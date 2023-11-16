/*
 * Copyright (c) 2016 Balabit
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "contextual-data-record-scanner.h"
#include "scanner/csv-scanner/csv-scanner.h"
#include "string-list.h"
#include "messages.h"
#include "cfg.h"

#include <string.h>

struct _ContextualDataRecordScanner
{
  ContextualDataRecord last_record;
  GlobalConfig *cfg;
  CSVScanner scanner;
  CSVScannerOptions options;
  gchar *filename;
  gchar *name_prefix;
};

static gboolean
_fetch_next(ContextualDataRecordScanner *self)
{
  if (!csv_scanner_scan_next(&self->scanner))
    {
      const gchar *columns[] = { "selector", "name", "value", NULL };
      gint column_index = csv_scanner_get_current_column(&self->scanner);
      const gchar *column_name = column_index < 3 ? columns[column_index] : "out-of-range";
      msg_error("add-contextual-data(): error parsing CSV file, expecting an additional column which was not found. Expecting (selector, name, value) triplets",
                evt_tag_str("target", column_name));
      return FALSE;
    }

  return TRUE;
}

static gboolean
_is_whole_record_parsed(ContextualDataRecordScanner *self)
{
  if (!csv_scanner_scan_next(&self->scanner) &&
      csv_scanner_is_scan_complete(&self->scanner))
    return TRUE;

  msg_error("add-contextual-data(): extra data found at the end of line, expecting (selector, name, value) triplets");
  return FALSE;
}

static gboolean
_fetch_selector(ContextualDataRecordScanner *self, ContextualDataRecord *record)
{
  if (!_fetch_next(self))
    return FALSE;
  record->selector = g_strdup(csv_scanner_get_current_value(&self->scanner));
  return TRUE;
}

static gboolean
_fetch_name(ContextualDataRecordScanner *self, ContextualDataRecord *record)
{
  if (!_fetch_next(self))
    return FALSE;

  gchar *name = g_strdup_printf("%s%s", self->name_prefix ? : "", csv_scanner_get_current_value(&self->scanner));
  record->value_handle = log_msg_get_value_handle(name);
  g_free(name);

  return TRUE;
}

static gboolean
_fetch_value(ContextualDataRecordScanner *self, ContextualDataRecord *record)
{
  if (!_fetch_next(self))
    return FALSE;

  const gchar *value_template = csv_scanner_get_current_value(&self->scanner);

  record->value = log_template_new(self->cfg, NULL);


  GError *error = NULL;
  gboolean success;

  if (cfg_is_config_version_older(self->cfg, VERSION_VALUE_3_21) &&
      strchr(value_template, '$') != NULL)
    {
      msg_warning("WARNING: the value field in add-contextual-data() CSV files has been changed "
                  "to be a template starting with " VERSION_3_21 ". You are using an older config "
                  "version and your CSV file contains a '$' character in this field, which needs "
                  "to be escaped as '$$' once you change your @version declaration in the "
                  "configuration. This message means that this string is now assumed to be a "
                  "literal (non-template) string for compatibility",
                  cfg_format_config_version_tag(self->cfg),
                  evt_tag_str("selector", record->selector),
                  evt_tag_str("name", log_msg_get_value_name(record->value_handle, NULL)),
                  evt_tag_str("value", value_template));
      log_template_compile_literal_string(record->value, value_template);
      success = TRUE;
    }
  else if (cfg_is_typing_feature_enabled(self->cfg))
    {
      /* typing feature is enabled */
      if (cfg_is_config_version_older(self->cfg, VERSION_VALUE_4_0))
        {
          /* old @config, use compat mode but warn if the format would become incompatible */
          if (strchr(value_template, '(') != NULL)
            {
              success = log_template_compile_with_type_hint(record->value, value_template, &error);
              if (!success)
                {
                  log_template_set_type_hint(record->value, "string", NULL);
                  msg_warning("WARNING: the value field in add-contextual-data() CSV files has been changed "
                              "to support typing from " FEATURE_TYPING_VERSION ". You are using an older config "
                              "version and your CSV file contains an unrecognized type-cast, probably a "
                              "parenthesis in the value field. This will be interpreted in the `type(value)' "
                              "format in future versions. Please add an "
                              "explicit string() cast as shown in the 'fixed-value' tag of this log message "
                              "or remove the parenthesis. The value column will be processed as a 'string' "
                              "expression",
                              cfg_format_config_version_tag(self->cfg),
                              evt_tag_str("selector", record->selector),
                              evt_tag_str("name", log_msg_get_value_name(record->value_handle, NULL)),
                              evt_tag_str("value", value_template),
                              evt_tag_printf("fixed-value", "string(%s)", value_template));
                  g_clear_error(&error);
                  success = log_template_compile(record->value, value_template, &error);
                }
            }
          else
            {
              success = log_template_compile(record->value, value_template, &error);
            }
        }
      else
        {
          /* new @config, use the new format with error handling */
          success = log_template_compile_with_type_hint(record->value, value_template, &error);
        }
    }
  else
    {
      /* typing feature is disabled, use old format, no warnings */
      success = log_template_compile(record->value, value_template, &error);
    }

  if (!success)
    {
      msg_error("add-contextual-data(): error compiling template",
                evt_tag_str("selector", record->selector),
                evt_tag_str("name", log_msg_get_value_name(record->value_handle, NULL)),
                evt_tag_str("value", value_template),
                evt_tag_str("error", error->message));
      g_clear_error(&error);
      return FALSE;
    }
  return TRUE;
}

static gboolean
_get_next_record(ContextualDataRecordScanner *self, const gchar *input, ContextualDataRecord *record)
{
  gboolean result = FALSE;

  csv_scanner_init(&self->scanner, &self->options, input);

  if (!_fetch_selector(self, record))
    goto error;

  if (!_fetch_name(self, record))
    goto error;

  if (!_fetch_value(self, record))
    goto error;

  if (!_is_whole_record_parsed(self))
    goto error;

  result = TRUE;

error:
  csv_scanner_deinit(&self->scanner);
  return result;
}

ContextualDataRecord *
contextual_data_record_scanner_get_next(ContextualDataRecordScanner *self,
                                        const gchar *input,
                                        const gchar *filename, gint lineno)
{
  contextual_data_record_init(&self->last_record);
  if (!_get_next_record(self, input, &self->last_record))
    {
      contextual_data_record_clean(&self->last_record);
      msg_error("add-contextual-data(): the failing line is",
                evt_tag_str("input", input),
                evt_tag_printf("filename", "%s:%d", filename, lineno));
      return NULL;
    }

  return &self->last_record;
}

void
contextual_data_record_scanner_free(ContextualDataRecordScanner *self)
{
  csv_scanner_options_clean(&self->options);
  g_free(self->name_prefix);
  g_free(self);
}

ContextualDataRecordScanner *
contextual_data_record_scanner_new(GlobalConfig *cfg, const gchar *name_prefix)
{
  ContextualDataRecordScanner *self =
    g_new0(ContextualDataRecordScanner, 1);

  self->cfg = cfg;

  csv_scanner_options_set_delimiters(&self->options, ",");
  csv_scanner_options_set_quote_pairs(&self->options, "\"\"''");
  csv_scanner_options_set_expected_columns(&self->options, 3);
  csv_scanner_options_set_flags(&self->options, CSV_SCANNER_STRIP_WHITESPACE);
  csv_scanner_options_set_dialect(&self->options, CSV_SCANNER_ESCAPE_DOUBLE_CHAR);
  self->name_prefix = g_strdup(name_prefix);

  return self;
}
