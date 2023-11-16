/*
 * Copyright (c) 2021 One Identity
 * Copyright (c) 2021 Xiaoyu Qiu
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

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include "regexp-parser.h"
#include "apphook.h"
#include "logmsg/logmsg.h"
#include "scratch-buffers.h"

void
setup(void)
{
  configuration = cfg_new_snippet();
  app_startup();
}

void
teardown(void)
{
  scratch_buffers_explicit_gc();
  app_shutdown();
  cfg_free(configuration);
}

TestSuite(regexp_parser, .init = setup, .fini = teardown);

typedef struct _RegexpParserTestParam
{
  const gchar *msg;
  const gchar *pattern;
  const gchar *prefix;
  gint flags;
  gboolean expected_result;
  const gchar *name;
  const gchar *value;
} RegexpParserTestParam;

static LogParser *
_construct_regexp_parser(const gchar *prefix, const gchar *pattern, gint flags)
{
  LogParser *p = regexp_parser_new(configuration);

  LogMatcherOptions *matcher_options = regexp_parser_get_matcher_options(p);
  matcher_options->flags |= flags;

  if (prefix != NULL)
    regexp_parser_set_prefix(p, prefix);
  if (pattern != NULL)
    regexp_parser_set_patterns(p, g_list_append(NULL, g_strdup(pattern)));

  return p;
}

ParameterizedTestParameters(regexp_parser, test_regexp_parser)
{
  static RegexpParserTestParam parser_params[] =
  {
    {.msg = "foo", .pattern = "(?<key>foo)", .prefix="", .flags = 0, .expected_result = TRUE, .name = "key", .value = "foo"},
    {.msg = "foo", .pattern = "(?<key>fo*)", .prefix="", .flags = 0, .expected_result = TRUE, .name = "key", .value = "foo"},
    {.msg = "foo", .pattern = "(?<key>fo*)", .prefix=".reg.", .flags = 0, .expected_result = TRUE, .name = ".reg.key", .value = "foo"},
    {.msg = "foo", .pattern = "(?<key>fo*)", .prefix=".reg.", .flags = 0, .expected_result = TRUE, .name = "key", .value = ""},
    {.msg = "foo", .pattern = "(?<key>foo)|(?<key>bar)", .prefix=".reg.", .expected_result = TRUE, .flags = LMF_DUPNAMES, .name = ".reg.key", .value = "foo"},
    {.msg = "abc", .pattern = "Abc", .prefix="", .flags = 0, .expected_result = FALSE, .name = NULL, .value = NULL},
    {.msg = "abc", .pattern = "(?<key>Abc)", .prefix="", .flags = LMF_ICASE, .expected_result = TRUE, .name = "key", .value = "abc"},

    /* store into a builtin value */
    {.msg = "abcdef", .pattern = "(?<PID>abc)", .prefix="", .flags = 0, .expected_result = TRUE, .name = "PID", .value = "abc"},
  };
  return cr_make_param_array(RegexpParserTestParam, parser_params, G_N_ELEMENTS(parser_params));
}

ParameterizedTest(RegexpParserTestParam *parser_param, regexp_parser, test_regexp_parser)
{
  LogParser *p = _construct_regexp_parser(parser_param->prefix, parser_param->pattern, parser_param->flags);
  gboolean result;
  GError *e = NULL;

  result = regexp_parser_compile(p, &e);
  cr_assert(result, "unexpected compiling failure; pattern=%s, error=%s\n", parser_param->pattern, e->message);

  LogMessage *msg = log_msg_new_empty();
  log_msg_set_value(msg, LM_V_MESSAGE, parser_param->msg, -1);

  LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;
  result = log_parser_process_message(p, &msg, &path_options);
  cr_assert_not((result && !parser_param->expected_result), "unexpected match; msg=%s\n", parser_param->msg);
  cr_assert_not((!result && parser_param->expected_result), "unexpected non-match; msg=%s\n", parser_param->msg);

  if (parser_param->name)
    {
      gssize len = -1;
      const gchar *value = log_msg_get_value_by_name(msg, parser_param->name, &len);
      cr_assert_str_eq(value, parser_param->value, "name: %s | value: %.*s, should be %s",
                       parser_param->name,
                       (gint) len, value,
                       parser_param->value);
    }

  log_pipe_unref(&p->super);
  log_msg_unref(msg);
}

Test(regexp_parser, test_regexp_parser_with_multiple_patterns)
{
  LogParser *p = regexp_parser_new(configuration);
  GList *patterns = NULL;
  patterns = g_list_append(patterns, g_strdup("(?<key>Abc)"));
  patterns = g_list_append(patterns, g_strdup("(?<key>abc)"));
  regexp_parser_set_patterns(p, patterns);
  regexp_parser_compile(p, NULL);

  LogMessage *msg = log_msg_new_empty();
  log_msg_set_value(msg, LM_V_MESSAGE, "abc", -1);

  LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;
  log_parser_process_message(p, &msg, &path_options);

  gssize len;
  const gchar *value = log_msg_get_value_by_name(msg, "key", &len);
  cr_assert(strncmp(value, "abc", 3) == 0,
            "Test regexp parser with multiple patterns failed: name: %s | value: %.*s, should be %s",
            "key",
            (gint) len, value,
            "abc");

  log_pipe_unref((LogPipe *)p);
  log_msg_unref(msg);
}
