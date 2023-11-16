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

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include "libtest/cr_template.h"

#include "context-info-db.h"
#include "apphook.h"
#include "scratch-buffers.h"
#include "cfg.h"
#include <stdio.h>
#include <string.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))


static void
_count_records(gpointer arg, const ContextualDataRecord *record)
{
  int *ctr = (int *) arg;
  ++(*ctr);
}

static void
_test_empty_db(ContextInfoDB *context_info_db)
{
  cr_assert_not(context_info_db_is_loaded(context_info_db) == TRUE,
                "Empty ContextInfoDB should be in unloaded state.");
  cr_assert_not(context_info_db_is_indexed(context_info_db) == TRUE,
                "Empty ContextInfoDB should be in un-indexed state.");
  cr_assert_not(context_info_db_contains(context_info_db, "selector") == TRUE,
                "Method context_info_db_contains should work with empty ContextInfoDB.");
  cr_assert_eq(context_info_db_number_of_records(context_info_db, "selector"), 0,
               "Method context_info_db_number should work with empty ContextInfoDB.");
  int ctr = 0;
  context_info_db_foreach_record(context_info_db, "selector", _count_records,
                                 (gpointer) & ctr);
  cr_assert_eq(ctr, 0,
               "Method context_info_db_foreach_record should work for with empty ContextInfoDB.");
}

Test(add_contextual_data, test_empty_db)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);

  _test_empty_db(context_info_db);

  context_info_db_unref(context_info_db);
}

Test(add_contextual_data, test_purge_empty_db)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);

  context_info_db_purge(context_info_db);
  _test_empty_db(context_info_db);

  context_info_db_unref(context_info_db);
}

Test(add_contextual_data, test_index_empty_db)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);

  context_info_db_index(context_info_db);
  _test_empty_db(context_info_db);

  context_info_db_unref(context_info_db);
}

static void
_fill_context_info_db(ContextInfoDB *context_info_db,
                      const gchar *selector_base, const gchar *name_base,
                      const gchar *value_base, int number_of_selectors,
                      int number_of_nv_pairs_per_selector)
{
  int i, j;
  for (i = 0; i < number_of_selectors; i++)
    {
      for (j = 0; j < number_of_nv_pairs_per_selector; j++)
        {
          gchar buf[256];

          ContextualDataRecord record;

          g_snprintf(buf, sizeof(buf), "%s-%d", selector_base, i);
          record.selector = g_strdup(buf);

          g_snprintf(buf, sizeof(buf), "%s-%d.%d", name_base, i, j);
          record.value_handle = log_msg_get_value_handle(buf);

          g_snprintf(buf, sizeof(buf), "%s-%d.%d", value_base, i, j);
          record.value = log_template_new(configuration, NULL);
          log_template_compile_literal_string(record.value, buf);
          context_info_db_insert(context_info_db, &record);
        }
    }
}

static gint
_g_strcmp(const gconstpointer a, gconstpointer b)
{
  return g_strcmp0((const gchar *) a, (const gchar *) b);
}

Test(add_contextual_data, test_insert)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);
  context_info_db_enable_ordering(context_info_db);

  _fill_context_info_db(context_info_db, "selector", "name", "value", 2, 5);
  int ctr = 0;
  cr_assert_eq(context_info_db_number_of_records(context_info_db, "selector-0"), 5,
               "selector-0 should have 5 nv-pairs");
  context_info_db_foreach_record(context_info_db, "selector-0", _count_records,
                                 (gpointer) & ctr);
  cr_assert_eq(ctr, 5, "foreach should find 5 nv-pairs for selector-0");
  cr_assert_eq(g_list_length(context_info_db_ordered_selectors(context_info_db)), 2,
               "2 different selectors were saved to the ordered list");

  context_info_db_unref(context_info_db);
}

Test(add_contextual_data, test_get_selectors)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);

  _fill_context_info_db(context_info_db, "selector", "name", "value", 2, 5);

  GList *selectors = context_info_db_get_selectors(context_info_db);
  GList *selector0 = g_list_find_custom(selectors, "selector-0", _g_strcmp);
  GList *selector1 = g_list_find_custom(selectors, "selector-1", _g_strcmp);

  cr_assert_str_eq((const gchar *)selector0->data, "selector-0");
  cr_assert_str_eq((const gchar *)selector1->data, "selector-1");


  context_info_db_unref(context_info_db);
  g_list_free(selectors);
}

typedef struct _TestNVPair
{
  const gchar *name;
  const gchar *value;
} TestNVPair;

typedef struct _TestNVPairStore
{
  TestNVPair *pairs;
  int ctr;
} TestNVPairStore;

static void
_foreach_get_nvpairs(gpointer arg, const ContextualDataRecord *record)
{
  TestNVPairStore *store = (TestNVPairStore *) arg;
  TestNVPair pair;
  GString *result = scratch_buffers_alloc();

  pair.name = log_msg_get_value_name(record->value_handle, NULL);

  LogMessage *msg = create_sample_message();
  log_template_format(record->value, msg, &DEFAULT_TEMPLATE_EVAL_OPTIONS, result);
  log_msg_unref(msg);

  pair.value = result->str;
  store->pairs[store->ctr++] = pair;
}

static void
_assert_context_info_db_contains_name_value_pairs_by_selector(ContextInfoDB *
    context_info_db,
    const gchar *
    selector,
    TestNVPair *
    expected_nvpairs,
    guint
    number_of_expected_nvpairs)
{
  TestNVPair result[number_of_expected_nvpairs];
  TestNVPairStore result_store = {.pairs = result, .ctr = 0 };

  context_info_db_foreach_record(context_info_db, selector,
                                 _foreach_get_nvpairs,
                                 (gpointer) & result_store);
  cr_assert_eq(result_store.ctr, number_of_expected_nvpairs);
  guint i;
  for (i = 0; i < number_of_expected_nvpairs; i++)
    {
      cr_assert_str_eq(result[i].name, expected_nvpairs[i].name);
      cr_assert_str_eq(result[i].value, expected_nvpairs[i].value);
    }
}

static void
_assert_import_csv_with_single_selector(gchar *csv_content, gchar *selector_to_check,
                                        TestNVPair *expected_nvpairs, gsize expected_nvpairs_size)
{
  FILE *fp = fmemopen(csv_content, strlen(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);
  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");
  fclose(fp);

  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      selector_to_check,
      expected_nvpairs,
      expected_nvpairs_size);

  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

Test(add_contextual_data, test_inserted_nv_pairs)
{
  ContextInfoDB *context_info_db = context_info_db_new(FALSE);

  _fill_context_info_db(context_info_db, "selector", "name", "value", 1, 3);

  TestNVPair expected_nvpairs[] =
  {
    {.name = "name-0.0", .value = "value-0.0"},
    {.name = "name-0.1", .value = "value-0.1"},
    {.name = "name-0.2", .value = "value-0.2"}
  };

  _assert_context_info_db_contains_name_value_pairs_by_selector
  (context_info_db, "selector-0", expected_nvpairs, ARRAY_SIZE(expected_nvpairs));
  context_info_db_unref(context_info_db);
}

Test(add_contextual_data, test_import_with_valid_csv)
{
  gchar csv_content[] = "selector1,name1,value1\n"
                        "selector1,name1.1,value1.1\n"
                        "selector2,name2,value2\n"
                        "selector3,name3,value3\n"
                        "selector3,name3.1,$(echo $HOST_FROM)";
  FILE *fp = fmemopen(csv_content, sizeof(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);
  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");
  cr_assert(context_info_db_is_loaded(db),
            "The context_info_db_is_loaded reports False after a successful import operation. ");
  cr_assert(context_info_db_is_indexed(db),
            "The context_info_db_is_indexed reports False after successful import&load operations.");
  fclose(fp);

  TestNVPair expected_nvpairs_selector1[] =
  {
    {.name = "name1", .value = "value1"},
    {.name = "name1.1", .value = "value1.1"},
  };

  TestNVPair expected_nvpairs_selector2[] =
  {
    {.name = "name2", .value = "value2"},
  };

  TestNVPair expected_nvpairs_selector3[] =
  {
    {.name = "name3", .value = "value3"},
    {.name = "name3.1", .value = "kismacska"},
  };

  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "selector1",
      expected_nvpairs_selector1,
      ARRAY_SIZE(expected_nvpairs_selector1));
  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "selector2",
      expected_nvpairs_selector2,
      ARRAY_SIZE(expected_nvpairs_selector2));
  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "selector3",
      expected_nvpairs_selector3,
      ARRAY_SIZE(expected_nvpairs_selector3));

  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

Test(add_contextual_data, test_import_from_csv_with_crlf_line_ending,
     .description = "RFC 4180: Each record should be located on a separate line, delimited by a line break (CRLF).")
{
  gchar csv_content[] = "selector1,name1,value1\r\n"
                        "selector1,name1.1,value1.1";

  TestNVPair expected_nvpairs[] =
  {
    {.name = "name1", .value = "value1"},
    {.name = "name1.1", .value = "value1.1"},
  };

  _assert_import_csv_with_single_selector(csv_content, "selector1", expected_nvpairs, ARRAY_SIZE(expected_nvpairs));
}

Test(add_contextual_data, test_import_from_csv_with_escaped_double_quote,
     .description = "RFC 4180: If double-quotes are used to enclose fields, then a double-quote appearing inside a "
                    "field must be escaped by preceding it with another double quote.")
{
  gchar csv_content[] = "selector1,name1,\"c\"\"cc\"";

  TestNVPair expected_nvpairs[] =
  {
    {.name = "name1", .value = "c\"cc"},
  };

  _assert_import_csv_with_single_selector(csv_content, "selector1", expected_nvpairs, ARRAY_SIZE(expected_nvpairs));
}

Test(add_contextual_data, test_import_with_invalid_csv_content)
{
  gchar csv_content[] = "xxx";
  FILE *fp = fmemopen(csv_content, strlen(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);

  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert_not(context_info_db_import(db, fp, "dummy.csv", scanner),
                "Successfully import an invalid CSV file.");
  cr_assert_not(context_info_db_is_loaded(db),
                "The context_info_db_is_loaded reports True after a failing import operation. ");
  cr_assert_not(context_info_db_is_indexed(db),
                "The context_info_db_is_indexed reports True after failing import&load operations.");

  fclose(fp);
  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

Test(add_contextual_data, test_import_with_csv_contains_invalid_line)
{
  gchar csv_content[] = "selector1,name1,value1\n"
                        ",value1.1\n";
  FILE *fp = fmemopen(csv_content, strlen(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);

  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert_not(context_info_db_import(db, fp, "dummy.csv", scanner),
                "Successfully import an invalid CSV file.");
  cr_assert_not(context_info_db_is_loaded(db),
                "The context_info_db_is_loaded reports True after a failing import operation. ");
  cr_assert_not(context_info_db_is_indexed(db),
                "The context_info_db_is_indexed reports True after failing import&load operations.");

  fclose(fp);
  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

struct TestNVPairPrefix
{
  TestNVPair expected;
  const gchar *prefix;
};

ParameterizedTestParameters(add_contextual_data, test_import_with_prefix)
{
  static struct TestNVPairPrefix params[] =
  {
    {
      .expected = {.name = "name1", .value = "value1"},
      .prefix = NULL
    },
    {
      .expected = {.name = "name1", .value = "value1"},
      .prefix = ""
    },
    {
      .expected = {.name = "aaaname1", .value = "value1"},
      .prefix = "aaa"
    },
    {
      .expected = {.name = "aaa.name1", .value = "value1"},
      .prefix = "aaa."
    },
    {
      .expected = {.name = ".aaa.name1", .value = "value1"},
      .prefix = ".aaa."
    },
    {
      .expected = {.name = ".name1", .value = "value1"},
      .prefix = "."
    },
    {
      .expected = {.name = "....name1", .value = "value1"},
      .prefix = "...."
    }
  };
  size_t nb_params = sizeof (params) / sizeof (struct TestNVPairPrefix);
  return cr_make_param_array(struct TestNVPairPrefix, params, nb_params);
}


ParameterizedTest(struct TestNVPairPrefix *param, add_contextual_data, test_import_with_prefix)
{
  gchar csv_content[] = "selector1,name1,value1";

  FILE *fp = fmemopen(csv_content, sizeof(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);
  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, param->prefix);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");
  cr_assert(context_info_db_is_loaded(db),
            "The context_info_db_is_loaded reports False after a successful import operation. ");
  cr_assert(context_info_db_is_indexed(db),
            "The context_info_db_is_indexed reports False after successful import&load operations.");
  fclose(fp);

  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "selector1",
      &param->expected,
      1);
}

Test(add_contextual_data, test_ignore_case_on)
{
  gchar csv_content[] = "LoCaLhOsT,tag1,value1";
  FILE *fp = fmemopen(csv_content, strlen(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(TRUE);

  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");

  cr_assert(context_info_db_contains(db, "Localhost"));
  cr_assert(context_info_db_contains(db, "localhost"));
  cr_assert(context_info_db_contains(db, "localhosT"));
  cr_assert(context_info_db_contains(db, "LOCALHOST"));
  cr_assert(context_info_db_contains(db, "LoCaLhOsT"));

  fclose(fp);
  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

Test(add_contextual_data, test_ignore_case_off)
{
  gchar csv_content[] = "LoCaLhOsT,tag1,value1";
  FILE *fp = fmemopen(csv_content, strlen(csv_content), "r");
  ContextInfoDB *db = context_info_db_new(FALSE);

  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");

  cr_assert_not(context_info_db_contains(db, "Localhost"));
  cr_assert_not(context_info_db_contains(db, "localhost"));
  cr_assert_not(context_info_db_contains(db, "localhosT"));
  cr_assert_not(context_info_db_contains(db, "LOCALHOST"));
  cr_assert(context_info_db_contains(db, "LoCaLhOsT"));

  fclose(fp);
  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

Test(add_contextual_data, test_selected_nvpairs_when_ignore_case_on)
{
  gchar csv_content[] = "selector,name1,value1\n"
                        "SeLeCtOr,name2,value2\n"
                        "sElEcToR,name3,value3\n"
                        "another,name4,value4";

  FILE *fp = fmemopen(csv_content, sizeof(csv_content), "r");

  ContextInfoDB *db = context_info_db_new(TRUE);
  ContextualDataRecordScanner *scanner =
    contextual_data_record_scanner_new(configuration, NULL);

  cr_assert(context_info_db_import(db, fp, "dummy.csv", scanner),
            "Failed to import valid CSV file.");
  cr_assert(context_info_db_is_loaded(db),
            "The context_info_db_is_loaded reports False after a successful import operation. ");
  cr_assert(context_info_db_is_indexed(db),
            "The context_info_db_is_indexed reports False after successful import&load operations.");
  fclose(fp);

  TestNVPair expected_nvpairs_selector1[] =
  {
    {.name = "name1", .value = "value1"},
    {.name = "name2", .value = "value2"},
    {.name = "name3", .value = "value3"},
  };

  TestNVPair expected_nvpairs_selector2[] =
  {
    {.name = "name4", .value = "value4"},
  };

  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "SELECTOR",
      expected_nvpairs_selector1,
      ARRAY_SIZE(expected_nvpairs_selector1));

  _assert_context_info_db_contains_name_value_pairs_by_selector(db,
      "another",
      expected_nvpairs_selector2,
      ARRAY_SIZE(expected_nvpairs_selector2));

  context_info_db_unref(db);
  contextual_data_record_scanner_free(scanner);
}

static void
setup(void)
{
  app_startup();
  configuration = cfg_new_snippet();

  init_template_tests();
  cfg_load_module(configuration, "syslogformat");
  cfg_load_module(configuration, "basicfuncs");
}

static void
teardown(void)
{
  scratch_buffers_explicit_gc();
  app_shutdown();
}

TestSuite(add_contextual_data, .init=setup, .fini=teardown);
