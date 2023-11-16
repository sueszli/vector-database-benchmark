/*
 * Tests for the libxlsxwriter library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/workbook.h"

/* Test the _write_defined_names() method. */
CTEST(workbook, write_defined_names) {


    char* got;
    char exp[] = "<definedNames><definedName name=\"_xlnm.Print_Titles\" localSheetId=\"0\">Sheet1!$1:$1</definedName></definedNames>";
    FILE* testfile = lxw_tmpfile(NULL);
    lxw_workbook *workbook = workbook_new(NULL);
    workbook->file = testfile;

    workbook_add_worksheet(workbook, NULL);

    _store_defined_name(workbook, "_xlnm.Print_Titles", "", "Sheet1!$1:$1", 0, 0);

    _write_defined_names(workbook);

    RUN_XLSX_STREQ(exp, got);

    lxw_workbook_free(workbook);
}



/* Test the _write_defined_name() method. */
CTEST(workbook, write_defined_names_sorted) {
    char* got;
    char exp[] = "<definedNames><definedName name=\"_Egg\">Sheet1!$A$1</definedName><definedName name=\"_Fog\">Sheet1!$A$1</definedName><definedName name=\"aaa\" localSheetId=\"1\">Sheet2!$A$1</definedName><definedName name=\"Abc\">Sheet1!$A$1</definedName><definedName name=\"Bar\" localSheetId=\"2\">'Sheet 3'!$A$1</definedName><definedName name=\"Bar\" localSheetId=\"0\">Sheet1!$A$1</definedName><definedName name=\"Bar\" localSheetId=\"1\">Sheet2!$A$1</definedName><definedName name=\"Baz\">0.98</definedName><definedName name=\"car\" localSheetId=\"2\">\"Saab 900\"</definedName></definedNames>";
    FILE* testfile = lxw_tmpfile(NULL);


    lxw_workbook *workbook = workbook_new(NULL);
    workbook->file = testfile;

    workbook_add_worksheet(workbook, NULL);
    workbook_add_worksheet(workbook, NULL);
    workbook_add_worksheet(workbook, "Sheet 3");


    workbook_define_name(workbook, "'Sheet 3'!Bar", "='Sheet 3'!$A$1");
    workbook_define_name(workbook, "Abc",           "=Sheet1!$A$1"   );
    workbook_define_name(workbook, "Baz",           "=0.98"          );
    workbook_define_name(workbook, "Sheet1!Bar",    "=Sheet1!$A$1"   );
    workbook_define_name(workbook, "Sheet2!Bar",    "=Sheet2!$A$1"   );
    workbook_define_name(workbook, "Sheet2!aaa",    "=Sheet2!$A$1"   );
    workbook_define_name(workbook, "'Sheet 3'!car", "=\"Saab 900\""  );
    workbook_define_name(workbook, "_Egg",          "=Sheet1!$A$1"   );
    workbook_define_name(workbook, "_Fog",          "=Sheet1!$A$1"   );

    _write_defined_names(workbook);

    RUN_XLSX_STREQ(exp, got);

    lxw_workbook_free(workbook);
}
