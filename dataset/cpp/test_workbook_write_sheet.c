/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/workbook.h"

// Test the _write_sheet() function.
CTEST(workbook, write_sheet1) {


    char* got;
    char exp[] = "<sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_workbook *workbook = workbook_new(NULL);
    workbook->file = testfile;

    _write_sheet(workbook, "Sheet1", 1, 0);

    RUN_XLSX_STREQ(exp, got);

    lxw_workbook_free(workbook);
}

// Test the _write_sheet() function.
CTEST(workbook, write_sheet2) {


    char* got;
    char exp[] = "<sheet name=\"Sheet1\" sheetId=\"1\" state=\"hidden\" r:id=\"rId1\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_workbook *workbook = workbook_new(NULL);
    workbook->file = testfile;

    _write_sheet(workbook, "Sheet1", 1, 1);

    RUN_XLSX_STREQ(exp, got);

    lxw_workbook_free(workbook);
}

// Test the _write_sheet() function.
CTEST(workbook, write_sheet3) {


    char* got;
    char exp[] = "<sheet name=\"Bits &amp; Bobs\" sheetId=\"1\" r:id=\"rId1\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_workbook *workbook = workbook_new(NULL);
    workbook->file = testfile;

    _write_sheet(workbook, "Bits & Bobs", 1, 0);

    RUN_XLSX_STREQ(exp, got);

    lxw_workbook_free(workbook);
}

