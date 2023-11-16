/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/styles.h"

// Test the _write_style_sheet() function.
CTEST(styles, write_style_sheet) {


    char* got;
    char exp[] = "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    styles->file = testfile;

    _write_style_sheet(styles);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
}

