/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/worksheet.h"

/* 1. Test the _write_sheet_views() method with split panes + selection. */
CTEST(worksheet, set_selection41) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"600\" topLeftCell=\"A21\" activePane=\"bottomLeft\"/><selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("A2:A2"));
    worksheet_split_panes_opt(worksheet, 15, 0, 20, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 2. Test the _write_sheet_views() method with split panes + selection. */
CTEST(worksheet, set_selection42) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"600\" topLeftCell=\"A21\" activePane=\"bottomLeft\"/><selection pane=\"bottomLeft\" activeCell=\"A21\" sqref=\"A21\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("A21:A21"));
    worksheet_split_panes_opt(worksheet, 15, 0, 20, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 3. Test the _write_sheet_views() method with split panes + selection. */
CTEST(worksheet, set_selection43) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"1350\" topLeftCell=\"E1\" activePane=\"topRight\"/><selection pane=\"topRight\" activeCell=\"B1\" sqref=\"B1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("B1:B1"));
    worksheet_split_panes_opt(worksheet, 0, 8.43, 0, 4);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 4. Test the _write_sheet_views() method with split panes + selection. */
CTEST(worksheet, set_selection44) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"1350\" topLeftCell=\"E1\" activePane=\"topRight\"/><selection pane=\"topRight\" activeCell=\"E1\" sqref=\"E1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("E1:E1"));
    worksheet_split_panes_opt(worksheet, 0, 8.43, 0, 4);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}
