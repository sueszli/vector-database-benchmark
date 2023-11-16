/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/worksheet.h"

/* 1. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes01) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"600\" topLeftCell=\"A2\"/><selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 15, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 2. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes02) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"900\" topLeftCell=\"A3\"/><selection pane=\"bottomLeft\" activeCell=\"A3\" sqref=\"A3\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 30, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 3. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes03) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"2400\" topLeftCell=\"A8\"/><selection pane=\"bottomLeft\" activeCell=\"A8\" sqref=\"A8\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 105, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 4. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes04) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"1350\" topLeftCell=\"B1\"/><selection pane=\"topRight\" activeCell=\"B1\" sqref=\"B1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 0, 8.43);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 5. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes05) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"2310\" topLeftCell=\"C1\"/><selection pane=\"topRight\" activeCell=\"C1\" sqref=\"C1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 0, 17.57);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 6. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes06) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"5190\" topLeftCell=\"F1\"/><selection pane=\"topRight\" activeCell=\"F1\" sqref=\"F1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 0, 45 );
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 7. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes07) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"1350\" ySplit=\"600\" topLeftCell=\"B2\"/><selection pane=\"topRight\" activeCell=\"B1\" sqref=\"B1\"/><selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/><selection pane=\"bottomRight\" activeCell=\"B2\" sqref=\"B2\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 15, 8.43);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 8. Test the _write_sheet_views() method with split panes. */
CTEST(worksheet, write_split_panes08) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"6150\" ySplit=\"1200\" topLeftCell=\"G4\"/><selection pane=\"topRight\" activeCell=\"G1\" sqref=\"G1\"/><selection pane=\"bottomLeft\" activeCell=\"A4\" sqref=\"A4\"/><selection pane=\"bottomRight\" activeCell=\"G4\" sqref=\"G4\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_split_panes(worksheet, 45, 54.14);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}
