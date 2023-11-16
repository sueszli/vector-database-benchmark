/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/worksheet.h"

/* 1. Test the _write_sheet_views() method with freeze panes + selection. */
CTEST(worksheet, set_selection11) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/><selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("A2:A2"));
    worksheet_freeze_panes(worksheet, 1, 0);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 2. Test the _write_sheet_views() method with freeze panes + selection. */
CTEST(worksheet, set_selection12) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"1\" topLeftCell=\"B1\" activePane=\"topRight\" state=\"frozen\"/><selection pane=\"topRight\" activeCell=\"B1\" sqref=\"B1\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("B1:B1"));
    worksheet_freeze_panes(worksheet, 0, 1);
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 3. Test the _write_sheet_views() method with freeze panes + selection. */
CTEST(worksheet, set_selection13) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"6\" ySplit=\"3\" topLeftCell=\"G4\" activePane=\"bottomRight\" state=\"frozen\"/><selection pane=\"topRight\" activeCell=\"G1\" sqref=\"G1\"/><selection pane=\"bottomLeft\" activeCell=\"A4\" sqref=\"A4\"/><selection pane=\"bottomRight\" activeCell=\"G4\" sqref=\"G4\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("G4:G4"));
    worksheet_freeze_panes(worksheet, CELL("G4"));
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}


/* 4. Test the _write_sheet_views() method with freeze panes + selection. */
CTEST(worksheet, set_selection14) {
    char* got;
    char exp[] = "<sheetViews><sheetView tabSelected=\"1\" workbookViewId=\"0\"><pane xSplit=\"6\" ySplit=\"3\" topLeftCell=\"G4\" activePane=\"bottomRight\" state=\"frozen\"/><selection pane=\"topRight\" activeCell=\"G1\" sqref=\"G1\"/><selection pane=\"bottomLeft\" activeCell=\"A4\" sqref=\"A4\"/><selection pane=\"bottomRight\" activeCell=\"I5\" sqref=\"I5\"/></sheetView></sheetViews>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_select(worksheet);
    worksheet_set_selection(worksheet, RANGE("I5:I5"));
    worksheet_freeze_panes(worksheet, CELL("G4"));
    _worksheet_write_sheet_views(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}
