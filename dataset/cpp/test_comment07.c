/*****************************************************************************
 * Test cases for libxlsxwriter.
 *
 * Test to compare output against Excel files.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "xlsxwriter.h"

int main() {

    lxw_workbook  *workbook  = workbook_new("test_comment07.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_write_comment(worksheet, CELL("A1"), "Some text");
    worksheet_write_comment(worksheet, CELL("A2"), "Some text");
    worksheet_write_comment(worksheet, CELL("A3"), "Some text");
    worksheet_write_comment(worksheet, CELL("A4"), "Some text");
    worksheet_write_comment(worksheet, CELL("A5"), "Some text");

    worksheet_show_comments(worksheet);

    worksheet_set_comments_author(worksheet, "John");

    return workbook_close(workbook);
}
