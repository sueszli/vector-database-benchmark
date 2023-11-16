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

    lxw_workbook  *workbook  = workbook_new("test_comment02.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_write_string(worksheet, CELL("A1"), "Foo", NULL);
    worksheet_write_comment(worksheet, CELL("B2"), "Some text");
    worksheet_write_comment(worksheet, CELL("D17"), "More text");

    worksheet_set_comments_author(worksheet, "John");

    return workbook_close(workbook);
}
