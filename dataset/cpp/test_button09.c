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

    lxw_workbook  *workbook  = workbook_new("test_button09.xlsx");
    lxw_worksheet *worksheet1 = workbook_add_worksheet(workbook, NULL);
    lxw_worksheet *worksheet2 = workbook_add_worksheet(workbook, NULL);

    worksheet_insert_button(worksheet2, CELL("C2"), NULL);

    worksheet_write_comment(worksheet1, CELL("A1"), "Foo");

    worksheet_set_comments_author(worksheet1, "John");

    return workbook_close(workbook);
}
