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

    lxw_workbook  *workbook  = workbook_new("test_table07.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_set_column(worksheet, COLS("C:F"), 10.288, NULL);

    worksheet_write_string(worksheet, CELL("A1"), "Foo", NULL);

    lxw_table_options options = {.no_header_row = LXW_TRUE};
    worksheet_add_table(worksheet, RANGE("C3:F13"), &options);

    return workbook_close(workbook);
}
