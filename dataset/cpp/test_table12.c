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

    lxw_workbook  *workbook  = workbook_new("test_table12.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_set_column(worksheet, COLS("C:F"), 10.288, NULL);

    worksheet_add_table(worksheet, RANGE("C2:F6"), NULL);

    worksheet_write_string(worksheet, 2, 2, "Foo", NULL);
    worksheet_write_string(worksheet, 3, 2, "Bar", NULL);
    worksheet_write_string(worksheet, 4, 2, "Baz", NULL);

    worksheet_write_number(worksheet, 2, 3, 1234, NULL);
    worksheet_write_number(worksheet, 3, 3, 1256, NULL);
    worksheet_write_number(worksheet, 4, 3, 2234, NULL);

    worksheet_write_number(worksheet, 2, 4, 2000, NULL);
    worksheet_write_number(worksheet, 3, 4, 4000, NULL);
    worksheet_write_number(worksheet, 4, 4, 3000, NULL);

    return workbook_close(workbook);
}
