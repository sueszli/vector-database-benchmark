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

    lxw_workbook  *workbook  = workbook_new("test_print_area07.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_print_area(worksheet, RANGE("A1:XFD1048576"));

    worksheet_write_string(worksheet, CELL("A1"), "Foo" , NULL);

    return workbook_close(workbook);
}
