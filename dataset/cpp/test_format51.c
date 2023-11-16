/*****************************************************************************
 * Test cases for libxlsxwriter.
 *
 * Simple test case to test data writing.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "xlsxwriter.h"


int main() {

    lxw_workbook  *workbook  = workbook_new("test_format51.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);
    lxw_format    *format;

    double value = 123.456;

    worksheet_set_column(worksheet, 0, 0, 12, NULL);

    format = workbook_add_format(workbook);
    format_set_num_format(format, "0.0");
    worksheet_write_number(worksheet, 0, 0, value, format);

    format = workbook_add_format(workbook);
    format_set_num_format(format, "0.000");
    worksheet_write_number(worksheet, 1, 0, value, format);

    format = workbook_add_format(workbook);
    format_set_num_format(format, "0.0000");
    worksheet_write_number(worksheet, 2, 0, value, format);

    format = workbook_add_format(workbook);
    format_set_num_format(format, "0.00000");
    worksheet_write_number(worksheet, 3, 0, value, format);

    return workbook_close(workbook);
}
