/*****************************************************************************
 * Test cases for libxlsxwriter.
 *
 * Simple test case to test worksheet set_row() and set_column().
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "xlsxwriter.h"

int main() {

    lxw_workbook  *workbook  = workbook_new("test_row_col_format05.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    lxw_format    *bold      = workbook_add_format(workbook);
    format_set_bold(bold);

    lxw_format    *italic    = workbook_add_format(workbook);
    format_set_italic(italic);

    worksheet_set_column(worksheet, 0, 0, 8.43, bold);
    worksheet_set_column(worksheet, 2, 2, 8.43, italic);

    return workbook_close(workbook);
}
