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

    lxw_workbook  *workbook  = workbook_new("test_set_row02.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    worksheet_set_row_pixels(worksheet, 0, 1,   NULL);
    worksheet_set_row_pixels(worksheet, 1, 2,   NULL);
    worksheet_set_row_pixels(worksheet, 2, 3,   NULL);
    worksheet_set_row_pixels(worksheet, 3, 4,   NULL);

    worksheet_set_row_pixels(worksheet, 11, 12, NULL);
    worksheet_set_row_pixels(worksheet, 12, 13, NULL);
    worksheet_set_row_pixels(worksheet, 13, 14, NULL);
    worksheet_set_row_pixels(worksheet, 14, 15, NULL);

    worksheet_set_row_pixels(worksheet, 18, 19, NULL);
    worksheet_set_row_pixels(worksheet, 20, 21, NULL);
    worksheet_set_row_pixels(worksheet, 21, 22, NULL);

    return workbook_close(workbook);
}
