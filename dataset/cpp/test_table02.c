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

    lxw_workbook  *workbook  = workbook_new("test_table02.xlsx");
    lxw_worksheet *worksheet1 = workbook_add_worksheet(workbook, NULL);
    lxw_worksheet *worksheet2 = workbook_add_worksheet(workbook, NULL);

    worksheet_set_column(worksheet1, COLS("B:J"), 10.288, NULL);
    worksheet_set_column(worksheet2, COLS("C:L"), 10.288, NULL);

    worksheet_add_table(worksheet1, RANGE("B3:E11"), NULL);
    worksheet_add_table(worksheet1, RANGE("G10:J16"), NULL);
    worksheet_add_table(worksheet1, RANGE("C18:F25"), NULL);

    worksheet_add_table(worksheet2, RANGE("I4:L11"), NULL);
    worksheet_add_table(worksheet2, RANGE("C16:H23"), NULL);

    return workbook_close(workbook);
}
