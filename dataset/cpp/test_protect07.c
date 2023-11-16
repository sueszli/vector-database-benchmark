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

    lxw_workbook  *workbook  = workbook_new("test_protect07.xlsx");
    workbook_add_worksheet(workbook, NULL);

    workbook_read_only_recommended(workbook);

    return workbook_close(workbook);
}
