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

    lxw_workbook  *workbook  = workbook_new("test_object_position20.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    lxw_row_col_options col_options = {.hidden = LXW_TRUE};
    worksheet_set_column_opt(worksheet, COLS("B:B"), 5, NULL, &col_options);

    lxw_image_options image_options = {.x_offset = 128};
    worksheet_insert_image_opt(worksheet, CELL("B9"), "images/red.png", &image_options);

    return workbook_close(workbook);
}
