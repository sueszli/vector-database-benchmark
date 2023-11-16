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

    lxw_workbook  *workbook  = workbook_new("test_button07.xlsm");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    lxw_button_options options = {.caption = "Hello", .macro = "say_hello"};

    worksheet_insert_button(worksheet, CELL("C2"), &options);

    workbook_add_vba_project(workbook, "images/vbaProject02.bin");

    return workbook_close(workbook);
}
