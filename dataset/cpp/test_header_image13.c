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

    lxw_workbook  *workbook  = workbook_new("test_header_image13.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    lxw_header_footer_options header_options = {.image_left   = "images/black_72.jpg",
                                                .image_center = "images/black_150.jpg",
                                                .image_right  = "images/black_300.jpg"};

    worksheet_set_header_opt(worksheet, "&L&G&C&G&R&G", &header_options);

    return workbook_close(workbook);
}
