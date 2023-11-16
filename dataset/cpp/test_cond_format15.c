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

    lxw_workbook  *workbook  = workbook_new("test_cond_format15.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);

    lxw_format *format1 = workbook_add_format(workbook);
    format_set_bg_color(format1, LXW_COLOR_RED);

    lxw_format *format2 = workbook_add_format(workbook);
    format_set_bg_color(format2, 0x92D050);

    worksheet_write_number(worksheet, CELL("A1"), 10 , NULL);
    worksheet_write_number(worksheet, CELL("A2"), 20 , NULL);
    worksheet_write_number(worksheet, CELL("A3"), 30 , NULL);
    worksheet_write_number(worksheet, CELL("A4"), 40 , NULL);

    lxw_conditional_format *conditional_format = calloc(1, sizeof(lxw_conditional_format));

    conditional_format->type         = LXW_CONDITIONAL_TYPE_CELL;
    conditional_format->criteria     = LXW_CONDITIONAL_CRITERIA_LESS_THAN;
    conditional_format->value        = 5;
    conditional_format->stop_if_true = LXW_TRUE;
    conditional_format->format       = format1;
    worksheet_conditional_format_cell(worksheet, CELL("A1"), conditional_format);

    conditional_format->type         = LXW_CONDITIONAL_TYPE_CELL;
    conditional_format->criteria     = LXW_CONDITIONAL_CRITERIA_GREATER_THAN;
    conditional_format->value        = 20;
    conditional_format->stop_if_true = LXW_TRUE;
    conditional_format->format       = format2;
    worksheet_conditional_format_cell(worksheet, CELL("A1"), conditional_format);

    free(conditional_format);
    return workbook_close(workbook);
}
