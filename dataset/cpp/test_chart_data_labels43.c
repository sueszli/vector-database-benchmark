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

    lxw_workbook  *workbook  = workbook_new("test_chart_data_labels43.xlsx");
    lxw_worksheet *worksheet = workbook_add_worksheet(workbook, NULL);
    lxw_chart     *chart     = workbook_add_chart(workbook, LXW_CHART_COLUMN);

    /* For testing, copy the randomly generated axis ids in the target file. */
    chart->axis_id_1 = 56185600;
    chart->axis_id_2 = 56187520;

    uint8_t data[5][3] = {
        {1, 2,  3},
        {2, 4,  6},
        {3, 6,  9},
        {4, 8,  12},
        {5, 10, 15}
    };

    int row, col;
    for (row = 0; row < 5; row++)
        for (col = 0; col < 3; col++)
            worksheet_write_number(worksheet, row, col, data[row][col], NULL);

    lxw_chart_series *series = chart_add_series(chart, NULL, "=Sheet1!$A$1:$A$5");

    lxw_chart_line line = {.color = LXW_COLOR_RED};
    lxw_chart_font font = {.bold = LXW_TRUE, .italic = LXW_TRUE, .color = LXW_COLOR_RED, .baseline = -1};

    lxw_chart_data_label data_label1 = {.line = &line, .font = &font};
    lxw_chart_data_label *data_labels[] = {&data_label1, NULL};

    chart_series_set_labels_custom(series, data_labels);

    chart_add_series(chart, NULL, "=Sheet1!$B$1:$B$5");
    chart_add_series(chart, NULL, "=Sheet1!$C$1:$C$5");

    worksheet_insert_chart(worksheet, CELL("E9"), chart);

    return workbook_close(workbook);
}
