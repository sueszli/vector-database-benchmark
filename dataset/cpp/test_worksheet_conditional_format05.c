/*
 * Tests for the libxlsxwriter library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/worksheet.h"
#include "../../../include/xlsxwriter/shared_strings.h"

// Test assembling a complete Worksheet file.
CTEST(worksheet, worksheet_condtional_format05) {

    char* got;
    char exp[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
              "<dimension ref=\"A1:A4\"/>"
              "<sheetViews>"
                "<sheetView tabSelected=\"1\" workbookViewId=\"0\"/>"
              "</sheetViews>"
              "<sheetFormatPr defaultRowHeight=\"15\"/>"
              "<sheetData>"
                "<row r=\"1\" spans=\"1:1\">"
                  "<c r=\"A1\">"
                    "<v>10</v>"
                  "</c>"
                "</row>"
                "<row r=\"2\" spans=\"1:1\">"
                  "<c r=\"A2\">"
                    "<v>20</v>"
                  "</c>"
                "</row>"
                "<row r=\"3\" spans=\"1:1\">"
                  "<c r=\"A3\">"
                    "<v>30</v>"
                  "</c>"
                "</row>"
                "<row r=\"4\" spans=\"1:1\">"
                  "<c r=\"A4\">"
                    "<v>40</v>"
                  "</c>"
                "</row>"
              "</sheetData>"
              "<conditionalFormatting sqref=\"A1:A4\">"
                "<cfRule type=\"aboveAverage\" priority=\"1\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"2\" aboveAverage=\"0\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"3\" equalAverage=\"1\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"4\" aboveAverage=\"0\" equalAverage=\"1\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"5\" stdDev=\"1\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"6\" aboveAverage=\"0\" stdDev=\"1\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"7\" stdDev=\"2\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"8\" aboveAverage=\"0\" stdDev=\"2\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"9\" stdDev=\"3\"/>"
                "<cfRule type=\"aboveAverage\" priority=\"10\" aboveAverage=\"0\" stdDev=\"3\"/>"
              "</conditionalFormatting>"
              "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>"
            "</worksheet>";

    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;
    worksheet_select(worksheet);

    worksheet_write_number(worksheet, CELL("A1"), 10, NULL);
    worksheet_write_number(worksheet, CELL("A2"), 20, NULL);
    worksheet_write_number(worksheet, CELL("A3"), 30, NULL);
    worksheet_write_number(worksheet, CELL("A4"), 40, NULL);

    lxw_conditional_format *conditional_format = calloc(1, sizeof(lxw_conditional_format));

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_ABOVE;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_BELOW;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_ABOVE_OR_EQUAL;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_BELOW_OR_EQUAL;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_1_STD_DEV_ABOVE;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_1_STD_DEV_BELOW;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_2_STD_DEV_ABOVE;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_2_STD_DEV_BELOW;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_3_STD_DEV_ABOVE;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    conditional_format->type     = LXW_CONDITIONAL_TYPE_AVERAGE;
    conditional_format->criteria = LXW_CONDITIONAL_CRITERIA_AVERAGE_3_STD_DEV_BELOW;
    worksheet_conditional_format_range(worksheet, RANGE("A1:A4"), conditional_format);

    free(conditional_format);

    lxw_worksheet_assemble_xml_file(worksheet);

    RUN_XLSX_STREQ_SHORT(exp, got);

    lxw_worksheet_free(worksheet);
}
