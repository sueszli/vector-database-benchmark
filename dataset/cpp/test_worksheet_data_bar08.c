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
CTEST(worksheet, worksheet_data_bar08) {

    char* got;
    char exp[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
              "<dimension ref=\"A1\"/>"
              "<sheetViews>"
                "<sheetView tabSelected=\"1\" workbookViewId=\"0\"/>"
              "</sheetViews>"
              "<sheetFormatPr defaultRowHeight=\"15\"/>"
              "<sheetData/>"
              "<conditionalFormatting sqref=\"A1\">"
                "<cfRule type=\"dataBar\" priority=\"1\">"
                  "<dataBar showValue=\"0\">"
                    "<cfvo type=\"min\" val=\"0\"/>"
                    "<cfvo type=\"max\" val=\"0\"/>"
                    "<color rgb=\"FF638EC6\"/>"
                  "</dataBar>"
                "</cfRule>"
              "</conditionalFormatting>"
              "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>"
            "</worksheet>";

    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;
    worksheet_select(worksheet);

    lxw_conditional_format *conditional_format = calloc(1, sizeof(lxw_conditional_format));

    conditional_format->type                           = LXW_CONDITIONAL_DATA_BAR;
    conditional_format->bar_only                       = LXW_TRUE;
    worksheet_conditional_format_cell(worksheet, CELL("A1"), conditional_format);

    free(conditional_format);
    lxw_worksheet_assemble_xml_file(worksheet);

    RUN_XLSX_STREQ_SHORT(exp, got);

    lxw_worksheet_free(worksheet);
}
