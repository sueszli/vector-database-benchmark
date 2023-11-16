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
CTEST(worksheet, worksheet_data_bar05) {

    char* got;
    char exp[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" xmlns:x14ac=\"http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac\" mc:Ignorable=\"x14ac\">"
              "<dimension ref=\"A1\"/>"
              "<sheetViews>"
                "<sheetView tabSelected=\"1\" workbookViewId=\"0\"/>"
              "</sheetViews>"
              "<sheetFormatPr defaultRowHeight=\"15\" x14ac:dyDescent=\"0.25\"/>"
              "<sheetData/>"
              "<conditionalFormatting sqref=\"A1\">"
                "<cfRule type=\"dataBar\" priority=\"1\">"
                  "<dataBar>"
                    "<cfvo type=\"min\"/>"
                    "<cfvo type=\"max\"/>"
                    "<color rgb=\"FF638EC6\"/>"
                  "</dataBar>"
                  "<extLst>"
                    "<ext xmlns:x14=\"http://schemas.microsoft.com/office/spreadsheetml/2009/9/main\" uri=\"{B025F937-C7B1-47D3-B67F-A62EFF666E3E}\">"
                      "<x14:id>{DA7ABA51-AAAA-BBBB-0001-000000000001}</x14:id>"
                    "</ext>"
                  "</extLst>"
                "</cfRule>"
              "</conditionalFormatting>"
              "<conditionalFormatting sqref=\"A2:B2\">"
                "<cfRule type=\"dataBar\" priority=\"2\">"
                  "<dataBar>"
                    "<cfvo type=\"min\"/>"
                    "<cfvo type=\"max\"/>"
                    "<color rgb=\"FF63C384\"/>"
                  "</dataBar>"
                  "<extLst>"
                    "<ext xmlns:x14=\"http://schemas.microsoft.com/office/spreadsheetml/2009/9/main\" uri=\"{B025F937-C7B1-47D3-B67F-A62EFF666E3E}\">"
                      "<x14:id>{DA7ABA51-AAAA-BBBB-0001-000000000002}</x14:id>"
                    "</ext>"
                  "</extLst>"
                "</cfRule>"
              "</conditionalFormatting>"
              "<conditionalFormatting sqref=\"A3:C3\">"
                "<cfRule type=\"dataBar\" priority=\"3\">"
                  "<dataBar>"
                    "<cfvo type=\"min\"/>"
                    "<cfvo type=\"max\"/>"
                    "<color rgb=\"FFFF555A\"/>"
                  "</dataBar>"
                  "<extLst>"
                    "<ext xmlns:x14=\"http://schemas.microsoft.com/office/spreadsheetml/2009/9/main\" uri=\"{B025F937-C7B1-47D3-B67F-A62EFF666E3E}\">"
                      "<x14:id>{DA7ABA51-AAAA-BBBB-0001-000000000003}</x14:id>"
                    "</ext>"
                  "</extLst>"
                "</cfRule>"
              "</conditionalFormatting>"
              "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>"
              "<extLst>"
                "<ext xmlns:x14=\"http://schemas.microsoft.com/office/spreadsheetml/2009/9/main\" uri=\"{78C0D931-6437-407d-A8EE-F0AAD7539E65}\">"
                  "<x14:conditionalFormattings>"
                    "<x14:conditionalFormatting xmlns:xm=\"http://schemas.microsoft.com/office/excel/2006/main\">"
                      "<x14:cfRule type=\"dataBar\" id=\"{DA7ABA51-AAAA-BBBB-0001-000000000001}\">"
                        "<x14:dataBar minLength=\"0\" maxLength=\"100\" border=\"1\" direction=\"leftToRight\" negativeBarBorderColorSameAsPositive=\"0\">"
                          "<x14:cfvo type=\"autoMin\"/>"
                          "<x14:cfvo type=\"autoMax\"/>"
                          "<x14:borderColor rgb=\"FF638EC6\"/>"
                          "<x14:negativeFillColor rgb=\"FFFF0000\"/>"
                          "<x14:negativeBorderColor rgb=\"FFFF0000\"/>"
                          "<x14:axisColor rgb=\"FF000000\"/>"
                        "</x14:dataBar>"
                      "</x14:cfRule>"
                      "<xm:sqref>A1</xm:sqref>"
                    "</x14:conditionalFormatting>"
                    "<x14:conditionalFormatting xmlns:xm=\"http://schemas.microsoft.com/office/excel/2006/main\">"
                      "<x14:cfRule type=\"dataBar\" id=\"{DA7ABA51-AAAA-BBBB-0001-000000000002}\">"
                        "<x14:dataBar minLength=\"0\" maxLength=\"100\" border=\"1\" direction=\"rightToLeft\" negativeBarBorderColorSameAsPositive=\"0\">"
                          "<x14:cfvo type=\"autoMin\"/>"
                          "<x14:cfvo type=\"autoMax\"/>"
                          "<x14:borderColor rgb=\"FF63C384\"/>"
                          "<x14:negativeFillColor rgb=\"FFFF0000\"/>"
                          "<x14:negativeBorderColor rgb=\"FFFF0000\"/>"
                          "<x14:axisColor rgb=\"FF000000\"/>"
                        "</x14:dataBar>"
                      "</x14:cfRule>"
                      "<xm:sqref>A2:B2</xm:sqref>"
                    "</x14:conditionalFormatting>"
                    "<x14:conditionalFormatting xmlns:xm=\"http://schemas.microsoft.com/office/excel/2006/main\">"
                      "<x14:cfRule type=\"dataBar\" id=\"{DA7ABA51-AAAA-BBBB-0001-000000000003}\">"
                        "<x14:dataBar minLength=\"0\" maxLength=\"100\" border=\"1\" negativeBarBorderColorSameAsPositive=\"0\">"
                          "<x14:cfvo type=\"autoMin\"/>"
                          "<x14:cfvo type=\"autoMax\"/>"
                          "<x14:borderColor rgb=\"FFFF555A\"/>"
                          "<x14:negativeFillColor rgb=\"FFFFFF00\"/>"
                          "<x14:negativeBorderColor rgb=\"FFFF0000\"/>"
                          "<x14:axisColor rgb=\"FF000000\"/>"
                        "</x14:dataBar>"
                      "</x14:cfRule>"
                      "<xm:sqref>A3:C3</xm:sqref>"
                    "</x14:conditionalFormatting>"
                  "</x14:conditionalFormattings>"
                "</ext>"
              "</extLst>"
            "</worksheet>";

    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;
    worksheet_select(worksheet);

    lxw_conditional_format *conditional_format = calloc(1, sizeof(lxw_conditional_format));

    conditional_format->type               = LXW_CONDITIONAL_DATA_BAR;
    conditional_format->bar_direction      = LXW_CONDITIONAL_BAR_DIRECTION_LEFT_TO_RIGHT;
    worksheet_conditional_format_cell(worksheet, CELL("A1"), conditional_format);
    memset(conditional_format, 0, sizeof(lxw_conditional_format));

    conditional_format->type               = LXW_CONDITIONAL_DATA_BAR;
    conditional_format->bar_color          = 0x63C384;
    conditional_format->bar_direction      = LXW_CONDITIONAL_BAR_DIRECTION_RIGHT_TO_LEFT;
    worksheet_conditional_format_range(worksheet, RANGE("A2:B2"), conditional_format);
    memset(conditional_format, 0, sizeof(lxw_conditional_format));

    conditional_format->type               = LXW_CONDITIONAL_DATA_BAR;
    conditional_format->bar_color          = 0xFF555A;
    conditional_format->bar_negative_color = 0xFFFF00;
    worksheet_conditional_format_range(worksheet, RANGE("A3:C3"), conditional_format);

    free(conditional_format);
    lxw_worksheet_assemble_xml_file(worksheet);

    RUN_XLSX_STREQ_SHORT(exp, got);

    lxw_worksheet_free(worksheet);
}
