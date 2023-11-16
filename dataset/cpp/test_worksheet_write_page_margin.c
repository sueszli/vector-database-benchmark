/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/worksheet.h"

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin01) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" "
                 "bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin02) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" "
                 "bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_margins(worksheet, -1, -1, -1, -1);
    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin03) {
    char* got;
    char exp[] = "<pageMargins left=\"0.8\" right=\"0.7\" top=\"0.75\" "
                 "bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_margins(worksheet, 0.8, -1, -1, -1);
    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin04) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.8\" top=\"0.75\" "
                 "bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_margins(worksheet, -1, 0.8, -1, -1);
    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin05) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.8\" "
                 "bottom=\"0.75\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_margins(worksheet, -1, -1, 0.8, -1);
    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin06) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" "
                 "bottom=\"0.8\" header=\"0.3\" footer=\"0.3\"/>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_margins(worksheet, -1, -1, -1, 0.8);
    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

/* Test the _write_page_margins() method. */
CTEST(worksheet, write_page_margin07) {
    char* got;
    char exp[] = "<pageMargins left=\"0.7\" right=\"0.7\" top=\"0.75\" "
                 "bottom=\"0.75\" header=\"0.2\" footer=\"0.4\"/>";
    FILE* testfile = lxw_tmpfile(NULL);
    lxw_header_footer_options header_options = {.margin = 0.2};
    lxw_header_footer_options footer_options = {.margin = 0.4};

    lxw_worksheet *worksheet = lxw_worksheet_new(NULL);
    worksheet->file = testfile;

    worksheet_set_header_opt(worksheet, "", &header_options);
    worksheet_set_footer_opt(worksheet, "", &footer_options);

    _worksheet_write_page_margins(worksheet);

    RUN_XLSX_STREQ(exp, got);

    lxw_worksheet_free(worksheet);
}

