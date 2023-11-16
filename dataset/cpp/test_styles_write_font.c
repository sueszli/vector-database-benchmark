/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/styles.h"

// Test the _write_font() function.
CTEST(styles, write_font01) {


    char* got;
    char exp[] = "<font><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font02) {


    char* got;
    char exp[] = "<font><b/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_bold(format);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font03) {


    char* got;
    char exp[] = "<font><i/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_italic(format);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font04) {


    char* got;
    char exp[] = "<font><u/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_underline(format, LXW_UNDERLINE_SINGLE);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font05) {


    char* got;
    char exp[] = "<font><strike/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_strikeout(format);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font06) {


    char* got;
    char exp[] = "<font><vertAlign val=\"superscript\"/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_script(format, LXW_FONT_SUPERSCRIPT);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font07) {


    char* got;
    char exp[] = "<font><vertAlign val=\"subscript\"/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_script(format, LXW_FONT_SUBSCRIPT);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font08) {


    char* got;
    char exp[] = "<font><sz val=\"11\"/><color theme=\"1\"/><name val=\"Arial\"/><family val=\"2\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_name(format, "Arial");

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font09) {


    char* got;
    char exp[] = "<font><sz val=\"12\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_size(format, 12);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font10) {


    char* got;
    char exp[] = "<font><outline/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_outline(format);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font11) {


    char* got;
    char exp[] = "<font><shadow/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_shadow(format);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font12) {


    char* got;
    char exp[] = "<font><sz val=\"11\"/><color rgb=\"FFFF0000\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_font_color(format, LXW_COLOR_RED);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font13) {


    char* got;
    char exp[] = "<font><b/><i/><strike/><outline/><shadow/><u/><vertAlign val=\"superscript\"/><sz val=\"12\"/><color rgb=\"FFFF0000\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_bold(format);
    format_set_italic(format);
    format_set_font_size(format, 12);
    format_set_font_color(format, LXW_COLOR_RED);
    format_set_font_strikeout(format);
    format_set_font_outline(format);
    format_set_font_shadow(format);
    format_set_font_script(format, LXW_FONT_SUPERSCRIPT);
    format_set_underline(format, LXW_UNDERLINE_SINGLE);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font14) {


    char* got;
    char exp[] = "<font><u val=\"double\"/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_underline(format, LXW_UNDERLINE_DOUBLE);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font15) {


    char* got;
    char exp[] = "<font><u val=\"singleAccounting\"/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_underline(format, LXW_UNDERLINE_SINGLE_ACCOUNTING);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}

// Test the _write_font() function.
CTEST(styles, write_font16) {


    char* got;
    char exp[] = "<font><u val=\"doubleAccounting\"/><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/><scheme val=\"minor\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_underline(format, LXW_UNDERLINE_DOUBLE_ACCOUNTING);

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}



// Test the _write_font() function.
CTEST(styles, write_font17) {


    char* got;
    char exp[] = "<font><u/><sz val=\"11\"/><color theme=\"10\"/><name val=\"Calibri\"/><family val=\"2\"/></font>";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_styles *styles = lxw_styles_new();
    lxw_format *format = lxw_format_new();

    format_set_underline(format, LXW_UNDERLINE_SINGLE);
    format_set_theme(format, 10);
    format->hyperlink = 1;

    styles->file = testfile;

    _write_font(styles, format, LXW_FALSE, LXW_FALSE);

    RUN_XLSX_STREQ(exp, got);

    lxw_styles_free(styles);
    lxw_format_free(format);
}
