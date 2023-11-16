/*
 * Tests for the libxlsxwriter library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/shared_strings.h"

// Test _xml_declaration().
CTEST(sst, xml_declaration) {

    char* got;
    char exp[] = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
    FILE* testfile = lxw_tmpfile(NULL);

    lxw_sst *sst = lxw_sst_new();
    sst->file = testfile;

    _sst_xml_declaration(sst);

    RUN_XLSX_STREQ(exp, got);

    lxw_sst_free(sst);
}
