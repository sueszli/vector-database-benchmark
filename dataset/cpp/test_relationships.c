/*
 * Tests for the lib_xlsx_writer library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/relationships.h"

// Test assembling a complete Relationships file.
CTEST(relationships, relationships01) {

    char* got;
    char exp[] =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
          "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
          "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme\" Target=\"theme/theme1.xml\"/>"
          "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
          "<Relationship Id=\"rId4\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>"
          "<Relationship Id=\"rId5\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/calcChain\" Target=\"calcChain.xml\"/>"
        "</Relationships>";

    FILE* testfile = lxw_tmpfile(NULL);

    lxw_relationships *rels = lxw_relationships_new();
    rels->file = testfile;

    lxw_add_document_relationship(rels, "/worksheet",     "worksheets/sheet1.xml");
    lxw_add_document_relationship(rels, "/theme",         "theme/theme1.xml");
    lxw_add_document_relationship(rels, "/styles",        "styles.xml");
    lxw_add_document_relationship(rels, "/sharedStrings", "sharedStrings.xml");
    lxw_add_document_relationship(rels, "/calcChain",     "calcChain.xml");


    lxw_relationships_assemble_xml_file(rels);

    RUN_XLSX_STREQ_SHORT(exp, got);

    lxw_free_relationships(rels);
}

// Test assembling a complete Relationships file.
CTEST(relationships, relationships02) {

    char* got;
    char exp[] =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
          "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink\" Target=\"www.foo.com\" TargetMode=\"External\"/>"
          "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink\" Target=\"link00.xlsx\" TargetMode=\"External\"/>"
        "</Relationships>";

    FILE* testfile = lxw_tmpfile(NULL);

    lxw_relationships *rels = lxw_relationships_new();
    rels->file = testfile;

    lxw_add_worksheet_relationship(rels, "/hyperlink", "www.foo.com", "External");
    lxw_add_worksheet_relationship(rels, "/hyperlink", "link00.xlsx", "External");

    lxw_relationships_assemble_xml_file(rels);

    RUN_XLSX_STREQ_SHORT(exp, got);

    lxw_free_relationships(rels);
}
