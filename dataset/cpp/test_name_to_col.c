/*
 * Tests for the libxlsxwriter library.
 *
 * Copyright 2014-2022, John McNamara, jmcnamara@cpan.org
 *
 */

#include "../ctest.h"
#include "../helper.h"

#include "../../../include/xlsxwriter/utility.h"

// Test _xl_get_col().
CTEST(utility, lxw_name_to_col) {

    ASSERT_EQUAL(0,     lxw_name_to_col("A1"));
    ASSERT_EQUAL(1,     lxw_name_to_col("B1"));
    ASSERT_EQUAL(2,     lxw_name_to_col("C1"));
    ASSERT_EQUAL(9,     lxw_name_to_col("J1"));
    ASSERT_EQUAL(24,    lxw_name_to_col("Y1"));
    ASSERT_EQUAL(25,    lxw_name_to_col("Z1"));
    ASSERT_EQUAL(26,    lxw_name_to_col("AA1"));
    ASSERT_EQUAL(254,   lxw_name_to_col("IU1"));
    ASSERT_EQUAL(255,   lxw_name_to_col("IV1"));
    ASSERT_EQUAL(256,   lxw_name_to_col("IW1"));
    ASSERT_EQUAL(16383, lxw_name_to_col("XFD1"));
    ASSERT_EQUAL(16384, lxw_name_to_col("XFE1"));
    ASSERT_EQUAL(0,     lxw_name_to_col("$A1"));
    ASSERT_EQUAL(0,     lxw_name_to_col("A$1"));
    ASSERT_EQUAL(0,     lxw_name_to_col("$A$1"));
}


// Test _xl_get_col_2().
CTEST(utility, lxw_name_to_col_2) {

    ASSERT_EQUAL(0,     lxw_name_to_col_2("AAA:A"));
    ASSERT_EQUAL(1,     lxw_name_to_col_2("AAA:B"));
    ASSERT_EQUAL(2,     lxw_name_to_col_2("AAA:C"));
    ASSERT_EQUAL(9,     lxw_name_to_col_2("AAA:J"));
    ASSERT_EQUAL(24,    lxw_name_to_col_2("AAA:Y"));
    ASSERT_EQUAL(25,    lxw_name_to_col_2("AAA:Z"));
    ASSERT_EQUAL(26,    lxw_name_to_col_2("AAA:AA"));
    ASSERT_EQUAL(254,   lxw_name_to_col_2("AAA:IU"));
    ASSERT_EQUAL(255,   lxw_name_to_col_2("AAA:IV"));
    ASSERT_EQUAL(256,   lxw_name_to_col_2("AAA:IW"));
    ASSERT_EQUAL(16383, lxw_name_to_col_2("AAA:XFD"));
    ASSERT_EQUAL(16384, lxw_name_to_col_2("AAA:XFE"));
    ASSERT_EQUAL(16384, lxw_name_to_col_2("AAA1:XFE1"));
    ASSERT_EQUAL(16384, lxw_name_to_col_2("$AAA:$XFE"));
}
