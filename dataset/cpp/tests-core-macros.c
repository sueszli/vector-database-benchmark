/*
 * Copyright (C) 2022 ML!PA Consulting GmbH
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include "embUnit.h"
#include "tests-core.h"
#include "macros/math.h"

static void test_math_signof(void)
{
    TEST_ASSERT_EQUAL_INT(1, SIGNOF(3));
    TEST_ASSERT_EQUAL_INT(-1, SIGNOF(-3));
    TEST_ASSERT_EQUAL_INT(-1, SIGNOF(INT32_MIN));
    TEST_ASSERT_EQUAL_INT(1, SIGNOF(UINT32_MAX));
}

static void test_math_div_round(void)
{
    size_t a = 20;

    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(a, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(21, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(22, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(-20, -7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(-21, -7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND(-22, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(20, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(21, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(22, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(-20, 7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(-21, 7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND(-22, 7));
    TEST_ASSERT_EQUAL_INT(0, DIV_ROUND(1, (long long) UINT32_MAX));

}

static void test_math_div_round_up(void)
{
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_UP(20, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_UP(21, 7));
    TEST_ASSERT_EQUAL_INT(4, DIV_ROUND_UP(22, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_UP(-20, -7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_UP(-21, -7));
    TEST_ASSERT_EQUAL_INT(4, DIV_ROUND_UP(-22, -7));
    TEST_ASSERT_EQUAL_INT(-2, DIV_ROUND_UP(20, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_UP(21, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_UP(22, -7));
    TEST_ASSERT_EQUAL_INT(-2, DIV_ROUND_UP(-20, 7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_UP(-21, 7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_UP(-22, 7));
    TEST_ASSERT_EQUAL_INT(1, DIV_ROUND_UP(1, (long long) UINT32_MAX));
}

static void test_math_div_round_inf(void)
{
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_INF(20, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_INF(21, 7));
    TEST_ASSERT_EQUAL_INT(4, DIV_ROUND_INF(22, 7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_INF(-20, -7));
    TEST_ASSERT_EQUAL_INT(3, DIV_ROUND_INF(-21, -7));
    TEST_ASSERT_EQUAL_INT(4, DIV_ROUND_INF(-22, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_INF(20, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_INF(21, -7));
    TEST_ASSERT_EQUAL_INT(-4, DIV_ROUND_INF(22, -7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_INF(-20, 7));
    TEST_ASSERT_EQUAL_INT(-3, DIV_ROUND_INF(-21, 7));
    TEST_ASSERT_EQUAL_INT(-4, DIV_ROUND_INF(-22, 7));
    TEST_ASSERT_EQUAL_INT(1, DIV_ROUND_INF(1, UINT32_MAX));
    TEST_ASSERT_EQUAL_INT(-1, DIV_ROUND_INF(1, (long long) INT32_MIN));

    TEST_ASSERT_EQUAL_INT(1536, MATH_ALIGN(1514, 64));
}

Test *tests_core_macros_tests(void)
{
    EMB_UNIT_TESTFIXTURES(fixtures) {
        new_TestFixture(test_math_signof),
        new_TestFixture(test_math_div_round),
        new_TestFixture(test_math_div_round_up),
        new_TestFixture(test_math_div_round_inf),
   };

    EMB_UNIT_TESTCALLER(core_macros_tests, NULL, NULL, fixtures);

    return (Test *)&core_macros_tests;
}
