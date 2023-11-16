/*
 * Copyright © 2021 Embedded Artistry LLC.
 * See LICENSE file for licensing information.
 */

// Cmocka needs these
// clang-format off
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
// clang-format on

#include <simple_fixed_point.h>
#include <fixed_point_tests.h>
#include <stdio.h>

/// Want to check some of these values for yourself?
/// Verify them with this page:
/// https://www.rfwireless-world.com/calculators/floating-vs-fixed-point-converter.html

static void double_to_fixed16_test(__attribute__((unused)) void** state)
{
	fixed_point_t output_round = double_to_fixed_round(11.5);
	fixed_point_t output_truncate = double_to_fixed_truncate(11.5);
	assert_int_equal(0x170, output_truncate);
	assert_int_equal(output_round, output_truncate);

	output_round = double_to_fixed_round(128);
	output_truncate = double_to_fixed_truncate(128);
	assert_int_equal(0x1000, output_truncate);
	assert_int_equal(output_round, output_truncate);

#ifndef USE_UNSIGNED_FIXED_POINT
	output_round = double_to_fixed_round(-128);
	output_truncate = double_to_fixed_truncate(-128);
	assert_int_equal((int16_t)0xf000, output_truncate);
	assert_int_equal(output_round, output_truncate);

	output_round = double_to_fixed_round(-0.968750);
	output_truncate = double_to_fixed_truncate(-0.968750);
	assert_int_equal((int16_t)0xffe1, output_truncate);
	assert_int_equal(output_round, output_truncate);
#endif

	output_round = double_to_fixed_round(128.28);
	output_truncate = double_to_fixed_truncate(128.28);
	assert_int_equal(0x1009, output_round);
	// Here, truncate loses precision vs round
	assert_int_equal(0x1008, output_truncate);

#ifndef USE_UNSIGNED_FIXED_POINT
	output_round = double_to_fixed_round(-64.28);
	output_truncate = double_to_fixed_truncate(-64.28);
	assert_int_equal((int16_t)0xf7f7, output_round);
	// Here, truncate loses precision vs round
	assert_int_equal((int16_t)0xf7f8, output_truncate);
#endif
}

static void fixed16_to_double_test(__attribute__((unused)) void** state)
{
	double output = fixed_to_double(0x1000);
	assert_float_equal(128.0, output, 0.01);

#ifndef USE_UNSIGNED_FIXED_POINT
	output = fixed_to_double(0xf000);
	assert_float_equal(-128.0, output, 0.01);

	output = fixed_to_double(0xf7f7);
	assert_float_equal(-64.28125, output, 0.01);

	// Check case presented in website comments
	output = fixed_to_double(0xffe1);
	assert_float_equal(-0.968750, output, 0.0001);
#endif

	output = fixed_to_double(0x1009);
	assert_float_equal(128.28125, output, 0.01);

	output = fixed_to_double(0x1008);
	assert_float_equal(128.25, output, 0.01);

	output = fixed_to_double(0x1F);
	assert_float_equal(0.968750, output, 0.0001);

	// Confirm equivalency
	output = fixed16_to_double(0x1008, 5);
	assert_float_equal(128.25, output, 0.01);

	// Check an alternate FP strategy - 10.6
	output = fixed16_to_double(0x1008, 6);
	assert_float_equal(64.125, output, 0.01);
}

#pragma mark - Public Functions -

int simple_fixed_point_test_suite(void)
{
	const struct CMUnitTest simple_fixed_point_tests[] = {
		cmocka_unit_test(double_to_fixed16_test),
		cmocka_unit_test(fixed16_to_double_test),
	};

	return cmocka_run_group_tests(simple_fixed_point_tests, NULL, NULL);
}
