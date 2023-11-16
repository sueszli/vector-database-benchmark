#include "common.h"
#include "vnstat_tests.h"
#include "common_tests.h"
#include "dbaccess.h"
#include "cfg.h"

START_TEST(printe_options)
{
	noexit = 2;
	cfg.uselogging = 0;
	ck_assert_int_eq(printe(PT_Info), 1);

	cfg.uselogging = 1;
	ck_assert_int_eq(printe(PT_Multiline), 1);

	noexit = 0;
	strcpy(errorstring, "dummy string");
	suppress_output();
	ck_assert_int_eq(printe(PT_Info), 1);
	ck_assert_int_eq(printe(PT_Warning), 1);
	ck_assert_int_eq(printe(PT_Error), 1);
	ck_assert_int_eq(printe(PT_Config), 1);
	ck_assert_int_eq(printe(PT_Multiline), 1);
	ck_assert_int_eq(printe(PT_ShortMultiline), 1);
	ck_assert_int_eq(printe(6), 1);
}
END_TEST

START_TEST(logprint_options)
{
	cfg.uselogging = 0;
	ck_assert_int_eq(logprint(PT_Info), 0);

	cfg.uselogging = 1;
	strcpy(cfg.logfile, "/dev/null");
	strcpy(errorstring, "dummy string");
	ck_assert_int_eq(logprint(PT_Info), 1);
	ck_assert_int_eq(logprint(PT_Warning), 1);
	ck_assert_int_eq(logprint(PT_Error), 1);
	ck_assert_int_eq(logprint(PT_Config), 1);
	ck_assert_int_eq(logprint(PT_Multiline), 0);
	ck_assert_int_eq(logprint(PT_ShortMultiline), 1);
	ck_assert_int_eq(logprint(6), 1);
}
END_TEST

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#endif
START_TEST(dmonth_return_within_range)
{
	int m;
	m = dmonth(_i);
	ck_assert_int_ge(m, 28);
	ck_assert_int_le(m, 31);
}
END_TEST
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

START_TEST(leapyears_are_known)
{
	ck_assert_int_eq(isleapyear(1995), 0);
	ck_assert_int_eq(isleapyear(1996), 1);
	ck_assert_int_eq(isleapyear(1997), 0);
	ck_assert_int_eq(isleapyear(1998), 0);
	ck_assert_int_eq(isleapyear(1999), 0);
	ck_assert_int_eq(isleapyear(2000), 1);
	ck_assert_int_eq(isleapyear(2001), 0);
	ck_assert_int_eq(isleapyear(2002), 0);
	ck_assert_int_eq(isleapyear(2003), 0);
	ck_assert_int_eq(isleapyear(2004), 1);
	ck_assert_int_eq(isleapyear(2005), 0);
	ck_assert_int_eq(isleapyear(2006), 0);
	ck_assert_int_eq(isleapyear(2007), 0);
	ck_assert_int_eq(isleapyear(2008), 1);
	ck_assert_int_eq(isleapyear(2009), 0);
	ck_assert_int_eq(isleapyear(2010), 0);
	ck_assert_int_eq(isleapyear(2011), 0);
	ck_assert_int_eq(isleapyear(2012), 1);
	ck_assert_int_eq(isleapyear(2013), 0);
	ck_assert_int_eq(isleapyear(2014), 0);
	ck_assert_int_eq(isleapyear(2015), 0);
	ck_assert_int_eq(isleapyear(2016), 1);
	ck_assert_int_eq(isleapyear(2017), 0);
	ck_assert_int_eq(isleapyear(2018), 0);
	ck_assert_int_eq(isleapyear(2019), 0);
	ck_assert_int_eq(isleapyear(2020), 1);
	ck_assert_int_eq(isleapyear(2021), 0);
}
END_TEST

START_TEST(mosecs_return_values)
{
	time_t a, b;
	cfg.monthrotate = 1;
	ck_assert_int_eq(mosecs(0, 0), 1);

	a = mosecs(172800, 173000);
	ck_assert_int_gt(a, 1);

	cfg.monthrotate = 2;
	b = mosecs(172800, 173000);
	ck_assert_int_gt(b, 1);

	ck_assert_int_gt(a, b);
}
END_TEST

START_TEST(mosecs_does_not_change_tz)
{
#if defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE) || defined(__APPLE__) || defined(__linux__)
	extern long timezone;
#else
	long timezone = 0;
#endif
	long timezone_before_call;

	tzset();
	timezone_before_call = timezone;

	ck_assert_int_eq(cfg.monthrotate, 1);
	ck_assert_int_ne(mosecs(1, 2), 0);
	ck_assert_int_ne(mosecs(1, 2), 1);
	ck_assert_int_eq(timezone_before_call, timezone);
}
END_TEST

START_TEST(mosecs_does_not_change_struct_tm_pointer_content)
{
	struct tm *stm;
	time_t current;

	current = time(NULL);
	stm = localtime(&current);

	ck_assert_int_eq(cfg.monthrotate, 1);
	ck_assert_int_eq(current, mktime(stm));
	ck_assert_int_ne(mosecs(1, 2), 0);
	ck_assert_int_ne(mosecs(1, 2), 1);
	ck_assert_int_eq(current, mktime(stm));
}
END_TEST

START_TEST(countercalc_no_change_32bit)
{
	uint64_t a, b;

	a = b = 0;
	ck_assert_int_eq(countercalc(&a, &b, 0), 0);
	ck_assert_int_eq(countercalc(&a, &b, -1), 0);
	a = b = 1;
	ck_assert_int_eq(countercalc(&a, &b, 0), 0);
	ck_assert_int_eq(countercalc(&a, &b, -1), 0);
}
END_TEST

START_TEST(countercalc_no_change_64bit)
{
	uint64_t a, b;

	a = b = 0;
	ck_assert_int_eq(countercalc(&a, &b, 1), 0);
	a = b = 1;
	ck_assert_int_eq(countercalc(&a, &b, 1), 0);
}
END_TEST

START_TEST(countercalc_small_change_32bit)
{
	uint64_t a, b;

	a = 0;
	b = 1;
	ck_assert_int_eq(countercalc(&a, &b, 0), 1);
	ck_assert_int_eq(countercalc(&a, &b, -1), 1);
	a = 1;
	b = 2;
	ck_assert_int_eq(countercalc(&a, &b, 0), 1);
	ck_assert_int_eq(countercalc(&a, &b, -1), 1);
	b = 3;
	ck_assert_int_eq(countercalc(&a, &b, 0), 2);
	ck_assert_int_eq(countercalc(&a, &b, -1), 2);
}
END_TEST

START_TEST(countercalc_small_change_64bit)
{
	uint64_t a, b;

	a = 0;
	b = 1;
	ck_assert_int_eq(countercalc(&a, &b, 1), 1);
	a = 1;
	b = 2;
	ck_assert_int_eq(countercalc(&a, &b, 1), 1);
	b = 3;
	ck_assert_int_eq(countercalc(&a, &b, 1), 2);
}
END_TEST

START_TEST(countercalc_rollover_with_32bit)
{
	uint64_t a, b;

	a = 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 0) == (MAX32 - 1));
	ck_assert(countercalc(&a, &b, -1) == (MAX32 - 1));
}
END_TEST

START_TEST(countercalc_rollover_with_64bit)
{
	uint64_t a, b;

	a = 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 1) == (MAX64 - 1));
}
END_TEST

START_TEST(countercalc_rollover_with_64bit_2)
{
	uint64_t a, b;

	a = MAX32 + 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 1) == (MAX64 - MAX32 - 1));
}
END_TEST

START_TEST(countercalc_rollover_with_32bit_starting_32bit)
{
	uint64_t a, b;

	a = MAX32 - 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 0) == 1);
	ck_assert(countercalc(&a, &b, -1) == 1);
}
END_TEST

START_TEST(countercalc_rollover_with_32bit_starting_over_32bit)
{
	uint64_t a, b;

	a = MAX32 + 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 0) == (MAX64 - MAX32 - 1));
	ck_assert(countercalc(&a, &b, -1) == (MAX64 - MAX32 - 1));
}
END_TEST

START_TEST(countercalc_rollover_with_64bit_starting_32bit)
{
	uint64_t a, b;

	a = MAX32 - 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 1) == (MAX64 - MAX32 + 1));
}
END_TEST

START_TEST(countercalc_rollover_with_64bit_starting_64bit)
{
	uint64_t a, b;

	a = MAX64 - 1;
	b = 0;
	ck_assert(countercalc(&a, &b, 1) == 1);
}
END_TEST

START_TEST(strncpy_nt_with_below_maximum_length_string)
{
	char dst[6];

	strncpy_nt(dst, "123", 6);
	ck_assert_str_eq(dst, "123");
}
END_TEST

START_TEST(strncpy_nt_with_maximum_length_string)
{
	char dst[6];

	strncpy_nt(dst, "12345", 6);
	ck_assert_str_eq(dst, "12345");
}
END_TEST

START_TEST(strncpy_nt_with_over_maximum_length_string)
{
	char dst[6];

	strncpy_nt(dst, "123456", 6);
	ck_assert_str_eq(dst, "12345");

	strncpy_nt(dst, "1234567890", 6);
	ck_assert_str_eq(dst, "12345");
}
END_TEST

START_TEST(isnumeric_empty)
{
	ck_assert_int_eq(isnumeric(""), 0);
}
END_TEST

START_TEST(isnumeric_it_is)
{
	ck_assert_int_eq(isnumeric("0"), 1);
	ck_assert_int_eq(isnumeric("1"), 1);
	ck_assert_int_eq(isnumeric("12"), 1);
	ck_assert_int_eq(isnumeric("123"), 1);
}
END_TEST

START_TEST(isnumeric_it_is_not)
{
	ck_assert_int_eq(isnumeric("a"), 0);
	ck_assert_int_eq(isnumeric("abc"), 0);
	ck_assert_int_eq(isnumeric("a1"), 0);
	ck_assert_int_eq(isnumeric("1a"), 0);
	ck_assert_int_eq(isnumeric("123abc"), 0);
	ck_assert_int_eq(isnumeric("/"), 0);
	ck_assert_int_eq(isnumeric("-"), 0);
}
END_TEST

START_TEST(getversion_returns_a_version)
{
	ck_assert_int_gt((int)strlen(getversion()), 1);
	ck_assert(strchr(getversion(), '_') == NULL);
	ck_assert(strchr(getversion(), '.') != NULL);
}
END_TEST

START_TEST(timeused_debug_outputs_something_expected_when_debug_is_enabled)
{
	int pipe, len;
	char buffer[512];
	memset(&buffer, '\0', sizeof(buffer));

	debug = 1;
	pipe = pipe_output();
	/* the assumption here is that the next two steps
	   can always execute in less than one second resulting
	   in a duration that starts with a zero */
	timeused_debug("that_func", 1);
	timeused_debug("that_func", 0);
	fflush(stdout);

	len = (int)read(pipe, buffer, 512);
	ck_assert_int_gt(len, 1);
	ck_assert_ptr_ne(strstr(buffer, "that_func() in 0"), NULL);
}
END_TEST

START_TEST(timeused_debug_does_not_output_anything_when_debug_is_disabled)
{
	int pipe, len;
	char buffer[512];
	memset(&buffer, '\0', sizeof(buffer));

	debug = 0;
	pipe = pipe_output();
	/* the assumption here is that the next two steps
	   can always execute in less than one second resulting
	   in a duration that starts with a zero */
	timeused_debug("other_func", 1);
	timeused_debug("other_func", 0);
	printf("-"); // stdout needs to contain something so that read doesn't block
	fflush(stdout);

	len = (int)read(pipe, buffer, 512);
	ck_assert_int_eq(len, 1);
}
END_TEST

START_TEST(timeused_tracks_used_time)
{
	double used;
	struct timespec ts;


	used = timeused("quick_func", 1);
	ck_assert(used == 0.0);

	ts.tv_sec = 0;
	ts.tv_nsec = 100000000; // 0.1 s
	nanosleep(&ts, NULL);

	used = timeused("quick_func", 0);
	ck_assert(used > 0.0);
}
END_TEST

__attribute__((noreturn))
START_TEST(can_panic)
{
	suppress_output();
	fclose(stderr);
	panicexit(__FILE__, __LINE__);
}
END_TEST

void add_common_tests(Suite *s)
{
	TCase *tc_common = tcase_create("Common");
	tcase_add_checked_fixture(tc_common, setup, teardown);
	tcase_add_unchecked_fixture(tc_common, setup, teardown);
	tcase_add_test(tc_common, printe_options);
	tcase_add_test(tc_common, logprint_options);
	tcase_add_loop_test(tc_common, dmonth_return_within_range, 0, 12);
	tcase_add_test(tc_common, leapyears_are_known);
	tcase_add_test(tc_common, mosecs_return_values);
	tcase_add_test(tc_common, mosecs_does_not_change_tz);
	tcase_add_test(tc_common, mosecs_does_not_change_struct_tm_pointer_content);
	tcase_add_test(tc_common, countercalc_no_change_32bit);
	tcase_add_test(tc_common, countercalc_no_change_64bit);
	tcase_add_test(tc_common, countercalc_small_change_32bit);
	tcase_add_test(tc_common, countercalc_small_change_64bit);
	tcase_add_test(tc_common, countercalc_rollover_with_32bit);
	tcase_add_test(tc_common, countercalc_rollover_with_64bit);
	tcase_add_test(tc_common, countercalc_rollover_with_64bit_2);
	tcase_add_test(tc_common, countercalc_rollover_with_32bit_starting_32bit);
	tcase_add_test(tc_common, countercalc_rollover_with_32bit_starting_over_32bit);
	tcase_add_test(tc_common, countercalc_rollover_with_64bit_starting_32bit);
	tcase_add_test(tc_common, countercalc_rollover_with_64bit_starting_64bit);
	tcase_add_test(tc_common, strncpy_nt_with_below_maximum_length_string);
	tcase_add_test(tc_common, strncpy_nt_with_maximum_length_string);
	tcase_add_test(tc_common, strncpy_nt_with_over_maximum_length_string);
	tcase_add_test(tc_common, isnumeric_empty);
	tcase_add_test(tc_common, isnumeric_it_is);
	tcase_add_test(tc_common, isnumeric_it_is_not);
	tcase_add_test(tc_common, getversion_returns_a_version);
	tcase_add_test(tc_common, timeused_debug_outputs_something_expected_when_debug_is_enabled);
	tcase_add_test(tc_common, timeused_debug_does_not_output_anything_when_debug_is_disabled);
	tcase_add_test(tc_common, timeused_tracks_used_time);
	tcase_add_exit_test(tc_common, can_panic, 1);
	suite_add_tcase(s, tc_common);
}
