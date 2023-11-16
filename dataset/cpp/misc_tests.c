#include "common.h"
#include "vnstat_tests.h"
#include "misc_tests.h"
#include "misc.h"
#include "dbsql.h"

START_TEST(getbtime_does_not_return_zero)
{
	ck_assert_int_gt(getbtime(), 0);
}
END_TEST

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#endif
START_TEST(getunitprefix_returns_something_with_all_cfg_combinations)
{
	const char *string;
	int j;

	cfg.unitmode = _i;
	for (j = 1; j <= (UNITPREFIXCOUNT + 1); j++) {
		string = getunitprefix(j);
		ck_assert_int_gt(strlen(string), 0);
	}
}
END_TEST

START_TEST(getrateunitprefix_returns_something_with_all_cfg_combinations)
{
	const char *string;
	int j;

	for (j = 1; j <= (UNITPREFIXCOUNT + 1); j++) {
		string = getrateunitprefix(_i, j);
		ck_assert_int_gt(strlen(string), 0);
	}
}
END_TEST

START_TEST(getunitdivisor_returns_something_with_all_cfg_combinations)
{
	int j;
	char div[16];

	for (j = 1; j <= (UNITPREFIXCOUNT + 1); j++) {
		snprintf(div, 15, "%" PRIu64 "", getunitdivisor(_i, j));
		if (j > UNITPREFIXCOUNT) {
			ck_assert_str_eq(div, "1");
		} else {
			ck_assert_str_ne(div, "0");
		}
	}
}
END_TEST
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

START_TEST(spacecheck_does_not_check_when_not_configured)
{
	cfg.spacecheck = 0;
	ck_assert_int_eq(spacecheck("/nonexistentpath"), 1);
}
END_TEST

START_TEST(spacecheck_checks_space)
{
	cfg.spacecheck = 1;
	/* it's assumed that /tmp isn't full */
	ck_assert_int_eq(spacecheck("/tmp"), 1);
}
END_TEST

START_TEST(spacecheck_fails_with_invalid_path)
{
	noexit = 1;
	cfg.spacecheck = 1;
	ck_assert_int_eq(spacecheck("/nonexistentpath"), 0);
}
END_TEST

START_TEST(getvalue_normal)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1.00 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1.00 MiB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_Normal), "1.00 GiB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_Normal), "1.00 TiB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_Normal), "1.00 PiB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_Normal), "1.00 EiB");
	cfg.unitmode = 1;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1.00 KB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1.00 MB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_Normal), "1.00 GB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_Normal), "1.00 TB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_Normal), "1.00 PB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_Normal), "1.00 EB");
	cfg.unitmode = 2;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1000, 0, RT_Normal), "1.00 kB");
	ck_assert_str_eq(getvalue(1000000, 0, RT_Normal), "1.00 MB");
	ck_assert_str_eq(getvalue(1000000000, 0, RT_Normal), "1.00 GB");
	ck_assert_str_eq(getvalue(1000000000000ULL, 0, RT_Normal), "1.00 TB");
	ck_assert_str_eq(getvalue(1000000000000000ULL, 0, RT_Normal), "1.00 PB");
	ck_assert_str_eq(getvalue(1000000000000000000ULL, 0, RT_Normal), "1.00 EB");
}
END_TEST

START_TEST(getvalue_estimate)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1.00 MiB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_Estimate), "1.00 GiB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_Estimate), "1.00 TiB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_Estimate), "1.00 PiB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_Estimate), "1.00 EiB");
	cfg.unitmode = 1;
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1.00 MB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_Estimate), "1.00 GB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_Estimate), "1.00 TB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_Estimate), "1.00 PB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_Estimate), "1.00 EB");
	cfg.unitmode = 2;
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1000, 0, RT_Estimate), "1 kB");
	ck_assert_str_eq(getvalue(1000000, 0, RT_Estimate), "1.00 MB");
	ck_assert_str_eq(getvalue(1000000000, 0, RT_Estimate), "1.00 GB");
	ck_assert_str_eq(getvalue(1000000000000ULL, 0, RT_Estimate), "1.00 TB");
	ck_assert_str_eq(getvalue(1000000000000000ULL, 0, RT_Estimate), "1.00 PB");
	ck_assert_str_eq(getvalue(1000000000000000000ULL, 0, RT_Estimate), "1.00 EB");
}
END_TEST

START_TEST(getvalue_imagescale)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MiB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_ImageScale), "1 GiB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_ImageScale), "1 TiB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_ImageScale), "1 PiB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_ImageScale), "1 EiB");
	cfg.unitmode = 1;
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MB");
	ck_assert_str_eq(getvalue(1073741824, 0, RT_ImageScale), "1 GB");
	ck_assert_str_eq(getvalue(1099511627776ULL, 0, RT_ImageScale), "1 TB");
	ck_assert_str_eq(getvalue(1125899906842624ULL, 0, RT_ImageScale), "1 PB");
	ck_assert_str_eq(getvalue(1152921504606846976ULL, 0, RT_ImageScale), "1 EB");
	cfg.unitmode = 2;
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1000, 0, RT_ImageScale), "1 kB");
	ck_assert_str_eq(getvalue(1000000, 0, RT_ImageScale), "1 MB");
	ck_assert_str_eq(getvalue(1000000000, 0, RT_ImageScale), "1 GB");
	ck_assert_str_eq(getvalue(1000000000000ULL, 0, RT_ImageScale), "1 TB");
	ck_assert_str_eq(getvalue(1000000000000000ULL, 0, RT_ImageScale), "1 PB");
	ck_assert_str_eq(getvalue(1000000000000000000ULL, 0, RT_ImageScale), "1 EB");
}
END_TEST

START_TEST(getvalue_padding)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	ck_assert_str_eq(getvalue(1024, 10, RT_Normal), "  1.00 KiB");
	cfg.unitmode = 1;
	ck_assert_str_eq(getvalue(1024, 10, RT_Normal), "   1.00 KB");
	cfg.unitmode = 2;
	ck_assert_str_eq(getvalue(1000, 10, RT_Normal), "   1.00 kB");
}
END_TEST

START_TEST(getvalue_zero_values)
{
	cfg.unitmode = 0;
	ck_assert_str_eq(getvalue(0, 0, RT_Normal), "0 B");
	ck_assert_str_eq(getvalue(0, 10, RT_Estimate), "   --     ");
	ck_assert_int_eq((int)strlen(getvalue(0, 10, RT_Estimate)), 10);
	ck_assert_int_eq((int)strlen(getvalue(0, 20, RT_Estimate)), 20);
	ck_assert_str_eq(getvalue(0, 0, RT_ImageScale), "0 B");
	cfg.unitmode = 1;
	ck_assert_str_eq(getvalue(0, 0, RT_Normal), "0 B");
	ck_assert_str_eq(getvalue(0, 10, RT_Estimate), "    --    ");
	ck_assert_int_eq((int)strlen(getvalue(0, 10, RT_Estimate)), 10);
	ck_assert_int_eq((int)strlen(getvalue(0, 20, RT_Estimate)), 20);
	ck_assert_str_eq(getvalue(0, 0, RT_ImageScale), "0 B");
	cfg.unitmode = 2;
	ck_assert_str_eq(getvalue(0, 0, RT_Normal), "0 B");
	ck_assert_str_eq(getvalue(0, 10, RT_Estimate), "    --    ");
	ck_assert_int_eq((int)strlen(getvalue(0, 10, RT_Estimate)), 10);
	ck_assert_int_eq((int)strlen(getvalue(0, 20, RT_Estimate)), 20);
	ck_assert_str_eq(getvalue(0, 0, RT_ImageScale), "0 B");
}
END_TEST

START_TEST(gettrafficrate_zero_interval)
{
	int i, j;

	for (i = 0; i <= 1; i++) {
		cfg.rateunit = i;
		for (j = 0; j <= 2; j++) {
			cfg.unitmode = j;
			ck_assert_str_eq(gettrafficrate(1, 0, 0), "n/a");
		}
	}
}
END_TEST

START_TEST(gettrafficrate_bytes)
{
	cfg.defaultdecimals = 2;
	cfg.rateunit = 0;
	cfg.unitmode = 0;
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1.00 MiB/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "1.00 GiB/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "1.00 TiB/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "1.00 PiB/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "1.00 EiB/s");
	cfg.unitmode = 1;
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.00 KB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100.00 KB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1.00 MB/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "1.00 GB/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "1.00 TB/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "1.00 PB/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "1.00 EB/s");
	cfg.unitmode = 2;
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(100000, 1, 0), "100.00 kB/s");
	ck_assert_str_eq(gettrafficrate(1000000, 10, 0), "100.00 kB/s");
	ck_assert_str_eq(gettrafficrate(1000000, 1, 0), "1.00 MB/s");
	ck_assert_str_eq(gettrafficrate(1000000000, 1, 0), "1.00 GB/s");
	ck_assert_str_eq(gettrafficrate(1000000000000ULL, 1, 0), "1.00 TB/s");
	ck_assert_str_eq(gettrafficrate(1000000000000000ULL, 1, 0), "1.00 PB/s");
	ck_assert_str_eq(gettrafficrate(1000000000000000000ULL, 1, 0), "1.00 EB/s");
}
END_TEST

START_TEST(gettrafficrate_bits)
{
	cfg.defaultdecimals = 2;
	cfg.rateunit = 1;

	cfg.rateunitmode = 1;
	cfg.unitmode = 0;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.39 Mbit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.59 Gbit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.80 Tbit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "9.01 Pbit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "9.22 Ebit/s");
	cfg.unitmode = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.39 Mbit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.59 Gbit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.80 Tbit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "9.01 Pbit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "9.22 Ebit/s");
	cfg.unitmode = 2;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.39 Mbit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.59 Gbit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.80 Tbit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "9.01 Pbit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "9.22 Ebit/s");

	cfg.rateunitmode = 0;
	cfg.unitmode = 0;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.00 Mibit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.00 Gibit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.00 Tibit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "8.00 Pibit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "8.00 Eibit/s");
	cfg.unitmode = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.00 Mibit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.00 Gibit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.00 Tibit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "8.00 Pibit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "8.00 Eibit/s");
	cfg.unitmode = 2;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "800.00 Kibit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.00 Mibit/s");
	ck_assert_str_eq(gettrafficrate(1073741824, 1, 0), "8.00 Gibit/s");
	ck_assert_str_eq(gettrafficrate(1099511627776ULL, 1, 0), "8.00 Tibit/s");
	ck_assert_str_eq(gettrafficrate(1125899906842624ULL, 1, 0), "8.00 Pibit/s");
	ck_assert_str_eq(gettrafficrate(1152921504606846976ULL, 1, 0), "8.00 Eibit/s");
}
END_TEST

START_TEST(gettrafficrate_interval_divides)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	cfg.rateunitmode = 1;
	cfg.rateunit = 0;
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(102400, 2, 0), "50.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(102400, 10, 0), "10.00 KiB/s");
	cfg.rateunit = 1;
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(102400, 2, 0), "409.60 kbit/s");
	ck_assert_str_eq(gettrafficrate(102400, 10, 0), "81.92 kbit/s");
}
END_TEST

START_TEST(gettrafficrate_padding)
{
	cfg.defaultdecimals = 2;
	cfg.unitmode = 0;
	cfg.rateunit = 0;
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 12), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 14), "  100.00 KiB/s");

	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(900, 1, 7), "900 B/s");
	ck_assert_str_eq(gettrafficrate(900, 1, 12), "     900 B/s");
	ck_assert_str_eq(gettrafficrate(900, 1, 14), "       900 B/s");
}
END_TEST

START_TEST(sighandler_sets_signal)
{
	debug = 1;
	intsignal = 0;
	disable_logprints();
	ck_assert(signal(SIGINT, sighandler) != SIG_ERR);
	ck_assert(signal(SIGHUP, sighandler) != SIG_ERR);
	ck_assert(signal(SIGTERM, sighandler) != SIG_ERR);

	ck_assert_int_eq(kill(getpid(), SIGINT), 0);
	ck_assert_int_eq(intsignal, SIGINT);

	ck_assert_int_eq(kill(getpid(), SIGHUP), 0);
	ck_assert_int_eq(intsignal, SIGHUP);

	ck_assert_int_eq(kill(getpid(), SIGTERM), 0);
	ck_assert_int_eq(intsignal, SIGTERM);
}
END_TEST

START_TEST(validatedatetime_can_detect_valid_strings)
{
	ck_assert_int_eq(validatedatetime("2018-03-24 01:23"), 1);
	ck_assert_int_eq(validatedatetime("1998-01-15 23:16"), 1);
	ck_assert_int_eq(validatedatetime("2018-03-24"), 1);
	ck_assert_int_eq(validatedatetime("1998-01-15"), 1);
	ck_assert_int_eq(validatedatetime("today"), 1);
}
END_TEST

START_TEST(validatedatetime_can_detect_invalid_strings)
{
	ck_assert_int_eq(validatedatetime("2018-03-24 01:23:12"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 01"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 01:23am"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 1:23"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 01-23"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 "), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 01:23 "), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24 "), 0);
	ck_assert_int_eq(validatedatetime("2018-o3-24"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24T01:23"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-24_01:23"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-241"), 0);
	ck_assert_int_eq(validatedatetime("2018/03/24"), 0);
	ck_assert_int_eq(validatedatetime("2018-03"), 0);
	ck_assert_int_eq(validatedatetime("1998-01"), 0);
	ck_assert_int_eq(validatedatetime("2018-03-"), 0);
	ck_assert_int_eq(validatedatetime("2018_03"), 0);
	ck_assert_int_eq(validatedatetime("2018-3"), 0);
	ck_assert_int_eq(validatedatetime("2018_03"), 0);
	ck_assert_int_eq(validatedatetime("2018"), 0);
	ck_assert_int_eq(validatedatetime("1998"), 0);
	ck_assert_int_eq(validatedatetime("9999"), 0);
	ck_assert_int_eq(validatedatetime("18-03"), 0);
	ck_assert_int_eq(validatedatetime("18"), 0);
	ck_assert_int_eq(validatedatetime(" "), 0);
	ck_assert_int_eq(validatedatetime(""), 0);
	ck_assert_int_eq(validatedatetime("wtf?"), 0);
	ck_assert_int_eq(validatedatetime("yesterday"), 0);
	ck_assert_int_eq(validatedatetime("tomorrow"), 0);
}
END_TEST

START_TEST(validatedatetime_does_not_validate_numbers)
{
	ck_assert_int_eq(validatedatetime("9999-99-99 99:99"), 1);
	ck_assert_int_eq(validatedatetime("0000-00-00 00:00"), 1);
	ck_assert_int_eq(validatedatetime("2018-03-24 01:90"), 1);
}
END_TEST

START_TEST(defaultdecimals_controls_the_number_of_decimals)
{
	cfg.unitmode = 0;
	cfg.rateunitmode = 1;

	cfg.defaultdecimals = 0;
	cfg.rateunit = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MiB");
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100 KiB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100 KiB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1 MiB/s");
	cfg.rateunit = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8 Mbit/s");

	cfg.defaultdecimals = 1;
	cfg.rateunit = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1.0 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1.0 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1.0 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MiB");
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.0 KiB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100.0 KiB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1.0 MiB/s");
	cfg.rateunit = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.2 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.2 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.4 Mbit/s");

	cfg.defaultdecimals = 2;
	cfg.rateunit = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1.00 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1.00 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1.00 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MiB");
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100.00 KiB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1.00 MiB/s");
	cfg.rateunit = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.20 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.39 Mbit/s");

	cfg.defaultdecimals = 3;
	cfg.rateunit = 0;
	ck_assert_str_eq(getvalue(100, 0, RT_Normal), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Normal), "1.000 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Normal), "1.000 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_Estimate), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_Estimate), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_Estimate), "1.000 MiB");
	ck_assert_str_eq(getvalue(100, 0, RT_ImageScale), "100 B");
	ck_assert_str_eq(getvalue(1024, 0, RT_ImageScale), "1 KiB");
	ck_assert_str_eq(getvalue(1048576, 0, RT_ImageScale), "1 MiB");
	ck_assert_str_eq(gettrafficrate(900, 1, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(9000, 10, 0), "900 B/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "100.000 KiB/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "100.000 KiB/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "1.000 MiB/s");
	cfg.rateunit = 1;
	ck_assert_str_eq(gettrafficrate(100, 1, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(1000, 10, 0), "800 bit/s");
	ck_assert_str_eq(gettrafficrate(102400, 1, 0), "819.200 kbit/s");
	ck_assert_str_eq(gettrafficrate(1024000, 10, 0), "819.200 kbit/s");
	ck_assert_str_eq(gettrafficrate(1048576, 1, 0), "8.389 Mbit/s");
}
END_TEST

START_TEST(issametimeslot_knows_the_none_list)
{
	ck_assert_int_eq(issametimeslot(LT_None, 10, 11), 0);
}
END_TEST

START_TEST(issametimeslot_handles_updates_before_the_entry_time)
{
	time_t entry, updated;

	entry = (time_t)get_timestamp(2019, 4, 15, 12, 31);
	updated = (time_t)get_timestamp(2019, 4, 15, 12, 30);

	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 0);
}
END_TEST

START_TEST(issametimeslot_knows_simple_slots)
{
	time_t entry, updated;

	entry = (time_t)get_timestamp(2019, 4, 15, 12, 30);

	updated = entry;
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = entry + 1;
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	entry = (time_t)get_timestamp(2019, 4, 1, 12, 30);
	updated = (time_t)get_timestamp(2019, 4, 1, 12, 34);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 1, 12, 35);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);

	entry = (time_t)get_timestamp(2019, 4, 1, 12, 0);
	updated = (time_t)get_timestamp(2019, 4, 1, 12, 59);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 1, 13, 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);

	entry = (time_t)get_timestamp(2019, 4, 1, 0, 0);
	updated = (time_t)get_timestamp(2019, 4, 1, 23, 59);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 2, 0, 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);

	entry = (time_t)get_timestamp(2019, 4, 1, 0, 0);
	updated = (time_t)get_timestamp(2019, 4, 30, 23, 59);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 5, 1, 0, 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);

	entry = (time_t)get_timestamp(2019, 1, 1, 0, 0);
	updated = (time_t)get_timestamp(2019, 12, 31, 23, 59);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	entry = (time_t)get_timestamp(2019, 1, 1, 0, 0);
	updated = (time_t)get_timestamp(2020, 1, 1, 0, 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 0);
}
END_TEST

START_TEST(issametimeslot_knows_its_slots)
{
	time_t entry, updated;

	entry = (time_t)get_timestamp(2019, 4, 15, 12, 30);

	/* the database has the entry timestamp stored with the first possible */
	/* time of the specific range resulting in many of the following scenarios */
	/* never happening during normal usage */

	updated = (time_t)get_timestamp(2019, 4, 15, 12, 32);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 15, 12, 35);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 15, 13, 00);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 16, 13, 00);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 30, 00, 00);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 4, 30, 23, 59);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 1);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2019, 5, 16, 13, 00);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	updated = (time_t)get_timestamp(2020, 5, 16, 13, 00);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 0);

	entry = (time_t)get_timestamp(2019, 4, 1, 0, 0);
	updated = (time_t)get_timestamp(2019, 5, 1, 0, 0);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 1);

	entry = (time_t)get_timestamp(2019, 12, 31, 23, 59);
	updated = (time_t)get_timestamp(2020, 1, 1, 0, 0);
	ck_assert_int_eq(issametimeslot(LT_5min, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Hour, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Day, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Top, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Month, entry, updated), 0);
	ck_assert_int_eq(issametimeslot(LT_Year, entry, updated), 0);
}
END_TEST

START_TEST(getperiodseconds_knows_fixed_not_ongoing_periods)
{
	time_t entry, created, updated;

	created = (time_t)get_timestamp(2010, 1, 1, 3, 30);
	entry = (time_t)get_timestamp(2021, 1, 1, 18, 0);
	updated = entry;

	ck_assert_int_eq(getperiodseconds(LT_None, entry, updated, created, 0), 0);
	ck_assert_int_eq(getperiodseconds(LT_5min, entry, updated, created, 0), 300);
	ck_assert_int_eq(getperiodseconds(LT_Hour, entry, updated, created, 0), 3600);
	ck_assert_int_eq(getperiodseconds(LT_Day, entry, updated, created, 0), 86400);
	ck_assert_int_eq(getperiodseconds(LT_Top, entry, updated, created, 0), 86400);
}
END_TEST

START_TEST(getperiodseconds_knows_dynamic_not_ongoing_periods)
{
	time_t entry, created, updated;

	created = (time_t)get_timestamp(2010, 1, 1, 3, 30);
	entry = (time_t)get_timestamp(2021, 1, 1, 18, 0);
	updated = entry;

	ck_assert_int_eq(getperiodseconds(LT_Month, entry, updated, created, 0), 2678400);

	/* 2021 isn't a leap year */
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 0), 31536000);

	entry = (time_t)get_timestamp(2020, 1, 1, 18, 0);
	updated = entry;

	/* 2020 is a leap year */
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 0), 31622400);
}
END_TEST

START_TEST(getperiodseconds_returns_zero_when_there_is_no_time_spent)
{
	time_t entry, created, updated;

	cfg.monthrotate = 1;
	created = (time_t)get_timestamp(2010, 1, 1, 3, 30);
	entry = (time_t)get_timestamp(2021, 1, 1, 0, 0);
	updated = entry;

	ck_assert_int_eq(getperiodseconds(LT_None, entry, updated, created, 1), 0);
	ck_assert_int_eq(getperiodseconds(LT_5min, entry, updated, created, 1), 0);
	ck_assert_int_eq(getperiodseconds(LT_Hour, entry, updated, created, 1), 0);
	ck_assert_int_eq(getperiodseconds(LT_Day, entry, updated, created, 1), 0);
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 1), 0);

	/* months are special due to cfg.monthrotate */
	ck_assert_int_eq(getperiodseconds(LT_Month, entry, updated, created, 1), 1);

	/* LT_Top always returns the same value */
	ck_assert_int_eq(getperiodseconds(LT_Top, entry, updated, created, 1), 86400);
}
END_TEST

START_TEST(getperiodseconds_knows_spent_ongoing_time)
{
	time_t entry, created, updated;

	cfg.monthrotate = 1;
	created = (time_t)get_timestamp(2010, 1, 1, 3, 30);
	entry = (time_t)get_timestamp(2021, 1, 2, 3, 46);
	updated = entry;

	ck_assert_int_eq(getperiodseconds(LT_None, entry, updated, created, 1), 0);
	ck_assert_int_eq(getperiodseconds(LT_5min, entry, updated, created, 1), 60);
	ck_assert_int_eq(getperiodseconds(LT_Hour, entry, updated, created, 1), 2760);
	ck_assert_int_eq(getperiodseconds(LT_Day, entry, updated, created, 1), 13560);
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 1), 99960);

	/* months are special due to cfg.monthrotate */
	ck_assert_int_eq(getperiodseconds(LT_Month, entry, updated, created, 1), 1);

	/* LT_Top always returns the same value */
	ck_assert_int_eq(getperiodseconds(LT_Top, entry, updated, created, 1), 86400);
}
END_TEST

START_TEST(getperiodseconds_knows_spent_ongoing_time_when_created_mid_period)
{
	time_t entry, created, updated;

	cfg.monthrotate = 1;
	entry = (time_t)get_timestamp(2021, 1, 2, 18, 33);
	updated = entry;

	created = (time_t)get_timestamp(2021, 1, 2, 18, 33);
	ck_assert_int_eq(getperiodseconds(LT_5min, entry, updated, created, 1), 180);
	ck_assert_int_eq(getperiodseconds(LT_Hour, entry, updated, created, 1), 1980);
	ck_assert_int_eq(getperiodseconds(LT_Day, entry, updated, created, 1), 66780);
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 1), 153180);

	//printf("u: %" PRIu64 "\nc: %" PRIu64 "\n", (uint64_t)updated, (uint64_t)created);

	created = (time_t)get_timestamp(2021, 1, 2, 18, 35);
	ck_assert_int_eq(getperiodseconds(LT_5min, entry, updated, created, 1), 60);
	ck_assert_int_eq(getperiodseconds(LT_Hour, entry, updated, created, 1), 1860);
	ck_assert_int_eq(getperiodseconds(LT_Day, entry, updated, created, 1), 66660);
	ck_assert_int_eq(getperiodseconds(LT_Year, entry, updated, created, 1), 153060);

	/* months are special due to cfg.monthrotate */
	ck_assert_int_eq(getperiodseconds(LT_Month, entry, updated, created, 1), 1);

	/* LT_Top always returns the same value */
	ck_assert_int_eq(getperiodseconds(LT_Top, entry, updated, created, 1), 86400);
}
END_TEST

START_TEST(getestimates_has_error_handling)
{
	time_t created, updated;
	uint64_t rx = 1, tx = 2;
	dbdatalist *datalist = NULL;

	created = (time_t)get_timestamp(2010, 1, 2, 3, 46);
	updated = (time_t)get_timestamp(2021, 1, 2, 3, 46);

	getestimates(&rx, &tx, LT_None, updated, created, &datalist);
	ck_assert_int_eq(rx, 0);
	ck_assert_int_eq(tx, 0);

	rx = 1;
	tx = 2;
	ck_assert_int_eq(dbdatalistadd(&datalist, 0, 0, 0, 1), 1);
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_eq(rx, 0);
	ck_assert_int_eq(tx, 0);
	dbdatalistfree(&datalist);

	rx = 1;
	tx = 2;
	ck_assert_int_eq(dbdatalistadd(&datalist, 1000000, 0, 0, 1), 1);
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_eq(rx, 0);
	ck_assert_int_eq(tx, 0);
	dbdatalistfree(&datalist);

	rx = 1;
	tx = 2;
	ck_assert_int_eq(dbdatalistadd(&datalist, 0, 1000000, 0, 1), 1);
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_eq(rx, 0);
	ck_assert_int_eq(tx, 0);
	dbdatalistfree(&datalist);

	rx = 1;
	tx = 2;
	ck_assert_int_eq(dbdatalistadd(&datalist, 1000000, 1000000, 0, 1), 1);
	getestimates(&rx, &tx, LT_None, updated, created, &datalist);
	ck_assert_int_eq(rx, 0);
	ck_assert_int_eq(tx, 0);
	dbdatalistfree(&datalist);
}
END_TEST

START_TEST(getestimates_has_a_crystal_ball)
{
	time_t created, updated;
	uint64_t rx = 1, tx = 2;
	dbdatalist *datalist = NULL;

	created = (time_t)get_timestamp(2010, 1, 1, 3, 30);

	cfg.monthrotate = 1;
	updated = (time_t)get_timestamp(2021, 1, 1, 3, 45);
	ck_assert_int_eq(dbdatalistadd(&datalist, 100000, 200000, updated, 1), 1);

	rx = 1;
	tx = 2;
	/* on the 5 minute so there's no calculation done */
	getestimates(&rx, &tx, LT_5min, updated, created, &datalist);
	ck_assert_int_eq(rx, 100000);
	ck_assert_int_eq(tx, 200000);

	updated = (time_t)get_timestamp(2021, 1, 1, 3, 46);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_5min, updated, created, &datalist);
	ck_assert_int_eq(rx, 499800);
	ck_assert_int_eq(tx, 999900);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Hour, updated, created, &datalist);
	ck_assert_int_eq(rx, 129600);
	ck_assert_int_eq(tx, 259200);

	updated = (time_t)get_timestamp(2021, 1, 2, 3, 0);

	rx = 1;
	tx = 2;
	/* on the hour so there's no calculation done */
	getestimates(&rx, &tx, LT_Hour, updated, created, &datalist);
	ck_assert_int_eq(rx, 100000);
	ck_assert_int_eq(tx, 200000);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_eq(rx, 777600);
	ck_assert_int_eq(tx, 1555200);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Month, updated, created, &datalist);
	ck_assert_int_eq(rx, 2678400);
	ck_assert_int_eq(tx, 5356800);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Year, updated, created, &datalist);
	ck_assert_int_eq(rx, 31536000);
	ck_assert_int_eq(tx, 63072000);

	dbdatalistfree(&datalist);
}
END_TEST

START_TEST(getestimates_still_has_a_crystal_ball_when_created_mid_day_period)
{
	time_t created, updated;
	uint64_t rx = 1, tx = 2;
	dbdatalist *datalist = NULL;

	cfg.monthrotate = 1;
	updated = (time_t)get_timestamp(2021, 1, 2, 0, 0);
	ck_assert_int_eq(dbdatalistadd(&datalist, 100000, 200000, updated, 1), 1);

	updated = (time_t)get_timestamp(2021, 1, 2, 12, 0);

	created = (time_t)get_timestamp(2021, 1, 1, 0, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_eq(rx, 172800);
	ck_assert_int_eq(tx, 345600);

	created = (time_t)get_timestamp(2021, 1, 2, 6, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Day, updated, created, &datalist);
	ck_assert_int_gt(rx, 172800);
	ck_assert_int_gt(tx, 345600);

	dbdatalistfree(&datalist);
}
END_TEST

START_TEST(getestimates_still_has_a_crystal_ball_when_created_mid_month_period)
{
	time_t created, updated;
	uint64_t rx = 1, tx = 2;
	dbdatalist *datalist = NULL;

	cfg.monthrotate = 1;
	updated = (time_t)get_timestamp(2021, 1, 1, 0, 0);
	ck_assert_int_eq(dbdatalistadd(&datalist, 100000000, 200000000, updated, 1), 1);

	updated = (time_t)get_timestamp(2021, 1, 15, 12, 0);

	created = (time_t)get_timestamp(2021, 1, 1, 0, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Month, updated, created, &datalist);
	ck_assert_int_eq(rx, 211593600);
	ck_assert_int_eq(tx, 425865600);

	created = (time_t)get_timestamp(2021, 1, 10, 6, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Month, updated, created, &datalist);
	ck_assert_int_gt(rx, 211593600);
	ck_assert_int_gt(tx, 425865600);

	dbdatalistfree(&datalist);
}
END_TEST

START_TEST(getestimates_still_has_a_crystal_ball_when_created_mid_year_period)
{
	time_t created, updated;
	uint64_t rx = 1, tx = 2;
	dbdatalist *datalist = NULL;

	cfg.monthrotate = 1;
	updated = (time_t)get_timestamp(2021, 1, 1, 0, 0);
	ck_assert_int_eq(dbdatalistadd(&datalist, 100000000, 200000000, updated, 1), 1);

	updated = (time_t)get_timestamp(2021, 8, 15, 12, 0);

	created = (time_t)get_timestamp(2021, 1, 1, 0, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Year, updated, created, &datalist);
	ck_assert_int_eq(rx, 157680000);
	ck_assert_int_eq(tx, 315360000);

	created = (time_t)get_timestamp(2021, 4, 10, 6, 0);

	rx = 1;
	tx = 2;
	getestimates(&rx, &tx, LT_Year, updated, created, &datalist);
	ck_assert_int_gt(rx, 157680000);
	ck_assert_int_gt(tx, 315360000);

	dbdatalistfree(&datalist);
}
END_TEST

START_TEST(ishelprequest_knows_what_a_help_request_is)
{
	ck_assert_int_eq(ishelprequest("--help"), 1);
	ck_assert_int_eq(ishelprequest("-?"), 1);
	ck_assert_int_eq(ishelprequest("?"), 1);
	ck_assert_int_eq(ishelprequest("help"), 0);
	ck_assert_int_eq(ishelprequest("-h"), 0);
	ck_assert_int_eq(ishelprequest("--helpme"), 0);
	ck_assert_int_eq(ishelprequest(""), 0);
	ck_assert_int_eq(ishelprequest("1"), 0);
	ck_assert_int_eq(ishelprequest("a"), 0);
}
END_TEST

void add_misc_tests(Suite *s)
{
	TCase *tc_misc = tcase_create("Misc");
	tcase_add_checked_fixture(tc_misc, setup, teardown);
	tcase_add_unchecked_fixture(tc_misc, setup, teardown);
	tcase_add_test(tc_misc, getbtime_does_not_return_zero);
	tcase_add_loop_test(tc_misc, getunitprefix_returns_something_with_all_cfg_combinations, 0, 2);
	tcase_add_loop_test(tc_misc, getrateunitprefix_returns_something_with_all_cfg_combinations, 0, 3);
	tcase_add_loop_test(tc_misc, getunitdivisor_returns_something_with_all_cfg_combinations, 0, 3);
	tcase_add_test(tc_misc, spacecheck_does_not_check_when_not_configured);
	tcase_add_test(tc_misc, spacecheck_checks_space);
	tcase_add_test(tc_misc, spacecheck_fails_with_invalid_path);
	tcase_add_test(tc_misc, getvalue_normal);
	tcase_add_test(tc_misc, getvalue_estimate);
	tcase_add_test(tc_misc, getvalue_imagescale);
	tcase_add_test(tc_misc, getvalue_padding);
	tcase_add_test(tc_misc, getvalue_zero_values);
	tcase_add_test(tc_misc, gettrafficrate_zero_interval);
	tcase_add_test(tc_misc, gettrafficrate_bytes);
	tcase_add_test(tc_misc, gettrafficrate_bits);
	tcase_add_test(tc_misc, gettrafficrate_interval_divides);
	tcase_add_test(tc_misc, gettrafficrate_padding);
	tcase_add_test(tc_misc, sighandler_sets_signal);
	tcase_add_test(tc_misc, validatedatetime_can_detect_valid_strings);
	tcase_add_test(tc_misc, validatedatetime_can_detect_invalid_strings);
	tcase_add_test(tc_misc, validatedatetime_does_not_validate_numbers);
	tcase_add_test(tc_misc, defaultdecimals_controls_the_number_of_decimals);
	tcase_add_test(tc_misc, issametimeslot_knows_the_none_list);
	tcase_add_test(tc_misc, issametimeslot_handles_updates_before_the_entry_time);
	tcase_add_test(tc_misc, issametimeslot_knows_simple_slots);
	tcase_add_test(tc_misc, issametimeslot_knows_its_slots);
	tcase_add_test(tc_misc, getperiodseconds_knows_fixed_not_ongoing_periods);
	tcase_add_test(tc_misc, getperiodseconds_knows_dynamic_not_ongoing_periods);
	tcase_add_test(tc_misc, getperiodseconds_returns_zero_when_there_is_no_time_spent);
	tcase_add_test(tc_misc, getperiodseconds_knows_spent_ongoing_time);
	tcase_add_test(tc_misc, getperiodseconds_knows_spent_ongoing_time_when_created_mid_period);
	tcase_add_test(tc_misc, getestimates_has_error_handling);
	tcase_add_test(tc_misc, getestimates_has_a_crystal_ball);
	tcase_add_test(tc_misc, getestimates_still_has_a_crystal_ball_when_created_mid_day_period);
	tcase_add_test(tc_misc, getestimates_still_has_a_crystal_ball_when_created_mid_month_period);
	tcase_add_test(tc_misc, getestimates_still_has_a_crystal_ball_when_created_mid_year_period);
	tcase_add_test(tc_misc, ishelprequest_knows_what_a_help_request_is);
	suite_add_tcase(s, tc_misc);
}
