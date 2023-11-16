/*
 * Copyright (c) 2002-2019 Balabit
 * Copyright (c) 1998-2019 Balázs Scheidler
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */
#include <criterion/criterion.h>
#include "libtest/fake-time.h"

#include "timeutils/unixtime.h"
#include "timeutils/wallclocktime.h"
#include "timeutils/conv.h"
#include "timeutils/cache.h"
#include "timeutils/format.h"


static void
_wct_initialize(WallClockTime *wct, const gchar *timestamp)
{
  gchar *end = wall_clock_time_strptime(wct, "%b %d %Y %H:%M:%S", timestamp);

  cr_assert(*end == 0, "error parsing WallClockTime initialization timestamp: %s, end: %s", timestamp, end);
}

Test(unixtime, unix_time_initialization)
{
  UnixTime ut = UNIX_TIME_INIT;

  cr_assert(!unix_time_is_set(&ut));

  /* Thu Dec 19 22:25:44 CET 2019 */
  fake_time(1576790744);
  unix_time_set_now(&ut);

  cr_assert(unix_time_is_set(&ut));
  cr_expect(ut.ut_sec == 1576790744);
  cr_expect(ut.ut_usec == 123000);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_unset(&ut);
  cr_assert(!unix_time_is_set(&ut));
}

Test(unixtime, unix_time_fix_timezone_adjusts_timestamp_as_if_was_parsed_assuming_the_incorrect_timezone)
{
  UnixTime ut = UNIX_TIME_INIT;
  WallClockTime wct = WALL_CLOCK_TIME_INIT;

  _wct_initialize(&wct, "Jan 19 2019 18:58:48");
  wct.wct_gmtoff = 3600;
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_fix_timezone(&ut, -5*3600);
  cr_expect(ut.ut_gmtoff == -5*3600);
  convert_unix_time_to_wall_clock_time(&ut, &wct);
  cr_expect(wct.wct_hour == 18);
  cr_expect(wct.wct_min == 58);
  cr_expect(wct.wct_sec == 48);
}

static void
_fix_timezone_with_tzinfo(UnixTime *base_ut, UnixTime *ut, gint base_ofs, const gchar *zone_name)
{
  *ut = *base_ut;
  ut->ut_sec += base_ofs;
  unix_time_fix_timezone_with_tzinfo(ut, cached_get_time_zone_info(zone_name));
}

Test(unixtime, unix_time_fix_timezone_with_tzinfo_to_a_zone_backwards_during_sprint_daylight_saving_hour)
{
  UnixTime ut = UNIX_TIME_INIT;
  UnixTime base_ut;

  /* We are assuming a timestamp in CET and fixing that to be US Eastern
   * time EST5EDT, e.g.  the timezone offset is being decreased from +3600
   * to -18000/-14400 (depending on DST).  This testcase contains a series
   * of tests as we go through the daylight saving start hour in the spring.
   * */

  /* Base timestamp: Mar 10 2019 02:00:00 CET,
   *
   * e.g.  if interpreted in EST5DST this is the start of the DST transition
   * hour in 2019.  We simulate that we assumed it was CET and then we "fix"
   * the timezone to EST5EDT */

  base_ut.ut_sec = 1552179600;
  base_ut.ut_usec = 0;
  base_ut.ut_gmtoff = 3600;

  /* TESTCASE: 1 second earlier than the DST transition hour */

  /* this "fix" will cause ut->ut_sec to be changed, it will go forward, but
   * still not be reaching the daylight saving start, short of a second */

  _fix_timezone_with_tzinfo(&base_ut, &ut, -1, "EST5EDT");


  /* still has not reached the DST start time */
  cr_assert(ut.ut_sec == 1552201200 - 1);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == -5*3600);

  /* TESTCASE: 1 second later, e.g.  Mar 10 2019 02:00:00 CET, which is
   * exactly the daylight saving start second.  It should be converted to
   * Mar 10 2019 03:00:00 EDT */

  _fix_timezone_with_tzinfo(&base_ut, &ut, 0, "EST5EDT");

  /* we are at exactly the DST start time */
  cr_assert(ut.ut_sec == 1552201200);
  /* thus the resulting timezone is EDT and not EST */
  cr_assert(ut.ut_gmtoff == -4*3600);

  /* TESTCASE: 30 minutes later, e.g. 02:30:00, that is converted to 03:30:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 1800, "EST5EDT");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1552201200 + 1800);
  /* thus the resulting timezone is EDT */
  cr_assert(ut.ut_gmtoff == -4*3600);

  /* TESTCASE: 1 hour second later, e.g. 03:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 3600, "EST5EDT");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1552201200);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == -4*3600);

  /* TESTCASE: 2 hours second later, e.g. 04:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 7200, "EST5EDT");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1552201200 + 3600);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == -4*3600);
}

Test(unixtime, unix_time_fix_timezone_with_tzinfo_to_a_zone_forwards_during_sprint_daylight_saving_hour)
{
  UnixTime ut = UNIX_TIME_INIT;
  UnixTime base_ut;

  /* Base timestamp: Mar 31 2019 02:00:00 EST5EDT,
   *
   * e.g.  if interpreted in CET this is the start of the DST transition
   * hour in 2019.  We simulate that we assumed it was EST5EDT and then we "fix"
   * the timezone to CET */

  base_ut.ut_sec = 1554012000;
  base_ut.ut_usec = 0;
  base_ut.ut_gmtoff = -4*3600;

  /* TESTCASE: 1 second earlier than the DST transition hour */
  /* this "fix" will cause ut->ut_sec to be changed, it will go forward, but
   * still not be reaching the daylight saving start, short of a second */
  _fix_timezone_with_tzinfo(&base_ut, &ut, -1, "CET");

  /* still has not reached the DST start time */
  cr_assert(ut.ut_sec == 1553994000 - 1);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == 3600);

  /* TESTCASE: 1 second later, e.g.  Mar 31 2019 02:00:00 CET, which is
   * exactly the daylight saving start second.  It should be converted to
   * Mar 31 2019 03:00:00 CEST */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 0, "CET");

  /* we are at exactly the DST start time */
  cr_assert(ut.ut_sec == 1553994000);
  /* thus the resulting timezone is EDT and not EST */
  cr_assert(ut.ut_gmtoff == 2*3600);

  /* TESTCASE: 30 minutes later, e.g. 02:30:00, that is converted to 03:30:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 1800, "CET");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1553994000 + 1800);
  /* thus the resulting timezone is EDT */
  cr_assert(ut.ut_gmtoff == 2*3600);

  /* TESTCASE: 1 hour second later, e.g. 03:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 3600, "CET");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1553994000);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == 2*3600);

  /* TESTCASE: 2 hours second later, e.g. 04:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 7200, "CET");

  /* we lose the hour, so the DST start time */
  cr_assert(ut.ut_sec == 1553994000 + 3600);
  /* thus the resulting timezone is still EST and not the daylight saving variant */
  cr_assert(ut.ut_gmtoff == 2*3600);
}

Test(unixtime, unix_time_fix_timezone_with_tzinfo_to_a_zone_backwards_during_autumn_daylight_saving_hour)
{
  UnixTime ut = UNIX_TIME_INIT;
  UnixTime base_ut;

  /* Base timestamp: Nov 3 2019 02:00:00 CET,
   *
   * e.g.  if interpreted in EST5DST this is the end of the DST transition
   * hour in 2019.  We simulate that we assumed it was CET and then we "fix"
   * the timezone to EST5EDT */

  base_ut.ut_sec = 1572742800;
  base_ut.ut_usec = 0;
  base_ut.ut_gmtoff = 3600;

  /* TESTCASE: 1 second earlier than the DST transition hour */

  /* this "fix" will cause ut->ut_sec to be changed, it will go forward, but
   * still not be reaching the daylight saving start, short of a second */
  _fix_timezone_with_tzinfo(&base_ut, &ut, -1, "EST5EDT");


  /* still has not reached the DST start time */
  cr_assert(ut.ut_sec == 1572760800 - 1);
  /* thus the resulting timezone is still EDT */
  cr_assert(ut.ut_gmtoff == -4*3600);

  /* TESTCASE: 1 second later, e.g.  Mar 10 2019 02:00:00 CET, which is
   * exactly the daylight saving start second.  It should be converted to
   * Mar 10 2019 03:00:00 EDT */

  _fix_timezone_with_tzinfo(&base_ut, &ut, 0, "EST5EDT");

  /* once passed the DST end threshold, we assume the 2nd 02:00:00AM, e.g.
   * the one with -05:00 offset.  This means that an hour is skipped in this
   * case in the seconds since 1970.01.01, while the offset decreased with
   * the same.  This means that it is not possible to represent a time
   * between 02:00:00 to 02:59:59 in the daylight saving period, unless the
   * timezone is explicitly available in the timestamp.  */

  cr_assert(ut.ut_sec == 1572760800 + 3600);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == -5*3600);

  /* TESTCASE: 30 minutes later, e.g. 02:30:00, that is converted to 03:30:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 1800, "EST5EDT");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572760800 + 3600 + 1800);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == -5*3600);

  /* TESTCASE: 1 hour second later, e.g. 03:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 3600, "EST5EDT");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572760800 + 3600 + 3600);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == -5*3600);

  /* TESTCASE: 2 hours second later, e.g. 04:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 7200, "EST5EDT");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572760800 + 3600 + 7200);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == -5*3600);
}

Test(unixtime, unix_time_fix_timezone_with_tzinfo_to_a_zone_forwards_during_autumn_daylight_saving_hour)
{
  UnixTime ut = UNIX_TIME_INIT;
  UnixTime base_ut;

  /* Base timestamp: Oct 27 2019 02:00:00 EST5EDT,
   *
   * e.g.  if interpreted in CET this is the end of the DST transition
   * hour in 2019.  We simulate that we assumed it was EST5EDT and then we "fix"
   * the timezone to CET */

  base_ut.ut_sec = 1572156000;
  base_ut.ut_usec = 0;
  base_ut.ut_gmtoff = -4*3600;

  /* testcase, 1 second earlier than the DST transition hour */

  /* this "fix" will cause ut->ut_sec to be changed, it will go forward, but
   * still not be reaching the daylight saving start, short of a second */

  _fix_timezone_with_tzinfo(&base_ut, &ut, -1, "CET");

  /* still has not reached the DST start time */
  cr_assert(ut.ut_sec == 1572134400 - 1);
  /* thus the resulting timezone is still CEST */
  cr_assert(ut.ut_gmtoff == 2*3600);

  /* TESTCASE: 1 second later, e.g.  Oct 27 2019 02:00:00 EDT, which is
   * exactly the daylight saving start second if interpreted in CET.  It
   * should be converted to Oct 27 2019 03:00:00 CET */

  _fix_timezone_with_tzinfo(&base_ut, &ut, 0, "CET");

  /* once passed the DST end threshold, we assume the 2nd 02:00:00AM, e.g.
   * the one with -05:00 offset.  This means that an hour is skipped in this
   * case in the seconds since 1970.01.01, while the offset decreased with
   * the same.  This means that it is not possible to represent a time
   * between 02:00:00 to 02:59:59 in the daylight saving period, unless the
   * timezone is explicitly available in the timestamp.  */

  cr_assert(ut.ut_sec == 1572134400 + 3600);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == 3600);


  /* TESTCASE: 30 minutes later, e.g. 02:30:00, that is converted to 03:30:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 1800, "CET");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572134400 + 3600 + 1800);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == 3600);

  /* TESTCASE: 1 hour second later, e.g. 03:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 3600, "CET");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572134400 + 3600 + 3600);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == 3600);

  /* TESTCASE: 2 hours second later, e.g. 04:00:00 */
  _fix_timezone_with_tzinfo(&base_ut, &ut, 7200, "CET");

  /* we lose the hour in ut_sec */
  cr_assert(ut.ut_sec == 1572134400 + 3600 + 7200);
  /* thus the resulting timezone is EST and not EDT */
  cr_assert(ut.ut_gmtoff == 3600);
}

Test(unixtime, unix_time_set_timezone_converts_the_timestamp_to_a_target_timezone_assuming_the_source_was_correct)
{
  UnixTime ut = UNIX_TIME_INIT;
  WallClockTime wct = WALL_CLOCK_TIME_INIT;

  _wct_initialize(&wct, "Jan 19 2019 18:58:48");
  wct.wct_gmtoff = 3600;
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_set_timezone(&ut, -5*3600);
  cr_expect(ut.ut_gmtoff == -5*3600);
  convert_unix_time_to_wall_clock_time(&ut, &wct);
  cr_expect(wct.wct_hour == 12);
  cr_expect(wct.wct_min == 58);
  cr_expect(wct.wct_sec == 48);
}

Test(unixtime, unix_time_set_timezone_with_tzinfo_calculates_dst_automatically)
{
  UnixTime ut = UNIX_TIME_INIT;
  WallClockTime wct = WALL_CLOCK_TIME_INIT;

  _wct_initialize(&wct, "Mar 10 2019 01:59:59");
  wct.wct_gmtoff = -5*3600;
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_gmtoff == -5*3600);

  unix_time_set_timezone_with_tzinfo(&ut, cached_get_time_zone_info("EST5EDT"));
  cr_expect(ut.ut_gmtoff == -5*3600);
  ut.ut_sec += 1;
  unix_time_set_timezone_with_tzinfo(&ut, cached_get_time_zone_info("EST5EDT"));
  cr_expect(ut.ut_gmtoff == -4*3600);

  _wct_initialize(&wct, "Nov 3 2019 01:59:59");
  wct.wct_gmtoff = -4*3600;
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_gmtoff == -4*3600);

  unix_time_set_timezone_with_tzinfo(&ut, cached_get_time_zone_info("EST5EDT"));
  cr_expect(ut.ut_gmtoff == -4*3600);

  ut.ut_sec += 1;
  unix_time_set_timezone_with_tzinfo(&ut, cached_get_time_zone_info("EST5EDT"));
  cr_expect(ut.ut_gmtoff == -5*3600);
}

Test(unixtime, unix_time_guess_timezone_for_even_hour_differences)
{
  UnixTime ut = UNIX_TIME_INIT;
  WallClockTime wct = WALL_CLOCK_TIME_INIT;


  /* Thu Dec 19 22:25:44 CET 2019 */
  fake_time(1576790744);

  /* this is one hour earlier than current time */
  _wct_initialize(&wct, "Dec 19 2019 21:25:44");
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_sec == 1576790744 - 3600);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_fix_timezone_assuming_the_time_matches_real_time(&ut);
  cr_expect(ut.ut_sec == 1576790744);
  cr_expect(ut.ut_gmtoff == 0);

  /* 13 hours earlier to test one extreme of the timezones, that is -12:00 */
  _wct_initialize(&wct, "Dec 19 2019 09:25:44");
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_sec == 1576790744 - 13*3600);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_fix_timezone_assuming_the_time_matches_real_time(&ut);
  cr_expect(ut.ut_sec == 1576790744);
  cr_expect(ut.ut_gmtoff == -12*3600);


  /* 13 hours later to test the other extreme, that is +14:00 */
  _wct_initialize(&wct, "Dec 20 2019 11:25:44");
  convert_wall_clock_time_to_unix_time(&wct, &ut);
  cr_expect(ut.ut_sec == 1576790744 + 13*3600);
  cr_expect(ut.ut_gmtoff == 3600);

  unix_time_fix_timezone_assuming_the_time_matches_real_time(&ut);
  cr_expect(ut.ut_sec == 1576790744);
  cr_expect(ut.ut_gmtoff == +14*3600, "%d", ut.ut_gmtoff);
}


Test(unixtime, unix_time_guess_timezone_for_quarter_hour_differences)
{
  UnixTime ut = UNIX_TIME_INIT;
  glong t, diff;
  gint number_of_noneven_timezones = 0;
  gint number_of_even_timezones = 0;

  /* Thu Dec 19 22:25:44 CET 2019 */
  t = 1576790744;
  fake_time(t);

  for (diff = - 13 * 3600; diff <= 14*3600; diff += 900)
    {
      ut.ut_sec = t + diff;
      ut.ut_gmtoff = 3600;
      if (unix_time_fix_timezone_assuming_the_time_matches_real_time(&ut))
        {
          cr_expect(ut.ut_sec == 1576790744);
          cr_expect(ut.ut_gmtoff == diff + 3600);

          if ((diff % 3600) != 0)
            number_of_noneven_timezones++;
          else
            number_of_even_timezones++;
        }
    }
  cr_assert(number_of_noneven_timezones == 17,
            "The expected number of timezones that are not at an even hour boundary does not match expectations: %d",
            number_of_noneven_timezones);

  /* -12:00 .. 00:00 .. +14:00 */
  cr_assert(number_of_even_timezones == 12 + 1 + 14,
            "The expected number of timezones that are at an even hour boundary does not match expectations: %d",
            number_of_even_timezones);
}

Test(unixtime, test_unix_time_diff_in_seconds)
{
  UnixTime ut1 = { 1, 123000 };
  UnixTime ut2 = { 2, 123000 };

  cr_assert(unix_time_diff_in_seconds(&ut1, &ut2) == -1);
  cr_assert(unix_time_diff_in_seconds(&ut2, &ut1) == 1);

  ut2.ut_sec = 1;
  ut2.ut_usec = 623000;

  cr_assert(unix_time_diff_in_seconds(&ut1, &ut2) == -1);
  cr_assert(unix_time_diff_in_seconds(&ut2, &ut1) == 1);

  ut2.ut_sec = 1;
  ut2.ut_usec = 622000;

  cr_assert(unix_time_diff_in_seconds(&ut1, &ut2) == 0);
  cr_assert(unix_time_diff_in_seconds(&ut2, &ut1) == 0);

  ut2.ut_sec = 1;
  ut2.ut_usec = 624000;

  cr_assert(unix_time_diff_in_seconds(&ut1, &ut2) == -1);
  cr_assert(unix_time_diff_in_seconds(&ut2, &ut1) == 1);



  /* >0.5 seconds in fractions rounded up */
  ut1.ut_sec = 0;
  ut1.ut_usec = 0;
  ut2.ut_sec = 1;
  ut2.ut_usec = 500001;
  cr_assert_eq(unix_time_diff_in_seconds(&ut1, &ut2), -2);
  cr_assert_eq(unix_time_diff_in_seconds(&ut2, &ut1), 2);

  /* < 0.5 seconds rounded down */
  ut1.ut_sec = 0;
  ut1.ut_usec = 980000;
  ut2.ut_sec = 1;
  ut2.ut_usec =  20000;
  cr_assert_eq(unix_time_diff_in_seconds(&ut2, &ut1), 0);
  cr_assert_eq(unix_time_diff_in_seconds(&ut1, &ut2), 0);

}

Test(unixtime, test_unix_time_diff_in_msec)
{
  UnixTime ut1 = { 1, 123000 };
  UnixTime ut2 = { 2, 123000 };

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -1000);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 1000);

  ut2.ut_sec = 1;
  ut2.ut_usec = 623000;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -500);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 500);

  ut2.ut_sec = 1;
  ut2.ut_usec = 622000;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -499);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 499);

  ut2.ut_sec = 1;
  ut2.ut_usec = 622499;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -499);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 499);

  ut2.ut_sec = 1;
  ut2.ut_usec = 622500;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -500);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 500);

  ut2.ut_sec = 1;
  ut2.ut_usec = 623499;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -500);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 500);

  ut2.ut_sec = 1;
  ut2.ut_usec = 623501;

  cr_assert(unix_time_diff_in_msec(&ut1, &ut2) == -501);
  cr_assert(unix_time_diff_in_msec(&ut2, &ut1) == 501);
}

static void
setup(void)
{
  setenv("TZ", "CET", TRUE);
  tzset();
}

static void
teardown(void)
{
}

TestSuite(unixtime, .init = setup, .fini = teardown);
