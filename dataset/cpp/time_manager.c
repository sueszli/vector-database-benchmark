/*
 * Copyright (C) 2020-2022 The opuntiaOS Project Authors.
 *  + Contributed by Nikita Melekhin <nimelehin@gmail.com>
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <drivers/driver_manager.h>
#include <libkern/log.h>
#include <time/time_manager.h>

// #define TIME_MANAGER_DEBUG

time_t ticks_since_boot = 0;
time_t ticks_since_second = 0;
static time_t time_since_boot = 0;
static time_t time_since_epoch = 0;
static uint32_t (*get_rtc)() = NULL;

static uint32_t pref_sum_of_days_in_mounts[] = {
    0,
    31,
    31 + 28, // the code that use that should add +1 autmoatically for a leap year
    31 + 28 + 31,
    31 + 28 + 31 + 30,
    31 + 28 + 31 + 30 + 31,
    31 + 28 + 31 + 30 + 31 + 30,
    31 + 28 + 31 + 30 + 31 + 30 + 31,
    31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
    31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
    31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,
    31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30,
    31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31,
};

bool timeman_is_leap_year(uint32_t year)
{
    return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
}

uint32_t timeman_days_in_years_since_epoch(uint32_t year)
{
    uint32_t days = 0;
    for (uint32_t y = 1970; y <= year; y++) {
        days += 365;
        if (timeman_is_leap_year(y)) {
            days++;
        }
    }
    return days;
}

/**
 * soy - start of the year
 */
uint32_t timeman_days_in_months_since_soy(uint8_t month, uint32_t year)
{
    uint32_t days = pref_sum_of_days_in_mounts[month];
    if (timeman_is_leap_year(year) && month >= 2) {
        days++;
    }
    return days;
}

time_t timeman_to_seconds_since_epoch(uint8_t secs, uint8_t mins, uint8_t hrs, uint8_t day, uint8_t month, uint32_t year)
{
    time_t res = timeman_days_in_years_since_epoch(year - 1) * 86400 + timeman_days_in_months_since_soy(month - 1, year) * 86400 + (day - 1) * 86400 + hrs * 3600 + mins * 60 + secs;
    return res;
}

int timeman_setup()
{
    uint8_t secs = 0, mins = 0, hrs = 0, day = 0, month = 0;
    uint32_t year = 1970;

    if (get_rtc) {
        time_since_epoch = get_rtc();
    } else {
        time_since_epoch = 0;
    }

#ifdef TIME_MANAGER_DEBUG
    log("Loaded date: %d", time_since_epoch);
#endif
    return 0;
}

void timeman_timer_tick()
{
    THIS_CPU->stat_ticks_since_boot++;
    if (system_cpu_id() != 0) {
        return;
    }

    atomic_add(&ticks_since_second, 1);

    if (ticks_since_second >= TIMER_TICKS_PER_SECOND) {
        atomic_add(&time_since_boot, 1);
        atomic_add(&time_since_epoch, 1);
        atomic_store(&ticks_since_second, 0);
    }
}

time_t timeman_seconds_since_epoch()
{
    return atomic_load(&time_since_epoch);
}

time_t timeman_seconds_since_boot()
{
    return atomic_load(&time_since_boot);
}

time_t timeman_get_ticks_from_last_second()
{
    return atomic_load(&ticks_since_second);
}

timespec_t timeman_timespec_since_epoch()
{
    timespec_t kts = { 0 };
    kts.tv_sec = timeman_seconds_since_epoch();
    kts.tv_nsec = timeman_get_ticks_from_last_second() * (1000000000 / timeman_ticks_per_second());
    return kts;
}

timespec_t timeman_timespec_since_boot()
{
    timespec_t kts = { 0 };
    kts.tv_sec = timeman_seconds_since_boot();
    kts.tv_nsec = timeman_get_ticks_from_last_second() * (1000000000 / timeman_ticks_per_second());
    return kts;
}

timeval_t timeman_timeval_since_epoch()
{
    timeval_t ktv = { 0 };
    ktv.tv_sec = timeman_seconds_since_epoch();
    ktv.tv_usec = timeman_get_ticks_from_last_second() * (1000000 / timeman_ticks_per_second());
    return ktv;
}

timeval_t timeman_timeval_since_boot()
{
    timeval_t ktv = { 0 };
    ktv.tv_sec = timeman_seconds_since_boot();
    ktv.tv_usec = timeman_get_ticks_from_last_second() * (1000000 / timeman_ticks_per_second());
    return ktv;
}

static void timeman_recieve_notification(uintptr_t msg, uintptr_t param)
{
    switch (msg) {
    case DEVMAN_NOTIFICATION_NEW_DEVICE:
        get_rtc = devman_function_handler((device_t*)param, DRIVER_RTC_GET_TIME);
        return;
    default:
        break;
    }
}

driver_desc_t _timeman_driver_info()
{
    driver_desc_t timeman_desc = { 0 };
    timeman_desc.type = DRIVER_TIME_MANAGER;
    timeman_desc.listened_device_mask = DEVICE_RTC;

    timeman_desc.system_funcs.recieve_notification = timeman_recieve_notification;
    return timeman_desc;
}

void timeman_install()
{
    devman_register_driver(_timeman_driver_info(), "timeman");
}
devman_register_driver_installation(timeman_install);
