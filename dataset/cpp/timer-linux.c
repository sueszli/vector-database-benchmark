/*
 * precise timer routines for Linux/UNIX
 * copyright (C) LGB & A'rpi/ASTRAL
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>

#include "common/common.h"
#include "timer.h"

static clockid_t clk_id;

void mp_sleep_ns(int64_t ns)
{
    if (ns < 0)
        return;
    struct timespec ts;
    ts.tv_sec  = ns / MP_TIME_S_TO_NS(1);
    ts.tv_nsec = ns % MP_TIME_S_TO_NS(1);
    nanosleep(&ts, NULL);
}

uint64_t mp_raw_time_ns(void)
{
    struct timespec tp = {0};
    clock_gettime(clk_id, &tp);
    return MP_TIME_S_TO_NS(tp.tv_sec) + tp.tv_nsec;
}

void mp_raw_time_init(void)
{
    static const clockid_t clock_ids[] = {
#ifdef CLOCK_MONOTONIC_RAW
        CLOCK_MONOTONIC_RAW,
#endif
        CLOCK_MONOTONIC,
    };

    struct timespec tp;
    for (int i = 0; i < MP_ARRAY_SIZE(clock_ids); ++i) {
        clk_id = clock_ids[i];
        if (!clock_gettime(clk_id, &tp))
            return;
    }
    fputs("No clock source available!\n", stderr);
    abort();
}
