/*
rvtimer.c - Timers, sleep functions
Copyright (C) 2021  LekKit <github.com/LekKit>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "rvtimer.h"
#include "compiler.h"
#include "utils.h"

#include <time.h>

#ifdef _WIN32
// Use QueryPerformanceCounter()
#include <windows.h>
#include "atomics.h"

static uint32_t qpc_crit = 0;
static uint64_t qpc_last = 0, qpc_freq = 0;

uint64_t rvtimer_clocksource(uint64_t freq)
{
    // Read the latest cached timer value from userspace
    uint64_t qpc_val = atomic_load_uint64_ex(&qpc_last, ATOMIC_ACQUIRE);
    if (!atomic_swap_uint32_ex(&qpc_crit, 1, ATOMIC_ACQUIRE)) {
        // Claimed the QPC lock, actually query the timer
        LARGE_INTEGER qpc = {0};
        if (qpc_freq == 0) {
            QueryPerformanceFrequency(&qpc);
            qpc_freq = qpc.QuadPart;
            if (qpc_freq == 0) rvvm_fatal("QueryPerformanceFrequency() failed!");
        }
        QueryPerformanceCounter(&qpc);
        if ((uint64_t)qpc.QuadPart < qpc_val) {
            DO_ONCE(rvvm_warn("Unstable clocksource (backward drift observed)"));
        } else {
            qpc_val = qpc.QuadPart;
            atomic_store_uint64_ex(&qpc_last, qpc_val, ATOMIC_RELEASE);
        }
        atomic_store_uint32_ex(&qpc_crit, 0, ATOMIC_RELEASE);
    }

    return rvtimer_convert_freq(qpc_val, qpc_freq, freq);
}

#elif defined(CLOCK_REALTIME) || defined(CLOCK_MONOTONIC)
// Use POSIX clock_gettime(), with a monotonic clock if possible
#include <unistd.h>
#if defined(CLOCK_MONOTONIC_RAW)
#define CHOSEN_POSIX_CLOCK CLOCK_MONOTONIC_RAW
#elif defined(CLOCK_MONOTONIC)
#define CHOSEN_POSIX_CLOCK CLOCK_MONOTONIC
#else
#define CHOSEN_POSIX_CLOCK CLOCK_REALTIME
#endif

uint64_t rvtimer_clocksource(uint64_t freq)
{
    struct timespec now = {0};
    clock_gettime(CHOSEN_POSIX_CLOCK, &now);
    return (now.tv_sec * freq) + (now.tv_nsec * freq / 1000000000ULL);
}

#elif defined(__APPLE__)
// Use mach_absolute_time() on older Mac OS
#include <unistd.h>
#include <mach/mach_time.h>

static mach_timebase_info_data_t mach_clk_freq = {0};

uint64_t rvtimer_clocksource(uint64_t freq)
{
    if (mach_clk_freq.denom == 0) {
        mach_timebase_info(&mach_clk_freq);
        if (mach_clk_freq.denom == 0) rvvm_fatal("mach_timebase_info() failed!");
    }
    return mach_absolute_time() * freq / mach_clk_freq.denom * mach_clk_freq.numer;
}

#else
// Use time() with no sub-second precision
#warning No OS support for precise clocksource!

uint64_t rvtimer_clocksource(uint64_t freq)
{
    return time(0) * freq;
}

#endif

#ifdef _POSIX_PRIORITY_SCHEDULING
#include <sched.h> // For sched_yield()
#endif

void rvtimer_init(rvtimer_t* timer, uint64_t freq)
{
    timer->freq = freq;
    // Some dumb rv32 OSes may ignore higher timecmp bits
    timer->timecmp = 0xFFFFFFFFU;
    rvtimer_rebase(timer, 0);
}

uint64_t rvtimer_get(rvtimer_t* timer)
{
    return rvtimer_clocksource(timer->freq) - timer->begin;
}

void rvtimer_rebase(rvtimer_t* timer, uint64_t time)
{
    timer->begin = rvtimer_clocksource(timer->freq) - time;
}

bool rvtimer_pending(rvtimer_t* timer)
{
    return rvtimer_get(timer) >= timer->timecmp;
}

void sleep_ms(uint32_t ms)
{
#ifdef _WIN32
#ifndef UNDER_CE
    static NTSTATUS (__stdcall *nt_setTR)(ULONG, BOOLEAN, PULONG) = NULL;
    DO_ONCE ({
        nt_setTR = (void*)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetTimerResolution");
    });
    if (nt_setTR) {
        ULONG cur;
        nt_setTR(5000, TRUE, &cur); // Set system clock resolution to 500us
        nt_setTR = NULL;
    }
#endif
    Sleep(ms);
#elif defined(CHOSEN_POSIX_CLOCK) || defined(__APPLE__)
    if (ms) {
        struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000, };
        while (nanosleep(&ts, &ts) < 0);
        return;
    }
#ifdef _POSIX_PRIORITY_SCHEDULING
    // Yield this thread time slice, as does Win32 Sleep(0)
    sched_yield();
#endif
#else
    UNUSED(ms);
#endif
}
