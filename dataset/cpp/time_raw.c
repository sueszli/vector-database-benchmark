#include <stdint.h>

// - nanoseconds since app initialization
// - microseconds since app initialization
// - milliseconds since app initialization
// - seconds since app initialization
// - minutes since app initialization
// - hours since app initialization

API uint64_t time_ns();
API uint64_t time_us();
API uint64_t time_ms();
API uint64_t time_ss();
API uint64_t time_hh();
API uint64_t time_mm();

API uint64_t time_raw(); // untested
API double   time_diff( uint64_t raw1, uint64_t raw2 ); // untested


// ----------------------------------------------------------------------------

#ifdef RAW_C
#pragma once
#include "../detect/detect.c" // platform

#define TIMER_E3 1000ULL
#define TIMER_E6 1000000ULL
#define TIMER_E9 1000000000ULL

#ifdef        CLOCK_MONOTONIC_RAW
#define TIME_MONOTONIC CLOCK_MONOTONIC_RAW
#elif defined CLOCK_MONOTONIC
#define TIME_MONOTONIC CLOCK_MONOTONIC
#else
// #define TIME_MONOTONIC CLOCK_REALTIME // untested
#endif

static uint64_t nanotimer() {
#  if WINDOWS || XBOX1
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (uint64_t)li.QuadPart;
#elif PS4
    return (uint64_t)sceKernelReadTsc();
#elif ANDROID
    return (uint64_t)clock();
#elif defined TIME_MONOTONIC
    struct timespec ts;
    clock_gettime(TIME_MONOTONIC, &ts);
    return (TIMER_E9 * (uint64_t)ts.tv_sec) + ts.tv_nsec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (TIMER_E6 * (uint64_t)tv.tv_sec) + tv.tv_usec;
#endif
}

static uint64_t nanofreq() {
#  if WINDOWS || XBOX1
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    return li.QuadPart;
#elif PS4
    return sceKernelGetTscFrequency();
#elif ANDROID
    return CLOCKS_PER_SEC;
#elif defined TIME_MONOTONIC
    return TIMER_E9;
#else
    return TIMER_E6;
#endif
}

// [ref] https://github.com/rust-lang/rust/blob/3809bbf47c8557bd149b3e52ceb47434ca8378d5/src/libstd/sys_common/mod.rs#L124
// Computes (a*b)/c without overflow, as long as both (a*b) and the overall result fit into 64-bits.
static
uint64_t time_muldiv64(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t q = a / c;
    uint64_t r = a % c;
    return q * b + r * b / c;
}

uint64_t time_ns() {
    static uint64_t epoch = 0;
    static uint64_t freq = 0;
    if( !epoch ) {
        epoch = nanotimer();
        freq = nanofreq();
    }
    return (uint64_t)time_muldiv64(nanotimer() - epoch, TIMER_E9, freq);
}
uint64_t time_us() {
    return time_ns() / TIMER_E3;
}
uint64_t time_ms() {
    return time_ns() / TIMER_E6;
}
uint64_t time_ss() {
    return time_ns() / TIMER_E9;
}
uint64_t time_mm() {
    return time_ss() / 60;
}
uint64_t time_hh() {
    return time_mm() / 60;
}

/* untested: */
uint64_t time_raw() {
    return nanotimer();
}
/* untested: */
double time_diff( uint64_t raw1, uint64_t raw2 ) {
    static double freq = 0; if(!freq) freq = 1.0 / nanofreq();
    return (raw1 < raw2 ? raw2 - raw1 : raw1 - raw2) * freq;

    //uint64_t ts = time_muldiv64(raw1 < raw2 ? raw2 - raw1 : raw1 - raw2, TIMER_E9, nanofreq());
    //return ts * 1.0e-9;
}


#endif


#ifdef RAW_DEMO
#include <stdio.h>
#include <stdlib.h>
int main() {
    uint64_t timer1 = time_ns();
    puts("hello world");
    timer1 = time_ns() - timer1;

    printf("print took %lluns\n", timer1);
    system("pause");
    printf("%02llu hours, %02llu minutes, %02llu seconds since app start\n", time_hh(), time_mm(), time_ss());


    uint64_t timer2a = time_raw();
    puts("hello world");
    uint64_t timer2b = time_raw();
    printf("%llu -> %llu = %fs", timer2a, timer2b, time_diff( timer2a, timer2b ) );
}
#endif
