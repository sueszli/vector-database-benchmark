
#include "cmos_clock.hpp"
#include "cmos.hpp"
#include <kernel/timers.hpp>
//#include <kernel/os.hpp>
#include <os.hpp>
#include <info>
#include "pit.hpp"
#include <rtc>
#include <util/units.hpp>

extern "C" uint16_t _cpu_sampling_freq_divider_;

namespace x86
{
  static uint64_t current_time;
  static uint64_t current_ticks;

  void CMOS_clock::init()
  {
    using namespace std::chrono;
    current_time  = CMOS::now().to_epoch();
    current_ticks = os::cycles_since_boot();

    INFO("CMOS", "Enabling regular clock sync for CMOS clock");
    // every minute recalibrate
    Timers::periodic(seconds(60), seconds(60),
      [] (Timers::id_t) {
        current_time  = CMOS::now().to_epoch();
        current_ticks = os::cycles_since_boot();
      });
  }

  uint64_t CMOS_clock::system_time()
  {
    auto ticks = os::cycles_since_boot() - current_ticks;
    auto diff  = (double) ticks / Hz(os::cpu_freq()).count();

    return (current_time + diff) * 1000000000ull;
  }
  timespec CMOS_clock::wall_clock()
  {
    using namespace util;
    auto ticks = os::cycles_since_boot() - current_ticks;
    auto diff  = (double) ticks / Hz(os::cpu_freq()).count();

    timespec tval;
    tval.tv_sec  = current_time + time_t(diff);
    tval.tv_nsec = diff * 1000000000ull;
    return tval;
  }

  KHz CMOS_clock::get_tsc_khz()
  {
    // Estimate CPU frequency
    INFO("CMOS", "Estimating CPU-frequency");
    INFO2("|");
    INFO2("+--(%d samples, %f sec. interval)", 18,
          (x86::PIT::FREQUENCY / _cpu_sampling_freq_divider_).count());
    INFO2("|");

    return KHz(MHz(PIT::get().estimate_CPU_frequency()));
  }
}
