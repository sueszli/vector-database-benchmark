/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck_gen_headers/config_kernel.h>

#include <tilck/common/basic_defs.h>
#include <tilck/common/printk.h>
#include <tilck/common/utils.h>

#include <tilck/kernel/hal.h>
#include <tilck/kernel/datetime.h>
#include <tilck/kernel/debug_utils.h>
#include <tilck/kernel/self_tests.h>
#include <tilck/kernel/sched.h>

extern u32 __tick_duration;
extern int __tick_adj_ticks_rem;
extern u32 clock_drift_adj_loop_delay;

void selftest_time(void)
{
   int drift;
   u32 orig_tick_duration = 0;
   u32 art_drift_p = 5;
   ulong var;

   if (!KRN_CLOCK_DRIFT_COMP) {
      printk("Skipping the test because KRN_CLOCK_DRIFT_COMP = 0.\n");
      goto out;
   }

   if (clock_drift_adj_loop_delay > 60 * TIMER_HZ) {

      printk("Test designed to run with clock_drift_adj_loop_delay <= 60s\n");
      printk("clock_drift_adj_loop_delay: %ds\n",
             clock_drift_adj_loop_delay / TIMER_HZ);

      printk("=> Skipping the artificial drift in the test\n");
      art_drift_p = 0;
   }

    /*
     * Increase tick's actual duration by 5% in order to produce quickly a
     * huge clock drift. Note: consider that __tick_duration is added to the
     * current time, TIMER_HZ times per second.
     *
     * For example, with TIMER_HZ=100:
     *
     *   td == 0.01 [ideal tick duration]
     *
     * Increasing `td` by 5%:
     *
     *   td == 0.0105
     *
     * Now after 1 second, we have an artificial drift of:
     *   0.0005 s * 100 = 0.05 s.
     *
     * After 20 seconds, we'll have a drift of 1 second.
     *
     * NOTE:
     *
     * A positive drift (calculated as: sys_ts - hw_ts) means that we're
     * going too fast and we have to add a _negative_ adjustment.
     *
     * A negative drift, means that we're lagging behind and we need to add a
     * _positive_ adjustment.
     */

   if (art_drift_p) {
      disable_interrupts(&var);
      {
         if (!__tick_adj_ticks_rem)
            orig_tick_duration = __tick_duration;
      }
      enable_interrupts(&var);

      if (!orig_tick_duration) {
         printk("Cannot start the test while there's a drift compensation.\n");
         return;
      }
   }

   printk("\n");
   printk("Clock drift correction self-test\n");
   printk("---------------------------------------------\n\n");

   for (int t = 0; !se_is_stop_requested(); t++) {

      drift = clock_get_second_drift();

      if (art_drift_p && t == 0) {

         /* Save the initial drift */
         printk("NOTE: Introduce artificial drift of %d%%\n", art_drift_p);

         disable_interrupts(&var);
         {
            __tick_duration = orig_tick_duration * (100+art_drift_p) / 100;
         }
         enable_interrupts(&var);

      } else if (art_drift_p && (t == 60 || t == 180)) {

         printk("NOTE: Remove any artificial drift\n");
         disable_interrupts(&var);
         {
            __tick_duration = orig_tick_duration;
         }
         enable_interrupts(&var);

      } else if (art_drift_p && t == 120) {

         printk("NOTE: Introduce artificial drift of -%d%%\n", art_drift_p);
         disable_interrupts(&var);
         {
            __tick_duration = orig_tick_duration * (100-art_drift_p) / 100;
         }
         enable_interrupts(&var);
      }

      printk(NO_PREFIX "[%06d seconds] Drift: %d\n", t, drift);
      kernel_sleep(TIMER_HZ);
   }

   if (art_drift_p) {
      disable_interrupts(&var);
      {
         __tick_duration = orig_tick_duration;
      }
      enable_interrupts(&var);
   }

out:
   if (se_is_stop_requested())
      se_interrupted_end();
   else
      se_regular_end();
}

REGISTER_SELF_TEST(time, se_manual, &selftest_time)

void selftest_delay(void)
{
   u64 before, after, elapsed;
   u32 us = 50000; /* 50 ms */

   disable_preemption();
   {
      before = get_ticks();
      delay_us(us);
      after = get_ticks();
   }
   enable_preemption();
   elapsed = after - before;

   printk("Expected in ticks: %u\n", us / (1000000 / TIMER_HZ));
   printk("Actual ticks:      %" PRIu64 "\n", elapsed);
}

REGISTER_SELF_TEST(delay, se_manual, &selftest_delay)

void selftest_clock_latency(void)
{
   struct datetime d;
   const int iters = 50000;
   u64 start, duration;

   printk("\n");
   printk("Clock read-latency self-test\n");
   printk("---------------------------------------------\n\n");

   start = RDTSC();
   disable_preemption();

   for (int i = 0; i < iters; i++) {
      hw_read_clock(&d);
   }

   enable_preemption();
   duration = (RDTSC() - start) / iters;
   printk("Latency: %" PRIu64 " RDTSC cycles\n", duration);
}

REGISTER_SELF_TEST(clock_latency, se_long, &selftest_clock_latency)
