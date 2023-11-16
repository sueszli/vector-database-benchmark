/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck_gen_headers/config_debug.h>

#include <tilck/common/basic_defs.h>
#include <tilck/common/printk.h>
#include <tilck/common/utils.h>
#include <tilck/common/string_util.h>

#include <tilck/kernel/process.h>
#include <tilck/kernel/sync.h>
#include <tilck/kernel/debug_utils.h>
#include <tilck/kernel/self_tests.h>
#include <tilck/kernel/timer.h>

#define KMUTEX_SEK_TH_ITERS 100000
#define KMUTEX_TH_COUNT        128

static struct kmutex test_mutex;
static int sek_vars[3];
static const int sek_set_1[3] = {1, 2, 3};
static const int sek_set_2[3] = {10, 20, 30};

static int tids[KMUTEX_TH_COUNT];
static int tid_by_idx1[KMUTEX_TH_COUNT];
static int tid_by_idx2[KMUTEX_TH_COUNT];
static volatile u8 ord_th_states[KMUTEX_TH_COUNT];
static volatile bool ord_test_done;
static int idx1, idx2;
static struct kmutex order_mutex;

static void sek_set_vars(const int *set)
{
   for (int i = 0; i < ARRAY_SIZE(sek_vars); i++) {
      sek_vars[i] = set[i];
      kernel_yield();
   }
}

static void sek_check_set_eq(const int *set)
{
   for (int i = 0; i < ARRAY_SIZE(sek_vars); i++) {
      VERIFY(sek_vars[i] == set[i]);
      kernel_yield();
   }
}

static void sek_thread(void *unused)
{
   for (int iter = 0; iter < KMUTEX_SEK_TH_ITERS; iter++) {

      if (UNLIKELY(se_is_stop_requested())) {
         printk("sek_thread: STOP at iter %d/%d\n", iter, KMUTEX_SEK_TH_ITERS);
         break;
      }

      kmutex_lock(&test_mutex);
      {
         kernel_yield();

         if (sek_vars[0] == sek_set_1[0]) {

            sek_check_set_eq(sek_set_1);
            sek_set_vars(sek_set_2);

         } else {

            sek_check_set_eq(sek_set_2);
            sek_set_vars(sek_set_1);
         }

         kernel_yield();
      }
      kmutex_unlock(&test_mutex);
      debug_no_deadlock_set_report_progress();

   } // for (int iter = 0; iter < KMUTEX_SEK_TH_ITERS; iter++)
}

void selftest_kmutex()
{
   int local_tids[3];

   kmutex_init(&test_mutex, 0);
   sek_set_vars(sek_set_1);

   debug_reset_no_deadlock_set();

   for (int i = 0; i < 3; i++) {
      local_tids[i] = kthread_create(sek_thread, 0, NULL);
      VERIFY(local_tids[i] > 0);
      debug_add_task_to_no_deadlock_set(local_tids[i]);
   }

   debug_add_task_to_no_deadlock_set(get_curr_tid());
   kthread_join_all(local_tids, ARRAY_SIZE(local_tids), true);
   debug_reset_no_deadlock_set();

   kmutex_destroy(&test_mutex);

   if (se_is_stop_requested())
      se_interrupted_end();
   else
      se_regular_end();
}

REGISTER_SELF_TEST(kmutex, se_med, &selftest_kmutex)

/* -------------------------------------------------- */
/*               Recursive mutex test                 */
/* -------------------------------------------------- */

static void test_kmutex_thread(void *arg)
{
   int tn = (int)(ulong)arg;

   printk("%i) before lock\n", tn);

   kmutex_lock(&test_mutex);

   printk("%i) under lock..\n", tn);

   // TODO: replace with delay_us()
   for (u32 i = 0; i < 128*MB; i++) {
      asmVolatile("nop");
   }

   kmutex_unlock(&test_mutex);

   printk("%i) after lock\n", tn);
}

static void test_kmutex_thread_trylock()
{
   printk("3) before trylock\n");

   bool locked = kmutex_trylock(&test_mutex);

   if (locked) {

      printk("3) trylock SUCCEEDED: under lock..\n");

      kmutex_unlock(&test_mutex);

      printk("3) after lock\n");

   } else {
      printk("3) trylock returned FALSE\n");
   }
}

void selftest_kmutex_rec()
{
   bool success;
   int local_tids[3];

   printk("kmutex recursive test\n");
   kmutex_init(&test_mutex, KMUTEX_FL_RECURSIVE);

   kmutex_lock(&test_mutex);
   printk("Locked once\n");

   kmutex_lock(&test_mutex);
   printk("Locked twice\n");

   success = kmutex_trylock(&test_mutex);

   if (!success) {
      panic("kmutex_trylock() failed on the same thread");
   }

   printk("Locked 3 times (last with trylock)\n");

   local_tids[0] = kthread_create(test_kmutex_thread_trylock, 0, NULL);
   VERIFY(local_tids[0] > 0);
   kthread_join(local_tids[0], true);

   kmutex_unlock(&test_mutex);
   printk("Unlocked once\n");

   kmutex_unlock(&test_mutex);
   printk("Unlocked twice\n");

   kmutex_unlock(&test_mutex);
   printk("Unlocked 3 times\n");

   local_tids[0] = kthread_create(&test_kmutex_thread, 0, (void*) 1);
   VERIFY(local_tids[0] > 0);

   local_tids[1] = kthread_create(&test_kmutex_thread, 0, (void*) 2);
   VERIFY(local_tids[1] > 0);

   local_tids[2] = kthread_create(&test_kmutex_thread_trylock, 0, NULL);
   VERIFY(local_tids[2] > 0);

   kthread_join_all(local_tids, ARRAY_SIZE(local_tids), true);
   kmutex_destroy(&test_mutex);
   se_regular_end();
}

REGISTER_SELF_TEST(kmutex_rec, se_med, &selftest_kmutex_rec)

/* -------------------------------------------------- */
/*               Strong order test                    */
/* -------------------------------------------------- */

/*
 * HOW IT WORKS
 * --------------
 *
 * The idea is to check that our kmutex implementation behaves like a strong
 * binary semaphore. In other words, if given task A tried to acquire the mutex
 * M before any given task B, on unlock() it MUST BE woken up and hold the mutex
 * BEFORE task B does.
 *
 * In order to do that, we create many threads and substantially make each one
 * of them to try to acquire the test_mutex. At the end, we would like to verify
 * that they acquired the mutex *in order*. But, what does *in order* mean?
 * How we do know which is the correct order? The creation of threads does NOT
 * have any order. For example: thread B, created AFTER thread A, may run before
 * it. Well, in order to do that, we use another mutex, called `order_mutex`.
 * Threads first get any order using `order_mutex` and then, in that order, they
 * try to acquire `test_mutex`. Of course, threads might be so fast that each
 * thread just acquires and releases both the mutexes without being preempted
 * and no thread really sleeps on kmutex_lock(). In order to prevent that, we
 * sleep while holding the `test_mutex`. For a better understanding, see the
 * comments below.
 */

static void kmutex_ord_th(void *arg)
{
   int tid = get_curr_tid();
   int local_id = (int)(long)arg;
   ord_th_states[local_id] = 1;

   /*
    * Since in practice, currently on Tilck, threads are executed pretty much
    * in the same order as they're created, we use the HACK below in order to
    * kind-of randomize the moment when they actually acquire the order_mutex,
    * simulating the general case where the `order_mutex` is strictly required.
    */
   kernel_sleep( ((u32)tid / sizeof(void *)) % 7 );
   ord_th_states[local_id] = 2;

   if (se_is_stop_requested())
      goto end;

   ord_th_states[local_id] = 3;
   kmutex_lock(&order_mutex);
   {
      tid_by_idx1[idx1++] = tid;

      /*
       * Note: disabling the preemption while holding the lock! This is *not*
       * a good practice and MUST BE avoided everywhere in real code except in
       * this test, where HACKS are needed in order to test the properties of
       * kmutex itself.
       */
      ord_th_states[local_id] = 4;
      disable_preemption();
   }
   kmutex_unlock(&order_mutex);
   ord_th_states[local_id] = 5;

   /*
    * Note: calling kmutex_lock() with preemption disabled! This is even worse
    * than calling kmutex_unlock() with preemption disabled. By definition,
    * it should *never* work because acquiring the mutex may require this thread
    * to go to sleep, if it has already an owner. But, for the purposes of this
    * test, we really need nobody to be able to preempt this thread in the
    * period of time between the acquisition of `order_mutex` and the attempt to
    * acquire `test_mutex` because we used `order_mutex` exactly in order to
    * make the attempts to acquire `test_mutex` happen all together. Ultimately,
    * we're testing that, if all threads try to lock `test_mutex` at the same
    * time, they're gonna to ultimately acquire the lock in the same order they
    * called kmutex_lock().
    */
   kmutex_lock(&test_mutex);
   {
      /*
       * Note: here, the preemption is enabled, even if we called kmutex_lock()
       * with preemption disabled. That's because of the "magic" kmutex flag
       * KMUTEX_FL_ALLOW_LOCK_WITH_PREEMPT_DISABLED designed specifically for
       * this self test. It allows the lock to be called while preemption is
       * disabled and it enables preemption forcibly, no matter what, before
       * going to sleep.
       */

      ord_th_states[local_id] = 6;
      ASSERT(is_preemption_enabled());
      tid_by_idx2[idx2++] = tid;

      /*
       * After registering this thread at position `idx2`, now sleep for 1 tick
       * WHILE holding the lock, in order to force all the other tasks to sleep
       * on kmutex_lock(), creating a queue. This is another trick necessary to
       * check that strong ordering actually occurs. Without it, threads might
       * be so fast that they just:
       *
       *    - acquire & release the order mutex without sleeping
       *    - acquire & release the test mutex without sleeping
       *
       * Therefore, the whole test will be pointless. Now instead, when
       * KMUTEX_STATS_ENABLED is 1, we can check that the order_mutex has
       * typically max_num_waiters = 0, while the test mutex has max_num_waiters
       * equals to almost its maxiumum (127). Typically, it's ~122.
       */

      if (!se_is_stop_requested()) {
         ord_th_states[local_id] = 7;
         kernel_sleep(1);
         ord_th_states[local_id] = 8;
      }
   }
   kmutex_unlock(&test_mutex);

end:
   ord_th_states[local_id] = 9;
}

static void
kmutex_ord_supervisor_thread()
{
   int time_ms = 0;

   while (!ord_test_done) {

      if (se_is_stop_requested())
         break;

      if (time_ms > 0 && (time_ms % 5000) == 0) {

         printk("[kmutex_ord_supervisor] %d sec elapsed\n", time_ms / 1000);

         int cnt[10] = {0};

         for (int i = 0; i < KMUTEX_TH_COUNT; i++)
            cnt[ord_th_states[i]]++;

         printk("[kmutex_ord_supervisor] Report per state:\n");

         for (int i = 0; i < 10; i++)
            printk("[kmutex_ord_supervisor] state %d: %d threads\n", i, cnt[i]);

         printk("\n\n");
      }

      kernel_sleep_ms(100);
      time_ms += 100;
   }
}

void selftest_kmutex_ord()
{
   u32 unlucky_threads = 0;
   int tid, supervisor_tid;

   idx1 = idx2 = 0;
   ord_test_done = false;
   bzero((void *)ord_th_states, sizeof(ord_th_states));
   kmutex_init(&test_mutex, KMUTEX_FL_ALLOW_LOCK_WITH_PREEMPT_DISABLED);
   kmutex_init(&order_mutex, 0);

   for (int i = 0; i < ARRAY_SIZE(tids); i++) {

      if ((tid = kthread_create(&kmutex_ord_th, 0, TO_PTR(i))) < 0)
         panic("[selftest] Unable to create kthread for kmutex_ord_th()");

      if (se_is_stop_requested())
         goto end;

      tids[i] = tid;
   }

   supervisor_tid = kthread_create(&kmutex_ord_supervisor_thread, 0, NULL);

   if (supervisor_tid < 0)
      panic("[selftest] Unable to create the supervisor kthread");

   kthread_join_all(tids, ARRAY_SIZE(tids), true);

   ord_test_done = true;
   kthread_join(supervisor_tid, true);

   if (se_is_stop_requested())
      goto end;

#if KMUTEX_STATS_ENABLED
   printk("order_mutex max waiters: %u\n", order_mutex.max_num_waiters);
   printk("test_mutex max waiters:  %u\n", test_mutex.max_num_waiters);
   VERIFY(test_mutex.max_num_waiters > 0);
#endif

   for (int i = 0; i < ARRAY_SIZE(tids); i++) {

      int t1, t2;

      t1 = tid_by_idx1[i];
      t2 = tid_by_idx2[i];

      if (t2 < 0) {
         unlucky_threads++;
         continue;
      }

      if (t2 != t1)
         panic("kmutex strong order test failed");
   }

   if (unlucky_threads > 0) {

      if (unlucky_threads > ARRAY_SIZE(tids) / 2)
         panic("Too many unlucky threads");

      printk("[selftests] Note: there were %u/%u unlucky threads",
             unlucky_threads, ARRAY_SIZE(tids));
   }

end:

   kmutex_destroy(&order_mutex);
   kmutex_destroy(&test_mutex);

   if (se_is_stop_requested())
      se_interrupted_end();
   else
      se_regular_end();
}

REGISTER_SELF_TEST(kmutex_ord, se_med, &selftest_kmutex_ord)
