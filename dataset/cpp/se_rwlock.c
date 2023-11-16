/* SPDX-License-Identifier: BSD-2-Clause */

#include <tilck/common/basic_defs.h>
#include <tilck/common/printk.h>

#include <tilck/kernel/rwlock.h>
#include <tilck/kernel/sched.h>
#include <tilck/kernel/sync.h>
#include <tilck/kernel/debug_utils.h>
#include <tilck/kernel/self_tests.h>
#include <tilck/kernel/timer.h>

#define RWLOCK_TH_ITERS    1000
#define RWLOCK_READERS       20
#define RWLOCK_WRITERS       20
#define RETRY_COUNT           4

static struct rwlock_rp test_rwlrp;
static struct rwlock_wp test_rwlwp;

static int se_rwlock_vars[3];
static const int se_rwlock_set_1[3] = {1, 2, 3};
static const int se_rwlock_set_2[3] = {10, 20, 30};
static ATOMIC(int) readers_running;
static ATOMIC(int) writers_running;

struct se_rwlock_ctx {

   void (*shlock)(void *);
   void (*shunlock)(void *);
   void (*exlock)(void *);
   void (*exunlock)(void *);
   void *arg;
};

static struct se_rwlock_ctx se_rp_ctx =
{
   .shlock = (void *) rwlock_rp_shlock,
   .shunlock = (void *) rwlock_rp_shunlock,
   .exlock = (void *) rwlock_rp_exlock,
   .exunlock = (void *) rwlock_rp_exunlock,
   .arg = (void *) &test_rwlrp,
};

static struct se_rwlock_ctx se_wp_ctx =
{
   .shlock = (void *) rwlock_wp_shlock,
   .shunlock = (void *) rwlock_wp_shunlock,
   .exlock = (void *) rwlock_wp_exlock,
   .exunlock = (void *) rwlock_wp_exunlock,
   .arg = (void *) &test_rwlwp,
};


static void se_rwlock_set_vars(const int *set)
{
   for (int i = 0; i < ARRAY_SIZE(se_rwlock_vars); i++) {
      se_rwlock_vars[i] = set[i];
      kernel_yield();
   }
}

static void se_rwlock_check_set_eq(const int *set)
{
   for (int i = 0; i < ARRAY_SIZE(se_rwlock_vars); i++) {
      VERIFY(se_rwlock_vars[i] == set[i]);
      kernel_yield();
   }
}

static void se_rwlock_read_thread(void *arg)
{
   struct se_rwlock_ctx *ctx = arg;
   readers_running++;

   for (int iter = 0; iter < RWLOCK_TH_ITERS; iter++) {

      if (se_is_stop_requested())
         break;

      ctx->shlock(ctx->arg);
      {
         if (se_rwlock_vars[0] == se_rwlock_set_1[0])
            se_rwlock_check_set_eq(se_rwlock_set_1);
         else
            se_rwlock_check_set_eq(se_rwlock_set_2);
      }
      ctx->shunlock(ctx->arg);
   }

   readers_running--;
}

static void se_rwlock_write_thread(void *arg)
{
   struct se_rwlock_ctx *ctx = arg;
   writers_running++;

   for (int iter = 0; iter < RWLOCK_TH_ITERS; iter++) {

      if (se_is_stop_requested())
         break;

      ctx->exlock(ctx->arg);
      {
         kernel_yield();

         if (se_rwlock_vars[0] == se_rwlock_set_1[0]) {

            se_rwlock_check_set_eq(se_rwlock_set_1);
            se_rwlock_set_vars(se_rwlock_set_2);

         } else {

            se_rwlock_check_set_eq(se_rwlock_set_2);
            se_rwlock_set_vars(se_rwlock_set_1);
         }

         kernel_yield();
      }
      ctx->exunlock(ctx->arg);
   }

   writers_running--;
}

static void se_rwlock_common(int *rt, int *wt, struct se_rwlock_ctx *ctx)
{
   se_rwlock_set_vars(se_rwlock_set_1);

   for (u32 i = 0; i < RWLOCK_READERS; i++) {
      rt[i] = kthread_create(se_rwlock_read_thread, 0, ctx);
      VERIFY(rt[i] > 0);
   }

   for (u32 i = 0; i < RWLOCK_WRITERS; i++) {
      wt[i] = kthread_create(se_rwlock_write_thread, 0, ctx);
      VERIFY(wt[i] > 0);
   }
}

void selftest_rwlock_rp()
{
   int rt[RWLOCK_READERS];
   int wt[RWLOCK_WRITERS];
   int retry;

   readers_running = writers_running = 0;
   rwlock_rp_init(&test_rwlrp);

   /*
    * Since we're testing a read-preferring rwlock, we except that, after all
    * readers have finished, there will still be writers running. At the same
    * way, we expect that, if we join first the writers, there won't be any
    * running readers.
    */

   printk("-------- sub-test: join readers and then writers -----------\n");
   for (retry = 0; retry < RETRY_COUNT; retry++) {

      se_rwlock_common(rt, wt, &se_rp_ctx);
      kthread_join_all(rt, ARRAY_SIZE(rt), true);
      printk("After readers, running writers: %d\n", writers_running);

      if (writers_running == 0) {

         kthread_join_all(wt, ARRAY_SIZE(wt), true);

         if (se_is_stop_requested())
            break;

         printk("running writers == 0, expected > 0. Re-try sub-test\n");
         continue;
      }

      /* writers_running > 0: ideal case */
      kthread_join_all(wt, ARRAY_SIZE(wt), true);
      break;
   }

   VERIFY(retry < RETRY_COUNT);

   if (se_is_stop_requested())
      goto end;

   printk("-------- sub-test: join writers and then readers -----------\n");
   for (retry = 0; retry < RETRY_COUNT; retry++) {

      se_rwlock_common(rt, wt, &se_rp_ctx);
      kthread_join_all(wt, ARRAY_SIZE(wt), true);

      if (se_is_stop_requested())
         goto end;

      printk("After writers, running readers: %d\n", readers_running);

      if (readers_running > 0) {

         kthread_join_all(rt, ARRAY_SIZE(rt), true);

         if (se_is_stop_requested())
            break;

         printk("running readers > 0, expected == 0. Re-try subtest.\n");
         continue;
      }

      /* readers_running == 0: ideal case */
      kthread_join_all(rt, ARRAY_SIZE(rt), true);
      break;
   }
   VERIFY(retry < RETRY_COUNT);

end:
   rwlock_rp_destroy(&test_rwlrp);

   if (se_is_stop_requested())
      se_interrupted_end();
   else
      se_regular_end();
}

REGISTER_SELF_TEST(rwlock_rp, se_med, &selftest_rwlock_rp)

void selftest_rwlock_wp()
{
   int rt[RWLOCK_READERS];
   int wt[RWLOCK_WRITERS];
   int retry;

   readers_running = writers_running = 0;
   rwlock_wp_init(&test_rwlwp, false);

   printk("-------- sub-test: join readers and then writers -----------\n");

   /*
    * Same as above, but in this case we're testing a write-preferring rwlock.
    * Therefore, after joining the readers, there should be 0 writers running;
    * after joining the writers, there should be some readers running.
    */

   for (retry = 0; retry < RETRY_COUNT; retry++) {

      se_rwlock_common(rt, wt, &se_wp_ctx);
      kthread_join_all(rt, ARRAY_SIZE(rt), true);
      printk("After readers, running writers: %d\n", writers_running);

      if (writers_running > 0) {

         kthread_join_all(wt, ARRAY_SIZE(wt), true);

         if (se_is_stop_requested())
            break;

         printk("running writers > 0, expected == 0. Re-try sub-test.\n");
         continue;
      }

      /* writers_running == 0: that's exactly we'd expect in the ideal case */
      kthread_join_all(wt, ARRAY_SIZE(wt), true);
      break;
   }
   VERIFY(retry < RETRY_COUNT);

   if (se_is_stop_requested())
      goto end;

   printk("-------- sub-test: join writers and then readers -----------\n");

   for (retry = 0; retry < RETRY_COUNT; retry++) {

      se_rwlock_common(rt, wt, &se_wp_ctx);
      kthread_join_all(wt, ARRAY_SIZE(wt), true);
      printk("After writers, running readers: %d\n", readers_running);

      if (readers_running == 0) {

         kthread_join_all(rt, ARRAY_SIZE(rt), true);

         if (se_is_stop_requested())
            break;

         printk("running readers == 0, expected > 0. Re-try sub-test\n");
         continue;
      }

      /* readers_running > 0: ideal case */
      kthread_join_all(rt, ARRAY_SIZE(rt), true);
      break;
   }
   VERIFY(retry < RETRY_COUNT);

end:
   rwlock_wp_destroy(&test_rwlwp);

   if (se_is_stop_requested())
      se_interrupted_end();
   else
      se_regular_end();
}

REGISTER_SELF_TEST(rwlock_wp, se_med, &selftest_rwlock_wp)
