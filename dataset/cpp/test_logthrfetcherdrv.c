/*
 * Copyright (c) 2018 Balabit
 * Copyright (c) 2018 László Várady <laszlo.varady@balabit.com>
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
#include "libtest/cr_template.h"

#include "logthrsource/logthrfetcherdrv.h"
#include "apphook.h"
#include "mainloop.h"
#include "mainloop-worker.h"
#include "cfg.h"
#include "stats/stats-counter.h"
#include "logsource.h"
#include "compat/time.h"

typedef struct _TestThreadedFetcherDriver
{
  LogThreadedFetcherDriver super;
  gint num_of_messages_to_generate;
  gint num_of_connection_failures_to_generate;
  gint connect_counter;
  gboolean try_again_first_time;
  gboolean no_data_first_time;

  GMutex lock;
  GCond cond;

} TestThreadedFetcherDriver;

MainLoopOptions main_loop_options = {0};
MainLoop *main_loop;

static const gchar *
_generate_persist_name(const LogPipe *s)
{
  return "test_threaded_fetcher_driver";
}

static void
_format_stats_key(LogThreadedSourceDriver *s, StatsClusterKeyBuilder *kb)
{
  stats_cluster_key_builder_add_legacy_label(kb, stats_cluster_label("driver", "test_threaded_fetcher_driver_stats"));
}

static void _source_queue_mock(LogPipe *s, LogMessage *msg, const LogPathOptions *path_options)
{
  LogSource *self = (LogSource *) s;

  stats_counter_inc(self->metrics.recvd_messages);
  log_pipe_forward_msg(s, msg, path_options);
}

static LogSource *
_get_source(TestThreadedFetcherDriver *self)
{
  return (LogSource *) self->super.super.worker;
}

static void
test_threaded_fetcher_free(LogPipe *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *) s;

  g_cond_clear(&self->cond);
  g_mutex_clear(&self->lock);

  log_threaded_fetcher_driver_free_method(s);
}

gboolean
test_threaded_fetcher_driver_init_method(LogPipe *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *)s;

  if (!log_threaded_fetcher_driver_init_method(s))
    return FALSE;

  /* mock out the hard-coded DNS lookup calls inside log_source_queue() */
  _get_source(self)->super.queue = _source_queue_mock;

  return TRUE;
}

static TestThreadedFetcherDriver *
test_threaded_fetcher_new(GlobalConfig *cfg)
{
  TestThreadedFetcherDriver *self = g_new0(TestThreadedFetcherDriver, 1);

  log_threaded_fetcher_driver_init_instance(&self->super, cfg);

  g_mutex_init(&self->lock);
  g_cond_init(&self->cond);

  self->super.super.super.super.super.init = test_threaded_fetcher_driver_init_method;

  self->super.super.format_stats_key = _format_stats_key;
  self->super.super.super.super.super.generate_persist_name = _generate_persist_name;
  self->super.super.super.super.super.free_fn = test_threaded_fetcher_free;

  return self;
}

static TestThreadedFetcherDriver *
create_threaded_fetcher(void)
{
  return test_threaded_fetcher_new(main_loop_get_current_config(main_loop));
}

static void
start_test_threaded_fetcher(TestThreadedFetcherDriver *s)
{
  cr_assert(log_pipe_init(&s->super.super.super.super.super));
  cr_assert(log_pipe_post_config_init(&s->super.super.super.super.super));
}

static void
wait_for_messages(TestThreadedFetcherDriver *s)
{
  g_mutex_lock(&s->lock);
  while (s->num_of_messages_to_generate > 0)
    g_cond_wait(&s->cond, &s->lock);
  g_mutex_unlock(&s->lock);
}

static void
stop_test_threaded_fetcher(TestThreadedFetcherDriver *s)
{
  main_loop_sync_worker_startup_and_teardown();
}

static void
destroy_test_threaded_fetcher(TestThreadedFetcherDriver *s)
{
  cr_assert(log_pipe_deinit(&s->super.super.super.super.super));
  log_pipe_unref(&s->super.super.super.super.super);
}

static void
setup(void)
{
  app_startup();
  main_loop = main_loop_get_instance();
  main_loop_init(main_loop, &main_loop_options);
}

static void
teardown(void)
{
  main_loop_deinit(main_loop);
  app_shutdown();
}

static LogThreadedFetchResult
_fetch(LogThreadedFetcherDriver *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *) s;

  if (self->num_of_connection_failures_to_generate
      && self->connect_counter <= self->num_of_connection_failures_to_generate)
    {
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_NOT_CONNECTED, NULL
      };
    }

  g_mutex_lock(&self->lock);
  if (self->num_of_messages_to_generate <= 0)
    {
      g_cond_signal(&self->cond);
      g_mutex_unlock(&self->lock);
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_ERROR, NULL
      };
    }

  LogMessage *msg = create_sample_message();

  self->num_of_messages_to_generate--;
  g_mutex_unlock(&self->lock);

  return (LogThreadedFetchResult)
  {
    .result = THREADED_FETCH_SUCCESS,
    .msg = msg
  };
}

static gboolean
_connect_fail_first_time(LogThreadedFetcherDriver *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *) s;

  self->connect_counter++;
  if (self->connect_counter == 1)
    return FALSE;

  return TRUE;
}

TestSuite(logthrfetcherdrv, .init = setup, .fini = teardown, .timeout = 10);

Test(logthrfetcherdrv, test_simple_fetch)
{
  TestThreadedFetcherDriver *s = create_threaded_fetcher();

  s->num_of_messages_to_generate = 10;
  s->super.fetch = _fetch;

  start_test_threaded_fetcher(s);
  wait_for_messages(s);
  stop_test_threaded_fetcher(s);

  StatsCounterItem *recvd_messages = _get_source(s)->metrics.recvd_messages;
  cr_assert(stats_counter_get(recvd_messages) == 10);

  destroy_test_threaded_fetcher(s);
}

Test(logthrfetcherdrv, test_reconnect)
{
  TestThreadedFetcherDriver *s = create_threaded_fetcher();

  s->num_of_messages_to_generate = 10;
  s->num_of_connection_failures_to_generate = 5;
  s->super.time_reopen = 0; /* immediate */
  s->super.connect = _connect_fail_first_time;
  s->super.fetch = _fetch;

  start_test_threaded_fetcher(s);
  wait_for_messages(s);
  stop_test_threaded_fetcher(s);

  StatsCounterItem *recvd_messages = _get_source(s)->metrics.recvd_messages;
  cr_assert(stats_counter_get(recvd_messages) == 10);
  cr_assert_geq(s->connect_counter, 6);

  destroy_test_threaded_fetcher(s);
}

static LogThreadedFetchResult
_fetch_for_try_again_test(LogThreadedFetcherDriver *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *) s;

  if (self->try_again_first_time)
    {
      self->try_again_first_time = FALSE;
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_TRY_AGAIN, NULL
      };
    }

  g_mutex_lock(&self->lock);
  if (self->num_of_messages_to_generate <= 0)
    {
      g_cond_signal(&self->cond);
      g_mutex_unlock(&self->lock);
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_ERROR, NULL
      };
    }

  LogMessage *msg = create_sample_message();

  self->num_of_messages_to_generate--;
  g_mutex_unlock(&self->lock);

  return (LogThreadedFetchResult)
  {
    .result = THREADED_FETCH_SUCCESS,
    .msg = msg
  };
}

Test(logthrfetcherdrv, test_try_again)
{
  TestThreadedFetcherDriver *s = create_threaded_fetcher();
  s->try_again_first_time = TRUE;

  s->num_of_messages_to_generate = 1;
  s->super.time_reopen = 10;
  s->super.fetch = _fetch_for_try_again_test;

  struct timespec start = {0};
  cr_assert(!clock_gettime(CLOCK_MONOTONIC, &start));

  start_test_threaded_fetcher(s);
  wait_for_messages(s);
  stop_test_threaded_fetcher(s);

  struct timespec stop = {0};
  cr_assert(!clock_gettime(CLOCK_MONOTONIC, &stop));

  // Should not pass time_reopen in case of try_again
  cr_assert(!stop.tv_sec - start.tv_sec < 2);

  destroy_test_threaded_fetcher(s);
}

static LogThreadedFetchResult
_fetch_for_no_data(LogThreadedFetcherDriver *s)
{
  TestThreadedFetcherDriver *self = (TestThreadedFetcherDriver *) s;

  if (self->no_data_first_time)
    {
      self->no_data_first_time = FALSE;
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_NO_DATA, NULL
      };
    }

  g_mutex_lock(&self->lock);
  if (self->num_of_messages_to_generate <= 0)
    {
      g_cond_signal(&self->cond);
      g_mutex_unlock(&self->lock);
      return (LogThreadedFetchResult)
      {
        THREADED_FETCH_ERROR, NULL
      };
    }

  LogMessage *msg = create_sample_message();

  self->num_of_messages_to_generate--;
  g_mutex_unlock(&self->lock);

  return (LogThreadedFetchResult)
  {
    .result = THREADED_FETCH_SUCCESS,
    .msg = msg
  };
}

Test(logthrfetcherdrv, test_no_data)
{
  TestThreadedFetcherDriver *s = create_threaded_fetcher();
  s->no_data_first_time = TRUE;

  s->num_of_messages_to_generate = 1;
  log_threaded_fetcher_driver_set_fetch_no_data_delay(&s->super.super.super.super, 1);
  s->super.fetch = _fetch_for_no_data;

  struct timespec start = {0};
  cr_assert(!clock_gettime(CLOCK_MONOTONIC, &start));

  start_test_threaded_fetcher(s);
  wait_for_messages(s);
  stop_test_threaded_fetcher(s);

  struct timespec stop = {0};
  cr_assert(!clock_gettime(CLOCK_MONOTONIC, &stop));

  cr_assert(stop.tv_sec - start.tv_sec >= 1);

  destroy_test_threaded_fetcher(s);
}
