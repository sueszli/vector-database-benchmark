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

#include "logthrsource/logthrsourcedrv.h"
#include "apphook.h"
#include "mainloop.h"
#include "mainloop-worker.h"
#include "cfg.h"
#include "stats/stats-counter.h"
#include "logsource.h"

typedef struct _TestThreadedSourceDriver
{
  LogThreadedSourceDriver super;
  gint num_of_messages_to_generate;
  gboolean suspended;
  gboolean exit_requested;
} TestThreadedSourceDriver;

MainLoopOptions main_loop_options = {0};
MainLoop *main_loop;

static void _request_exit(LogThreadedSourceDriver *s);
static void _run_simple(LogThreadedSourceDriver *s);
static void _run_using_blocking_posts(LogThreadedSourceDriver *s);

static const gchar *
_generate_persist_name(const LogPipe *s)
{
  return "test_threaded_source_driver";
}

static void
_format_stats_key(LogThreadedSourceDriver *s, StatsClusterKeyBuilder *kb)
{
  stats_cluster_key_builder_add_legacy_label(kb, stats_cluster_label("driver", "test_threaded_source_driver_stats"));
}

static void
_source_queue_mock(LogPipe *s, LogMessage *msg, const LogPathOptions *path_options)
{
  LogSource *self = (LogSource *) s;

  stats_counter_inc(self->metrics.recvd_messages);
  log_pipe_forward_msg(s, msg, path_options);
}

static LogSource *
_get_source(TestThreadedSourceDriver *self)
{
  return (LogSource *) self->super.worker;
}

gboolean
test_threaded_source_driver_init_method(LogPipe *s)
{

  TestThreadedSourceDriver *self = (TestThreadedSourceDriver *)s;

  if (!log_threaded_source_driver_init_method(s))
    return FALSE;

  /* mock out the hard-coded DNS lookup calls inside log_source_queue() */
  _get_source(self)->super.queue = _source_queue_mock;

  return TRUE;
}


static TestThreadedSourceDriver *
test_threaded_sd_new(GlobalConfig *cfg, gboolean blocking_post)
{
  TestThreadedSourceDriver *self = g_new0(TestThreadedSourceDriver, 1);

  log_threaded_source_driver_init_instance(&self->super, cfg);

  self->super.super.super.super.init = test_threaded_source_driver_init_method;
  self->super.format_stats_key = _format_stats_key;
  self->super.super.super.super.generate_persist_name = _generate_persist_name;

  self->super.request_exit = _request_exit;
  if (blocking_post)
    self->super.run = _run_using_blocking_posts;
  else
    self->super.run = _run_simple;

  return self;
}

static TestThreadedSourceDriver *
create_threaded_source(void)
{
  return test_threaded_sd_new(main_loop_get_current_config(main_loop), FALSE);
}

static TestThreadedSourceDriver *
create_threaded_source_blocking(void)
{
  return test_threaded_sd_new(main_loop_get_current_config(main_loop), TRUE);
}

static void
start_test_threaded_source(TestThreadedSourceDriver *s)
{
  cr_assert(log_pipe_init(&s->super.super.super.super));
  cr_assert(log_pipe_post_config_init(&s->super.super.super.super));
}

static void
request_exit_and_wait_for_stop(TestThreadedSourceDriver *s)
{
  main_loop_sync_worker_startup_and_teardown();
}

static void
destroy_test_threaded_source(TestThreadedSourceDriver *s)
{
  cr_assert(log_pipe_deinit(&s->super.super.super.super));
  log_pipe_unref(&s->super.super.super.super);
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

static void
_run_using_blocking_posts(LogThreadedSourceDriver *s)
{
  TestThreadedSourceDriver *self = (TestThreadedSourceDriver *) s;

  for (gint i = 0; i < self->num_of_messages_to_generate; ++i)
    {
      LogMessage *msg = create_sample_message();
      log_threaded_source_blocking_post(&self->super, msg);
    }
}

static void
_run_simple(LogThreadedSourceDriver *s)
{
  TestThreadedSourceDriver *self = (TestThreadedSourceDriver *) s;

  for (gint i = 0; i < self->num_of_messages_to_generate; ++i)
    {
      LogMessage *msg = create_sample_message();
      log_threaded_source_post(&self->super, msg);

      if (!log_threaded_source_free_to_send(&self->super))
        {
          self->suspended = TRUE;
          break;
        }
    }
}

static void
_request_exit(LogThreadedSourceDriver *s)
{
  TestThreadedSourceDriver *self = (TestThreadedSourceDriver *) s;
  self->exit_requested = TRUE;
}

static void
_do_not_ack_messages(LogPipe *s, LogMessage *msg, const LogPathOptions *path_options)
{
  log_msg_unref(msg);
}

TestSuite(logthrsourcedrv, .init = setup, .fini = teardown, .timeout = 10);

Test(logthrsourcedrv, test_threaded_source_blocking_post)
{
  TestThreadedSourceDriver *s = create_threaded_source_blocking();

  s->num_of_messages_to_generate = 10;

  start_test_threaded_source(s);
  request_exit_and_wait_for_stop(s);

  StatsCounterItem *recvd_messages = _get_source(s)->metrics.recvd_messages;
  cr_assert(stats_counter_get(recvd_messages) == 10);
  cr_assert(s->exit_requested);

  destroy_test_threaded_source(s);
}

Test(logthrsourcedrv, test_threaded_source_suspend)
{
  TestThreadedSourceDriver *s = create_threaded_source();

  s->num_of_messages_to_generate = 5;
  s->super.worker_options.super.init_window_size = 5;
  s->super.super.super.super.queue = _do_not_ack_messages;

  start_test_threaded_source(s);
  request_exit_and_wait_for_stop(s);

  StatsCounterItem *recvd_messages = _get_source(s)->metrics.recvd_messages;
  cr_assert(stats_counter_get(recvd_messages) == 5);
  cr_assert(s->suspended);
  cr_assert(s->exit_requested);

  destroy_test_threaded_source(s);
}
