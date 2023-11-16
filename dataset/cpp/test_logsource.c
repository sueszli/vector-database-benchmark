/*
 * Copyright (c) 2019 One Identity
 * Copyright (c) 2019 László Várady
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
#include <criterion/parameterized.h>

#include "logsource.h"
#include "logmsg/logmsg.h"
#include "logpipe.h"
#include "cfg.h"
#include "apphook.h"
#include "dynamic-window-pool.h"

#include <syslog.h>
#include <string.h>

#define TEST_SOURCE_GROUP "test_source_group"
#define TEST_STATS_ID "test_stats_id"

GlobalConfig *cfg;
LogSourceOptions source_options;

void
setup(void)
{
  cfg = cfg_new_snippet();
  log_source_options_defaults(&source_options);
  app_startup();
}

void
teardown(void)
{
  app_shutdown();
  log_source_options_destroy(&source_options);
  cfg_free(cfg);
}

typedef struct TestSource
{
  LogSource super;
  gsize wakeup_count;
} TestSource;

void
test_source_wakeup(LogSource *s)
{
  TestSource *source = (TestSource *) s;
  source->wakeup_count++;
}

LogSource *
test_source_init(LogSourceOptions *options)
{
  TestSource *source = g_new0(TestSource, 1);
  log_source_init_instance(&source->super, cfg);
  source->super.wakeup = test_source_wakeup;

  log_source_options_init(options, cfg, TEST_SOURCE_GROUP);
  log_source_set_options(&source->super, options, TEST_STATS_ID, NULL, TRUE, NULL);
  cr_assert(log_pipe_init(&source->super.super));
  return &source->super;
}

void
test_source_destroy(LogSource *source)
{
  log_pipe_deinit(&source->super);
  log_pipe_unref(&source->super);
}

typedef struct TestPipe
{
  LogPipe super;
  GQueue *messages;
  gsize messages_count;
} TestPipe;

static void
test_pipe_queue(LogPipe *s, LogMessage *msg, const LogPathOptions *path_options)
{
  TestPipe *pipe = (TestPipe *) s;

  g_queue_push_tail(pipe->messages, msg);
  pipe->messages_count++;
}

TestPipe *
test_pipe_init(void)
{
  TestPipe *pipe = g_new0(TestPipe, 1);
  log_pipe_init_instance(&pipe->super, cfg);
  pipe->super.queue = test_pipe_queue;

  pipe->messages = g_queue_new();

  cr_assert(log_pipe_init(&pipe->super));
  return pipe;
}

void
test_pipe_ack_messages(TestPipe *pipe, gsize ack_count)
{
  for (gsize i = 0; i < ack_count; ++i)
    {
      LogMessage *msg = g_queue_pop_head(pipe->messages);
      cr_assert(msg);
      pipe->messages_count--;

      LogPathOptions path_options = { .ack_needed = TRUE };
      log_msg_drop(msg, &path_options, AT_PROCESSED);
    }
}

void
test_pipe_destroy(TestPipe *pipe)
{
  log_pipe_deinit(&pipe->super);

  g_queue_free(pipe->messages);
  log_pipe_unref(&pipe->super);
}


const gchar *
resolve_sockaddr_to_hostname(gsize *result_len, GSockAddr *saddr, const HostResolveOptions *host_resolve_options)
{
  static const gchar *test_hostname = "resolved-test-host";
  *result_len = strlen(test_hostname);
  return test_hostname;
}

typedef struct _MangleHostnameParams
{
  gboolean keep_hostname;
  gboolean chain_hostnames;
  const gchar *input_host;
  const gchar *expected_hostname;

  guint32 msg_flags;
} MangleHostnameParams;

ParameterizedTestParameters(log_source, test_mangle_hostname)
{
  static MangleHostnameParams test_params[] =
  {
    { .keep_hostname = TRUE, .chain_hostnames = FALSE, "msg-test-host", "msg-test-host" },
    { .keep_hostname = TRUE, .chain_hostnames = FALSE, NULL, "resolved-test-host" },
    { .keep_hostname = TRUE, .chain_hostnames = FALSE, "", "resolved-test-host" },
    { .keep_hostname = TRUE, .chain_hostnames = TRUE, "msg-test-host", "msg-test-host" },
    { .keep_hostname = FALSE, .chain_hostnames = TRUE, NULL, "resolved-test-host/resolved-test-host" },
    { .keep_hostname = FALSE, .chain_hostnames = TRUE, "", "resolved-test-host/resolved-test-host" },
    { .keep_hostname = FALSE, .chain_hostnames = FALSE, "msg-test-host", "resolved-test-host" },
    { .keep_hostname = FALSE, .chain_hostnames = TRUE, "msg-test-host", "msg-test-host/resolved-test-host" },
    {
      .keep_hostname = FALSE, .chain_hostnames = TRUE, "msg-test-host", TEST_SOURCE_GROUP "@resolved-test-host",
      .msg_flags = LF_LOCAL
    },
  };

  return cr_make_param_array(MangleHostnameParams, test_params, G_N_ELEMENTS(test_params));
}

ParameterizedTest(MangleHostnameParams *test_params, log_source, test_mangle_hostname)
{
  source_options.keep_hostname = test_params->keep_hostname;
  source_options.chain_hostnames = test_params->chain_hostnames;
  LogSource *source = test_source_init(&source_options);

  LogMessage *msg = log_msg_new_empty();

  if (test_params->input_host)
    log_msg_set_value(msg, LM_V_HOST, test_params->input_host, -1);

  msg->flags |= test_params->msg_flags;

  log_source_mangle_hostname(source, msg);

  const gchar *actual_hostname = log_msg_get_value(msg, LM_V_HOST, NULL);
  cr_assert_str_eq(actual_hostname, test_params->expected_hostname);

  log_msg_unref(msg);
  test_source_destroy(source);
}

Test(log_source, test_chain_hostname_truncates_long_chained_hostnames)
{
  source_options.chain_hostnames = TRUE;
  LogSource *source = test_source_init(&source_options);
  LogMessage *msg = log_msg_new_empty();

  const gsize long_hostname_size = 512;
  gchar long_hostname[long_hostname_size];
  memset(long_hostname, 'Z', long_hostname_size);
  log_msg_set_value(msg, LM_V_HOST, long_hostname, long_hostname_size);

  log_source_mangle_hostname(source, msg);

  const gchar *actual_hostname = log_msg_get_value(msg, LM_V_HOST, NULL);
  gsize expected_hostname_len = 255;

  cr_assert_eq(strlen(actual_hostname), expected_hostname_len);
  cr_assert_arr_eq(actual_hostname, long_hostname, expected_hostname_len);

  log_msg_unref(msg);
  test_source_destroy(source);
}

Test(log_source, test_host_and_program_override)
{
  source_options.host_override = g_strdup("test-host-override");
  source_options.program_override = g_strdup("test-program-override");
  LogSource *source = test_source_init(&source_options);

  LogMessage *msg = log_msg_new_empty();
  log_msg_set_value(msg, LM_V_HOST, "hostname-to-override", -1);
  log_msg_set_value(msg, LM_V_PROGRAM, "program-to-override", -1);

  log_msg_ref(msg);
  log_source_post(source, msg);

  const gchar *actual_hostname = log_msg_get_value(msg, LM_V_HOST, NULL);
  cr_expect_str_eq(actual_hostname, source_options.host_override);
  const gchar *actual_program = log_msg_get_value(msg, LM_V_PROGRAM, NULL);
  cr_expect_str_eq(actual_program, source_options.program_override);

  log_msg_unref(msg);
  test_source_destroy(source);
}

Test(log_source, test_source_tags)
{
  GList *tags = NULL;
  tags = g_list_prepend(tags, g_strdup("tag1"));
  tags = g_list_prepend(tags, g_strdup("tag2"));
  log_source_options_set_tags(&source_options, tags);

  LogSource *source = test_source_init(&source_options);

  LogMessage *msg = log_msg_new_empty();
  log_msg_ref(msg);
  log_source_post(source, msg);

  cr_expect(log_msg_is_tag_by_name(msg, "tag1"));
  cr_expect(log_msg_is_tag_by_name(msg, "tag2"));
  cr_expect(log_msg_is_tag_by_name(msg, ".source." TEST_SOURCE_GROUP));

  log_msg_unref(msg);
  test_source_destroy(source);
}


static void
_post_messages(LogSource *source, gsize messages_to_send)
{
  for (gsize i = 0; i < messages_to_send; ++i)
    {
      LogMessage *msg = log_msg_new_empty();
      log_source_post(source, msg);
    }
}

Test(log_source, test_suspend)
{
  source_options.init_window_size = 3;

  LogSource *source = test_source_init(&source_options);
  TestPipe *next_pipe = test_pipe_init();
  log_pipe_append(&source->super, &next_pipe->super);

  cr_assert_eq(log_source_get_init_window_size(source), 3);
  cr_assert(log_source_free_to_send(source));

  _post_messages(source, 1);
  cr_assert(log_source_free_to_send(source));

  _post_messages(source, 2);
  cr_assert_not(log_source_free_to_send(source));

  test_pipe_ack_messages(next_pipe, 2);
  cr_assert(log_source_free_to_send(source));

  test_pipe_ack_messages(next_pipe, 1);
  cr_assert(log_source_free_to_send(source));

  test_pipe_destroy(next_pipe);
  test_source_destroy(source);
}

Test(log_source, test_wakeup)
{
  source_options.init_window_size = 3;

  LogSource *source = test_source_init(&source_options);
  TestPipe *next_pipe = test_pipe_init();
  log_pipe_append(&source->super, &next_pipe->super);

  _post_messages(source, 3);
  cr_expect_not(log_source_free_to_send(source));

  test_pipe_ack_messages(next_pipe, 1);
  cr_assert_eq(((TestSource *) source)->wakeup_count, 1);

  _post_messages(source, 1);
  cr_expect_not(log_source_free_to_send(source));

  test_pipe_ack_messages(next_pipe, 3);
  cr_assert_eq(((TestSource *) source)->wakeup_count, 2);

  test_pipe_destroy(next_pipe);
  test_source_destroy(source);
}

Test(log_source, test_forced_suspend_and_wakeup)
{
  LogSource *source = test_source_init(&source_options);

  cr_assert(log_source_free_to_send(source));

  log_source_flow_control_suspend(source);
  cr_assert_not(log_source_free_to_send(source));

  log_source_flow_control_adjust_when_suspended(source, 1);
  cr_assert_not(log_source_free_to_send(source));
  cr_assert_eq(((TestSource *) source)->wakeup_count, 0);

  log_source_flow_control_adjust(source, 1);
  cr_assert(log_source_free_to_send(source));
  cr_assert_eq(((TestSource *) source)->wakeup_count, 1);

  test_source_destroy(source);
}


static gboolean
test_mangle_callback_forbidden(GlobalConfig *config, LogMessage *msg, gpointer user_data)
{
  return strstr(log_msg_get_value(msg, LM_V_MESSAGE, NULL), "forbidden") == NULL;
}

static gboolean
test_mangle_callback_tag(GlobalConfig *config, LogMessage *msg, gpointer user_data)
{
  log_msg_set_tag_by_name(msg, "tagged");
  return TRUE;
}

static void
expect_forbidden_message_dropped(LogSource *source)
{
  LogMessage *msg = log_msg_new_internal(LOG_INFO | LOG_SYSLOG, "This is a forbidden message");
  log_msg_ref(msg);
  log_source_post(source, msg);

  cr_expect_not(log_msg_is_tag_by_name(msg, "tagged"),
                "Message should not be tagged, the message should have been dropped in test_mangle_callback_forbidden");
  log_msg_unref(msg);
}

static void
expect_regular_message_forwarded(LogSource *source)
{
  LogMessage *msg = log_msg_new_internal(LOG_INFO | LOG_SYSLOG, "Message");
  log_msg_ref(msg);
  log_source_post(source, msg);

  cr_expect(log_msg_is_tag_by_name(msg, "tagged"));
  log_msg_unref(msg);
}

Test(log_source, test_mangle_callback)
{
  register_source_mangle_callback(cfg, test_mangle_callback_forbidden);
  register_source_mangle_callback(cfg, test_mangle_callback_tag);

  LogSource *source = test_source_init(&source_options);
  TestPipe *next_pipe = test_pipe_init();
  log_pipe_append(&source->super, &next_pipe->super);

  expect_forbidden_message_dropped(source);
  cr_assert_eq(next_pipe->messages_count, 0);

  expect_regular_message_forwarded(source);
  cr_assert_eq(next_pipe->messages_count, 1);

  test_pipe_ack_messages(next_pipe, 1);

  test_pipe_destroy(next_pipe);
  test_source_destroy(source);
}


static DynamicWindowPool *
test_dynamic_window_pool_init(gsize pool_size)
{
  DynamicWindowPool *pool = dynamic_window_pool_new(pool_size);
  pool->balanced_window = pool_size;
  dynamic_window_pool_init(pool);

  return pool;
}

Test(log_source, test_dynamic_window_is_disabled_by_default)
{
  LogSource *source = test_source_init(&source_options);

  cr_assert_not(log_source_is_dynamic_window_enabled(source));

  test_source_destroy(source);
}

Test(log_source, test_dynamic_window)
{
  source_options.init_window_size = 0;

  LogSource *source = test_source_init(&source_options);
  log_source_set_name(source, "test-source-name");

  cr_assert_not(log_source_free_to_send(source));

  const gsize pool_size = 1000;
  DynamicWindowPool *pool = test_dynamic_window_pool_init(pool_size);
  log_source_enable_dynamic_window(source, pool);
  cr_assert(log_source_is_dynamic_window_enabled(source));

  /* currently unused */
  log_source_dynamic_window_update_statistics(source);

  pool->balanced_window = 200;
  log_source_dynamic_window_realloc(source);
  cr_assert(log_source_free_to_send(source),
            "Source should not be suspended as it should own free dynamic window slots");
  cr_assert_eq(pool->free_window, pool->pool_size - pool->balanced_window);

  dynamic_window_pool_unref(pool);
  test_source_destroy(source);
}

static void
_try_to_reclaim_all_dynamic_window_slots(LogSource *source, DynamicWindowPool *pool)
{
  pool->balanced_window = 0;
  log_source_dynamic_window_realloc(source);
}

Test(log_source, test_dynamic_window_reclaim)
{
  source_options.init_window_size = 1;
  LogSource *source = test_source_init(&source_options);
  TestPipe *next_pipe = test_pipe_init();
  log_pipe_append(&source->super, &next_pipe->super);

  const gsize pool_size = 10;
  DynamicWindowPool *pool = test_dynamic_window_pool_init(pool_size);
  log_source_enable_dynamic_window(source, pool);
  log_source_dynamic_window_realloc(source);

  const gsize num_of_pending_messages = pool->pool_size + source_options.init_window_size;
  _post_messages(source, num_of_pending_messages);
  cr_assert_not(log_source_free_to_send(source),
                "Source should be suspended, its window is filled with pending messages");

  _try_to_reclaim_all_dynamic_window_slots(source, pool);
  cr_assert_not(log_source_free_to_send(source));
  cr_assert_eq(pool->free_window, 0,
               "Incorrect free pool size; window should not be released, the source has pending messages");

  test_pipe_ack_messages(next_pipe, num_of_pending_messages);

  _try_to_reclaim_all_dynamic_window_slots(source, pool);
  cr_assert(log_source_free_to_send(source), "The initial static window should be available");
  cr_assert_eq(pool->free_window, pool->pool_size,
               "Incorrect free pool size; window should be reclaimed");

  dynamic_window_pool_unref(pool);
  test_pipe_destroy(next_pipe);
  test_source_destroy(source);
}

TestSuite(log_source, .init = setup, .fini = teardown);
