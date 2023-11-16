/*
 * Copyright (c) 2013-2015 Balabit
 * Copyright (c) 2013 Juhász Viktor <jviktor@balabit.hu>
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

#include "messages.h"
#include "control/control.h"
#include "control/control-commands.h"
#include "control/control-connection.h"
#include "control-server-dummy.h"
#include "mainloop-control.h"
#include "stats/stats-control.h"
#include "control/control-server.h"
#include "stats/stats-cluster.h"
#include "stats/stats-registry.h"
#include "apphook.h"
#include "cfg-path.h"


ControlServer *control_server;
ControlConnection *control_connection;
MainLoop *main_loop;

void
setup(void)
{
  MainLoopOptions main_loop_options = {0};
  app_startup();

  main_loop = main_loop_get_instance();
  main_loop_init(main_loop, &main_loop_options);
  main_loop_register_control_commands(main_loop);
  stats_register_control_commands();
  control_server = control_server_dummy_new();
  control_connection = control_connection_dummy_new(control_server);
  control_connection_start_watches(control_connection);
}

void
teardown(void)
{
  control_server_connection_closed(control_server, control_connection);
  control_server_free(control_server);
  main_loop_deinit(main_loop);
  app_shutdown();
  reset_control_command_list();
}

TestSuite(control_cmds, .init = setup, .fini = teardown);

static void
_send_request(const gchar *request)
{
  control_connection_dummy_set_input(control_connection, request);
  control_connection_dummy_reset_output(control_connection);
}

static void
_fetch_response(const gchar **response)
{
  *response = control_connection_dummy_get_output(control_connection);
}

static void
_run_command(const gchar *request, const gchar **response)
{
  _send_request(request);
  control_connection->handle_input(control_connection);
  _fetch_response(response);
}

static gboolean
first_line_eq(const gchar *buf, const gchar *expected)
{
  const gchar *nl = strchr(buf, '\n');

  return strncmp(buf, expected, nl - buf) == 0;
}

Test(control_cmds, test_listfiles)
{
  const gchar *response;
  const gchar *db_file = "/opt/syslog-ng/var/db/patterndb.xml";
  GString *expected = g_string_new("");

  cfg_path_track_file(main_loop_get_current_config(main_loop), db_file, "path_check");

  _run_command("LISTFILES", &response);
  g_string_printf(expected, "path_check: %s", db_file);
  cr_assert(first_line_eq(response, expected->str), "Bad reply: [%s]", response);

  g_string_free(expected, TRUE);
}

Test(control_cmds, test_log)
{
  const gchar *response;

  _run_command("LOG", &response);
  cr_assert(first_line_eq(response, "FAIL Invalid arguments received"),
            "Bad reply: [%s]", response);

  _run_command("LOG fakelog", &response);
  cr_assert(first_line_eq(response, "FAIL Invalid arguments received"),
            "Bad reply: [%s]", response);

  msg_set_log_level(0);
  _run_command("LOG VERBOSE", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 0"),
            "Bad reply: [%s]", response);

  _run_command("LOG VERBOSE ON", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 1"),
            "Bad reply: [%s]", response);
  cr_assert_eq(verbose_flag, 1, "Flag isn't changed");

  _run_command("LOG VERBOSE OFF", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 0"),
            "Bad reply: [%s]", response);
  cr_assert_eq(verbose_flag, 0, "Flag isn't changed");


  msg_set_log_level(1);
  _run_command("LOG DEBUG", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 1"),
            "Bad reply: [%s]", response);

  _run_command("LOG DEBUG ON", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 2"),
            "Bad reply: [%s]", response);
  cr_assert_eq(debug_flag, 1, "Flag isn't changed");

  _run_command("LOG DEBUG OFF", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 1"),
            "Bad reply: [%s]", response);
  cr_assert_eq(debug_flag, 0, "Flag isn't changed");

  msg_set_log_level(2);
  _run_command("LOG TRACE", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 2"),
            "Bad reply: [%s]", response);

  _run_command("LOG TRACE ON", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 3"),
            "Bad reply: [%s]", response);
  cr_assert_eq(trace_flag, 1, "Flag isn't changed");

  _run_command("LOG TRACE OFF", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 2"),
            "Bad reply: [%s]", response);
  cr_assert_eq(trace_flag, 0, "Flag isn't changed");

}

Test(control_cmds, test_log_level)
{
  const gchar *response;

  msg_set_log_level(0);
  _run_command("LOG LEVEL foo", &response);
  cr_assert(first_line_eq(response, "FAIL Invalid arguments received"),
            "Bad reply: [%s]", response);

  msg_set_log_level(0);
  _run_command("LOG LEVEL debug", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 2"),
            "Bad reply: [%s]", response);

  msg_set_log_level(1);
  _run_command("LOG LEVEL", &response);
  cr_assert(first_line_eq(response, "OK syslog-ng log level set to 1"),
            "Bad reply: [%s]", response);

}

Test(control_cmds, test_stats)
{
  StatsCounterItem *counter = NULL;
  gchar **stats_result;
  const gchar *response;

  stats_lock();
  StatsClusterKey sc_key;
  stats_cluster_logpipe_key_legacy_set(&sc_key, SCS_CENTER, "id", "received" );
  stats_register_counter(0, &sc_key, SC_TYPE_PROCESSED, &counter);
  stats_unlock();

  _run_command("STATS", &response);

  stats_result = g_strsplit(response, "\n", 2);
  cr_assert_str_eq(stats_result[0], "SourceName;SourceId;SourceInstance;State;Type;Number",
                   "Bad reply");
  g_strfreev(stats_result);
}

Test(control_cmds, test_reset_stats)
{
  StatsCounterItem *counter = NULL;
  const gchar *response;

  stats_lock();
  StatsClusterKey sc_key;
  stats_cluster_logpipe_key_legacy_set(&sc_key, SCS_CENTER, "id", "received" );
  stats_register_counter(0, &sc_key, SC_TYPE_PROCESSED, &counter);
  stats_counter_set(counter, 666);
  stats_unlock();

  _run_command("RESET_STATS", &response);
  cr_assert(first_line_eq(response, "OK The statistics of syslog-ng have been reset to 0."), "Bad reply");

  _run_command("STATS", &response);
  cr_assert_str_eq(response,
                   "SourceName;SourceId;SourceInstance;State;Type;Number\ncenter;id;received;a;processed;0\n.\n",
                   "Bad reply");
}

static void
_original_replace(ControlConnection *cc, GString *result, gpointer user_data, gboolean *cancelled)
{
}

static void
_new_replace(ControlConnection *cc, GString *result, gpointer user_data, gboolean *cancelled)
{
}

static void
_assert_control_command_eq(ControlCommand *cmd, ControlCommand *cmd_other)
{
  cr_assert_eq(cmd->func, cmd_other->func);
  cr_assert_str_eq(cmd->command_name, cmd_other->command_name);
  cr_assert_eq(cmd->user_data, cmd_other->user_data);
}

Test(control_cmds, test_replace_existing_command)
{
  control_register_command("REPLACE", _original_replace, (gpointer)0xbaadf00d, FALSE);
  ControlCommand *cmd = control_find_command("REPLACE");
  ControlCommand expected_original =
  {
    .func = _original_replace,
    .command_name = "REPLACE",
    .user_data = (gpointer)0xbaadf00d
  };

  _assert_control_command_eq(cmd, &expected_original);

  control_replace_command("REPLACE", _new_replace, (gpointer)0xd006f00d, FALSE);
  ControlCommand *new_cmd = control_find_command("REPLACE");
  ControlCommand expected_new =
  {
    .func = _new_replace,
    .command_name = "REPLACE",
    .user_data = (gpointer) 0xd006f00d
  };
  _assert_control_command_eq(new_cmd, &expected_new);
}

Test(control_cmds, test_replace_non_existing_command)
{
  control_replace_command("REPLACE", _new_replace, (gpointer)0xd006f00d, FALSE);
  ControlCommand *new_cmd = control_find_command("REPLACE");
  ControlCommand expected_new =
  {
    .func = _new_replace,
    .command_name = "REPLACE",
    .user_data = (gpointer) 0xd006f00d
  };
  _assert_control_command_eq(new_cmd, &expected_new);
}
