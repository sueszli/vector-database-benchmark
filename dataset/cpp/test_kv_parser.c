/*
 * Copyright (c) 2015-2019 Balabit
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 */

#include <criterion/criterion.h>
#include "libtest/msg_parse_lib.h"

#include "kv-parser.h"
#include "apphook.h"
#include "scratch-buffers.h"


GlobalConfig *cfg;
LogParser *kv_parser;

static LogMessage *
parse_kv_into_log_message_no_check(const gchar *kv)
{
  LogMessage *msg;
  LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;
  LogParser *cloned_parser;

  cloned_parser = (LogParser *) log_pipe_clone(&kv_parser->super);
  msg = log_msg_new_empty();
  log_msg_set_value(msg, LM_V_MESSAGE, kv, -1);
  if (!log_parser_process_message(cloned_parser, &msg, &path_options))
    {
      log_msg_unref(msg);
      log_pipe_unref(&cloned_parser->super);
      return NULL;
    }
  log_pipe_unref(&cloned_parser->super);
  return msg;
}

static LogMessage *
parse_kv_into_log_message(const gchar *kv)
{
  LogMessage *msg;

  msg = parse_kv_into_log_message_no_check(kv);
  cr_assert_not_null(msg, "expected kv-parser success and it returned failure, kv=%s", kv);
  return msg;
}

void
setup(void)
{
  app_startup();
  cfg = cfg_new_snippet();
  kv_parser = kv_parser_new(cfg);
  log_pipe_init((LogPipe *)kv_parser);
}

void
teardown(void)
{
  log_pipe_deinit((LogPipe *)kv_parser);
  log_pipe_unref(&kv_parser->super);
  scratch_buffers_explicit_gc();
  cfg_free(cfg);
  app_shutdown();
}


Test(kv_parser, test_basics)
{
  LogMessage *msg;

  msg = parse_kv_into_log_message("foo=bar");
  assert_log_message_value(msg, log_msg_get_value_handle("foo"), "bar");
  log_msg_unref(msg);

  kv_parser_set_prefix(kv_parser, ".prefix.");
  msg = parse_kv_into_log_message("foo=bar");
  assert_log_message_value(msg, log_msg_get_value_handle(".prefix.foo"), "bar");
  log_msg_unref(msg);
}

Test(kv_parser, test_using_template_to_parse_input)
{
  LogMessage *msg;
  LogTemplate *template;

  template = log_template_new(NULL, NULL);
  log_template_compile_literal_string(template, "foo=bar");
  log_parser_set_template(kv_parser, template);
  msg = parse_kv_into_log_message("foo=this-value-doesnot-matter-as-template-overrides");
  assert_log_message_value(msg, log_msg_get_value_handle("foo"), "bar");
  log_msg_unref(msg);
}

Test(kv_parser, test_audit)
{
  LogMessage *msg;

  msg = parse_kv_into_log_message("type=EXECVE msg=audit(1436899154.146:186135): argc=6 a0=\"modprobe\" a1=\"--set-version=3.19.0-22-generic\" a2=\"--ignore-install\" a3=\"--quiet\" a4=\"--show-depends\" a5=\"sata_sis\"");
  assert_log_message_value_by_name(msg, "type", "EXECVE");
  assert_log_message_value_by_name(msg, "msg", "audit(1436899154.146:186135):");
  assert_log_message_value_by_name(msg, "argc", "6");
  assert_log_message_value_by_name(msg, "a0", "modprobe");
  assert_log_message_value_by_name(msg, "a1", "--set-version=3.19.0-22-generic");
  assert_log_message_value_by_name(msg, "a2", "--ignore-install");
  assert_log_message_value_by_name(msg, "a3", "--quiet");
  assert_log_message_value_by_name(msg, "a4", "--show-depends");
  assert_log_message_value_by_name(msg, "a5", "sata_sis");
  log_msg_unref(msg);

  msg = parse_kv_into_log_message("type=LOGIN msg=audit(1437419821.034:2972): pid=4160 uid=0 auid=0 ses=221 msg='op=PAM:session_close acct=\"root\" exe=\"/usr/sbin/cron\" hostname=? addr=? terminal=cron res=success'");
  assert_log_message_value_by_name(msg, "type", "LOGIN");
  /*  assert_log_message_value_by_name(msg, "msg", "audit(1437419821.034:2972):"); */
  assert_log_message_value_by_name(msg, "pid", "4160");
  assert_log_message_value_by_name(msg, "uid", "0");
  assert_log_message_value_by_name(msg, "auid", "0");
  assert_log_message_value_by_name(msg, "ses", "221");
  assert_log_message_value_by_name(msg, "msg",
                                   "op=PAM:session_close acct=\"root\" exe=\"/usr/sbin/cron\" hostname=? addr=? terminal=cron res=success");
  log_msg_unref(msg);
}

Test(kv_parser, test_extract_stray_words)
{
  LogMessage *msg;

  kv_parser_set_stray_words_value_name(kv_parser, "stray");
  kv_parser_set_prefix(kv_parser, ".junos.");
  kv_parser_set_pair_separator(kv_parser, ";");
  log_pipe_deinit((LogPipe *)kv_parser);
  log_pipe_init((LogPipe *)kv_parser);
  msg = parse_kv_into_log_message("VSYS=public; Slot=5/1; protocol=17; source-ip=10.116.214.221; source-port=50989; "
                                  "destination-ip=172.16.236.16; destination-port=162;time=2016/02/18 16:00:07; "
                                  "interzone-emtn_s1_vpn-enodeb_om inbound; policy=370;");
  assert_log_message_value_by_name(msg, ".junos.VSYS", "public");
  assert_log_message_value_by_name(msg, ".junos.Slot", "5/1");
  assert_log_message_value_by_name(msg, ".junos.protocol", "17");
  assert_log_message_value_by_name(msg, ".junos.source-ip", "10.116.214.221");
  assert_log_message_value_by_name(msg, ".junos.source-port", "50989");
  assert_log_message_value_by_name(msg, ".junos.destination-ip", "172.16.236.16");
  assert_log_message_value_by_name(msg, ".junos.destination-port", "162");
  assert_log_message_value_by_name(msg, ".junos.time", "2016/02/18 16:00:07");
  assert_log_message_value_by_name(msg, ".junos.policy", "370");
  assert_log_message_value_by_name(msg, "stray", "\"interzone-emtn_s1_vpn-enodeb_om inbound;\"");
  log_msg_unref(msg);

}

TestSuite(kv_parser, .init = setup, .fini = teardown);
