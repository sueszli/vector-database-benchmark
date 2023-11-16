/*
 * Copyright (c) 2018 Balabit
 * Copyright (c) 2018 László Várady <laszlo.varady@balabit.com>
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
 *
 */

#include "msg-generator-source.h"

#include "logsource.h"
#include "logpipe.h"
#include "logmsg/logmsg.h"
#include "template/templates.h"
#include "timeutils/misc.h"
#include "ack-tracker/ack_tracker_factory.h"

#include <iv.h>

struct MsgGeneratorSource
{
  LogSource super;
  MsgGeneratorSourceOptions *options;
  struct iv_timer timer;
  gint num;
};

static void
_stop_timer(MsgGeneratorSource *self)
{
  if (iv_timer_registered(&self->timer))
    iv_timer_unregister(&self->timer);
}

static void
_start_timer(MsgGeneratorSource *self)
{
  iv_validate_now();
  self->timer.expires = iv_now;
  timespec_add_msec(&self->timer.expires, self->options->freq);

  iv_timer_register(&self->timer);
}

static void
_start_timer_immediately(MsgGeneratorSource *self)
{
  iv_validate_now();
  self->timer.expires = iv_now;

  iv_timer_register(&self->timer);
}

static gboolean
_init(LogPipe *s)
{
  MsgGeneratorSource *self = (MsgGeneratorSource *) s;
  if (!log_source_init(s))
    return FALSE;

  _start_timer_immediately(self);

  return TRUE;
}

static gboolean
_deinit(LogPipe *s)
{
  MsgGeneratorSource *self = (MsgGeneratorSource *) s;

  _stop_timer(self);

  return log_source_deinit(s);
}

static void
_free(LogPipe *s)
{
  log_source_free(s);
}

static void
_add_name_value(gpointer key, gpointer value, gpointer data)
{
  gchar *name = (gchar *) key;
  LogTemplate *val = (LogTemplate *) value;
  LogMessage *msg = (LogMessage *) data;
  LogMessageValueType type;

  GString *msg_body = g_string_sized_new(128);
  log_template_format_value_and_type(val, msg, &DEFAULT_TEMPLATE_EVAL_OPTIONS, msg_body, &type);
  log_msg_set_value_by_name_with_type(msg, name, msg_body->str, msg_body->len, type);
  g_string_free(msg_body, TRUE);

}

static void
_send_generated_message(MsgGeneratorSource *self)
{
  if (!log_source_free_to_send(&self->super))
    return;

  LogMessage *msg = log_msg_new_empty();
  g_hash_table_foreach(self->options->name_value, _add_name_value, msg);
  log_msg_set_value(msg, LM_V_MESSAGE, "-- Generated message. --", -1);

  if (self->options->template)
    {
      GString *msg_body = g_string_sized_new(128);
      log_template_format(self->options->template, msg, &DEFAULT_TEMPLATE_EVAL_OPTIONS, msg_body);
      log_msg_set_value(msg, LM_V_MESSAGE, msg_body->str, msg_body->len);
      g_string_free(msg_body, TRUE);
    }

  msg_debug("Incoming generated message", evt_tag_str("msg", log_msg_get_value(msg, LM_V_MESSAGE, NULL)));
  log_source_post(&self->super, msg);
}

static void
_timer_expired(void *cookie)
{
  MsgGeneratorSource *self = (MsgGeneratorSource *) cookie;

  _send_generated_message(self);

  if (self->options->max_num > 0)
    {
      self->num++;
      if (self->num < self->options->max_num)
        {
          _start_timer(self);
        }
    }
  else
    {
      _start_timer(self);
    }
}

gboolean
msg_generator_source_init(MsgGeneratorSource *self)
{
  return log_pipe_init(&self->super.super);
}

gboolean
msg_generator_source_deinit(MsgGeneratorSource *self)
{
  return log_pipe_deinit(&self->super.super);
}

void
msg_generator_source_free(MsgGeneratorSource *self)
{
  log_pipe_unref(&self->super.super);
}

void
msg_generator_source_set_options(MsgGeneratorSource *self, MsgGeneratorSourceOptions *options,
                                 const gchar *stats_id, StatsClusterKeyBuilder *kb, gboolean threaded,
                                 gboolean pos_tracked, LogExprNode *expr_node)
{
  log_source_set_options(&self->super, &options->super, stats_id, kb, threaded, expr_node);

  AckTrackerFactory *factory = pos_tracked ? consecutive_ack_tracker_factory_new() :
                               instant_ack_tracker_bookmarkless_factory_new();
  log_source_set_ack_tracker_factory(&self->super, factory);

  self->options = options;
}

MsgGeneratorSource *
msg_generator_source_new(GlobalConfig *cfg)
{
  MsgGeneratorSource *self = g_new0(MsgGeneratorSource, 1);
  log_source_init_instance(&self->super, cfg);

  IV_TIMER_INIT(&self->timer);
  self->timer.cookie = self;
  self->timer.handler = _timer_expired;

  self->super.super.init = _init;
  self->super.super.deinit = _deinit;
  self->super.super.free_fn = _free;

  return self;
}
