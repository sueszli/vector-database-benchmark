/*
 * Copyright (c) 2019 Balazs Scheidler <bazsi77@gmail.com>
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
#include "rewrite-fix-timezone.h"
#include "scratch-buffers.h"
#include "timeutils/cache.h"

typedef struct _RewriteFixTimeZone RewriteFixTimeZone;

struct _RewriteFixTimeZone
{
  LogRewrite super;
  LogTemplate *zone_template;
  gint stamp;
};

void
rewrite_fix_time_zone_set_zone_template_ref(LogRewrite *s, LogTemplate *zone_template)
{
  RewriteFixTimeZone *self = (RewriteFixTimeZone *) s;

  log_template_unref(self->zone_template);
  self->zone_template = zone_template;
}

void
rewrite_fix_time_zone_set_time_stamp(LogRewrite *s, gint stamp)
{
  RewriteFixTimeZone *self = (RewriteFixTimeZone *) s;
  self->stamp = stamp;
}

static void
_process(LogRewrite *s, LogMessage **pmsg, const LogPathOptions *path_options)
{
  RewriteFixTimeZone *self = (RewriteFixTimeZone *) s;
  GString *result = scratch_buffers_alloc();
  LogMessage *msg = *pmsg;

  log_template_format(self->zone_template, msg, &DEFAULT_TEMPLATE_EVAL_OPTIONS, result);

  UnixTime stamp = msg->timestamps[self->stamp];
  glong implied_gmtoff = stamp.ut_gmtoff;
  TimeZoneInfo *tzinfo = cached_get_time_zone_info(result->str);

  unix_time_fix_timezone_with_tzinfo(&stamp, tzinfo);
  if (stamp.ut_gmtoff != msg->timestamps[self->stamp].ut_gmtoff)
    {
      /* only clone the message in case the time is indeed to be changed */
      msg = log_msg_make_writable(pmsg, path_options);

      msg->timestamps[self->stamp] = stamp;
    }
  msg_trace("fix-timezone(): adjusting message timezone assuming it was improperly recognized",
            evt_tag_str("new_timezone", result->str),
            evt_tag_long("implied_gmtoff", implied_gmtoff),
            evt_tag_long("new_gmtoff", msg->timestamps[self->stamp].ut_gmtoff));
}

static LogPipe *
_clone(LogPipe *s)
{
  RewriteFixTimeZone *self = (RewriteFixTimeZone *) s;
  LogRewrite *cloned;

  cloned = rewrite_fix_time_zone_new(s->cfg);
  log_rewrite_clone_method(cloned, &self->super);

  rewrite_fix_time_zone_set_zone_template_ref(cloned, log_template_ref(self->zone_template));
  rewrite_fix_time_zone_set_time_stamp(cloned, self->stamp);

  return &cloned->super;
}

static void
_free(LogPipe *s)
{
  RewriteFixTimeZone *self = (RewriteFixTimeZone *) s;

  log_template_unref(self->zone_template);
  log_rewrite_free_method(s);
}

LogRewrite *
rewrite_fix_time_zone_new(GlobalConfig *cfg)
{
  RewriteFixTimeZone *self = g_new0(RewriteFixTimeZone, 1);

  log_rewrite_init_instance(&self->super, cfg);
  self->super.super.free_fn = _free;
  self->super.super.clone = _clone;
  self->super.process = _process;
  self->stamp = LM_TS_STAMP;
  return &self->super;
}
