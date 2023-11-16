/*
 * Copyright (c) 2002-2013, 2015 Balabit
 * Copyright (c) 1998-2013, 2015 Balázs Scheidler
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
#include "correlation-context.h"
#include "logmsg/logmsg.h"
#include "scratch-buffers.h"
#include <string.h>

static gint
_compare_messages_with_trivial_template(gconstpointer a, gconstpointer b, gpointer user_data)
{
  LogMessage *am = *(LogMessage **) a;
  LogMessage *bm = *(LogMessage **) b;
  gssize av_len, bv_len;
  LogTemplate *sort_key = (LogTemplate *) user_data;

  const gchar *am_key = log_template_get_trivial_value(sort_key, am, &av_len);
  const gchar *bm_key = log_template_get_trivial_value(sort_key, bm, &bv_len);

  if (am_key == NULL && bm_key == NULL)
    return 0;

  if (am_key == NULL)
    return -1;

  if (bm_key == NULL)
    return 1;

  gssize cmp_len = MIN(av_len, bv_len);
  return strncmp(am_key, bm_key, cmp_len);
}

static gint
_compare_messages_with_nontrivial_template(gconstpointer a, gconstpointer b, gpointer user_data)
{
  LogMessage *am = *(LogMessage **) a;
  LogMessage *bm = *(LogMessage **) b;
  LogTemplate *sort_key = (LogTemplate *) user_data;

  ScratchBuffersMarker mark;
  GString *am_key = scratch_buffers_alloc_and_mark(&mark);
  GString *bm_key = scratch_buffers_alloc();

  log_template_format(sort_key, am, &DEFAULT_TEMPLATE_EVAL_OPTIONS, am_key);
  log_template_format(sort_key, bm, &DEFAULT_TEMPLATE_EVAL_OPTIONS, bm_key);
  gint result = strcmp(am_key->str, bm_key->str);

  scratch_buffers_reclaim_marked(mark);
  return result;
}

void
correlation_context_sort(CorrelationContext *self, LogTemplate *sort_key)
{
  if (log_template_is_trivial(sort_key))
    g_ptr_array_sort_with_data(self->messages, _compare_messages_with_trivial_template, sort_key);
  else
    g_ptr_array_sort_with_data(self->messages, _compare_messages_with_nontrivial_template, sort_key);
}

void
correlation_context_init(CorrelationContext *self, const CorrelationKey *key)
{
  self->clear = correlation_context_clear_method;

  self->messages = g_ptr_array_new();
  memcpy(&self->key, key, sizeof(self->key));

  if (self->key.pid)
    self->key.pid = g_strdup(self->key.pid);
  if (self->key.program)
    self->key.program = g_strdup(self->key.program);
  if (self->key.host)
    self->key.host = g_strdup(self->key.host);
  self->ref_cnt = 1;
  self->free_fn = correlation_context_free_method;
}

void
correlation_context_clear_method(CorrelationContext *self)
{
  for (gint i = 0; i < self->messages->len; i++)
    {
      log_msg_unref((LogMessage *) g_ptr_array_index(self->messages, i));
    }
  g_ptr_array_set_size(self->messages, 0);
}

void
correlation_context_free_method(CorrelationContext *self)
{
  correlation_context_clear(self);
  g_ptr_array_free(self->messages, TRUE);

  if (self->key.host)
    g_free((gchar *) self->key.host);
  if (self->key.program)
    g_free((gchar *) self->key.program);
  if (self->key.pid)
    g_free((gchar *) self->key.pid);
  g_free(self->key.session_id);
}

CorrelationContext *
correlation_context_new(CorrelationKey *key)
{
  CorrelationContext *self = g_new0(CorrelationContext, 1);

  correlation_context_init(self, key);
  return self;
}

CorrelationContext *
correlation_context_ref(CorrelationContext *self)
{
  self->ref_cnt++;
  return self;
}

void
correlation_context_unref(CorrelationContext *self)
{
  if (--self->ref_cnt == 0)
    {
      if (self->free_fn)
        self->free_fn(self);
      g_free(self);
    }
}
