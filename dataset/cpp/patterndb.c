/*
 * Copyright (c) 2002-2013 Balabit
 * Copyright (c) 1998-2013 Balázs Scheidler
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

#include "patterndb.h"
#include "pdb-action.h"
#include "pdb-rule.h"
#include "pdb-program.h"
#include "pdb-ruleset.h"
#include "pdb-load.h"
#include "pdb-context.h"
#include "pdb-ratelimit.h"
#include "pdb-lookup-params.h"
#include "correlation.h"
#include "logmsg/logmsg.h"
#include "template/templates.h"
#include "str-utils.h"
#include "filter/filter-expr-parser.h"
#include "logpipe.h"
#include "timeutils/cache.h"
#include "timeutils/misc.h"

#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

static NVHandle context_id_handle = 0;

#define EXPECTED_NUMBER_OF_MESSAGES_EMITTED 32

typedef struct _PDBProcessParams
{
  PDBRule *rule;
  PDBAction *action;
  PDBContext *context;
  LogMessage *msg;
  gpointer emitted_messages[EXPECTED_NUMBER_OF_MESSAGES_EMITTED];
  GPtrArray *emitted_messages_overflow;
  gint num_emitted_messages;
} PDBProcessParams;

struct _PatternDB
{
  GMutex ruleset_lock;
  PDBRuleSet *ruleset;
  CorrelationState *correlation;
  LogTemplate *program_template;
  GHashTable *rate_limits;
  PatternDBEmitFunc emit;
  gpointer emit_data;
  gchar *prefix;
};

/* This function is called to populate the emitted_messages array in
 * process_params.  It only manipulates per-thread data structure so it does
 * not require locks but does not mind them being locked either.  */
static void
_emit_message(PatternDB *self, PDBProcessParams *process_params, LogMessage *msg)
{
  if (!self->emit)
    return;

  if (process_params->num_emitted_messages < EXPECTED_NUMBER_OF_MESSAGES_EMITTED)
    {
      process_params->emitted_messages[process_params->num_emitted_messages++] = msg;
    }
  else
    {
      if (!process_params->emitted_messages_overflow)
        process_params->emitted_messages_overflow = g_ptr_array_new();

      g_ptr_array_add(process_params->emitted_messages_overflow, msg);
    }
  log_msg_ref(msg);
}

static void
_send_emitted_message_array(PatternDB *self, gpointer *values, gsize len)
{
  /* if emit is NULL, we don't store any entries in the arrays, so no need
   * to check it here.  */

  for (gint i = 0; i < len; i++)
    {
      LogMessage *msg = values[i];

      self->emit(msg, self->emit_data);
      log_msg_unref(msg);
    }
}

/* This function is called to flush the accumulated list of messages that
 * are generated during rule evaluation.  We must not hold any locks within
 * PatternDB when doing this, as it will cause log_pipe_queue() calls to
 * subsequent elements in the message pipeline, which in turn may recurse
 * into PatternDB.  This works as process_params itself is per-thread
 * (actually an auto variable on the stack), and this is called without
 * locks held at the end of a pattern_db_process() invocation. */
static void
_flush_emitted_messages(PatternDB *self, PDBProcessParams *process_params)
{
  /* send inline elements */
  _send_emitted_message_array(self, process_params->emitted_messages, process_params->num_emitted_messages);
  process_params->num_emitted_messages = 0;
  if (process_params->emitted_messages_overflow)
    {
      /* send overflow area */
      _send_emitted_message_array(self, process_params->emitted_messages_overflow->pdata,
                                  process_params->emitted_messages_overflow->len);
      g_ptr_array_free(process_params->emitted_messages_overflow, TRUE);
      process_params->emitted_messages_overflow = NULL;
    }
}

/*
 * Timing
 * ======
 *
 * The time tries to follow the message stream, e.g. it is independent from
 * the current system time.  Whenever a message comes in, its timestamp
 * moves the current time forward, which means it is quite easy to process
 * logs from the past, correlation timeouts will be measured in "message
 * time".  There's one exception to this rule: when the patterndb is idle
 * (e.g.  no messages are coming in), the current system time is used to
 * measure as real time passes, and that will also increase the time of the
 * correlation engine. This is based on the following assumptions:
 *
 *    1) dbparser can only be idle in case on-line logs are processed
 *       (otherwise messages are read from the disk much faster)
 *
 *    2) if on-line processing is done, it is expected that messages have
 *       roughly correct timestamps, e.g. if 1 second passes in current
 *       system time, further incoming messages will have a timestamp close
 *       to this.
 *
 * Thus whenever the patterndb is idle, a timer tick callback arrives, which
 * checks the real elapsed time between the last message (or last tick) and
 * increments the current known time with this value.
 *
 * This behaviour makes it possible to properly work in these use-cases:
 *
 *    1) process a log file stored on disk, containing messages in the past
 *    2) process an incoming message stream on-line, expiring correlation
 *    states even if there are no incoming messages
 *
 */


/*********************************************
 * Rule evaluation
 *********************************************/

static gboolean
_is_action_within_rate_limit(PatternDB *db, PDBProcessParams *process_params)
{
  PDBRule *rule = process_params->rule;
  PDBAction *action = process_params->action;
  LogMessage *msg = process_params->msg;
  GString *buffer = g_string_sized_new(256);

  CorrelationKey key;
  PDBRateLimit *rl;
  guint64 now;

  if (action->rate == 0)
    return TRUE;

  g_string_printf(buffer, "%s:%d", rule->rule_id, action->id);
  correlation_key_init(&key, rule->context.scope, msg, buffer->str);

  rl = g_hash_table_lookup(db->rate_limits, &key);
  if (!rl)
    {
      rl = pdb_rate_limit_new(&key);
      g_hash_table_insert(db->rate_limits, &rl->key, rl);
      g_string_free(buffer, FALSE);
    }
  else
    {
      g_string_free(buffer, TRUE);
    }

  now = correlation_state_get_time(db->correlation);
  if (rl->last_check == 0)
    {
      rl->last_check = now;
      rl->buckets = action->rate;
    }
  else
    {
      /* quick and dirty fixed point arithmetic, 8 bit fraction part */
      gint new_credits = (((glong) (now - rl->last_check)) << 8) / ((((glong) action->rate_quantum) << 8) / action->rate);

      if (new_credits)
        {
          /* ok, enough time has passed to increase the current credit.
           * Deposit the new credits in bucket but make sure we don't permit
           * more than the maximum rate. */

          rl->buckets = MIN(rl->buckets + new_credits, action->rate);
          rl->last_check = now;
        }
    }
  if (rl->buckets)
    {
      rl->buckets--;
      return TRUE;
    }
  return FALSE;
}

static gboolean
_is_action_triggered(PatternDB *db, PDBProcessParams *process_params, PDBActionTrigger trigger)
{
  PDBAction *action = process_params->action;
  PDBContext *context = process_params->context;
  LogMessage *msg = process_params->msg;

  if (action->trigger != trigger)
    return FALSE;

  if (action->condition)
    {
      if (context
          && !filter_expr_eval_with_context(action->condition, (LogMessage **) context->super.messages->pdata,
                                            context->super.messages->len, &DEFAULT_TEMPLATE_EVAL_OPTIONS))
        return FALSE;
      if (!context && !filter_expr_eval(action->condition, msg))
        return FALSE;
    }

  if (!_is_action_within_rate_limit(db, process_params))
    return FALSE;

  return TRUE;
}

static LogMessage *
_generate_synthetic_message(PDBProcessParams *process_params)
{
  PDBAction *action = process_params->action;
  PDBContext *context = process_params->context;
  LogMessage *msg = process_params->msg;

  if (context)
    return synthetic_message_generate_with_context(&action->content.message, &context->super);
  else
    return synthetic_message_generate_without_context(&action->content.message, msg);
}

static void
_execute_action_message(PatternDB *db, PDBProcessParams *process_params)
{
  LogMessage *genmsg;

  genmsg = _generate_synthetic_message(process_params);
  _emit_message(db, process_params, genmsg);
  log_msg_unref(genmsg);
}

static void pattern_db_expire_entry(TimerWheel *wheel, guint64 now, gpointer user_data, gpointer caller_context);

static void
_execute_action_create_context(PatternDB *db, PDBProcessParams *process_params)
{
  CorrelationKey key;
  PDBAction *action = process_params->action;
  PDBRule *rule = process_params->rule;
  PDBContext *triggering_context = process_params->context;
  LogMessage *triggering_msg = process_params->msg;
  GString *buffer = g_string_sized_new(256);
  PDBContext *new_context;
  LogMessage *context_msg;
  SyntheticContext *syn_context;
  SyntheticMessage *syn_message;

  syn_context = &action->content.create_context.context;
  syn_message = &action->content.create_context.message;
  if (triggering_context)
    {
      context_msg = synthetic_message_generate_with_context(syn_message, &triggering_context->super);
      log_template_format_with_context(syn_context->id_template,
                                       (LogMessage **) triggering_context->super.messages->pdata, triggering_context->super.messages->len,
                                       &DEFAULT_TEMPLATE_EVAL_OPTIONS, buffer);
    }
  else
    {
      context_msg = synthetic_message_generate_without_context(syn_message, triggering_msg);
      log_template_format(syn_context->id_template,
                          triggering_msg,
                          &DEFAULT_TEMPLATE_EVAL_OPTIONS, buffer);
    }

  msg_debug("Explicit create-context action, starting a new context",
            evt_tag_str("rule", rule->rule_id),
            evt_tag_str("context", buffer->str),
            evt_tag_int("context_timeout", syn_context->timeout),
            evt_tag_int("context_expiration", correlation_state_get_time(db->correlation) + syn_context->timeout));

  correlation_key_init(&key, syn_context->scope, context_msg, buffer->str);
  new_context = pdb_context_new(&key);
  correlation_state_tx_store_context(db->correlation, &new_context->super, rule->context.timeout);
  g_string_free(buffer, FALSE);

  g_ptr_array_add(new_context->super.messages, context_msg);

  new_context->rule = pdb_rule_ref(rule);
}

static void
_execute_action(PatternDB *db, PDBProcessParams *process_params)
{
  PDBAction *action = process_params->action;

  switch (action->content_type)
    {
    case RAC_NONE:
      break;
    case RAC_MESSAGE:
      _execute_action_message(db, process_params);
      break;
    case RAC_CREATE_CONTEXT:
      _execute_action_create_context(db, process_params);
      break;
    default:
      g_assert_not_reached();
      break;
    }
}

static void
_execute_action_if_triggered(PatternDB *db, PDBProcessParams *process_params, PDBActionTrigger trigger)
{
  if (_is_action_triggered(db, process_params, trigger))
    _execute_action(db, process_params);
}

static void
_execute_rule_actions(PatternDB *db, PDBProcessParams *process_params, PDBActionTrigger trigger)
{
  gint i;
  PDBRule *rule = process_params->rule;

  if (!rule->actions)
    return;

  for (i = 0; i < rule->actions->len; i++)
    {
      process_params->action = (PDBAction *) g_ptr_array_index(rule->actions, i);

      _execute_action_if_triggered(db, process_params, trigger);
    }
}

/*********************************************************
 * PatternDB
 *********************************************************/

/* NOTE: this function requires PatternDB reader/writer lock to be
 * write-locked.
 *
 * Currently, it is, as timer_wheel_set_time() is only called with that
 * precondition, and timer-wheel callbacks are only called from within
 * timer_wheel_set_time().
 */

static void
pattern_db_expire_entry(TimerWheel *wheel, guint64 now, gpointer user_data, gpointer caller_context)
{
  PDBContext *context = user_data;
  PatternDB *pdb = (PatternDB *) timer_wheel_get_associated_data(wheel);
  LogMessage *msg = correlation_context_get_last_message(&context->super);
  PDBProcessParams *process_params = caller_context;

  msg_debug("Expiring patterndb correlation context",
            evt_tag_str("last_rule", context->rule->rule_id),
            evt_tag_long("utc", correlation_state_get_time(pdb->correlation)));
  process_params->context = context;
  process_params->rule = context->rule;
  process_params->msg = msg;

  _execute_rule_actions(pdb, process_params, RAT_TIMEOUT);
  context->super.timer = NULL;
  correlation_state_tx_remove_context(pdb->correlation, &context->super);

  /* pdb_context_free is automatically called when returning from
     this function by the timerwheel code as a destroy notify
     callback. */
}

/*
 * This function can be called any time when pattern-db is not processing
 * messages, but we expect the correlation timer to move forward.  It
 * doesn't need to be called absolutely regularly as it'll use the current
 * system time to determine how much time has passed since the last
 * invocation.  See the timing comment at pattern_db_process() for more
 * information.
 */
void
pattern_db_timer_tick(PatternDB *self)
{
  PDBProcessParams process_params = {0};

  if (correlation_state_timer_tick(self->correlation, &process_params))
    {
      msg_debug("Advancing patterndb current time because of timer tick",
                evt_tag_long("utc", correlation_state_get_time(self->correlation)));
    }
  _flush_emitted_messages(self, &process_params);
}

/* NOTE: lock should be acquired for writing before calling this function. */
static void
_advance_time_based_on_message(PatternDB *self, PDBProcessParams *process_params, const UnixTime *ls)
{
  correlation_state_set_time(self->correlation, ls->ut_sec, process_params);

  msg_debug("Advancing patterndb current time because of an incoming message",
            evt_tag_long("utc", correlation_state_get_time(self->correlation)));
}

void
pattern_db_advance_time(PatternDB *self, gint timeout)
{
  PDBProcessParams process_params= {0};

  correlation_state_advance_time(self->correlation, timeout, &process_params);
  _flush_emitted_messages(self, &process_params);
}

gboolean
pattern_db_reload_ruleset(PatternDB *self, GlobalConfig *cfg, const gchar *pdb_file)
{
  PDBRuleSet *new_ruleset;

  new_ruleset = pdb_rule_set_new(self->prefix);
  if (!pdb_rule_set_load(new_ruleset, cfg, pdb_file, NULL))
    {
      pdb_rule_set_free(new_ruleset);
      return FALSE;
    }
  else
    {
      g_mutex_lock(&self->ruleset_lock);
      if (self->ruleset)
        pdb_rule_set_free(self->ruleset);
      self->ruleset = new_ruleset;
      g_mutex_unlock(&self->ruleset_lock);
      return TRUE;
    }
}


void
pattern_db_set_emit_func(PatternDB *self, PatternDBEmitFunc emit, gpointer emit_data)
{
  self->emit = emit;
  self->emit_data = emit_data;
}

void
pattern_db_set_program_template(PatternDB *self, LogTemplate *program_template)
{
  log_template_unref(self->program_template);
  self->program_template = log_template_ref(program_template);
}

const gchar *
pattern_db_get_ruleset_pub_date(PatternDB *self)
{
  return self->ruleset->pub_date;
}

const gchar *
pattern_db_get_ruleset_version(PatternDB *self)
{
  return self->ruleset->version;
}

PDBRuleSet *
pattern_db_get_ruleset(PatternDB *self)
{
  return self->ruleset;
}

static gboolean
_pattern_db_is_empty(PatternDB *self)
{
  return (G_UNLIKELY(!self->ruleset) || self->ruleset->is_empty);
}

static void
_pattern_db_process_matching_rule(PatternDB *self, PDBProcessParams *process_params)
{
  PDBContext *context = NULL;
  PDBRule *rule = process_params->rule;
  LogMessage *msg = process_params->msg;
  GString *buffer = g_string_sized_new(32);

  correlation_state_tx_begin(self->correlation);
  if (rule->context.id_template)
    {
      CorrelationKey key;

      log_template_format(rule->context.id_template, msg, &DEFAULT_TEMPLATE_EVAL_OPTIONS, buffer);
      log_msg_set_value(msg, context_id_handle, buffer->str, -1);

      correlation_key_init(&key, rule->context.scope, msg, buffer->str);
      context = (PDBContext *) correlation_state_tx_lookup_context(self->correlation, &key);
      if (!context)
        {
          msg_debug("Correlation context lookup failure, starting a new context",
                    evt_tag_str("rule", rule->rule_id),
                    evt_tag_str("context", buffer->str),
                    evt_tag_int("context_timeout", rule->context.timeout),
                    evt_tag_int("context_expiration", correlation_state_get_time(self->correlation) + rule->context.timeout));
          context = pdb_context_new(&key);
          correlation_state_tx_store_context(self->correlation, &context->super, rule->context.timeout);
          g_string_steal(buffer);
        }
      else
        {
          msg_debug("Correlation context lookup successful",
                    evt_tag_str("rule", rule->rule_id),
                    evt_tag_str("context", buffer->str),
                    evt_tag_int("context_timeout", rule->context.timeout),
                    evt_tag_int("context_expiration", correlation_state_get_time(self->correlation) + rule->context.timeout),
                    evt_tag_int("num_messages", context->super.messages->len));
        }

      g_ptr_array_add(context->super.messages, log_msg_ref(msg));

      correlation_state_tx_update_context(self->correlation, &context->super, rule->context.timeout);
      if (context->rule != rule)
        {
          if (context->rule)
            pdb_rule_unref(context->rule);
          context->rule = pdb_rule_ref(rule);
        }
    }
  else
    {
      context = NULL;
    }

  process_params->context = context;

  synthetic_message_apply(&rule->msg, context ? &context->super : NULL, msg);

  _execute_rule_actions(self, process_params, RAT_MATCH);

  pdb_rule_unref(rule);
  correlation_state_tx_end(self->correlation);

  if (context)
    log_msg_write_protect(msg);

  g_string_free(buffer, TRUE);
}

static void
_pattern_db_advance_time_and_flush_expired(PatternDB *self, LogMessage *msg)
{
  PDBProcessParams process_params = {0};

  _advance_time_based_on_message(self, &process_params, &msg->timestamps[LM_TS_STAMP]);
  _flush_emitted_messages(self, &process_params);
}

static gboolean
_pattern_db_process(PatternDB *self, PDBLookupParams *lookup, GArray *dbg_list)
{
  LogMessage *msg = lookup->msg;
  PDBProcessParams process_params_p = {0};
  PDBProcessParams *process_params = &process_params_p;

  g_mutex_lock(&self->ruleset_lock);
  if (_pattern_db_is_empty(self))
    {
      g_mutex_unlock(&self->ruleset_lock);
      return FALSE;
    }
  process_params->rule = pdb_ruleset_lookup(self->ruleset, lookup, dbg_list);
  process_params->msg = msg;
  g_mutex_unlock(&self->ruleset_lock);

  _pattern_db_advance_time_and_flush_expired(self, msg);

  if (process_params->rule)
    _pattern_db_process_matching_rule(self, process_params);

  _flush_emitted_messages(self, process_params);

  return process_params->rule != NULL;
}

gboolean
pattern_db_process(PatternDB *self, LogMessage *msg)
{
  PDBLookupParams lookup;

  pdb_lookup_params_init(&lookup, msg, self->program_template);
  return _pattern_db_process(self, &lookup, NULL);
}

gboolean
pattern_db_process_with_custom_message(PatternDB *self, LogMessage *msg, const gchar *message, gssize message_len)
{
  PDBLookupParams lookup;

  pdb_lookup_params_init(&lookup, msg, self->program_template);
  pdb_lookup_params_override_message(&lookup, message, message_len);
  return _pattern_db_process(self, &lookup, NULL);
}

void
pattern_db_debug_ruleset(PatternDB *self, LogMessage *msg, GArray *dbg_list)
{
  PDBLookupParams lookup;

  pdb_lookup_params_init(&lookup, msg, NULL);
  _pattern_db_process(self, &lookup, dbg_list);
}

void
pattern_db_expire_state(PatternDB *self)
{
  PDBProcessParams process_params = {0};

  correlation_state_expire_all(self->correlation, &process_params);
  _flush_emitted_messages(self, &process_params);

}

static void
_init_state(PatternDB *self)
{
  self->rate_limits = g_hash_table_new_full(correlation_key_hash, correlation_key_equal, NULL,
                                            (GDestroyNotify) pdb_rate_limit_free);
  self->correlation = correlation_state_new(pattern_db_expire_entry);
  timer_wheel_set_associated_data(self->correlation->timer_wheel, self, NULL);
}

static void
_destroy_state(PatternDB *self)
{
  g_hash_table_destroy(self->rate_limits);
  correlation_state_unref(self->correlation);
  self->correlation = NULL;
}


/* NOTE: this function is for testing only and is not expecting parallel
 * threads taking actions within the same PatternDB instance. */
void
pattern_db_forget_state(PatternDB *self)
{
  _destroy_state(self);
  _init_state(self);
}

PatternDB *
pattern_db_new(const gchar *prefix)
{
  PatternDB *self = g_new0(PatternDB, 1);

  self->prefix = g_strdup(prefix);
  self->ruleset = pdb_rule_set_new(self->prefix);
  g_mutex_init(&self->ruleset_lock);
  _init_state(self);
  return self;
}

void
pattern_db_free(PatternDB *self)
{
  g_free(self->prefix);
  log_template_unref(self->program_template);
  if (self->ruleset)
    pdb_rule_set_free(self->ruleset);
  _destroy_state(self);
  g_mutex_clear(&self->ruleset_lock);
  g_free(self);
}

void
pattern_db_global_init(void)
{
  context_id_handle = log_msg_get_value_handle(".classifier.context_id");
  pdb_rule_set_global_init();
}
