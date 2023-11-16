/*
 * Copyright (c) 2002-2012 Balabit
 * Copyright (c) 1998-2012 Balázs Scheidler
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

#include "logqueue.h"
#include "logpipe.h"
#include "messages.h"
#include "serialize.h"
#include "stats/stats-registry.h"
#include "mainloop-worker.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <iv_thread.h>

QueueType log_queue_fifo_type = "FIFO";

/*
 * LogFifo is a scalable first-in-first-output queue implementation, that:
 *
 *   - has a per-thread, unlocked input queue where threads can put their items
 *
 *   - has a locked wait-queue where items go once the per-thread input
 *     would be overflown or if the input thread goes to sleep (e.g.  one
 *     lock acquisition per a longer period)
 *
 *   - has an unlocked output queue where items from the wait queue go, once
 *     it becomes depleted.
 *
 * This means that items flow in this sequence from one list to the next:
 *
 *    input queue (per-thread) -> wait queue (locked) -> output queue (single-threaded)
 *
 * Fastpath is:
 *   - input threads putting elements on their per-thread queue (lockless)
 *   - output threads removing elements from the output queue (lockless)
 *
 * Slowpath:
 *   - input queue is overflown (or the input thread goes to sleep), wait
 *     queue mutex is grabbed, all elements are put to the wait queue.
 *
 *   - output queue is depleted, wait queue mutex is grabbed, all elements
 *     on the wait queue is put to the output queue
 *
 * Threading assumptions:
 *   - the head of the queue is only manipulated from the output thread
 *   - the tail of the queue is only manipulated from the input threads
 *
 */

typedef struct _InputQueue
{
  struct iv_list_head items;
  WorkerBatchCallback cb;
  guint32 len;
  guint32 non_flow_controlled_len;
  guint16 finish_cb_registered;
} InputQueue;

typedef struct _OverflowQueue
{
  struct iv_list_head items;
  gint len;
  gint non_flow_controlled_len;
} OverflowQueue;

typedef struct _LogQueueFifo
{
  LogQueue super;

  /* scalable qoverflow implementation */
  OverflowQueue output_queue;
  OverflowQueue wait_queue;
  OverflowQueue backlog_queue; /* entries that were sent but not acked yet */

  gint log_fifo_size;

  /* legacy: flow-controlled messages are included in the log_fifo_size limit */
  gboolean use_legacy_fifo_size;

  gint num_input_queues;
  InputQueue input_queues[0];
} LogQueueFifo;

/* NOTE: this is inherently racy. If the LogQueue->lock is taken, then the
 * race is limited to the changes in output_queue queue changes.
 *
 * In the output thread, this means that this can get race-free. In the
 * input thread, the output_queue can change because of a
 * log_queue_fifo_rewind_backlog().
 *
 */

static void
iv_list_update_msg_size(LogQueueFifo *self, struct iv_list_head *head)
{
  LogMessage *msg;
  struct iv_list_head *ilh, *ilh2;
  iv_list_for_each_safe(ilh, ilh2, head)
  {
    msg = iv_list_entry(ilh, LogMessageQueueNode, list)->msg;
    log_queue_memory_usage_add(&self->super, log_msg_get_size(msg));
  }
}

static gint64
log_queue_fifo_get_length(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;

  return self->wait_queue.len + self->output_queue.len;
}

static gint64
log_queue_fifo_get_non_flow_controlled_length(LogQueueFifo *self)
{
  return self->wait_queue.non_flow_controlled_len + self->output_queue.non_flow_controlled_len;
}

gboolean
log_queue_fifo_is_empty_racy(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  gboolean has_message_in_queue = FALSE;
  g_mutex_lock(&self->super.lock);
  if (log_queue_fifo_get_length(s) > 0)
    {
      has_message_in_queue = TRUE;
    }
  else
    {
      gint i;
      for (i = 0; i < self->num_input_queues && !has_message_in_queue; i++)
        {
          has_message_in_queue |= self->input_queues[i].finish_cb_registered;
        }
    }
  g_mutex_unlock(&self->super.lock);
  return !has_message_in_queue;
}

/* NOTE: this is inherently racy, can only be called if log processing is suspended (e.g. reload time) */
static gboolean
log_queue_fifo_keep_on_reload(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  return log_queue_fifo_get_length(s) > 0 || self->backlog_queue.len > 0;
}

static inline void
log_queue_fifo_drop_messages_from_input_queue(LogQueueFifo *self, InputQueue *input_queue, gint num_of_messages_to_drop)
{
  LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;

  struct iv_list_head *item = input_queue->items.next;
  for (gint dropped = 0; dropped < num_of_messages_to_drop;)
    {
      LogMessageQueueNode *node = iv_list_entry(item, LogMessageQueueNode, list);

      item = item->next;

      path_options.ack_needed = node->ack_needed;
      path_options.flow_control_requested = node->flow_control_requested;

      if (!self->use_legacy_fifo_size && path_options.flow_control_requested)
        continue;

      iv_list_del(&node->list);
      input_queue->len--;
      log_queue_dropped_messages_inc(&self->super);
      log_msg_free_queue_node(node);

      LogMessage *msg = node->msg;
      if (path_options.flow_control_requested)
        log_msg_drop(msg, &path_options, AT_SUSPENDED);
      else
        {
          input_queue->non_flow_controlled_len--;
          log_msg_drop(msg, &path_options, AT_PROCESSED);
        }

      dropped++;
    }

  msg_debug("Destination queue full, dropping messages",
            evt_tag_int("log_fifo_size", self->log_fifo_size),
            evt_tag_int("number_of_dropped_messages", num_of_messages_to_drop),
            evt_tag_str("persist_name", self->super.persist_name));
}

static inline gboolean
log_queue_fifo_calculate_num_of_messages_to_drop(LogQueueFifo *self, InputQueue *input_queue,
                                                 gint *num_of_messages_to_drop)
{
  /* since we're in the input thread, queue_len will be racy.  It can
   * increase due to log_queue_fifo_rewind_backlog() and can also decrease
   * as items are removed from the output queue using log_queue_pop_head().
   *
   * The only reason we're using it here is to check for qoverflow
   * overflows, however the only side-effect of the race (if lost) is that
   * we would lose a couple of message too many or add some more messages to
   * qoverflow than permitted by the user.  Since if flow-control is used,
   * the fifo size should be sized larger than the potential window sizes,
   * otherwise we can lose messages anyway, this is not deemed a cost to
   * justify proper locking in this case.
   */

  guint16 input_queue_len;
  gint queue_len;

  if (G_UNLIKELY(self->use_legacy_fifo_size))
    {
      queue_len = log_queue_fifo_get_length(&self->super);
      input_queue_len = input_queue->len;
    }
  else
    {
      queue_len = log_queue_fifo_get_non_flow_controlled_length(self);
      input_queue_len = input_queue->non_flow_controlled_len;
    }

  gboolean drop_messages = queue_len + input_queue_len > self->log_fifo_size;
  if (!drop_messages)
    return FALSE;

  /* NOTE: MAX is needed here to ensure that the lost race on queue_len
   * doesn't result in n < 0 */
  *num_of_messages_to_drop = input_queue_len - MAX(0, (self->log_fifo_size - queue_len));

  return TRUE;
}

/* move items from the per-thread input queue to the lock-protected "wait" queue */
static void
log_queue_fifo_move_input_unlocked(LogQueueFifo *self, gint thread_index)
{
  gint num_of_messages_to_drop;
  gboolean drop_messages = log_queue_fifo_calculate_num_of_messages_to_drop(self, &self->input_queues[thread_index],
                           &num_of_messages_to_drop);

  if (drop_messages)
    {
      /* slow path, the input thread's queue would overflow the queue, let's drop some messages */
      log_queue_fifo_drop_messages_from_input_queue(self, &self->input_queues[thread_index], num_of_messages_to_drop);
    }

  log_queue_queued_messages_add(&self->super, self->input_queues[thread_index].len);
  iv_list_update_msg_size(self, &self->input_queues[thread_index].items);

  iv_list_splice_tail_init(&self->input_queues[thread_index].items, &self->wait_queue.items);
  self->wait_queue.len += self->input_queues[thread_index].len;
  self->wait_queue.non_flow_controlled_len += self->input_queues[thread_index].non_flow_controlled_len;
  self->input_queues[thread_index].len = 0;
  self->input_queues[thread_index].non_flow_controlled_len = 0;
}

/* move items from the per-thread input queue to the lock-protected
 * "wait" queue, but grabbing locks first. This is registered as a
 * callback to be called when the input worker thread finishes its
 * job.
 */
static gpointer
log_queue_fifo_move_input(gpointer user_data)
{
  LogQueueFifo *self = (LogQueueFifo *) user_data;
  gint thread_index;

  thread_index = main_loop_worker_get_thread_index();
  g_assert(thread_index >= 0);

  g_mutex_lock(&self->super.lock);
  log_queue_fifo_move_input_unlocked(self, thread_index);
  log_queue_push_notify(&self->super);
  g_mutex_unlock(&self->super.lock);
  self->input_queues[thread_index].finish_cb_registered = FALSE;
  log_queue_unref(&self->super);
  return NULL;
}

/* lock must be held */
static inline gboolean
_message_has_to_be_dropped(LogQueueFifo *self, const LogPathOptions *path_options)
{
  if (G_UNLIKELY(self->use_legacy_fifo_size))
    return log_queue_fifo_get_length(&self->super) >= self->log_fifo_size;

  return !path_options->flow_control_requested
         && log_queue_fifo_get_non_flow_controlled_length(self) >= self->log_fifo_size;
}

static inline void
_drop_message(LogMessage *msg, const LogPathOptions *path_options)
{
  if (path_options->flow_control_requested)
    {
      log_msg_drop(msg, path_options, AT_SUSPENDED);
      return;
    }

  log_msg_drop(msg, path_options, AT_PROCESSED);
}

/**
 * Assumed to be called from one of the input threads. If the thread_index
 * cannot be determined, the item is put directly in the wait queue.
 *
 * Puts the message to the queue, and logs an error if it caused the
 * queue to be full.
 *
 * It attempts to put the item to the per-thread input queue.
 *
 * NOTE: It consumes the reference passed by the caller.
 **/
static void
log_queue_fifo_push_tail(LogQueue *s, LogMessage *msg, const LogPathOptions *path_options)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  gint thread_index;
  LogMessageQueueNode *node;

  thread_index = main_loop_worker_get_thread_index();

  /* if this thread has an ID than the number of input queues we have (due
   * to a config change), handle the load via the slow path */

  if (thread_index >= self->num_input_queues)
    thread_index = -1;

  /* NOTE: we don't use high-water marks for now, as log_fetch_limit
   * limits the number of items placed on the per-thread input queue
   * anyway, and any sane number decreased the performance measurably.
   *
   * This means that per-thread input queues contain _all_ items that
   * a single poll iteration produces. And once the reader is finished
   * (either because the input is depleted or because of
   * log_fetch_limit / window_size) the whole bunch is propagated to
   * the "wait" queue.
   */

  if (thread_index >= 0)
    {
      /* fastpath, use per-thread input FIFOs */
      if (!self->input_queues[thread_index].finish_cb_registered)
        {
          /* this is the first item in the input FIFO, register a finish
           * callback to make sure it gets moved to the wait_queue if the
           * input thread finishes
           * One reference should be held, while the callback is registered
           * avoiding use-after-free situation
           */

          main_loop_worker_register_batch_callback(&self->input_queues[thread_index].cb);
          self->input_queues[thread_index].finish_cb_registered = TRUE;
          log_queue_ref(&self->super);
        }

      log_msg_write_protect(msg);
      node = log_msg_alloc_queue_node(msg, path_options);
      iv_list_add_tail(&node->list, &self->input_queues[thread_index].items);
      self->input_queues[thread_index].len++;

      if (!path_options->flow_control_requested)
        self->input_queues[thread_index].non_flow_controlled_len++;

      log_msg_unref(msg);
      return;
    }

  /* slow path, put the pending item and the whole input queue to the wait_queue */

  g_mutex_lock(&self->super.lock);

  if (_message_has_to_be_dropped(self, path_options))
    {
      log_queue_dropped_messages_inc(&self->super);
      g_mutex_unlock(&self->super.lock);

      _drop_message(msg, path_options);

      msg_debug("Destination queue full, dropping message",
                evt_tag_int("queue_len", log_queue_fifo_get_length(&self->super)),
                evt_tag_int("log_fifo_size", self->log_fifo_size),
                evt_tag_str("persist_name", self->super.persist_name));
      return;
    }

  log_msg_write_protect(msg);
  node = log_msg_alloc_queue_node(msg, path_options);

  iv_list_add_tail(&node->list, &self->wait_queue.items);
  self->wait_queue.len++;

  if (!path_options->flow_control_requested)
    self->wait_queue.non_flow_controlled_len++;

  log_queue_push_notify(&self->super);
  log_queue_queued_messages_inc(&self->super);

  log_queue_memory_usage_add(&self->super, log_msg_get_size(msg));
  g_mutex_unlock(&self->super.lock);

  log_msg_unref(msg);
}

/*
 * Can only run from the output thread.
 */
static inline void
_move_items_from_wait_queue_to_output_queue(LogQueueFifo *self)
{
  /* slow path, output queue is empty, get some elements from the wait queue */
  g_mutex_lock(&self->super.lock);
  iv_list_splice_tail_init(&self->wait_queue.items, &self->output_queue.items);
  self->output_queue.len = self->wait_queue.len;
  self->output_queue.non_flow_controlled_len = self->wait_queue.non_flow_controlled_len;
  self->wait_queue.len = 0;
  self->wait_queue.non_flow_controlled_len = 0;
  g_mutex_unlock(&self->super.lock);
}

/*
 * Can only run from the output thread.
 */
static LogMessage *
log_queue_fifo_peek_head(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  LogMessageQueueNode *node;
  LogMessage *msg = NULL;

  if (self->output_queue.len == 0)
    _move_items_from_wait_queue_to_output_queue(self);

  if (self->output_queue.len == 0)
    return NULL;

  node = iv_list_entry(self->output_queue.items.next, LogMessageQueueNode, list);
  msg = node->msg;

  return msg;
}

/*
 * Can only run from the output thread.
 *
 * NOTE: this returns a reference which the caller must take care to free.
 */
static LogMessage *
log_queue_fifo_pop_head(LogQueue *s, LogPathOptions *path_options)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  LogMessageQueueNode *node;
  LogMessage *msg = NULL;

  if (self->output_queue.len == 0)
    _move_items_from_wait_queue_to_output_queue(self);

  if (self->output_queue.len > 0)
    {
      node = iv_list_entry(self->output_queue.items.next, LogMessageQueueNode, list);

      msg = node->msg;
      path_options->ack_needed = node->ack_needed;
      self->output_queue.len--;

      if (!node->flow_control_requested)
        self->output_queue.non_flow_controlled_len--;

      iv_list_del_init(&node->list);
    }
  else
    {
      /* no items either on the wait queue nor the output queue.
       *
       * NOTE: the input queues may contain items even in this case,
       * however we don't touch them here, they'll be migrated to the
       * wait_queue once the input threads finish their processing (or
       * the high watermark is reached). Also, they are unlocked, so
       * no way to touch them safely.
       */
      return NULL;
    }
  log_queue_queued_messages_dec(&self->super);
  log_queue_memory_usage_sub(&self->super, log_msg_get_size(msg));

  /* push to backlog */
  log_msg_ref(msg);
  iv_list_add_tail(&node->list, &self->backlog_queue.items);
  self->backlog_queue.len++;

  if (!node->flow_control_requested)
    self->backlog_queue.non_flow_controlled_len++;

  return msg;
}

/*
 * Can only run from the output thread.
 */
static void
log_queue_fifo_ack_backlog(LogQueue *s, gint rewind_count)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  LogMessage *msg;
  LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;
  gint pos;

  for (pos = 0; pos < rewind_count && self->backlog_queue.len > 0; pos++)
    {
      LogMessageQueueNode *node;
      node = iv_list_entry(self->backlog_queue.items.next, LogMessageQueueNode, list);
      msg = node->msg;

      iv_list_del(&node->list);
      self->backlog_queue.len--;

      if (!node->flow_control_requested)
        self->backlog_queue.non_flow_controlled_len--;

      path_options.ack_needed = node->ack_needed;
      log_msg_ack(msg, &path_options, AT_PROCESSED);
      log_msg_free_queue_node(node);
      log_msg_unref(msg);
    }
}


/*
 * log_queue_rewind_backlog_all:
 *
 * Move items on our backlog back to our qoverflow queue. Please note that this
 * function does not really care about qoverflow size, it has to put the backlog
 * somewhere. The backlog is emptied as that will be filled if we send the
 * items again.
 *
 * NOTE: this is assumed to be called from the output thread.
 */
static void
log_queue_fifo_rewind_backlog_all(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;

  iv_list_update_msg_size(self, &self->backlog_queue.items);
  iv_list_splice_tail_init(&self->backlog_queue.items, &self->output_queue.items);

  self->output_queue.len += self->backlog_queue.len;
  self->output_queue.non_flow_controlled_len += self->backlog_queue.non_flow_controlled_len;
  log_queue_queued_messages_add(&self->super, self->backlog_queue.len);
  self->backlog_queue.len = 0;
  self->backlog_queue.non_flow_controlled_len = 0;
}

static void
log_queue_fifo_rewind_backlog(LogQueue *s, guint rewind_count)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  guint pos;

  if (rewind_count > self->backlog_queue.len)
    rewind_count = self->backlog_queue.len;

  for (pos = 0; pos < rewind_count; pos++)
    {
      LogMessageQueueNode *node = iv_list_entry(self->backlog_queue.items.prev, LogMessageQueueNode, list);
      /*
       * Because the message go to the backlog only in case of pop_head
       * and pop_head add ack and ref when it pushes the message into the backlog
       * The rewind must decrease the ack and ref too
       */
      iv_list_del_init(&node->list);
      iv_list_add(&node->list, &self->output_queue.items);

      self->backlog_queue.len--;
      self->output_queue.len++;

      if (!node->flow_control_requested)
        {
          self->backlog_queue.non_flow_controlled_len--;
          self->output_queue.non_flow_controlled_len++;
        }

      log_queue_queued_messages_inc(&self->super);
      log_queue_memory_usage_add(&self->super, log_msg_get_size(node->msg));
    }
}

static void
log_queue_fifo_free_queue(struct iv_list_head *q)
{
  while (!iv_list_empty(q))
    {
      LogMessageQueueNode *node;
      LogPathOptions path_options = LOG_PATH_OPTIONS_INIT;
      LogMessage *msg;

      node = iv_list_entry(q->next, LogMessageQueueNode, list);
      iv_list_del(&node->list);

      path_options.ack_needed = node->ack_needed;
      msg = node->msg;
      log_msg_free_queue_node(node);
      log_msg_ack(msg, &path_options, AT_ABORTED);
      log_msg_unref(msg);
    }
}

static void
log_queue_fifo_free(LogQueue *s)
{
  LogQueueFifo *self = (LogQueueFifo *) s;
  gint i;

  for (i = 0; i < self->num_input_queues; i++)
    {
      g_assert(self->input_queues[i].finish_cb_registered == FALSE);
      log_queue_fifo_free_queue(&self->input_queues[i].items);
    }

  log_queue_fifo_free_queue(&self->wait_queue.items);
  log_queue_fifo_free_queue(&self->output_queue.items);
  log_queue_fifo_free_queue(&self->backlog_queue.items);
  log_queue_free_method(s);
}

LogQueue *
log_queue_fifo_new(gint log_fifo_size, const gchar *persist_name, gint stats_level,
                   StatsClusterKeyBuilder *driver_sck_builder, StatsClusterKeyBuilder *queue_sck_builder)
{
  LogQueueFifo *self;

  gint max_threads = main_loop_worker_get_max_number_of_threads();
  self = g_malloc0(sizeof(LogQueueFifo) + max_threads * sizeof(self->input_queues[0]));

  if (queue_sck_builder)
    {
      stats_cluster_key_builder_push(queue_sck_builder);
      stats_cluster_key_builder_set_name_prefix(queue_sck_builder, "memory_queue_");
    }

  log_queue_init_instance(&self->super, persist_name, stats_level, driver_sck_builder, queue_sck_builder);
  self->super.type = log_queue_fifo_type;
  self->super.get_length = log_queue_fifo_get_length;
  self->super.is_empty_racy = log_queue_fifo_is_empty_racy;
  self->super.keep_on_reload = log_queue_fifo_keep_on_reload;
  self->super.push_tail = log_queue_fifo_push_tail;
  self->super.pop_head = log_queue_fifo_pop_head;
  self->super.peek_head = log_queue_fifo_peek_head;
  self->super.ack_backlog = log_queue_fifo_ack_backlog;
  self->super.rewind_backlog = log_queue_fifo_rewind_backlog;
  self->super.rewind_backlog_all = log_queue_fifo_rewind_backlog_all;

  self->super.free_fn = log_queue_fifo_free;

  self->num_input_queues = max_threads;
  for (gint i = 0; i < self->num_input_queues; i++)
    {
      INIT_IV_LIST_HEAD(&self->input_queues[i].items);
      worker_batch_callback_init(&self->input_queues[i].cb);
      self->input_queues[i].cb.func = log_queue_fifo_move_input;
      self->input_queues[i].cb.user_data = self;
    }
  INIT_IV_LIST_HEAD(&self->wait_queue.items);
  INIT_IV_LIST_HEAD(&self->output_queue.items);
  INIT_IV_LIST_HEAD(&self->backlog_queue.items);

  self->log_fifo_size = log_fifo_size;

  if (queue_sck_builder)
    stats_cluster_key_builder_pop(queue_sck_builder);

  return &self->super;
}

LogQueue *
log_queue_fifo_legacy_new(gint log_fifo_size, const gchar *persist_name, gint stats_level,
                          StatsClusterKeyBuilder *driver_sck_builder, StatsClusterKeyBuilder *queue_sck_builder)
{
  LogQueueFifo *self = (LogQueueFifo *) log_queue_fifo_new(log_fifo_size, persist_name, stats_level,
                                                           driver_sck_builder, queue_sck_builder);
  self->use_legacy_fifo_size = TRUE;
  return &self->super;
}

QueueType
log_queue_fifo_get_type(void)
{
  return log_queue_fifo_type;
}
