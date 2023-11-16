/*
 *   Copyright (C) 2007-2019 Tristan Heaven <tristan@tristanheaven.net>
 *
 *   This file is part of GtkHash.
 *
 *   GtkHash is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   GtkHash is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with GtkHash. If not, see <https://gnu.org/licenses/gpl-2.0.txt>.
 */

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <glib.h>

#include "hash-file.h"
#include "hash-func.h"
#include "hash-lib.h"
#include "digest-format.h"

#ifndef G_SOURCE_FUNC
#define G_SOURCE_FUNC(f) ((GSourceFunc) (void (*)(void)) (f))
#endif

// This lib can use GDK 2 or 3, but doesn't link either directly.
// Try to avoid potential ABI/API mismatch issues by only declaring
// necessary gdk.h functions...
guint gdk_threads_add_idle(GSourceFunc, gpointer);
guint gdk_threads_add_timeout(guint, GSourceFunc, gpointer);
// (this might cause other problems)

static gboolean gtkhash_hash_file_source_func(struct hash_file_s *data);
static void gtkhash_hash_file_hash_thread_func(struct hash_func_s *func,
	struct hash_file_s *data);

enum hash_file_state_e {
	HASH_FILE_STATE_IDLE,
	HASH_FILE_STATE_START,
	HASH_FILE_STATE_OPEN,
	HASH_FILE_STATE_GET_SIZE,
	HASH_FILE_STATE_READ,
	HASH_FILE_STATE_HASH,
	HASH_FILE_STATE_HASH_FINISH,
	HASH_FILE_STATE_CLOSE,
	HASH_FILE_STATE_FINISH,
	HASH_FILE_STATE_CALLBACK,
};

struct hash_file_s {
	goffset file_size, total_read;
	const void *cb_data;
	const char *uri;
	GFile *file;
	const uint8_t *hmac_key;
	size_t key_size;
	GCancellable *cancellable;
	GFileInputStream *stream;
	gssize just_read;
	uint8_t *buffer;
	GTimer *timer;
	GThreadPool *thread_pool;
	struct hash_func_s *funcs;
	enum hash_file_state_e state;
	enum digest_format_e format;
	int threads;
	unsigned int report_source;
	unsigned int source;
	GMutex mtx;
};

static void gtkhash_hash_file_add_source(struct hash_file_s *data)
{
	g_mutex_lock(&data->mtx);
	g_assert(!data->source);
	data->source = g_idle_add(G_SOURCE_FUNC(gtkhash_hash_file_source_func),
		data);
	g_mutex_unlock(&data->mtx);
}

static void gtkhash_hash_file_remove_source(struct hash_file_s *data)
{
	g_mutex_lock(&data->mtx);
	if (G_UNLIKELY(!g_source_remove(data->source)))
		g_assert_not_reached();
	data->source = 0;
	g_mutex_unlock(&data->mtx);
}

void gtkhash_hash_file_cancel(struct hash_file_s *data)
{
	g_cancellable_cancel(data->cancellable);
}

static gboolean gtkhash_hash_file_report_source_func(struct hash_file_s *data)
{
	if (data->report_source && data->total_read)
		gtkhash_hash_file_report_cb((void *)data->cb_data, data->file_size,
			data->total_read, data->timer);

	return true;
}

static void gtkhash_hash_file_add_report_source(struct hash_file_s *data)
{
	g_assert(!data->report_source);
	data->report_source = gdk_threads_add_timeout(HASH_FILE_REPORT_INTERVAL,
		G_SOURCE_FUNC(gtkhash_hash_file_report_source_func), data);
}

static void gtkhash_hash_file_remove_report_source(struct hash_file_s *data)
{
	if (data->report_source) {
		if (G_UNLIKELY(!g_source_remove(data->report_source)))
			g_assert_not_reached();
		data->report_source = 0;
	}
}

static void gtkhash_hash_file_start(struct hash_file_s *data)
{
	g_assert(data->uri);

	int funcs_enabled = 0;

	for (int i = 0; i < HASH_FUNCS_N; i++) {
		if (data->funcs[i].enabled) {
			gtkhash_hash_lib_start(&data->funcs[i], data->hmac_key,
				data->key_size);
			funcs_enabled++;
		}
	}

	g_assert(funcs_enabled > 0);

	// Determine max number of threads to use
	const int cpus = g_get_num_processors();
	const int threads = CLAMP(MIN(funcs_enabled, cpus), 1, HASH_FUNCS_N);

	g_atomic_int_set(&data->threads, 0);
	data->thread_pool = g_thread_pool_new(
		(GFunc)gtkhash_hash_file_hash_thread_func, data, threads, true, NULL);

	data->file = g_file_new_for_uri(data->uri);
	data->just_read = 0;
	data->buffer = g_malloc(HASH_FILE_BUFFER_SIZE);
	data->timer = g_timer_new();
	data->total_read = 0;

	data->state = HASH_FILE_STATE_OPEN;
}

static void gtkhash_hash_file_open_finish(G_GNUC_UNUSED GObject *source,
	GAsyncResult *res, struct hash_file_s *data)
{
	data->stream = g_file_read_finish(data->file, res, NULL);
	if (G_UNLIKELY(!data->stream &&
		!g_cancellable_is_cancelled(data->cancellable)))
	{
		g_warning("failed to open file (%s)", data->uri);
		g_cancellable_cancel(data->cancellable);
	}

	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		if (data->stream)
			data->state = HASH_FILE_STATE_CLOSE;
		else
			data->state = HASH_FILE_STATE_FINISH;
	} else
		data->state = HASH_FILE_STATE_GET_SIZE;

	gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_open(struct hash_file_s *data)
{
	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		data->state = HASH_FILE_STATE_FINISH;
		return;
	}

	gtkhash_hash_file_remove_source(data);
	g_file_read_async(data->file, G_PRIORITY_DEFAULT, data->cancellable,
		(GAsyncReadyCallback)gtkhash_hash_file_open_finish, data);
}

static void gtkhash_hash_file_get_size_finish(G_GNUC_UNUSED GObject *source,
	GAsyncResult *res, struct hash_file_s *data)
{
	GError *error = NULL;
	GFileInfo *info = g_file_input_stream_query_info_finish(
		data->stream, res, &error);

	if (G_UNLIKELY(!info)) {
		g_warning("query info: %s", error->message);
		g_error_free(error);

		data->state = HASH_FILE_STATE_CLOSE;
		g_cancellable_cancel(data->cancellable);
	} else {
		data->file_size = g_file_info_get_size(info);
		g_object_unref(info);

		if (data->file_size == 0)
			data->state = HASH_FILE_STATE_HASH;
		else {
			data->state = HASH_FILE_STATE_READ;
			gtkhash_hash_file_add_report_source(data);
		}
	}

	gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_get_size(struct hash_file_s *data)
{
	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		data->state = HASH_FILE_STATE_CLOSE;
		return;
	}

	gtkhash_hash_file_remove_source(data);
	g_file_input_stream_query_info_async(data->stream,
		G_FILE_ATTRIBUTE_STANDARD_SIZE, G_PRIORITY_DEFAULT, data->cancellable,
		(GAsyncReadyCallback)gtkhash_hash_file_get_size_finish, data);
}

static void gtkhash_hash_file_read_finish(G_GNUC_UNUSED GObject *source,
	GAsyncResult *res, struct hash_file_s *data)
{
	data->just_read = g_input_stream_read_finish(
		G_INPUT_STREAM(data->stream), res, NULL);

	if (G_UNLIKELY(data->just_read == -1) &&
		!g_cancellable_is_cancelled(data->cancellable))
	{
		g_warning("failed to read file (%s)", data->uri);
		g_cancellable_cancel(data->cancellable);
	} else if (G_UNLIKELY(data->just_read == 0)) {
		g_warning("unexpected EOF (%s)", data->uri);
		g_cancellable_cancel(data->cancellable);
	} else {
		data->total_read += data->just_read;
		if (G_UNLIKELY(data->total_read > data->file_size)) {
			g_warning("read %" G_GOFFSET_FORMAT
				" more bytes than expected (%s)", data->total_read -
				data->file_size, data->uri);
			g_cancellable_cancel(data->cancellable);
		} else
			data->state = HASH_FILE_STATE_HASH;
	}

	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable)))
		data->state = HASH_FILE_STATE_CLOSE;

	gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_read(struct hash_file_s *data)
{
	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		data->state = HASH_FILE_STATE_CLOSE;
		return;
	}

	gtkhash_hash_file_remove_source(data);
	g_input_stream_read_async(G_INPUT_STREAM(data->stream),
		data->buffer, HASH_FILE_BUFFER_SIZE, G_PRIORITY_DEFAULT,
		data->cancellable, (GAsyncReadyCallback)gtkhash_hash_file_read_finish,
		data);
}

static void gtkhash_hash_file_hash_thread_func(struct hash_func_s *func,
	struct hash_file_s *data)
{
	gtkhash_hash_lib_update(func, data->buffer, data->just_read);

	if (g_atomic_int_dec_and_test(&data->threads))
		gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_hash(struct hash_file_s *data)
{
	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		data->state = HASH_FILE_STATE_CLOSE;
		return;
	}

	gtkhash_hash_file_remove_source(data);
	data->state = HASH_FILE_STATE_HASH_FINISH;

	g_atomic_int_inc(&data->threads);
	for (int i = 0; i < HASH_FUNCS_N; i++) {
		if (data->funcs[i].enabled) {
			g_atomic_int_inc(&data->threads);
			g_thread_pool_push(data->thread_pool, &data->funcs[i], NULL);
		}
	}

	if (g_atomic_int_dec_and_test(&data->threads))
		gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_hash_finish(struct hash_file_s *data)
{
	g_assert(g_atomic_int_get(&data->threads) == 0);

	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		data->state = HASH_FILE_STATE_CLOSE;
		return;
	}

	if (data->total_read >= data->file_size)
		data->state = HASH_FILE_STATE_CLOSE;
	else
		data->state = HASH_FILE_STATE_READ;
}

static void gtkhash_hash_file_close_finish(G_GNUC_UNUSED GObject *source,
	GAsyncResult *res, struct hash_file_s *data)
{
	if (G_UNLIKELY(!g_input_stream_close_finish(G_INPUT_STREAM(data->stream), res, NULL) &&
		!g_cancellable_is_cancelled(data->cancellable)))
	{
		g_warning("failed to close file (%s)", data->uri);
	}

	g_object_unref(data->stream);

	gtkhash_hash_file_remove_report_source(data);
	data->state = HASH_FILE_STATE_FINISH;
	gtkhash_hash_file_add_source(data);
}

static void gtkhash_hash_file_close(struct hash_file_s *data)
{
	gtkhash_hash_file_remove_source(data);
	g_input_stream_close_async(G_INPUT_STREAM(data->stream),
		G_PRIORITY_DEFAULT, data->cancellable,
		(GAsyncReadyCallback)gtkhash_hash_file_close_finish, data);
}

static void gtkhash_hash_file_finish(struct hash_file_s *data)
{
	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		for (int i = 0; i < HASH_FUNCS_N; i++)
			if (data->funcs[i].enabled)
				gtkhash_hash_lib_stop(&data->funcs[i]);
	} else {
		for (int i = 0; i < HASH_FUNCS_N; i++)
			if (data->funcs[i].enabled)
				gtkhash_hash_lib_finish(&data->funcs[i]);
	}

	g_object_unref(data->file);
	data->file = NULL;

	g_free(data->buffer);
	data->buffer = NULL;

	g_timer_destroy(data->timer);
	data->timer = NULL;

	g_thread_pool_free(data->thread_pool, true, false);
	data->thread_pool = NULL;

	data->state = HASH_FILE_STATE_CALLBACK;
}

static gboolean gtkhash_hash_file_callback_stop_func(void *cb_data)
{
	gtkhash_hash_file_stop_cb(cb_data);

	return false;
}

static gboolean gtkhash_hash_file_callback_finish_func(
	struct hash_file_s *data)
{
	for (int i = 0; i < HASH_FUNCS_N; i++) {
		if (!data->funcs[i].enabled)
			continue;

		char *digest = gtkhash_hash_func_get_digest(&data->funcs[i],
			data->format);

		gtkhash_hash_file_digest_cb(i, digest, (void *)data->cb_data);

		g_free(digest);

		gtkhash_hash_func_clear_digest(&data->funcs[i]);
	}

	gtkhash_hash_file_finish_cb((void *)data->cb_data);

	return false;
}

static void gtkhash_hash_file_callback(struct hash_file_s *data)
{
	gtkhash_hash_file_remove_source(data);
	data->state = HASH_FILE_STATE_IDLE;

	if (G_UNLIKELY(g_cancellable_is_cancelled(data->cancellable))) {
		gdk_threads_add_idle(gtkhash_hash_file_callback_stop_func,
			(void *)data->cb_data);
	} else {
		gdk_threads_add_idle(
			G_SOURCE_FUNC(gtkhash_hash_file_callback_finish_func), data);
	}

	g_object_unref(data->cancellable);
	data->cancellable = NULL;
}

static gboolean gtkhash_hash_file_source_func(struct hash_file_s *data)
{
	static void (* const state_funcs[])(struct hash_file_s *) = {
		[HASH_FILE_STATE_IDLE]        = NULL,
		[HASH_FILE_STATE_START]       = gtkhash_hash_file_start,
		[HASH_FILE_STATE_OPEN]        = gtkhash_hash_file_open,
		[HASH_FILE_STATE_GET_SIZE]    = gtkhash_hash_file_get_size,
		[HASH_FILE_STATE_READ]        = gtkhash_hash_file_read,
		[HASH_FILE_STATE_HASH]        = gtkhash_hash_file_hash,
		[HASH_FILE_STATE_HASH_FINISH] = gtkhash_hash_file_hash_finish,
		[HASH_FILE_STATE_CLOSE]       = gtkhash_hash_file_close,
		[HASH_FILE_STATE_FINISH]      = gtkhash_hash_file_finish,
		[HASH_FILE_STATE_CALLBACK]    = gtkhash_hash_file_callback,
	};

	state_funcs[data->state](data);

	return true;
}

void gtkhash_hash_file(struct hash_file_s *data, const char *uri,
	const enum digest_format_e format, const uint8_t *hmac_key,
	const size_t key_size, const void *cb_data)
{
	g_assert(data);
	g_assert(uri && *uri);
	g_assert(DIGEST_FORMAT_IS_VALID(format));
	g_assert(data->state == HASH_FILE_STATE_IDLE);
	g_assert(data->report_source == 0);
	g_assert(data->source == 0);
	g_assert(!data->cancellable);

	data->uri = uri;
	data->format = format;
	data->hmac_key = hmac_key;
	data->key_size = key_size;
	data->cb_data = cb_data;
	data->cancellable = g_cancellable_new();

	data->state = HASH_FILE_STATE_START;
	gtkhash_hash_file_add_source(data);
}

struct hash_file_s *gtkhash_hash_file_new(struct hash_func_s *funcs)
{
	struct hash_file_s *data = g_new(struct hash_file_s, 1);

	data->file_size = 0;
	data->total_read = 0;
	data->cb_data = NULL;
	data->uri = NULL;
	data->file = NULL;
	data->hmac_key = NULL;
	data->key_size = 0;
	data->cancellable = NULL;
	data->stream = NULL;
	data->just_read = 0;
	data->buffer = NULL;
	data->timer = NULL;
	data->thread_pool = NULL;
	data->funcs = funcs;
	data->state = HASH_FILE_STATE_IDLE;
	data->format = DIGEST_FORMAT_INVALID;
	g_atomic_int_set(&data->threads, 0);
	data->report_source = 0;
	data->source = 0;
	g_mutex_init(&data->mtx);

	return data;
}

void gtkhash_hash_file_free(struct hash_file_s *data)
{
	g_assert(data);

	// Shouldn't still be running
	g_assert(data->state == HASH_FILE_STATE_IDLE);
	g_assert(data->report_source == 0);
	g_assert(data->source == 0);

	g_mutex_clear(&data->mtx);

	g_free(data);
}
