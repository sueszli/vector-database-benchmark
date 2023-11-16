/*
 * SPDX-License-Identifier: ISC
 * SPDX-URL: https://spdx.org/licenses/ISC.html
 *
 * Copyright (C) 2010-2012 William Pitcock <nenolod@dereferenced.org>
 * Copyright (C) 2018 Atheme Development Group (https://atheme.github.io/)
 *
 * Platform-agnostic database backend layer.
 */

#include <atheme.h>
#include "internal.h"

static mowgli_patricia_t *db_types = NULL;

const struct database_module *db_mod = NULL;

struct database_handle *
db_open(const char *filename, enum database_transaction txn)
{
	return_val_if_fail(db_mod != NULL, NULL);
	return_val_if_fail(db_mod->db_open != NULL, NULL);

	return db_mod->db_open(filename, txn);
}

void
db_close(struct database_handle *db)
{
	return_if_fail(db_mod != NULL);
	return_if_fail(db_mod->db_close != NULL);

	return db_mod->db_close(db);
}

void
db_parse(struct database_handle *db)
{
	return_if_fail(db_mod != NULL);
	return_if_fail(db_mod->db_parse != NULL);

	return db_mod->db_parse(db);
}

bool
db_read_next_row(struct database_handle *db)
{
	return_val_if_fail(db != NULL, NULL);
	return_val_if_fail(db->vt != NULL, NULL);
	return_val_if_fail(db->vt->read_next_row != NULL, NULL);

	return db->vt->read_next_row(db);
}

const char *
db_read_word(struct database_handle *db)
{
	return_val_if_fail(db != NULL, NULL);
	return_val_if_fail(db->vt != NULL, NULL);
	return_val_if_fail(db->vt->read_word != NULL, NULL);

	return db->vt->read_word(db);
}

const char *
db_read_str(struct database_handle *db)
{
	return_val_if_fail(db != NULL, NULL);
	return_val_if_fail(db->vt != NULL, NULL);
	return_val_if_fail(db->vt->read_str != NULL, NULL);

	return db->vt->read_str(db);
}

bool
db_read_int(struct database_handle *db, int *r)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->read_int != NULL, false);

	return db->vt->read_int(db, r);
}

bool
db_read_uint(struct database_handle *db, unsigned int *r)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->read_uint != NULL, false);

	return db->vt->read_uint(db, r);
}

bool
db_read_time(struct database_handle *db, time_t *r)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->read_time != NULL, false);

	return db->vt->read_time(db, r);
}

const char *
db_sread_word(struct database_handle *db)
{
	const char *w = db_read_word(db);
	if (!w)
	{
		slog(LG_ERROR, "db-sread-word: needed word at file %s line %u token %u", db->file, db->line, db->token);
		slog(LG_ERROR, "db-sread-word: exiting to avoid data loss");
		exit(EXIT_FAILURE);
	}
	return w;
}

const char *
db_sread_str(struct database_handle *db)
{
	const char *w = db_read_str(db);
	if (!w)
	{
		slog(LG_ERROR, "db-sread-multiword: needed multiword at file %s line %u token %u", db->file, db->line, db->token);
		slog(LG_ERROR, "db-sread-multiword: exiting to avoid data loss");
		exit(EXIT_FAILURE);
	}
	return w;
}

int
db_sread_int(struct database_handle *db)
{
	int r;
	bool ok = db_read_int(db, &r);

	if (!ok)
	{
		slog(LG_ERROR, "db-read-int: needed int at file %s line %u token %u", db->file, db->line, db->token);
		slog(LG_ERROR, "db-read-int: exiting to avoid data loss");
		exit(EXIT_FAILURE);
	}
	return r;
}

unsigned int
db_sread_uint(struct database_handle *db)
{
	unsigned int r;
	bool ok = db_read_uint(db, &r);

	if (!ok)
	{
		slog(LG_ERROR, "db-read-uint: needed int at file %s line %u token %u", db->file, db->line, db->token);
		slog(LG_ERROR, "db-read-uint: exiting to avoid data loss");
		exit(EXIT_FAILURE);
	}
	return r;
}

time_t
db_sread_time(struct database_handle *db)
{
	time_t r;
	bool ok = db_read_time(db, &r);

	if (!ok)
	{
		slog(LG_ERROR, "db-read-time: needed int at file %s line %u token %u", db->file, db->line, db->token);
		slog(LG_ERROR, "db-read-time: exiting to avoid data loss");
		exit(EXIT_FAILURE);
	}
	return r;
}

bool
db_start_row(struct database_handle *db, const char *type)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->start_row != NULL, false);

	return db->vt->start_row(db, type);
}

bool
db_write_word(struct database_handle *db, const char *word)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->write_word != NULL, false);

	return db->vt->write_word(db, word);
}

bool
db_write_str(struct database_handle *db, const char *str)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->write_str != NULL, false);

	return db->vt->write_str(db, str);
}

bool
db_write_int(struct database_handle *db, int num)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->write_int != NULL, false);

	return db->vt->write_int(db, num);
}

bool
db_write_uint(struct database_handle *db, unsigned int num)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->write_uint != NULL, false);

	return db->vt->write_uint(db, num);
}

bool
db_write_time(struct database_handle *db, time_t tm)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->write_time != NULL, false);

	return db->vt->write_time(db, tm);
}

bool
db_commit_row(struct database_handle *db)
{
	return_val_if_fail(db != NULL, false);
	return_val_if_fail(db->vt != NULL, false);
	return_val_if_fail(db->vt->commit_row != NULL, false);

	return db->vt->commit_row(db);
}

void
db_register_type_handler(const char *type, database_handler_fn fun)
{
	return_if_fail(db_types != NULL);
	return_if_fail(type != NULL);
	return_if_fail(fun != NULL);

	mowgli_patricia_add(db_types, type, fun);
}

void
db_unregister_type_handler(const char *type)
{
	return_if_fail(db_types != NULL);
	return_if_fail(type != NULL);

	mowgli_patricia_delete(db_types, type);
}

void
db_process(struct database_handle *db, const char *type)
{
	database_handler_fn fun;

	return_if_fail(db_types != NULL);
	return_if_fail(db != NULL);
	return_if_fail(type != NULL);

	fun = mowgli_patricia_retrieve(db_types, type);

	if (!fun)
	{
		fun = mowgli_patricia_retrieve(db_types, "???");
	}

	fun(db, type);
}

bool ATHEME_FATTR_PRINTF(2, 3)
db_write_format(struct database_handle *db, const char *fmt, ...)
{
	va_list va;
	char buf[BUFSIZE];

	va_start(va, fmt);
	vsnprintf(buf, BUFSIZE, fmt, va);
	va_end(va);

	return db_write_word(db, buf);
}

void
db_init(void)
{
	db_types = mowgli_patricia_create(strcasecanon);

	if (db_types == NULL)
	{
		slog(LG_ERROR, "db_init(): object allocator failure");
		exit(EXIT_FAILURE);
	}
}
