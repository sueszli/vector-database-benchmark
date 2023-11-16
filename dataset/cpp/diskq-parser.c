/*
 * Copyright (c) 2002-2016 Balabit
 * Copyright (c) 2016 Viktor Juhasz <viktor.juhasz@balabit.com>
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

#include "diskq.h"
#include "cfg-parser.h"
#include "diskq-grammar.h"

extern int diskq_debug;
int diskq_parse(CfgLexer *lexer, LogDriverPlugin **instance, gpointer arg);

static CfgLexerKeyword diskq_keywords[] =
{
  { "disk_buffer",       KW_DISK_BUFFER },
  { "mem_buf_length",           KW_FLOW_CONTROL_WINDOW_SIZE },
  { "flow_control_window_size", KW_FLOW_CONTROL_WINDOW_SIZE },
  { "disk_buf_size",     KW_CAPACITY_BYTES },
  { "capacity_bytes",    KW_CAPACITY_BYTES },
  { "reliable",          KW_RELIABLE },
  { "compaction",        KW_COMPACTION },
  { "mem_buf_size",              KW_FLOW_CONTROL_WINDOW_BYTES },
  { "flow_control_window_bytes", KW_FLOW_CONTROL_WINDOW_BYTES },
  { "qout_size",         KW_FRONT_CACHE_SIZE },
  { "front_cache_size",  KW_FRONT_CACHE_SIZE },
  { "dir",               KW_DIR },
  { "truncate_size_ratio", KW_TRUNCATE_SIZE_RATIO },
  { "prealloc",          KW_PREALLOC },
  { "stats",             KW_STATS },
  { "freq",              KW_FREQ },
  { NULL }
};

CfgParser diskq_parser =
{
#if SYSLOG_NG_ENABLE_DEBUG
  .debug_flag = &diskq_debug,
#endif
  .name = "disk_buffer",
  .keywords = diskq_keywords,
  .parse = (int (*)(CfgLexer *lexer, gpointer *instance, gpointer arg)) diskq_parse,
  .cleanup = (void (*)(gpointer)) log_driver_plugin_free,

};

CFG_PARSER_IMPLEMENT_LEXER_BINDING(diskq_, DISKQ_, LogDriverPlugin **)
