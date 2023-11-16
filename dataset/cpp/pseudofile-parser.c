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

#include "pseudofile.h"
#include "cfg-parser.h"
#include "pseudofile-grammar.h"

extern int pseudofile_debug;

int pseudofile_parse(CfgLexer *lexer, LogDriver **instance, gpointer arg);

static CfgLexerKeyword pseudofile_keywords[] =
{
  { "pseudofile",               KW_PSEUDOFILE },

  { NULL }
};

CfgParser pseudofile_parser =
{
#if SYSLOG_NG_ENABLE_DEBUG
  .debug_flag = &pseudofile_debug,
#endif
  .name = "pseudofile",
  .keywords = pseudofile_keywords,
  .parse = (gint (*)(CfgLexer *, gpointer *, gpointer)) pseudofile_parse,
  .cleanup = (void (*)(gpointer)) log_pipe_unref,
};

CFG_PARSER_IMPLEMENT_LEXER_BINDING(pseudofile_, PSEUDOFILE_, LogDriver **)
