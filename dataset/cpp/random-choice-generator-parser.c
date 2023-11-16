/*
 * Copyright (c) 2023 Attila Szakacs
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

#include "driver.h"
#include "cfg-parser.h"
#include "random-choice-generator-grammar.h"

extern int random_choice_generator_debug;

int random_choice_generator_parse(CfgLexer *lexer, LogDriver **instance, gpointer arg);

static CfgLexerKeyword random_choice_generator_keywords[] =
{
  { "random_choice_generator",         KW_RANDOM_CHOICE_GENERATOR },
  { "choices",                         KW_CHOICES },
  { "freq",                            KW_FREQ },
  { NULL }
};

CfgParser random_choice_generator_parser =
{
#if SYSLOG_NG_ENABLE_DEBUG
  .debug_flag = &random_choice_generator_debug,
#endif
  .name = "random_choice_generator",
  .keywords = random_choice_generator_keywords,
  .parse = (gint (*)(CfgLexer *, gpointer *, gpointer)) random_choice_generator_parse,
  .cleanup = (void (*)(gpointer)) log_pipe_unref,
};

CFG_PARSER_IMPLEMENT_LEXER_BINDING(random_choice_generator_, random_choice_generator_, LogDriver **)
