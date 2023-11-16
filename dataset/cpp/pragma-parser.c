/*
 * Copyright (c) 2002-2011 Balabit
 * Copyright (c) 1998-2011 Balázs Scheidler
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

#include "pragma-parser.h"
#include "pragma-grammar.h"

#include <stdlib.h>

extern int pragma_debug;
int pragma_parse(CfgLexer *lexer, gpointer *result, gpointer arg);

guint
process_version_string(gchar *value)
{
  gchar *p, *end;
  gint major, minor;

  if (strlen(value) > strlen("xxx.yyy"))
    return 0;

  if (value[0] == '+' || value[0] == '-')
    return 0;

  p = strchr(value, '.');
  if (p == value)
    return 0;
  if (p)
    {
      major = strtol(value, &end, 10);
      if (major < 0)
        return 0;
      if (end == p)
        {
          minor = strtol(p+1, &end, 10);
          if (minor < 0)
            return 0;
          if (*end == '\0')
            {
              return (major << 8) + minor;
            }
        }
    }
  return 0;
}

static CfgLexerKeyword pragma_keywords[] =
{
  { "version",            KW_VERSION, },
  { "current",            KW_VERSION_CURRENT },
  { "include",            KW_INCLUDE, },
  { "module",             KW_MODULE, },
  { "define",             KW_DEFINE, },
  { "requires",           KW_REQUIRES, },
  { "line",               KW_LINE },
  { "config_id",          KW_CONFIG_ID },
  { CFG_KEYWORD_STOP },
};

CfgParser pragma_parser =
{
#if SYSLOG_NG_ENABLE_DEBUG
  .debug_flag = &pragma_debug,
#endif
  .name = "pragma",
  .context = LL_CONTEXT_PRAGMA,
  .keywords = pragma_keywords,
  .parse = pragma_parse,
};

CFG_PARSER_IMPLEMENT_LEXER_BINDING(pragma_, PRAGMA_, gpointer *)
