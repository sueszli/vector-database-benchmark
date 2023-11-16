/*
 * Copyright (c) 2018 Balabit
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

#include <criterion/criterion.h>
#include "libtest/grab-logging.h"
#include "libtest/mock-cfg-parser.h"

#include "apphook.h"
#include "cfg-grammar.h"

#define TESTDATA_DIR TOP_SRCDIR "/modules/confgen/tests"

CfgParserMock *parser = NULL;

static void
_input(const gchar *input)
{
  cfg_parser_mock_input(parser, input);
}

static void
_next_token(void)
{
  cfg_parser_mock_next_token(parser);
}

static CFG_STYPE *
_current_token(void)
{
  return parser->yylval;
}

#define assert_token_type(expected)                                     \
  cr_assert_eq(_current_token()->type, expected, "Unexpected token type %d != %d", _current_token()->type, expected);

#define assert_parser_string(expected)                          \
  _next_token();                                                        \
  assert_token_type(LL_STRING);                                        \
  cr_assert_str_eq(_current_token()->cptr, expected, "Unexpected string value parsed >>>%s<<< != >>>%s<<<", _current_token()->cptr, expected);

#define assert_parser_character_token(expected)                          \
  _next_token();                                                        \
  assert_token_type((gint) expected);

#define assert_parser_identifier(expected) \
  _next_token();                                                        \
  assert_token_type(LL_IDENTIFIER);                                         \
  cr_assert_str_eq(_current_token()->cptr, expected, "Unexpected identifier parsed >>>%s<<< != >>>%s<<<", _current_token()->cptr, expected);


Test(confgen, confgen_script_output_is_included_into_the_config)
{
  parser->lexer->ignore_pragma = FALSE;

  cfg_lexer_push_context(parser->lexer, main_parser.context, main_parser.keywords, main_parser.name);
  _input(
    "@module confgen context(root) name(confgentest) exec('"TESTDATA_DIR "/confgentest.sh')\n"
    "from-config1\n"
    "confgentest()\n"
    "from-config2\n");

  assert_parser_identifier("from-config1");
  assert_parser_identifier("from-confgen1");
  assert_parser_identifier("from-confgen2");
  assert_parser_identifier("from-config2");
  cfg_lexer_pop_context(parser->lexer);
}

Test(confgen, confgen_unknown_context_is_reported_as_an_error)
{
  parser->lexer->ignore_pragma = FALSE;

  start_grabbing_messages();
  cfg_lexer_push_context(parser->lexer, main_parser.context, main_parser.keywords, main_parser.name);
  _input(
    "@module confgen context(unknown-context) name(confgentest) exec('"TESTDATA_DIR "/confgentest.sh')\n"
    "from-config1\n"
    "confgentest()\n"
    "from-config2\n");

  _next_token();
  assert_parser_identifier("from-config1");
  assert_grabbed_log_contains("context value is unknown");
  /* confgen is not registered */
  assert_parser_identifier("confgentest");
  assert_parser_character_token('(');
  assert_parser_character_token(')');
  assert_parser_identifier("from-config2");
  cfg_lexer_pop_context(parser->lexer);
  stop_grabbing_messages();
}

static void
setup(void)
{
  app_startup();
  configuration = cfg_new_snippet();
  parser = cfg_parser_mock_new();
}

static void
teardown(void)
{
  cfg_parser_mock_free(parser);
  cfg_free(configuration);
  configuration = NULL;
  app_shutdown();
}

TestSuite(confgen, .init = setup, .fini = teardown);
