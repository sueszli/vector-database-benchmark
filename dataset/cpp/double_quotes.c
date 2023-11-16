#include <stic.h>

#include <stdlib.h> /* free() */
#include <string.h> /* memset() */

#include "../../src/engine/parsing.h"
#include "../../src/engine/var.h"

#include "asserts.h"

TEST(empty_ok)
{
	ASSERT_OK("\"\"", "");
}

TEST(simple_ok)
{
	ASSERT_OK("\"test\"", "test");
}

TEST(not_closed_error)
{
	ASSERT_FAIL("\"test", PE_MISSING_QUOTE);
}

TEST(concatenation)
{
	ASSERT_OK("\"NV\".\"AR\"", "NVAR");
	ASSERT_OK("\"NV\" .\"AR\"", "NVAR");
	ASSERT_OK("\"NV\". \"AR\"", "NVAR");
	ASSERT_OK("\"NV\" . \"AR\"", "NVAR");
}

TEST(double_quote_escaping_ok)
{
	ASSERT_OK("\"\\\"\"", "\"");
}

TEST(special_chars_ok)
{
	ASSERT_OK("\"\\t\"", "\t");
}

TEST(spaces_ok)
{
	ASSERT_OK("\" s y \"", " s y ");
}

TEST(dot_ok)
{
	ASSERT_OK("\"a . c\"", "a . c");
}

TEST(very_long_string)
{
	char string[8192];
	string[0] = '\"';
	memset(string + 1, '0', sizeof(string) - 2U);
	string[sizeof(string) - 1U] = '\0';

	ASSERT_FAIL(string, PE_INTERNAL);
}

/* vim: set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab cinoptions-=(0 : */
/* vim: set cinoptions+=t0 filetype=c : */
