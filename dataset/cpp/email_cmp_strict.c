/**
 * @file
 * Test code for email_cmp_strict()
 *
 * @authors
 * Copyright (C) 2019 Richard Russon <rich@flatcap.org>
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define TEST_NO_MAIN
#include "config.h"
#include "acutest.h"
#include <stddef.h>
#include "email/lib.h"

void test_email_cmp_strict(void)
{
  // bool email_cmp_strict(const struct Email *e1, const struct Email *e2);

  {
    struct Email e = { 0 };
    TEST_CHECK(!email_cmp_strict(NULL, &e));
  }

  {
    struct Email e = { 0 };
    TEST_CHECK(!email_cmp_strict(&e, NULL));
  }
}
