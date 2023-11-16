/**
 * @file
 * Test code for is_descendant()
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

void test_is_descendant(void)
{
  // bool is_descendant(struct MuttThread *a, struct MuttThread *b);

  {
    struct MuttThread muttthread = { 0 };
    TEST_CHECK(!is_descendant(NULL, &muttthread));
  }

  {
    struct MuttThread muttthread = { 0 };
    TEST_CHECK(!is_descendant(&muttthread, NULL));
  }
}
