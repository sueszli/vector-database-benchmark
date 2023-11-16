/***
  This file is part of systemd.

  Copyright 2012 Lennart Poettering

  systemd is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.

  systemd is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with systemd; If not, see <http://www.gnu.org/licenses/>.
***/

#include <stdio.h>

#include "util.h"

int
main(int argc, char *argv[])
{
	char *p;

	assert_se(p = strdup("\tFoobar\tbar\twaldo\t"));
	assert_se(strip_tab_ansi(&p, NULL));
	fprintf(stdout, "<%s>\n", p);
	assert_se(streq(p, "        Foobar        bar        waldo        "));
	free(p);

	assert_se(p = strdup(ANSI_HIGHLIGHT_ON
			  "Hello" ANSI_HIGHLIGHT_OFF ANSI_HIGHLIGHT_RED_ON
			  " world!" ANSI_HIGHLIGHT_OFF));
	assert_se(strip_tab_ansi(&p, NULL));
	fprintf(stdout, "<%s>\n", p);
	assert_se(streq(p, "Hello world!"));
	free(p);

	assert_se(p = strdup("\x1B[\x1B[\t\x1B[" ANSI_HIGHLIGHT_ON "\x1B["
			     "Hello" ANSI_HIGHLIGHT_OFF ANSI_HIGHLIGHT_RED_ON
			     " world!" ANSI_HIGHLIGHT_OFF));
	assert_se(strip_tab_ansi(&p, NULL));
	assert_se(streq(p, "\x1B[\x1B[        \x1B[\x1B[Hello world!"));
	free(p);

	assert_se(p = strdup("\x1B[waldo"));
	assert_se(strip_tab_ansi(&p, NULL));
	assert_se(streq(p, "\x1B[waldo"));
	free(p);

	return 0;
}
