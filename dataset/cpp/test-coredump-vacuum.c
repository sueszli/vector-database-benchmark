/***
  This file is part of systemd.

  Copyright 2012 Zbigniew Jędrzejewski-Szmek

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

#include <stdlib.h>

#include "coredump-vacuum.h"

int
main(int argc, char *argv[])
{
	if (coredump_vacuum(-1, (off_t)-1, 70 * 1024) < 0)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
