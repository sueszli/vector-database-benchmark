/***
  This file is part of systemd.

  Copyright 2013 Lennart Poettering

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

#include <sys/socket.h>
#include <string.h>

#include "af-list.h"
#include "util.h"

static const struct af_name *lookup_af(register const char *str,
	register GPERF_LEN_TYPE len);

#include "af-from-name.h"
#include "af-to-name.h"

const char *
af_to_name(int id)
{
	if (id <= 0)
		return NULL;

	if (id >= (int)ELEMENTSOF(af_names))
		return NULL;

	return af_names[id];
}

int
af_from_name(const char *name)
{
	const struct af_name *sc;

	assert(name);

	sc = lookup_af(name, strlen(name));
	if (!sc)
		return AF_UNSPEC;

	return sc->id;
}

int
af_max(void)
{
	return ELEMENTSOF(af_names);
}
