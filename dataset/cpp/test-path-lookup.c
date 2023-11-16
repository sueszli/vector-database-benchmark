/***
  This file is part of systemd.

  Copyright 2014 Zbigniew Jędrzejewski-Szmek

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

#include <sys/types.h>
#include <sys/stat.h>

#include "log.h"
#include "path-lookup.h"
#include "strv.h"

static void
test_paths(SystemdRunningAs running_as, bool personal)
{
	char template[] = "/tmp/test-path-lookup.XXXXXXX";

	_cleanup_lookup_paths_free_ LookupPaths lp = {};
	char *exists, *not ;

	assert_se(mkdtemp(template));
	exists = strjoina(template, "/exists");
	assert_se(mkdir(exists, 0755) == 0);
	not = strjoina(template, "/not");

	assert_se(lookup_paths_init(&lp, running_as, personal, NULL, exists,
			  not, not ) == 0);

	assert_se(!strv_isempty(lp.unit_path));
	assert_se(strv_contains(lp.unit_path, exists));
	assert_se(strv_contains(lp.unit_path, not ));

	assert_se(rm_rf_dangerous(template, false, true, false) >= 0);
}

static void
print_generator_paths(SystemdRunningAs running_as)
{
	_cleanup_strv_free_ char **paths;
	char **dir;

	log_info("Generators dirs (%s):",
		running_as == SYSTEMD_SYSTEM ? "system" : "user");

	paths = generator_paths(running_as);
	STRV_FOREACH (dir, paths)
		log_info("        %s", *dir);
}

int
main(int argc, char **argv)
{
	log_set_max_level(LOG_DEBUG);
	log_parse_environment();
	log_open();

	test_paths(SYSTEMD_SYSTEM, false);
	test_paths(SYSTEMD_SYSTEM, true);
	test_paths(SYSTEMD_USER, false);
	test_paths(SYSTEMD_USER, true);

	print_generator_paths(SYSTEMD_SYSTEM);
	print_generator_paths(SYSTEMD_USER);

	return EXIT_SUCCESS;
}
