/*
 * Copyright 2011, Oliver Tappe <zooey@hirschkaefer.de>
 * Distributed under the terms of the MIT License.
 */


#include "package_repo.h"

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const char* __progname;
const char* kCommandName = __progname;


static const char* kUsage =
	"Usage: %s <command> <command args>\n"
	"Creates or inspects a Haiku package repository file.\n"
	"\n"
	"Commands:\n"
	"  create [ <options> ] <repo-info> <package-file ...> \n"
	"    Creates a package repository from the information found in \n"
	"    <repo-info>, adding the given package files.\n"
	"\n"
	"    -C <dir>   - Change to directory <dir> before starting.\n"
	"    -q         - be quiet (don't show any output except for errors).\n"
	"    -v         - be verbose (list package attributes as encountered).\n"
	"\n"
	"  list [ <options> ] <package-repo>\n"
	"    Lists the contents of package repository file <package-repo>.\n"
	"\n"
	"    -v         - be verbose (list attributes of all packages found).\n"
	"\n"
	"  update [ <options> ] <source-repo> <new-repo> <package-list-file> \n"
	"    Creates package repository file <new-repo> with all the packages\n"
	"    contained in <package-list-file>. If possible, package-infos are\n"
	"    taken from <source-repo> to avoid the need for recomputing the\n"
	"    checksum of all packages.\n"
	"    <old-repo> and <new-repo> can be the same file.\n"
	"\n"
	"    -C <dir>   - Change to directory <dir> before starting.\n"
	"    -q         - be quiet (don't show any output except for errors).\n"
	"    -v         - be verbose (list package attributes as encountered).\n"
	"\n"
	"Common Options:\n"
	"  -h, --help   - Print this usage info.\n"
;


void
print_usage_and_exit(bool error)
{
    fprintf(error ? stderr : stdout, kUsage, kCommandName);
    exit(error ? 1 : 0);
}


int
main(int argc, const char* const* argv)
{
	if (argc < 2)
		print_usage_and_exit(true);

	const char* command = argv[1];
	if (strcmp(command, "create") == 0)
		return command_create(argc - 1, argv + 1);

	if (strcmp(command, "list") == 0)
		return command_list(argc - 1, argv + 1);

	if (strcmp(command, "update") == 0)
		return command_update(argc - 1, argv + 1);

	if (strcmp(command, "help") == 0)
		print_usage_and_exit(false);
	else
		print_usage_and_exit(true);

	// never gets here
	return 0;
}
