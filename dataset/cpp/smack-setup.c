/***
  This file is part of systemd.

  Copyright (C) 2013 Intel Corporation
  Authors:
        Nathaniel Chen <nathaniel.chen@intel.com>

  systemd is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  systemd is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with systemd; If not, see <http://www.gnu.org/licenses/>.
***/

#include <sys/types.h>
#include <sys/mount.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fileio.h"
#include "label.h"
#include "log.h"
#include "macro.h"
#include "smack-setup.h"
#include "util.h"

#define SMACK_CONFIG "/etc/smack/accesses.d/"
#define CIPSO_CONFIG "/etc/smack/cipso.d/"

#ifdef HAVE_SMACK

#include <sys/vfs.h>

static int
write_rules(const char *dstpath, const char *srcdir)
{
	_cleanup_fclose_ FILE *dst = NULL;
	_cleanup_closedir_ DIR *dir = NULL;
	struct dirent *entry;
	char buf[NAME_MAX];
	int dfd = -1;
	int r = 0;

	dst = fopen(dstpath, "we");
	if (!dst) {
		if (errno != ENOENT)
			log_warning_errno(errno, "Failed to open %s: %m",
				dstpath);
		return -errno; /* negative error */
	}

	/* write rules to dst from every file in the directory */
	dir = opendir(srcdir);
	if (!dir) {
		if (errno != ENOENT)
			log_warning_errno(errno, "Failed to opendir %s: %m",
				srcdir);
		return errno; /* positive on purpose */
	}

	dfd = dirfd(dir);
	assert(dfd >= 0);

	FOREACH_DIRENT (entry, dir, return 0) {
		int fd;
		_cleanup_fclose_ FILE *policy = NULL;

		fd = openat(dfd, entry->d_name, O_RDONLY | O_CLOEXEC);
		if (fd < 0) {
			if (r == 0)
				r = -errno;
			log_warning_errno(errno, "Failed to open %s: %m",
				entry->d_name);
			continue;
		}

		policy = fdopen(fd, "re");
		if (!policy) {
			if (r == 0)
				r = -errno;
			safe_close(fd);
			log_error_errno(errno, "Failed to open %s: %m",
				entry->d_name);
			continue;
		}

		/* load2 write rules in the kernel require a line buffered stream */
		FOREACH_LINE(buf, policy,
			log_error_errno(errno,
				"Failed to read line from %s: %m",
				entry->d_name))
		{
			if (!fputs(buf, dst)) {
				if (r == 0)
					r = -EINVAL;
				log_error("Failed to write line to %s",
					dstpath);
				break;
			}
			if (fflush(dst)) {
				if (r == 0)
					r = -errno;
				log_error_errno(errno,
					"Failed to flush writes to %s: %m",
					dstpath);
				break;
			}
		}
	}

	return r;
}

#endif

int
mac_smack_setup(bool *loaded_policy)
{
#ifdef HAVE_SMACK

	int r;

	assert(loaded_policy);

	r = write_rules("/sys/fs/smackfs/load2", SMACK_CONFIG);
	switch (r) {
	case -ENOENT:
		log_debug("Smack is not enabled in the kernel.");
		return 0;
	case ENOENT:
		log_debug("Smack access rules directory " SMACK_CONFIG
			  " not found");
		return 0;
	case 0:
		log_info("Successfully loaded Smack policies.");
		break;
	default:
		log_warning("Failed to load Smack access rules: %s, ignoring.",
			strerror(abs(r)));
		return 0;
	}

#ifdef SMACK_RUN_LABEL
	r = write_string_file("/proc/self/attr/current", SMACK_RUN_LABEL);
	if (r)
		log_warning("Failed to set SMACK label \"%s\" on self: %s",
			SMACK_RUN_LABEL, strerror(-r));
#endif

	r = write_rules("/sys/fs/smackfs/cipso2", CIPSO_CONFIG);
	switch (r) {
	case -ENOENT:
		log_debug("Smack/CIPSO is not enabled in the kernel.");
		return 0;
	case ENOENT:
		log_debug("Smack/CIPSO access rules directory " CIPSO_CONFIG
			  " not found");
		return 0;
	case 0:
		log_info("Successfully loaded Smack/CIPSO policies.");
		break;
	default:
		log_warning(
			"Failed to load Smack/CIPSO access rules: %s, ignoring.",
			strerror(abs(r)));
		return 0;
	}

	*loaded_policy = true;

#endif

	return 0;
}
