/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        systemd integration.
 *
 * Author:      Alexandre Cassen, <acassen@linux-vs.org>
 *
 *              This program is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *              See the GNU General Public License for more details.
 *
 *              This program is free software; you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License
 *              as published by the Free Software Foundation; either version
 *              2 of the License, or (at your option) any later version.
 *
 * Copyright (C) 2020-2020 Alexandre Cassen, <acassen@gmail.com>
 */

#include "config.h"

#include <stdbool.h>
#include <systemd/sd-daemon.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

#include "systemd.h"
#include "logger.h"

static bool parent_is_systemd;

bool
check_parent_systemd(void)
{
	char stat_buf[13];	/* "1 (systemd) " */
	int fd;
	int len;

	/* If our parent is not the init process, it can't be systemd */
	if (getppid() != 1)
		return false;

	if ((fd = open("/proc/1/stat", O_RDONLY)) == -1)
		return false;

	len = read(fd, stat_buf, sizeof(stat_buf) - 1);
	close(fd);
	if (len < 0)
		return false;

	stat_buf[len] = '\0';

	if (strcmp(stat_buf, "1 (systemd) "))
		return false;

	/* systemd sets $NOTIFY_SOCKET to allow returning of notify information,
	 * but it is only set if the service file has "Type=notify" */
	if (!getenv("NOTIFY_SOCKET"))
		return false;

	parent_is_systemd = true;

	return true;
}

void
systemd_notify_running(void)
{
	if (parent_is_systemd)
		sd_notifyf(0, "READY=1\nMAINPID=%lu", (unsigned long)getpid());
}

void
systemd_notify_reloading(void)
{
	if (parent_is_systemd)
		sd_notify(0, "RELOADING=1");
}

void
systemd_notify_error(int error_code)
{
	if (parent_is_systemd)
		sd_notifyf(0, "ERRNO=%d", error_code);
}

void
systemd_notify_stopping(void)
{
	if (parent_is_systemd)
		sd_notify(0, "STOPPING=1");
}

/* Clear the environment variable NOTIFY_SOCKET to stop any
 * child processes we create sending notify messages to systemd */
void
systemd_unset_notify(void)
{
	/* This is only called in a child process of the main
	 * process. If the systemd service type is notify (e.g. Debian),
	 * then systemd will log:
	 *   keepalived.service: Got notification message from PID nnnn, but reception only permitted for main PID nnnm
	 * We therefore need to appear as though our parent process (the main process) is sending the message.
	 *
	 * In fact, it seems a bit daft to send a message just to clear a local environment variable, but this seems to
	 * be the only option offered by systemd.
	 */
	if (parent_is_systemd) {
#ifdef HAVE_SD_PID_NOTIFY
		sd_pid_notify(getppid(), 1, "");
#else
		/* The above systemd message will be logged, but we don't have the functionality to stop it other
		 * than setting NotifyAccess=all, but that is dangerous. */
		sd_notify(1, "");
#endif
	}
}
