/***
  This file is part of systemd.

  Copyright 2010 Lennart Poettering

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
#include <net/if.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "loopback-setup.h"
#include "macro.h"
#include "missing.h"
#include "socket-util.h"
#include "util.h"

#ifdef SVC_PLATFORM_Linux
#include <asm/types.h>
#include "rtnl-util.h"
#include "sd-rtnl.h"
#endif

#ifdef SVC_PLATFORM_Linux
static int
start_loopback(sd_rtnl *rtnl)
{
	_cleanup_rtnl_message_unref_ sd_rtnl_message *req = NULL;
	int r;

	r = sd_rtnl_message_new_link(rtnl, &req, RTM_SETLINK, LOOPBACK_IFINDEX);
	if (r < 0)
		return r;

	r = sd_rtnl_message_link_set_flags(req, IFF_UP, IFF_UP);
	if (r < 0)
		return r;

	r = sd_rtnl_call(rtnl, req, 0, NULL);
	if (r < 0)
		return r;

	return 0;
}

static bool
check_loopback(sd_rtnl *rtnl)
{
	_cleanup_rtnl_message_unref_ sd_rtnl_message *req = NULL, *reply = NULL;
	unsigned flags;
	int r;

	r = sd_rtnl_message_new_link(rtnl, &req, RTM_GETLINK, LOOPBACK_IFINDEX);
	if (r < 0)
		return false;

	r = sd_rtnl_call(rtnl, req, 0, &reply);
	if (r < 0)
		return false;

	r = sd_rtnl_message_link_get_flags(reply, &flags);
	if (r < 0)
		return false;

	return flags & IFF_UP;
}
#endif

int
loopback_setup(void)
{
#ifdef SVC_PLATFORM_Linux
	_cleanup_rtnl_unref_ sd_rtnl *rtnl = NULL;
	int r;

	r = sd_rtnl_open(&rtnl, 0);
	if (r < 0)
		return r;

	r = start_loopback(rtnl);
	if (r < 0) {
		/* If we lack the permissions to configure the
                 * loopback device, but we find it to be already
                 * configured, let's exit cleanly, in order to
                 * supported unprivileged containers. */
		if (r == -EPERM && check_loopback(rtnl))
			return 0;

		return log_warning_errno(r,
			"Failed to configure loopback device: %m");
	}

	return 0;
#else
	unimplemented();
	return -ENOTSUP;
#endif
}
