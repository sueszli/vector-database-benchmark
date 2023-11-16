/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        Linux namespace handling.
 *
 * Author:      Quentin Armitage <quentin@armitage.org.uk>
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
 * Copyright (C) 2016-2017 Alexandre Cassen, <acassen@gmail.com>
 */

/*******************************************************************************
 *
 * Running keepalived in a namespace provides isolation from other instances of
 * keepalived running on the same system, and is useful for a variety of reasons.
 *
 * In order not to have to specify different pid files for each instance of
 * keepalived, if keepalived is running in a network namespace it will also create
 * its own mount namespace, and will slave bind mount a unique directory
 * (/run/keepalived/NAMESPACE) on /run/keepalived, so keepalived will
 * write its usual pid files (but to /run/keepalived rather than to /run),
 * and outside the mount namespace these will be visible at
 * /run/keepalived/NAMESPACE.
 *
 * If you are familiar with network namespaces, then you will know what you can do
 * with them. If not, then the following scenarios should give you an idea of what
 * can be done, and why they might be helpful.
 *
 * If you wish to test keepalived, but don't wish to interfere with the
 * networking on a live system, or you wish to test multiple instances, but all
 * in one machine, then the following will set up such an environment.
 *
 * Three "machine" configuration:
 *
 *      netns1                       netns2                       netns3
 * ------------------           ------------------           ------------------
 * |                |           |  -----------   |           |                |
 * |                |           |  |   br0   |   |           |                |
 * |                |           |  -----------   |           |                |
 * |                |           |    |     |     |           |                |
 * |       e        |           |    e     e     |           |       e        |
 * |       t        |           |    t     t     |           |       t        |
 * |       h        |           |    h     h     |           |       h        |
 * |       0        |           |    0     1     |           |       0        |
 * |       |        |           |    |     |     |           |       |        |
 * ------------------           ------------------           ------------------
 *         |                         |     |                         |
 *         ---------------------------     ---------------------------
 *
 * NOTE: it is possible that `ip netns add NAME` will create the namespace but
 * include all the network links, rather than just lo. To check this, after
 * creating a namespace, run `ip netns exec NAME ip link show` and if it shows
 * all the network links, then you have this problem. To work around the problem,
 * prefix all the `ip netns add NAME` commands with unshare --net, e.g.
 *   unshare --net ip netns add NAME
 *
 * Create the namespaces
 * # ip netns add netns1
 * # ip netns add netns2
 * # ip netns add netns3
 *
 * Bring up the loopback interfaces
 * # ip netns exec netns1 ip link set lo up
 * # ip netns exec netns2 ip link set lo up
 * # ip netns exec netns3 ip link set lo up
 *
 * Create link between netns1 and netns2
 * # ip netns exec netns2 ip link add 1.eth0 type veth peer name eth0
 * # ip netns exec netns2 ip link set eth0 netns netns1
 *
 * Create link between netns2 and netns3
 * # ip netns exec netns2 ip link add 3.eth1 type veth peer name eth0
 * # ip netns exec netns2 ip link set eth0 netns netns3
 *
 * Make the link names in netns2 easier to remember
 * # ip netns exec netns2 ip link set 1.eth0 name eth0
 * # ip netns exec netns2 ip link set 3.eth1 name eth1
 *
 * Bring up the interfaces
 * # ip netns exec netns1 ip link set eth0 up
 * # ip netns exec netns2 ip link set eth0 up
 * # ip netns exec netns2 ip link set eth1 up
 * # ip netns exec netns3 ip link set eth0 up
 *
 * Bridge eth0 and eth1 in netns2
 * # ip netns exec netns2 ip link add br0 type bridge
 * # ip netns exec netns2 ip link set br0 up
 *
 * Connect eth0 and eth1 to br0 in netns2
 * # ip netns exec netns2 ip link set eth0 master br0
 * # ip netns exec netns2 ip link set eth1 master br0
 *
 * Configure some addresses
 * # ip netns exec netns1 ip addr add 10.2.0.1/24 broadcast 10.2.0.255 dev eth0
 * # ip netns exec netns2 ip addr add 10.2.0.2/24 broadcast 10.2.0.255 dev br0
 * # ip netns exec netns3 ip addr add 10.2.0.3/24 broadcast 10.2.0.255 dev eth0
 *
 * Test it
 * # ip netns exec netns1 ping 10.2.0.2		# netns1 can talk to netns2
 * # ip netns exec netns1 ping 10.2.0.3		# netns1 can talk to netns3 (bridge is working)
 *
 * If you want to enter multiple commands in a net namespace, then try:
 * # ip netns exec netns1 bash
 * # PS1="netns1 # "
 * netns1 #
 *
 * Create three configuration files, keepalived.netns1.conf etc
 * and in each config file in the global_defs section specify
 * net_namespace netns1        # or netns2 or netns3 as appropriate
 * global_defs {
 *		....
 *
 * Now run three instances of keepalived. Note, keepalived handles
 * joining the appropriate network namespace, and so the commands don't
 * need to be prefixed with 'ip netns exec netns1'.
 * # keepalived -f /etc/keepalived/keepalived.netns1.conf
 * # keepalived -f /etc/keepalived/keepalived.netns2.conf
 * # keepalived -f /etc/keepalived/keepalived.netns3.conf
 *
 * The syslog output will have the network namespace name appended to the
 * ident.
 *
 * If you want to connect the setup above to the real world, add the following:
 * # ip link add veth0 type veth peer name veth1
 * # ip link set veth1 netns netns2
 * # ip link set up veth0
 * # ip link set veth1 netns netns2
 * # ip netns exec netns2 ip link set up veth1
 * # ip netns exec netns2 ip link set veth1 master br0
 * # ip link add br0 type bridge
 * # ip link set br0 up
 * # ip link set veth0 master br0
 * # ip link set eth0 master br0
 * # ip link add addr 10.2.0.4/24 broadcast 10.2.0.255 dev br0
 *
 * There are further possibilities. If the above configuration is set up on two
 * separate machines, a tunnel could be established between the two netns2 instances
 * and the masters of each end of the tunnels set to br0. Alternatively, a new vlan
 * could be set up in (or moved to) the two netns2 instances, and added to the br0
 * bridges.
 *
 ******************************************************************************/

#include "config.h"

#include <fcntl.h>
#include <sched.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/mount.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/socket.h>

#ifdef LIBIPVS_USE_NL
#ifdef _HAVE_LIBNL1_
#include <linux/types.h>
#endif
#include <linux/netlink.h>	/* Required for netlink/headers.h prior to libnl v3.3.0 */
#include <netlink/socket.h>
#include <netlink/genl/genl.h>
#ifdef _LIBNL_DYNAMIC_
#include "libnl_link.h"
#endif
#endif

#include "namespaces.h"
#include "memory.h"
#include "logger.h"
#include "pidfile.h"
#include "utils.h"
#include "bitops.h"


/* Local data */
static const char *netns_dir = RUNSTATEDIR "/netns/";
static char *mount_dirname;
static bool run_mount_set;

void
free_dirname(void)
{
	FREE_PTR(mount_dirname);
	mount_dirname = NULL;
}

static void
set_run_mount(const char *net_namespace)
{
	bool error;

	/* /run/keepalived/NAMESPACE */
	mount_dirname = MALLOC(strlen(KEEPALIVED_PID_DIR) + 1 + strlen(net_namespace));
	if (!mount_dirname) {
		log_message(LOG_INFO, "Unable to allocate memory for pid file dirname");
		return;
	}

	strcpy(mount_dirname, KEEPALIVED_PID_DIR);
	strcat(mount_dirname, net_namespace);

	/* We want the directory to have rwxr-xr-x permissions */
	if (umask_val & (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))
		umask(umask_val & ~(S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH));

	error = mkdir(mount_dirname, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) && errno != EEXIST;

	/* Restore our default umask */
	if (umask_val & (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))
		umask(umask_val);

	if (error) {
		log_message(LOG_INFO, "Unable to create directory %s", mount_dirname);
		free_dirname();
		return;
	}

	if (unshare(CLONE_NEWNS)) {
		log_message(LOG_INFO, "mount unshare failed (%d) '%s'", errno, strerror(errno));
		return;
	}

	/* Make all mounts unshared - systemd makes them shared by default */
	if (mount("", "/", NULL, MS_REC | MS_SLAVE, NULL))
		log_message(LOG_INFO, "Mount slave failed, error (%d) '%s'", errno, strerror(errno));

	if (mount(mount_dirname, pid_directory, NULL, MS_BIND, NULL))
		log_message(LOG_INFO, "Mount failed, error (%d) '%s'", errno, strerror(errno));

	run_mount_set = true;
}

static void
unmount_run(void)
{
	if (!run_mount_set)
		return;

	if (umount(pid_directory))
		log_message(LOG_INFO, "unmount of %s failed - errno %d", pid_directory, errno);
	if (mount_dirname) {
		if (rmdir(mount_dirname) && errno != ENOTEMPTY && errno != EBUSY)
			log_message(LOG_INFO, "unlink of %s failed - error (%d) '%s'", mount_dirname, errno, strerror(errno));
		free_dirname();
	}
}

bool
set_namespaces(const char* net_namespace)
{
	char *netns_path;
	int fd;

	netns_path = MALLOC(strlen(netns_dir) + strlen(net_namespace) + 1);
	if (!netns_path) {
		log_message(LOG_INFO, "Unable to malloc for set_namespaces()");
		return false;
	}

	strcpy(netns_path, netns_dir);
	strcat(netns_path, net_namespace);

	fd = open(netns_path, O_RDONLY);
	if (fd == -1) {
		log_message(LOG_INFO, "Failed to open %s", netns_path);
		goto err;
	}

	if (setns(fd, CLONE_NEWNET)) {
		log_message(LOG_INFO, "setns() failed with error %d", errno);
		goto err;
	}

	close(fd);

	if (!__test_bit(CONFIG_TEST_BIT, &debug))
		set_run_mount(net_namespace);

	FREE_PTR(netns_path);
	netns_path = NULL;

	return true;

err:
	if (fd != -1)
		close(fd);
	FREE_PTR(netns_path);
	netns_path = NULL;

	return false;
}

void
clear_namespaces(void)
{
	unmount_run();
}

/* get default namespace file descriptor to be able to place fd in a given
 * namespace
 */
static int
open_current_namespace(void)
{
	return open("/proc/self/ns/net", O_RDONLY | O_CLOEXEC);
}

/* opens namespace for ipvs communication */
static int open_ipvs_namespace(const char *ns_name)
{
	char netns_path[PATH_MAX];
	const char *path;

	if (ns_name[0]) {
		if (snprintf(netns_path, sizeof(netns_path), "/var/run/netns/%s", ns_name) < 0)
			return -1;
		path = netns_path;
	} else
		path = "/proc/1/ns/net";

	return open(path, O_RDONLY | O_CLOEXEC);
}

#ifdef _INCLUDE_UNUSED_CODE_
/* opens a socket in the <nsfd> namespace with the parameters <domain>,
 * <type> and <protocol> and returns the FD or -1 in case of error (check errno).
 */
int
socket_netns(int nsfd, int domain, int type, int protocol)
{
	int sock;

	if (default_namespace >= 0 && nsfd && setns(nsfd, CLONE_NEWNET) == -1)
		return -1;

	sock = socket(domain, type, protocol);

	if (default_namespace >= 0 && nsfd && setns(default_namespace, CLONE_NEWNET) == -1) {
		if (sock >= 0)
			close(sock);
		return -1;
	}
	return sock;
}
#endif

int
set_netns_name(const char *netns_name)
{
	int cur_net_namespace = -1;
	int new_net_namespace;
	int ret;

	if (!netns_name)
		return -1;

	new_net_namespace = open_ipvs_namespace(netns_name);
	cur_net_namespace = open_current_namespace();

	if (cur_net_namespace < 0 || new_net_namespace < 0) {
		if (cur_net_namespace >= 0)
			close(cur_net_namespace);
		else if (new_net_namespace >= 0)
			close(new_net_namespace);
		log_message(LOG_INFO, "Failed to open namespace fds, errno %d (%m)", errno);
		return -1;
	}

	ret = setns(new_net_namespace, CLONE_NEWNET);
	close(new_net_namespace);

	if (ret < 0) {
		close(cur_net_namespace);
		log_message(LOG_INFO, "Open socket - unable to set namespace, errno %d (%m)", errno);
		return -1;
	}

	return cur_net_namespace;
}

void
restore_net_namespace(int cur_net_namespace)
{
	int ret;

	ret = setns(cur_net_namespace, CLONE_NEWNET);
	close(cur_net_namespace);

	if (ret < 0) {
		log_message(LOG_INFO, "Open socket - unable to restore default namespace, errno %d (%m)", errno);
		return;
	}
}

int
socket_netns_name(const char *netns_name, int domain, int type, int protocol)
{
	int cur_net_namespace = -1;
	int sock;

	if (netns_name &&
	    (cur_net_namespace = set_netns_name(netns_name)) < 0)
		return -1;

	sock = socket(domain, type, protocol);

	if (cur_net_namespace >= 0)
		restore_net_namespace(cur_net_namespace);

	return sock;
}

#ifdef LIBIPVS_USE_NL
/* netlink connect within ipvs namespace */
int
nl_ipvs_connect(const char *netns_name, struct nl_sock *sock)
{
	int cur_net_namespace = -1;
	int ret;

	if (netns_name &&
	    (cur_net_namespace = set_netns_name(netns_name)) < 0)
		return -1;

	ret = genl_connect(sock);

	if (cur_net_namespace >= 0)
		restore_net_namespace(cur_net_namespace);

	return ret < 0 ? -1 : 0;
}
#endif
