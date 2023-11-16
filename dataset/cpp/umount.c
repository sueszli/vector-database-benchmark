/***
  This file is part of systemd.

  Copyright 2010 ProFUSION embedded systems

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

#include <sys/mount.h>
#include <sys/swap.h>
#include <linux/dm-ioctl.h>
#include <linux/loop.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "fstab-util.h"
#include "libudev.h"
#include "list.h"
#include "mount-setup.h"
#include "path-util.h"
#include "udev-util.h"
#include "umount.h"
#include "util.h"
#include "virt.h"

typedef struct MountPoint {
	char *path;
	char *options;
	char *type;
	dev_t devnum;
	IWLIST_FIELDS(struct MountPoint, mount_point);
} MountPoint;

static void
mount_point_free(MountPoint **head, MountPoint *m)
{
	assert(head);
	assert(m);

	IWLIST_REMOVE(mount_point, *head, m);

	free(m->path);
	free(m);
}

static void
mount_points_list_free(MountPoint **head)
{
	assert(head);

	while (*head)
		mount_point_free(head, *head);
}

static int
mount_points_list_get(MountPoint **head)
{
	_cleanup_fclose_ FILE *proc_self_mountinfo = NULL;
	unsigned int i;

	assert(head);

	proc_self_mountinfo = fopen("/proc/self/mountinfo", "re");
	if (!proc_self_mountinfo)
		return -errno;

	for (i = 1;; i++) {
		_cleanup_free_ char *path = NULL, *options = NULL, *type = NULL;
		char *p = NULL;
		MountPoint *m;
		int k;

		k = fscanf(proc_self_mountinfo,
			"%*s " /* (1) mount id */
			"%*s " /* (2) parent id */
			"%*s " /* (3) major:minor */
			"%*s " /* (4) root */
			"%ms " /* (5) mount point */
			"%*s" /* (6) mount flags */
			"%*[^-]" /* (7) optional fields */
			"- " /* (8) separator */
			"%ms " /* (9) file system type */
			"%*s" /* (10) mount source */
			"%ms" /* (11) mount options */
			"%*[^\n]", /* some rubbish at the end */
			&path, &type, &options);
		if (k != 3) {
			if (k == EOF)
				break;

			log_warning("Failed to parse /proc/self/mountinfo:%u.",
				i);
			continue;
		}

		p = cunescape(path);
		if (!p)
			return -ENOMEM;

		/* Ignore mount points we can't unmount because they
                 * are API or because we are keeping them open (like
                 * /dev/console). Also, ignore all mounts below API
                 * file systems, since they are likely virtual too,
                 * and hence not worth spending time on. Also, in
                 * unprivileged containers we might lack the rights to
                 * unmount these things, hence don't bother. */
		if (mount_point_is_api(p) || mount_point_ignore(p) ||
			path_startswith(p, "/dev") ||
			path_startswith(p, "/sys") ||
			path_startswith(p, "/proc")) {
			free(p);
			continue;
		}

		m = new0(MountPoint, 1);
		if (!m) {
			free(p);
			return -ENOMEM;
		}

		m->path = p;
		m->options = options;
		options = NULL;
		m->type = type;
		type = NULL;

		IWLIST_PREPEND(mount_point, *head, m);
	}

	return 0;
}

static int
swap_list_get(MountPoint **head)
{
	_cleanup_fclose_ FILE *proc_swaps = NULL;
	unsigned int i;

	assert(head);

	if (!(proc_swaps = fopen("/proc/swaps", "re")))
		return (errno == ENOENT) ? 0 : -errno;

	(void)fscanf(proc_swaps, "%*s %*s %*s %*s %*s\n");

	for (i = 2;; i++) {
		MountPoint *swap;
		char *dev = NULL, *d;
		int k;

		if ((k = fscanf(proc_swaps,
			     "%ms " /* device/file */
			     "%*s " /* type of swap */
			     "%*s " /* swap size */
			     "%*s " /* used */
			     "%*s\n", /* priority */
			     &dev)) != 1) {
			if (k == EOF)
				break;

			log_warning("Failed to parse /proc/swaps:%u.", i);

			free(dev);
			continue;
		}

		if (endswith(dev, " (deleted)")) {
			free(dev);
			continue;
		}

		d = cunescape(dev);
		free(dev);

		if (!d) {
			return -ENOMEM;
		}

		if (!(swap = new0(MountPoint, 1))) {
			free(d);
			return -ENOMEM;
		}

		swap->path = d;
		IWLIST_PREPEND(mount_point, *head, swap);
	}

	return 0;
}

static int
loopback_list_get(MountPoint **head)
{
	_cleanup_udev_enumerate_unref_ struct udev_enumerate *e = NULL;
	struct udev_list_entry *item = NULL, *first = NULL;
	_cleanup_udev_unref_ struct udev *udev = NULL;
	int r;

	assert(head);

	udev = udev_new();
	if (!udev)
		return -ENOMEM;

	e = udev_enumerate_new(udev);
	if (!e)
		return -ENOMEM;

	r = udev_enumerate_add_match_subsystem(e, "block");
	if (r < 0)
		return r;

	r = udev_enumerate_add_match_sysname(e, "loop*");
	if (r < 0)
		return r;

	r = udev_enumerate_add_match_sysattr(e, "loop/backing_file", NULL);
	if (r < 0)
		return r;

	r = udev_enumerate_scan_devices(e);
	if (r < 0)
		return r;

	first = udev_enumerate_get_list_entry(e);
	udev_list_entry_foreach(item, first)
	{
		MountPoint *lb;
		_cleanup_udev_device_unref_ struct udev_device *d;
		char *loop;
		const char *dn;

		d = udev_device_new_from_syspath(udev,
			udev_list_entry_get_name(item));
		if (!d)
			return -ENOMEM;

		dn = udev_device_get_devnode(d);
		if (!dn)
			continue;

		loop = strdup(dn);
		if (!loop)
			return -ENOMEM;

		lb = new0(MountPoint, 1);
		if (!lb) {
			free(loop);
			return -ENOMEM;
		}

		lb->path = loop;
		IWLIST_PREPEND(mount_point, *head, lb);
	}

	return 0;
}

static int
dm_list_get(MountPoint **head)
{
	_cleanup_udev_enumerate_unref_ struct udev_enumerate *e = NULL;
	struct udev_list_entry *item = NULL, *first = NULL;
	_cleanup_udev_unref_ struct udev *udev = NULL;
	int r;

	assert(head);

	udev = udev_new();
	if (!udev)
		return -ENOMEM;

	e = udev_enumerate_new(udev);
	if (!e)
		return -ENOMEM;

	r = udev_enumerate_add_match_subsystem(e, "block");
	if (r < 0)
		return r;

	r = udev_enumerate_add_match_sysname(e, "dm-*");
	if (r < 0)
		return r;

	r = udev_enumerate_scan_devices(e);
	if (r < 0)
		return r;

	first = udev_enumerate_get_list_entry(e);
	udev_list_entry_foreach(item, first)
	{
		MountPoint *m;
		_cleanup_udev_device_unref_ struct udev_device *d;
		dev_t devnum;
		char *node;
		const char *dn;

		d = udev_device_new_from_syspath(udev,
			udev_list_entry_get_name(item));
		if (!d)
			return -ENOMEM;

		devnum = udev_device_get_devnum(d);
		dn = udev_device_get_devnode(d);
		if (major(devnum) == 0 || !dn)
			continue;

		node = strdup(dn);
		if (!node)
			return -ENOMEM;

		m = new (MountPoint, 1);
		if (!m) {
			free(node);
			return -ENOMEM;
		}

		m->path = node;
		m->devnum = devnum;
		IWLIST_PREPEND(mount_point, *head, m);
	}

	return 0;
}

static int
delete_loopback(const char *device)
{
	_cleanup_close_ int fd = -1;
	int r;

	fd = open(device, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return errno == ENOENT ? 0 : -errno;

	r = ioctl(fd, LOOP_CLR_FD, 0);
	if (r >= 0)
		return 1;

	/* ENXIO: not bound, so no error */
	if (errno == ENXIO)
		return 0;

	return -errno;
}

static int
delete_dm(dev_t devnum)
{
	_cleanup_close_ int fd = -1;
	int r;
	struct dm_ioctl dm = {
		.version = { DM_VERSION_MAJOR, DM_VERSION_MINOR,
			DM_VERSION_PATCHLEVEL },
		.data_size = sizeof(dm),
		.dev = devnum,
	};

	assert(major(devnum) != 0);

	fd = open("/dev/mapper/control", O_RDWR | O_CLOEXEC);
	if (fd < 0)
		return -errno;

	r = ioctl(fd, DM_DEV_REMOVE, &dm);
	return r >= 0 ? 0 : -errno;
}

static int
remount_with_timeout(MountPoint *m, char *options, int *n_failed)
{
	pid_t pid;
	int r;

	BLOCK_SIGNALS(SIGCHLD);

	/* Due to the possiblity of a remount operation hanging, we
         * fork a child process and set a timeout. If the timeout
         * lapses, the assumption is that that particular remount
         * failed. */
	pid = fork();
	if (pid < 0)
		return log_error_errno(errno, "Failed to fork: %m");

	if (pid == 0) {
		log_info("Remounting '%s' read-only in with options '%s'.",
			m->path, options);

		/* Start the mount operation here in the child */
		r = mount(NULL, m->path, NULL, MS_REMOUNT | MS_RDONLY, options);
		if (r < 0)
			log_error_errno(errno,
				"Failed to remount '%s' read-only: %m",
				m->path);

		_exit(r < 0 ? EXIT_FAILURE : EXIT_SUCCESS);
	}

	r = wait_for_terminate_with_timeout(pid, DEFAULT_TIMEOUT_USEC);
	if (r == -ETIMEDOUT) {
		log_error_errno(errno,
			"Remounting '%s' - timed out, issuing SIGKILL to PID " PID_FMT
			".",
			m->path, pid);
		(void)kill(pid, SIGKILL);
	} else if (r < 0)
		log_error_errno(r, "Failed to wait for process: %m");

	return r;
}

static int
umount_with_timeout(MountPoint *m, bool *changed)
{
	pid_t pid;
	int r;

	BLOCK_SIGNALS(SIGCHLD);

	/* Due to the possiblity of a umount operation hanging, we
         * fork a child process and set a timeout. If the timeout
         * lapses, the assumption is that that particular umount
         * failed. */
	pid = fork();
	if (pid < 0)
		return log_error_errno(errno, "Failed to fork: %m");

	if (pid == 0) {
		log_info("Unmounting '%s'.", m->path);

		/* Start the mount operation here in the child Using MNT_FORCE
                 * causes some filesystems (e.g. FUSE and NFS and other network
                 * filesystems) to abort any pending requests and return -EIO
                 * rather than blocking indefinitely. If the filesysten is
                 * "busy", this may allow processes to die, thus making the
                 * filesystem less busy so the unmount might succeed (rather
                 * then return EBUSY).*/
		r = umount2(m->path, MNT_FORCE);
		if (r < 0)
			log_error_errno(errno, "Failed to unmount %s: %m",
				m->path);

		_exit(r < 0 ? EXIT_FAILURE : EXIT_SUCCESS);
	}

	r = wait_for_terminate_with_timeout(pid, DEFAULT_TIMEOUT_USEC);
	if (r == -ETIMEDOUT) {
		log_error_errno(errno,
			"Unmounting '%s' - timed out, issuing SIGKILL to PID " PID_FMT
			".",
			m->path, pid);
		(void)kill(pid, SIGKILL);
	} else if (r < 0)
		log_error_errno(r, "Failed to wait for process: %m");

	return r;
}

static int
mount_points_list_umount(MountPoint **head, bool *changed)
{
	MountPoint *m, *n;
	int n_failed = 0;

	assert(head);

	IWLIST_FOREACH_SAFE (mount_point, m, n, *head) {
		/* If we are in a container, don't attempt to
                   read-only mount anything as that brings no real
                   benefits, but might confuse the host, as we remount
                   the superblock here, not the bind mount.
                   If the filesystem is a network fs, also skip the
                   remount.  It brings no value (we cannot leave
                   a "dirty fs") and could hang if the network is down.
                   Note that umount2() is more careful and will not
                   hang because of the network being down. */
		if (detect_container(NULL) <= 0 &&
			!fstype_is_network(m->type)) {
			_cleanup_free_ char *options = NULL;
			/* MS_REMOUNT requires that the data parameter
                         * should be the same from the original mount
                         * except for the desired changes. Since we want
                         * to remount read-only, we should filter out
                         * rw (and ro too, because it confuses the kernel) */
			(void)fstab_filter_options(m->options, "rw\0ro\0", NULL,
				NULL, &options);

			/* We always try to remount directories
                         * read-only first, before we go on and umount
                         * them.
                         *
                         * Mount points can be stacked. If a mount
                         * point is stacked below / or /usr, we
                         * cannot umount or remount it directly,
                         * since there is no way to refer to the
                         * underlying mount. There's nothing we can do
                         * about it for the general case, but we can
                         * do something about it if it is aliased
                         * somehwere else via a bind mount. If we
                         * explicitly remount the super block of that
                         * alias read-only we hence should be
                         * relatively safe regarding keeping the fs we
                         * can otherwise not see dirty.
                         *
                         * Since the remount can hang in the instance of
                         * remote filesystems, we remount asynchronously
                         * and skip the subsequent umount if it fails */
			if (remount_with_timeout(m, options, &n_failed) < 0)
				continue;
		}

		/* Skip / and /usr since we cannot unmount that
                 * anyway, since we are running from it. They have
                 * already been remounted ro. */
		if (path_equal(m->path, "/")
#ifndef HAVE_SPLIT_USR
			|| path_equal(m->path, "/usr")
#endif
		)
			continue;

		/* Trying to umount */
		if (umount_with_timeout(m, changed) < 0)
			n_failed++;
		else {
			if (changed)
				*changed = true;

			mount_point_free(head, m);
		}
	}

	return n_failed;
}

static int
swap_points_list_off(MountPoint **head, bool *changed)
{
	MountPoint *m, *n;
	int n_failed = 0;

	assert(head);

	IWLIST_FOREACH_SAFE (mount_point, m, n, *head) {
		log_info("Deactivating swap %s.", m->path);
		if (swapoff(m->path) == 0) {
			if (changed)
				*changed = true;

			mount_point_free(head, m);
		} else {
			log_warning_errno(errno,
				"Could not deactivate swap %s: %m", m->path);
			n_failed++;
		}
	}

	return n_failed;
}

static int
loopback_points_list_detach(MountPoint **head, bool *changed)
{
	MountPoint *m, *n;
	int n_failed = 0, k;
	struct stat root_st;

	assert(head);

	k = lstat("/", &root_st);

	IWLIST_FOREACH_SAFE (mount_point, m, n, *head) {
		int r;
		struct stat loopback_st;

		if (k >= 0 && major(root_st.st_dev) != 0 &&
			lstat(m->path, &loopback_st) >= 0 &&
			root_st.st_dev == loopback_st.st_rdev) {
			n_failed++;
			continue;
		}

		log_info("Detaching loopback %s.", m->path);
		r = delete_loopback(m->path);
		if (r >= 0) {
			if (r > 0 && changed)
				*changed = true;

			mount_point_free(head, m);
		} else {
			log_warning_errno(errno,
				"Could not detach loopback %s: %m", m->path);
			n_failed++;
		}
	}

	return n_failed;
}

static int
dm_points_list_detach(MountPoint **head, bool *changed)
{
	MountPoint *m, *n;
	int n_failed = 0, k;
	struct stat root_st;

	assert(head);

	k = lstat("/", &root_st);

	IWLIST_FOREACH_SAFE (mount_point, m, n, *head) {
		int r;

		if (k >= 0 && major(root_st.st_dev) != 0 &&
			root_st.st_dev == m->devnum) {
			n_failed++;
			continue;
		}

		log_info("Detaching DM %u:%u.", major(m->devnum),
			minor(m->devnum));
		r = delete_dm(m->devnum);
		if (r >= 0) {
			if (changed)
				*changed = true;

			mount_point_free(head, m);
		} else {
			log_warning_errno(errno, "Could not detach DM %s: %m",
				m->path);
			n_failed++;
		}
	}

	return n_failed;
}

int
umount_all(bool *changed)
{
	int r;
	bool umount_changed;
	IWLIST_HEAD(MountPoint, mp_IWLIST_HEAD);

	IWLIST_HEAD_INIT(mp_IWLIST_HEAD);
	r = mount_points_list_get(&mp_IWLIST_HEAD);
	if (r < 0)
		goto end;

	/* retry umount, until nothing can be umounted anymore */
	do {
		umount_changed = false;

		mount_points_list_umount(&mp_IWLIST_HEAD, &umount_changed);
		if (umount_changed)
			*changed = true;

	} while (umount_changed);

end:
	mount_points_list_free(&mp_IWLIST_HEAD);

	return r;
}

int
swapoff_all(bool *changed)
{
	int r;
	IWLIST_HEAD(MountPoint, swap_IWLIST_HEAD);

	IWLIST_HEAD_INIT(swap_IWLIST_HEAD);

	r = swap_list_get(&swap_IWLIST_HEAD);
	if (r < 0)
		goto end;

	r = swap_points_list_off(&swap_IWLIST_HEAD, changed);

end:
	mount_points_list_free(&swap_IWLIST_HEAD);

	return r;
}

int
loopback_detach_all(bool *changed)
{
	int r;
	IWLIST_HEAD(MountPoint, loopback_IWLIST_HEAD);

	IWLIST_HEAD_INIT(loopback_IWLIST_HEAD);

	r = loopback_list_get(&loopback_IWLIST_HEAD);
	if (r < 0)
		goto end;

	r = loopback_points_list_detach(&loopback_IWLIST_HEAD, changed);

end:
	mount_points_list_free(&loopback_IWLIST_HEAD);

	return r;
}

int
dm_detach_all(bool *changed)
{
	int r;
	IWLIST_HEAD(MountPoint, dm_IWLIST_HEAD);

	IWLIST_HEAD_INIT(dm_IWLIST_HEAD);

	r = dm_list_get(&dm_IWLIST_HEAD);
	if (r < 0)
		goto end;

	r = dm_points_list_detach(&dm_IWLIST_HEAD, changed);

end:
	mount_points_list_free(&dm_IWLIST_HEAD);

	return r;
}
