/***
  This file is part of systemd.

  Copyright 2011 Lennart Poettering

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

#include <sys/inotify.h>
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>

#include "cgroup-util.h"
#include "fileio.h"
#include "login-shared.h"
#include "macro.h"
#include "sd-login.h"
#include "strv.h"
#include "util.h"

_public_ int
sd_pid_get_session(pid_t pid, char **session)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(session, -EINVAL);

	return cg_pid_get_session(pid, session);
}

_public_ int
sd_pid_get_unit(pid_t pid, char **unit)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(unit, -EINVAL);

	return cg_pid_get_unit(pid, unit);
}

_public_ int
sd_pid_get_user_unit(pid_t pid, char **unit)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(unit, -EINVAL);

	return cg_pid_get_user_unit(pid, unit);
}

_public_ int
sd_pid_get_machine_name(pid_t pid, char **name)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(name, -EINVAL);

	return cg_pid_get_machine_name(pid, name);
}

_public_ int
sd_pid_get_slice(pid_t pid, char **slice)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(slice, -EINVAL);

	return cg_pid_get_slice(pid, slice);
}

_public_ int
sd_pid_get_owner_uid(pid_t pid, uid_t *uid)
{
	assert_return(pid >= 0, -EINVAL);
	assert_return(uid, -EINVAL);

	return cg_pid_get_owner_uid(pid, uid);
}

_public_ int
sd_peer_get_session(int fd, char **session)
{
	struct socket_ucred ucred = {};
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(session, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_session(ucred.pid, session);
}

_public_ int
sd_peer_get_owner_uid(int fd, uid_t *uid)
{
	struct socket_ucred ucred;
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(uid, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_owner_uid(ucred.pid, uid);
}

_public_ int
sd_peer_get_unit(int fd, char **unit)
{
	struct socket_ucred ucred;
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(unit, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_unit(ucred.pid, unit);
}

_public_ int
sd_peer_get_user_unit(int fd, char **unit)
{
	struct socket_ucred ucred;
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(unit, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_user_unit(ucred.pid, unit);
}

_public_ int
sd_peer_get_machine_name(int fd, char **machine)
{
	struct socket_ucred ucred;
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(machine, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_machine_name(ucred.pid, machine);
}

_public_ int
sd_peer_get_slice(int fd, char **slice)
{
	struct socket_ucred ucred;
	int r;

	assert_return(fd >= 0, -EINVAL);
	assert_return(slice, -EINVAL);

	r = getpeercred(fd, &ucred);
	if (r < 0)
		return r;

	return cg_pid_get_slice(ucred.pid, slice);
}

static int
file_of_uid(uid_t uid, char **p)
{
	assert(p);

	if (asprintf(p, SVC_PKGRUNSTATEDIR "/users/" UID_FMT, uid) < 0)
		return -ENOMEM;

	return 0;
}

_public_ int
sd_uid_get_state(uid_t uid, char **state)
{
	_cleanup_free_ char *p = NULL;
	char *s = NULL;
	int r;

	assert_return(state, -EINVAL);

	r = file_of_uid(uid, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "STATE", &s, NULL);
	if (r == -ENOENT) {
		free(s);
		s = strdup("offline");
		if (!s)
			return -ENOMEM;

	} else if (r < 0) {
		free(s);
		return r;
	} else if (!s)
		return -EIO;

	*state = s;
	return 0;
}

_public_ int
sd_uid_get_display(uid_t uid, char **session)
{
	_cleanup_free_ char *p = NULL, *s = NULL;
	int r;

	assert_return(session, -EINVAL);

	r = file_of_uid(uid, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "DISPLAY", &s, NULL);
	if (r < 0)
		return r;

	if (isempty(s))
		return -ENOENT;

	*session = s;
	s = NULL;

	return 0;
}

_public_ int
sd_uid_is_on_seat(uid_t uid, int require_active, const char *seat)
{
	_cleanup_free_ char *t = NULL, *s = NULL, *p = NULL;
	size_t l;
	int r;
	const char *word, *variable, *state;

	assert_return(seat, -EINVAL);

	variable = require_active ? "ACTIVE_UID" : "UIDS";

	p = strappend(SVC_PKGRUNSTATEDIR "/seats/", seat);
	if (!p)
		return -ENOMEM;

	r = parse_env_file(p, NEWLINE, variable, &s, NULL);

	if (r < 0)
		return r;

	if (!s)
		return -EIO;

	if (asprintf(&t, UID_FMT, uid) < 0)
		return -ENOMEM;

	FOREACH_WORD(word, l, s, state)
	{
		if (strneq(t, word, l))
			return 1;
	}

	return 0;
}

static int
uid_get_array(uid_t uid, const char *variable, char ***array)
{
	_cleanup_free_ char *p = NULL, *s = NULL;
	char **a;
	int r;

	r = file_of_uid(uid, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, variable, &s, NULL);
	if (r < 0) {
		if (r == -ENOENT) {
			if (array)
				*array = NULL;
			return 0;
		}

		return r;
	}

	if (!s) {
		if (array)
			*array = NULL;
		return 0;
	}

	a = strv_split(s, " ");

	if (!a)
		return -ENOMEM;

	strv_uniq(a);
	r = strv_length(a);

	if (array)
		*array = a;
	else
		strv_free(a);

	return r;
}

_public_ int
sd_uid_get_sessions(uid_t uid, int require_active, char ***sessions)
{
	return uid_get_array(uid,
		require_active == 0	   ? "ONLINE_SESSIONS" :
			require_active > 0 ? "ACTIVE_SESSIONS" :
						   "SESSIONS",
		sessions);
}

_public_ int
sd_uid_get_seats(uid_t uid, int require_active, char ***seats)
{
	return uid_get_array(uid,
		require_active == 0	   ? "ONLINE_SEATS" :
			require_active > 0 ? "ACTIVE_SEATS" :
						   "SEATS",
		seats);
}

static int
file_of_session(const char *session, char **_p)
{
	char *p;
	int r;

	assert(_p);

	if (session) {
		if (!session_id_valid(session))
			return -EINVAL;

		p = strappend(SVC_PKGRUNSTATEDIR "/sessions/", session);
	} else {
		_cleanup_free_ char *buf = NULL;

		r = sd_pid_get_session(0, &buf);
		if (r < 0)
			return r;

		p = strappend(SVC_PKGRUNSTATEDIR "/sessions/", buf);
	}

	if (!p)
		return -ENOMEM;

	*_p = p;
	return 0;
}

_public_ int
sd_session_is_active(const char *session)
{
	int r;
	_cleanup_free_ char *p = NULL, *s = NULL;

	r = file_of_session(session, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "ACTIVE", &s, NULL);
	if (r < 0)
		return r;

	if (!s)
		return -EIO;

	return parse_boolean(s);
}

_public_ int
sd_session_is_remote(const char *session)
{
	int r;
	_cleanup_free_ char *p = NULL, *s = NULL;

	r = file_of_session(session, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "REMOTE", &s, NULL);
	if (r < 0)
		return r;

	if (!s)
		return -EIO;

	return parse_boolean(s);
}

_public_ int
sd_session_get_state(const char *session, char **state)
{
	_cleanup_free_ char *p = NULL, *s = NULL;
	int r;

	assert_return(state, -EINVAL);

	r = file_of_session(session, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "STATE", &s, NULL);
	if (r < 0)
		return r;
	else if (!s)
		return -EIO;

	*state = s;
	s = NULL;

	return 0;
}

_public_ int
sd_session_get_uid(const char *session, uid_t *uid)
{
	int r;
	_cleanup_free_ char *p = NULL, *s = NULL;

	assert_return(uid, -EINVAL);

	r = file_of_session(session, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "UID", &s, NULL);
	if (r < 0)
		return r;

	if (!s)
		return -EIO;

	return parse_uid(s, uid);
}

static int
session_get_string(const char *session, const char *field, char **value)
{
	_cleanup_free_ char *p = NULL, *s = NULL;
	int r;

	assert_return(value, -EINVAL);

	r = file_of_session(session, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, field, &s, NULL);
	if (r < 0)
		return r;

	if (isempty(s))
		return -ENOENT;

	*value = s;
	s = NULL;
	return 0;
}

_public_ int
sd_session_get_seat(const char *session, char **seat)
{
	return session_get_string(session, "SEAT", seat);
}

_public_ int
sd_session_get_tty(const char *session, char **tty)
{
	return session_get_string(session, "TTY", tty);
}

_public_ int
sd_session_get_vt(const char *session, unsigned *vtnr)
{
	_cleanup_free_ char *vtnr_string = NULL;
	unsigned u;
	int r;

	r = session_get_string(session, "VTNR", &vtnr_string);
	if (r < 0)
		return r;

	r = safe_atou(vtnr_string, &u);
	if (r < 0)
		return r;

	*vtnr = u;
	return 0;
}

_public_ int
sd_session_get_service(const char *session, char **service)
{
	return session_get_string(session, "SERVICE", service);
}

_public_ int
sd_session_get_type(const char *session, char **type)
{
	return session_get_string(session, "TYPE", type);
}

_public_ int
sd_session_get_class(const char *session, char **class)
{
	return session_get_string(session, "CLASS", class);
}

_public_ int
sd_session_get_desktop(const char *session, char **desktop)
{
	_cleanup_free_ char *escaped = NULL;
	char *t;
	int r;

	assert_return(desktop, -EINVAL);

	r = session_get_string(session, "DESKTOP", &escaped);
	if (r < 0)
		return r;

	t = cunescape(escaped);
	if (!t)
		return -ENOMEM;

	*desktop = t;
	return 0;
}

_public_ int
sd_session_get_display(const char *session, char **display)
{
	return session_get_string(session, "DISPLAY", display);
}

_public_ int
sd_session_get_remote_user(const char *session, char **remote_user)
{
	return session_get_string(session, "REMOTE_USER", remote_user);
}

_public_ int
sd_session_get_remote_host(const char *session, char **remote_host)
{
	return session_get_string(session, "REMOTE_HOST", remote_host);
}

static int
file_of_seat(const char *seat, char **_p)
{
	char *p;
	int r;

	assert(_p);

	if (seat)
		p = strappend(SVC_PKGRUNSTATEDIR "/seats/", seat);
	else {
		_cleanup_free_ char *buf = NULL;

		r = sd_session_get_seat(NULL, &buf);
		if (r < 0)
			return r;

		p = strappend(SVC_PKGRUNSTATEDIR "/seats/", buf);
	}

	if (!p)
		return -ENOMEM;

	*_p = p;
	p = NULL;
	return 0;
}

_public_ int
sd_seat_get_active(const char *seat, char **session, uid_t *uid)
{
	_cleanup_free_ char *p = NULL, *s = NULL, *t = NULL;
	int r;

	assert_return(session || uid, -EINVAL);

	r = file_of_seat(seat, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "ACTIVE", &s, "ACTIVE_UID", &t, NULL);
	if (r < 0)
		return r;

	if (session && !s)
		return -ENOENT;

	if (uid && !t)
		return -ENOENT;

	if (uid && t) {
		r = parse_uid(t, uid);
		if (r < 0)
			return r;
	}

	if (session && s) {
		*session = s;
		s = NULL;
	}

	return 0;
}

_public_ int
sd_seat_get_sessions(const char *seat, char ***sessions, uid_t **uids,
	unsigned *n_uids)
{
	_cleanup_free_ char *p = NULL, *s = NULL, *t = NULL;
	_cleanup_strv_free_ char **a = NULL;
	_cleanup_free_ uid_t *b = NULL;
	unsigned n = 0;
	int r;

	r = file_of_seat(seat, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, "SESSIONS", &s, "ACTIVE_SESSIONS", &t,
		NULL);

	if (r < 0)
		return r;

	if (s) {
		a = strv_split(s, " ");
		if (!a)
			return -ENOMEM;
	}

	if (uids && t) {
		const char *word, *state;
		size_t l;

		FOREACH_WORD(word, l, t, state)
		n++;

		if (n > 0) {
			unsigned i = 0;

			b = new (uid_t, n);
			if (!b)
				return -ENOMEM;

			FOREACH_WORD(word, l, t, state)
			{
				_cleanup_free_ char *k = NULL;

				k = strndup(word, l);
				if (!k)
					return -ENOMEM;

				r = parse_uid(k, b + i);

				if (r < 0)
					continue;

				i++;
			}
		}
	}

	r = strv_length(a);

	if (sessions) {
		*sessions = a;
		a = NULL;
	}

	if (uids) {
		*uids = b;
		b = NULL;
	}

	if (n_uids)
		*n_uids = n;

	return r;
}

static int
seat_get_can(const char *seat, const char *variable)
{
	_cleanup_free_ char *p = NULL, *s = NULL;
	int r;

	assert_return(variable, -EINVAL);

	r = file_of_seat(seat, &p);
	if (r < 0)
		return r;

	r = parse_env_file(p, NEWLINE, variable, &s, NULL);
	if (r < 0)
		return r;
	if (!s)
		return 0;

	return parse_boolean(s);
}

_public_ int
sd_seat_can_multi_session(const char *seat)
{
	return seat_get_can(seat, "CAN_MULTI_SESSION");
}

_public_ int
sd_seat_can_tty(const char *seat)
{
	return seat_get_can(seat, "CAN_TTY");
}

_public_ int
sd_seat_can_graphical(const char *seat)
{
	return seat_get_can(seat, "CAN_GRAPHICAL");
}

_public_ int
sd_get_seats(char ***seats)
{
	return get_files_in_directory(SVC_PKGRUNSTATEDIR "/seats/", seats);
}

_public_ int
sd_get_sessions(char ***sessions)
{
	return get_files_in_directory(SVC_PKGRUNSTATEDIR "/sessions/",
		sessions);
}

_public_ int
sd_get_uids(uid_t **users)
{
	_cleanup_closedir_ DIR *d;
	int r = 0;
	unsigned n = 0;
	_cleanup_free_ uid_t *l = NULL;

	d = opendir(SVC_PKGRUNSTATEDIR "/users/");
	if (!d)
		return -errno;

	for (;;) {
		struct dirent *de;
		int k;
		uid_t uid;

		errno = 0;
		de = readdir(d);
		if (!de && errno != 0)
			return -errno;

		if (!de)
			break;

		dirent_ensure_type(d, de);

		if (!dirent_is_file(de))
			continue;

		k = parse_uid(de->d_name, &uid);
		if (k < 0)
			continue;

		if (users) {
			if ((unsigned)r >= n) {
				uid_t *t;

				n = MAX(16, 2 * r);
				t = realloc(l, sizeof(uid_t) * n);
				if (!t)
					return -ENOMEM;

				l = t;
			}

			assert((unsigned)r < n);
			l[r++] = uid;
		} else
			r++;
	}

	if (users) {
		*users = l;
		l = NULL;
	}

	return r;
}

_public_ int
sd_get_machine_names(char ***machines)
{
	char **l = NULL, **a, **b;
	int r;

	assert_return(machines, -EINVAL);

	r = get_files_in_directory(SVC_PKGRUNSTATEDIR "/machines/", &l);
	if (r < 0)
		return r;

	if (l) {
		r = 0;

		/* Filter out the unit: symlinks */
		for (a = l, b = l; *a; a++) {
			if (startswith(*a, "unit:"))
				free(*a);
			else {
				*b = *a;
				b++;
				r++;
			}
		}

		*b = NULL;
	}

	*machines = l;
	return r;
}

_public_ int
sd_machine_get_class(const char *machine, char **class)
{
	_cleanup_free_ char *c = NULL;
	const char *p;
	int r;

	assert_return(machine_name_is_valid(machine), -EINVAL);
	assert_return(class, -EINVAL);

	p = strjoina(SVC_PKGRUNSTATEDIR "/machines/", machine);
	r = parse_env_file(p, NEWLINE, "CLASS", &c, NULL);
	if (r < 0)
		return r;
	if (!c)
		return -EIO;

	*class = c;
	c = NULL;

	return 0;
}

_public_ int
sd_machine_get_ifindices(const char *machine, int **ifindices)
{
	_cleanup_free_ char *netif = NULL;
	size_t l, allocated = 0, nr = 0;
	int *ni = NULL;
	const char *p, *word, *state;
	int r;

	assert_return(machine_name_is_valid(machine), -EINVAL);
	assert_return(ifindices, -EINVAL);

	p = strjoina(SVC_PKGRUNSTATEDIR "/machines/", machine);
	r = parse_env_file(p, NEWLINE, "NETIF", &netif, NULL);
	if (r < 0)
		return r;
	if (!netif) {
		*ifindices = NULL;
		return 0;
	}

	FOREACH_WORD(word, l, netif, state)
	{
		char buf[l + 1];
		int ifi;

		*(char *)(mempcpy(buf, word, l)) = 0;

		if (safe_atoi(buf, &ifi) < 0)
			continue;
		if (ifi <= 0)
			continue;

		if (!GREEDY_REALLOC(ni, allocated, nr + 1)) {
			free(ni);
			return -ENOMEM;
		}

		ni[nr++] = ifi;
	}

	*ifindices = ni;
	return nr;
}

static inline int
MONITOR_TO_FD(sd_login_monitor *m)
{
	return (int)(unsigned long)m - 1;
}

static inline sd_login_monitor *
FD_TO_MONITOR(int fd)
{
	return (sd_login_monitor *)(unsigned long)(fd + 1);
}

_public_ int
sd_login_monitor_new(const char *category, sd_login_monitor **m)
{
	int fd, k;
	bool good = false;

	assert_return(m, -EINVAL);

	fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	if (fd < 0)
		return -errno;

	if (!category || streq(category, "seat")) {
		k = inotify_add_watch(fd, SVC_PKGRUNSTATEDIR "/seats/",
			IN_MOVED_TO | IN_DELETE);
		if (k < 0) {
			safe_close(fd);
			return -errno;
		}

		good = true;
	}

	if (!category || streq(category, "session")) {
		k = inotify_add_watch(fd, SVC_PKGRUNSTATEDIR "/sessions/",
			IN_MOVED_TO | IN_DELETE);
		if (k < 0) {
			safe_close(fd);
			return -errno;
		}

		good = true;
	}

	if (!category || streq(category, "uid")) {
		k = inotify_add_watch(fd, SVC_PKGRUNSTATEDIR "/users/",
			IN_MOVED_TO | IN_DELETE);
		if (k < 0) {
			safe_close(fd);
			return -errno;
		}

		good = true;
	}

	if (!category || streq(category, "machine")) {
		k = inotify_add_watch(fd, SVC_PKGRUNSTATEDIR "/machines/",
			IN_MOVED_TO | IN_DELETE);
		if (k < 0) {
			safe_close(fd);
			return -errno;
		}

		good = true;
	}

	if (!good) {
		close_nointr(fd);
		return -EINVAL;
	}

	*m = FD_TO_MONITOR(fd);
	return 0;
}

_public_ sd_login_monitor *
sd_login_monitor_unref(sd_login_monitor *m)
{
	int fd;

	assert_return(m, NULL);

	fd = MONITOR_TO_FD(m);
	close_nointr(fd);

	return NULL;
}

_public_ int
sd_login_monitor_flush(sd_login_monitor *m)
{
	assert_return(m, -EINVAL);

	return flush_fd(MONITOR_TO_FD(m));
}

_public_ int
sd_login_monitor_get_fd(sd_login_monitor *m)
{
	assert_return(m, -EINVAL);

	return MONITOR_TO_FD(m);
}

_public_ int
sd_login_monitor_get_events(sd_login_monitor *m)
{
	assert_return(m, -EINVAL);

	/* For now we will only return POLLIN here, since we don't
         * need anything else ever for inotify.  However, let's have
         * this API to keep our options open should we later on need
         * it. */
	return POLLIN;
}

_public_ int
sd_login_monitor_get_timeout(sd_login_monitor *m, uint64_t *timeout_usec)
{
	assert_return(m, -EINVAL);
	assert_return(timeout_usec, -EINVAL);

	/* For now we will only return (uint64_t) -1, since we don't
         * need any timeout. However, let's have this API to keep our
         * options open should we later on need it. */
	*timeout_usec = (uint64_t)-1;
	return 0;
}
