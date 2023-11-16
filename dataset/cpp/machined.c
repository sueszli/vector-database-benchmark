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

#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <string.h>
#include <unistd.h>

#include "bus-error.h"
#include "bus-util.h"
#include "cgroup-util.h"
#include "conf-parser.h"
#include "label.h"
#include "machine-image.h"
#include "machined.h"
#include "mkdir.h"
#include "sd-daemon.h"
#include "strv.h"

Manager *
manager_new(void)
{
	Manager *m;
	int r;

	m = new0(Manager, 1);
	if (!m)
		return NULL;

	m->machines = hashmap_new(&string_hash_ops);
	m->machine_units = hashmap_new(&string_hash_ops);
	m->machine_leaders = hashmap_new(NULL);

	if (!m->machines || !m->machine_units || !m->machine_leaders) {
		manager_free(m);
		return NULL;
	}

	r = sd_event_default(&m->event);
	if (r < 0) {
		manager_free(m);
		return NULL;
	}

	sd_event_set_watchdog(m->event, true);

	return m;
}

void
manager_free(Manager *m)
{
	Machine *machine;
	Image *i;

	assert(m);

	while ((machine = hashmap_first(m->machines)))
		machine_free(machine);

	hashmap_free(m->machines);
	hashmap_free(m->machine_units);
	hashmap_free(m->machine_leaders);

	while ((i = hashmap_steal_first(m->image_cache)))
		image_unref(i);

	hashmap_free(m->image_cache);

	sd_event_source_unref(m->image_cache_defer_event);

	bus_verify_polkit_async_registry_free(m->polkit_registry);

	sd_bus_unref(m->bus);
	sd_event_unref(m->event);

	free(m);
}

int
manager_enumerate_machines(Manager *m)
{
	_cleanup_closedir_ DIR *d = NULL;
	struct dirent *de;
	int r = 0;

	assert(m);

	/* Read in machine data stored on disk */
	d = opendir(SVC_PKGRUNSTATEDIR "/machines");
	if (!d) {
		if (errno == ENOENT)
			return 0;

		log_error_errno(errno,
			"Failed to open " SVC_PKGRUNSTATEDIR "/machines: %m");
		return -errno;
	}

	FOREACH_DIRENT (de, d, return -errno) {
		struct Machine *machine;
		int k;

		if (!dirent_is_file(de))
			continue;

		/* Ignore symlinks that map the unit name to the machine */
		if (startswith(de->d_name, "unit:"))
			continue;

		k = manager_add_machine(m, de->d_name, &machine);
		if (k < 0) {
			log_error_errno(k,
				"Failed to add machine by file name %s: %m",
				de->d_name);

			r = k;
			continue;
		}

		machine_add_to_gc_queue(machine);

		k = machine_load(machine);
		if (k < 0)
			r = k;
	}

	return r;
}

static int
manager_connect_bus(Manager *m)
{
	_cleanup_bus_error_free_ sd_bus_error error = SD_BUS_ERROR_NULL;
	int r;

	assert(m);
	assert(!m->bus);

	r = sd_bus_default_system(&m->bus);
	if (r < 0)
		return log_error_errno(r,
			"Failed to connect to system bus: %m");

	r = sd_bus_add_object_vtable(m->bus, NULL, "/org/freedesktop/machine1",
		SVC_MACHINED_DBUS_INTERFACE ".Manager", manager_vtable, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add manager object vtable: %m");

	r = sd_bus_add_fallback_vtable(m->bus, NULL,
		"/org/freedesktop/machine1/machine",
		SVC_MACHINED_DBUS_INTERFACE ".Machine", machine_vtable,
		machine_object_find, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add machine object vtable: %m");

	r = sd_bus_add_node_enumerator(m->bus, NULL,
		"/org/freedesktop/machine1/machine", machine_node_enumerator,
		m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add machine enumerator: %m");

	r = sd_bus_add_fallback_vtable(m->bus, NULL,
		"/org/freedesktop/machine1/image",
		SVC_MACHINED_DBUS_INTERFACE ".Image", image_vtable,
		image_object_find, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add image object vtable: %m");

	r = sd_bus_add_node_enumerator(m->bus, NULL,
		"/org/freedesktop/machine1/image", image_node_enumerator, m);
	if (r < 0)
		return log_error_errno(r, "Failed to add image enumerator: %m");

	r = sd_bus_add_match(m->bus, NULL,
		"type='signal',"
		"sender='" SVC_DBUS_BUSNAME "',"
		"interface='" SVC_DBUS_INTERFACE ".Manager',"
		"member='JobRemoved',"
		"path='/org/freedesktop/systemd1'",
		match_job_removed, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add match for JobRemoved: %m");

	r = sd_bus_add_match(m->bus, NULL,
		"type='signal',"
		"sender='" SVC_DBUS_BUSNAME "',"
		"interface='" SVC_DBUS_INTERFACE ".Manager',"
		"member='UnitRemoved',"
		"path='/org/freedesktop/systemd1'",
		match_unit_removed, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add match for UnitRemoved: %m");

	r = sd_bus_add_match(m->bus, NULL,
		"type='signal',"
		"sender='" SVC_DBUS_BUSNAME "',"
		"interface='org.freedesktop.DBus.Properties',"
		"member='PropertiesChanged'",
		match_properties_changed, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add match for PropertiesChanged: %m");

	r = sd_bus_add_match(m->bus, NULL,
		"type='signal',"
		"sender='" SVC_DBUS_BUSNAME "',"
		"interface='" SVC_DBUS_INTERFACE ".Manager',"
		"member='Reloading',"
		"path='/org/freedesktop/systemd1'",
		match_reloading, m);
	if (r < 0)
		return log_error_errno(r,
			"Failed to add match for Reloading: %m");

	r = sd_bus_call_method(m->bus, SVC_DBUS_BUSNAME,
		"/org/freedesktop/systemd1", SVC_DBUS_INTERFACE ".Manager",
		"Subscribe", &error, NULL, NULL);
	if (r < 0) {
		log_error("Failed to enable subscription: %s",
			bus_error_message(&error, r));
		return r;
	}

	r = sd_bus_request_name(m->bus, SVC_MACHINED_DBUS_BUSNAME, 0);
	if (r < 0)
		return log_error_errno(r, "Failed to register name: %m");

	r = sd_bus_attach_event(m->bus, m->event, 0);
	if (r < 0)
		return log_error_errno(r,
			"Failed to attach bus to event loop: %m");

	return 0;
}

void
manager_gc(Manager *m, bool drop_not_started)
{
	Machine *machine;

	assert(m);

	while ((machine = m->machine_gc_queue)) {
		IWLIST_REMOVE(gc_queue, m->machine_gc_queue, machine);
		machine->in_gc_queue = false;

		if (!machine_check_gc(machine, drop_not_started)) {
			machine_stop(machine);
			machine_free(machine);
		}
	}
}

int
manager_startup(Manager *m)
{
	Machine *machine;
	Iterator i;
	int r;

	assert(m);

	/* Connect to the bus */
	r = manager_connect_bus(m);
	if (r < 0)
		return r;

	/* Deserialize state */
	manager_enumerate_machines(m);

	/* Remove stale objects before we start them */
	manager_gc(m, false);

	/* And start everything */
	HASHMAP_FOREACH (machine, m->machines, i)
		machine_start(machine, NULL, NULL);

	return 0;
}

static bool
check_idle(void *userdata)
{
	Manager *m = userdata;

	manager_gc(m, true);

	return hashmap_isempty(m->machines);
}

int
manager_run(Manager *m)
{
	assert(m);

	return bus_event_loop_with_idle(m->event, m->bus,
		SVC_MACHINED_DBUS_BUSNAME, DEFAULT_EXIT_USEC, check_idle, m);
}

int
main(int argc, char *argv[])
{
	Manager *m = NULL;
	int r;

	log_set_target(LOG_TARGET_AUTO);
	log_set_facility(LOG_AUTH);
	log_parse_environment();
	log_open();

	umask(0022);

	if (argc != 1) {
		log_error("This program takes no arguments.");
		r = -EINVAL;
		goto finish;
	}

	/* Always create the directories people can create inotify
         * watches in. Note that some applications might check for the
         * existence of $SVC_PKGRUNSTATEDIR/machines/ to determine whether
         * machined is available, so please always make sure this
         * check stays in. */
	mkdir_label(SVC_PKGRUNSTATEDIR "/machines", 0755);

	m = manager_new();
	if (!m) {
		r = log_oom();
		goto finish;
	}

	r = manager_startup(m);
	if (r < 0) {
		log_error_errno(r, "Failed to fully start up daemon: %m");
		goto finish;
	}

	log_debug("systemd-machined running as pid " PID_FMT, getpid());

	sd_notify(false,
		"READY=1\n"
		"STATUS=Processing requests...");

	r = manager_run(m);

	log_debug("systemd-machined stopped as pid " PID_FMT, getpid());

finish:
	if (m)
		manager_free(m);

	return r < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
