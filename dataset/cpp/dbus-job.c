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

#include "dbus-job.h"
#include "dbus.h"
#include "job.h"
#include "log.h"
#include "sd-bus.h"
#include "selinux-access.h"

static BUS_DEFINE_PROPERTY_GET_ENUM(property_get_type, job_type, JobType);
static BUS_DEFINE_PROPERTY_GET_ENUM(property_get_state, job_state, JobState);

static int
verify_sys_admin_or_owner_sync(sd_bus_message *message, Job *j,
	sd_bus_error *error)
{
	int r;

	if (sd_bus_track_contains(j->clients,
		    sd_bus_message_get_sender(message)))
		return 0; /* One of the job owners is calling us */

	r = sd_bus_query_sender_privilege(message, 0);
	if (r < 0)
		return r;
	if (r == 0)
		return sd_bus_error_setf(error, SD_BUS_ERROR_ACCESS_DENIED,
			"Access denied to perform action");

	/* Root has called us */
	return 0;
}

static int
property_get_unit(sd_bus *bus, const char *path, const char *interface,
	const char *property, sd_bus_message *reply, void *userdata,
	sd_bus_error *error)
{
	_cleanup_free_ char *p = NULL;
	Job *j = userdata;

	assert(bus);
	assert(reply);
	assert(j);

	p = unit_dbus_path(j->unit);
	if (!p)
		return -ENOMEM;

	return sd_bus_message_append(reply, "(so)", j->unit->id, p);
}

int
bus_job_method_cancel(sd_bus *bus, sd_bus_message *message, void *userdata,
	sd_bus_error *error)
{
	Job *j = userdata;
	int r;

	assert(bus);
	assert(message);
	assert(j);

	r = verify_sys_admin_or_owner_sync(message, j, error);
	if (r < 0)
		return r;

	r = mac_selinux_unit_access_check(j->unit, message, "stop", error);
	if (r < 0)
		return r;

	job_finish_and_invalidate(j, JOB_CANCELED, true, false);

	return sd_bus_reply_method_return(message, NULL);
}

const sd_bus_vtable bus_job_vtable[] = { SD_BUS_VTABLE_START(0),
	SD_BUS_METHOD("Cancel", NULL, NULL, bus_job_method_cancel,
		SD_BUS_VTABLE_UNPRIVILEGED),
	SD_BUS_PROPERTY("Id", "u", NULL, offsetof(Job, id),
		SD_BUS_VTABLE_PROPERTY_CONST),
	SD_BUS_PROPERTY("Unit", "(so)", property_get_unit, 0,
		SD_BUS_VTABLE_PROPERTY_CONST),
	SD_BUS_PROPERTY("JobType", "s", property_get_type, offsetof(Job, type),
		SD_BUS_VTABLE_PROPERTY_CONST),
	SD_BUS_PROPERTY("State", "s", property_get_state, offsetof(Job, state),
		SD_BUS_VTABLE_PROPERTY_EMITS_CHANGE),
	SD_BUS_VTABLE_END };

static int
send_new_signal(sd_bus *bus, void *userdata)
{
	_cleanup_bus_message_unref_ sd_bus_message *m = NULL;
	_cleanup_free_ char *p = NULL;
	Job *j = userdata;
	int r;

	assert(bus);
	assert(j);

	p = job_dbus_path(j);
	if (!p)
		return -ENOMEM;

	r = sd_bus_message_new_signal(bus, &m, "/org/freedesktop/systemd1",
		SVC_DBUS_INTERFACE ".Manager", "JobNew");
	if (r < 0)
		return r;

	r = sd_bus_message_append(m, "uos", j->id, p, j->unit->id);
	if (r < 0)
		return r;

	return sd_bus_send(bus, m, NULL);
}

static int
send_changed_signal(sd_bus *bus, void *userdata)
{
	_cleanup_free_ char *p = NULL;
	Job *j = userdata;

	assert(bus);
	assert(j);

	p = job_dbus_path(j);
	if (!p)
		return -ENOMEM;

	return sd_bus_emit_properties_changed(bus, p, SVC_DBUS_INTERFACE ".Job",
		"State", NULL);
}

void
bus_job_send_change_signal(Job *j)
{
	int r;

	assert(j);

	if (j->in_dbus_queue) {
		IWLIST_REMOVE(dbus_queue, j->manager->dbus_job_queue, j);
		j->in_dbus_queue = false;
	}

	r = bus_foreach_bus(j->manager, j->clients,
		j->sent_dbus_new_signal ? send_changed_signal : send_new_signal,
		j);
	if (r < 0)
		log_debug_errno(r,
			"Failed to send job change signal for %u: %m", j->id);

	j->sent_dbus_new_signal = true;
}

static int
send_removed_signal(sd_bus *bus, void *userdata)
{
	_cleanup_bus_message_unref_ sd_bus_message *m = NULL;
	_cleanup_free_ char *p = NULL;
	Job *j = userdata;
	int r;

	assert(bus);
	assert(j);

	p = job_dbus_path(j);
	if (!p)
		return -ENOMEM;

	r = sd_bus_message_new_signal(bus, &m, "/org/freedesktop/systemd1",
		SVC_DBUS_INTERFACE ".Manager", "JobRemoved");
	if (r < 0)
		return r;

	r = sd_bus_message_append(m, "uoss", j->id, p, j->unit->id,
		job_result_to_string(j->result));
	if (r < 0)
		return r;

	return sd_bus_send(bus, m, NULL);
}

void
bus_job_send_removed_signal(Job *j)
{
	int r;

	assert(j);

	if (!j->sent_dbus_new_signal)
		bus_job_send_change_signal(j);

	r = bus_foreach_bus(j->manager, j->clients, send_removed_signal, j);
	if (r < 0)
		log_debug_errno(r,
			"Failed to send job remove signal for %u: %m", j->id);
}
