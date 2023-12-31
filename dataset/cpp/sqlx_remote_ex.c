/*
OpenIO SDS sqliterepo
Copyright (C) 2014 Worldline, as part of Redcurrant
Copyright (C) 2015-2019 OpenIO SAS, as part of OpenIO SDS
Copyright (C) 2021 OVH SAS

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library.
*/

#include <errno.h>
#include <stdlib.h>

#include <metautils/lib/metautils.h>
#include <metautils/lib/metacomm.h>
#include <sqliterepo/sqlite_utils.h>
#include <sqliterepo/sqlx_remote.h>
#include <sqliterepo/sqlx_remote_ex.h>

GError*
sqlx_remote_execute_RESYNC_many(gchar **targets, GByteArray *sid,
		const struct sqlx_name_s *name, gint64 deadline)
{
	(void) sid;
	GError *err = NULL;
	GByteArray *req = sqlx_pack_RESYNC(name, -1, deadline);
	struct gridd_client_s **clients = gridd_client_create_many(
			targets, req, NULL, NULL);
	metautils_gba_unref(req);
	req = NULL;

	if (clients == NULL) {
		err = NEWERROR(0, "Failed to create gridd clients");
		return err;
	}

	gridd_clients_set_timeout(clients,
			oio_clamp_timeout(10 * G_TIME_SPAN_SECOND, deadline));

	gridd_clients_start(clients);
	err = gridd_clients_loop(clients);

	for (struct gridd_client_s **p = clients; clients && *p; p++) {
		if ((err = gridd_client_error(*p))) {
			GRID_WARN("Database resync attempts failed: (%d) %s",
					err->code, err->message);
			g_clear_error(&err);
		}
	}

	gridd_clients_free(clients);
	return err;
}
