/*
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include "api.hh"

namespace netw { class messaging_service; }

namespace api {

void set_messaging_service(http_context& ctx, httpd::routes& r, sharded<netw::messaging_service>& ms);
void unset_messaging_service(http_context& ctx, httpd::routes& r);

}
