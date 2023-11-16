/*
 * Copyright 2019-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <string>
#include <string_view>
#include <array>
#include "gc_clock.hh"
#include "utils/loading_cache.hh"

namespace service {
class storage_proxy;
}

namespace alternator {

using key_cache = utils::loading_cache<std::string, std::string, 1>;

future<std::string> get_key_from_roles(service::storage_proxy& proxy, std::string username);

}
