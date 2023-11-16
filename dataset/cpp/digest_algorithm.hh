/*
 * Copyright (C) 2016-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace query {

enum class digest_algorithm : uint8_t {
    none = 0,  // digest not required
    MD5 = 1,
    legacy_xxHash_without_null_digest = 2,
    xxHash = 3, // default algorithm
};

}
