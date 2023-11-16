/*
 * Copyright (C) 2018-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <string_view>
#include <functional>
#include <iosfwd>
#include <optional>

#include <seastar/core/sstring.hh>

#include "seastarx.hh"

namespace auth {

class role_or_anonymous final {
public:
    std::optional<sstring> name{};

    role_or_anonymous() = default;
    role_or_anonymous(std::string_view name) : name(name) {
    }
    friend bool operator==(const role_or_anonymous&, const role_or_anonymous&) noexcept = default;
};

std::ostream& operator<<(std::ostream&, const role_or_anonymous&);

bool is_anonymous(const role_or_anonymous&) noexcept;

}

namespace std {

template <>
struct hash<auth::role_or_anonymous> {
    size_t operator()(const auth::role_or_anonymous& mr) const {
        return hash<std::optional<sstring>>()(mr.name);
    }
};

}
