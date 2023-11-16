/*
 * Copyright (C) 2018-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <fmt/format.h>

namespace streaming {

enum class stream_reason : uint8_t {
    unspecified,
    bootstrap,
    decommission,
    removenode,
    rebuild,
    repair,
    replace,
    tablet_migration,
};

}

template <>
struct fmt::formatter<streaming::stream_reason> : fmt::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const streaming::stream_reason& r, FormatContext& ctx) const {
        using enum streaming::stream_reason;
        switch (r) {
        case unspecified:
            return formatter<std::string_view>::format("unspecified", ctx);
        case bootstrap:
            return formatter<std::string_view>::format("bootstrap", ctx);
        case decommission:
            return formatter<std::string_view>::format("decommission", ctx);
        case removenode:
            return formatter<std::string_view>::format("removenode", ctx);
        case rebuild:
            return formatter<std::string_view>::format("rebuild", ctx);
        case repair:
            return formatter<std::string_view>::format("repair", ctx);
        case replace:
            return formatter<std::string_view>::format("replace", ctx);
        case tablet_migration:
            return formatter<std::string_view>::format("tablet migration", ctx);
        }
        std::abort();
    }
};
