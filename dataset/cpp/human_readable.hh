/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

/*
 * Copyright (C) 2020-present ScyllaDB
 */

#pragma once

#include <cinttypes>
#include <iosfwd>

namespace utils {

struct human_readable_value {
    uint16_t value;  // [0, 1024)
    char suffix; // 0 -> no suffix
};

std::ostream& operator<<(std::ostream& os, const human_readable_value& val);

/// Convert a size to a human readable representation.
///
/// The human-readable representation has at most 4 digits
/// and a letter appropriate to the power of two the number has to be multiplied
/// with to arrive to the original number (with some loss of precision).
/// The different powers of two are the conventional 2 ** (N * 10) variants:
/// * N=0: (B)ytes
/// * N=1: (K)bytes
/// * N=2: (M)bytes
/// * N=3: (G)bytes
/// * N=4: (T)bytes
///
/// Examples:
/// * 87665 will be converted to 87K
/// * 1024 will be converted to 1K
human_readable_value to_hr_size(uint64_t size);

} // namespace utils
