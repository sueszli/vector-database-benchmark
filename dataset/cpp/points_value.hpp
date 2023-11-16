/**
 * Copyright (c) 2011-2023 libbitcoin developers (see AUTHORS)
 *
 * This file is part of libbitcoin.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef LIBBITCOIN_SYSTEM_WALLET_POINTS_VALUE_HPP
#define LIBBITCOIN_SYSTEM_WALLET_POINTS_VALUE_HPP

#include <numeric>
#include <bitcoin/system/define.hpp>
#include <bitcoin/system/wallet/point_value.hpp>

namespace libbitcoin {
namespace system {
namespace wallet {

class BC_API points_value
{
public:
    enum class selection
    {
        /// The smallest single sufficient unspent output, if one exists, or a
        /// sufficient set of unspent outputs, if such a set exists. The set is
        /// minimal by number of outputs but not necessarily by total value.
        greedy,

        /// A set of individually sufficient unspent outputs. Each individual
        /// member of the set is sufficient. Return ascending order by value.
        individual
    };

    /// Select outpoints for a spend from a list of unspent outputs.
    static void select(points_value& out, const points_value& unspent,
        uint64_t minimum_value, selection option=selection::greedy) NOEXCEPT;

    /// Total value of the current set of points.
    uint64_t value() const NOEXCEPT;

    /// A set of valued points.
    point_value::list points{};

private:
    static void greedy(points_value& out, const points_value& unspent,
        uint64_t minimum_value) NOEXCEPT;

    static void individual(points_value& out, const points_value& unspent,
        uint64_t minimum_value) NOEXCEPT;
};

} // namespace chain
} // namespace system
} // namespace libbitcoin

#endif
