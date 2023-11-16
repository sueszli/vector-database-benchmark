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
#include <bitcoin/system/error/error_t.hpp>

#include <bitcoin/system/define.hpp>
#include <bitcoin/system/error/macros.hpp>

namespace libbitcoin {
namespace system {
namespace error {
    
DEFINE_ERROR_T_MESSAGE_MAP(error)
{
    { success, "success" },
    { unknown, "unknown error" },
    { not_found, "object does not exist" },
    { not_implemented, "feature not implemented" }
    ////{ error_last, "unmapped code" }
};

DEFINE_ERROR_T_CATEGORY(error, "bc", "system code")

} // namespace error
} // namespace system
} // namespace libbitcoin
