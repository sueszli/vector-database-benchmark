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

// Sponsored in part by Digital Contract Design, LLC

#ifndef LIBBITCOIN_SYSTEM_WALLET_NEUTRINO_FILTER_HPP
#define LIBBITCOIN_SYSTEM_WALLET_NEUTRINO_FILTER_HPP

#include <istream>
#include <memory>
#include <bitcoin/system/data/data.hpp>
#include <bitcoin/system/define.hpp>
#include <bitcoin/system/chain/chain.hpp>
#include <bitcoin/system/wallet/addresses/payment_address.hpp>

namespace libbitcoin {
namespace system {
namespace neutrino {

struct BC_API block_filter
{
    hash_digest hash;
    data_chunk filter;
};

bool BC_API compute_filter(const chain::block& block,
    data_chunk& out_filter) NOEXCEPT;

hash_digest BC_API compute_filter_header(const hash_digest& previous_block,
    const data_chunk& filter) NOEXCEPT;

bool BC_API match_filter(const block_filter& filter,
    const chain::script& script) NOEXCEPT;

bool BC_API match_filter(const block_filter& filter,
    const chain::scripts& scripts) NOEXCEPT;

bool BC_API match_filter(const block_filter& filter,
    const wallet::payment_address& address) NOEXCEPT;

bool BC_API match_filter(const block_filter& filter,
    const wallet::payment_address::list& addresses) NOEXCEPT;

} // namespace neutrino
} // namespace system
} // namespace libbitcoin

#endif
