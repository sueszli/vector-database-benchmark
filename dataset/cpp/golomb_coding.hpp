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

#ifndef LIBBITCOIN_SYSTEM_CRYPTO_GOLOMB_CODING
#define LIBBITCOIN_SYSTEM_CRYPTO_GOLOMB_CODING

#include <istream>
#include <ostream>
#include <bitcoin/system/data/data.hpp>
#include <bitcoin/system/define.hpp>
#include <bitcoin/system/hash/hash.hpp>

namespace libbitcoin {
namespace system {
namespace golomb {

// Golomb-coded set construction
// ----------------------------------------------------------------------------

BC_API data_chunk construct(const data_stack& items, uint8_t bits,
    const half_hash& entropy, uint64_t target_false_positive_rate) NOEXCEPT;

BC_API data_chunk construct(const data_stack& items, uint8_t bits,
    const siphash_key& entropy, uint64_t target_false_positive_rate) NOEXCEPT;

BC_API void construct(std::ostream& stream, const data_stack& items,
    uint8_t bits, const half_hash& entropy,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API void construct(std::ostream& stream, const data_stack& items,
    uint8_t bits, const siphash_key& entropy,
    uint64_t target_false_positive_rate) NOEXCEPT;

// Single element match
// ----------------------------------------------------------------------------

BC_API bool match(const data_chunk& target, const data_chunk& compressed_set,
    uint64_t set_size, const half_hash& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_chunk& target, const data_chunk& compressed_set,
    uint64_t set_size, const siphash_key& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_chunk& target, std::istream& compressed_set,
    uint64_t set_size, const half_hash& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_chunk& target, std::istream& compressed_set,
    uint64_t set_size, const siphash_key& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

// Intersection match
// ----------------------------------------------------------------------------

BC_API bool match(const data_stack& targets, const data_chunk& compressed_set,
    uint64_t set_size, const half_hash& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_stack& targets, const data_chunk& compressed_set,
    uint64_t set_size, const siphash_key& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_stack& targets, std::istream& compressed_set,
    uint64_t set_size, const half_hash& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

BC_API bool match(const data_stack& targets, std::istream& compressed_set,
    uint64_t set_size, const siphash_key& entropy, uint8_t bits,
    uint64_t target_false_positive_rate) NOEXCEPT;

} // namespace golomb
} // namespace system
} // namespace libbitcoin

#endif
