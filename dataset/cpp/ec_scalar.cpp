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
#include <bitcoin/system/wallet/keys/ec_scalar.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <bitcoin/system/crypto/crypto.hpp>
#include <bitcoin/system/math/math.hpp>
#include <bitcoin/system/stream/stream.hpp>

namespace libbitcoin {
namespace system {

// construction
// ----------------------------------------------------------------------------

ec_scalar::ec_scalar() NOEXCEPT
  : secret_(null_hash)
{
}

ec_scalar::ec_scalar(ec_secret&& secret) NOEXCEPT
  : secret_(std::move(secret))
{
}

ec_scalar::ec_scalar(const ec_secret& secret) NOEXCEPT
  : secret_(secret)
{
}

ec_scalar::ec_scalar(int64_t value) NOEXCEPT
  : ec_scalar(from_int64(value))
{
}

// assignment operators
// ----------------------------------------------------------------------------

ec_scalar& ec_scalar::operator=(ec_secret&& secret) NOEXCEPT
{
    secret_ = std::move(secret);
    return *this;
}

ec_scalar& ec_scalar::operator=(const ec_secret& secret) NOEXCEPT
{
    secret_ = secret;
    return *this;
}

// private
ec_scalar ec_scalar::from_int64(int64_t value) NOEXCEPT
{
    // Short-circuit writing a zero.
    if (is_zero(value))
        return {};

    ec_secret secret = null_hash;
    write::bytes::copy writer(
    { 
        std::prev(secret.end(), sizeof(value)),
        secret.end()
    });

    // All hashes and secrets are stored as big-endian by convention.
    // absolute(minimum<int64_t>) overflows to zero, intended behavior.
    writer.write_8_bytes_big_endian(absolute(value));

    return value > 0 ? ec_scalar{ secret } : -ec_scalar{ secret };
}

// arithmetic assignment operators
// ----------------------------------------------------------------------------

ec_scalar& ec_scalar::operator+=(const ec_scalar& scalar) NOEXCEPT
{
    *this = (*this + scalar);
    return *this;
}

ec_scalar& ec_scalar::operator-=(const ec_scalar& scalar) NOEXCEPT
{
    *this = (*this - scalar);
    return *this;
}

ec_scalar& ec_scalar::operator*=(const ec_scalar& scalar) NOEXCEPT
{
    *this = (*this * scalar);
    return *this;
}

// unary operators (const)
// ----------------------------------------------------------------------------

ec_scalar ec_scalar::operator-() const NOEXCEPT
{
    auto out = secret_;
    if (!ec_negate(out))
        return {};
    
    return ec_scalar{ std::move(out) };
}

// binary math operators (const)
// ----------------------------------------------------------------------------

ec_scalar operator+(const ec_scalar& left, const ec_scalar& right) NOEXCEPT
{
    ec_secret out = left.secret();
    if (!ec_add(out, right.secret()))
        return {};
    
    return ec_scalar{ std::move(out) };
}

ec_scalar operator-(const ec_scalar& left, const ec_scalar& right) NOEXCEPT
{
    return left + -right;
}

ec_scalar operator*(const ec_scalar& left, const ec_scalar& right) NOEXCEPT
{
    auto out = left.secret();
    if (!ec_multiply(out, right.secret()))
        return {};
    
    return ec_scalar{ std::move(out) };
}

// comparison operators
// ----------------------------------------------------------------------------

bool operator==(int64_t left, const ec_scalar& right) NOEXCEPT
{
    return ec_scalar(left) == right;
}

bool operator!=(int64_t left, const ec_scalar& right) NOEXCEPT
{
    return !(left == right);
}

bool operator==(const ec_scalar& left, int64_t right) NOEXCEPT
{
    return right == left;
}

bool operator!=(const ec_scalar& left, int64_t right) NOEXCEPT
{
    return !(right == left);
}

bool operator==(const ec_scalar& left, const ec_scalar& right) NOEXCEPT
{
    // Compare arrays from left and right in reverse order since scalars are
    // encoded in big endian format, with leading bytes zero for small scalars.
    return std::equal(left.secret().rbegin(), left.secret().rend(),
        right.secret().rbegin());
}

bool operator!=(const ec_scalar& left, const ec_scalar& right) NOEXCEPT
{
    return !(left == right);
}

// cast operators
// ----------------------------------------------------------------------------

ec_scalar::operator bool() const NOEXCEPT
{
    // Use reverse comparison, typically only 1 byte compare if non-zero.
    return !((*this) == null_hash);
}

ec_scalar::operator const ec_secret&() const NOEXCEPT
{
    return secret_;
}

// properties
// ----------------------------------------------------------------------------

const ec_secret& ec_scalar::secret() const NOEXCEPT
{
    return secret_;
}

} // namespace system
} // namespace libbitcoin
