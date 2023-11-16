//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "crypto/password_hash.h"

#include "base/logging.h"

#include <openssl/evp.h>

namespace base {

namespace {

//--------------------------------------------------------------------------------------------------
template <typename InputT, typename OutputT>
OutputT hashT(PasswordHash::Type type, std::string_view password, InputT salt)
{
    DCHECK_EQ(type, PasswordHash::Type::SCRYPT);

    // CPU/Memory cost parameter, must be larger than 1, a power of 2, and less than 2^(128 * r / 8).
    static const uint64_t N = 16384;

    // Block size parameter.
    static const uint64_t r = 8;

    // Parallelization parameter, a positive integer less than or equal to ((2^32-1) * hLen) / MFLen
    // where hLen is 32 and MFlen is 128 * r.
    static const uint64_t p = 2;

    // 32MB
    static const uint64_t max_mem = 32 * 1024 * 1024;

    OutputT result;
    result.resize(PasswordHash::kBytesSize);

    int ret = EVP_PBE_scrypt(password.data(), password.size(),
                             reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
                             N, r, p, max_mem,
                             reinterpret_cast<uint8_t*>(result.data()), result.size());
    CHECK_EQ(ret, 1) << "EVP_PBE_scrypt failed";

    return result;
}

} // namespace

//--------------------------------------------------------------------------------------------------
// static
ByteArray PasswordHash::hash(Type type, std::string_view password, const ByteArray& salt)
{
    return hashT<const ByteArray, ByteArray>(type, password, salt);
}

//--------------------------------------------------------------------------------------------------
// static
std::string PasswordHash::hash(Type type, std::string_view password, std::string_view salt)
{
    return hashT<std::string_view, std::string>(type, password, salt);
}

} // namespace base
