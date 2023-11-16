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

#include "relay/session_key.h"

#include "base/crypto/generic_hash.h"
#include "base/crypto/random.h"

namespace relay {

//--------------------------------------------------------------------------------------------------
SessionKey::SessionKey() = default;

SessionKey::SessionKey(base::KeyPair&& key_pair, base::ByteArray&& iv)
    : key_pair_(std::move(key_pair)),
      iv_(std::move(iv))
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
SessionKey::SessionKey(SessionKey&& other) noexcept
    : key_pair_(std::move(other.key_pair_)),
      iv_(std::move(other.iv_))
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
SessionKey& SessionKey::operator=(SessionKey&& other) noexcept
{
    if (&other != this)
    {
        key_pair_ = std::move(other.key_pair_);
        iv_ = std::move(other.iv_);
    }

    return *this;
}

//--------------------------------------------------------------------------------------------------
SessionKey::~SessionKey() = default;

//--------------------------------------------------------------------------------------------------
// static
SessionKey SessionKey::create()
{
    base::KeyPair key_pair = base::KeyPair::create(base::KeyPair::Type::X25519);
    if (!key_pair.isValid())
        return SessionKey();

    base::ByteArray iv = base::Random::byteArray(12);
    if (iv.empty())
        return SessionKey();

    return SessionKey(std::move(key_pair), std::move(iv));
}

//--------------------------------------------------------------------------------------------------
bool SessionKey::isValid() const
{
    return key_pair_.isValid() && !iv_.empty();
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SessionKey::privateKey() const
{
    return key_pair_.privateKey();
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SessionKey::publicKey() const
{
    return key_pair_.publicKey();
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SessionKey::sessionKey(std::string_view peer_public_key) const
{
    base::ByteArray temp = key_pair_.sessionKey(base::fromStdString(peer_public_key));
    if (temp.empty())
        return base::ByteArray();

    return base::GenericHash::hash(base::GenericHash::Type::BLAKE2s256, temp);
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SessionKey::iv() const
{
    return iv_;
}

} // namespace relay
