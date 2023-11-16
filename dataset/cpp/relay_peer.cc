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

#include "base/peer/relay_peer.h"

#include "base/endian_util.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/crypto/generic_hash.h"
#include "base/crypto/key_pair.h"
#include "base/crypto/message_encryptor_openssl.h"
#include "base/message_loop/message_loop.h"
#include "base/message_loop/message_pump_asio.h"
#include "base/net/tcp_channel.h"
#include "base/strings/unicode.h"
#include "proto/relay_peer.pb.h"

#include <asio/connect.hpp>
#include <asio/write.hpp>

namespace base {

namespace {

std::string endpointsToString(const asio::ip::tcp::resolver::results_type& endpoints)
{
    std::string str;

    for (auto it = endpoints.begin(); it != endpoints.end();)
    {
        str += it->endpoint().address().to_string();
        if (++it != endpoints.end())
            str += ", ";
    }

    return str;
}

} // namespace

//--------------------------------------------------------------------------------------------------
RelayPeer::RelayPeer()
    : io_context_(MessageLoop::current()->pumpAsio()->ioContext()),
      socket_(io_context_),
      resolver_(io_context_)
{
    LOG(LS_INFO) << "Ctor";
}

//--------------------------------------------------------------------------------------------------
RelayPeer::~RelayPeer()
{
    LOG(LS_INFO) << "Dtor";
    delegate_ = nullptr;

    std::error_code ignored_code;
    socket_.cancel(ignored_code);
    socket_.close(ignored_code);
}

//--------------------------------------------------------------------------------------------------
void RelayPeer::start(const proto::ConnectionOffer& offer, Delegate* delegate)
{
    delegate_ = delegate;
    connection_offer_ = offer;

    DCHECK(delegate_);

    const proto::RelayCredentials& credentials = connection_offer_.relay();

    message_ = authenticationMessage(credentials.key(), credentials.secret());

    LOG(LS_INFO) << "Start resolving for " << credentials.host() << ":" << credentials.port();

    resolver_.async_resolve(local8BitFromUtf16(utf16FromUtf8(credentials.host())),
                            std::to_string(credentials.port()),
        [this](const std::error_code& error_code,
               const asio::ip::tcp::resolver::results_type& endpoints)
    {
        if (error_code)
        {
            if (error_code != asio::error::operation_aborted)
                onErrorOccurred(FROM_HERE, error_code);
            return;
        }

        LOG(LS_INFO) << "Resolved endpoints: " << endpointsToString(endpoints);
        LOG(LS_INFO) << "Start connecting...";

        asio::async_connect(socket_, endpoints,
                            [this](const std::error_code& error_code,
                                   const asio::ip::tcp::endpoint& endpoint)
        {
            if (error_code)
            {
                if (error_code != asio::error::operation_aborted)
                {
                    onErrorOccurred(FROM_HERE, error_code);
                }
                else
                {
                    LOG(LS_ERROR) << "Operation aborted";
                }
                return;
            }

            LOG(LS_INFO) << "Connected to: " << endpoint.address().to_string();
            onConnected();
        });
    });
}

//--------------------------------------------------------------------------------------------------
void RelayPeer::onConnected()
{
    if (message_.empty())
    {
        onErrorOccurred(FROM_HERE, std::error_code());
        return;
    }

    message_size_ = base::EndianUtil::toBig(static_cast<uint32_t>(message_.size()));

    asio::async_write(socket_, asio::const_buffer(&message_size_, sizeof(message_size_)),
        [this](const std::error_code& error_code, size_t bytes_transferred)
    {
        if (error_code)
        {
            if (error_code != asio::error::operation_aborted)
            {
                onErrorOccurred(FROM_HERE, error_code);
            }
            else
            {
                LOG(LS_ERROR) << "Operation aborted";
            }
            return;
        }

        if (bytes_transferred != sizeof(message_size_))
        {
            onErrorOccurred(FROM_HERE, std::error_code());
            return;
        }

        asio::async_write(socket_, asio::const_buffer(message_.data(), message_.size()),
                          [this](const std::error_code& error_code, size_t bytes_transferred)
        {
            if (error_code)
            {
                if (error_code != asio::error::operation_aborted)
                {
                    onErrorOccurred(FROM_HERE, error_code);
                }
                else
                {
                    LOG(LS_ERROR) << "Operation aborted";
                }
                return;
            }

            if (bytes_transferred != message_.size())
            {
                onErrorOccurred(FROM_HERE, std::error_code());
                return;
            }

            is_finished_ = true;
            if (delegate_)
            {
                std::unique_ptr<TcpChannel> channel =
                    std::unique_ptr<TcpChannel>(new TcpChannel(std::move(socket_)));
                channel->setHostId(connection_offer_.host_data().host_id());

                delegate_->onRelayConnectionReady(std::move(channel));
            }
            else
            {
                LOG(LS_ERROR) << "Invalid delegate";
            }
        });
    });
}

//--------------------------------------------------------------------------------------------------
void RelayPeer::onErrorOccurred(const Location& location, const std::error_code& error_code)
{
    LOG(LS_ERROR) << "Failed to connect to relay server: "
                  << utf16FromLocal8Bit(error_code.message()) << " ("
                  << location.toString() << ")";

    is_finished_ = true;
    if (delegate_)
    {
        delegate_->onRelayConnectionError();
    }
    else
    {
        LOG(LS_ERROR) << "Invalid delegate";
    }
}

//--------------------------------------------------------------------------------------------------
// static
ByteArray RelayPeer::authenticationMessage(const proto::RelayKey& key, const std::string& secret)
{
    if (key.type() != proto::RelayKey::TYPE_X25519)
    {
        LOG(LS_ERROR) << "Unsupported key type: " << key.type();
        return ByteArray();
    }

    if (key.encryption() != proto::RelayKey::ENCRYPTION_CHACHA20_POLY1305)
    {
        LOG(LS_ERROR) << "Unsupported encryption type: " << key.encryption();
        return ByteArray();
    }

    if (key.public_key().empty())
    {
        LOG(LS_ERROR) << "Empty public key";
        return ByteArray();
    }

    if (key.iv().empty())
    {
        LOG(LS_ERROR) << "Empty IV";
        return ByteArray();
    }

    if (secret.empty())
    {
        LOG(LS_ERROR) << "Empty secret";
        return ByteArray();
    }

    KeyPair key_pair = KeyPair::create(KeyPair::Type::X25519);
    if (!key_pair.isValid())
    {
        LOG(LS_ERROR) << "KeyPair::create failed";
        return ByteArray();
    }

    ByteArray temp = key_pair.sessionKey(fromStdString(key.public_key()));
    if (temp.empty())
    {
        LOG(LS_ERROR) << "Failed to create session key";
        return ByteArray();
    }

    ByteArray session_key = base::GenericHash::hash(base::GenericHash::Type::BLAKE2s256, temp);

    std::unique_ptr<MessageEncryptor> encryptor =
        MessageEncryptorOpenssl::createForChaCha20Poly1305(session_key, fromStdString(key.iv()));
    if (!encryptor)
    {
        LOG(LS_ERROR) << "createForChaCha20Poly1305 failed";
        return ByteArray();
    }

    std::string encrypted_secret;
    encrypted_secret.resize(encryptor->encryptedDataSize(secret.size()));
    if (!encryptor->encrypt(secret.data(), secret.size(), encrypted_secret.data()))
    {
        LOG(LS_ERROR) << "encrypt failed";
        return ByteArray();
    }

    proto::PeerToRelay message;

    message.set_key_id(key.key_id());
    message.set_public_key(base::toStdString(key_pair.publicKey()));
    message.set_data(std::move(encrypted_secret));

    return serialize(message);
}

} // namespace base
