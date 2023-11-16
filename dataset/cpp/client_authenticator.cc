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

#include "base/peer/client_authenticator.h"

#include "base/cpuid_util.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/sys_info.h"
#include "base/crypto/generic_hash.h"
#include "base/crypto/key_pair.h"
#include "base/crypto/random.h"
#include "base/crypto/srp_constants.h"
#include "base/crypto/srp_math.h"
#include "base/strings/unicode.h"

namespace base {

namespace {

const size_t kIvSize = 12; // 12 bytes.

//--------------------------------------------------------------------------------------------------
bool verifyNg(std::string_view N, std::string_view g)
{
    switch (N.size())
    {
        case 512: // 4096 bit
        {
            if (N != kSrpNgPair_4096.first || g != kSrpNgPair_4096.second)
                return false;
        }
        break;

        case 768: // 6144 bit
        {
            if (N != kSrpNgPair_6144.first || g != kSrpNgPair_6144.second)
                return false;
        }
        break;

        case 1024: // 8192 bit
        {
            if (N != kSrpNgPair_8192.first || g != kSrpNgPair_8192.second)
                return false;
        }
        break;

        // We do not allow groups less than 512 bytes (4096 bits).
        default:
            return false;
    }

    return true;
}

} // namespace

//--------------------------------------------------------------------------------------------------
ClientAuthenticator::ClientAuthenticator(std::shared_ptr<TaskRunner> task_runner)
    : Authenticator(std::move(task_runner))
{
    LOG(LS_INFO) << "Ctor";
}

//--------------------------------------------------------------------------------------------------
ClientAuthenticator::~ClientAuthenticator()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::setPeerPublicKey(const ByteArray& public_key)
{
    peer_public_key_ = public_key;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::setIdentify(proto::Identify identify)
{
    identify_ = identify;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::setUserName(std::u16string_view username)
{
    username_ = username;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::setPassword(std::u16string_view password)
{
    password_ = password;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::setSessionType(uint32_t session_type)
{
    session_type_ = session_type;
}

//--------------------------------------------------------------------------------------------------
bool ClientAuthenticator::onStarted()
{
    internal_state_ = InternalState::SEND_CLIENT_HELLO;
    sendClientHello();
    return true;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::onReceived(const ByteArray& buffer)
{
    switch (internal_state_)
    {
        case InternalState::READ_SERVER_HELLO:
        {
            if (readServerHello(buffer))
            {
                if (identify_ == proto::IDENTIFY_ANONYMOUS)
                {
                    internal_state_ = InternalState::READ_SESSION_CHALLENGE;
                }
                else
                {
                    internal_state_ = InternalState::SEND_IDENTIFY;
                    sendIdentify();
                }
            }
        }
        break;

        case InternalState::READ_SERVER_KEY_EXCHANGE:
        {
            if (readServerKeyExchange(buffer))
            {
                internal_state_ = InternalState::SEND_CLIENT_KEY_EXCHANGE;
                sendClientKeyExchange();
            }
        }
        break;

        case InternalState::READ_SESSION_CHALLENGE:
        {
            if (readSessionChallenge(buffer))
            {
                internal_state_ = InternalState::SEND_SESSION_RESPONSE;
                sendSessionResponse();
            }
        }
        break;

        default:
        {
            NOTREACHED();
        }
        break;
    }
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::onWritten()
{
    switch (internal_state_)
    {
        case InternalState::SEND_CLIENT_HELLO:
        {
            LOG(LS_INFO) << "Sended: ClientHello";
            internal_state_ = InternalState::READ_SERVER_HELLO;
        }
        break;

        case InternalState::SEND_IDENTIFY:
        {
            LOG(LS_INFO) << "Sended: Identify";
            internal_state_ = InternalState::READ_SERVER_KEY_EXCHANGE;
        }
        break;

        case InternalState::SEND_CLIENT_KEY_EXCHANGE:
        {
            LOG(LS_INFO) << "Sended: ClientKeyExchange";
            internal_state_ = InternalState::READ_SESSION_CHALLENGE;
            if (!onSessionKeyChanged())
            {
                finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
                return;
            }
        }
        break;

        case InternalState::SEND_SESSION_RESPONSE:
        {
            LOG(LS_INFO) << "Sended: SessionResponse";
            finish(FROM_HERE, ErrorCode::SUCCESS);
        }
        break;

        default:
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::sendClientHello()
{
    // We do not allow anonymous connections without a public key.
    if (identify_ == proto::IDENTIFY_ANONYMOUS && peer_public_key_.empty())
    {
        finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
        return;
    }

    std::unique_ptr<proto::ClientHello> client_hello = std::make_unique<proto::ClientHello>();

    uint32_t encryption = proto::ENCRYPTION_CHACHA20_POLY1305;

#if defined(ARCH_CPU_X86_FAMILY)
    if (CpuidUtil::hasAesNi())
        encryption |= proto::ENCRYPTION_AES256_GCM;
#endif

    client_hello->set_encryption(encryption);
    client_hello->set_identify(identify_);

    if (!peer_public_key_.empty())
    {
        encrypt_iv_ = Random::byteArray(kIvSize);
        if (encrypt_iv_.empty())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }

        KeyPair key_pair = KeyPair::create(KeyPair::Type::X25519);
        if (!key_pair.isValid())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }

        ByteArray temp = key_pair.sessionKey(peer_public_key_);
        if (temp.empty())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }

        session_key_ = GenericHash::hash(GenericHash::Type::BLAKE2s256, temp);
        if (session_key_.empty())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }

        ByteArray public_key = key_pair.publicKey();
        if (public_key.empty())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }

        client_hello->set_public_key(toStdString(public_key));
        client_hello->set_iv(toStdString(encrypt_iv_));
    }

    LOG(LS_INFO) << "Sending: ClientHello";
    sendMessage(*client_hello);
}

//--------------------------------------------------------------------------------------------------
bool ClientAuthenticator::readServerHello(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: ServerHello";

    std::unique_ptr<proto::ServerHello> server_hello = std::make_unique<proto::ServerHello>();
    if (!parse(buffer, server_hello.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    LOG(LS_INFO) << "Encryption: " << server_hello->encryption();

    encryption_ = server_hello->encryption();
    switch (encryption_)
    {
        case proto::ENCRYPTION_AES256_GCM:
        case proto::ENCRYPTION_CHACHA20_POLY1305:
            break;

        default:
            finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
            return false;
    }

    decrypt_iv_ = fromStdString(server_hello->iv());

    if (session_key_.empty() != decrypt_iv_.empty())
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    if (!session_key_.empty() && !onSessionKeyChanged())
    {
        finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::sendIdentify()
{
    std::unique_ptr<proto::SrpIdentify> identify = std::make_unique<proto::SrpIdentify>();
    identify->set_username(utf8FromUtf16(username_));

    LOG(LS_INFO) << "Sending: Identify";
    sendMessage(*identify);
}

//--------------------------------------------------------------------------------------------------
bool ClientAuthenticator::readServerKeyExchange(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: ServerKeyExchange";

    std::unique_ptr<proto::SrpServerKeyExchange> server_key_exchange =
        std::make_unique<proto::SrpServerKeyExchange>();
    if (!parse(buffer, server_key_exchange.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    if (server_key_exchange->salt().empty() || server_key_exchange->b().empty())
    {
        LOG(LS_ERROR) << "Salt size: " << server_key_exchange->salt().size();
        LOG(LS_ERROR) << "B size: " << server_key_exchange->b().size();

        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    if (!verifyNg(server_key_exchange->number(), server_key_exchange->generator()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    N_ = BigNum::fromStdString(server_key_exchange->number());
    g_ = BigNum::fromStdString(server_key_exchange->generator());
    s_ = BigNum::fromStdString(server_key_exchange->salt());
    B_ = BigNum::fromStdString(server_key_exchange->b());
    decrypt_iv_ = fromStdString(server_key_exchange->iv());

    a_ = BigNum::fromByteArray(Random::byteArray(128)); // 1024 bits.
    A_ = SrpMath::calc_A(a_, N_, g_);
    encrypt_iv_ = Random::byteArray(kIvSize);

    if (!SrpMath::verify_B_mod_N(B_, N_))
    {
        LOG(LS_ERROR) << "Invalid B or N";
        return false;
    }

    BigNum u = SrpMath::calc_u(A_, B_, N_);
    BigNum x = SrpMath::calc_x(s_, username_, password_);
    BigNum key = SrpMath::calcClientKey(N_, B_, g_, x, a_, u);
    if (!key.isValid())
    {
        LOG(LS_ERROR) << "Empty encryption key generated";
        return false;
    }

    // AES256-GCM and ChaCha20-Poly1305 requires 256 bit key.
    GenericHash hash(GenericHash::BLAKE2s256);

    if (!session_key_.empty())
        hash.addData(session_key_);
    hash.addData(key.toByteArray());

    session_key_ = hash.result();
    return true;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::sendClientKeyExchange()
{
    std::unique_ptr<proto::SrpClientKeyExchange> client_key_exchange =
        std::make_unique<proto::SrpClientKeyExchange>();
    client_key_exchange->set_a(A_.toStdString());
    client_key_exchange->set_iv(toStdString(encrypt_iv_));

    LOG(LS_INFO) << "Sending: ClientKeyExchange";
    sendMessage(*client_key_exchange);
}

//--------------------------------------------------------------------------------------------------
bool ClientAuthenticator::readSessionChallenge(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: SessionChallenge";

    std::unique_ptr<proto::SessionChallenge> challenge =
        std::make_unique<proto::SessionChallenge>();
    if (!parse(buffer, challenge.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return false;
    }

    if (!(challenge->session_types() & session_type_))
    {
        finish(FROM_HERE, ErrorCode::SESSION_DENIED);
        return false;
    }

    setPeerVersion(challenge->version());
    setPeerOsName(challenge->os_name());
    setPeerComputerName(challenge->computer_name());

    LOG(LS_INFO) << "Server Version: " << peerVersion();
    LOG(LS_INFO) << "Server Name: " << challenge->computer_name();
    LOG(LS_INFO) << "Server OS: " << challenge->os_name();
    LOG(LS_INFO) << "Server CPU Cores: " << challenge->cpu_cores();
    LOG(LS_INFO) << "Server Arch: " << challenge->arch();

    return true;
}

//--------------------------------------------------------------------------------------------------
void ClientAuthenticator::sendSessionResponse()
{
    std::unique_ptr<proto::SessionResponse> response = std::make_unique<proto::SessionResponse>();
    response->set_session_type(session_type_);

    proto::Version* version = response->mutable_version();
    version->set_major(ASPIA_VERSION_MAJOR);
    version->set_minor(ASPIA_VERSION_MINOR);
    version->set_patch(ASPIA_VERSION_PATCH);
    version->set_revision(GIT_COMMIT_COUNT);

    response->set_os_name(SysInfo::operatingSystemName());
    response->set_computer_name(SysInfo::computerName());
    response->set_cpu_cores(static_cast<uint32_t>(SysInfo::processorThreads()));

#if defined(ARCH_CPU_X86)
    response->set_arch("x86");
#elif defined(ARCH_CPU_X86_64)
    response->set_arch("x86_64");
#elif defined(ARCH_CPU_ARMEL)
    response->set_arch("arm");
#elif defined(ARCH_CPU_ARM64)
    response->set_arch("arm64");
#else
    response->set_arch(std::string());
#endif

    LOG(LS_INFO) << "Sending: SessionResponse";
    sendMessage(*response);
}

} // namespace base
