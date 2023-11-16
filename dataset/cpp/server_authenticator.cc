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

#include "base/peer/server_authenticator.h"

#include "base/bitset.h"
#include "base/cpuid_util.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/sys_info.h"
#include "base/crypto/generic_hash.h"
#include "base/crypto/random.h"
#include "base/crypto/srp_constants.h"
#include "base/crypto/srp_math.h"
#include "base/peer/user_list.h"
#include "base/strings/unicode.h"

namespace base {

namespace {

constexpr size_t kIvSize = 12;

//--------------------------------------------------------------------------------------------------
const char* identifyToString(proto::Identify identify)
{
    switch (identify)
    {
        case proto::IDENTIFY_ANONYMOUS:
            return "IDENTIFY_ANONYMOUS";

        case proto::IDENTIFY_SRP:
            return "IDENTIFY_SRP";

        default:
            return "UNKNOWN";
    }
}

} // namespace

//--------------------------------------------------------------------------------------------------
ServerAuthenticator::ServerAuthenticator(std::shared_ptr<TaskRunner> task_runner)
    : Authenticator(std::move(task_runner))
{
    LOG(LS_INFO) << "Ctor";
}

//--------------------------------------------------------------------------------------------------
ServerAuthenticator::~ServerAuthenticator()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::setUserList(base::local_shared_ptr<UserListBase> user_list)
{
    user_list_ = std::move(user_list);
    DCHECK(user_list_);
}

//--------------------------------------------------------------------------------------------------
bool ServerAuthenticator::setPrivateKey(const ByteArray& private_key)
{
    // The method must be called before calling start().
    if (state() != State::STOPPED)
        return false;

    if (private_key.empty())
    {
        LOG(LS_ERROR) << "An empty private key is not valid";
        return false;
    }

    key_pair_ = KeyPair::fromPrivateKey(private_key);
    if (!key_pair_.isValid())
    {
        LOG(LS_ERROR) << "Failed to load private key. Perhaps the key is incorrect";
        return false;
    }

    encrypt_iv_ = Random::byteArray(kIvSize);
    if (encrypt_iv_.empty())
    {
        LOG(LS_ERROR) << "An empty IV is not valid";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool ServerAuthenticator::setAnonymousAccess(
    AnonymousAccess anonymous_access, uint32_t session_types)
{
    // The method must be called before calling start().
    if (state() != State::STOPPED)
        return false;

    if (anonymous_access == AnonymousAccess::ENABLE)
    {
        if (!key_pair_.isValid())
        {
            LOG(LS_ERROR) << "When anonymous access is enabled, a private key must be installed";
            return false;
        }

        if (!session_types)
        {
            LOG(LS_ERROR) << "When anonymous access is enabled, there must be at least one "
                          << "session for anonymous access";
            return false;
        }

        session_types_ = session_types;
    }
    else
    {
        session_types_ = 0;
    }

    anonymous_access_ = anonymous_access;
    return true;
}

//--------------------------------------------------------------------------------------------------
bool ServerAuthenticator::onStarted()
{
    internal_state_ = InternalState::READ_CLIENT_HELLO;

    // We do not allow anonymous access without a private key.
    if (anonymous_access_ == AnonymousAccess::ENABLE && !key_pair_.isValid())
    {
        finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
        return false;
    }

    if (anonymous_access_ == AnonymousAccess::ENABLE)
    {
        // When anonymous access is enabled, a private key must be installed.
        if (!key_pair_.isValid())
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return false;
        }

        // When anonymous access is enabled, there must be at least one session for anonymous access.
        if (!session_types_)
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return false;
        }
    }
    else
    {
        // If anonymous access is disabled, then there should not be allowed sessions by default.
        if (session_types_)
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return false;
        }
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onReceived(const ByteArray& buffer)
{
    switch (internal_state_)
    {
        case InternalState::READ_CLIENT_HELLO:
            onClientHello(buffer);
            break;

        case InternalState::READ_IDENTIFY:
            onIdentify(buffer);
            break;

        case InternalState::READ_CLIENT_KEY_EXCHANGE:
            onClientKeyExchange(buffer);
            break;

        case InternalState::READ_SESSION_RESPONSE:
            onSessionResponse(buffer);
            break;

        default:
            NOTREACHED();
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onWritten()
{
    switch (internal_state_)
    {
        case InternalState::SEND_SERVER_HELLO:
        {
            LOG(LS_INFO) << "Sended: ServerHello";

            if (!session_key_.empty())
            {
                if (!onSessionKeyChanged())
                    return;
            }

            switch (identify_)
            {
                case proto::IDENTIFY_SRP:
                {
                    internal_state_ = InternalState::READ_IDENTIFY;
                }
                break;

                case proto::IDENTIFY_ANONYMOUS:
                {
                    internal_state_ = InternalState::SEND_SESSION_CHALLENGE;
                    doSessionChallenge();
                }
                break;

                default:
                    NOTREACHED();
                    break;
            }
        }
        break;

        case InternalState::SEND_SERVER_KEY_EXCHANGE:
        {
            LOG(LS_INFO) << "Sended: ServerKeyExchange";
            internal_state_ = InternalState::READ_CLIENT_KEY_EXCHANGE;
        }
        break;

        case InternalState::SEND_SESSION_CHALLENGE:
        {
            LOG(LS_INFO) << "Sended: SessionChallenge";
            internal_state_ = InternalState::READ_SESSION_RESPONSE;
        }
        break;

        default:
            NOTREACHED();
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onClientHello(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: ClientHello";

    std::unique_ptr<proto::ClientHello> client_hello = std::make_unique<proto::ClientHello>();
    if (!parse(buffer, client_hello.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    const uint32_t encryption = client_hello->encryption();

    LOG(LS_INFO) << "Supported by client:";

    if (encryption & proto::ENCRYPTION_AES256_GCM)
        LOG(LS_INFO) << "ENCRYPTION_AES256_GCM";

    if (encryption & proto::ENCRYPTION_CHACHA20_POLY1305)
        LOG(LS_INFO) << "ENCRYPTION_CHACHA20_POLY1305";

    LOG(LS_INFO) << "Identify: " << identifyToString(client_hello->identify());

    if (!(encryption & proto::ENCRYPTION_AES256_GCM) &&
        !(encryption & proto::ENCRYPTION_CHACHA20_POLY1305))
    {
        // No encryption methods supported.
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    identify_ = client_hello->identify();
    switch (identify_)
    {
        // SRP is always supported.
        case proto::IDENTIFY_SRP:
            break;

        case proto::IDENTIFY_ANONYMOUS:
        {
            // If anonymous method is not allowed.
            if (anonymous_access_ != AnonymousAccess::ENABLE)
            {
                finish(FROM_HERE, ErrorCode::ACCESS_DENIED);
                return;
            }
        }
        break;

        default:
        {
            // Unsupported identication method.
            finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
            return;
        }
    }

    std::unique_ptr<proto::ServerHello> server_hello = std::make_unique<proto::ServerHello>();

    if (key_pair_.isValid())
    {
        ByteArray peer_public_key = fromStdString(client_hello->public_key());
        decrypt_iv_ = fromStdString(client_hello->iv());

        if (peer_public_key.empty() != decrypt_iv_.empty())
        {
            finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
            return;
        }

        if (!peer_public_key.empty() && !decrypt_iv_.empty())
        {
            ByteArray temp = key_pair_.sessionKey(peer_public_key);
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

            DCHECK(!encrypt_iv_.empty());
            server_hello->set_iv(toStdString(encrypt_iv_));
        }
    }

    bool has_aes_ni = false;

#if defined(ARCH_CPU_X86_FAMILY)
    has_aes_ni = CpuidUtil::hasAesNi();
#endif

    if ((encryption & proto::ENCRYPTION_AES256_GCM) && has_aes_ni)
    {
        LOG(LS_INFO) << "Both sides have hardware support AES. Using AES256 GCM";
        // If both sides of the connection support AES, then method AES256 GCM is the fastest option.
        server_hello->set_encryption(proto::ENCRYPTION_AES256_GCM);
    }
    else
    {
        LOG(LS_INFO) << "Using ChaCha20+Poly1305";
        // Otherwise, we use ChaCha20+Poly1305. This works faster in the absence of hardware
        // support AES.
        server_hello->set_encryption(proto::ENCRYPTION_CHACHA20_POLY1305);
    }

    // Now we are in the authentication phase.
    internal_state_ = InternalState::SEND_SERVER_HELLO;
    encryption_ = server_hello->encryption();

    LOG(LS_INFO) << "Sending: ServerHello";
    sendMessage(*server_hello);
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onIdentify(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: Identify";

    std::unique_ptr<proto::SrpIdentify> identify = std::make_unique<proto::SrpIdentify>();
    if (!parse(buffer, identify.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    user_name_ = std::move(*identify->mutable_username());
    if (user_name_.empty())
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    LOG(LS_INFO) << "Username: '" << user_name_ << "'";

    do
    {
        std::u16string user_name_utf16 = base::utf16FromUtf8(user_name_);
        ByteArray seed_key;
        User user;

        if (user_list_)
        {
            user = user_list_->find(user_name_utf16);
            seed_key = user_list_->seedKey();
        }
        else
        {
            LOG(LS_INFO) << "UserList is nullptr";
        }

        if (seed_key.empty())
        {
            LOG(LS_INFO) << "Empty seed key. Using random 64 bytes";
            seed_key = base::Random::byteArray(64);
        }

        if (user.isValid())
        {
            LOG(LS_INFO) << "User '" << user_name_ << "' found (enabled: "
                         << ((user.flags & User::ENABLED) != 0) << ")";
        }
        else
        {
            LOG(LS_INFO) << "User '" << user_name_ << "' NOT found";
        }

        if (user.isValid() && (user.flags & User::ENABLED))
        {
            session_types_ = user.sessions;

            std::optional<SrpNgPair> Ng_pair = pairByGroup(user.group);
            if (Ng_pair.has_value())
            {
                N_ = BigNum::fromStdString(Ng_pair->first);
                g_ = BigNum::fromStdString(Ng_pair->second);
                s_ = BigNum::fromByteArray(user.salt);
                v_ = BigNum::fromByteArray(user.verifier);
                break;
            }
            else
            {
                LOG(LS_ERROR) << "User '" << user.name << "' has an invalid SRP group";
            }
        }

        session_types_ = 0;

        GenericHash hash(GenericHash::BLAKE2b512);
        hash.addData(seed_key);
        hash.addData(user_name_);

        N_ = BigNum::fromStdString(kSrpNgPair_8192.first);
        g_ = BigNum::fromStdString(kSrpNgPair_8192.second);
        s_ = BigNum::fromByteArray(hash.result());
        v_ = SrpMath::calc_v(user_name_utf16, seed_key, s_, N_, g_);
    }
    while (false);

    b_ = BigNum::fromByteArray(Random::byteArray(128)); // 1024 bits.
    B_ = SrpMath::calc_B(b_, N_, g_, v_);

    if (!N_.isValid() || !g_.isValid() || !s_.isValid() || !B_.isValid())
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    internal_state_ = InternalState::SEND_SERVER_KEY_EXCHANGE;
    encrypt_iv_ = Random::byteArray(kIvSize);

    std::unique_ptr<proto::SrpServerKeyExchange> server_key_exchange =
        std::make_unique<proto::SrpServerKeyExchange>();

    server_key_exchange->set_number(N_.toStdString());
    server_key_exchange->set_generator(g_.toStdString());
    server_key_exchange->set_salt(s_.toStdString());
    server_key_exchange->set_b(B_.toStdString());
    server_key_exchange->set_iv(toStdString(encrypt_iv_));

    LOG(LS_INFO) << "Sending: ServerKeyExchange";
    sendMessage(*server_key_exchange);
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onClientKeyExchange(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: ClientKeyExchange";

    std::unique_ptr<proto::SrpClientKeyExchange> client_key_exchange =
        std::make_unique<proto::SrpClientKeyExchange>();
    if (!parse(buffer, client_key_exchange.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    A_ = BigNum::fromStdString(client_key_exchange->a());
    decrypt_iv_ = fromStdString(client_key_exchange->iv());

    if (!A_.isValid() || decrypt_iv_.empty())
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    ByteArray srp_key = createSrpKey();
    if (srp_key.empty())
    {
        finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
        return;
    }

    switch (encryption_)
    {
        // AES256-GCM and ChaCha20-Poly1305 requires 256 bit key.
        case proto::ENCRYPTION_AES256_GCM:
        case proto::ENCRYPTION_CHACHA20_POLY1305:
        {
            GenericHash hash(GenericHash::BLAKE2s256);

            if (!session_key_.empty())
                hash.addData(session_key_);
            hash.addData(srp_key);

            session_key_ = hash.result();
        }
        break;

        default:
        {
            finish(FROM_HERE, ErrorCode::UNKNOWN_ERROR);
            return;
        }
    }

    if (!onSessionKeyChanged())
        return;

    internal_state_ = InternalState::SEND_SESSION_CHALLENGE;
    doSessionChallenge();
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::doSessionChallenge()
{
    std::unique_ptr<proto::SessionChallenge> session_challenge =
        std::make_unique<proto::SessionChallenge>();
    session_challenge->set_session_types(session_types_);

    proto::Version* version = session_challenge->mutable_version();
    version->set_major(ASPIA_VERSION_MAJOR);
    version->set_minor(ASPIA_VERSION_MINOR);
    version->set_patch(ASPIA_VERSION_PATCH);
    version->set_revision(GIT_COMMIT_COUNT);

    session_challenge->set_os_name(SysInfo::operatingSystemName());
    session_challenge->set_computer_name(SysInfo::computerName());
    session_challenge->set_cpu_cores(static_cast<uint32_t>(SysInfo::processorThreads()));

#if defined(ARCH_CPU_X86)
    session_challenge->set_arch("x86");
#elif defined(ARCH_CPU_X86_64)
    session_challenge->set_arch("x86_64");
#elif defined(ARCH_CPU_ARMEL)
    session_challenge->set_arch("arm");
#elif defined(ARCH_CPU_ARM64)
    session_challenge->set_arch("arm64");
#else
    session_challenge->set_arch(std::string());
#endif

    LOG(LS_INFO) << "Sending: SessionChallenge";
    sendMessage(*session_challenge);
}

//--------------------------------------------------------------------------------------------------
void ServerAuthenticator::onSessionResponse(const ByteArray& buffer)
{
    LOG(LS_INFO) << "Received: SessionResponse";

    std::unique_ptr<proto::SessionResponse> session_response =
        std::make_unique<proto::SessionResponse>();
    if (!parse(buffer, session_response.get()))
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    setPeerVersion(session_response->version());
    setPeerOsName(session_response->os_name());
    setPeerComputerName(session_response->computer_name());

    LOG(LS_INFO) << "Client Session Type: " << session_response->session_type();
    LOG(LS_INFO) << "Client Version: " << peerVersion();
    LOG(LS_INFO) << "Client Name: " << session_response->computer_name();
    LOG(LS_INFO) << "Client OS: " << session_response->os_name();
    LOG(LS_INFO) << "Client CPU Cores: " << session_response->cpu_cores();
    LOG(LS_INFO) << "Client Arch: " << session_response->arch();

    BitSet<uint32_t> session_type = session_response->session_type();
    if (session_type.count() != 1)
    {
        finish(FROM_HERE, ErrorCode::PROTOCOL_ERROR);
        return;
    }

    session_type_ = session_type.value();
    if (!(session_types_ & session_type_))
    {
        finish(FROM_HERE, ErrorCode::SESSION_DENIED);
        return;
    }

    // Authentication completed successfully.
    finish(FROM_HERE, ErrorCode::SUCCESS);
}

//--------------------------------------------------------------------------------------------------
ByteArray ServerAuthenticator::createSrpKey()
{
    if (!SrpMath::verify_A_mod_N(A_, N_))
    {
        LOG(LS_ERROR) << "SrpMath::verify_A_mod_N failed";
        return ByteArray();
    }

    BigNum u = SrpMath::calc_u(A_, B_, N_);
    BigNum server_key = SrpMath::calcServerKey(A_, v_, u, b_, N_);

    return server_key.toByteArray();
}

} // namespace base
