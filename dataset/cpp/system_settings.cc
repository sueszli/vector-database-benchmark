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

#include "host/system_settings.h"

#include "base/crypto/password_generator.h"
#include "base/crypto/password_hash.h"
#include "base/crypto/random.h"
#include "base/peer/user_list.h"

namespace host {

namespace {

const size_t kPasswordHashSaltSize = 256;

} // namespace

//--------------------------------------------------------------------------------------------------
SystemSettings::SystemSettings()
    : settings_(base::JsonSettings::Scope::SYSTEM, "aspia", "host")
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
SystemSettings::~SystemSettings() = default;

//--------------------------------------------------------------------------------------------------
// static
bool SystemSettings::createPasswordHash(
    std::string_view password, base::ByteArray* hash, base::ByteArray* salt)
{
    if (password.empty() || !hash || !salt)
        return false;

    base::ByteArray salt_temp = base::Random::byteArray(kPasswordHashSaltSize);
    if (salt_temp.empty())
        return false;

    base::ByteArray hash_temp =
        base::PasswordHash::hash(base::PasswordHash::SCRYPT, password, salt_temp);
    if (hash_temp.empty())
        return false;

    *salt = std::move(salt_temp);
    *hash = std::move(hash_temp);
    return true;
}

//--------------------------------------------------------------------------------------------------
// static
bool SystemSettings::isValidPassword(std::string_view password)
{
    if (password.empty())
        return false;

    SystemSettings settings;

    base::ByteArray password_hash_salt = settings.passwordHashSalt();
    base::ByteArray password_hash = settings.passwordHash();

    if (password_hash_salt.empty() || password_hash.empty())
        return false;

    base::ByteArray verifiable_password_hash =
        base::PasswordHash::hash(base::PasswordHash::SCRYPT, password, password_hash_salt);
    if (verifiable_password_hash.empty())
        return false;

    return base::equals(verifiable_password_hash, password_hash);
}

//--------------------------------------------------------------------------------------------------
const std::filesystem::path& SystemSettings::filePath() const
{
    return settings_.filePath();
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::isWritable() const
{
    return settings_.isWritable();
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::sync()
{
    settings_.sync();
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::flush()
{
    return settings_.flush();
}

//--------------------------------------------------------------------------------------------------
uint16_t SystemSettings::tcpPort() const
{
    return settings_.get<uint16_t>("TcpPort", DEFAULT_HOST_TCP_PORT);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setTcpPort(uint16_t port)
{
    settings_.set<uint16_t>("TcpPort", port);
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::isRouterEnabled() const
{
    return settings_.get<bool>("RouterEnabled", false);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setRouterEnabled(bool enable)
{
    settings_.set<bool>("RouterEnabled", enable);
}

//--------------------------------------------------------------------------------------------------
std::u16string SystemSettings::routerAddress() const
{
    return settings_.get<std::u16string>("RouterAddress");
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setRouterAddress(const std::u16string& address)
{
    settings_.set<std::u16string>("RouterAddress", address);
}

//--------------------------------------------------------------------------------------------------
uint16_t SystemSettings::routerPort() const
{
    return settings_.get<uint16_t>("RouterPort", DEFAULT_ROUTER_TCP_PORT);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setRouterPort(uint16_t port)
{
    settings_.set<uint16_t>("RouterPort", port);
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SystemSettings::routerPublicKey() const
{
    return settings_.get<base::ByteArray>("RouterPublicKey");
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setRouterPublicKey(const base::ByteArray& key)
{
    settings_.set<base::ByteArray>("RouterPublicKey", key);
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<base::UserList> SystemSettings::userList() const
{
    std::unique_ptr<base::UserList> users = base::UserList::createEmpty();

    for (const auto& item : settings_.getArray("Users"))
    {
        base::User user;

        user.name     = item.get<std::u16string>("Name");
        user.group    = item.get<std::string>("Group");
        user.salt     = item.get<base::ByteArray>("Salt");
        user.verifier = item.get<base::ByteArray>("Verifier");
        user.sessions = item.get<uint32_t>("Sessions");
        user.flags    = item.get<uint32_t>("Flags");

        users->add(user);
    }

    base::ByteArray seed_key = settings_.get<base::ByteArray>("SeedKey");
    if (seed_key.empty())
        seed_key = base::Random::byteArray(64);

    users->setSeedKey(seed_key);
    return users;
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setUserList(const base::UserList& users)
{
    // Clear the old list of users.
    settings_.remove("Users");

    base::Settings::Array users_array;

    for (const auto& user : users.list())
    {
        base::Settings item;
        item.set("Name", user.name);
        item.set("Group", user.group);
        item.set("Salt", user.salt);
        item.set("Verifier", user.verifier);
        item.set("Sessions", user.sessions);
        item.set("Flags", user.flags);

        users_array.emplace_back(std::move(item));
    }

    base::ByteArray seed_key = users.seedKey();
    if (seed_key.empty())
        seed_key = base::Random::byteArray(64);

    settings_.setArray("Users", users_array);
    settings_.set("SeedKey", seed_key);
}

//--------------------------------------------------------------------------------------------------
std::string SystemSettings::updateServer() const
{
    return settings_.get<std::string>("UpdateServer", DEFAULT_UPDATE_SERVER);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setUpdateServer(const std::string& server)
{
    settings_.set("UpdateServer", server);
}

//--------------------------------------------------------------------------------------------------
uint32_t SystemSettings::preferredVideoCapturer() const
{
    return settings_.get<uint32_t>("PreferredVideoCapturer", 0);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setPreferredVideoCapturer(uint32_t type)
{
    settings_.set("PreferredVideoCapturer", type);
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::passwordProtection() const
{
    return settings_.get<bool>("PasswordProtection", false);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setPasswordProtection(bool enable)
{
    settings_.set("PasswordProtection", enable);
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SystemSettings::passwordHash() const
{
    return settings_.get<base::ByteArray>("PasswordHash");
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setPasswordHash(const base::ByteArray& hash)
{
    settings_.set("PasswordHash", hash);
}

//--------------------------------------------------------------------------------------------------
base::ByteArray SystemSettings::passwordHashSalt() const
{
    return settings_.get<base::ByteArray>("PasswordHashSalt");
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setPasswordHashSalt(const base::ByteArray& salt)
{
    settings_.set("PasswordHashSalt", salt);
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::oneTimePassword() const
{
    return settings_.get<bool>("OneTimePassword", true);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setOneTimePassword(bool enable)
{
    settings_.set("OneTimePassword", enable);
}

//--------------------------------------------------------------------------------------------------
std::chrono::milliseconds SystemSettings::oneTimePasswordExpire() const
{
    static const std::chrono::milliseconds kDefaultValue { 5 * 60 * 1000 }; // 5 minutes.
    static const std::chrono::milliseconds kMinValue { 0 };
    static const std::chrono::milliseconds kMaxValue { 24 * 60 * 60 * 1000 }; // 24 hours.

    std::chrono::milliseconds value(
        settings_.get<int64_t>("OneTimePasswordExpire", kDefaultValue.count()));

    if (value < kMinValue)
        value = kMinValue;
    else if (value > kMaxValue)
        value = kMaxValue;

    return value;
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setOneTimePasswordExpire(const std::chrono::milliseconds& interval)
{
    settings_.set("OneTimePasswordExpire", interval.count());
}

//--------------------------------------------------------------------------------------------------
int SystemSettings::oneTimePasswordLength() const
{
    static const int kDefaultValue = 8;
    static const int kMinValue = 6;
    static const int kMaxValue = 16;

    int value = settings_.get<int>("OneTimePasswordLength", kDefaultValue);

    if (value < kMinValue)
        value = kMinValue;
    else if (value > kMaxValue)
        value = kMaxValue;

    return value;
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setOneTimePasswordLength(int length)
{
    settings_.set("OneTimePasswordLength", length);
}

//--------------------------------------------------------------------------------------------------
uint32_t SystemSettings::oneTimePasswordCharacters() const
{
    uint32_t kDefaultValue = base::PasswordGenerator::DIGITS | base::PasswordGenerator::LOWER_CASE |
        base::PasswordGenerator::UPPER_CASE;

    uint32_t value = settings_.get<uint32_t>("OneTimePasswordCharacters", kDefaultValue);

    if (!(value & base::PasswordGenerator::DIGITS) &&
        !(value & base::PasswordGenerator::LOWER_CASE) &&
        !(value & base::PasswordGenerator::UPPER_CASE))
    {
        value = kDefaultValue;
    }

    return value;
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setOneTimePasswordCharacters(uint32_t characters)
{
    settings_.set("OneTimePasswordCharacters", characters);
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::connConfirm() const
{
    return settings_.get<bool>("ConnConfirm", false);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setConnConfirm(bool enable)
{
    settings_.set("ConnConfirm", enable);
}

//--------------------------------------------------------------------------------------------------
SystemSettings::NoUserAction SystemSettings::connConfirmNoUserAction() const
{
    return static_cast<NoUserAction>(
        settings_.get<int>("ConnConfirmNoUserAction", static_cast<int>(NoUserAction::ACCEPT)));
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setConnConfirmNoUserAction(NoUserAction action)
{
    settings_.set("ConnConfirmNoUserAction", static_cast<int>(action));
}

//--------------------------------------------------------------------------------------------------
std::chrono::milliseconds SystemSettings::autoConnConfirmInterval() const
{
    static const std::chrono::milliseconds kDefaultValue { 0 };
    static const std::chrono::milliseconds kMinValue { 0 };
    static const std::chrono::milliseconds kMaxValue { 60 * 1000 }; // 60 seconds.

    std::chrono::milliseconds value(
        settings_.get<int64_t>("AutoConnConfirmInterval", kDefaultValue.count()));

    if (value < kMinValue)
        value = kMinValue;
    else if (value > kMaxValue)
        value = kMaxValue;

    return value;
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setAutoConnConfirmInterval(const std::chrono::milliseconds& interval)
{
    settings_.set("AutoConnConfirmInterval", interval.count());
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::isApplicationShutdownDisabled() const
{
    return settings_.get<bool>("ApplicationShutdownDisabled", false);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setApplicationShutdownDisabled(bool value)
{
    settings_.set("ApplicationShutdownDisabled", value);
}

//--------------------------------------------------------------------------------------------------
bool SystemSettings::isAutoUpdateEnabled() const
{
    return settings_.get<bool>("AutoUpdateEnabled", true);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setAutoUpdateEnabled(bool enable)
{
    settings_.set("AutoUpdateEnabled", enable);
}

//--------------------------------------------------------------------------------------------------
int SystemSettings::updateCheckFrequency() const
{
    return settings_.get<int>("UpdateCheckFrequency", 7);
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setUpdateCheckFrequency(int days)
{
    settings_.set("UpdateCheckFrequency", days);
}

//--------------------------------------------------------------------------------------------------
int64_t SystemSettings::lastUpdateCheck() const
{
    return settings_.get<int64_t>("LastUpdateCheck");
}

//--------------------------------------------------------------------------------------------------
void SystemSettings::setLastUpdateCheck(int64_t timepoint)
{
    settings_.set("LastUpdateCheck", timepoint);
}

} // namespace host
