////////////////////////////////////////////////////////////////////////////
//
// Copyright 2016 Realm Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
////////////////////////////////////////////////////////////////////////////

#include <realm/object-store/sync/sync_user.hpp>

#include <realm/object-store/sync/app.hpp>
#include <realm/object-store/sync/app_credentials.hpp>
#include <realm/object-store/sync/generic_network_transport.hpp>
#include <realm/object-store/sync/impl/sync_metadata.hpp>
#include <realm/object-store/sync/mongo_client.hpp>
#include <realm/object-store/sync/sync_manager.hpp>
#include <realm/object-store/sync/sync_session.hpp>

#include <realm/util/base64.hpp>

namespace realm {

static std::string base64_decode(const std::string& in)
{
    std::string out;
    out.resize(util::base64_decoded_size(in.size()));
    util::base64_decode(in, &out[0], out.size());
    return out;
}

static std::vector<std::string> split_token(const std::string& jwt)
{
    constexpr static char delimiter = '.';

    std::vector<std::string> parts;
    size_t pos = 0, start_from = 0;

    while ((pos = jwt.find(delimiter, start_from)) != std::string::npos) {
        parts.push_back(jwt.substr(start_from, pos - start_from));
        start_from = pos + 1;
    }

    parts.push_back(jwt.substr(start_from));

    if (parts.size() != 3) {
        throw app::AppError(ErrorCodes::BadToken, "jwt missing parts");
    }

    return parts;
}

RealmJWT::RealmJWT(const std::string& token)
    : token(token)
{
    auto parts = split_token(this->token);

    auto json_str = base64_decode(parts[1]);
    auto json = static_cast<bson::BsonDocument>(bson::parse(json_str));

    this->expires_at = static_cast<int64_t>(json["exp"]);
    this->issued_at = static_cast<int64_t>(json["iat"]);

    if (json.find("user_data") != json.end()) {
        this->user_data = static_cast<bson::BsonDocument>(json["user_data"]);
    }
}

SyncUserIdentity::SyncUserIdentity(const std::string& id, const std::string& provider_type)
    : id(id)
    , provider_type(provider_type)
{
}

SyncUserContextFactory SyncUser::s_binding_context_factory;
std::mutex SyncUser::s_binding_context_factory_mutex;

SyncUser::SyncUser(const std::string& refresh_token, const std::string& id, const std::string& access_token,
                   const std::string& device_id, SyncManager* sync_manager)
    : m_state(State::LoggedIn)
    , m_identity(id)
    , m_refresh_token(RealmJWT(refresh_token))
    , m_access_token(RealmJWT(access_token))
    , m_device_id(device_id)
    , m_sync_manager(sync_manager)
{
    REALM_ASSERT(!access_token.empty() && !refresh_token.empty());
    {
        std::lock_guard lock(s_binding_context_factory_mutex);
        if (s_binding_context_factory) {
            m_binding_context = s_binding_context_factory();
        }
    }

    m_sync_manager->perform_metadata_update([&](const auto& manager) NO_THREAD_SAFETY_ANALYSIS {
        auto metadata = manager.get_or_make_user_metadata(m_identity);
        metadata->set_state_and_tokens(State::LoggedIn, m_access_token.token, m_refresh_token.token);
        metadata->set_device_id(m_device_id);
        m_legacy_identities = metadata->legacy_identities();
        this->m_user_profile = metadata->profile();
    });
}

SyncUser::SyncUser(const SyncUserMetadata& data, SyncManager* sync_manager)
    : m_state(data.state())
    , m_legacy_identities(data.legacy_identities())
    , m_identity(data.identity())
    , m_refresh_token(RealmJWT(data.refresh_token()))
    , m_access_token(RealmJWT(data.access_token()))
    , m_user_identities(data.identities())
    , m_user_profile(data.profile())
    , m_device_id(data.device_id())
    , m_sync_manager(sync_manager)
{
    // Check for inconsistent state in the metadata Realm. This shouldn't happen,
    // but previous versions could sometimes mark a user as logged in with an
    // empty refresh token.
    if (m_state == State::LoggedIn && (m_refresh_token.token.empty() || m_access_token.token.empty())) {
        m_state = State::LoggedOut;
        m_refresh_token = {};
        m_access_token = {};
    }

    {
        std::lock_guard lock(s_binding_context_factory_mutex);
        if (s_binding_context_factory) {
            m_binding_context = s_binding_context_factory();
        }
    }
}

std::shared_ptr<SyncManager> SyncUser::sync_manager() const
{
    util::CheckedLockGuard lk(m_mutex);
    if (m_state == State::Removed) {
        throw app::AppError(
            ErrorCodes::ClientUserNotFound,
            util::format("Cannot start a sync session for user '%1' because this user has been removed.",
                         m_identity));
    }
    REALM_ASSERT(m_sync_manager);
    return m_sync_manager->shared_from_this();
}

void SyncUser::detach_from_sync_manager()
{
    util::CheckedLockGuard lk(m_mutex);
    REALM_ASSERT(m_sync_manager);
    m_state = SyncUser::State::Removed;
    m_sync_manager = nullptr;
}

std::vector<std::shared_ptr<SyncSession>> SyncUser::all_sessions()
{
    util::CheckedLockGuard lock(m_mutex);
    std::vector<std::shared_ptr<SyncSession>> sessions;
    if (m_state == State::Removed) {
        return sessions;
    }
    for (auto it = m_sessions.begin(); it != m_sessions.end();) {
        if (auto ptr_to_session = it->second.lock()) {
            sessions.emplace_back(std::move(ptr_to_session));
            it++;
            continue;
        }
        // This session is bad, destroy it.
        it = m_sessions.erase(it);
    }
    return sessions;
}

std::shared_ptr<SyncSession> SyncUser::session_for_on_disk_path(const std::string& path)
{
    util::CheckedLockGuard lock(m_mutex);
    if (m_state == State::Removed) {
        return nullptr;
    }
    auto it = m_sessions.find(path);
    if (it == m_sessions.end()) {
        return nullptr;
    }
    auto locked = it->second.lock();
    if (!locked) {
        // Remove the session from the map, because it has fatally errored out or the entry is invalid.
        m_sessions.erase(it);
    }
    return locked;
}

void SyncUser::log_in(const std::string& access_token, const std::string& refresh_token)
{
    REALM_ASSERT(!access_token.empty());
    REALM_ASSERT(!refresh_token.empty());
    std::vector<std::shared_ptr<SyncSession>> sessions_to_revive;
    {
        util::CheckedLockGuard lock1(m_mutex);
        util::CheckedLockGuard lock2(m_tokens_mutex);
        m_state = State::LoggedIn;
        m_access_token = RealmJWT(access_token);
        m_refresh_token = RealmJWT(refresh_token);
        sessions_to_revive = revive_sessions();

        m_sync_manager->perform_metadata_update([&](const auto& manager) {
            auto metadata = manager.get_or_make_user_metadata(m_identity);
            metadata->set_state_and_tokens(State::LoggedIn, access_token, refresh_token);
        });
    }
    // (Re)activate all pending sessions.
    // Note that we do this after releasing the lock, since the session may
    // need to access protected User state in the process of binding itself.
    for (auto& session : sessions_to_revive) {
        session->revive_if_needed();
    }

    emit_change_to_subscribers(*this);
}

void SyncUser::invalidate()
{
    {
        util::CheckedLockGuard lock1(m_mutex);
        util::CheckedLockGuard lock2(m_tokens_mutex);
        m_state = State::Removed;
        m_access_token = {};
        m_refresh_token = {};

        m_sync_manager->perform_metadata_update([&](const auto& manager) {
            auto metadata = manager.get_or_make_user_metadata(m_identity);
            metadata->set_state_and_tokens(State::Removed, "", "");
        });
    }
    emit_change_to_subscribers(*this);
}

std::vector<std::shared_ptr<SyncSession>> SyncUser::revive_sessions()
{
    std::vector<std::shared_ptr<SyncSession>> sessions_to_revive;
    sessions_to_revive.reserve(m_waiting_sessions.size());
    for (auto& [path, weak_session] : m_waiting_sessions) {
        if (auto ptr = weak_session.lock()) {
            m_sessions[path] = ptr;
            sessions_to_revive.emplace_back(std::move(ptr));
        }
    }
    m_waiting_sessions.clear();
    return sessions_to_revive;
}

void SyncUser::update_access_token(std::string&& token)
{
    {
        util::CheckedLockGuard lock(m_mutex);
        if (m_state != State::LoggedIn)
            return;

        util::CheckedLockGuard lock2(m_tokens_mutex);
        m_access_token = RealmJWT(std::move(token));
        m_sync_manager->perform_metadata_update([&, raw_access_token = m_access_token.token](const auto& manager) {
            auto metadata = manager.get_or_make_user_metadata(m_identity);
            metadata->set_access_token(raw_access_token);
        });
    }

    emit_change_to_subscribers(*this);
}

std::vector<SyncUserIdentity> SyncUser::identities() const
{
    util::CheckedLockGuard lock(m_mutex);
    return m_user_identities;
}

void SyncUser::log_out()
{
    // We'll extend the lifetime of SyncManager while holding m_mutex so that we know it's safe to call methods on it
    // after we've been marked as logged out.
    std::shared_ptr<SyncManager> sync_manager_shared;
    {
        util::CheckedLockGuard lock(m_mutex);
        bool is_anonymous = false;
        {
            util::CheckedLockGuard lock2(m_tokens_mutex);
            if (m_state != State::LoggedIn) {
                return;
            }
            is_anonymous = do_is_anonymous();
            m_state = State::LoggedOut;
            m_access_token = RealmJWT{};
            m_refresh_token = RealmJWT{};
        }

        if (is_anonymous) {
            // An Anonymous user can not log back in.
            // Mark the user as 'dead' in the persisted metadata Realm.
            m_state = State::Removed;
            m_sync_manager->perform_metadata_update([&](const auto& manager) {
                auto metadata = manager.get_or_make_user_metadata(m_identity, false);
                if (metadata)
                    metadata->remove();
            });
        }
        else {
            m_sync_manager->perform_metadata_update([&](const auto& manager) {
                auto metadata = manager.get_or_make_user_metadata(m_identity);
                metadata->set_state_and_tokens(State::LoggedOut, "", "");
            });
        }
        sync_manager_shared = m_sync_manager->shared_from_this();
        // Move all active sessions into the waiting sessions pool. If the user is
        // logged back in, they will automatically be reactivated.
        for (auto& [path, weak_session] : m_sessions) {
            if (auto ptr = weak_session.lock()) {
                ptr->force_close();
                m_waiting_sessions[path] = std::move(ptr);
            }
        }
        m_sessions.clear();
    }

    sync_manager_shared->log_out_user(*this);
    emit_change_to_subscribers(*this);
}

bool SyncUser::is_logged_in() const
{
    util::CheckedLockGuard lock(m_mutex);
    return m_state == State::LoggedIn;
}

bool SyncUser::is_anonymous() const
{
    util::CheckedLockGuard lock(m_mutex);
    util::CheckedLockGuard lock2(m_tokens_mutex);
    return do_is_anonymous();
}

bool SyncUser::do_is_anonymous() const
{
    return m_state == State::LoggedIn && m_user_identities.size() == 1 &&
           m_user_identities[0].provider_type == app::IdentityProviderAnonymous;
}

std::string SyncUser::refresh_token() const
{
    util::CheckedLockGuard lock(m_tokens_mutex);
    return m_refresh_token.token;
}

std::string SyncUser::access_token() const
{
    util::CheckedLockGuard lock(m_tokens_mutex);
    return m_access_token.token;
}

std::string SyncUser::device_id() const
{
    util::CheckedLockGuard lock(m_mutex);
    return m_device_id;
}

bool SyncUser::has_device_id() const
{
    util::CheckedLockGuard lock(m_mutex);
    return !m_device_id.empty() && m_device_id != "000000000000000000000000";
}

SyncUser::State SyncUser::state() const
{
    util::CheckedLockGuard lock(m_mutex);
    return m_state;
}

SyncUserProfile SyncUser::user_profile() const
{
    util::CheckedLockGuard lock(m_mutex);
    return m_user_profile;
}

util::Optional<bson::BsonDocument> SyncUser::custom_data() const
{
    util::CheckedLockGuard lock(m_tokens_mutex);
    return m_access_token.user_data;
}

void SyncUser::update_user_profile(std::vector<SyncUserIdentity> identities, SyncUserProfile profile)
{
    util::CheckedLockGuard lock(m_mutex);
    if (m_state == SyncUser::State::Removed) {
        return;
    }

    m_user_identities = std::move(identities);
    m_user_profile = std::move(profile);

    m_sync_manager->perform_metadata_update([&](const auto& manager) NO_THREAD_SAFETY_ANALYSIS {
        auto metadata = manager.get_or_make_user_metadata(m_identity);
        metadata->set_identities(m_user_identities);
        metadata->set_user_profile(m_user_profile);
    });
}

void SyncUser::register_session(std::shared_ptr<SyncSession> session)
{
    const std::string& path = session->path();
    util::CheckedUniqueLock lock(m_mutex);
    switch (m_state) {
        case State::LoggedIn:
            m_sessions[path] = session;
            break;
        case State::LoggedOut:
            m_waiting_sessions[path] = session;
            break;
        case State::Removed:
            break;
    }
}

app::MongoClient SyncUser::mongo_client(const std::string& service_name)
{
    util::CheckedLockGuard lk(m_mutex);
    REALM_ASSERT(m_state == SyncUser::State::LoggedIn);
    return app::MongoClient(shared_from_this(), m_sync_manager->app().lock(), service_name);
}

void SyncUser::set_binding_context_factory(SyncUserContextFactory factory)
{
    std::lock_guard<std::mutex> lock(s_binding_context_factory_mutex);
    s_binding_context_factory = std::move(factory);
}

void SyncUser::refresh_custom_data(util::UniqueFunction<void(util::Optional<app::AppError>)> completion_block)
    REQUIRES(!m_mutex)
{
    refresh_custom_data(false, std::move(completion_block));
}

void SyncUser::refresh_custom_data(bool update_location,
                                   util::UniqueFunction<void(util::Optional<app::AppError>)> completion_block)
{
    std::shared_ptr<app::App> app;
    std::shared_ptr<SyncUser> user;
    {
        util::CheckedLockGuard lk(m_mutex);
        if (m_state != SyncUser::State::Removed) {
            user = shared_from_this();
        }
        if (m_sync_manager) {
            app = m_sync_manager->app().lock();
        }
    }
    if (!user) {
        completion_block(app::AppError(
            ErrorCodes::ClientUserNotFound,
            util::format("Cannot initiate a refresh on user '%1' because the user has been removed", m_identity)));
    }
    else if (!app) {
        completion_block(app::AppError(
            ErrorCodes::ClientAppDeallocated,
            util::format("Cannot initiate a refresh on user '%1' because the app has been deallocated", m_identity)));
    }
    else {
        std::weak_ptr<SyncUser> weak_user = user->weak_from_this();
        app->refresh_custom_data(user, update_location,
                                 [completion_block = std::move(completion_block), weak_user](auto error) {
                                     if (auto strong = weak_user.lock()) {
                                         strong->emit_change_to_subscribers(*strong);
                                     }
                                     completion_block(error);
                                 });
    }
}

bool SyncUser::access_token_refresh_required() const
{
    using namespace std::chrono;
    constexpr size_t buffer_seconds = 5; // arbitrary
    util::CheckedLockGuard lock(m_tokens_mutex);
    const auto now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count() +
                     m_seconds_to_adjust_time_for_testing.load(std::memory_order_relaxed);
    const auto threshold = now - buffer_seconds;
    return !m_access_token.token.empty() && m_access_token.expires_at < static_cast<int64_t>(threshold);
}

} // namespace realm

namespace std {
size_t hash<realm::SyncUserIdentity>::operator()(const realm::SyncUserIdentity& k) const
{
    return ((hash<string>()(k.id) ^ (hash<string>()(k.provider_type) << 1)) >> 1);
}
} // namespace std
