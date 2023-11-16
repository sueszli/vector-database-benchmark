////////////////////////////////////////////////////////////////////////////
//
// Copyright 2021 Realm Inc.
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

#pragma once

#ifdef REALM_ENABLE_SYNC
#ifdef REALM_ENABLE_AUTH_TESTS

#include <util/sync/sync_test_utils.hpp>

#include <realm/object-store/property.hpp>
#include <realm/object-store/object_schema.hpp>
#include <realm/object-store/schema.hpp>

#include <realm/object-store/sync/app_credentials.hpp>
#include <realm/object-store/sync/generic_network_transport.hpp>

#include <realm/util/logger.hpp>

#include <external/json/json.hpp>
#include <external/mpark/variant.hpp>

namespace realm {
app::Response do_http_request(const app::Request& request);

class AdminAPIEndpoint {
public:
    app::Response get(const std::vector<std::pair<std::string, std::string>>& params = {}) const;
    app::Response patch(std::string body) const;
    app::Response post(std::string body) const;
    app::Response put(std::string body) const;
    app::Response del() const;
    nlohmann::json get_json(const std::vector<std::pair<std::string, std::string>>& params = {}) const;
    nlohmann::json patch_json(nlohmann::json body) const;
    nlohmann::json post_json(nlohmann::json body) const;
    nlohmann::json put_json(nlohmann::json body) const;

    AdminAPIEndpoint operator[](StringData name) const;

protected:
    friend class AdminAPISession;
    AdminAPIEndpoint(std::string url, std::string access_token)
        : m_url(std::move(url))
        , m_access_token(std::move(access_token))
    {
    }

    app::Response do_request(app::Request request) const;

private:
    std::string m_url;
    std::string m_access_token;
};

struct AppCreateConfig;

class AdminAPISession {
public:
    static AdminAPISession login(const AppCreateConfig& config);

    enum class APIFamily { Admin, Private };
    AdminAPIEndpoint apps(APIFamily family = APIFamily::Admin) const;
    void revoke_user_sessions(const std::string& user_id, const std::string& app_id) const;
    void disable_user_sessions(const std::string& user_id, const std::string& app_id) const;
    void enable_user_sessions(const std::string& user_id, const std::string& app_id) const;
    bool verify_access_token(const std::string& access_token, const std::string& app_id) const;
    void set_development_mode_to(const std::string& app_id, bool enable) const;
    void delete_app(const std::string& app_id) const;
    void trigger_client_reset(const std::string& app_id, int64_t file_ident) const;
    void migrate_to_flx(const std::string& app_id, const std::string& service_id, bool migrate_to_flx) const;

    struct Service {
        std::string id;
        std::string name;
        std::string type;
        int64_t version;
        int64_t last_modified;
    };
    struct ServiceConfig {
        enum class SyncMode { Partitioned, Flexible } mode = SyncMode::Partitioned;
        std::string database_name;
        util::Optional<nlohmann::json> partition;
        util::Optional<nlohmann::json> queryable_field_names;
        util::Optional<nlohmann::json> permissions;
        std::string state;
        bool recovery_is_disabled = false;
        std::string_view sync_service_name()
        {
            if (mode == SyncMode::Flexible) {
                return "flexible_sync";
            }
            else {
                return "sync";
            }
        }
    };

    std::vector<Service> get_services(const std::string& app_id) const;
    std::vector<std::string> get_errors(const std::string& app_id) const;
    Service get_sync_service(const std::string& app_id) const;
    ServiceConfig get_config(const std::string& app_id, const Service& service) const;
    ServiceConfig disable_sync(const std::string& app_id, const std::string& service_id,
                               ServiceConfig sync_config) const;
    ServiceConfig pause_sync(const std::string& app_id, const std::string& service_id,
                             ServiceConfig sync_config) const;
    ServiceConfig enable_sync(const std::string& app_id, const std::string& service_id,
                              ServiceConfig sync_config) const;
    ServiceConfig set_disable_recovery_to(const std::string& app_id, const std::string& service_id,
                                          ServiceConfig sync_config, bool disable) const;
    bool is_sync_enabled(const std::string& app_id) const;
    bool is_sync_terminated(const std::string& app_id) const;
    bool is_initial_sync_complete(const std::string& app_id) const;

    struct MigrationStatus {
        std::string statusMessage;
        bool isMigrated = false;
        bool isCancelable = false;
        bool isRevertible = false;
        bool complete = false;
    };

    MigrationStatus get_migration_status(const std::string& app_id) const;

    const std::string& admin_url() const noexcept
    {
        return m_base_url;
    }

private:
    AdminAPISession(std::string admin_url, std::string access_token, std::string group_id)
        : m_base_url(std::move(admin_url))
        , m_access_token(std::move(access_token))
        , m_group_id(std::move(group_id))
    {
    }

    AdminAPIEndpoint service_config_endpoint(const std::string& app_id, const std::string& service_id) const;

    std::string m_base_url;
    std::string m_access_token;
    std::string m_group_id;
};

struct AppCreateConfig {
    struct FunctionDef {
        std::string name;
        std::string source;
        bool is_private;
    };

    struct UserPassAuthConfig {
        bool auto_confirm;
        std::string confirm_email_subject;
        std::string confirmation_function_name;
        std::string email_confirmation_url;
        std::string reset_function_name;
        std::string reset_password_subject;
        std::string reset_password_url;
        bool run_confirmation_function;
        bool run_reset_function;
    };

    struct ServiceRoleDocumentFilters {
        nlohmann::json read;
        nlohmann::json write;
    };

    // ServiceRole represents the set of permissions used MongoDB-based services (Flexible Sync, DataAPI, GraphQL,
    // etc.). In flexible sync, roles are assigned on a per-table, per-session basis by the server. NB: there are
    // restrictions on the role configuration when used with flexible sync. See
    // https://www.mongodb.com/docs/atlas/app-services/rules/sync-compatibility/ for more information.
    struct ServiceRole {
        std::string name;

        // apply_when describes when a role applies. Set it to an empty JSON expression ("{}") if
        // the role should always apply
        nlohmann::json apply_when = nlohmann::json::object();

        // document_filters describe which objects can be read from/written to, as
        // specified by the below read and write expressions. Set both to true to give read/write
        // access on all objects
        ServiceRoleDocumentFilters document_filters;

        // insert_filter and delete_filter describe which objects can be created and erased by the client,
        // respectively. Set both to true if all objects can be created/erased by the client
        nlohmann::json insert_filter;
        nlohmann::json delete_filter;

        // read and write describe the permissions for "read-all-fields"/"write-all-fields" behavior. Set both to true
        // if all fields should have read/write access
        nlohmann::json read;
        nlohmann::json write;

        // NB: for more granular field-level permissions, the "fields" and "additional_fields" keys can be included in
        // a service role to describe which fields individually can be read/written. These fields have been omitted
        // here for simplicity
    };

    struct FLXSyncConfig {
        std::vector<std::string> queryable_fields;
    };

    std::string app_name;
    std::string app_url;
    std::string admin_url;
    std::string admin_username;
    std::string admin_password;

    std::string mongo_uri;
    std::string mongo_dbname;

    Schema schema;
    Property partition_key;
    bool dev_mode_enabled;
    util::Optional<FLXSyncConfig> flx_sync_config;

    std::vector<FunctionDef> functions;

    util::Optional<UserPassAuthConfig> user_pass_auth;
    util::Optional<std::string> custom_function_auth;
    bool enable_api_key_auth = false;
    bool enable_anonymous_auth = false;
    bool enable_custom_token_auth = false;

    std::vector<ServiceRole> service_roles;

    std::shared_ptr<util::Logger> logger;
};

realm::Schema get_default_schema();
AppCreateConfig default_app_config();
AppCreateConfig minimal_app_config(const std::string& name, const Schema& schema);

struct AppSession {
    std::string client_app_id;
    std::string server_app_id;
    AdminAPISession admin_api;
    AppCreateConfig config;
};
AppSession create_app(const AppCreateConfig& config);

class SynchronousTestTransport : public app::GenericNetworkTransport {
public:
    void send_request_to_server(const app::Request& request,
                                util::UniqueFunction<void(const app::Response&)>&& completion) override
    {
        {
            std::lock_guard barrier(m_mutex);
        }
        completion(do_http_request(request));
    }

    void block()
    {
        m_mutex.lock();
    }
    void unblock()
    {
        m_mutex.unlock();
    }

private:
    std::mutex m_mutex;
};

// This will create a new test app in the baas server - base_url and admin_url
// are automatically set
AppSession get_runtime_app_session();

std::string get_mongodb_server();

template <typename Factory>
inline app::App::Config get_config(Factory factory, const AppSession& app_session)
{
    return {app_session.client_app_id,
            factory,
            app_session.config.app_url,
            util::none,
            {"Object Store Platform Version Blah", "An sdk version", "An sdk name", "A device name",
             "A device version", "A framework name", "A framework version", "A bundle id"}};
}

} // namespace realm

#endif // REALM_ENABLE_AUTH_TESTS
#endif // REALM_ENABLE_SYNC
