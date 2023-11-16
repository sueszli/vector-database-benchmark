/*
 * Copyright (C) 2016-present ScyllaDB
 *
 * Modified by ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#include "auth/password_authenticator.hh"

#include <algorithm>
#include <chrono>
#include <random>
#include <string_view>
#include <optional>

#include <boost/algorithm/cxx11/all_of.hpp>
#include <seastar/core/seastar.hh>

#include "auth/authenticated_user.hh"
#include "auth/common.hh"
#include "auth/passwords.hh"
#include "auth/roles-metadata.hh"
#include "cql3/untyped_result_set.hh"
#include "log.hh"
#include "service/migration_manager.hh"
#include "utils/class_registrator.hh"
#include "replica/database.hh"
#include "cql3/query_processor.hh"
#include "db/config.hh"

namespace auth {

constexpr std::string_view password_authenticator_name("org.apache.cassandra.auth.PasswordAuthenticator");

// name of the hash column.
static constexpr std::string_view SALTED_HASH = "salted_hash";
static constexpr std::string_view DEFAULT_USER_NAME = meta::DEFAULT_SUPERUSER_NAME;
static const sstring DEFAULT_USER_PASSWORD = sstring(meta::DEFAULT_SUPERUSER_NAME);

static logging::logger plogger("password_authenticator");

// To ensure correct initialization order, we unfortunately need to use a string literal.
static const class_registrator<
        authenticator,
        password_authenticator,
        cql3::query_processor&,
        ::service::migration_manager&> password_auth_reg("org.apache.cassandra.auth.PasswordAuthenticator");

static thread_local auto rng_for_salt = std::default_random_engine(std::random_device{}());

static std::string_view get_config_value(std::string_view value, std::string_view def) {
    return value.empty() ? def : value;
}

std::string password_authenticator::default_superuser(const db::config& cfg) {
    return std::string(get_config_value(cfg.auth_superuser_name(), DEFAULT_USER_NAME));
}

password_authenticator::~password_authenticator() {
}

password_authenticator::password_authenticator(cql3::query_processor& qp, ::service::migration_manager& mm)
    : _qp(qp)
    , _migration_manager(mm)
    , _stopped(make_ready_future<>()) 
    , _superuser(default_superuser(qp.db().get_config()))
{}

static bool has_salted_hash(const cql3::untyped_result_set_row& row) {
    return !row.get_or<sstring>(SALTED_HASH, "").empty();
}

static const sstring& update_row_query() {
    static const sstring update_row_query = format("UPDATE {} SET {} = ? WHERE {} = ?",
            meta::roles_table::qualified_name,
            SALTED_HASH,
            meta::roles_table::role_col_name);
    return update_row_query;
}

static const sstring legacy_table_name{"credentials"};

bool password_authenticator::legacy_metadata_exists() const {
    return _qp.db().has_schema(meta::AUTH_KS, legacy_table_name);
}

future<> password_authenticator::migrate_legacy_metadata() const {
    plogger.info("Starting migration of legacy authentication metadata.");
    static const sstring query = format("SELECT * FROM {}.{}", meta::AUTH_KS, legacy_table_name);

    return _qp.execute_internal(
            query,
            db::consistency_level::QUORUM,
            internal_distributed_query_state(),
            cql3::query_processor::cache_internal::no).then([this](::shared_ptr<cql3::untyped_result_set> results) {
        return do_for_each(*results, [this](const cql3::untyped_result_set_row& row) {
            auto username = row.get_as<sstring>("username");
            auto salted_hash = row.get_as<sstring>(SALTED_HASH);

            return _qp.execute_internal(
                    update_row_query(),
                    consistency_for_user(username),
                    internal_distributed_query_state(),
                    {std::move(salted_hash), username},
                    cql3::query_processor::cache_internal::no).discard_result();
        }).finally([results] {});
    }).then([] {
       plogger.info("Finished migrating legacy authentication metadata.");
    }).handle_exception([](std::exception_ptr ep) {
        plogger.error("Encountered an error during migration!");
        std::rethrow_exception(ep);
    });
}

future<> password_authenticator::create_default_if_missing() const {
    return default_role_row_satisfies(_qp, &has_salted_hash, _superuser).then([this](bool exists) {
        if (!exists) {
            std::string salted_pwd(get_config_value(_qp.db().get_config().auth_superuser_salted_password(), ""));
            if (salted_pwd.empty()) {
                salted_pwd = passwords::hash(DEFAULT_USER_PASSWORD, rng_for_salt);
            }
            return _qp.execute_internal(
                    update_row_query(),
                    db::consistency_level::QUORUM,
                    internal_distributed_query_state(),
                    {salted_pwd, _superuser},
                    cql3::query_processor::cache_internal::no).then([](auto&&) {
                plogger.info("Created default superuser authentication record.");
            });
        }

        return make_ready_future<>();
    });
}

future<> password_authenticator::start() {
     return once_among_shards([this] {
         auto f = create_metadata_table_if_missing(
                 meta::roles_table::name,
                 _qp,
                 meta::roles_table::creation_query(),
                 _migration_manager);

         _stopped = do_after_system_ready(_as, [this] {
             return async([this] {
                 _migration_manager.wait_for_schema_agreement(_qp.db().real_database(), db::timeout_clock::time_point::max(), &_as).get0();

                 if (any_nondefault_role_row_satisfies(_qp, &has_salted_hash, _superuser).get0()) {
                     if (legacy_metadata_exists()) {
                         plogger.warn("Ignoring legacy authentication metadata since nondefault data already exist.");
                     }

                     return;
                 }

                 if (legacy_metadata_exists()) {
                     migrate_legacy_metadata().get0();
                     return;
                 }

                 create_default_if_missing().get0();
             });
         });

         return f;
     });
 }

future<> password_authenticator::stop() {
    _as.request_abort();
    return _stopped.handle_exception_type([] (const sleep_aborted&) { }).handle_exception_type([](const abort_requested_exception&) {});
}

db::consistency_level password_authenticator::consistency_for_user(std::string_view role_name) {
    // TODO: this is plain dung. Why treat hardcoded default special, but for example a user-created
    // super user uses plain LOCAL_ONE?
    if (role_name == DEFAULT_USER_NAME) {
        return db::consistency_level::QUORUM;
    }
    return db::consistency_level::LOCAL_ONE;
}

std::string_view password_authenticator::qualified_java_name() const {
    return password_authenticator_name;
}

bool password_authenticator::require_authentication() const {
    return true;
}

authentication_option_set password_authenticator::supported_options() const {
    return authentication_option_set{authentication_option::password};
}

authentication_option_set password_authenticator::alterable_options() const {
    return authentication_option_set{authentication_option::password};
}

future<authenticated_user> password_authenticator::authenticate(
                const credentials_map& credentials) const {
    if (!credentials.contains(USERNAME_KEY)) {
        throw exceptions::authentication_exception(format("Required key '{}' is missing", USERNAME_KEY));
    }
    if (!credentials.contains(PASSWORD_KEY)) {
        throw exceptions::authentication_exception(format("Required key '{}' is missing", PASSWORD_KEY));
    }

    auto& username = credentials.at(USERNAME_KEY);
    auto& password = credentials.at(PASSWORD_KEY);

    // Here was a thread local, explicit cache of prepared statement. In normal execution this is
    // fine, but since we in testing set up and tear down system over and over, we'd start using
    // obsolete prepared statements pretty quickly.
    // Rely on query processing caching statements instead, and lets assume
    // that a map lookup string->statement is not gonna kill us much.
    return futurize_invoke([this, username, password] {
        static const sstring query = format("SELECT {} FROM {} WHERE {} = ?",
                SALTED_HASH,
                meta::roles_table::qualified_name,
                meta::roles_table::role_col_name);

        return _qp.execute_internal(
                query,
                consistency_for_user(username),
                internal_distributed_query_state(),
                {username},
                cql3::query_processor::cache_internal::yes);
    }).then_wrapped([=](future<::shared_ptr<cql3::untyped_result_set>> f) {
        try {
            auto res = f.get0();
            auto salted_hash = std::optional<sstring>();
            if (!res->empty()) {
                salted_hash = res->one().get_opt<sstring>(SALTED_HASH);
            }
            if (!salted_hash || !passwords::check(password, *salted_hash)) {
                throw exceptions::authentication_exception("Username and/or password are incorrect");
            }
            return make_ready_future<authenticated_user>(username);
        } catch (std::system_error &) {
            std::throw_with_nested(exceptions::authentication_exception("Could not verify password"));
        } catch (exceptions::request_execution_exception& e) {
            std::throw_with_nested(exceptions::authentication_exception(e.what()));
        } catch (exceptions::authentication_exception& e) {
            std::throw_with_nested(e);
        } catch (exceptions::unavailable_exception& e) {
            std::throw_with_nested(exceptions::authentication_exception(e.get_message()));
        } catch (...) {
            std::throw_with_nested(exceptions::authentication_exception("authentication failed"));
        }
    });
}

future<> password_authenticator::create(std::string_view role_name, const authentication_options& options) const {
    if (!options.password) {
        return make_ready_future<>();
    }

    return _qp.execute_internal(
            update_row_query(),
            consistency_for_user(role_name),
            internal_distributed_query_state(),
            {passwords::hash(*options.password, rng_for_salt), sstring(role_name)},
            cql3::query_processor::cache_internal::no).discard_result();
}

future<> password_authenticator::alter(std::string_view role_name, const authentication_options& options) const {
    if (!options.password) {
        return make_ready_future<>();
    }

    static const sstring query = format("UPDATE {} SET {} = ? WHERE {} = ?",
            meta::roles_table::qualified_name,
            SALTED_HASH,
            meta::roles_table::role_col_name);

    return _qp.execute_internal(
            query,
            consistency_for_user(role_name),
            internal_distributed_query_state(),
            {passwords::hash(*options.password, rng_for_salt), sstring(role_name)},
            cql3::query_processor::cache_internal::no).discard_result();
}

future<> password_authenticator::drop(std::string_view name) const {
    static const sstring query = format("DELETE {} FROM {} WHERE {} = ?",
            SALTED_HASH,
            meta::roles_table::qualified_name,
            meta::roles_table::role_col_name);

    return _qp.execute_internal(
            query, consistency_for_user(name),
            internal_distributed_query_state(),
            {sstring(name)},
            cql3::query_processor::cache_internal::no).discard_result();
}

future<custom_options> password_authenticator::query_custom_options(std::string_view role_name) const {
    return make_ready_future<custom_options>();
}

const resource_set& password_authenticator::protected_resources() const {
    static const resource_set resources({make_data_resource(meta::AUTH_KS, meta::roles_table::name)});
    return resources;
}

::shared_ptr<sasl_challenge> password_authenticator::new_sasl_challenge() const {
    return ::make_shared<plain_sasl_challenge>([this](std::string_view username, std::string_view password) {
        credentials_map credentials{};
        credentials[USERNAME_KEY] = sstring(username);
        credentials[PASSWORD_KEY] = sstring(password);
        return this->authenticate(credentials);
    });
}

}
