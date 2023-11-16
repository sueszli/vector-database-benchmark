/*
 * Copyright 2023 Redpanda Data, Inc.
 *
 * Licensed as a Redpanda Enterprise file under the Redpanda Community
 * License (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * https://github.com/redpanda-data/redpanda/blob/master/licenses/rcl.md
 */

#include "security/audit/audit_log_manager.h"

#include "cluster/controller.h"
#include "cluster/ephemeral_credential_frontend.h"
#include "kafka/client/client.h"
#include "kafka/client/config_utils.h"
#include "kafka/protocol/produce.h"
#include "kafka/protocol/schemata/produce_response.h"
#include "kafka/server/handlers/topics/types.h"
#include "security/acl.h"
#include "security/audit/client_probe.h"
#include "security/audit/logger.h"
#include "security/audit/schemas/application_activity.h"
#include "security/audit/schemas/types.h"
#include "security/ephemeral_credential_store.h"
#include "storage/parser_utils.h"
#include "utils/retry.h"

#include <seastar/core/sleep.hh>
#include <seastar/core/smp.hh>
#include <seastar/coroutine/maybe_yield.hh>

#include <memory>

namespace security::audit {

static constexpr std::string_view subsystem_name = "Audit System";

std::ostream& operator<<(std::ostream& os, event_type t) {
    switch (t) {
    case event_type::management:
        return os << "management";
    case event_type::produce:
        return os << "produce";
    case event_type::consume:
        return os << "consume";
    case event_type::describe:
        return os << "describe";
    case event_type::heartbeat:
        return os << "heartbeat";
    case event_type::authenticate:
        return os << "authenticated";
    case event_type::unknown:
        return os << "unknown";
    default:
        return os << "invalid";
    }
}

class audit_sink;

/// Contains a kafka client and a sempahore to bound the memory allocated
/// by it. This class may be allocated/deallocated on the owning shard depending
/// on the value of the global audit toggle config option (audit_enabled)
class audit_client {
public:
    audit_client(
      audit_sink* sink, cluster::controller*, kafka::client::configuration&);

    /// Initializes the client (with all necessary auth) and connects to the
    /// remote broker. If successful requests to create audit topic and all
    /// necessary ACLs will be made. produce() cannot be called until
    /// initialization completes with success.
    ss::future<> initialize();

    /// Closes the gate and flushes all buffers before severing connection to
    /// broker(s)
    ss::future<> shutdown();

    /// Produces to the audit topic, internal partitioner assigns partitions
    /// to the batches provided. Blocks if semaphore is exhausted.
    ss::future<>
    produce(std::vector<kafka::client::record_essence>, audit_probe& probe);

    bool is_initialized() const { return _is_initialized; }

private:
    ss::future<> update_status(kafka::error_code);
    ss::future<> update_status(kafka::produce_response);
    ss::future<> configure();
    ss::future<> mitigate_error(std::exception_ptr);
    ss::future<> create_internal_topic();
    ss::future<> set_auditing_permissions();
    ss::future<> inform(model::node_id id);
    ss::future<> do_inform(model::node_id id);
    ss::future<> set_client_credentials();

private:
    kafka::error_code _last_errc{kafka::error_code::unknown_server_error};
    ss::abort_source _as;
    ss::gate _gate;
    bool _is_initialized{false};
    size_t _max_buffer_size;
    ssx::semaphore _send_sem;
    kafka::client::client _client;
    audit_sink* _sink;
    cluster::controller* _controller;
    std::unique_ptr<client_probe> _probe;
};

/// Allocated only on the shard responsible for owning the kafka client, its
/// lifetime is the duration of the audit_log_manager. Contains a gate/mutex to
/// synchronize around actions around the internal client which may be
/// started/stopped on demand.
class audit_sink {
public:
    using auth_misconfigured_t = ss::bool_class<struct auth_misconfigured_tag>;

    audit_sink(
      audit_log_manager* audit_mgr,
      cluster::controller* controller,
      kafka::client::configuration& config) noexcept;

    /// Starts a kafka::client if none is allocated, backgrounds the work
    ss::future<> start();

    /// Closes all gates, deallocates client returns when all has completed
    ss::future<> stop();

    /// Produce to the audit topic within the context of the internal locks,
    /// ensuring toggling of the audit master switch happens in lock step with
    /// calls to produce()
    ss::future<> produce(std::vector<kafka::client::record_essence> records);

    /// Allocates and connects, or deallocates and shuts down the audit client
    void toggle(bool enabled);

    /// Returns true if _client has a value
    bool is_enabled() const { return _client != nullptr; }

private:
    ss::future<> update_auth_status(auth_misconfigured_t);

    ss::future<> do_toggle(bool enabled);

    /// Primitives for ensuring background work and toggling of switch w/ async
    /// work occur in lock step
    ss::gate _gate;
    mutex _mutex;

    /// Reference to audit manager so synchronization with its fibers may occur.
    /// Supports pausing and resuming these fibers so the client can safely be
    /// deallocated when no more work is concurrently entering the system
    audit_log_manager* _audit_mgr;

    /// audit_client and members necessary to pass to its constructor
    std::unique_ptr<audit_client> _client;
    cluster::controller* _controller;
    kafka::client::configuration& _config;

    friend class audit_client;
};

audit_client::audit_client(
  audit_sink* sink,
  cluster::controller* controller,
  kafka::client::configuration& client_config)
  : _max_buffer_size(config::shard_local_cfg().audit_client_max_buffer_size())
  , _send_sem(_max_buffer_size, "audit_log_producer_semaphore")
  , _client(
      config::to_yaml(client_config, config::redact_secrets::no),
      [this](std::exception_ptr eptr) { return mitigate_error(eptr); })
  , _sink(sink)
  , _controller(controller) {}

ss::future<> audit_client::initialize() {
    static const auto base_backoff = 250ms;
    exp_backoff_policy backoff_policy;
    while (!_as.abort_requested()) {
        try {
            co_await configure();
            _is_initialized = true;
            break;
        } catch (...) {
            /// Sleep, then try again
        }
        auto next = backoff_policy.next_backoff();
        co_await ss::sleep_abortable(base_backoff * next, _as)
          .handle_exception_type([](const ss::sleep_aborted&) {});
    }
    if (_is_initialized) {
        _probe = std::make_unique<client_probe>();
        _probe->setup_metrics([this]() {
            return 1.0
                   - (static_cast<double>(_send_sem.available_units()) / static_cast<double>(_max_buffer_size));
        });
    }
}

ss::future<> audit_client::set_client_credentials() {
    /// Set ephemeral credential
    auto& frontend = _controller->get_ephemeral_credential_frontend().local();
    auto pw = co_await frontend.get(audit_principal);
    if (pw.err != cluster::errc::success) {
        throw std::runtime_error(fmt::format(
          "Failed to fetch credential for principal: {}", audit_principal));
    }

    _client.config().sasl_mechanism.set_value(pw.credential.mechanism());
    _client.config().scram_username.set_value(pw.credential.user()());
    _client.config().scram_password.set_value(pw.credential.password()());
}

ss::future<> audit_client::configure() {
    try {
        co_await set_client_credentials();
        co_await set_auditing_permissions();
        co_await create_internal_topic();

        /// Retries should be "infinite", to avoid dropping data, there is a
        /// known issue within the client setting this value to size_t::max
        _client.config().retries.set_value(10000);
        vlog(adtlog.info, "Audit log client initialized");
    } catch (...) {
        vlog(
          adtlog.warn,
          "Audit log client failed to initialize: {}",
          std::current_exception());
        throw;
    }
}

ss::future<> audit_client::set_auditing_permissions() {
    /// Give permissions to create and write to the audit topic
    security::acl_entry acl_create_entry{
      audit_principal,
      security::acl_host::wildcard_host(),
      security::acl_operation::create,
      security::acl_permission::allow};

    security::acl_entry acl_write_entry{
      audit_principal,
      security::acl_host::wildcard_host(),
      security::acl_operation::write,
      security::acl_permission::allow};

    security::resource_pattern audit_topic_pattern{
      security::resource_type::topic,
      model::kafka_audit_logging_topic,
      security::pattern_type::literal};

    co_await _controller->get_security_frontend().local().create_acls(
      {security::acl_binding{audit_topic_pattern, acl_create_entry},
       security::acl_binding{audit_topic_pattern, acl_write_entry}},
      5s);
}

/// `update_auth_status` should not be called frequently since this method
/// occurs on the hot path and calls to `update_auth_status` call will boil down
/// to an invoke_on_all() call. Conditionals are wrapped around the call to
/// `update_auth_status` so that its only called when the errc changes to/from
/// a desired condition.
ss::future<> audit_client::update_status(kafka::error_code errc) {
    /// If the status changed to erraneous from anything else
    if (errc == kafka::error_code::illegal_sasl_state) {
        if (_last_errc != kafka::error_code::illegal_sasl_state) {
            co_await _sink->update_auth_status(
              audit_sink::auth_misconfigured_t::yes);
        }
    } else if (_last_errc == kafka::error_code::illegal_sasl_state) {
        /// The status changed from erraneous to anything else
        if (errc != kafka::error_code::illegal_sasl_state) {
            co_await _sink->update_auth_status(
              audit_sink::auth_misconfigured_t::no);
        }
    }
    _last_errc = errc;
}

ss::future<> audit_client::update_status(kafka::produce_response response) {
    /// This method should almost always call update_status() with a value of
    /// no error code. That is because kafka client mitigation will be called in
    /// the case there is a produce error, and an erraneous response will only
    /// be returned when the retry count is exhausted, which will never occur
    /// since it is artificially set high to have the effect of always retrying
    absl::flat_hash_set<kafka::error_code> errcs;
    for (const auto& topic_response : response.data.responses) {
        for (const auto& partition_response : topic_response.partitions) {
            errcs.emplace(partition_response.error_code);
        }
    }
    if (errcs.empty()) {
        vlog(seclog.warn, "Empty produce response recieved");
        co_return;
    }
    auto errc = *errcs.begin();
    if (errcs.contains(kafka::error_code::illegal_sasl_state)) {
        errc = kafka::error_code::illegal_sasl_state;
    }
    co_await update_status(errc);
}

ss::future<> audit_client::mitigate_error(std::exception_ptr eptr) {
    if (_gate.is_closed() || _as.abort_requested()) {
        /// TODO: Investigate looping behavior on shutdown
        co_return;
    }
    vlog(adtlog.trace, "mitigate_error: {}", eptr);
    auto f = ss::now();
    try {
        std::rethrow_exception(eptr);
    } catch (kafka::client::broker_error const& ex) {
        f = update_status(ex.error);
        if (ex.error == kafka::error_code::sasl_authentication_failed) {
            f = f.then([this, ex]() {
                return inform(ex.node_id).then([this]() {
                    return _client.connect();
                });
            });
        } else {
            throw;
        }
    } catch (...) {
        throw;
    }
    co_await std::move(f);
}

ss::future<> audit_client::inform(model::node_id id) {
    vlog(adtlog.trace, "inform: {}", id);

    // Inform a particular node
    if (id != kafka::client::unknown_node_id) {
        return do_inform(id);
    }

    // Inform all nodes
    return seastar::parallel_for_each(
      _controller->get_members_table().local().node_ids(),
      [this](model::node_id id) { return do_inform(id); });
}

ss::future<> audit_client::do_inform(model::node_id id) {
    auto& fe = _controller->get_ephemeral_credential_frontend().local();
    auto ec = co_await fe.inform(id, audit_principal);
    vlog(adtlog.info, "Informed: broker: {}, ec: {}", id, ec);
}

ss::future<> audit_client::create_internal_topic() {
    constexpr std::string_view retain_forever = "-1";
    constexpr std::string_view seven_days = "604800000";
    kafka::creatable_topic audit_topic{
      .name = model::kafka_audit_logging_topic,
      .num_partitions = config::shard_local_cfg().audit_log_num_partitions(),
      .replication_factor
      = config::shard_local_cfg().audit_log_replication_factor(),
      .assignments = {},
      .configs = {
        kafka::createable_topic_config{
          .name = ss::sstring(kafka::topic_property_retention_bytes),
          .value{retain_forever}},
        kafka::createable_topic_config{
          .name = ss::sstring(kafka::topic_property_retention_duration),
          .value{seven_days}}}};
    vlog(
      adtlog.info, "Creating audit log topic with settings: {}", audit_topic);
    const auto resp = co_await _client.create_topic({std::move(audit_topic)});
    if (resp.data.topics.size() != 1) {
        throw std::runtime_error(
          fmt::format("Unexpected create topics response: {}", resp.data));
    }
    const auto& topic = resp.data.topics[0];
    if (topic.error_code == kafka::error_code::none) {
        vlog(adtlog.debug, "Auditing: created audit log topic: {}", topic);
    } else if (topic.error_code == kafka::error_code::topic_already_exists) {
        vlog(adtlog.debug, "Auditing: topic already exists");
        co_await _client.update_metadata();
    } else {
        if (topic.error_code == kafka::error_code::invalid_replication_factor) {
            vlog(
              adtlog.warn,
              "Auditing: invalid replication factor on audit topic, "
              "check/modify settings, then disable and re-enable "
              "'audit_enabled'");
        }
        const auto msg = topic.error_message.has_value() ? *topic.error_message
                                                         : "<no_err_msg>";
        throw std::runtime_error(
          fmt::format("{} - error_code: {}", msg, topic.error_code));
    }
}

ss::future<> audit_client::shutdown() {
    /// Repeated calls to shutdown() are possible, although unlikely
    if (_as.abort_requested()) {
        co_return;
    }
    vlog(adtlog.info, "Shutting down audit client");
    _as.request_abort();
    _send_sem.broken();
    co_await _client.stop();
    co_await _gate.close();
    _probe.reset(nullptr);
    vlog(adtlog.info, "Audit client stopped");
}

ss::future<> audit_client::produce(
  std::vector<kafka::client::record_essence> records, audit_probe& probe) {
    /// TODO: Produce with acks=1, atm -1 is hardcoded into client
    const auto records_size = [](const auto& records) {
        std::size_t size = 0;
        for (const auto& r : records) {
            if (r.value) {
                /// auditing does not fill in any of the fields of the
                /// record_essence other then the value member
                size += r.value->size_bytes();
            }
        }
        return size;
    };

    try {
        const auto size_bytes = records_size(records);
        vlog(
          adtlog.trace,
          "Obtaining {} units from auditing semaphore",
          size_bytes);
        auto units = co_await ss::get_units(_send_sem, size_bytes);
        ssx::spawn_with_gate(
          _gate,
          [this,
           &probe,
           units = std::move(units),
           records = std::move(records)]() mutable {
              const auto n_records = records.size();
              return _client
                .produce_records(
                  model::kafka_audit_logging_topic, std::move(records))
                .then([this, n_records, &probe](kafka::produce_response r) {
                    bool errored = std::any_of(
                      r.data.responses.cbegin(),
                      r.data.responses.cend(),
                      [](const kafka::topic_produce_response& tp) {
                          return std::any_of(
                            tp.partitions.cbegin(),
                            tp.partitions.cend(),
                            [](const kafka::partition_produce_response& p) {
                                return p.error_code != kafka::error_code::none;
                            });
                      });
                    if (errored) {
                        if (_as.abort_requested()) {
                            vlog(
                              adtlog.warn,
                              "{} audit records dropped, shutting down",
                              n_records);
                        } else {
                            vlog(
                              adtlog.error,
                              "{} audit records dropped",
                              n_records);
                        }
                        probe.audit_error();
                    } else {
                        probe.audit_event();
                    }
                    return update_status(std::move(r));
                })
                .finally([units = std::move(units)] {});
          });
    } catch (const ss::broken_semaphore&) {
        vlog(
          adtlog.debug,
          "Shutting down the auditor kafka::client, semaphore broken");
    }
    co_return;
}

/// audit_sink

audit_sink::audit_sink(
  audit_log_manager* audit_mgr,
  cluster::controller* controller,
  kafka::client::configuration& config) noexcept
  : _audit_mgr(audit_mgr)
  , _controller(controller)
  , _config(config) {}

ss::future<> audit_sink::start() {
    toggle(true);
    return ss::now();
}

ss::future<> audit_sink::stop() {
    _mutex.broken();
    if (_client) {
        co_await _client->shutdown();
    }
    co_await _gate.close();
}

ss::future<>
audit_sink::update_auth_status(auth_misconfigured_t auth_misconfigured) {
    return _audit_mgr->container().invoke_on_all(
      [auth_misconfigured](audit_log_manager& mgr) {
          mgr._auth_misconfigured = (bool)auth_misconfigured;
      });
}

ss::future<>
audit_sink::produce(std::vector<kafka::client::record_essence> records) {
    /// No locks/gates since the calls to this method are done in controlled
    /// context of other synchronization primitives
    vassert(_client, "produce() called on a null client");
    co_await _client->produce(std::move(records), _audit_mgr->probe());
}

void audit_sink::toggle(bool enabled) {
    ssx::spawn_with_gate(
      _gate, [this, enabled]() { return do_toggle(enabled); });
}

ss::future<> audit_sink::do_toggle(bool enabled) {
    try {
        ssx::semaphore_units lock;
        if (enabled && !_client) {
            lock = co_await _mutex.get_units();
            _client = std::make_unique<audit_client>(
              this, _controller, _config);
            co_await _client->initialize();
            if (_client->is_initialized()) {
                /// Only if shutdown succeeded before initializtion could
                /// complete would this case evaluate to false
                co_await _audit_mgr->resume();
            }
        } else if (!enabled && _client) {
            /// Call to shutdown does not exist within the lock so that
            /// shutting down isn't blocked on the lock held above in the
            /// case initialize() doesn't complete. This is common if for
            /// example the audit topic is improperly configured
            /// intitialization will forever hang.
            co_await _client->shutdown();
            lock = co_await _mutex.get_units();
            co_await _audit_mgr->pause();
            _client.reset(nullptr);
        }
    } catch (const ss::broken_semaphore&) {
        vlog(adtlog.info, "Failed to toggle audit status, shutting down");
    } catch (...) {
        vlog(
          adtlog.error,
          "Failed to toggle audit status: {}",
          std::current_exception());
    }
}

/// audit_log_manager

void audit_log_manager::set_enabled_events() {
    using underlying_enum_t = std::underlying_type_t<event_type>;
    _enabled_event_types = underlying_enum_t(0);
    for (const auto& e : _audit_event_types()) {
        const auto as_uint = underlying_enum_t(string_to_event_type(e));
        _enabled_event_types[as_uint] = true;
    }
    vassert(
      !is_audit_event_enabled(event_type::unknown),
      "Unknown event_type observed");
}

audit_log_manager::audit_log_manager(
  cluster::controller* controller, kafka::client::configuration& client_config)
  : _audit_enabled(config::shard_local_cfg().audit_enabled.bind())
  , _queue_drain_interval_ms(
      config::shard_local_cfg().audit_queue_drain_interval_ms.bind())
  , _max_queue_elements_per_shard(
      config::shard_local_cfg().audit_max_queue_elements_per_shard.bind())
  , _audit_event_types(
      config::shard_local_cfg().audit_enabled_event_types.bind())
  , _controller(controller)
  , _config(client_config) {
    if (ss::this_shard_id() == client_shard_id) {
        _sink = std::make_unique<audit_sink>(this, controller, client_config);
    }

    _drain_timer.set_callback([this] {
        ssx::spawn_with_gate(_gate, [this]() {
            return ss::get_units(_active_drain, 1)
              .then([this](auto units) mutable {
                  return drain()
                    .handle_exception([&probe = probe()](std::exception_ptr e) {
                        vlog(
                          adtlog.warn,
                          "Exception in audit_log_manager fiber: {}",
                          e);
                        probe.audit_error();
                    })
                    .finally([this, units = std::move(units)] {
                        _drain_timer.arm(_queue_drain_interval_ms());
                    });
              });
        });
    });
    set_enabled_events();
    _audit_event_types.watch([this] { set_enabled_events(); });
}

audit_log_manager::~audit_log_manager() = default;

bool audit_log_manager::is_audit_event_enabled(event_type event_type) const {
    using underlying_enum_t
      = std::underlying_type_t<security::audit::event_type>;
    return _enabled_event_types.test(underlying_enum_t(event_type));
}

ss::future<> audit_log_manager::start() {
    _probe = std::make_unique<audit_probe>();
    _probe->setup_metrics([this] {
        return static_cast<double>(pending_events())
               / static_cast<double>(_max_queue_elements_per_shard());
    });
    if (ss::this_shard_id() != client_shard_id) {
        co_return;
    }
    _audit_enabled.watch([this] {
        try {
            if (!enqueue_app_lifecycle_event(
                  _audit_enabled() ? application_lifecycle::activity_id::start
                                   : application_lifecycle::activity_id::stop,
                  ss::sstring{subsystem_name})) {
                vlog(adtlog.error, "Failed to enqueue audit lifecycle event");
                if (_audit_enabled()) {
                    throw std::runtime_error(
                      "Failed to enqueue audit lifecycle event for audit "
                      "system start");
                }
            }
            _sink->toggle(_audit_enabled());
        } catch (const ss::gate_closed_exception&) {
            vlog(
              adtlog.debug, "Failed to toggle auditing state, shutting down");
        } catch (...) {
            vlog(
              adtlog.error,
              "Failed to toggle auditing state: {}",
              std::current_exception());
        }
    });
    if (_audit_enabled()) {
        vlog(adtlog.info, "Starting audit_log_manager");
        if (!this->enqueue_app_lifecycle_event(
              application_lifecycle::activity_id::start,
              ss::sstring{subsystem_name})) {
            vlog(adtlog.error, "Failed to enqueue audit lifecycle start event");
            // TODO I should error here, yeah?
        }
        co_await _sink->start();
    }
}

ss::future<> audit_log_manager::stop() {
    if (ss::this_shard_id() == client_shard_id) {
        if (!enqueue_app_lifecycle_event(
              application_lifecycle::activity_id::stop,
              ss::sstring{subsystem_name})) {
            vlog(
              adtlog.error, "Failed to enqueue audit subsystem shutdown event");
        }
    }
    _drain_timer.cancel();
    _as.request_abort();
    if (ss::this_shard_id() == client_shard_id) {
        vlog(adtlog.info, "Shutting down audit log manager");
        co_await _sink->stop();
    }
    if (!_gate.is_closed()) {
        /// Gate may already be closed if ::pause() had been called
        co_await _gate.close();
    }
    _probe.reset(nullptr);
    if (_queue.size() > 0) {
        vlog(
          adtlog.debug,
          "{} records were not pushed to the audit log before shutdown",
          _queue.size());
    }
}

ss::future<> audit_log_manager::pause() {
    return container().invoke_on_all([](audit_log_manager& mgr) {
        /// Wait until drain() has completed, with timer cancelled it can be
        /// ensured no more work will be performed
        return ss::get_units(mgr._active_drain, 1).then([&mgr](auto) {
            mgr._drain_timer.cancel();
        });
    });
}

ss::future<> audit_log_manager::resume() {
    return container().invoke_on_all([](audit_log_manager& mgr) {
        /// If the timer is already armed that is a bug
        vassert(
          !mgr._drain_timer.armed(),
          "Timer is already armed upon call to ::resume");
        mgr._drain_timer.arm(mgr._queue_drain_interval_ms());
    });
}

bool audit_log_manager::is_client_enabled() const {
    vassert(
      ss::this_shard_id() == client_shard_id,
      "Must be called on audit client shard");
    return _sink->is_enabled();
}

bool audit_log_manager::report_redpanda_app_event(is_started app_started) {
    return enqueue_app_lifecycle_event(
      app_started == is_started::yes
        ? application_lifecycle::activity_id::start
        : application_lifecycle::activity_id::stop);
}

bool audit_log_manager::do_enqueue_audit_event(
  std::unique_ptr<security::audit::ocsf_base_impl> msg) {
    auto& map = _queue.get<underlying_unordered_map>();
    auto it = map.find(msg->key());
    if (it == map.end()) {
        if (_queue.size() >= _max_queue_elements_per_shard()) {
            vlog(
              adtlog.warn,
              "Unable to enqueue audit message: {} >= {}",
              _queue.size(),
              _max_queue_elements_per_shard());
            probe().audit_error();
            return false;
        }
        auto& list = _queue.get<underlying_list>();
        vlog(adtlog.trace, "Successfully enqueued audit event {}", *msg);
        list.push_back(std::move(msg));
    } else {
        vlog(adtlog.trace, "incrementing count of event {}", *msg);
        auto now = security::audit::timestamp_t{
          std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count()};
        (*it)->increment(now);
    }
    return true;
}

ss::future<> audit_log_manager::drain() {
    if (_queue.empty() || _as.abort_requested()) {
        co_return;
    }

    vlog(
      adtlog.debug,
      "Attempting to drain {} audit events from sharded queue",
      _queue.size());

    /// Combine all batched audit msgs into record_essences
    std::vector<kafka::client::record_essence> essences;
    auto records = std::exchange(_queue, underlying_t{});
    auto& records_seq = records.get<underlying_list>();
    while (!records_seq.empty()) {
        const auto& front = records_seq.front();
        auto as_json = front->to_json();
        records_seq.pop_front();
        iobuf b;
        b.append(as_json.c_str(), as_json.size());
        essences.push_back(
          kafka::client::record_essence{.value = std::move(b)});
        co_await ss::maybe_yield();
    }

    /// This call may block if the audit_clients semaphore is exhausted,
    /// this represents the amount of memory used within its kafka::client
    /// produce batch queue. If the semaphore blocks it will apply
    /// backpressure here, and the \ref _queue will begin to fill closer to
    /// capacity. When it hits capacity, enqueue_audit_event() will block.
    co_await container().invoke_on(
      client_shard_id,
      [recs = std::move(essences)](audit_log_manager& mgr) mutable {
          return mgr._sink->produce(std::move(recs));
      });
}

std::optional<audit_log_manager::audit_event_passthrough>
audit_log_manager::should_enqueue_audit_event() const {
    if (!_audit_enabled()) {
        return std::make_optional(audit_event_passthrough::yes);
    }
    if (_as.abort_requested()) {
        /// Prevent auditing new messages when shutdown starts that way the
        /// queue may be entirely flushed before shutdown
        return std::make_optional(audit_event_passthrough::no);
    }
    if (_auth_misconfigured) {
        /// Audit logging depends on having auth enabled, if it is not
        /// then messages are rejected for increased observability into why
        /// things are not working.
        vlog(
          adtlog.warn,
          "Audit message rejected due to misconfigured authorization");
        return std::make_optional(audit_event_passthrough::no);
    }
    return std::nullopt;
}

std::optional<audit_log_manager::audit_event_passthrough>
audit_log_manager::should_enqueue_audit_event(kafka::api_key api) const {
    if (auto val = should_enqueue_audit_event(); val.has_value()) {
        return val;
    }
    if (!is_audit_event_enabled(kafka_api_to_event_type(api))) {
        return std::make_optional(audit_event_passthrough::yes);
    }
    return std::nullopt;
}

std::optional<audit_log_manager::audit_event_passthrough>
audit_log_manager::should_enqueue_audit_event(event_type type) const {
    if (auto val = should_enqueue_audit_event(); val.has_value()) {
        return val;
    }
    if (!is_audit_event_enabled(type)) {
        return std::make_optional(audit_event_passthrough::yes);
    }
    return std::nullopt;
}

} // namespace security::audit
