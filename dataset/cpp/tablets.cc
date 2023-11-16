/*
 * Copyright (C) 2023-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "locator/tablet_replication_strategy.hh"
#include "locator/tablets.hh"
#include "locator/tablet_metadata_guard.hh"
#include "locator/tablet_sharder.hh"
#include "locator/token_range_splitter.hh"
#include "types/types.hh"
#include "types/tuple.hh"
#include "types/set.hh"
#include "utils/hash.hh"
#include "db/system_keyspace.hh"
#include "cql3/query_processor.hh"
#include "cql3/untyped_result_set.hh"
#include "replica/database.hh"
#include "utils/stall_free.hh"

#include <seastar/core/coroutine.hh>
#include <seastar/coroutine/maybe_yield.hh>

namespace locator {

seastar::logger tablet_logger("tablets");


static
write_replica_set_selector get_selector_for_writes(tablet_transition_stage stage) {
    switch (stage) {
        case tablet_transition_stage::allow_write_both_read_old:
            return write_replica_set_selector::previous;
        case tablet_transition_stage::write_both_read_old:
            return write_replica_set_selector::both;
        case tablet_transition_stage::streaming:
            return write_replica_set_selector::both;
        case tablet_transition_stage::write_both_read_new:
            return write_replica_set_selector::both;
        case tablet_transition_stage::use_new:
            return write_replica_set_selector::next;
        case tablet_transition_stage::cleanup:
            return write_replica_set_selector::next;
        case tablet_transition_stage::end_migration:
            return write_replica_set_selector::next;
    }
    on_internal_error(tablet_logger, format("Invalid tablet transition stage: {}", static_cast<int>(stage)));
}

static
read_replica_set_selector get_selector_for_reads(tablet_transition_stage stage) {
    switch (stage) {
        case tablet_transition_stage::allow_write_both_read_old:
            return read_replica_set_selector::previous;
        case tablet_transition_stage::write_both_read_old:
            return read_replica_set_selector::previous;
        case tablet_transition_stage::streaming:
            return read_replica_set_selector::previous;
        case tablet_transition_stage::write_both_read_new:
            return read_replica_set_selector::next;
        case tablet_transition_stage::use_new:
            return read_replica_set_selector::next;
        case tablet_transition_stage::cleanup:
            return read_replica_set_selector::next;
        case tablet_transition_stage::end_migration:
            return read_replica_set_selector::next;
    }
    on_internal_error(tablet_logger, format("Invalid tablet transition stage: {}", static_cast<int>(stage)));
}

tablet_transition_info::tablet_transition_info(tablet_transition_stage stage, tablet_replica_set next, tablet_replica pending_replica)
    : stage(stage)
    , next(std::move(next))
    , pending_replica(std::move(pending_replica))
    , writes(get_selector_for_writes(stage))
    , reads(get_selector_for_reads(stage))
{ }

tablet_migration_streaming_info get_migration_streaming_info(const tablet_info& tinfo, const tablet_transition_info& trinfo) {
    tablet_migration_streaming_info result = {
        .read_from = std::unordered_set<tablet_replica>(tinfo.replicas.begin(), tinfo.replicas.end()),
        .written_to = std::unordered_set<tablet_replica>(trinfo.next.begin(), trinfo.next.end())
    };
    for (auto&& r : trinfo.next) {
        result.read_from.erase(r);
    }
    for (auto&& r : tinfo.replicas) {
        result.written_to.erase(r);
    }
    return result;
}

tablet_replica get_leaving_replica(const tablet_info& tinfo, const tablet_transition_info& trinfo) {
    std::unordered_set<tablet_replica> leaving(tinfo.replicas.begin(), tinfo.replicas.end());
    for (auto&& r : trinfo.next) {
        leaving.erase(r);
    }
    if (leaving.empty()) {
        throw std::runtime_error(format("No leaving replicas"));
    }
    if (leaving.size() > 1) {
        throw std::runtime_error(format("More than one leaving replica"));
    }
    return *leaving.begin();
}

const tablet_map& tablet_metadata::get_tablet_map(table_id id) const {
    try {
        return _tablets.at(id);
    } catch (const std::out_of_range&) {
        throw std::runtime_error(format("Tablet map not found for table {}", id));
    }
}

tablet_map& tablet_metadata::get_tablet_map(table_id id) {
    return const_cast<tablet_map&>(
        const_cast<const tablet_metadata*>(this)->get_tablet_map(id));
}

void tablet_metadata::set_tablet_map(table_id id, tablet_map map) {
    _tablets.insert_or_assign(id, std::move(map));
}

future<> tablet_metadata::clear_gently() {
    for (auto&& [id, map] : _tablets) {
        co_await map.clear_gently();
    }
    co_return;
}

tablet_map::tablet_map(size_t tablet_count)
        : _log2_tablets(log2ceil(tablet_count)) {
    if (tablet_count != 1ul << _log2_tablets) {
        on_internal_error(tablet_logger, format("Tablet count not a power of 2: {}", tablet_count));
    }
    _tablets.resize(tablet_count);
}

void tablet_map::check_tablet_id(tablet_id id) const {
    if (size_t(id) >= tablet_count()) {
        throw std::logic_error(format("Invalid tablet id: {} >= {}", id, tablet_count()));
    }
}

const tablet_info& tablet_map::get_tablet_info(tablet_id id) const {
    check_tablet_id(id);
    return _tablets[size_t(id)];
}

tablet_id tablet_map::get_tablet_id(token t) const {
    return tablet_id(dht::compaction_group_of(_log2_tablets, t));
}

dht::token tablet_map::get_last_token(tablet_id id) const {
    check_tablet_id(id);
    return dht::last_token_of_compaction_group(_log2_tablets, size_t(id));
}

dht::token tablet_map::get_first_token(tablet_id id) const {
    if (id == first_tablet()) {
        return dht::first_token();
    } else {
        return dht::next_token(get_last_token(tablet_id(size_t(id) - 1)));
    }
}

dht::token_range tablet_map::get_token_range(tablet_id id) const {
    if (id == first_tablet()) {
        return dht::token_range::make({dht::minimum_token(), false}, {get_last_token(id), true});
    } else {
        return dht::token_range::make({get_last_token(tablet_id(size_t(id) - 1)), false}, {get_last_token(id), true});
    }
}

void tablet_map::set_tablet(tablet_id id, tablet_info info) {
    check_tablet_id(id);
    _tablets[size_t(id)] = std::move(info);
}

void tablet_map::set_tablet_transition_info(tablet_id id, tablet_transition_info info) {
    check_tablet_id(id);
    _transitions.insert_or_assign(id, std::move(info));
}

future<> tablet_map::for_each_tablet(seastar::noncopyable_function<void(tablet_id, const tablet_info&)> func) const {
    std::optional<tablet_id> tid = first_tablet();
    for (const tablet_info& ti : tablets()) {
        co_await coroutine::maybe_yield();
        func(*tid, ti);
        tid = next_tablet(*tid);
    }
}

void tablet_map::clear_transitions() {
    _transitions.clear();
}

std::optional<shard_id> tablet_map::get_shard(tablet_id tid, host_id host) const {
    auto&& info = get_tablet_info(tid);

    for (auto&& r : info.replicas) {
        if (r.host == host) {
            return r.shard;
        }
    }

    auto tinfo = get_tablet_transition_info(tid);
    if (tinfo && tinfo->pending_replica.host == host) {
        return tinfo->pending_replica.shard;
    }

    return std::nullopt;
}

future<> tablet_map::clear_gently() {
    return utils::clear_gently(_tablets);
}

const tablet_transition_info* tablet_map::get_tablet_transition_info(tablet_id id) const {
    auto i = _transitions.find(id);
    if (i == _transitions.end()) {
        return nullptr;
    }
    return &i->second;
}

// The names are persisted in system tables so should not be changed.
static const std::unordered_map<tablet_transition_stage, sstring> tablet_transition_stage_to_name = {
    {tablet_transition_stage::allow_write_both_read_old, "allow_write_both_read_old"},
    {tablet_transition_stage::write_both_read_old, "write_both_read_old"},
    {tablet_transition_stage::write_both_read_new, "write_both_read_new"},
    {tablet_transition_stage::streaming, "streaming"},
    {tablet_transition_stage::use_new, "use_new"},
    {tablet_transition_stage::cleanup, "cleanup"},
    {tablet_transition_stage::end_migration, "end_migration"},
};

static const std::unordered_map<sstring, tablet_transition_stage> tablet_transition_stage_from_name = std::invoke([] {
    std::unordered_map<sstring, tablet_transition_stage> result;
    for (auto&& [v, s] : tablet_transition_stage_to_name) {
        result.emplace(s, v);
    }
    return result;
});

sstring tablet_transition_stage_to_string(tablet_transition_stage stage) {
    auto i = tablet_transition_stage_to_name.find(stage);
    if (i == tablet_transition_stage_to_name.end()) {
        on_internal_error(tablet_logger, format("Invalid tablet transition stage: {}", static_cast<int>(stage)));
    }
    return i->second;
}

tablet_transition_stage tablet_transition_stage_from_string(const sstring& name) {
    return tablet_transition_stage_from_name.at(name);
}

std::ostream& operator<<(std::ostream& out, tablet_id id) {
    return out << size_t(id);
}

std::ostream& operator<<(std::ostream& out, const tablet_replica& r) {
    return out << r.host << ":" << r.shard;
}

std::ostream& operator<<(std::ostream& out, const tablet_map& r) {
    if (r.tablet_count() == 0) {
        return out << "{}";
    }
    out << "{";
    bool first = true;
    tablet_id tid = r.first_tablet();
    for (auto&& tablet : r._tablets) {
        if (!first) {
            out << ",";
        }
        out << format("\n    [{}]: last_token={}, replicas={}", tid, r.get_last_token(tid), tablet.replicas);
        if (auto tr = r.get_tablet_transition_info(tid)) {
            out << format(", stage={}, new_replicas={}, pending={}", tr->stage, tr->next, tr->pending_replica);
        }
        first = false;
        tid = *r.next_tablet(tid);
    }
    return out << "\n  }";
}

std::ostream& operator<<(std::ostream& out, const tablet_metadata& tm) {
    out << "{";
    bool first = true;
    for (auto&& [id, map] : tm._tablets) {
        if (!first) {
            out << ",";
        }
        out << "\n  " << id << ": " << map;
        first = false;
    }
    return out << "\n}";
}

size_t tablet_map::external_memory_usage() const {
    size_t result = _tablets.external_memory_usage();
    for (auto&& tablet : _tablets) {
        result += tablet.replicas.external_memory_usage();
    }
    return result;
}

// Estimates the external memory usage of std::unordered_map<>.
// Does not include external memory usage of elements.
template <typename K, typename V>
static size_t estimate_external_memory_usage(const std::unordered_map<K, V>& map) {
    return map.bucket_count() * sizeof(void*) + map.size() * (sizeof(std::pair<const K, V>) + 8);
}

size_t tablet_metadata::external_memory_usage() const {
    size_t result = estimate_external_memory_usage(_tablets);
    for (auto&& [id, map] : _tablets) {
        result += map.external_memory_usage();
    }
    return result;
}

class tablet_effective_replication_map : public effective_replication_map {
    table_id _table;
    tablet_sharder _sharder;
private:
    gms::inet_address get_endpoint_for_host_id(host_id host) const {
        auto endpoint_opt = _tmptr->get_endpoint_for_host_id(host);
        if (!endpoint_opt) {
            on_internal_error(tablet_logger, format("Host ID {} not found in the cluster", host));
        }
        return *endpoint_opt;
    }
    inet_address_vector_replica_set to_replica_set(const tablet_replica_set& replicas) const {
        inet_address_vector_replica_set result;
        result.reserve(replicas.size());
        for (auto&& replica : replicas) {
            result.emplace_back(get_endpoint_for_host_id(replica.host));
        }
        return result;
    }
    const tablet_map& get_tablet_map() const {
        return _tmptr->tablets().get_tablet_map(_table);
    }
public:
    tablet_effective_replication_map(table_id table,
                                     replication_strategy_ptr rs,
                                     token_metadata_ptr tmptr,
                                     size_t replication_factor)
            : effective_replication_map(std::move(rs), std::move(tmptr), replication_factor)
            , _table(table)
            , _sharder(*_tmptr, table)
    { }

    virtual ~tablet_effective_replication_map() = default;

    virtual inet_address_vector_replica_set get_natural_endpoints(const token& search_token) const override {
        auto&& tablets = get_tablet_map();
        auto tablet = tablets.get_tablet_id(search_token);
        auto* info = tablets.get_tablet_transition_info(tablet);
        auto&& replicas = std::invoke([&] () -> const tablet_replica_set& {
            if (!info) {
                return tablets.get_tablet_info(tablet).replicas;
            }
            switch (info->writes) {
                case write_replica_set_selector::previous:
                    [[fallthrough]];
                case write_replica_set_selector::both:
                    return tablets.get_tablet_info(tablet).replicas;
                case write_replica_set_selector::next: {
                    return info->next;
                }
            }
            on_internal_error(tablet_logger, format("Invalid replica selector", static_cast<int>(info->writes)));
        });
        tablet_logger.trace("get_natural_endpoints({}): table={}, tablet={}, replicas={}", search_token, _table, tablet, replicas);
        return to_replica_set(replicas);
    }

    virtual inet_address_vector_replica_set get_natural_endpoints_without_node_being_replaced(const token& search_token) const override {
        auto result = get_natural_endpoints(search_token);
        maybe_remove_node_being_replaced(*_tmptr, *_rs, result);
        return result;
    }

    virtual inet_address_vector_topology_change get_pending_endpoints(const token& search_token) const override {
        auto&& tablets = get_tablet_map();
        auto tablet = tablets.get_tablet_id(search_token);
        auto&& info = tablets.get_tablet_transition_info(tablet);
        if (!info) {
            return {};
        }
        switch (info->writes) {
            case write_replica_set_selector::previous:
                return {};
            case write_replica_set_selector::both:
                tablet_logger.trace("get_pending_endpoints({}): table={}, tablet={}, replica={}",
                                    search_token, _table, tablet, info->pending_replica);
                return {get_endpoint_for_host_id(info->pending_replica.host)};
            case write_replica_set_selector::next:
                return {};
        }
        on_internal_error(tablet_logger, format("Invalid replica selector", static_cast<int>(info->writes)));
    }

    virtual inet_address_vector_replica_set get_endpoints_for_reading(const token& search_token) const override {
        auto&& tablets = get_tablet_map();
        auto tablet = tablets.get_tablet_id(search_token);
        auto&& info = tablets.get_tablet_transition_info(tablet);
        auto&& replicas = std::invoke([&] () -> const tablet_replica_set& {
            if (!info) {
                return tablets.get_tablet_info(tablet).replicas;
            }
            switch (info->reads) {
                case read_replica_set_selector::previous:
                    return tablets.get_tablet_info(tablet).replicas;
                case read_replica_set_selector::next: {
                    return info->next;
                }
            }
            on_internal_error(tablet_logger, format("Invalid replica selector", static_cast<int>(info->reads)));
        });
        tablet_logger.trace("get_endpoints_for_reading({}): table={}, tablet={}, replicas={}", search_token, _table, tablet, replicas);
        auto result = to_replica_set(replicas);
        maybe_remove_node_being_replaced(*_tmptr, *_rs, result);
        return result;
    }

    virtual bool has_pending_ranges(inet_address endpoint) const override {
        const auto host_id = _tmptr->get_host_id_if_known(endpoint);
        if (!host_id.has_value()) {
            return false;
        }
        for (const auto& [id, transition_info]: get_tablet_map().transitions()) {
            if (transition_info.pending_replica.host == *host_id) {
                return true;
            }
        }
        return false;
    }

    virtual std::unique_ptr<token_range_splitter> make_splitter() const override {
        class splitter : public token_range_splitter {
            token_metadata_ptr _tmptr; // To keep the tablet map alive.
            const tablet_map& _tmap;
            std::optional<tablet_id> _next;
        public:
            splitter(token_metadata_ptr tmptr, const tablet_map& tmap)
                : _tmptr(std::move(tmptr))
                , _tmap(tmap)
            { }

            void reset(dht::ring_position_view pos) override {
                _next = _tmap.get_tablet_id(pos.token());
            }

            std::optional<dht::token> next_token() override {
                if (!_next) {
                    return std::nullopt;
                }
                auto t = _tmap.get_last_token(*_next);
                _next = _tmap.next_tablet(*_next);
                return t;
            }
        };
        return std::make_unique<splitter>(_tmptr, get_tablet_map());
    }

    const dht::sharder& get_sharder(const schema& s) const override {
        return _sharder;
    }
};

size_t tablet_aware_replication_strategy::parse_initial_tablets(const sstring& val) const {
    try {
        return std::stol(val);
    } catch (...) {
        throw exceptions::configuration_exception(format("\"initial_tablets\" must be numeric; found {}", val));
    }
}

void tablet_aware_replication_strategy::validate_tablet_options(const gms::feature_service& fs,
                                                                const replication_strategy_config_options& opts) const {
    for (auto& c: opts) {
        if (c.first == "initial_tablets") {
            if (!fs.tablets) {
                throw exceptions::configuration_exception("Tablet replication is not enabled");
            }
            parse_initial_tablets(c.second);
        }
    }
}

void tablet_aware_replication_strategy::process_tablet_options(abstract_replication_strategy& ars,
                                                               replication_strategy_config_options& opts) {
    auto i = opts.find("initial_tablets");
    if (i != opts.end()) {
        _initial_tablets = parse_initial_tablets(i->second);
        ars._uses_tablets = true;
        mark_as_per_table(ars);
        opts.erase(i);
    }
}

std::unordered_set<sstring> tablet_aware_replication_strategy::recognized_tablet_options() const {
    std::unordered_set<sstring> opts;
    opts.insert("initial_tablets");
    return opts;
}

effective_replication_map_ptr tablet_aware_replication_strategy::do_make_replication_map(
        table_id table, replication_strategy_ptr rs, token_metadata_ptr tm, size_t replication_factor) const {
    return seastar::make_shared<tablet_effective_replication_map>(table, std::move(rs), std::move(tm), replication_factor);
}

void tablet_metadata_guard::check() noexcept {
    auto erm = _table->get_effective_replication_map();
    auto& tmap = erm->get_token_metadata_ptr()->tablets().get_tablet_map(_tablet.table);
    auto* trinfo = tmap.get_tablet_transition_info(_tablet.tablet);
    if (bool(_stage) != bool(trinfo) || (_stage && _stage != trinfo->stage)) {
        _abort_source.request_abort();
    } else {
        _erm = std::move(erm);
        subscribe();
    }
}

tablet_metadata_guard::tablet_metadata_guard(replica::table& table, global_tablet_id tablet)
    : _table(table.shared_from_this())
    , _tablet(tablet)
    , _erm(table.get_effective_replication_map())
{
    subscribe();
    if (auto* trinfo = get_tablet_map().get_tablet_transition_info(tablet.tablet)) {
        _stage = trinfo->stage;
    }
}

void tablet_metadata_guard::subscribe() {
    _callback = _erm->get_validity_abort_source().subscribe([this] () noexcept {
        check();
    });
}

}

auto fmt::formatter<locator::global_tablet_id>::format(const locator::global_tablet_id& id, fmt::format_context& ctx) const
        -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}:{}", id.table, id.tablet);
}

auto fmt::formatter<locator::tablet_transition_stage>::format(const locator::tablet_transition_stage& stage, fmt::format_context& ctx) const
        -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", locator::tablet_transition_stage_to_string(stage));
}
