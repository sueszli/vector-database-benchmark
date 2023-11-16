/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <type_traits>
#include <seastar/core/coroutine.hh>
#include <seastar/coroutine/parallel_for_each.hh>
#include <seastar/util/file.hh>
#include <boost/range/adaptor/map.hpp>
#include <boost/algorithm/string.hpp>
#include "sstables/sstable_directory.hh"
#include "sstables/sstables.hh"
#include "sstables/sstables_manager.hh"
#include "compaction/compaction_manager.hh"
#include "log.hh"
#include "sstable_directory.hh"
#include "utils/lister.hh"
#include "utils/overloaded_functor.hh"
#include "utils/directories.hh"
#include "replica/database.hh"
#include "db/system_keyspace.hh"

static logging::logger dirlog("sstable_directory");

namespace sstables {

bool manifest_json_filter(const fs::path&, const directory_entry& entry) {
    // Filter out directories. If type of the entry is unknown - check its name.
    if (entry.type.value_or(directory_entry_type::regular) != directory_entry_type::directory && (entry.name == "manifest.json" || entry.name == "schema.cql")) {
        return false;
    }

    return true;
}

sstable_directory::filesystem_components_lister::filesystem_components_lister(std::filesystem::path dir)
        : _directory(dir)
        , _state(std::make_unique<scan_state>())
{
}

sstable_directory::system_keyspace_components_lister::system_keyspace_components_lister(db::system_keyspace& sys_ks, sstring location)
        : _sys_ks(sys_ks)
        , _location(std::move(location))
{
}

std::unique_ptr<sstable_directory::components_lister>
sstable_directory::make_components_lister() {
    return std::visit(overloaded_functor {
        [this] (const data_dictionary::storage_options::local& loc) mutable -> std::unique_ptr<sstable_directory::components_lister> {
            return std::make_unique<sstable_directory::filesystem_components_lister>(_sstable_dir);
        },
        [this] (const data_dictionary::storage_options::s3& os) mutable -> std::unique_ptr<sstable_directory::components_lister> {
            return std::make_unique<sstable_directory::system_keyspace_components_lister>(_manager.system_keyspace(), _sstable_dir.native());
        }
    }, _storage_opts->value);
}

sstable_directory::sstable_directory(replica::table& table,
        sstable_state state,
        io_error_handler_gen error_handler_gen)
    : sstable_directory(
        table.get_sstables_manager(),
        table.schema(),
        table.get_effective_replication_map()->get_sharder(*table.schema()),
        table.get_storage_options_ptr(),
        table.dir(),
        std::move(state),
        std::move(error_handler_gen)
    )
{}

sstable_directory::sstable_directory(sstables_manager& manager,
        schema_ptr schema,
        const dht::sharder& sharder,
        lw_shared_ptr<const data_dictionary::storage_options> storage_opts,
        sstring table_dir,
        sstable_state state,
        io_error_handler_gen error_handler_gen)
    : _manager(manager)
    , _schema(std::move(schema))
    , _storage_opts(std::move(storage_opts))
    , _table_dir(std::move(table_dir))
    , _state(state)
    , _sstable_dir(make_path(_table_dir, _state))
    , _error_handler_gen(error_handler_gen)
    , _storage(make_storage(_manager, *_storage_opts, _table_dir, _state))
    , _lister(make_components_lister())
    , _sharder(sharder)
    , _unshared_remote_sstables(smp::count)
{}

void sstable_directory::filesystem_components_lister::handle(sstables::entry_descriptor desc, fs::path filename) {
    // TODO: decorate sstable_directory with some noncopyable_function<shard_id (generation_type)>
    //       to communicate how different tables place sstables into shards.
    if (!sstables::sstable_generation_generator::maybe_owned_by_this_shard(desc.generation)) {
        return;
    }

    dirlog.trace("for SSTable directory, scanning {}", filename);
    _state->generations_found.emplace(desc.generation, filename);

    switch (desc.component) {
    case component_type::TemporaryStatistics:
        // We generate TemporaryStatistics when we rewrite the Statistics file,
        // for instance on mutate_level. We should delete it - so we mark it for deletion
        // here, but just the component. The old statistics file should still be there
        // and we'll go with it.
        _state->files_for_removal.insert(filename.native());
        break;
    case component_type::TOC:
        _state->descriptors.emplace(desc.generation, std::move(desc));
        break;
    case component_type::TemporaryTOC:
        _state->temp_toc_found.push_back(std::move(desc));
        break;
    default:
        // Do nothing, and will validate when trying to load the file.
        break;
    }
}

void sstable_directory::validate(sstables::shared_sstable sst, process_flags flags) const {
    schema_ptr s = sst->get_schema();
    if (s->is_counter() && !sst->has_scylla_component()) {
        sstring error = "Direct loading non-Scylla SSTables containing counters is not supported.";
        if (flags.enable_dangerous_direct_import_of_cassandra_counters) {
            dirlog.info("{} But trying to continue on user's request.", error);
        } else {
            dirlog.error("{} Use sstableloader instead.", error);
            throw std::runtime_error(fmt::format("{} Use sstableloader instead.", error));
        }
    }
    if (s->is_view() && !flags.allow_loading_materialized_view) {
        throw std::runtime_error("Loading Materialized View SSTables is not supported. Re-create the view instead.");
    }
    if (!sst->is_uploaded()) {
        sst->validate_originating_host_id();
    }
}

future<sstables::shared_sstable> sstable_directory::load_sstable(sstables::entry_descriptor desc, sstables::sstable_open_config cfg) const {
    auto sst = _manager.make_sstable(_schema, _table_dir, *_storage_opts, desc.generation, _state, desc.version, desc.format, gc_clock::now(), _error_handler_gen);
    co_await sst->load(_sharder, cfg);
    co_return sst;
}

future<sstables::shared_sstable> sstable_directory::load_sstable(sstables::entry_descriptor desc, process_flags flags) const {
    auto sst = co_await load_sstable(std::move(desc), flags.sstable_open_config);
    validate(sst, flags);
    if (flags.need_mutate_level) {
        dirlog.trace("Mutating {} to level 0\n", sst->get_filename());
        co_await sst->mutate_sstable_level(0);
    }
    co_return sst;
}

future<>
sstable_directory::process_descriptor(sstables::entry_descriptor desc, process_flags flags) {
    if (desc.version > _max_version_seen) {
        _max_version_seen = desc.version;
    }

    if (flags.sort_sstables_according_to_owner) {
        co_await sort_sstable(std::move(desc), flags);
    } else {
        dirlog.debug("Added {} to unsorted sstables list", sstable_filename(desc));
        _unsorted_sstables.push_back(co_await load_sstable(std::move(desc), flags));
    }
}

future<std::vector<shard_id>> sstable_directory::get_shards_for_this_sstable(const sstables::entry_descriptor& desc, process_flags flags) const {
    auto sst = _manager.make_sstable(_schema, _table_dir, *_storage_opts, desc.generation, _state, desc.version, desc.format, gc_clock::now(), _error_handler_gen);
    co_await sst->load_owner_shards(_sharder);
    validate(sst, flags);
    co_return sst->get_shards_for_this_sstable();
}

future<foreign_sstable_open_info> sstable_directory::get_open_info_for_this_sstable(const sstables::entry_descriptor& desc) const {
    auto sst = co_await load_sstable(std::move(desc));
    co_return co_await sst->get_open_info();
}

future<>
sstable_directory::sort_sstable(sstables::entry_descriptor desc, process_flags flags) {
    auto shards = co_await get_shards_for_this_sstable(desc, flags);
    if (shards.size() == 1) {
        if (shards[0] == this_shard_id()) {
            dirlog.trace("{} identified as a local unshared SSTable", sstable_filename(desc));
            _unshared_local_sstables.push_back(co_await load_sstable(std::move(desc), flags));
        } else {
            dirlog.trace("{} identified as a remote unshared SSTable, shard={}", sstable_filename(desc), shards[0]);
            _unshared_remote_sstables[shards[0]].push_back(std::move(desc));
        }
    } else {
        dirlog.trace("{} identified as a shared SSTable, shards={}", sstable_filename(desc), shards);
        _shared_sstable_info.push_back(co_await get_open_info_for_this_sstable(desc));
    }
}

sstring sstable_directory::sstable_filename(const sstables::entry_descriptor& desc) const {
    return sstable::filename(_sstable_dir.native(), _schema->ks_name(), _schema->cf_name(), desc.version, desc.generation, desc.format, component_type::Data);
}

generation_type
sstable_directory::highest_generation_seen() const {
    return _max_generation_seen;
}

sstables::sstable_version_types
sstable_directory::highest_version_seen() const {
    return _max_version_seen;
}

future<> sstable_directory::prepare(process_flags flags) {
    return _lister->prepare(*this, flags, *_storage);
}

future<> sstable_directory::filesystem_components_lister::prepare(sstable_directory& dir, process_flags flags, storage& st) {
    if (dir._state == sstable_state::quarantine) {
        if (!co_await file_exists(_directory.native())) {
            co_return;
        }
    }

    // verify owner and mode on the sstables directory
    // and all its subdirectories, except for "snapshots"
    // as there could be a race with scylla-manager that might
    // delete snapshots concurrently
    co_await utils::directories::verify_owner_and_mode(_directory, utils::directories::recursive::no);
    co_await lister::scan_dir(_directory, lister::dir_entry_types::of<directory_entry_type::directory>(), [] (fs::path dir, directory_entry de) -> future<> {
        if (de.name != sstables::snapshots_dir) {
            co_await utils::directories::verify_owner_and_mode(dir / de.name, utils::directories::recursive::yes);
        }
    });

    if (flags.garbage_collect) {
        co_await garbage_collect(st);
    }
}

future<> sstable_directory::system_keyspace_components_lister::prepare(sstable_directory& dir, process_flags flags, storage& st) {
    if (flags.garbage_collect) {
        co_await garbage_collect(st);
    }
}

future<> sstable_directory::process_sstable_dir(process_flags flags) {
    dirlog.debug("Start processing directory {} for SSTables (storage {})", _sstable_dir, _storage_opts->type_string());
    return _lister->process(*this, flags);
}

future<> sstable_directory::filesystem_components_lister::process(sstable_directory& directory, process_flags flags) {
    if (directory._state == sstable_state::quarantine) {
        if (!co_await file_exists(_directory.native())) {
            co_return;
        }
    }

    // It seems wasteful that each shard is repeating this scan, and to some extent it is.
    // However, we still want to open the files and especially call process_dir() in a distributed
    // fashion not to overload any shard. Also in the common case the SSTables will all be
    // unshared and be on the right shard based on their generation number. In light of that there are
    // two advantages of having each shard repeat the directory listing:
    //
    // - The directory listing part already interacts with data_structures inside scan_state. We
    //   would have to either transfer a lot of file information among shards or complicate the code
    //   to make sure they all update their own version of scan_state and then merge it.
    // - If all shards scan in parallel, they can start loading sooner. That is faster than having
    //   a separate step to fetch all files, followed by another step to distribute and process.

    directory_lister lister(_directory, lister::dir_entry_types::of<directory_entry_type::regular>(), &manifest_json_filter);
    std::exception_ptr ex;
    try {
        while (true) {
            auto de = co_await lister.get();
            if (!de) {
                break;
            }
            auto component_path = _directory / de->name;
            auto [ comps, ks, cf ] = sstables::parse_path(component_path);
            handle(std::move(comps), component_path);
        }
    } catch (...) {
        ex = std::current_exception();
    }
    co_await lister.close();
    if (ex) {
        dirlog.debug("Could not process sstable directory {}: {}", _directory, ex);
        // FIXME: waiting for https://github.com/scylladb/seastar/pull/1090
        // co_await coroutine::return_exception(std::move(ex));
        std::rethrow_exception(std::move(ex));
    }

    // Always okay to delete files with a temporary TOC. We want to do it before we process
    // the generations seen: it's okay to reuse those generations since the files will have
    // been deleted anyway.
    for (auto& desc: _state->temp_toc_found) {
        auto range = _state->generations_found.equal_range(desc.generation);
        for (auto it = range.first; it != range.second; ++it) {
            auto& path = it->second;
            dirlog.trace("Scheduling to remove file {}, from an SSTable with a Temporary TOC", path.native());
            _state->files_for_removal.insert(path.native());
        }
        _state->generations_found.erase(range.first, range.second);
        _state->descriptors.erase(desc.generation);
    }

    auto msg = format("After {} scanned, {} descriptors found, {} different files found",
            _directory, _state->descriptors.size(), _state->generations_found.size());

    if (!_state->generations_found.empty()) {
        directory._max_generation_seen =  boost::accumulate(_state->generations_found | boost::adaptors::map_keys, sstables::generation_type{}, [] (generation_type a, generation_type b) {
            return std::max<generation_type>(a, b);
        });

        msg = format("{}, highest generation seen: {}", msg, directory._max_generation_seen);
    } else {
        msg = format("{}, no numeric generation was seen", msg);
    }

    dirlog.debug("{}", msg);

    // _descriptors is everything with a TOC. So after we remove this, what's left is
    // SSTables for which a TOC was not found.
    auto descriptors = std::move(_state->descriptors);
    co_await directory.parallel_for_each_restricted(descriptors, [this, flags, &directory] (std::pair<const generation_type, sstables::entry_descriptor>& t) {
        auto& desc = std::get<1>(t);
        _state->generations_found.erase(desc.generation);
        // This will try to pre-load this file and throw an exception if it is invalid
        return directory.process_descriptor(std::move(desc), flags);
    });

    // For files missing TOC, it depends on where this is coming from.
    // If scylla was supposed to have generated this SSTable, this is not okay and
    // we refuse to proceed. If this coming from, say, an import, then we just delete,
    // log and proceed.
    for (auto& path : _state->generations_found | boost::adaptors::map_values) {
        if (flags.throw_on_missing_toc) {
            throw sstables::malformed_sstable_exception(format("At directory: {}: no TOC found for SSTable {}!. Refusing to boot", _directory.native(), path.native()));
        } else {
            dirlog.info("Found incomplete SSTable {} at directory {}. Removing", path.native(), _directory.native());
            _state->files_for_removal.insert(path.native());
        }
    }
}

future<> sstable_directory::system_keyspace_components_lister::process(sstable_directory& directory, process_flags flags) {
    return _sys_ks.sstables_registry_list(_location, [this, flags, &directory] (sstring status, sstable_state state, entry_descriptor desc) {
        if (state != directory._state) {
            return make_ready_future<>();
        }
        if (status != "sealed") {
            dirlog.warn("Skip processing {} {} entry from {} (must have been picked up by garbage collector)", status, desc.generation, _location);
            return make_ready_future<>();
        }
        if (!sstable_generation_generator::maybe_owned_by_this_shard(desc.generation)) {
            return make_ready_future<>();
        }

        dirlog.debug("Processing {} entry from {}", desc.generation, _location);
        return directory.process_descriptor(std::move(desc), flags);
    });
}

future<> sstable_directory::commit_directory_changes() {
    return _lister->commit().finally([x = std::move(_lister)] {});
}

future<> sstable_directory::filesystem_components_lister::commit() {
    // Remove all files scheduled for removal
    return parallel_for_each(std::exchange(_state->files_for_removal, {}), [] (sstring path) {
        dirlog.info("Removing file {}", path);
        return remove_file(std::move(path));
    });
}

future<> sstable_directory::system_keyspace_components_lister::commit() {
    return make_ready_future<>();
}

future<> sstable_directory::system_keyspace_components_lister::garbage_collect(storage& st) {
    return do_with(std::set<generation_type>(), [this, &st] (auto& gens_to_remove) {
        return _sys_ks.sstables_registry_list(_location, [&st, &gens_to_remove] (sstring status, sstable_state state, entry_descriptor desc) {
            if (status == "sealed") {
                return make_ready_future<>();
            }

            dirlog.info("Removing dangling {} {} entry", desc.generation, status);
            gens_to_remove.insert(desc.generation);
            return st.remove_by_registry_entry(std::move(desc));
        }).then([this, &gens_to_remove] {
            return parallel_for_each(gens_to_remove, [this] (auto gen) {
                return _sys_ks.sstables_registry_delete_entry(_location, gen);
            });
        });
    });
}

future<>
sstable_directory::move_foreign_sstables(sharded<sstable_directory>& source_directory) {
    return parallel_for_each(boost::irange(0u, smp::count), [this, &source_directory] (unsigned shard_id) mutable {
        auto info_vec = std::exchange(_unshared_remote_sstables[shard_id], {});
        if (info_vec.empty()) {
            return make_ready_future<>();
        }
        // Should be empty, since an SSTable that belongs to this shard is not remote.
        assert(shard_id != this_shard_id());
        dirlog.debug("Moving {} unshared SSTables to shard {} ", info_vec.size(), shard_id);
        return source_directory.invoke_on(shard_id, &sstables::sstable_directory::load_foreign_sstables, std::move(info_vec));
    });
}

future<shared_sstable> sstable_directory::load_foreign_sstable(foreign_sstable_open_info& info) {
    auto sst = _manager.make_sstable(_schema, _table_dir, *_storage_opts, info.generation, _state, info.version, info.format, gc_clock::now(), _error_handler_gen);
    co_await sst->load(std::move(info));
    co_return sst;
}

future<>
sstable_directory::load_foreign_sstables(sstable_entry_descriptor_vector info_vec) {
    co_await parallel_for_each_restricted(info_vec, [this] (const sstables::entry_descriptor& info) {
        return load_sstable(info).then([this] (auto sst) {
            _unshared_local_sstables.push_back(sst);
            return make_ready_future<>();
        });
    });
}

future<>
sstable_directory::remove_sstables(std::vector<sstables::shared_sstable> sstlist) {
    dirlog.debug("Removing {} SSTables", sstlist.size());
    return parallel_for_each(std::move(sstlist), [] (const sstables::shared_sstable& sst) {
        dirlog.trace("Removing SSTable {}", sst->get_filename());
        return sst->unlink().then([sst] {});
    });
}

future<>
sstable_directory::collect_output_unshared_sstables(std::vector<sstables::shared_sstable> resharded_sstables, can_be_remote remote_ok) {
    dirlog.debug("Collecting {} output SSTables (remote={})", resharded_sstables.size(), remote_ok);
    return parallel_for_each(std::move(resharded_sstables), [this, remote_ok] (sstables::shared_sstable sst) {
        auto shards = sst->get_shards_for_this_sstable();
        assert(shards.size() == 1);
        auto shard = shards[0];

        if (shard == this_shard_id()) {
            dirlog.trace("Collected output SSTable {} already local", sst->get_filename());
            _unshared_local_sstables.push_back(std::move(sst));
            return make_ready_future<>();
        }

        if (!remote_ok) {
            return make_exception_future<>(std::runtime_error("Unexpected remote sstable"));
        }

        dirlog.trace("Collected output SSTable {} is remote. Storing it", sst->get_filename());
        _unshared_remote_sstables[shard].push_back(sst->get_descriptor(component_type::Data));
        return make_ready_future<>();
    });
}

future<>
sstable_directory::remove_unshared_sstables(std::vector<sstables::shared_sstable> sstlist) {
    // When removing input sstables from reshaping: Those SSTables used to be in the unshared local
    // list. So not only do we have to remove them, we also have to update the list. Because we're
    // dealing with a vector it's easier to just reconstruct the list.
    dirlog.debug("Removing {} unshared SSTables", sstlist.size());
    return do_with(std::move(sstlist), std::unordered_set<sstables::shared_sstable>(),
            [this] (std::vector<sstables::shared_sstable>& sstlist, std::unordered_set<sstables::shared_sstable>& exclude) {

        for (auto& sst : sstlist) {
            exclude.insert(sst);
        }

        auto old = std::exchange(_unshared_local_sstables, {});

        for (auto& sst : old) {
            if (!exclude.contains(sst)) {
                _unshared_local_sstables.push_back(sst);
            }
        }

        // Do this last for exception safety. If there is an exception on unlink we
        // want to at least leave the SSTable unshared list in a sane state.
        return remove_sstables(std::move(sstlist)).then([] {
            dirlog.debug("Finished removing all SSTables");
        });
    });
}


future<>
sstable_directory::do_for_each_sstable(std::function<future<>(sstables::shared_sstable)> func) {
    auto sstables = std::move(_unshared_local_sstables);
    co_await parallel_for_each_restricted(sstables, std::move(func));
}

future<>
sstable_directory::filter_sstables(std::function<future<bool>(sstables::shared_sstable)> func) {
    std::vector<sstables::shared_sstable> filtered;
    co_await parallel_for_each_restricted(_unshared_local_sstables, [func = std::move(func), &filtered] (sstables::shared_sstable sst) -> future<> {
        auto keep = co_await func(sst);
        if (keep) {
            filtered.emplace_back(sst);
        }
    });
    _unshared_local_sstables = std::move(filtered);
}

template <std::ranges::range Container, typename Func>
requires std::is_invocable_r_v<future<>, Func, typename std::ranges::range_value_t<Container>&>
future<>
sstable_directory::parallel_for_each_restricted(Container& c, Func func) {
    co_await max_concurrent_for_each(c, _manager.dir_semaphore()._concurrency, [&] (auto& el) -> future<>{
        auto units = co_await get_units(_manager.dir_semaphore()._sem, 1);
        co_await func(el);
    });
}

void
sstable_directory::store_phaser(utils::phased_barrier::operation op) {
    _operation_barrier.emplace(std::move(op));
}

sstable_directory::sstable_open_info_vector
sstable_directory::retrieve_shared_sstables() {
    return std::exchange(_shared_sstable_info, {});
}

bool sstable_directory::compare_sstable_storage_prefix(const sstring& prefix_a, const sstring& prefix_b) noexcept {
    size_t size_a = prefix_a.size();
    if (prefix_a.back() == '/') {
        size_a--;
    }
    size_t size_b = prefix_b.size();
    if (prefix_b.back() == '/') {
        size_b--;
    }
    return size_a == size_b && sstring::traits_type::compare(prefix_a.begin(), prefix_b.begin(), size_a) == 0;
}

future<> sstable_directory::delete_with_pending_deletion_log(std::vector<shared_sstable> ssts) {
    return seastar::async([ssts = std::move(ssts)] {
        shared_sstable first = nullptr;
        min_max_tracker<generation_type> gen_tracker;

        for (const auto& sst : ssts) {
            gen_tracker.update(sst->generation());

            if (first == nullptr) {
                first = sst;
            } else {
                // All sstables are assumed to be in the same column_family, hence
                // sharing their base directory. Since lexicographical comparison of
                // paths is not the same as their actualy equivalence, this should
                // rather check for fs::equivalent call on _storage.prefix()-s. But
                // since we know that the worst thing filesystem storage driver can
                // do is to prepend/drop the trailing slash, it should be enough to
                // compare prefixes of both ... prefixes
                assert(compare_sstable_storage_prefix(first->_storage->prefix(), sst->_storage->prefix()));
            }
        }

        sstring pending_delete_dir = first->_storage->prefix() + "/" + sstables::pending_delete_dir;
        sstring pending_delete_log = format("{}/sstables-{}-{}.log", pending_delete_dir, gen_tracker.min(), gen_tracker.max());
        sstring tmp_pending_delete_log = pending_delete_log + ".tmp";
        sstlog.trace("Writing {}", tmp_pending_delete_log);
        try {
            touch_directory(pending_delete_dir).get();
            auto oflags = open_flags::wo | open_flags::create | open_flags::exclusive;
            // Create temporary pending_delete log file.
            auto f = open_file_dma(tmp_pending_delete_log, oflags).get0();
            // Write all toc names into the log file.
            auto out = make_file_output_stream(std::move(f), 4096).get0();
            auto close_out = deferred_close(out);

            for (const auto& sst : ssts) {
                auto toc = sst->component_basename(component_type::TOC);
                out.write(toc).get();
                out.write("\n").get();
            }

            out.flush().get();
            close_out.close_now();

            auto dir_f = open_directory(pending_delete_dir).get0();
            // Once flushed and closed, the temporary log file can be renamed.
            rename_file(tmp_pending_delete_log, pending_delete_log).get();

            // Guarantee that the changes above reached the disk.
            dir_f.flush().get();
            dir_f.close().get();
            sstlog.debug("{} written successfully.", pending_delete_log);
        } catch (...) {
            sstlog.warn("Error while writing {}: {}. Ignoring.", pending_delete_log, std::current_exception());
        }

        parallel_for_each(ssts, [] (shared_sstable sst) {
            return sst->unlink(sstables::storage::sync_dir::no);
        }).get();

        sync_directory(first->_storage->prefix()).get();

        // Once all sstables are deleted, the log file can be removed.
        // Note: the log file will be removed also if unlink failed to remove
        // any sstable and ignored the error.
        try {
            remove_file(pending_delete_log).get();
            sstlog.debug("{} removed.", pending_delete_log);
        } catch (...) {
            sstlog.warn("Error removing {}: {}. Ignoring.", pending_delete_log, std::current_exception());
        }
    });
}

// FIXME: Go through maybe_delete_large_partitions_entry on recovery since
// this is an indication we crashed in the middle of delete_with_pending_deletion_log
future<> sstable_directory::filesystem_components_lister::replay_pending_delete_log(fs::path pending_delete_log) {
    sstlog.debug("Reading pending_deletes log file {}", pending_delete_log);
    fs::path pending_delete_dir = pending_delete_log.parent_path();
    try {
        sstring sstdir = pending_delete_dir.parent_path().native();
        auto text = co_await seastar::util::read_entire_file_contiguous(pending_delete_log);

        sstring all(text.begin(), text.end());
        std::vector<sstring> basenames;
        boost::split(basenames, all, boost::is_any_of("\n"), boost::token_compress_on);
        auto tocs = boost::copy_range<std::vector<sstring>>(basenames | boost::adaptors::filtered([] (auto&& basename) { return !basename.empty(); }));
        co_await parallel_for_each(tocs, [&sstdir] (const sstring& name) {
            return remove_by_toc_name(sstdir + "/" + name);
        });
        sstlog.debug("Replayed {}, removing", pending_delete_log);
        co_await remove_file(pending_delete_log.native());
    } catch (...) {
        sstlog.warn("Error replaying {}: {}. Ignoring.", pending_delete_log, std::current_exception());
    }
}

future<> sstable_directory::filesystem_components_lister::garbage_collect(storage& st) {
    // First pass, cleanup temporary sstable directories and sstables pending delete.
    co_await cleanup_column_family_temp_sst_dirs();
    co_await handle_sstables_pending_delete();
}

future<> sstable_directory::filesystem_components_lister::cleanup_column_family_temp_sst_dirs() {
    std::vector<future<>> futures;

    co_await lister::scan_dir(_directory, lister::dir_entry_types::of<directory_entry_type::directory>(), [&] (fs::path sstdir, directory_entry de) {
        // push futures that remove files/directories into an array of futures,
        // so that the supplied callback will not block scan_dir() from
        // reading the next entry in the directory.
        fs::path dirpath = sstdir / de.name;
        if (dirpath.extension().string() == tempdir_extension) {
            sstlog.info("Found temporary sstable directory: {}, removing", dirpath);
            futures.push_back(io_check([dirpath = std::move(dirpath)] () { return lister::rmdir(dirpath); }));
        }
        return make_ready_future<>();
    });

    co_await when_all_succeed(futures.begin(), futures.end()).discard_result();
}

future<> sstable_directory::filesystem_components_lister::handle_sstables_pending_delete() {
    auto pending_delete_dir = _directory / sstables::pending_delete_dir;
    auto exists = co_await file_exists(pending_delete_dir.native());
    if (!exists) {
        co_return;
    }

    std::vector<future<>> futures;

    co_await lister::scan_dir(pending_delete_dir, lister::dir_entry_types::of<directory_entry_type::regular>(), [this, &futures] (fs::path dir, directory_entry de) {
        // push nested futures that remove files/directories into an array of futures,
        // so that the supplied callback will not block scan_dir() from
        // reading the next entry in the directory.
        fs::path file_path = dir / de.name;
        if (file_path.extension() == ".tmp") {
            sstlog.info("Found temporary pending_delete log file: {}, deleting", file_path);
            futures.push_back(remove_file(file_path.string()));
        } else if (file_path.extension() == ".log") {
            sstlog.info("Found pending_delete log file: {}, replaying", file_path);
            auto f = replay_pending_delete_log(std::move(file_path));
            futures.push_back(std::move(f));
        } else {
            sstlog.debug("Found unknown file in pending_delete directory: {}, ignoring", file_path);
        }
        return make_ready_future<>();
    });

    co_await when_all_succeed(futures.begin(), futures.end()).discard_result();
}

future<sstables::generation_type>
highest_generation_seen(sharded<sstables::sstable_directory>& directory) {
    co_return co_await directory.map_reduce0(
        std::mem_fn(&sstables::sstable_directory::highest_generation_seen),
        sstables::generation_type{},
        [] (sstables::generation_type a, sstables::generation_type b) {
            return std::max(a, b);
        });
}

}
