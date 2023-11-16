///////////////////////////////////////////////////////////////////////////
//
// Copyright 2022 Realm Inc.
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

#include "realm/util/functional.hpp"
#include <realm/sync/noinst/client_history_impl.hpp>

#include <realm/util/compression.hpp>
#include <realm/util/features.h>
#include <realm/util/scope_exit.hpp>
#include <realm/sync/changeset.hpp>
#include <realm/sync/changeset_parser.hpp>
#include <realm/sync/instruction_applier.hpp>
#include <realm/sync/instruction_replication.hpp>
#include <realm/sync/noinst/client_reset.hpp>
#include <realm/transaction.hpp>
#include <realm/version.hpp>

#include <algorithm>
#include <ctime>
#include <cstring>
#include <utility>

namespace realm::sync {

void ClientHistory::set_client_file_ident_in_wt(version_type current_version, SaltedFileIdent client_file_ident)
{
    ensure_updated(current_version); // Throws
    prepare_for_write();             // Throws

    Array& root = m_arrays->root;
    m_group->set_sync_file_id(client_file_ident.ident); // Throws
    root.set(s_client_file_ident_salt_iip,
             RefOrTagged::make_tagged(client_file_ident.salt)); // Throws
}


void ClientHistory::set_client_reset_adjustments(version_type current_version, SaltedFileIdent client_file_ident,
                                                 SaltedVersion server_version, BinaryData uploadable_changeset)
{
    ensure_updated(current_version); // Throws
    prepare_for_write();             // Throws

    version_type client_version = m_sync_history_base_version + sync_history_size();
    REALM_ASSERT(client_version == current_version); // For now
    DownloadCursor download_progress = {server_version.version, 0};
    UploadCursor upload_progress = {0, 0};
    Array& root = m_arrays->root;
    m_group->set_sync_file_id(client_file_ident.ident); // Throws
    root.set(s_client_file_ident_salt_iip,
             RefOrTagged::make_tagged(client_file_ident.salt)); // Throws
    root.set(s_progress_download_server_version_iip,
             RefOrTagged::make_tagged(download_progress.server_version)); // Throws
    root.set(s_progress_download_client_version_iip,
             RefOrTagged::make_tagged(download_progress.last_integrated_client_version)); // Throws
    root.set(s_progress_latest_server_version_iip,
             RefOrTagged::make_tagged(server_version.version)); // Throws
    root.set(s_progress_latest_server_version_salt_iip,
             RefOrTagged::make_tagged(server_version.salt)); // Throws
    root.set(s_progress_upload_client_version_iip,
             RefOrTagged::make_tagged(upload_progress.client_version)); // Throws
    root.set(s_progress_upload_server_version_iip,
             RefOrTagged::make_tagged(upload_progress.last_integrated_server_version)); // Throws
    root.set(s_progress_downloaded_bytes_iip,
             RefOrTagged::make_tagged(0)); // Throws
    root.set(s_progress_downloadable_bytes_iip,
             RefOrTagged::make_tagged(0)); // Throws
    root.set(s_progress_uploaded_bytes_iip,
             RefOrTagged::make_tagged(0)); // Throws
    root.set(s_progress_uploadable_bytes_iip,
             RefOrTagged::make_tagged(0)); // Throws

    // Discard existing synchronization history
    do_trim_sync_history(sync_history_size()); // Throws

    m_progress_download = download_progress;
    m_client_reset_changeset = uploadable_changeset; // Picked up by prepare_changeset()
}

std::vector<ClientHistory::LocalChange> ClientHistory::get_local_changes(version_type current_version) const
{
    ensure_updated(current_version); // Throws
    std::vector<ClientHistory::LocalChange> changesets;
    if (!m_arrays || m_arrays->changesets.is_empty())
        return changesets;

    sync::version_type begin_version = 0;
    {
        sync::version_type local_version;
        SaltedFileIdent local_ident;
        SyncProgress local_progress;
        get_status(local_version, local_ident, local_progress);
        begin_version = local_progress.upload.client_version;
    }

    version_type end_version = m_sync_history_base_version + sync_history_size();
    if (begin_version < m_sync_history_base_version)
        begin_version = m_sync_history_base_version;

    for (version_type version = begin_version; version < end_version; ++version) {
        std::size_t ndx = std::size_t(version - m_sync_history_base_version);
        std::int_fast64_t origin_file_ident = m_arrays->origin_file_idents.get(ndx);
        bool not_from_server = (origin_file_ident == 0);
        if (not_from_server) {
            changesets.push_back({version, m_arrays->changesets.get(ndx)});
        }
    }
    return changesets;
}

void ClientHistory::set_local_origin_timestamp_source(util::UniqueFunction<timestamp_type()> source_fn)
{
    m_local_origin_timestamp_source = std::move(source_fn);
}

// Overriding member function in realm::Replication
void ClientReplication::initialize(DB& sg)
{
    SyncReplication::initialize(sg); // Throws
    m_history.initialize(sg);
}


// Overriding member function in realm::Replication
auto ClientReplication::get_history_type() const noexcept -> HistoryType
{
    return hist_SyncClient;
}


// Overriding member function in realm::Replication
int ClientReplication::get_history_schema_version() const noexcept
{
    return get_client_history_schema_version();
}


// Overriding member function in realm::Replication
bool ClientReplication::is_upgradable_history_schema(int stored_schema_version) const noexcept
{
    if (stored_schema_version == 11) {
        return true;
    }
    return false;
}


// Overriding member function in realm::Replication
void ClientReplication::upgrade_history_schema(int stored_schema_version)
{
    // upgrade_history_schema() is called only when there is a need to upgrade
    // (`stored_schema_version < get_server_history_schema_version()`), and only
    // when is_upgradable_history_schema() returned true (`stored_schema_version
    // >= 1`).
    REALM_ASSERT(stored_schema_version < get_client_history_schema_version());
    REALM_ASSERT(stored_schema_version >= 11);
    int orig_schema_version = stored_schema_version;
    int schema_version = orig_schema_version;

    if (schema_version < 12) {
        m_history.compress_stored_changesets();
        schema_version = 12;
    }

    // NOTE: Future migration steps go here.

    REALM_ASSERT(schema_version == get_client_history_schema_version());

    // Record migration event
    m_history.record_current_schema_version(); // Throws
}

void ClientHistory::compress_stored_changesets()
{
    using gf = _impl::GroupFriend;
    Allocator& alloc = gf::get_alloc(*m_group);
    auto ref = gf::get_history_ref(*m_group);
    Arrays arrays{alloc, *m_group, ref};

    util::AppendBuffer<char> compressed_buffer;
    util::AppendBuffer<char> decompressed_buffer;
    util::compression::CompressMemoryArena arena;
    auto columns = {&arrays.reciprocal_transforms, &arrays.changesets};
    for (auto column : columns) {
        for (size_t i = 0; i < column->size(); ++i) {
            ChunkedBinaryData data(*column, i);
            if (data.is_null())
                continue;
            data.copy_to(compressed_buffer);
            util::compression::allocate_and_compress_nonportable(arena, compressed_buffer, decompressed_buffer);
            column->set(i, BinaryData{decompressed_buffer.data(), decompressed_buffer.size()}); // Throws
        }
    }
}

// Overriding member function in realm::Replication
auto ClientReplication::prepare_changeset(const char* data, size_t size, version_type orig_version) -> version_type
{
    m_history.ensure_updated(orig_version);
    m_history.prepare_for_write(); // Throws

    BinaryData ct_changeset{data, size};
    auto& buffer = get_instruction_encoder().buffer();
    BinaryData sync_changeset(buffer.data(), buffer.size());

    return m_history.add_changeset(ct_changeset, sync_changeset); // Throws
}

util::UniqueFunction<SyncReplication::WriteValidator> ClientReplication::make_write_validator(Transaction& tr)
{
    if (!m_write_validator_factory) {
        return {};
    }

    return m_write_validator_factory(tr);
}

void ClientHistory::get_status(version_type& current_client_version, SaltedFileIdent& client_file_ident,
                               SyncProgress& progress, bool* has_pending_client_reset) const
{
    TransactionRef rt = m_db->start_read(); // Throws
    version_type current_client_version_2 = rt->get_version();

    SaltedFileIdent client_file_ident_2{rt->get_sync_file_id(), 0};
    SyncProgress progress_2;
    using gf = _impl::GroupFriend;
    if (ref_type ref = gf::get_history_ref(*rt)) {
        Array root(m_db->get_alloc());
        root.init_from_ref(ref);
        client_file_ident_2.salt = salt_type(root.get_as_ref_or_tagged(s_client_file_ident_salt_iip).get_as_int());
        progress_2.latest_server_version.version =
            version_type(root.get_as_ref_or_tagged(s_progress_latest_server_version_iip).get_as_int());
        progress_2.latest_server_version.salt =
            version_type(root.get_as_ref_or_tagged(s_progress_latest_server_version_salt_iip).get_as_int());
        progress_2.download.server_version =
            version_type(root.get_as_ref_or_tagged(s_progress_download_server_version_iip).get_as_int());
        progress_2.download.last_integrated_client_version =
            version_type(root.get_as_ref_or_tagged(s_progress_download_client_version_iip).get_as_int());
        progress_2.upload.client_version =
            version_type(root.get_as_ref_or_tagged(s_progress_upload_client_version_iip).get_as_int());
        progress_2.upload.last_integrated_server_version =
            version_type(root.get_as_ref_or_tagged(s_progress_upload_server_version_iip).get_as_int());
    }

    current_client_version = current_client_version_2;
    client_file_ident = client_file_ident_2;
    progress = progress_2;

    REALM_ASSERT(current_client_version >= s_initial_version + 0);
    if (current_client_version == s_initial_version + 0)
        current_client_version = 0;

    if (has_pending_client_reset) {
        *has_pending_client_reset = _impl::client_reset::has_pending_reset(*rt).has_value();
    }
}

SaltedFileIdent ClientHistory::get_client_file_ident(const Transaction& rt) const
{
    SaltedFileIdent client_file_ident{rt.get_sync_file_id(), 0};

    using gf = _impl::GroupFriend;
    if (ref_type ref = gf::get_history_ref(rt)) {
        Array root(m_db->get_alloc());
        root.init_from_ref(ref);
        client_file_ident.salt = salt_type(root.get_as_ref_or_tagged(s_client_file_ident_salt_iip).get_as_int());
    }

    return client_file_ident;
}

void ClientHistory::set_client_file_ident(SaltedFileIdent client_file_ident, bool fix_up_object_ids)
{
    REALM_ASSERT(client_file_ident.ident != 0);

    TransactionRef wt = m_db->start_write(); // Throws
    version_type local_version = wt->get_version();
    ensure_updated(local_version); // Throws
    prepare_for_write();           // Throws

    Array& root = m_arrays->root;
    REALM_ASSERT(wt->get_sync_file_id() == 0);
    wt->set_sync_file_id(client_file_ident.ident);
    root.set(s_client_file_ident_salt_iip,
             RefOrTagged::make_tagged(client_file_ident.salt)); // Throws
    root.set(s_progress_download_client_version_iip, RefOrTagged::make_tagged(0));
    root.set(s_progress_upload_client_version_iip, RefOrTagged::make_tagged(0));

    if (fix_up_object_ids) {
        fix_up_client_file_ident_in_stored_changesets(*wt, client_file_ident.ident); // Throws
    }

    // Note: This transaction produces an empty changeset. Empty changesets are
    // not uploaded to the server.
    wt->commit(); // Throws
}


// Overriding member function in realm::sync::ClientHistoryBase
void ClientHistory::set_sync_progress(const SyncProgress& progress, const std::uint_fast64_t* downloadable_bytes,
                                      VersionInfo& version_info)
{
    TransactionRef wt = m_db->start_write(); // Throws
    version_type local_version = wt->get_version();
    ensure_updated(local_version); // Throws
    prepare_for_write();           // Throws

    update_sync_progress(progress, downloadable_bytes, wt); // Throws

    // Note: This transaction produces an empty changeset. Empty changesets are
    // not uploaded to the server.
    version_type new_version = wt->commit(); // Throws
    version_info.realm_version = new_version;
    version_info.sync_version = {new_version, 0};
}

void ClientHistory::find_uploadable_changesets(UploadCursor& upload_progress, version_type end_version,
                                               std::vector<UploadChangeset>& uploadable_changesets,
                                               version_type& locked_server_version) const
{
    TransactionRef rt = m_db->start_read(); // Throws
    auto& alloc = m_db->get_alloc();
    using gf = _impl::GroupFriend;
    ref_type ref = gf::get_history_ref(*rt);
    REALM_ASSERT(ref);

    Arrays arrays(alloc, *rt, ref);
    const auto sync_history_size = arrays.changesets.size();
    const auto sync_history_base_version = rt->get_version() - sync_history_size;

    std::size_t accum_byte_size_soft_limit = 131072;   // 128 KB
    std::size_t accum_byte_size_hard_limit = 16777216; // server-imposed limit
    std::size_t accum_byte_size = 0;

    version_type begin_version_2 = std::max(upload_progress.client_version, sync_history_base_version);
    version_type end_version_2 = std::max(end_version, sync_history_base_version);
    version_type last_integrated_upstream_version = upload_progress.last_integrated_server_version;

    while (accum_byte_size < accum_byte_size_soft_limit) {
        HistoryEntry entry;
        version_type last_integrated_upstream_version_2 = last_integrated_upstream_version;
        version_type version = find_sync_history_entry(arrays, sync_history_base_version, begin_version_2,
                                                       end_version_2, entry, last_integrated_upstream_version_2);

        if (version == 0) {
            begin_version_2 = end_version_2;
            last_integrated_upstream_version = last_integrated_upstream_version_2;
            break;
        }

        ChunkedBinaryInputStream is(entry.changeset);
        size_t size = util::compression::get_uncompressed_size_from_header(is);
        if (accum_byte_size + size >= accum_byte_size_hard_limit && !uploadable_changesets.empty())
            break;
        accum_byte_size += size;
        last_integrated_upstream_version = last_integrated_upstream_version_2;
        begin_version_2 = version;

        UploadChangeset uc;
        util::AppendBuffer<char> decompressed;
        ChunkedBinaryInputStream is_2(entry.changeset);
        auto ec = util::compression::decompress_nonportable(is_2, decompressed);
        if (ec == util::compression::error::decompress_unsupported) {
            REALM_TERMINATE(
                "Synchronized Realm files with unuploaded local changes cannot be copied between platforms.");
        }
        REALM_ASSERT_3(ec, ==, std::error_code{});

        uc.origin_timestamp = entry.origin_timestamp;
        uc.origin_file_ident = entry.origin_file_ident;
        uc.progress = UploadCursor{version, entry.remote_version};
        uc.changeset = BinaryData{decompressed.data(), decompressed.size()};
        uc.buffer = decompressed.release().release();
        uploadable_changesets.push_back(std::move(uc)); // Throws
    }

    upload_progress = {std::min(begin_version_2, end_version), last_integrated_upstream_version};

    locked_server_version = arrays.root.get_as_ref_or_tagged(s_progress_download_server_version_iip).get_as_int();
}


void ClientHistory::integrate_server_changesets(
    const SyncProgress& progress, const std::uint_fast64_t* downloadable_bytes,
    util::Span<const RemoteChangeset> incoming_changesets, VersionInfo& version_info, DownloadBatchState batch_state,
    util::Logger& logger, const TransactionRef& transact,
    util::UniqueFunction<void(const TransactionRef&, util::Span<Changeset>)> run_in_write_tr)
{
    REALM_ASSERT(incoming_changesets.size() != 0);
    REALM_ASSERT(
        (transact->get_transact_stage() == DB::transact_Writing && batch_state != DownloadBatchState::SteadyState) ||
        (transact->get_transact_stage() == DB::transact_Reading && batch_state == DownloadBatchState::SteadyState));
    std::vector<Changeset> changesets;
    changesets.resize(incoming_changesets.size()); // Throws

    // Parse incoming changesets without holding the write lock unless 'transact' is specified.
    try {
        for (std::size_t i = 0; i < incoming_changesets.size(); ++i) {
            const RemoteChangeset& changeset = incoming_changesets[i];
            parse_remote_changeset(changeset, changesets[i]); // Throws
            changesets[i].transform_sequence = i;
        }
    }
    catch (const BadChangesetError& e) {
        throw IntegrationException(ErrorCodes::BadChangeset,
                                   util::format("Failed to parse received changeset: %1", e.what()),
                                   ProtocolError::bad_changeset);
    }

    VersionID new_version{0, 0};
    auto num_changesets = incoming_changesets.size();
    util::Span<Changeset> changesets_to_integrate(changesets);
    const bool allow_lock_release = batch_state == DownloadBatchState::SteadyState;

    // Ideally, this loop runs only once, but it can run up to `incoming_changesets.size()` times, depending on the
    // number of times the sync client yields the write lock to allow the user to commit their changes.
    // In each iteration, at least one changeset is transformed and committed.
    // In FLX, all changesets are committed at once in the bootstrap phase (i.e, in one iteration).
    while (!changesets_to_integrate.empty()) {
        if (transact->get_transact_stage() == DB::transact_Reading) {
            transact->promote_to_write(); // Throws
        }
        VersionID old_version = transact->get_version_of_current_transaction();
        version_type local_version = old_version.version;
        auto sync_file_id = transact->get_sync_file_id();
        REALM_ASSERT(sync_file_id != 0);

        ensure_updated(local_version); // Throws
        prepare_for_write();           // Throws

        std::uint64_t downloaded_bytes_in_transaction = 0;
        auto changesets_transformed_count = transform_and_apply_server_changesets(
            changesets_to_integrate, transact, logger, downloaded_bytes_in_transaction, allow_lock_release);

        // downloaded_bytes always contains the total number of downloaded bytes
        // from the Realm. downloaded_bytes must be persisted in the Realm, since
        // the downloaded changesets are trimmed after use, and since it would be
        // expensive to traverse the entire history.
        Array& root = m_arrays->root;
        auto downloaded_bytes =
            std::uint64_t(root.get_as_ref_or_tagged(s_progress_downloaded_bytes_iip).get_as_int());
        downloaded_bytes += downloaded_bytes_in_transaction;
        root.set(s_progress_downloaded_bytes_iip, RefOrTagged::make_tagged(downloaded_bytes)); // Throws

        const RemoteChangeset& last_changeset = incoming_changesets[changesets_transformed_count - 1];
        auto changesets_for_cb = changesets_to_integrate.first(changesets_transformed_count);
        changesets_to_integrate = changesets_to_integrate.sub_span(changesets_transformed_count);
        incoming_changesets = incoming_changesets.sub_span(changesets_transformed_count);

        // During the bootstrap phase in flexible sync, the server sends multiple download messages with the same
        // synthetic server version that represents synthetic changesets generated from state on the server.
        if (batch_state == DownloadBatchState::LastInBatch && changesets_to_integrate.empty()) {
            update_sync_progress(progress, downloadable_bytes, transact); // Throws
        }
        // Always update progress for download messages from steady state.
        else if (batch_state == DownloadBatchState::SteadyState && !changesets_to_integrate.empty()) {
            auto partial_progress = progress;
            partial_progress.download.server_version = last_changeset.remote_version;
            partial_progress.download.last_integrated_client_version = last_changeset.last_integrated_local_version;
            update_sync_progress(partial_progress, downloadable_bytes, transact); // Throws
        }
        else if (batch_state == DownloadBatchState::SteadyState && changesets_to_integrate.empty()) {
            update_sync_progress(progress, downloadable_bytes, transact); // Throws
        }
        if (run_in_write_tr) {
            run_in_write_tr(transact, changesets_for_cb);
        }

        // The reason we can use the `origin_timestamp`, and the `origin_file_ident`
        // from the last transformed changeset, and ignore all the other changesets, is
        // that these values are actually irrelevant for changesets of remote origin
        // stored in the client-side history (for now), except that
        // `origin_file_ident` is required to be nonzero, to mark it as having been
        // received from the server.
        HistoryEntry entry;
        entry.origin_timestamp = last_changeset.origin_timestamp;
        entry.origin_file_ident = last_changeset.origin_file_ident;
        entry.remote_version = last_changeset.remote_version;
        add_sync_history_entry(entry); // Throws

        // Tell prepare_commit()/add_changeset() not to write a history entry for
        // this transaction as we already did it.
        REALM_ASSERT(!m_applying_server_changeset);
        m_applying_server_changeset = true;
        // Commit and continue to write if in bootstrap phase and there are still changes to integrate.
        if (batch_state == DownloadBatchState::MoreToCome ||
            (batch_state == DownloadBatchState::LastInBatch && !changesets_to_integrate.empty())) {
            new_version = transact->commit_and_continue_writing(); // Throws
        }
        else {
            new_version = transact->commit_and_continue_as_read(); // Throws
        }

        logger.debug("Integrated %1 changesets out of %2", changesets_transformed_count, num_changesets);
    }

    REALM_ASSERT(new_version.version > 0);
    REALM_ASSERT(
        (batch_state == DownloadBatchState::MoreToCome && transact->get_transact_stage() == DB::transact_Writing) ||
        (batch_state != DownloadBatchState::MoreToCome && transact->get_transact_stage() == DB::transact_Reading));
    version_info.realm_version = new_version.version;
    version_info.sync_version = {new_version.version, 0};
}


size_t ClientHistory::transform_and_apply_server_changesets(util::Span<Changeset> changesets_to_integrate,
                                                            TransactionRef transact, util::Logger& logger,
                                                            std::uint64_t& downloaded_bytes, bool allow_lock_release)
{
    REALM_ASSERT(transact->get_transact_stage() == DB::transact_Writing);

    if (!m_replication.apply_server_changes()) {
        std::for_each(changesets_to_integrate.begin(), changesets_to_integrate.end(), [&](const Changeset c) {
            downloaded_bytes += c.original_changeset_size;
        });
        // Skip over all changesets if they don't need to be transformed and applied.
        return changesets_to_integrate.size();
    }

    version_type local_version = transact->get_version_of_current_transaction().version;
    auto sync_file_id = transact->get_sync_file_id();

    try {
        for (auto& changeset : changesets_to_integrate) {
            REALM_ASSERT(changeset.last_integrated_remote_version <= local_version);
            REALM_ASSERT(changeset.origin_file_ident > 0 && changeset.origin_file_ident != sync_file_id);

            // It is possible that the synchronization history has been trimmed
            // to a point where a prefix of the merge window is no longer
            // available, but this can only happen if that prefix consisted
            // entirely of upload skippable entries. Since such entries (those
            // that are empty or of remote origin) will be skipped by the
            // transformer anyway, we can simply clamp the beginning of the
            // merge window to the beginning of the synchronization history,
            // when this situation occurs.
            //
            // See trim_sync_history() for further details.
            if (changeset.last_integrated_remote_version < m_sync_history_base_version)
                changeset.last_integrated_remote_version = m_sync_history_base_version;
        }

        constexpr std::size_t commit_byte_size_limit = 102400; // 100 KB

        auto changeset_applier = [&](const Changeset* transformed_changeset) -> bool {
            InstructionApplier applier{*transact};
            {
                TempShortCircuitReplication tscr{m_replication};
                applier.apply(*transformed_changeset, &logger); // Throws
            }
            downloaded_bytes += transformed_changeset->original_changeset_size;

            return !(allow_lock_release && m_db->other_writers_waiting_for_lock() &&
                     transact->get_commit_size() >= commit_byte_size_limit);
        };
        sync::Transformer transformer;
        auto changesets_transformed_count = transformer.transform_remote_changesets(
            *this, sync_file_id, local_version, changesets_to_integrate, changeset_applier, logger); // Throws
        return changesets_transformed_count;
    }
    catch (const BadChangesetError& e) {
        throw IntegrationException(ErrorCodes::BadChangeset,
                                   util::format("Failed to apply received changeset: %1", e.what()),
                                   ProtocolError::bad_changeset);
    }
    catch (const TransformError& e) {
        throw IntegrationException(ErrorCodes::BadChangeset,
                                   util::format("Failed to transform received changeset: %1", e.what()),
                                   ProtocolError::bad_changeset);
    }
}


void ClientHistory::get_upload_download_bytes(DB* db, std::uint_fast64_t& downloaded_bytes,
                                              std::uint_fast64_t& downloadable_bytes,
                                              std::uint_fast64_t& uploaded_bytes,
                                              std::uint_fast64_t& uploadable_bytes,
                                              std::uint_fast64_t& snapshot_version)
{
    TransactionRef rt = db->start_read(); // Throws
    version_type current_client_version = rt->get_version();

    downloaded_bytes = 0;
    downloadable_bytes = 0;
    uploaded_bytes = 0;
    uploadable_bytes = 0;
    snapshot_version = current_client_version;

    using gf = _impl::GroupFriend;
    if (ref_type ref = gf::get_history_ref(*rt)) {
        Array root(db->get_alloc());
        root.init_from_ref(ref);
        downloaded_bytes = root.get_as_ref_or_tagged(s_progress_downloaded_bytes_iip).get_as_int();
        downloadable_bytes = root.get_as_ref_or_tagged(s_progress_downloadable_bytes_iip).get_as_int();
        uploadable_bytes = root.get_as_ref_or_tagged(s_progress_uploadable_bytes_iip).get_as_int();
        uploaded_bytes = root.get_as_ref_or_tagged(s_progress_uploaded_bytes_iip).get_as_int();
    }
}

auto ClientHistory::find_history_entry(version_type begin_version, version_type end_version,
                                       HistoryEntry& entry) const noexcept -> version_type
{
    version_type last_integrated_server_version;
    return find_sync_history_entry(*m_arrays, m_sync_history_base_version, begin_version, end_version, entry,
                                   last_integrated_server_version);
}


ChunkedBinaryData ClientHistory::get_reciprocal_transform(version_type version, bool& is_compressed) const
{
    is_compressed = true;
    REALM_ASSERT(version > m_sync_history_base_version);

    std::size_t index = to_size_t(version - m_sync_history_base_version) - 1;
    REALM_ASSERT(index < sync_history_size());

    ChunkedBinaryData reciprocal{m_arrays->reciprocal_transforms, index};
    if (!reciprocal.is_null())
        return reciprocal;
    return ChunkedBinaryData{m_arrays->changesets, index};
}


void ClientHistory::set_reciprocal_transform(version_type version, BinaryData data)
{
    REALM_ASSERT(version > m_sync_history_base_version);

    std::size_t index = size_t(version - m_sync_history_base_version) - 1;
    REALM_ASSERT(index < sync_history_size());

    if (data.is_null()) {
        m_arrays->reciprocal_transforms.set(index, BinaryData{"", 0}); // Throws
        return;
    }

    auto compressed = util::compression::allocate_and_compress_nonportable(data);
    m_arrays->reciprocal_transforms.set(index, BinaryData{compressed.data(), compressed.size()}); // Throws
}


auto ClientHistory::find_sync_history_entry(Arrays& arrays, version_type base_version, version_type begin_version,
                                            version_type end_version, HistoryEntry& entry,
                                            version_type& last_integrated_server_version) noexcept -> version_type
{
    if (begin_version == 0)
        begin_version = s_initial_version + 0;

    REALM_ASSERT(begin_version <= end_version);
    REALM_ASSERT(begin_version >= base_version);
    REALM_ASSERT(end_version <= base_version + arrays.changesets.size());
    std::size_t n = to_size_t(end_version - begin_version);
    std::size_t offset = to_size_t(begin_version - base_version);
    for (std::size_t i = 0; i < n; ++i) {
        std::int_fast64_t origin_file_ident = arrays.origin_file_idents.get(offset + i);
        last_integrated_server_version = version_type(arrays.remote_versions.get(offset + i));
        bool not_from_server = (origin_file_ident == 0);
        if (not_from_server) {
            ChunkedBinaryData chunked_changeset(arrays.changesets, offset + i);
            if (chunked_changeset.size() > 0) {
                entry.origin_file_ident = file_ident_type(origin_file_ident);
                entry.remote_version = last_integrated_server_version;
                entry.origin_timestamp = timestamp_type(arrays.origin_timestamps.get(offset + i));
                entry.changeset = chunked_changeset;
                return begin_version + i + 1;
            }
        }
    }
    return 0;
}

// sum_of_history_entry_sizes calculates the sum of the changeset sizes of the
// local history entries that produced a version that succeeds `begin_version`
// and precedes `end_version`.
std::uint_fast64_t ClientHistory::sum_of_history_entry_sizes(version_type begin_version,
                                                             version_type end_version) const noexcept
{
    if (begin_version >= end_version)
        return 0;

    REALM_ASSERT(m_arrays->changesets.is_attached());
    REALM_ASSERT(m_arrays->origin_file_idents.is_attached());
    REALM_ASSERT(end_version <= m_sync_history_base_version + sync_history_size());

    version_type begin_version_2 = begin_version;
    version_type end_version_2 = end_version;
    clamp_sync_version_range(begin_version_2, end_version_2);

    std::uint_fast64_t sum_of_sizes = 0;

    std::size_t n = to_size_t(end_version_2 - begin_version_2);
    std::size_t offset = to_size_t(begin_version_2 - m_sync_history_base_version);
    for (std::size_t i = 0; i < n; ++i) {

        // Only local changesets are considered
        if (m_arrays->origin_file_idents.get(offset + i) != 0)
            continue;

        ChunkedBinaryData changeset(m_arrays->changesets, offset + i);
        ChunkedBinaryInputStream in{changeset};
        sum_of_sizes += util::compression::get_uncompressed_size_from_header(in);
    }

    return sum_of_sizes;
}

void ClientHistory::prepare_for_write()
{
    if (m_arrays) {
        REALM_ASSERT(m_arrays->root.size() == s_root_size);
        return;
    }

    m_arrays.emplace(*m_db, *m_group);
}


Replication::version_type ClientHistory::add_changeset(BinaryData ct_changeset, BinaryData sync_changeset)
{
    // FIXME: BinaryColumn::set() currently interprets BinaryData(0,0) as
    // null. It should probably be changed such that BinaryData(0,0) is always
    // interpreted as the empty string. For the purpose of setting null values,
    // BinaryColumn::set() should accept values of type Optional<BinaryData>().
    if (ct_changeset.is_null())
        ct_changeset = BinaryData("", 0);
    m_arrays->ct_history.add(ct_changeset); // Throws

    REALM_ASSERT(!m_applying_server_changeset || !m_client_reset_changeset);

    // If we're applying a changeset from the server then we should have already
    // added the history entry and don't need to do so here
    if (m_applying_server_changeset) {
        // We need to unset this before committing the write, as it's guarded
        // by the write lock
        m_applying_server_changeset = false;
        REALM_ASSERT(m_ct_history_base_version + ct_history_size() ==
                     m_sync_history_base_version + sync_history_size());
        REALM_ASSERT(sync_changeset.size() == 0);
        return m_ct_history_base_version + ct_history_size();
    }

    BinaryData changeset = sync_changeset;
    if (m_client_reset_changeset) {
        // When performing a client reset, `sync_changeset` is generated from
        // the operations performed to bring the local Realm in sync with the
        // server, so we don't want to send that to the server. Instead we
        // send m_client_reset_changeset which is the recovered local writes
        // (or null in discard local mode).
        changeset = *std::exchange(m_client_reset_changeset, util::none);
    }

    HistoryEntry entry;
    entry.origin_timestamp = m_local_origin_timestamp_source();
    entry.origin_file_ident = 0; // Of local origin
    entry.remote_version = m_progress_download.server_version;
    entry.changeset = changeset;
    add_sync_history_entry(entry); // Throws

    // uploadable_bytes is updated at every local Realm change. The total
    // number of uploadable bytes must be persisted in the Realm, since the
    // synchronization history is trimmed. Even if the synchronization
    // history wasn't trimmed, it would be expensive to traverse the entire
    // history at every access to uploadable bytes.
    Array& root = m_arrays->root;
    std::uint_fast64_t uploadable_bytes = root.get_as_ref_or_tagged(s_progress_uploadable_bytes_iip).get_as_int();
    uploadable_bytes += entry.changeset.size();
    root.set(s_progress_uploadable_bytes_iip, RefOrTagged::make_tagged(uploadable_bytes));

    return m_ct_history_base_version + ct_history_size();
}

void ClientHistory::add_sync_history_entry(const HistoryEntry& entry)
{
    REALM_ASSERT(m_arrays->changesets.size() == sync_history_size());
    REALM_ASSERT(m_arrays->reciprocal_transforms.size() == sync_history_size());
    REALM_ASSERT(m_arrays->remote_versions.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_file_idents.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_timestamps.size() == sync_history_size());

    if (!entry.changeset.is_null()) {
        auto changeset = entry.changeset.get_first_chunk();
        auto compressed = util::compression::allocate_and_compress_nonportable(changeset);
        m_arrays->changesets.add(BinaryData{compressed.data(), compressed.size()}); // Throws
    }
    else {
        m_arrays->changesets.add(BinaryData("", 0)); // Throws
    }

    m_arrays->reciprocal_transforms.add(BinaryData{});                                            // Throws
    m_arrays->remote_versions.insert(realm::npos, std::int_fast64_t(entry.remote_version));       // Throws
    m_arrays->origin_file_idents.insert(realm::npos, std::int_fast64_t(entry.origin_file_ident)); // Throws
    m_arrays->origin_timestamps.insert(realm::npos, std::int_fast64_t(entry.origin_timestamp));   // Throws
}


void ClientHistory::update_sync_progress(const SyncProgress& progress, const std::uint_fast64_t* downloadable_bytes,
                                         TransactionRef)
{
    Array& root = m_arrays->root;

    // Progress must never decrease
    if (auto current = version_type(root.get_as_ref_or_tagged(s_progress_latest_server_version_iip).get_as_int());
        progress.latest_server_version.version < current) {
        throw IntegrationException(ErrorCodes::SyncProtocolInvariantFailed,
                                   util::format("latest server version cannot decrease (current: %1, new: %2)",
                                                current, progress.latest_server_version.version),
                                   ProtocolError::bad_progress);
    }
    if (auto current = version_type(root.get_as_ref_or_tagged(s_progress_download_server_version_iip).get_as_int());
        progress.download.server_version < current) {
        throw IntegrationException(
            ErrorCodes::SyncProtocolInvariantFailed,
            util::format("server version of download cursor cannot decrease (current: %1, new: %2)", current,
                         progress.download.server_version),
            ProtocolError::bad_progress);
    }
    if (auto current = version_type(root.get_as_ref_or_tagged(s_progress_download_client_version_iip).get_as_int());
        progress.download.last_integrated_client_version < current) {
        throw IntegrationException(
            ErrorCodes::SyncProtocolInvariantFailed,
            util::format("last integrated client version of download cursor cannot decrease (current: %1, new: %2)",
                         current, progress.download.last_integrated_client_version),
            ProtocolError::bad_progress);
    }
    if (auto current = version_type(root.get_as_ref_or_tagged(s_progress_upload_client_version_iip).get_as_int());
        progress.upload.client_version < current) {
        throw IntegrationException(
            ErrorCodes::SyncProtocolInvariantFailed,
            util::format("client version of upload cursor cannot decrease (current: %1, new: %2)", current,
                         progress.upload.client_version),
            ProtocolError::bad_progress);
    }
    const auto last_integrated_server_version = progress.upload.last_integrated_server_version;
    if (auto current = version_type(root.get_as_ref_or_tagged(s_progress_upload_server_version_iip).get_as_int());
        last_integrated_server_version > 0 && last_integrated_server_version < current) {
        throw IntegrationException(
            ErrorCodes::SyncProtocolInvariantFailed,
            util::format("last integrated server version of upload cursor cannot decrease (current: %1, new: %2)",
                         current, last_integrated_server_version),
            ProtocolError::bad_progress);
    }

    auto uploaded_bytes = std::uint_fast64_t(root.get_as_ref_or_tagged(s_progress_uploaded_bytes_iip).get_as_int());
    auto previous_upload_client_version =
        version_type(root.get_as_ref_or_tagged(s_progress_upload_client_version_iip).get_as_int());
    uploaded_bytes += sum_of_history_entry_sizes(previous_upload_client_version, progress.upload.client_version);

    root.set(s_progress_download_server_version_iip,
             RefOrTagged::make_tagged(progress.download.server_version)); // Throws
    root.set(s_progress_download_client_version_iip,
             RefOrTagged::make_tagged(progress.download.last_integrated_client_version)); // Throws
    root.set(s_progress_latest_server_version_iip,
             RefOrTagged::make_tagged(progress.latest_server_version.version)); // Throws
    root.set(s_progress_latest_server_version_salt_iip,
             RefOrTagged::make_tagged(progress.latest_server_version.salt)); // Throws
    root.set(s_progress_upload_client_version_iip,
             RefOrTagged::make_tagged(progress.upload.client_version)); // Throws
    if (progress.upload.last_integrated_server_version > 0) {
        root.set(s_progress_upload_server_version_iip,
                 RefOrTagged::make_tagged(progress.upload.last_integrated_server_version)); // Throws
    }

    if (downloadable_bytes) {
        root.set(s_progress_downloadable_bytes_iip,
                 RefOrTagged::make_tagged(*downloadable_bytes)); // Throws
    }
    root.set(s_progress_uploaded_bytes_iip,
             RefOrTagged::make_tagged(uploaded_bytes)); // Throws

    m_progress_download = progress.download;

    trim_sync_history(); // Throws
}


void ClientHistory::trim_ct_history()
{
    version_type begin = m_ct_history_base_version;
    version_type end = m_version_of_oldest_bound_snapshot;
    REALM_ASSERT(end >= begin);

    std::size_t n = std::size_t(end - begin);
    if (n == 0)
        return;

    // The new changeset is always added before set_oldest_bound_version()
    // is called. Therefore, the trimming operation can never leave the
    // history empty.
    REALM_ASSERT(n < ct_history_size());

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = (n - 1) - i;
        m_arrays->ct_history.erase(j);
    }

    m_ct_history_base_version += n;

    REALM_ASSERT(m_ct_history_base_version + ct_history_size() == m_sync_history_base_version + sync_history_size());
}


// Trimming rules for synchronization history:
//
// Let C be the latest client version that was integrated on the server prior to
// the latest server version currently integrated by the client
// (`m_progress_download.last_integrated_client_version`).
//
// Definition: An *upload skippable history entry* is one whose changeset is
// either empty, or of remote origin.
//
// Then, a history entry, E, can be trimmed away if it preceeds C, or E is
// upload skippable, and there are no upload nonskippable entries between C and
// E.
//
// Since the history representation is contiguous, it is necessary that the
// trimming rule upholds the following invariant:
//
// > If a changeset can be trimmed, then any earlier changeset can also be
// > trimmed.
//
// Note that C corresponds to the earliest possible beginning of the merge
// window for the next incoming changeset from the server.
void ClientHistory::trim_sync_history()
{
    version_type begin = m_sync_history_base_version;
    version_type end = std::max(m_progress_download.last_integrated_client_version, s_initial_version + 0);
    // Note: At this point, `end` corresponds to C in the description above.

    // `end` (`m_progress_download.last_integrated_client_version`) will precede
    // the beginning of the history, if we trimmed beyond
    // `m_progress_download.last_integrated_client_version` during the previous
    // trimming session. Since new entries, that have now become eligible for
    // scanning, may also be upload skippable, we need to continue the scan from
    // the beginning of the history in that case.
    if (end < begin)
        end = begin;

    // FIXME: It seems like in some cases, a particular history entry that
    // terminates the scan may get examined over and over every time
    // trim_history() is called. For this reason, it seems like it would be
    // worth considering to cache the outcome.

    // FIXME: It seems like there is a significant overlap between what is going
    // on here and in a place like find_uploadable_changesets(). Maybe there is
    // grounds for some refactoring to take that into account, especially, to
    // avoid scanning the same parts of the history for the same information
    // multiple times.

    {
        std::size_t offset = std::size_t(end - begin);
        std::size_t n = std::size_t(sync_history_size() - offset);
        std::size_t i = 0;
        while (i < n) {
            std::int_fast64_t origin_file_ident = m_arrays->origin_file_idents.get(offset + i);
            bool of_local_origin = (origin_file_ident == 0);
            if (of_local_origin) {
                std::size_t pos = 0;
                BinaryData chunk = m_arrays->changesets.get_at(offset + i, pos);
                bool nonempty = (chunk.size() > 0);
                if (nonempty)
                    break; // Not upload skippable
            }
            ++i;
        }
        end += i;
    }

    std::size_t n = std::size_t(end - begin);
    do_trim_sync_history(n); // Throws
}

bool ClientHistory::no_pending_local_changes(version_type version) const
{
    ensure_updated(version);
    for (size_t i = 0; i < sync_history_size(); i++) {
        if (m_arrays->origin_file_idents.get(i) == 0) {
            std::size_t pos = 0;
            BinaryData chunk = m_arrays->changesets.get_at(i, pos);
            if (chunk.size() > 0)
                return false;
        }
    }
    return true;
}

void ClientHistory::do_trim_sync_history(std::size_t n)
{
    REALM_ASSERT(m_arrays->changesets.size() == sync_history_size());
    REALM_ASSERT(m_arrays->reciprocal_transforms.size() == sync_history_size());
    REALM_ASSERT(m_arrays->remote_versions.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_file_idents.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_timestamps.size() == sync_history_size());
    REALM_ASSERT(n <= sync_history_size());

    if (n > 0) {
        // FIXME: shouldn't this be using truncate()?
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n - 1) - i;
            m_arrays->changesets.erase(j); // Throws
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n - 1) - i;
            m_arrays->reciprocal_transforms.erase(j); // Throws
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n - 1) - i;
            m_arrays->remote_versions.erase(j); // Throws
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n - 1) - i;
            m_arrays->origin_file_idents.erase(j); // Throws
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n - 1) - i;
            m_arrays->origin_timestamps.erase(j); // Throws
        }

        m_sync_history_base_version += n;
    }
}

void ClientHistory::fix_up_client_file_ident_in_stored_changesets(Transaction& group,
                                                                  file_ident_type client_file_ident)
{
    // Must be in write transaction!

    REALM_ASSERT(client_file_ident != 0);
    auto promote_global_key = [client_file_ident](GlobalKey* oid) {
        if (oid->hi() == 0) {
            // client_file_ident == 0
            *oid = GlobalKey{uint64_t(client_file_ident), oid->lo()};
            return true;
        }
        return false;
    };

    Group::TableNameBuffer buffer;
    auto get_table_for_class = [&](StringData class_name) -> ConstTableRef {
        REALM_ASSERT(class_name.size() < Group::max_table_name_length - 6);
        return group.get_table(Group::class_name_to_table_name(class_name, buffer));
    };

    util::compression::CompressMemoryArena arena;
    util::AppendBuffer<char> compressed;

    // Fix up changesets.
    Array& root = m_arrays->root;
    uint64_t uploadable_bytes = root.get_as_ref_or_tagged(s_progress_uploadable_bytes_iip).get_as_int();
    for (size_t i = 0; i < sync_history_size(); ++i) {
        // We could have opened a pre-provisioned realm file. In this case we can skip the entries downloaded
        // from the server.
        if (m_arrays->origin_file_idents.get(i) != 0)
            continue;

        ChunkedBinaryData changeset{m_arrays->changesets, i};
        ChunkedBinaryInputStream is{changeset};
        size_t decompressed_size;
        auto decompressed = util::compression::decompress_nonportable_input_stream(is, decompressed_size);
        if (!decompressed)
            continue;
        Changeset log;
        parse_changeset(*decompressed, log);

        bool did_modify = false;
        auto last_class_name = InternString::npos;
        ConstTableRef selected_table;
        for (auto instr : log) {
            if (!instr)
                continue;

            if (auto obj_instr = instr->get_if<Instruction::ObjectInstruction>()) {
                // Cache the TableRef
                if (obj_instr->table != last_class_name) {
                    StringData class_name = log.get_string(obj_instr->table);
                    last_class_name = obj_instr->table;
                    selected_table = get_table_for_class(class_name);
                }

                // Fix up instructions using GlobalKey to identify objects.
                if (auto global_key = mpark::get_if<GlobalKey>(&obj_instr->object)) {
                    did_modify = promote_global_key(global_key);
                }

                // Fix up the payload for Set and ArrayInsert.
                Instruction::Payload* payload = nullptr;
                if (auto set_instr = instr->get_if<Instruction::Update>()) {
                    payload = &set_instr->value;
                }
                else if (auto list_insert_instr = instr->get_if<Instruction::ArrayInsert>()) {
                    payload = &list_insert_instr->value;
                }

                if (payload && payload->type == Instruction::Payload::Type::Link) {
                    if (auto global_key = mpark::get_if<GlobalKey>(&payload->data.link.target)) {
                        did_modify = promote_global_key(global_key);
                    }
                }
            }
        }

        if (did_modify) {
            ChangesetEncoder::Buffer modified;
            encode_changeset(log, modified);
            util::compression::allocate_and_compress_nonportable(arena, modified, compressed);
            m_arrays->changesets.set(i, BinaryData{compressed.data(), compressed.size()}); // Throws

            uploadable_bytes += modified.size() - decompressed_size;
        }
    }

    root.set(s_progress_uploadable_bytes_iip, RefOrTagged::make_tagged(uploadable_bytes));
}

void ClientHistory::set_group(Group* group, bool updated)
{
    _impl::History::set_group(group, updated);
    if (m_arrays)
        _impl::GroupFriend::set_history_parent(*m_group, m_arrays->root);
}

void ClientHistory::record_current_schema_version()
{
    using gf = _impl::GroupFriend;
    Allocator& alloc = gf::get_alloc(*m_group);
    auto ref = gf::get_history_ref(*m_group);
    REALM_ASSERT(ref != 0);
    Array root{alloc};
    gf::set_history_parent(*m_group, root);
    root.init_from_ref(ref);
    Array schema_versions{alloc};
    schema_versions.set_parent(&root, s_schema_versions_iip);
    schema_versions.init_from_parent();
    version_type snapshot_version = m_db->get_version_of_latest_snapshot();
    record_current_schema_version(schema_versions, snapshot_version); // Throws
}


void ClientHistory::record_current_schema_version(Array& schema_versions, version_type snapshot_version)
{
    static_assert(s_schema_versions_size == 4, "");
    REALM_ASSERT(schema_versions.size() == s_schema_versions_size);

    Allocator& alloc = schema_versions.get_alloc();
    {
        Array sv_schema_versions{alloc};
        sv_schema_versions.set_parent(&schema_versions, s_sv_schema_versions_iip);
        sv_schema_versions.init_from_parent();
        int schema_version = get_client_history_schema_version();
        sv_schema_versions.add(schema_version); // Throws
    }
    {
        Array sv_library_versions{alloc};
        sv_library_versions.set_parent(&schema_versions, s_sv_library_versions_iip);
        sv_library_versions.init_from_parent();
        const char* library_version = REALM_VERSION_STRING;
        std::size_t size = std::strlen(library_version);
        Array value{alloc};
        bool context_flag = false;
        value.create(Array::type_Normal, context_flag, size); // Throws
        _impl::ShallowArrayDestroyGuard adg{&value};
        using uchar = unsigned char;
        for (std::size_t i = 0; i < size; ++i)
            value.set(i, std::int_fast64_t(uchar(library_version[i]))); // Throws
        sv_library_versions.add(std::int_fast64_t(value.get_ref()));    // Throws
        adg.release();                                                  // Ownership transferred to parent array
    }
    {
        Array sv_snapshot_versions{alloc};
        sv_snapshot_versions.set_parent(&schema_versions, s_sv_snapshot_versions_iip);
        sv_snapshot_versions.init_from_parent();
        sv_snapshot_versions.add(std::int_fast64_t(snapshot_version)); // Throws
    }
    {
        Array sv_timestamps{alloc};
        sv_timestamps.set_parent(&schema_versions, s_sv_timestamps_iip);
        sv_timestamps.init_from_parent();
        std::time_t timestamp = std::time(nullptr);
        sv_timestamps.add(std::int_fast64_t(timestamp)); // Throws
    }
}

// Overriding member function in realm::_impl::History
void ClientHistory::update_from_ref_and_version(ref_type ref, version_type version)
{
    if (ref == 0) {
        // No history
        m_ct_history_base_version = version;
        m_sync_history_base_version = version;
        m_arrays.reset();
        m_progress_download = {0, 0};
        return;
    }
    if (REALM_LIKELY(m_arrays)) {
        m_arrays->init_from_ref(ref);
    }
    else {
        REALM_ASSERT_RELEASE(m_group);
        m_arrays.emplace(m_db->get_alloc(), *m_group, ref);
    }

    m_ct_history_base_version = version - ct_history_size();
    m_sync_history_base_version = version - sync_history_size();
    REALM_ASSERT(m_arrays->changesets.size() == sync_history_size());
    REALM_ASSERT(m_arrays->reciprocal_transforms.size() == sync_history_size());
    REALM_ASSERT(m_arrays->remote_versions.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_file_idents.size() == sync_history_size());
    REALM_ASSERT(m_arrays->origin_timestamps.size() == sync_history_size());

    const Array& root = m_arrays->root;
    m_progress_download.server_version =
        version_type(root.get_as_ref_or_tagged(s_progress_download_server_version_iip).get_as_int());
    m_progress_download.last_integrated_client_version =
        version_type(root.get_as_ref_or_tagged(s_progress_download_client_version_iip).get_as_int());
}


// Overriding member function in realm::_impl::History
void ClientHistory::update_from_parent(version_type current_version)
{
    using gf = _impl::GroupFriend;
    ref_type ref = gf::get_history_ref(*m_group);
    update_from_ref_and_version(ref, current_version); // Throws
}


// Overriding member function in realm::_impl::History
void ClientHistory::get_changesets(version_type begin_version, version_type end_version,
                                   BinaryIterator* iterators) const noexcept
{
    REALM_ASSERT(begin_version <= end_version);
    REALM_ASSERT(begin_version >= m_ct_history_base_version);
    REALM_ASSERT(end_version <= m_ct_history_base_version + ct_history_size());
    std::size_t n = to_size_t(end_version - begin_version);
    REALM_ASSERT(n == 0 || m_arrays);
    std::size_t offset = to_size_t(begin_version - m_ct_history_base_version);
    for (std::size_t i = 0; i < n; ++i)
        iterators[i] = BinaryIterator(&m_arrays->ct_history, offset + i);
}


// Overriding member function in realm::_impl::History
void ClientHistory::set_oldest_bound_version(version_type version)
{
    REALM_ASSERT(version >= m_version_of_oldest_bound_snapshot);
    if (version > m_version_of_oldest_bound_snapshot) {
        m_version_of_oldest_bound_snapshot = version;
        trim_ct_history(); // Throws
    }
}

// Overriding member function in realm::_impl::History
void ClientHistory::verify() const
{
#ifdef REALM_DEBUG
    // The size of the continuous transactions history can only be zero when the
    // Realm is in the initial empty state where top-ref is null.
    REALM_ASSERT(ct_history_size() != 0 || m_ct_history_base_version == s_initial_version + 0);

    if (!m_arrays) {
        REALM_ASSERT(m_progress_download.server_version == 0);
        REALM_ASSERT(m_progress_download.last_integrated_client_version == 0);
        return;
    }
    m_arrays->verify();

    auto& root = m_arrays->root;
    version_type progress_download_server_version =
        version_type(root.get_as_ref_or_tagged(s_progress_download_server_version_iip).get_as_int());
    version_type progress_download_client_version =
        version_type(root.get_as_ref_or_tagged(s_progress_download_client_version_iip).get_as_int());
    REALM_ASSERT(progress_download_server_version == m_progress_download.server_version);
    REALM_ASSERT(progress_download_client_version == m_progress_download.last_integrated_client_version);
    REALM_ASSERT(progress_download_client_version <= m_sync_history_base_version + sync_history_size());
    version_type remote_version_of_last_entry = 0;
    if (auto size = sync_history_size())
        remote_version_of_last_entry = m_arrays->remote_versions.get(size - 1);
    REALM_ASSERT(progress_download_server_version >= remote_version_of_last_entry);

    // Verify that there is no cooked history.
    Array cooked_history{m_db->get_alloc()};
    cooked_history.set_parent(&root, s_cooked_history_iip);
    REALM_ASSERT(cooked_history.get_ref_from_parent() == 0);
#endif // REALM_DEBUG
}

ClientHistory::Arrays::Arrays(Allocator& alloc) noexcept
    : root(alloc)
    , ct_history(alloc)
    , changesets(alloc)
    , reciprocal_transforms(alloc)
    , remote_versions(alloc)
    , origin_file_idents(alloc)
    , origin_timestamps(alloc)
{
}

ClientHistory::Arrays::Arrays(DB& db, Group& group)
    : Arrays(db.get_alloc())
{
    auto& alloc = db.get_alloc();
    {
        bool context_flag = false;
        std::size_t size = s_root_size;
        root.create(Array::type_HasRefs, context_flag, size); // Throws
    }
    _impl::DeepArrayDestroyGuard dg{&root};

    ct_history.set_parent(&root, s_ct_history_iip);
    ct_history.create(); // Throws
    changesets.set_parent(&root, s_changesets_iip);
    changesets.create(); // Throws
    reciprocal_transforms.set_parent(&root, s_reciprocal_transforms_iip);
    reciprocal_transforms.create(); // Throws
    remote_versions.set_parent(&root, s_remote_versions_iip);
    remote_versions.create(); // Throws
    origin_file_idents.set_parent(&root, s_origin_file_idents_iip);
    origin_file_idents.create(); // Throws
    origin_timestamps.set_parent(&root, s_origin_timestamps_iip);
    origin_timestamps.create(); // Throws

    { // `schema_versions` table
        Array schema_versions{alloc};
        bool context_flag = false;
        std::size_t size = s_schema_versions_size;
        schema_versions.create(Array::type_HasRefs, context_flag, size); // Throws
        _impl::DeepArrayDestroyGuard adg{&schema_versions};

        auto create_array = [&](NodeHeader::Type type, int ndx_in_parent) {
            MemRef mem = Array::create_empty_array(type, context_flag, alloc);
            ref_type ref = mem.get_ref();
            _impl::DeepArrayRefDestroyGuard ardg{ref, alloc};
            schema_versions.set_as_ref(ndx_in_parent, ref); // Throws
            ardg.release();                                 // Ownership transferred to parent array
        };
        create_array(Array::type_Normal, s_sv_schema_versions_iip);
        create_array(Array::type_HasRefs, s_sv_library_versions_iip);
        create_array(Array::type_Normal, s_sv_snapshot_versions_iip);
        create_array(Array::type_Normal, s_sv_timestamps_iip);

        version_type snapshot_version = db.get_version_of_latest_snapshot();
        record_current_schema_version(schema_versions, snapshot_version);  // Throws
        root.set_as_ref(s_schema_versions_iip, schema_versions.get_ref()); // Throws
        adg.release();                                                     // Ownership transferred to parent array
    }
    _impl::GroupFriend::prepare_history_parent(group, root, Replication::hist_SyncClient,
                                               get_client_history_schema_version(), 0); // Throws
    // Note: gf::prepare_history_parent() also ensures the the root array has a
    // slot for the history ref.
    root.update_parent(); // Throws
    dg.release();
}

ClientHistory::Arrays::Arrays(Allocator& alloc, Group& parent, ref_type ref)
    : Arrays(alloc)
{
    using gf = _impl::GroupFriend;
    root.init_from_ref(ref);
    gf::set_history_parent(parent, root);

    ct_history.set_parent(&root, s_ct_history_iip);
    changesets.set_parent(&root, s_changesets_iip);
    reciprocal_transforms.set_parent(&root, s_reciprocal_transforms_iip);
    remote_versions.set_parent(&root, s_remote_versions_iip);
    origin_file_idents.set_parent(&root, s_origin_file_idents_iip);
    origin_timestamps.set_parent(&root, s_origin_timestamps_iip);

    init_from_ref(ref); // Throws

    Array cooked_history{alloc};
    cooked_history.set_parent(&root, s_cooked_history_iip);
    // We should have no cooked history in existing Realms.
    REALM_ASSERT(cooked_history.get_ref_from_parent() == 0);
}

void ClientHistory::Arrays::Arrays::init_from_ref(ref_type ref)
{
    root.init_from_ref(ref);
    REALM_ASSERT(root.size() == s_root_size);
    {
        ref_type ref_2 = root.get_as_ref(s_ct_history_iip);
        ct_history.init_from_ref(ref_2); // Throws
    }
    {
        ref_type ref_2 = root.get_as_ref(s_changesets_iip);
        changesets.init_from_ref(ref_2); // Throws
    }
    {
        ref_type ref_2 = root.get_as_ref(s_reciprocal_transforms_iip);
        reciprocal_transforms.init_from_ref(ref_2); // Throws
    }
    remote_versions.init_from_parent();    // Throws
    origin_file_idents.init_from_parent(); // Throws
    origin_timestamps.init_from_parent();  // Throws
}

void ClientHistory::Arrays::verify() const
{
#ifdef REALM_DEBUG
    root.verify();
    ct_history.verify();
    changesets.verify();
    reciprocal_transforms.verify();
    remote_versions.verify();
    origin_file_idents.verify();
    origin_timestamps.verify();
    REALM_ASSERT(root.size() == s_root_size);
    REALM_ASSERT(reciprocal_transforms.size() == changesets.size());
    REALM_ASSERT(remote_versions.size() == changesets.size());
    REALM_ASSERT(origin_file_idents.size() == changesets.size());
    REALM_ASSERT(origin_timestamps.size() == changesets.size());
#endif // REALM_DEBUG
}

} // namespace realm::sync
