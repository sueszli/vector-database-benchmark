/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <algorithm>
#include <assert.h>
#include <unistd.h>
#include <eventql/util/io/fileutil.h>
#include <eventql/db/partition.h>
#include <eventql/db/partition_writer.h>
#include <eventql/db/file_tracker.h>
#include <eventql/db/metadata_file.h>
#include <eventql/db/partition_reader.h>
#include <eventql/db/server_allocator.h>
#include <eventql/db/metadata_operations.pb.h>
#include <eventql/db/metadata_coordinator.h>
#include <eventql/db/metadata_client.h>
#include <eventql/db/compaction_worker.h>
#include <eventql/db/replication_state.h>
#include <eventql/util/logging.h>
#include <eventql/util/random.h>
#include <eventql/util/wallclock.h>
#include <eventql/io/sstable/SSTableWriter.h>
#include "eventql/eventql.h"

namespace eventql {

PartitionWriter::PartitionWriter(
    PartitionSnapshotRef* head) :
    head_(head),
    frozen_(false) {}

void PartitionWriter::lock() {
  mutex_.lock();
}

void PartitionWriter::unlock() {
  mutex_.unlock();
}

void PartitionWriter::freeze() {
  frozen_ = true;
}

const size_t LSMPartitionWriter::kDefaultPartitionSplitThresholdBytes = 1024llu * 1024llu * 512llu;
const size_t LSMPartitionWriter::kDefaultPartitionSplitThresholdRows = 2000000llu;
const size_t LSMPartitionWriter::kMaxArenaRecordsSoft = 1024 * 128;
const size_t LSMPartitionWriter::kMaxArenaRecordsHard = 1024 * 1024 * 2;
const size_t LSMPartitionWriter::kMaxLSMTables = 96;
const size_t LSMPartitionWriter::kSplitRetryInterval = 5 * kMicrosPerSecond;

LSMPartitionWriter::LSMPartitionWriter(
    DatabaseContext* dbctx,
    RefPtr<Partition> partition,
    PartitionSnapshotRef* head) :
    PartitionWriter(head),
    partition_(partition),
    compaction_strategy_(
        new SimpleCompactionStrategy(
            partition_,
            dbctx->lsm_index_cache)),
    dbctx_(dbctx),
    partition_split_threshold_bytes_(kDefaultPartitionSplitThresholdBytes),
    partition_split_threshold_rows_(kDefaultPartitionSplitThresholdRows),
    split_started_(false),
    split_cancelled_(false) {
  auto table = partition_->getTable();
  auto table_cfg = table->config();

  if (table_cfg.config().override_partition_split_threshold() > 0) {
    partition_split_threshold_bytes_ =
        table_cfg.config().override_partition_split_threshold();
  }
}

LSMPartitionWriter::~LSMPartitionWriter() {
  ScopedLock<std::mutex> split_lk(split_mutex_);
  if (split_started_) {
    split_cancelled_ = true;
    split_cv_.notify_all();
    split_lk.unlock();
    split_thread_.join();
  }
}

Set<SHA1Hash> LSMPartitionWriter::insertRecords(
    const ShreddedRecordList& records) {
  HashMap<SHA1Hash, uint64_t> rec_versions;
  for (size_t i = 0; i < records.getNumRecords(); ++i) {
    rec_versions.emplace(records.getRecordID(i), 0);
  }

  // opportunistically fetch indexes before going into critical section
  auto snap = head_->getSnapshot();
  Set<String> prepared_indexes;
  {
    const auto& tables = snap->state.lsm_tables();
    for (auto tbl = tables.rbegin(); tbl != tables.rend(); ++tbl) {
      auto idx_path = FileUtil::joinPaths(snap->rel_path, tbl->filename());
      auto idx = dbctx_->lsm_index_cache->lookup(idx_path);
      idx->lookup(&rec_versions);
      prepared_indexes.insert(idx_path);
    }
  }

  std::unique_lock<std::mutex> lk(mutex_);
  if (frozen_) {
    RAISE(kIllegalStateError, "partition is frozen");
  }

  snap = head_->getSnapshot();
  const auto& tables = snap->state.lsm_tables();
  if (size_t(tables.size()) > kMaxLSMTables) {
    RAISE(kRuntimeError, "partition is overloaded, can't insert");
  }

  logTrace(
      "tsdb",
      "Insert $0 record into partition $1/$2/$3",
      records.getNumRecords(),
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString());

  Set<SHA1Hash> inserted_ids;
  try {
    if (snap->compacting_arena.get() != nullptr) {
      for (auto& r : rec_versions) {
        auto v = snap->compacting_arena->fetchRecordVersion(r.first);
        if (v > r.second) {
          r.second = v;
        }
      }
    }

    for (auto tbl = tables.rbegin(); tbl != tables.rend(); ++tbl) {
      auto idx_path = FileUtil::joinPaths(snap->rel_path, tbl->filename());
      if (prepared_indexes.count(idx_path) > 0) {
        continue;
      }

      auto idx = dbctx_->lsm_index_cache->lookup(idx_path);
      idx->lookup(&rec_versions);
    }

    Vector<bool> record_flags_skip(records.getNumRecords(), false);
    Vector<bool> record_flags_update(records.getNumRecords(), false);

    if (!rec_versions.empty()) {
      for (size_t i = 0; i < records.getNumRecords(); ++i) {
        const auto& record_id = records.getRecordID(i);
        auto headv = rec_versions[record_id];
        if (headv > 0) {
          assert(headv > 1400000000000000);
          record_flags_update[i] = true;
        }

        auto thisv = records.getRecordVersion(i);
        assert(thisv > 1400000000000000);

        if (thisv <= headv) {
          record_flags_skip[i] = true;
          continue;
        }
      }
    }

    inserted_ids = snap->head_arena->insertRecords(
        records,
        record_flags_skip,
        record_flags_update);

    lk.unlock();
  } catch (const std::exception& e) {
    logCritical("evqld", "error in insert routine: $0", e.what());
    abort();
  }

  if (needsPromptCommit()) {
    dbctx_->compaction_worker->enqueuePartition(partition_, true);
  }

  if (needsUrgentCommit()) {
    logWarning(
        "evqld",
        "Partition $0/$1/$2 needs urgent commit -- overloaded or "
        "kMaxArenaRecordsHard too low?",
        snap->state.tsdb_namespace(),
        snap->state.table_key(),
        snap->key.toString());

    commit();
  }

  if (needsUrgentCompaction()) {
    logWarning(
        "evqld",
        "Partition $0/$1/$2 needs urgent compaction -- overloaded or "
        "kMaxLSMTables too low?",
        snap->state.tsdb_namespace(),
        snap->state.table_key(),
        snap->key.toString());

    compact();
  }

  return inserted_ids;
}

bool LSMPartitionWriter::needsCommit() {
  return head_->getSnapshot()->head_arena->size() > 0;
}

bool LSMPartitionWriter::needsPromptCommit() {
  if (head_->getSnapshot()->head_arena->size() < kMaxArenaRecordsSoft) {
    return false;
  }

  if (compactionRunning()) {
    return false;
  }

  return true;
}

bool LSMPartitionWriter::needsUrgentCommit() {
  return head_->getSnapshot()->head_arena->size() > kMaxArenaRecordsHard;
}

bool LSMPartitionWriter::needsCompaction() {
  if (needsCommit()) {
    return true;
  }

  auto snap = head_->getSnapshot();
  return compaction_strategy_->needsCompaction(
      Vector<LSMTableRef>(
          snap->state.lsm_tables().begin(),
          snap->state.lsm_tables().end()));
}

bool LSMPartitionWriter::needsUrgentCompaction() {
  auto snap = head_->getSnapshot();
  return compaction_strategy_->needsUrgentCompaction(
      Vector<LSMTableRef>(
          snap->state.lsm_tables().begin(),
          snap->state.lsm_tables().end()));
}

bool LSMPartitionWriter::commit() {
  ScopedLock<std::mutex> commit_lk(commit_mutex_);
  RefPtr<PartitionArena> arena;

  // flip arenas if records pending
  {
    ScopedLock<std::mutex> write_lk(mutex_);
    auto snap = head_->getSnapshot()->clone();
    if (snap->compacting_arena.get() == nullptr &&
        snap->head_arena->size() > 0) {
      snap->compacting_arena = snap->head_arena;
      snap->head_arena = mkRef(
          new PartitionArena(*partition_->getTable()->schema()));
      head_->setSnapshot(snap);
    }
    arena = snap->compacting_arena;
  }

  // flush arena to disk if pending
  bool commited = false;
  if (arena.get() && arena->size() > 0) {
    auto snap = head_->getSnapshot();
    auto filename = Random::singleton()->hex64();
    auto filepath = FileUtil::joinPaths(snap->base_path, filename);
    auto t0 = WallClock::unixMicros();
    {
      auto rc = arena->writeToDisk(filepath, snap->state.lsm_sequence() + 1);
      if (!rc.isSuccess()) {
        logError(
            "evqld",
            "Error while commiting partition $0/$1/$2: $3",
            snap->state.tsdb_namespace(),
            snap->state.table_key(),
            snap->key.toString(),
            rc.message());

        return false;
      }
    }

    auto t1 = WallClock::unixMicros();

    logDebug(
        "evqld",
        "Committing partition $3/$4/$5 (num_records=$0, sequence=$1..$2), took $6s",
        arena->size(),
        snap->state.lsm_sequence() + 1,
        snap->state.lsm_sequence() + arena->size(),
        snap->state.tsdb_namespace(),
        snap->state.table_key(),
        snap->key.toString(),
        (double) (t1 - t0) / 1000000.0f);

    ScopedLock<std::mutex> write_lk(mutex_);
    snap = head_->getSnapshot()->clone();
    auto tblref = snap->state.add_lsm_tables();
    tblref->set_filename(filename);
    tblref->set_first_sequence(snap->state.lsm_sequence() + 1);
    tblref->set_last_sequence(snap->state.lsm_sequence() + arena->size());
    tblref->set_size_bytes(FileUtil::size(filepath + ".cst"));
    tblref->set_size_rows(arena->size());
    tblref->set_has_skiplist(arena->hasSkiplist());
    tblref->set_has_updates(arena->hasUpdate());
    snap->state.set_lsm_sequence(snap->state.lsm_sequence() + arena->size());
    snap->compacting_arena = nullptr;
    snap->writeToDisk();
    head_->setSnapshot(snap);
    commited = true;
  }

  commit_lk.unlock();

  if (needsSplit()) {
    auto rc = split();
    if (!rc.isSuccess()) {
      logWarning("evqld", "partition split failed: $0", rc.message());
    }
  }

  return commited;
}

bool LSMPartitionWriter::compactionRunning() {
  ScopedLock<std::mutex> compact_lk(compaction_mutex_, std::defer_lock);
  if (compact_lk.try_lock()) {
    return false;
  } else {
    return true;
  }
}

bool LSMPartitionWriter::compact(bool force /* = false */) {
  ScopedLock<std::mutex> compact_lk(compaction_mutex_, std::defer_lock);
  if (!compact_lk.try_lock()) {
    return false;
  }

  auto dirty = commit();

  // fetch current table list
  auto snap = head_->getSnapshot()->clone();

  Vector<LSMTableRef> new_tables;
  Vector<LSMTableRef> old_tables(
      snap->state.lsm_tables().begin(),
      snap->state.lsm_tables().end());

  if (!force && !compaction_strategy_->needsCompaction(old_tables)) {
    return dirty;
  }

  // compact
  auto t0 = WallClock::unixMicros();
  if (!compaction_strategy_->compact(old_tables, &new_tables)) {
    return dirty;
  }
  auto t1 = WallClock::unixMicros();

  logDebug(
      "evqld",
      "Compacting partition $0/$1/$2, took $3s",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString(),
      (double) (t1 - t0) / 1000000.0f);

  // commit table list
  ScopedLock<std::mutex> write_lk(mutex_);
  snap = head_->getSnapshot()->clone();

  if (size_t(snap->state.lsm_tables().size()) < old_tables.size()) {
    RAISE(kConcurrentModificationError, "concurrent compaction");
  }

  size_t i = 0;
  for (const auto& tbl : snap->state.lsm_tables()) {
    if (i < old_tables.size()) {
      if (old_tables[i].filename() != tbl.filename()) {
        RAISE(kConcurrentModificationError, "concurrent compaction");
      }
    } else {
      new_tables.push_back(tbl);
    }

    ++i;
  }

  snap->state.mutable_lsm_tables()->Clear();
  for (const auto& tbl :  new_tables) {
    *snap->state.add_lsm_tables() = tbl;
  }

  snap->writeToDisk();
  head_->setSnapshot(snap);
  write_lk.unlock();

  // delete
  Set<String> delete_filenames;
  for (const auto& tbl : old_tables) {
    delete_filenames.emplace(tbl.filename());
  }
  for (const auto& tbl : new_tables) {
    delete_filenames.erase(tbl.filename());
  }

  compact_lk.unlock();

  {
    Set<String> delete_filenames_full;
    for (const auto& f : delete_filenames) {
      auto fpath = FileUtil::joinPaths(snap->rel_path, f);
      delete_filenames_full.insert(fpath + ".cst");
      delete_filenames_full.insert(fpath + ".idx");
      dbctx_->lsm_index_cache->flush(fpath);
    }

    dbctx_->file_tracker->deleteFiles(delete_filenames_full);
  }

  // maybe split this partition
  if (needsSplit()) {
    auto rc = split();
    if (!rc.isSuccess()) {
      logWarning("evqld", "partition split failed: $0", rc.message());
    }
  }

  return true;
}

bool LSMPartitionWriter::needsSplit() const {
  if (partition_->getTable()->hasUserDefinedPartitions()) {
    return false;
  }

  auto snap = head_->getSnapshot();
  if (snap->state.is_splitting() || split_started_) {
    return false;
  }

  switch (snap->state.lifecycle_state()) {
    case PDISCOVERY_LOAD:
    case PDISCOVERY_SERVE:
      break;
    case PDISCOVERY_UNLOAD:
    case PDISCOVERY_UNKNOWN:
      return false;
  }

  size_t size_bytes = 0;
  size_t size_rows = 0;
  for (const auto& tbl : snap->state.lsm_tables()) {
    size_bytes += tbl.size_bytes();
    size_rows += tbl.size_rows();
  }

  return
      size_bytes > partition_split_threshold_bytes_ ||
      size_rows > partition_split_threshold_rows_;
}

Status LSMPartitionWriter::split() {
  ScopedLock<std::mutex> split_lk(split_mutex_, std::defer_lock);
  if (!split_lk.try_lock() || split_started_) {
    return Status(eConcurrentModificationError, "split is already running");
  }

  auto snap = head_->getSnapshot();
  auto table = partition_->getTable();
  auto keyspace = table->getKeyspaceType();

  String midpoint;
  {
    auto cmp = [keyspace] (const String& a, const String& b) -> bool {
      return comparePartitionKeys(
          keyspace,
          encodePartitionKey(keyspace, a),
          encodePartitionKey(keyspace, b)) < 0;
    };

    LSMPartitionReader reader(table, snap);
    String minval;
    String maxval;
    auto rc = reader.findMedianValue(
        table->getPartitionKey(),
        cmp,
        &minval,
        &midpoint,
        &maxval);

    if (!rc.isSuccess()) {
      return rc;
    }

    if (minval == midpoint || maxval == midpoint) {
      return Status(eRuntimeError, "no suitable split point found");
    }
  }

  logInfo(
      "evqld",
      "Splitting partition $0/$1/$2 at '$3'",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString(),
      midpoint);

  split_started_ = true;
  split_thread_ = std::thread([this, midpoint] {
    while (!commitSplit(midpoint).isSuccess()) {
      ScopedLock<std::mutex> lk(split_mutex_);
      if (split_cancelled_) {
        return;
      }

      split_cv_.wait_for(lk, std::chrono::microseconds(kSplitRetryInterval));
      if (split_cancelled_) {
        return;
      }
    }
  });

  return Status::success();
}

Status LSMPartitionWriter::commitSplit(const std::string& midpoint) {
  auto snap = head_->getSnapshot();
  auto table = partition_->getTable();
  auto keyspace = table->getKeyspaceType();

  if (snap->state.is_splitting()) {
    return Status::success();
  }

  switch (snap->state.lifecycle_state()) {
    case PDISCOVERY_LOAD:
    case PDISCOVERY_SERVE:
      break;
    case PDISCOVERY_UNLOAD:
    case PDISCOVERY_UNKNOWN:
      return Status(eSuccess);
  }

  // FIXME make explicit discovery request and check if we're already splitting
  // to work around local metadata update delay?

  logInfo(
      "evqld",
      "Comitting partition split for $0/$1/$2 at '$3'",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString(),
      midpoint);

  auto cconf = dbctx_->config_directory->getClusterConfig();
  auto split_partition_id_low = Random::singleton()->sha1();
  auto split_partition_id_high = Random::singleton()->sha1();

  SplitPartitionOperation op;
  op.set_partition_id(snap->key.data(), snap->key.size());
  op.set_split_point(encodePartitionKey(keyspace, midpoint));
  op.set_split_partition_id_low(
      split_partition_id_low.data(),
      split_partition_id_low.size());
  op.set_split_partition_id_high(
      split_partition_id_high.data(),
      split_partition_id_high.size());
  op.set_placement_id(Random::singleton()->random64());

  if (table->config().config().enable_async_split()) {
    op.set_finalize_immediately(true);
  }

  std::vector<String> split_servers_low;
  {
    auto rc = dbctx_->server_alloc->allocateServers(
        ServerAllocator::MUST_ALLOCATE,
        cconf.replication_factor(),
        Set<String>{},
        &split_servers_low);
    if (!rc.isSuccess()) {
      return rc;
    }
  }

  for (const auto& s : split_servers_low) {
    op.add_split_servers_low(s);
  }

  std::vector<String> split_servers_high;
  {
    auto rc = dbctx_->server_alloc->allocateServers(
        ServerAllocator::MUST_ALLOCATE,
        cconf.replication_factor(),
        Set<String>{},
        &split_servers_high);
    if (!rc.isSuccess()) {
      return rc;
    }
  }

  for (const auto& s : split_servers_high) {
    op.add_split_servers_high(s);
  }

  auto table_config = dbctx_->config_directory->getTableConfig(
      snap->state.tsdb_namespace(),
      snap->state.table_key());

  MetadataOperation envelope(
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      METAOP_SPLIT_PARTITION,
      SHA1Hash(
          table_config.metadata_txnid().data(),
          table_config.metadata_txnid().size()),
      Random::singleton()->sha1(),
      *msg::encode(op));

  return dbctx_->metadata_coordinator->performAndCommitOperation(
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      envelope);
}

ReplicationState LSMPartitionWriter::fetchReplicationState() const {
  auto snap = head_->getSnapshot();
  auto repl_state = snap->state.replication_state();
  String tbl_uuid((char*) snap->uuid().data(), snap->uuid().size());

  if (repl_state.uuid() == tbl_uuid) {
    return repl_state;
  } else {
    ReplicationState state;
    state.set_uuid(tbl_uuid);
    return state;
  }
}

void LSMPartitionWriter::commitReplicationState(const ReplicationState& state) {
  ScopedLock<std::mutex> write_lk(mutex_);
  auto snap = head_->getSnapshot()->clone();
  if (state.uuid() == snap->state.replication_state().uuid()) {
    mergeReplicationState(snap->state.mutable_replication_state(), &state);
  } else {
    *snap->state.mutable_replication_state() = state;
  }
  snap->writeToDisk();
  head_->setSnapshot(snap);
}

Status LSMPartitionWriter::applyMetadataChange(
    const PartitionDiscoveryResponse& discovery_info) {
  ScopedLock<std::mutex> write_lk(mutex_);
  auto snap = head_->getSnapshot()->clone();

  logTrace(
      "evqld",
      "Applying metadata change to partition $0/$1/$2: $3",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString(),
      discovery_info.DebugString());

  if (snap->state.last_metadata_txnseq() >= discovery_info.txnseq()) {
    return Status::success();
  }

  // if we're alreading unloading this partition, fast forward any new
  // replication targets (since they can't be new joins but must be splits
  // of the previous replication targets)
  if (snap->state.lifecycle_state() == PDISCOVERY_UNLOAD) {
    fastForwardReplicationState(
        snap.get(),
        snap->state.mutable_replication_state(),
        discovery_info);
  }

  snap->state.set_last_metadata_txnid(discovery_info.txnid());
  snap->state.set_last_metadata_txnseq(discovery_info.txnseq());
  snap->state.set_lifecycle_state(discovery_info.code());
  snap->state.set_is_splitting(discovery_info.is_splitting());

  // backfill keyrange
  if (snap->state.partition_keyrange_end().size() == 0 &&
      discovery_info.keyrange_end().size() > 0) {
    snap->state.set_partition_keyrange_end(discovery_info.keyrange_end());
  }

  snap->state.mutable_split_partition_ids()->Clear();
  for (const auto& p : discovery_info.split_partition_ids()) {
    snap->state.add_split_partition_ids(p);
  }

  snap->state.set_has_joining_servers(false);
  snap->state.mutable_replication_targets()->Clear();
  for (const auto& dt : discovery_info.replication_targets()) {
    auto pt = snap->state.add_replication_targets();
    pt->set_server_id(dt.server_id());
    pt->set_placement_id(dt.placement_id());
    pt->set_partition_id(dt.partition_id());
    pt->set_keyrange_begin(dt.keyrange_begin());
    pt->set_keyrange_end(dt.keyrange_end());

    if (dt.is_joining()) {
      pt->set_is_joining(true);
      snap->state.set_has_joining_servers(true);
    }
  }

  snap->writeToDisk();
  head_->setSnapshot(snap);

  return Status::success();
}

void LSMPartitionWriter::fastForwardReplicationState(
    const PartitionSnapshot* snap,
    ReplicationState* repl_state,
    const PartitionDiscoveryResponse& discovery_info) {
  /* find old low replication watermark */
  std::map<std::string, std::vector<uint64_t>> old_repl_offsets;
  for (const auto& t : snap->state.replication_targets()) {
    auto replica_offset = replicatedOffsetFor(*repl_state, t);
    old_repl_offsets[t.partition_id()].emplace_back(replica_offset);
  }

  uint64_t low_watermark = 0;
  bool low_watermark_found = false;
  for (auto& t : old_repl_offsets) {
    uint64_t this_watermark = 0;
    if (t.second.size() > 1) {
      std::sort(t.second.begin(), t.second.end());
      this_watermark = t.second[(t.second.size() - 1) / 2];
    } else {
      this_watermark = t.second[0];
    }

    if (!low_watermark_found || this_watermark < low_watermark) {
      low_watermark = this_watermark;
      low_watermark_found = true;
    }
  }

  if (low_watermark == 0) {
    return;
  }

  /* default new replication targets to old watermark */
  logInfo(
      "evqld",
      "Fast-forwarding replication targets for partition $0/$1/$2 to $3",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString(),
      low_watermark);

  for (const auto& t : discovery_info.replication_targets()) {
    auto replica_offset = replicatedOffsetFor(*repl_state, t);
    if (replica_offset == 0) {
      setReplicatedOffsetFor(repl_state, t, low_watermark);
    }
  }
}

} // namespace tdsb
