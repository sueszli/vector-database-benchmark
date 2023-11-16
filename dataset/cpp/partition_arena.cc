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
#include <assert.h>
#include "eventql/eventql.h"
#include <eventql/db/partition_arena.h>
#include <eventql/db/tablet_index.h>
#include <eventql/util/protobuf/MessageDecoder.h>

namespace eventql {

PartitionArena::PartitionArena(
    const msg::MessageSchema& schema) :
    schema_(schema),
    num_records_(0),
    has_update_(false),
    has_skiplist_(false),
    cstable_schema_(cstable::TableSchema::fromProtobuf(schema)),
    cstable_schema_ext_(cstable_schema_),
    opened_(false) {
  cstable_schema_ext_.addBool("__lsm_is_update", false);
  cstable_schema_ext_.addBool("__lsm_skip", false);
  cstable_schema_ext_.addString("__lsm_id", false);
  cstable_schema_ext_.addUnsignedInteger("__lsm_version", false);
  cstable_schema_ext_.addUnsignedInteger("__lsm_sequence", false);
}

void PartitionArena::open() {
  cstable_file_.reset(
      new cstable::CSTableFile(
          cstable::BinaryFormatVersion::v0_2_0,
          cstable_schema_ext_));

  cstable_writer_ = cstable::CSTableWriter::openFile(cstable_file_.get());

  is_update_col_ = cstable_writer_->getColumnWriter("__lsm_is_update");
  id_col_ = cstable_writer_->getColumnWriter("__lsm_id");
  version_col_ = cstable_writer_->getColumnWriter("__lsm_version");

  opened_ = true;
}

Set<SHA1Hash> PartitionArena::insertRecords(
    const ShreddedRecordList& records,
    Vector<bool> skip_flags,
    const Vector<bool>& update_flags) {
  assert(records.getNumRecords() == skip_flags.size());
  assert(records.getNumRecords() == update_flags.size());

  ScopedLock<std::mutex> lk(mutex_);

  if (!opened_) {
    open();
  }

  Set<SHA1Hash> inserted_ids;
  size_t insert_count = 0;
  for (size_t i = 0; i < records.getNumRecords(); ++i) {
    if (skip_flags[i]) {
      continue;
    }

    const auto& record_id = records.getRecordID(i);
    auto record_version = records.getRecordVersion(i);
    assert(record_version > 1400000000000000);

    auto old = record_versions_.find(record_id);
    if (old == record_versions_.end()) {
      // record does not exist in arena
    } else if (old->second.version < record_version) {
      // record does exist in arena, but the one we're inserting is newer
      skiplist_[old->second.position] = true;
      has_skiplist_ = true;
    } else {
      // record in arena is newer or equal to the one we're inserting, skip
      skip_flags[i] = true;
      continue;
    }

    has_update_ = has_update_ || update_flags[i];

    is_update_col_->writeBoolean(0, 0, update_flags[i]);
    id_col_->writeString(0, 0, (const char*) record_id.data(), record_id.size());
    version_col_->writeUnsignedInt(0, 0, record_version);
    cstable_writer_->addRow();

    RecordVersion rversion;
    inserted_ids.insert(record_id);
    ++insert_count;
    rversion.version = record_version;
    rversion.position = num_records_++;
    record_versions_[record_id] = rversion;
    vmap_[record_id] = record_version;
    skiplist_.push_back(false);
  }

  Set<String> written_columns;
  for (size_t j = 0; j < records.getNumColumns(); ++j) {
    auto col_reader = records.getColumn(j);
    if (!cstable_writer_->hasColumn(col_reader->column_name)) {
      continue;
    }

    written_columns.insert(col_reader->column_name);
    auto col_writer = cstable_writer_->getColumnWriter(col_reader->column_name);
    auto col_maxdlvl = col_writer->maxDefinitionLevel();
    size_t cur_rec = 0;
    size_t nvals = col_reader->values.size();
    for (size_t i = 0; i < nvals; ++i) {
      const auto& v = col_reader->values[i];

      if (v.rlvl == 0 && i > 0) {
        ++cur_rec;
      }

      if (!skip_flags[cur_rec]) {
        if (v.dlvl == col_maxdlvl) {
          col_writer->writeString(v.rlvl, v.dlvl, v.value);
        } else {
          col_writer->writeNull(v.rlvl, v.dlvl);
        }
      }
    }

    assert(cur_rec + 1 == records.getNumRecords());
  }

  for (const auto& col : cstable_writer_->columns()) {
    if (written_columns.count(col.column_name) > 0) {
      continue;
    }

    auto col_writer = cstable_writer_->getColumnWriter(col.column_name);
    for (size_t i = 0; i < insert_count; ++i) {
      col_writer->writeNull(0, 0);
    }
  }

  cstable_writer_->commit();
  return inserted_ids;
}

uint64_t PartitionArena::fetchRecordVersion(const SHA1Hash& record_id) {
  ScopedLock<std::mutex> lk(mutex_);
  auto rec = record_versions_.find(record_id);
  if (rec == record_versions_.end()) {
    return 0;
  } else {
    return rec->second.version;
  }
}

size_t PartitionArena::size() const {
  ScopedLock<std::mutex> lk(mutex_);
  return num_records_;
}

Status PartitionArena::writeToDisk(
    const String& filename,
    uint64_t sequence) {
  if (!opened_) {
    return Status(eRuntimeError, "partition arena is empty");
  }

  auto sequence_col = cstable_writer_->getColumnWriter("__lsm_sequence");
  auto skip_col = cstable_writer_->getColumnWriter("__lsm_skip");
  for (size_t i = 0; i < num_records_; ++i) {
    sequence_col->writeUnsignedInt(0, 0, sequence++);
    skip_col->writeBoolean(0, 0, skiplist_[i]);
  }

  cstable_writer_->commit();
  auto file = File::openFile(filename + ".cst", File::O_WRITE | File::O_CREATE);
  cstable_file_->writeFile(file.fd());

  LSMTableIndex::write(vmap_, filename + ".idx");
  return Status::success();
}

cstable::CSTableFile* PartitionArena::getCSTableFile() const {
  return cstable_file_.get();
}

PartitionArena::SkiplistReader PartitionArena::getSkiplistReader() const {
  ScopedLock<std::mutex> lk(mutex_);
  SkiplistReader reader;
  reader.position = 0;
  reader.skiplist = skiplist_;
  return reader;
}

bool PartitionArena::SkiplistReader::readNext() {
  return skiplist[position++];
}

size_t PartitionArena::SkiplistReader::size() const {
  return skiplist.size();
}

bool PartitionArena::hasUpdate() const {
  return has_update_;
}

bool PartitionArena::hasSkiplist() const {
  return has_skiplist_;
}

} // namespace eventql

