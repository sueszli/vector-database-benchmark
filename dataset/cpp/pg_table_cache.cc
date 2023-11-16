// Copyright (c) YugaByte, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.
//

#include "yb/tserver/pg_table_cache.h"

#include <unordered_map>

#include "yb/client/client.h"
#include "yb/client/table.h"

#include "yb/gutil/thread_annotations.h"

#include "yb/tserver/pg_client.pb.h"

#include "yb/util/scope_exit.h"

namespace yb {
namespace tserver {

namespace {

struct CacheEntry {
  std::promise<Result<client::YBTablePtr>> promise;
  std::shared_future<Result<client::YBTablePtr>> future;
  master::GetTableSchemaResponsePB schema;

  CacheEntry() : future(promise.get_future()) {
  }
};

} // namespace

class PgTableCache::Impl {
 public:
  explicit Impl(std::shared_future<client::YBClient*> client_future)
      : client_future_(client_future) {}

  Status GetInfo(
      const TableId& table_id,
      client::YBTablePtr* table,
      master::GetTableSchemaResponsePB* schema) {
    auto entry = GetEntry(table_id);
    const auto& table_result = entry->future.get();
    RETURN_NOT_OK(table_result);
    *table = *table_result;
    *schema = entry->schema;
    return Status::OK();
  }

  Result<client::YBTablePtr> Get(const TableId& table_id) {
    return GetEntry(table_id)->future.get();
  }

  void Invalidate(const TableId& table_id) {
    std::lock_guard lock(mutex_);
    cache_.erase(table_id);
  }

  void InvalidateAll(CoarseTimePoint invalidation_time) {
    std::lock_guard lock(mutex_);
    if (last_cache_invalidation_ > invalidation_time) {
      return;
    }
    last_cache_invalidation_ = CoarseMonoClock::now();
    cache_.clear();
  }

 private:
  client::YBClient& client() {
    return *client_future_.get();
  }

  std::shared_ptr<CacheEntry> GetEntry(const TableId& table_id) {
    auto p = DoGetEntry(table_id);
    if (p.second) {
      LoadEntry(table_id, p.first.get());
    }
    return p.first;
  }

  std::pair<std::shared_ptr<CacheEntry>, bool> DoGetEntry(const TableId& table_id) {
    std::lock_guard lock(mutex_);
    auto it = cache_.find(table_id);
    if (it != cache_.end()) {
      return std::make_pair(it->second, false);
    }
    it = cache_.emplace(table_id, std::make_shared<CacheEntry>()).first;
    return std::make_pair(it->second, true);
  }

  Status OpenTable(
      const TableId& table_id,
      client::YBTablePtr* table,
      master::GetTableSchemaResponsePB* schema) {
    RETURN_NOT_OK(client().OpenTable(table_id, table, schema));
    RSTATUS_DCHECK(
        (**table).table_type() == client::YBTableType::PGSQL_TABLE_TYPE, RuntimeError,
        "Wrong table type");
    return Status::OK();
  }

  void LoadEntry(const TableId& table_id, CacheEntry* entry) {
    client::YBTablePtr table;
    bool finished = false;
    auto se = ScopeExit([entry, &finished] {
      if (finished) {
        return;
      }
      entry->promise.set_value(STATUS(InternalError, "Unexpected return"));
    });
    auto status = OpenTable(table_id, &table, &entry->schema);
    if (!status.ok()) {
      Invalidate(table_id);
      entry->promise.set_value(std::move(status));
      finished = true;
      return;
    }

    entry->promise.set_value(table);
    finished = true;
  }

  std::shared_future<client::YBClient*> client_future_;
  std::mutex mutex_;
  std::unordered_map<TableId, std::shared_ptr<CacheEntry>> cache_ GUARDED_BY(mutex_);
  CoarseTimePoint last_cache_invalidation_ GUARDED_BY(mutex_);
};

PgTableCache::PgTableCache(std::shared_future<client::YBClient*> client_future)
    : impl_(new Impl(std::move(client_future))) {
}

PgTableCache::~PgTableCache() {
}

Status PgTableCache::GetInfo(
    const TableId& table_id,
    client::YBTablePtr* table,
    master::GetTableSchemaResponsePB* schema) {
  return impl_->GetInfo(table_id, table, schema);
}

Result<client::YBTablePtr> PgTableCache::Get(const TableId& table_id) {
  return impl_->Get(table_id);
}

void PgTableCache::Invalidate(const TableId& table_id) {
  impl_->Invalidate(table_id);
}

void PgTableCache::InvalidateAll(CoarseTimePoint invalidation_time) {
  impl_->InvalidateAll(invalidation_time);
}

}  // namespace tserver
}  // namespace yb
