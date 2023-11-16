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
#include "eventql/config/config_directory_standalone.h"
#include "eventql/db/database.h"
#include "eventql/db/monitor.h"

namespace eventql {

StandaloneConfigDirectory::StandaloneConfigDirectory(
    const String& datadir,
    const String& listen_addr,
    DatabaseContext* dbctx /* = nullptr */) :
    listen_addr_(listen_addr),
    dbctx_(dbctx) {
  cluster_config_.set_replication_factor(1);

  mdb::MDBOptions opts;
  opts.data_filename = "standalone.db";
  opts.lock_filename = "standalone.db.lck";
  db_ = mdb::MDB::open(datadir, opts);
}

Status StandaloneConfigDirectory::start(bool create /* = false */) {
  if (db_.get()) {
    auto txn = db_->startTransaction(false);
    auto cursor = txn->getCursor();

    for (int i = 0; ; ++i) {
      Buffer key;
      Buffer value;
      if (i == 0) {
        if (!cursor->getFirst(&key, &value)) {
          break;
        }
      } else {
        if (!cursor->getNext(&key, &value)) {
          break;
        }
      }

      auto db_key = key.toString();

      if (StringUtil::beginsWith(db_key, "db~")) {
        auto cfg = msg::decode<NamespaceConfig>(value);
        namespaces_.emplace(cfg.customer(), cfg);
      }

      if (StringUtil::beginsWith(db_key, "tbl~")) {
        auto cfg = msg::decode<TableDefinition>(value);
        tables_.emplace(
            StringUtil::format("$0~$1", cfg.customer(), cfg.table_name()),
            cfg);
      }
    }

    cursor->close();
    txn->abort();
  }

  return Status::success();
}

void StandaloneConfigDirectory::stop() {}

ClusterConfig StandaloneConfigDirectory::getClusterConfig() const {
  std::unique_lock<std::mutex> lk(mutex_);
  return cluster_config_;
}

void StandaloneConfigDirectory::updateClusterConfig(ClusterConfig config) {
  std::unique_lock<std::mutex> lk(mutex_);
  cluster_config_ = config;
  auto callbacks = on_cluster_change_;
  lk.unlock();
  for (const auto& cb : callbacks) {
    cb(config);
  }
}

void StandaloneConfigDirectory::setClusterConfigChangeCallback(
    Function<void (const ClusterConfig& cfg)> fn) {
  std::unique_lock<std::mutex> lk(mutex_);
  on_cluster_change_.emplace_back(fn);
}

String StandaloneConfigDirectory::getServerID() const {
  return "localhost";
}

bool StandaloneConfigDirectory::electLeader() {
  return true;
}

String StandaloneConfigDirectory::getLeader() const {
  return "localhost";
}

ServerConfig StandaloneConfigDirectory::getServerConfig(
    const String& server_name) const {
  if (server_name != "localhost") {
    RAISEF(kNotFoundError, "server not found: $0", server_name);
  }

  ServerConfig c;
  c.set_server_id("localhost");
  c.set_server_addr(listen_addr_);
  c.set_server_status(SERVER_UP);

  auto ss = c.mutable_server_stats();
  double load_factor = 0.0f;
  if (dbctx_ && dbctx_->monitor) {
    auto m = dbctx_->monitor;
    ss->set_load_factor(m->getLoadFactor());
    ss->set_disk_used(m->getDiskUsed());
    ss->set_disk_available(m->getDiskAvailable());
    ss->set_partitions_loaded(m->getPartitionsLoaded());
    ss->set_partitions_assigned(m->getPartitionsAssigned());
  } else {
    ss->set_load_factor(0.0f);
  }

  return c;
}

Vector<ServerConfig> StandaloneConfigDirectory::listServers() const {
  Vector<ServerConfig> servers;
  servers.emplace_back(getServerConfig("localhost"));
  return servers;
}

void StandaloneConfigDirectory::setServerConfigChangeCallback(
    Function<void (const ServerConfig& cfg)> fn) {
  // void
}

void StandaloneConfigDirectory::updateServerConfig(ServerConfig cfg) {
  // void
}

RefPtr<NamespaceConfigRef> StandaloneConfigDirectory::getNamespaceConfig(
    const String& customer_key) const {
  std::unique_lock<std::mutex> lk(mutex_);
  auto iter = namespaces_.find(customer_key);
  if (iter == namespaces_.end()) {
    RAISEF(kNotFoundError, "namespace not found: $0", customer_key);
  }

  return mkRef(new NamespaceConfigRef(iter->second));
}

void StandaloneConfigDirectory::listNamespaces(
    Function<void (const NamespaceConfig& cfg)> fn) const {
  std::unique_lock<std::mutex> lk(mutex_);
  for (const auto& ns : namespaces_) {
    fn(ns.second);
  }
}

void StandaloneConfigDirectory::setNamespaceConfigChangeCallback(
    Function<void (const NamespaceConfig& cfg)> fn) {
  std::unique_lock<std::mutex> lk(mutex_);
  on_namespace_change_.emplace_back(fn);
}

void StandaloneConfigDirectory::updateNamespaceConfig(NamespaceConfig cfg) {
  std::unique_lock<std::mutex> lk(mutex_);
  namespaces_.emplace(cfg.customer(), cfg);
  auto callbacks = on_namespace_change_;

  if (db_.get()) {
    auto db_key = StringUtil::format("db~$0", cfg.customer());
    auto db_val = *msg::encode(cfg);
    auto txn = db_->startTransaction(false);
    txn->update(
        db_key.data(),
        db_key.size(),
        db_val.data(),
        db_val.size());

    txn->commit();
  }

  lk.unlock();
  for (const auto& cb : callbacks) {
    cb(cfg);
  }
}

TableDefinition StandaloneConfigDirectory::getTableConfig(
    const String& db_namespace,
    const String& table_name,
    bool allow_cache /* = true */) {
  std::unique_lock<std::mutex> lk(mutex_);
  auto iter = tables_.find(db_namespace + "~" + table_name);
  if (iter == tables_.end()) {
    RAISEF(kNotFoundError, "table not found: $0", table_name);
  }

  return iter->second;
}

void StandaloneConfigDirectory::updateTableConfig(
    const TableDefinition& table,
    bool force /* = false */) {
  std::unique_lock<std::mutex> lk(mutex_);
  auto table_id = table.customer() + "~" + table.table_name();
  auto old_table_version = 0;
  auto old_table = tables_.find(table_id);
  if (old_table != tables_.end()) {
    old_table_version = old_table->second.version();
  }

  if (table.version() != old_table_version) {
    RAISEF(
        kIllegalArgumentError,
        "conflicting update to table config: $0",
        table.table_name());
  }

  auto new_table = table;
  new_table.set_version(old_table_version + 1);

  tables_[table_id] = new_table;

  if (db_.get()) {
    auto db_key = StringUtil::format(
        "tbl~$0~$1",
        new_table.customer(),
        new_table.table_name());

    auto db_val = *msg::encode(new_table);
    auto txn = db_->startTransaction(false);
    txn->update(
        db_key.data(),
        db_key.size(),
        db_val.data(),
        db_val.size());

    txn->commit();
  }

  auto callbacks = on_table_change_;
  lk.unlock();
  for (const auto& cb : callbacks) {
    cb(new_table);
  }
}

void StandaloneConfigDirectory::listTables(
    Function<void (const TableDefinition& table)> fn) const {
  std::unique_lock<std::mutex> lk(mutex_);
  for (const auto& tbl : tables_) {
    fn(tbl.second);
  }
}

void StandaloneConfigDirectory::setTableConfigChangeCallback(
    Function<void (const TableDefinition& tbl)> fn) {
  std::unique_lock<std::mutex> lk(mutex_);
  on_table_change_.emplace_back(fn);
}

} // namespace eventql

