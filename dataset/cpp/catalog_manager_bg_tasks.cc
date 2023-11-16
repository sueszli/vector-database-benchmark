// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// The following only applies to changes made to this file as part of YugaByte development.
//
// Portions Copyright (c) YugaByte, Inc.
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

#include "yb/master/catalog_manager_bg_tasks.h"

#include <memory>

#include "yb/gutil/casts.h"

#include "yb/master/cluster_balance.h"
#include "yb/master/master.h"
#include "yb/master/ts_descriptor.h"
#include "yb/master/tablet_split_manager.h"
#include "yb/master/ysql_backends_manager.h"

#include "yb/util/debug-util.h"
#include "yb/util/flags.h"
#include "yb/util/monotime.h"
#include "yb/util/mutex.h"
#include "yb/util/status_log.h"
#include "yb/util/thread.h"

using std::shared_ptr;
using std::vector;

METRIC_DEFINE_event_stats(
    server, load_balancer_duration, "Load balancer duration",
    yb::MetricUnit::kMilliseconds, "Duration of one load balancer run (in milliseconds)");

DEFINE_RUNTIME_int32(catalog_manager_bg_task_wait_ms, 1000,
    "Amount of time the catalog manager background task thread waits between runs");

DEFINE_RUNTIME_int32(load_balancer_initial_delay_secs, yb::master::kDelayAfterFailoverSecs,
             "Amount of time to wait between becoming master leader and enabling the load "
             "balancer.");

DEFINE_RUNTIME_bool(sys_catalog_respect_affinity_task, true,
            "Whether the master sys catalog tablet respects cluster config preferred zones "
            "and sends step down requests to a preferred leader.");

DEFINE_RUNTIME_bool(ysql_enable_auto_analyze_service, false,
                    "Enable the Auto Analyze service which automatically triggers ANALYZE to "
                    "update table statistics for tables which have changed more than a "
                    "configurable threshold.");
TAG_FLAG(ysql_enable_auto_analyze_service, experimental);

DEFINE_test_flag(bool, pause_catalog_manager_bg_loop_start, false,
                 "Pause the bg tasks thread at the beginning of the loop.");

DEFINE_test_flag(bool, pause_catalog_manager_bg_loop_end, false,
                 "Pause the bg tasks thread at the end of the loop.");

DECLARE_bool(enable_ysql);
DECLARE_bool(TEST_echo_service_enabled);

namespace yb {
namespace master {

typedef std::unordered_map<TableId, std::list<CDCStreamInfoPtr>> TableStreamIdsMap;

CatalogManagerBgTasks::CatalogManagerBgTasks(CatalogManager *catalog_manager)
    : closing_(false),
      pending_updates_(false),
      cond_(&lock_),
      thread_(nullptr),
      catalog_manager_(catalog_manager),
      load_balancer_duration_(METRIC_load_balancer_duration.Instantiate(
          catalog_manager->master_->metric_entity())) {
}

void CatalogManagerBgTasks::Wake() {
  MutexLock lock(lock_);
  pending_updates_ = true;
  cond_.Broadcast();
}

void CatalogManagerBgTasks::Wait(int msec) {
  MutexLock lock(lock_);
  if (closing_.load()) return;
  if (!pending_updates_) {
    cond_.TimedWait(MonoDelta::FromMilliseconds(msec));
  }
  pending_updates_ = false;
}

void CatalogManagerBgTasks::WakeIfHasPendingUpdates() {
  MutexLock lock(lock_);
  if (pending_updates_) {
    cond_.Broadcast();
  }
}

Status CatalogManagerBgTasks::Init() {
  RETURN_NOT_OK(yb::Thread::Create("catalog manager", "bgtasks",
      &CatalogManagerBgTasks::Run, this, &thread_));
  return Status::OK();
}

void CatalogManagerBgTasks::Shutdown() {
  {
    bool closing_expected = false;
    if (!closing_.compare_exchange_strong(closing_expected, true)) {
      VLOG(2) << "CatalogManagerBgTasks already shut down";
      return;
    }
  }

  Wake();
  if (thread_ != nullptr) {
    CHECK_OK(ThreadJoiner(thread_.get()).Join());
  }
}

void CatalogManagerBgTasks::TryResumeBackfillForTables(
    const LeaderEpoch& epoch, std::unordered_set<TableId>* tables) {
  for (auto it = tables->begin(); it != tables->end(); it = tables->erase(it)) {
    const auto& table_info_result = catalog_manager_->FindTableById(*it);
    if (!table_info_result.ok()) {
      LOG(WARNING) << "Table Info not found for id " << *it;
      continue;
    }
    const auto& table_info = *table_info_result;
    // Get schema version.
    uint32_t version = table_info->LockForRead()->pb.version();
    const auto& tablets = table_info->GetTablets();
    for (const auto& tablet : tablets) {
      LOG(INFO) << "PITR: Try resuming backfill for tablet " << tablet->id()
                << ". If it is not a table for which backfill needs to be resumed"
                << " then this is a NO-OP";
      auto s = catalog_manager_->HandleTabletSchemaVersionReport(
          tablet.get(), version, epoch, table_info);
      // If schema version changed since PITR restore then backfill should restart
      // by virtue of that particular alter if needed.
      WARN_NOT_OK(s, Format("PITR: Resume backfill failed for tablet ", tablet->id()));
    }
  }
}

void CatalogManagerBgTasks::Run() {
  while (!closing_.load()) {
    TEST_PAUSE_IF_FLAG(TEST_pause_catalog_manager_bg_loop_start);
    // Perform assignment processing.
    SCOPED_LEADER_SHARED_LOCK(l, catalog_manager_);
    if (!l.catalog_status().ok()) {
      LOG(WARNING) << "Catalog manager background task thread going to sleep: "
                   << l.catalog_status().ToString();
    } else if (l.leader_status().ok()) {
      // Clear metrics for dead tservers.
      vector<shared_ptr<TSDescriptor>> descs;
      const auto& ts_manager = catalog_manager_->master_->ts_manager();
      ts_manager->GetAllDescriptors(&descs);
      for (auto& ts_desc : descs) {
        if (!ts_desc->IsLive()) {
          ts_desc->ClearMetrics();
        }
      }

      if (FLAGS_TEST_echo_service_enabled) {
        WARN_NOT_OK(
            catalog_manager_->CreateTestEchoService(l.epoch()),
            "Failed to create Test Echo service");
      }

      // TODO(auto-analyze, #19464): we allow enabling this service at runtime. We should also allow
      // disabling this service at runtime i.e., the service should stop on the tserver hosting it
      // when the flag is set to false.
      if (GetAtomicFlag(&FLAGS_ysql_enable_auto_analyze_service)) {
        WARN_NOT_OK(catalog_manager_->CreatePgAutoAnalyzeService(l.epoch()),
                    "Failed to create Auto Analyze service");
      }

      // Report metrics.
      catalog_manager_->ReportMetrics();

      // Cleanup old tasks from tracker.
      catalog_manager_->tasks_tracker_->CleanupOldTasks();

      TabletInfos to_delete;
      TableToTabletInfos to_process;

      // Get list of tablets not yet running or already replaced.
      catalog_manager_->ExtractTabletsToProcess(&to_delete, &to_process);

      bool processed_tablets = false;
      if (!to_process.empty()) {
        // For those tablets which need to be created in this round, assign replicas.
        TSDescriptorVector ts_descs = catalog_manager_->GetAllLiveNotBlacklistedTServers();
        CMGlobalLoadState global_load_state;
        catalog_manager_->InitializeGlobalLoadState(ts_descs, &global_load_state);
        // Transition tablet assignment state from preparing to creating, send
        // and schedule creation / deletion RPC messages, etc.
        // This is done table by table.
        for (const auto& entries : to_process) {
          LOG(INFO) << "Processing pending assignments for table: " << entries.first;
          Status s = catalog_manager_->ProcessPendingAssignmentsPerTable(
              entries.first, entries.second, l.epoch(), &global_load_state);
          WARN_NOT_OK(s, "Assignment failed");
          // Set processed_tablets as true if the call succeeds for at least one table.
          processed_tablets = processed_tablets || s.ok();
          // TODO Add tests for this in the revision that makes
          // create/alter fault tolerant.
        }
      }

      // Trigger pending backfills.
      std::unordered_set<TableId> table_map;
      {
        std::lock_guard lock(catalog_manager_->backfill_mutex_);
        table_map.swap(catalog_manager_->pending_backfill_tables_);
      }
      TryResumeBackfillForTables(l.epoch(), &table_map);

      // Do the LB enabling check
      if (!processed_tablets) {
        if (catalog_manager_->TimeSinceElectedLeader() >
            MonoDelta::FromSeconds(FLAGS_load_balancer_initial_delay_secs)) {
          auto start = CoarseMonoClock::Now();
          catalog_manager_->load_balance_policy_->RunLoadBalancer(l.epoch());
          load_balancer_duration_->Increment(ToMilliseconds(CoarseMonoClock::now() - start));
        }
      }

      std::vector<scoped_refptr<TableInfo>> tables;
      TabletInfoMap tablet_info_map;
      {
        CatalogManager::SharedLock lock(catalog_manager_->mutex_);
        auto tables_it = catalog_manager_->tables_->GetPrimaryTables();
        tables = std::vector(std::begin(tables_it), std::end(tables_it));
        tablet_info_map = *catalog_manager_->tablet_map_;
      }
      catalog_manager_->tablet_split_manager()->MaybeDoSplitting(
          tables, tablet_info_map, l.epoch());

      if (!to_delete.empty() || catalog_manager_->AreTablesDeleting()) {
        catalog_manager_->CleanUpDeletedTables(l.epoch());
      }

      {
        // Find if there have been any new tables added to any namespace with an active cdcsdk
        // stream.
        TableStreamIdsMap table_unprocessed_streams_map;
        // In case of master leader restart of leadership changes, we will scan all streams for
        // unprocessed tables, but from the second iteration onwards we will only consider the
        // 'cdcsdk_unprocessed_tables' field of CDCStreamInfo object stored in the cdc_state_map.
        Status s =
            catalog_manager_->FindCDCSDKStreamsForAddedTables(&table_unprocessed_streams_map);

        if (s.ok() && !table_unprocessed_streams_map.empty()) {
          s = catalog_manager_->ProcessNewTablesForCDCSDKStreams(
              table_unprocessed_streams_map, l.epoch());
        }
        if (!s.ok()) {
          YB_LOG_EVERY_N(WARNING, 10)
              << "Encountered failure while trying to add unprocessed tables to cdc_state table: "
              << s.ToString();
        }
      }

      // Ensure the master sys catalog tablet follows the cluster's affinity specification.
      if (FLAGS_sys_catalog_respect_affinity_task) {
        Status s = catalog_manager_->SysCatalogRespectLeaderAffinity();
        if (!s.ok()) {
          YB_LOG_EVERY_N(INFO, 10) << s.message().ToBuffer();
        }
      }

      if (FLAGS_enable_ysql) {
        // Start the tablespace background task.
        catalog_manager_->StartTablespaceBgTaskIfStopped();

        // Start the pg catalog versions background task.
        catalog_manager_->StartPgCatalogVersionsBgTaskIfStopped();
      }

      // Restart CDCSDK parent tablet deletion bg task.
      catalog_manager_->StartCDCParentTabletDeletionTaskIfStopped();

      // Run background tasks related to XCluster & CDC Schema.
      WARN_NOT_OK(
          catalog_manager_->RunXClusterBgTasks(l.epoch()), "Failed XCluster Background Task");

      // Abort inactive YSQL BackendsCatalogVersionJob jobs.
      catalog_manager_->master_->ysql_backends_manager()->AbortInactiveJobs();

      // Set the universe_uuid field in the cluster config if not already set.
      WARN_NOT_OK(catalog_manager_->SetUniverseUuidIfNeeded(l.epoch()),
                  "Failed SetUniverseUuidIfNeeded Task");

      was_leader_ = true;
    } else {
      // leader_status is not ok.
      if (was_leader_) {
        LOG(INFO) << "Begin one-time cleanup on losing leadership";
        load_balancer_duration_->Reset();
        catalog_manager_->ResetMetrics();
        catalog_manager_->ResetTasksTrackers();
        catalog_manager_->master_->ysql_backends_manager()->AbortAllJobs();
        was_leader_ = false;
      }
    }
    // Wait for a notification or a timeout expiration.
    //  - CreateTable will call Wake() to notify about the tablets to add
    //  - HandleReportedTablet/ProcessPendingAssignments will call WakeIfHasPendingUpdates()
    //    to notify about tablets creation.
    //  - DeleteTable will call Wake() to finish destructing any table internals
    l.Unlock();
    TEST_PAUSE_IF_FLAG(TEST_pause_catalog_manager_bg_loop_end);
    Wait(FLAGS_catalog_manager_bg_task_wait_ms);
  }
  VLOG(1) << "Catalog manager background task thread shutting down";
}

}  // namespace master
}  // namespace yb
