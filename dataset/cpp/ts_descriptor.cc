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
//

#include "yb/master/ts_descriptor.h"

#include <vector>

#include "yb/common/common.pb.h"
#include "yb/common/wire_protocol.h"
#include "yb/common/wire_protocol.pb.h"

#include "yb/master/master_fwd.h"
#include "yb/master/master_util.h"
#include "yb/master/catalog_manager_util.h"
#include "yb/master/master_cluster.pb.h"
#include "yb/master/master_heartbeat.pb.h"

#include "yb/util/atomic.h"
#include "yb/util/flags.h"
#include "yb/util/status_format.h"

DEFINE_UNKNOWN_int32(tserver_unresponsive_timeout_ms, 60 * 1000,
             "The period of time that a Master can go without receiving a heartbeat from a "
             "tablet server before considering it unresponsive. Unresponsive servers are not "
             "selected when assigning replicas during table creation or re-replication.");
TAG_FLAG(tserver_unresponsive_timeout_ms, advanced);


namespace yb {
namespace master {

Result<TSDescriptorPtr> TSDescriptor::RegisterNew(
    const NodeInstancePB& instance,
    const TSRegistrationPB& registration,
    CloudInfoPB local_cloud_info,
    rpc::ProxyCache* proxy_cache,
    RegisteredThroughHeartbeat registered_through_heartbeat) {
  auto result = std::make_shared<TSDescriptor>(
      instance.permanent_uuid(), registered_through_heartbeat);
  RETURN_NOT_OK(result->Register(instance, registration, std::move(local_cloud_info), proxy_cache));
  return std::move(result);
}

TSDescriptor::TSDescriptor(std::string perm_id,
                           RegisteredThroughHeartbeat registered_through_heartbeat)
    : permanent_uuid_(std::move(perm_id)),
      has_tablet_report_(false),
      has_faulty_drive_(false),
      recent_replica_creations_(0),
      last_replica_creations_decay_(MonoTime::Now()),
      num_live_replicas_(0),
      registered_through_heartbeat_(registered_through_heartbeat) {
  if (registered_through_heartbeat_) {
    last_heartbeat_ = MonoTime::Now();
  }
}

Status TSDescriptor::Register(const NodeInstancePB& instance,
                              const TSRegistrationPB& registration,
                              CloudInfoPB local_cloud_info,
                              rpc::ProxyCache* proxy_cache) {
  std::lock_guard l(lock_);
  return RegisterUnlocked(instance, registration, std::move(local_cloud_info), proxy_cache);
}

Status TSDescriptor::RegisterUnlocked(
    const NodeInstancePB& instance,
    const TSRegistrationPB& registration,
    CloudInfoPB local_cloud_info,
    rpc::ProxyCache* proxy_cache) {
  CHECK_EQ(instance.permanent_uuid(), permanent_uuid_);

  int64_t latest_seqno = ts_information_
      ? ts_information_->tserver_instance().instance_seqno()
      : -1;
  if (instance.instance_seqno() < latest_seqno) {
    return STATUS(AlreadyPresent,
      strings::Substitute("Cannot register with sequence number $0:"
                          " Already have a registration from sequence number $1",
                          instance.instance_seqno(),
                          latest_seqno));
  } else if (instance.instance_seqno() == latest_seqno) {
    // It's possible that the TS registered, but our response back to it
    // got lost, so it's trying to register again with the same sequence
    // number. That's fine.
    LOG(INFO) << "Processing retry of TS registration from " << instance.ShortDebugString();
  }

  latest_seqno = instance.instance_seqno();
  // After re-registering, make the TS re-report its tablets.
  has_tablet_report_ = false;

  ts_information_ = std::make_shared<TSInformationPB>();
  ts_information_->mutable_registration()->CopyFrom(registration);
  ts_information_->mutable_tserver_instance()->set_permanent_uuid(permanent_uuid_);
  ts_information_->mutable_tserver_instance()->set_instance_seqno(latest_seqno);

  placement_id_ = generate_placement_id(registration.common().cloud_info());

  proxies_.reset();

  placement_uuid_ = "";
  if (registration.common().has_placement_uuid()) {
    placement_uuid_ = registration.common().placement_uuid();
  }
  local_cloud_info_ = std::move(local_cloud_info);
  proxy_cache_ = proxy_cache;

  capabilities_.clear();
  capabilities_.insert(registration.capabilities().begin(), registration.capabilities().end());

  return Status::OK();
}

std::string TSDescriptor::placement_uuid() const {
  SharedLock<decltype(lock_)> l(lock_);
  return placement_uuid_;
}

std::string TSDescriptor::generate_placement_id(const CloudInfoPB& ci) {
  return strings::Substitute(
      "$0:$1:$2", ci.placement_cloud(), ci.placement_region(), ci.placement_zone());
}

std::string TSDescriptor::placement_id() const {
  SharedLock<decltype(lock_)> l(lock_);
  return placement_id_;
}

void TSDescriptor::UpdateHeartbeat(const TSHeartbeatRequestPB* req) {
  DCHECK_GE(req->num_live_tablets(), 0);
  DCHECK_GE(req->leader_count(), 0);
  {
    std::lock_guard l(lock_);
    last_heartbeat_ = MonoTime::Now();
    num_live_replicas_ = req->num_live_tablets();
    leader_count_ = req->leader_count();
    physical_time_ = req->ts_physical_time();
    hybrid_time_ = HybridTime::FromPB(req->ts_hybrid_time());
    heartbeat_rtt_ = MonoDelta::FromMicroseconds(req->rtt_us());
    if (req->has_faulty_drive()) {
      has_faulty_drive_ = req->faulty_drive();
    }
  }
}

MonoDelta TSDescriptor::TimeSinceHeartbeat() const {
  auto last_heartbeat = LastHeartbeatTime();
  return MonoTime::Now().GetDeltaSince(last_heartbeat ? last_heartbeat : MonoTime::kMin);
}

MonoTime TSDescriptor::LastHeartbeatTime() const {
  SharedLock<decltype(lock_)> l(lock_);
  return last_heartbeat_;
}

int64_t TSDescriptor::latest_seqno() const {
  SharedLock<decltype(lock_)> l(lock_);
  return ts_information_->tserver_instance().instance_seqno();
}

bool TSDescriptor::has_tablet_report() const {
  SharedLock<decltype(lock_)> l(lock_);
  return has_tablet_report_;
}

void TSDescriptor::set_has_tablet_report(bool has_report) {
  std::lock_guard l(lock_);
  has_tablet_report_ = has_report;
}

bool TSDescriptor::has_faulty_drive() const {
  SharedLock<decltype(lock_)> l(lock_);
  return has_faulty_drive_;
}

bool TSDescriptor::registered_through_heartbeat() const {
  return registered_through_heartbeat_;
}

void TSDescriptor::DecayRecentReplicaCreationsUnlocked() {
  // In most cases, we won't have any recent replica creations, so
  // we don't need to bother calling the clock, etc.
  if (recent_replica_creations_ == 0) return;

  const double kHalflifeSecs = 60;
  MonoTime now = MonoTime::Now();
  double secs_since_last_decay = now.GetDeltaSince(last_replica_creations_decay_).ToSeconds();
  recent_replica_creations_ *= pow(0.5, secs_since_last_decay / kHalflifeSecs);

  // If sufficiently small, reset down to 0 to take advantage of the fast path above.
  if (recent_replica_creations_ < 1e-12) {
    recent_replica_creations_ = 0;
  }
  last_replica_creations_decay_ = now;
}

void TSDescriptor::IncrementRecentReplicaCreations() {
  std::lock_guard l(lock_);
  DecayRecentReplicaCreationsUnlocked();
  recent_replica_creations_ += 1;
}

double TSDescriptor::RecentReplicaCreations() {
  std::lock_guard l(lock_);
  DecayRecentReplicaCreationsUnlocked();
  return recent_replica_creations_;
}

TSRegistrationPB TSDescriptor::GetRegistration() const {
  SharedLock<decltype(lock_)> l(lock_);
  return ts_information_->registration();
}

const std::shared_ptr<TSInformationPB> TSDescriptor::GetTSInformationPB() const {
  SharedLock<decltype(lock_)> l(lock_);
  CHECK(ts_information_) << "No stored information";
  return ts_information_;
}

bool TSDescriptor::MatchesCloudInfo(const CloudInfoPB& cloud_info) const {
  SharedLock<decltype(lock_)> l(lock_);
  const auto& ts_ci = ts_information_->registration().common().cloud_info();

  // cloud_info should be a prefix of ts_ci.
  return CatalogManagerUtil::IsCloudInfoPrefix(cloud_info, ts_ci);
}

CloudInfoPB TSDescriptor::GetCloudInfo() const {
  SharedLock<decltype(lock_)> l(lock_);
  return ts_information_->registration().common().cloud_info();
}

bool TSDescriptor::IsBlacklisted(const BlacklistSet& blacklist) const {
  TSRegistrationPB reg = GetRegistration();
  return yb::master::IsBlacklisted(reg.common(), blacklist);
}

bool TSDescriptor::IsRunningOn(const HostPortPB& hp) const {
  TSRegistrationPB reg = GetRegistration();
  return yb::master::IsRunningOn(reg.common(), hp);
}

Result<HostPort> TSDescriptor::GetHostPortUnlocked() const {
  const auto& addr = DesiredHostPort(ts_information_->registration().common(), local_cloud_info_);
  if (addr.host().empty()) {
    return STATUS_FORMAT(NetworkError, "Unable to find the TS address for $0: $1",
                         permanent_uuid_, ts_information_->registration().ShortDebugString());
  }

  return HostPortFromPB(addr);
}

bool TSDescriptor::IsAcceptingLeaderLoad(const ReplicationInfoPB& replication_info) const {
  if (IsReadOnlyTS(replication_info)) {
    // Read-only ts are not voting and therefore cannot be leaders.
    return false;
  }

  if (replication_info.affinitized_leaders_size() == 0 &&
      replication_info.multi_affinitized_leaders_size() == 0) {
    // If there are no affinitized leaders, all ts can be leaders.
    return true;
  }

  for (const auto& zone_set : replication_info.multi_affinitized_leaders()) {
    for (const CloudInfoPB& cloud_info : zone_set.zones()) {
      if (MatchesCloudInfo(cloud_info)) {
        return true;
      }
    }
  }

  // Handle old un-updated config if any
  for (const CloudInfoPB& cloud_info : replication_info.affinitized_leaders()) {
    if (MatchesCloudInfo(cloud_info)) {
      return true;
    }
  }
  return false;
}

void TSDescriptor::UpdateMetrics(const TServerMetricsPB& metrics) {
  std::lock_guard l(lock_);
  ts_metrics_.total_memory_usage = metrics.total_ram_usage();
  ts_metrics_.total_sst_file_size = metrics.total_sst_file_size();
  ts_metrics_.uncompressed_sst_file_size = metrics.uncompressed_sst_file_size();
  ts_metrics_.num_sst_files = metrics.num_sst_files();
  ts_metrics_.read_ops_per_sec = metrics.read_ops_per_sec();
  ts_metrics_.write_ops_per_sec = metrics.write_ops_per_sec();
  ts_metrics_.uptime_seconds = metrics.uptime_seconds();
  ts_metrics_.path_metrics.clear();
  for (const auto& path_metric : metrics.path_metrics()) {
    ts_metrics_.path_metrics[path_metric.path_id()] =
        { path_metric.used_space(), path_metric.total_space() };
  }
  ts_metrics_.disable_tablet_split_if_default_ttl = metrics.disable_tablet_split_if_default_ttl();
}

void TSDescriptor::GetMetrics(TServerMetricsPB* metrics) {
  CHECK(metrics);
  SharedLock<decltype(lock_)> l(lock_);
  metrics->set_total_ram_usage(ts_metrics_.total_memory_usage);
  metrics->set_total_sst_file_size(ts_metrics_.total_sst_file_size);
  metrics->set_uncompressed_sst_file_size(ts_metrics_.uncompressed_sst_file_size);
  metrics->set_num_sst_files(ts_metrics_.num_sst_files);
  metrics->set_read_ops_per_sec(ts_metrics_.read_ops_per_sec);
  metrics->set_write_ops_per_sec(ts_metrics_.write_ops_per_sec);
  metrics->set_uptime_seconds(ts_metrics_.uptime_seconds);
  for (const auto& path_metric : ts_metrics_.path_metrics) {
    auto* new_path_metric = metrics->add_path_metrics();
    new_path_metric->set_path_id(path_metric.first);
    new_path_metric->set_used_space(path_metric.second.used_space);
    new_path_metric->set_total_space(path_metric.second.total_space);
  }
  metrics->set_disable_tablet_split_if_default_ttl(ts_metrics_.disable_tablet_split_if_default_ttl);
}

bool TSDescriptor::HasTabletDeletePending() const {
  SharedLock<decltype(lock_)> l(lock_);
  return !tablets_pending_delete_.empty();
}

bool TSDescriptor::IsTabletDeletePending(const std::string& tablet_id) const {
  SharedLock<decltype(lock_)> l(lock_);
  return tablets_pending_delete_.count(tablet_id);
}

std::string TSDescriptor::PendingTabletDeleteToString() const {
  SharedLock<decltype(lock_)> l(lock_);
  return yb::ToString(tablets_pending_delete_);
}

void TSDescriptor::AddPendingTabletDelete(const std::string& tablet_id) {
  std::lock_guard l(lock_);
  tablets_pending_delete_.insert(tablet_id);
}

void TSDescriptor::ClearPendingTabletDelete(const std::string& tablet_id) {
  std::lock_guard l(lock_);
  tablets_pending_delete_.erase(tablet_id);
}

std::size_t TSDescriptor::NumTasks() const {
  SharedLock<decltype(lock_)> l(lock_);
  return tablets_pending_delete_.size();
}

bool TSDescriptor::IsLive() const {
  return TimeSinceHeartbeat().ToMilliseconds() <
         GetAtomicFlag(&FLAGS_tserver_unresponsive_timeout_ms) && !IsRemoved();
}

bool TSDescriptor::IsLiveAndHasReported() const {
  return IsLive() && has_tablet_report();
}

std::string TSDescriptor::ToString() const {
  SharedLock<decltype(lock_)> l(lock_);
  return Format("{ permanent_uuid: $0 registration: $1 placement_id: $2 }",
                permanent_uuid_, ts_information_->registration(), placement_id_);
}

bool TSDescriptor::IsReadOnlyTS(const ReplicationInfoPB& replication_info) const {
  const PlacementInfoPB& placement_info = replication_info.live_replicas();
  if (placement_info.has_placement_uuid()) {
    return placement_info.placement_uuid() != placement_uuid();
  }
  return !placement_uuid().empty();
}
} // namespace master
} // namespace yb
