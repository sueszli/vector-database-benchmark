// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "simplemaintenancescanner.h"
#include <vespa/storage/distributor/distributor_bucket_space.h>
#include <ostream>
#include <cassert>

namespace storage::distributor {

SimpleMaintenanceScanner::SimpleMaintenanceScanner(BucketPriorityDatabase& bucketPriorityDb,
                                                   const MaintenancePriorityGenerator& priorityGenerator,
                                                   const DistributorBucketSpaceRepo& bucketSpaceRepo)
    : _bucketPriorityDb(bucketPriorityDb),
      _priorityGenerator(priorityGenerator),
      _bucketSpaceRepo(bucketSpaceRepo),
      _bucketSpaceItr(_bucketSpaceRepo.begin()),
      _bucketCursor(),
      _pendingMaintenance()
{
}

SimpleMaintenanceScanner::~SimpleMaintenanceScanner() = default;

bool
SimpleMaintenanceScanner::GlobalMaintenanceStats::operator==(const GlobalMaintenanceStats& rhs) const noexcept
{
    return pending == rhs.pending;
}

void
SimpleMaintenanceScanner::GlobalMaintenanceStats::merge(const GlobalMaintenanceStats& rhs) noexcept
{
    for (size_t i = 0; i < pending.size(); ++i) {
        pending[i] += rhs.pending[i];
    }
}

void
SimpleMaintenanceScanner::PendingMaintenanceStats::merge(const PendingMaintenanceStats& rhs)
{
    global.merge(rhs.global);
    perNodeStats.merge(rhs.perNodeStats);
}

SimpleMaintenanceScanner::PendingMaintenanceStats::PendingMaintenanceStats() noexcept = default;
SimpleMaintenanceScanner::PendingMaintenanceStats::~PendingMaintenanceStats() = default;
SimpleMaintenanceScanner::PendingMaintenanceStats::PendingMaintenanceStats(const PendingMaintenanceStats &) = default;
SimpleMaintenanceScanner::PendingMaintenanceStats::PendingMaintenanceStats(PendingMaintenanceStats &&) noexcept = default;
SimpleMaintenanceScanner::PendingMaintenanceStats &
SimpleMaintenanceScanner::PendingMaintenanceStats::operator = (PendingMaintenanceStats &&) noexcept = default;

SimpleMaintenanceScanner::PendingMaintenanceStats
SimpleMaintenanceScanner::PendingMaintenanceStats::fetch_and_reset() {
    PendingMaintenanceStats prev = std::move(*this);
    global = GlobalMaintenanceStats();
    perNodeStats.reset(prev.perNodeStats.numNodes());
    return prev;
}

MaintenanceScanner::ScanResult
SimpleMaintenanceScanner::scanNext()
{
    for (;;) {
        if (_bucketSpaceItr == _bucketSpaceRepo.end()) {
            return ScanResult::createDone();
        }
        const auto &bucketDb(_bucketSpaceItr->second->getBucketDatabase());
        BucketDatabase::Entry entry(bucketDb.getNext(_bucketCursor));
        if (!entry.valid()) {
            ++_bucketSpaceItr;
            _bucketCursor = document::BucketId();
            continue;
        }
        countBucket(_bucketSpaceItr->first, entry.getBucketInfo());
        prioritizeBucket(document::Bucket(_bucketSpaceItr->first, entry.getBucketId()));
        _bucketCursor = entry.getBucketId();
        return ScanResult::createNotDone(_bucketSpaceItr->first, std::move(entry));
    }
}

SimpleMaintenanceScanner::PendingMaintenanceStats
SimpleMaintenanceScanner::fetch_and_reset()
{
    _bucketCursor = document::BucketId();
    _bucketSpaceItr = _bucketSpaceRepo.begin();
    return _pendingMaintenance.fetch_and_reset();
}

void
SimpleMaintenanceScanner::countBucket(document::BucketSpace bucketSpace, const BucketInfo &info)
{
    NodeMaintenanceStatsTracker &perNodeStats = _pendingMaintenance.perNodeStats;
    uint32_t nodeCount = info.getNodeCount();
    for (uint32_t i = 0; i < nodeCount; ++i) {
        const BucketCopy &node = info.getNodeRef(i);
        perNodeStats.incTotal(node.getNode(), bucketSpace);
    }
}

void
SimpleMaintenanceScanner::prioritizeBucket(const document::Bucket &bucket)
{
    MaintenancePriorityAndType pri(_priorityGenerator.prioritize(bucket, _pendingMaintenance.perNodeStats));
    if (pri.requiresMaintenance()) {
        _bucketPriorityDb.setPriority(PrioritizedBucket(bucket, pri.getPriority().getPriority()));
        assert(pri.getType() != MaintenanceOperation::OPERATION_COUNT);
        ++_pendingMaintenance.global.pending[pri.getType()];
    }
}

std::ostream&
operator<<(std::ostream& os, const SimpleMaintenanceScanner::GlobalMaintenanceStats& stats)
{
    using MO = MaintenanceOperation;
    os << "delete bucket: "        << stats.pending[MO::DELETE_BUCKET]
       << ", merge bucket: "       << stats.pending[MO::MERGE_BUCKET]
       << ", split bucket: "       << stats.pending[MO::SPLIT_BUCKET]
       << ", join bucket: "        << stats.pending[MO::JOIN_BUCKET]
       << ", set bucket state: "   << stats.pending[MO::SET_BUCKET_STATE]
       << ", garbage collection: " << stats.pending[MO::GARBAGE_COLLECTION];
    return os;
}

}
