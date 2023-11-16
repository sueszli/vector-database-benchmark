// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "statechecker.h"
#include "distributor_bucket_space.h"
#include "distributor_stripe_component.h"
#include <vespa/vdslib/distribution/distribution.h>
#include <vespa/vdslib/state/clusterstate.h>
#include <sstream>

#include <vespa/log/log.h>
LOG_SETUP(".distributor.statechecker");

namespace storage::distributor {

namespace {

class StoredResultImpl
    : public StateChecker::ResultImpl
{
    mutable IdealStateOperation::UP _operation;
    MaintenancePriority _priority;
public:
    StoredResultImpl(const StoredResultImpl &) = delete;
    StoredResultImpl & operator = (const StoredResultImpl &) = delete;
    StoredResultImpl()
        : _operation(),
          _priority(MaintenancePriority::NO_MAINTENANCE_NEEDED)
    {}

    StoredResultImpl(IdealStateOperation::UP operation,
                     MaintenancePriority priority)
        : _operation(std::move(operation)),
          _priority(priority)
    {}

    IdealStateOperation::UP createOperation() override {
        return std::move(_operation);
    }

    MaintenancePriority getPriority() const override {
        return _priority;
    }

    MaintenanceOperation::Type getType() const override {
        assert(_operation.get());
        return _operation->getType();
    }
};

}

StateChecker::Result
StateChecker::Result::noMaintenanceNeeded()
{
    return Result({});
}

StateChecker::Result
StateChecker::Result::createStoredResult(IdealStateOperation::UP operation, MaintenancePriority::Priority priority)
{
    return Result(std::make_unique<StoredResultImpl>(std::move(operation), MaintenancePriority(priority)));
}

StateChecker::Context::Context(const DistributorNodeContext& node_ctx_in,
                               const DistributorStripeOperationContext& op_ctx_in,
                               const DistributorBucketSpace& distributorBucketSpace,
                               NodeMaintenanceStatsTracker& statsTracker,
                               const document::Bucket& bucket_)
    : bucket(bucket_),
      siblingBucket(op_ctx_in.get_sibling(bucket.getBucketId())),
      systemState(distributorBucketSpace.getClusterState()),
      pending_cluster_state(op_ctx_in.pending_cluster_state_or_null(bucket_.getBucketSpace())),
      distributorConfig(op_ctx_in.distributor_config()),
      distribution(distributorBucketSpace.getDistribution()),
      gcTimeCalculator(op_ctx_in.bucket_id_hasher(), distributorConfig.getGarbageCollectionInterval()),
      idealStateBundle(distributorBucketSpace.get_ideal_service_layer_nodes_bundle(bucket.getBucketId())),
      node_ctx(node_ctx_in),
      op_ctx(op_ctx_in),
      db(distributorBucketSpace.getBucketDatabase()),
      stats(statsTracker),
      merges_inhibited_in_bucket_space(distributorBucketSpace.merges_inhibited()),
      _entry()
{ }

StateChecker::Context::~Context() = default;

void
StateChecker::Context::fillParentAndChildBuckets()
{
    db.getAll(getBucketId(), entries);
    if (entries.empty()) {
        LOG(spam, "Did not find bucket %s in bucket database", bucket.toString().c_str());
    }
}

void
StateChecker::Context::fillSiblingBucket()
{
    siblingEntry = db.get(siblingBucket);
}

const BucketDatabase::Entry*
StateChecker::Context::getEntryForPrimaryBucket() const
{
    for (auto & e : entries) {
        if (e.getBucketId() == getBucketId() && ! e->getNodes().empty()) {
            return &e;
        }
    }
    return nullptr;
}


std::string
StateChecker::Context::toString() const
{
    std::ostringstream ss;
    ss << "entries: {";

    for (uint32_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << entries[i].getBucketId() << ": [" << entries[i]->toString() << "]";
    }

    ss << "}, state: " << systemState;
    return ss.str();
}

}
