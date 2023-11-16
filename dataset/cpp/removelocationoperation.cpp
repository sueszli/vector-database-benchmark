// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "removelocationoperation.h"
#include <vespa/storageapi/message/removelocation.h>
#include <vespa/document/bucket/bucketselector.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/select/parser.h>
#include <vespa/storage/distributor/distributor_bucket_space.h>
#include <vespa/vdslib/state/clusterstate.h>

#include <vespa/log/log.h>
LOG_SETUP(".distributor.operations.external.remove_location");

using document::BucketSpace;

namespace storage::distributor {

RemoveLocationOperation::RemoveLocationOperation(
        const DistributorNodeContext& node_ctx,
        DistributorStripeOperationContext& op_ctx,
        const DocumentSelectionParser& parser,
        DistributorBucketSpace &bucketSpace,
        std::shared_ptr<api::RemoveLocationCommand> msg,
        PersistenceOperationMetricSet& metric)
    : Operation(),
      _tracker(metric, std::make_shared<api::RemoveLocationReply>(*msg), node_ctx, op_ctx, _cancel_scope),
      _msg(std::move(msg)),
      _node_ctx(node_ctx),
      _parser(parser),
      _bucketSpace(bucketSpace)
{}

RemoveLocationOperation::~RemoveLocationOperation() = default;

int
RemoveLocationOperation::getBucketId(
        const DistributorNodeContext& node_ctx,
        const DocumentSelectionParser& parser,
        const api::RemoveLocationCommand& cmd, document::BucketId& bid)
{
    document::BucketSelector bucketSel(node_ctx.bucket_id_factory());
    std::unique_ptr<document::BucketSelector::BucketVector> exprResult
        = bucketSel.select(*parser.parse_selection(cmd.getDocumentSelection()));

    if (!exprResult.get()) {
        return 0;
    } else if (exprResult->size() != 1) {
        return exprResult->size();
    } else {
        bid = (*exprResult)[0];
        return 1;
    }
}

void
RemoveLocationOperation::onStart(DistributorStripeMessageSender& sender)
{
    document::BucketId bid;
    int count = getBucketId(_node_ctx, _parser, *_msg, bid);

    if (count != 1) {
        _tracker.fail(sender,
                      api::ReturnCode(api::ReturnCode::ILLEGAL_PARAMETERS,
                                      "Document selection could not be mapped to a single location"));
    }

    std::vector<BucketDatabase::Entry> entries;
    _bucketSpace.getBucketDatabase().getAll(bid, entries);

    bool sent = false;
    for (uint32_t j = 0; j < entries.size(); ++j) {
        const BucketDatabase::Entry& e = entries[j];

        std::vector<uint16_t> nodes = e->getNodes();

        for (uint32_t i = 0; i < nodes.size(); i++) {
            auto command = std::make_shared<api::RemoveLocationCommand>(_msg->getDocumentSelection(),
                                                                        document::Bucket(_msg->getBucket().getBucketSpace(), e.getBucketId()));

            copyMessageSettings(*_msg, *command);
            _tracker.queueCommand(std::move(command), nodes[i]);
            sent = true;
        }
    }

    if (!sent) {
        LOG(debug,
            "Remove location %s failed since no available nodes found. "
            "System state is %s",
            _msg->toString().c_str(),
            _bucketSpace.getClusterState().toString().c_str());

        _tracker.fail(sender, api::ReturnCode(api::ReturnCode::OK));
    } else {
        _tracker.flushQueue(sender);
    }
};


void
RemoveLocationOperation::onReceive(
        DistributorStripeMessageSender& sender,
        const std::shared_ptr<api::StorageReply> & msg)
{
    _tracker.receiveReply(sender, static_cast<api::BucketInfoReply&>(*msg));
}

void
RemoveLocationOperation::onClose(DistributorStripeMessageSender& sender)
{
    _tracker.fail(sender, api::ReturnCode(api::ReturnCode::ABORTED,
                                          "Process is shutting down"));
}

}
