// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <tests/common/dummystoragelink.h>
#include <tests/common/testhelper.h>
#include <tests/common/teststorageapp.h>
#include <vespa/config/helper/configgetter.hpp>
#include <vespa/document/test/make_document_bucket.h>
#include <vespa/messagebus/dynamicthrottlepolicy.h>
#include <vespa/storage/persistence/messages.h>
#include <vespa/storage/storageserver/mergethrottler.h>
#include <vespa/storageapi/message/bucket.h>
#include <vespa/storageapi/message/state.h>
#include <vespa/vdslib/state/clusterstate.h>
#include <vespa/vespalib/gtest/gtest.h>
#include <vespa/vespalib/util/exceptions.h>
#include <vespa/vespalib/util/size_literals.h>
#include <unordered_set>
#include <memory>
#include <iterator>
#include <vector>
#include <chrono>
#include <climits>
#include <thread>

using namespace document;
using namespace storage::api;
using document::test::makeDocumentBucket;
using namespace ::testing;

namespace storage {

namespace {

using StorServerConfig = vespa::config::content::core::StorServerConfig;
using StorServerConfigBuilder = vespa::config::content::core::StorServerConfigBuilder;

vespalib::string _storage("storage");

std::unique_ptr<StorServerConfig> default_server_config() {
    vdstestlib::DirConfig dir_config(getStandardConfig(true));
    auto cfg_uri = ::config::ConfigUri(dir_config.getConfigId());
    return config_from<StorServerConfig>(cfg_uri);
}

struct MergeBuilder {
    document::BucketId           _bucket;
    api::Timestamp               _maxTimestamp;
    std::vector<uint16_t>        _nodes;
    std::vector<uint16_t>        _chain;
    std::unordered_set<uint16_t> _source_only;
    uint64_t                     _clusterStateVersion;
    uint32_t                     _memory_usage;
    bool                         _unordered;

    explicit MergeBuilder(const document::BucketId& bucket)
        : _bucket(bucket),
          _maxTimestamp(1234),
          _chain(),
          _clusterStateVersion(1),
          _memory_usage(0),
          _unordered(false)
    {
        nodes(0, 1, 2);
    }

    ~MergeBuilder();

    MergeBuilder& nodes(uint16_t n0) {
        _nodes.clear();
        _nodes.push_back(n0);
        return *this;
    }
    MergeBuilder& nodes(uint16_t n0, uint16_t n1) {
        _nodes.clear();
        _nodes.push_back(n0);
        _nodes.push_back(n1);
        return *this;
    }
    MergeBuilder& nodes(uint16_t n0, uint16_t n1, uint16_t n2) {
        _nodes.clear();
        _nodes.push_back(n0);
        _nodes.push_back(n1);
        _nodes.push_back(n2);
        return *this;
    }
    MergeBuilder& maxTimestamp(api::Timestamp maxTs) {
        _maxTimestamp = maxTs;
        return *this;
    }
    MergeBuilder& clusterStateVersion(uint64_t csv) {
        _clusterStateVersion = csv;
        return *this;
    }
    MergeBuilder& chain(uint16_t n0) {
        _chain.clear();
        _chain.push_back(n0);
        return *this;
    }
    MergeBuilder& chain(uint16_t n0, uint16_t n1) {
        _chain.clear();
        _chain.push_back(n0);
        _chain.push_back(n1);
        return *this;
    }
    MergeBuilder& chain(uint16_t n0, uint16_t n1, uint16_t n2) {
        _chain.clear();
        _chain.push_back(n0);
        _chain.push_back(n1);
        _chain.push_back(n2);
        return *this;
    }
    MergeBuilder& source_only(uint16_t node) {
        _source_only.insert(node);
        return *this;
    }
    MergeBuilder& memory_usage(uint32_t usage_bytes) {
        _memory_usage = usage_bytes;
        return *this;
    }
    MergeBuilder& unordered(bool is_unordered) {
        _unordered = is_unordered;
        return *this;
    }

    api::MergeBucketCommand::SP create() const {
        std::vector<api::MergeBucketCommand::Node> n;
        for (uint32_t i = 0; i < _nodes.size(); ++i) {
            uint16_t node = _nodes[i];
            bool source_only = (_source_only.find(node) != _source_only.end());
            n.emplace_back(node, source_only);
        }
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(_bucket), n, _maxTimestamp,
                _clusterStateVersion, _chain);
        cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, _nodes[0]));
        cmd->set_estimated_memory_footprint(_memory_usage);
        cmd->set_use_unordered_forwarding(_unordered);
        return cmd;
    }
};

MergeBuilder::~MergeBuilder() = default;

std::shared_ptr<api::SetSystemStateCommand>
makeSystemStateCmd(const std::string& state)
{
    return std::make_shared<api::SetSystemStateCommand>(lib::ClusterState(state));
}

} // anon ns

struct MergeThrottlerTest : Test {
    static constexpr int _storageNodeCount = 3;
    static constexpr int _messageWaitTime = 100;

    // Using n storage node links and dummy servers
    std::vector<std::shared_ptr<DummyStorageLink> > _topLinks;
    std::vector<std::shared_ptr<TestServiceLayerApp> > _servers;
    std::vector<MergeThrottler*> _throttlers;
    std::vector<DummyStorageLink*> _bottomLinks;

    MergeThrottlerTest();
    ~MergeThrottlerTest() override;

    void SetUp() override;
    void TearDown() override;

    MergeThrottler& throttler(size_t idx) noexcept {
        assert(idx < _throttlers.size());
        return *_throttlers[idx];
    }

    api::MergeBucketCommand::SP sendMerge(const MergeBuilder&);

    void send_and_expect_reply(
        const std::shared_ptr<api::StorageMessage>& msg,
        const api::MessageType& expectedReplyType,
        api::ReturnCode::Result expectedResultCode);

    std::shared_ptr<api::StorageMessage> send_and_expect_forwarding(const std::shared_ptr<api::StorageMessage>& msg);

    void fill_throttler_queue_with_n_commands(uint16_t throttler_index, size_t queued_count);
    void receive_chained_merge_with_full_queue(bool disable_queue_limits, bool unordered_fwd = false);

    std::shared_ptr<api::MergeBucketCommand> peek_throttler_queue_top(size_t throttler_idx) {
        auto& queue = _throttlers[throttler_idx]->getMergeQueue();
        assert(!queue.empty());
        auto merge = std::dynamic_pointer_cast<api::MergeBucketCommand>(queue.begin()->_msg);
        assert(merge);
        return merge;
    }

    [[nodiscard]] uint32_t throttler_max_merges_pending(uint16_t throttler_index) const noexcept;
};

MergeThrottlerTest::MergeThrottlerTest() = default;
MergeThrottlerTest::~MergeThrottlerTest() = default;

void
MergeThrottlerTest::SetUp()
{
    auto config = default_server_config();
    for (int i = 0; i < _storageNodeCount; ++i) {
        auto server = std::make_unique<TestServiceLayerApp>(NodeIndex(i));
        server->setClusterState(lib::ClusterState("distributor:100 storage:100 version:1"));
        std::unique_ptr<DummyStorageLink> top;

        top = std::make_unique<DummyStorageLink>();
        auto* throttler = new MergeThrottler(*config, server->getComponentRegister(), vespalib::HwInfo());
        // MergeThrottler will be sandwiched in between two dummy links
        top->push_back(std::unique_ptr<StorageLink>(throttler));
        auto* bottom = new DummyStorageLink;
        throttler->push_back(std::unique_ptr<StorageLink>(bottom));

        _servers.push_back(std::shared_ptr<TestServiceLayerApp>(server.release()));
        _throttlers.push_back(throttler);
        _bottomLinks.push_back(bottom);
        top->open();
        _topLinks.push_back(std::shared_ptr<DummyStorageLink>(top.release()));
    }
}

void
MergeThrottlerTest::TearDown()
{
    for (auto& link : _topLinks) {
        if (link->getState() == StorageLink::OPENED) {
            link->close();
            link->flush();
        }
        link.reset();
    }
    _topLinks.clear();
    _bottomLinks.clear();
    _throttlers.clear();
    _servers.clear();
}

namespace {

template <typename Iterator>
bool
checkChain(const StorageMessage::SP& msg,
           Iterator first, Iterator end)
{
    auto& cmd = dynamic_cast<const MergeBucketCommand&>(*msg);
    if (cmd.getChain().size() != static_cast<size_t>(std::distance(first, end))) {
        return false;
    }
    return std::equal(cmd.getChain().begin(), cmd.getChain().end(), first);
}

void waitUntilMergeQueueIs(MergeThrottler& throttler, size_t sz, int timeout)
{
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        size_t count;
        {
            std::lock_guard lock(throttler.getStateLock());
            count = throttler.getMergeQueue().size();
        }
        if (count == sz) {
            break;
        }
        auto now = std::chrono::steady_clock::now();
        if (now - start > std::chrono::seconds(timeout)) {
            std::ostringstream os;
            os << "Timeout while waiting for merge queue with " << sz << " items. Had "
               << count << " at timeout.";
            throw vespalib::IllegalStateException(os.str(), VESPA_STRLOC);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

}

uint32_t
MergeThrottlerTest::throttler_max_merges_pending(uint16_t throttler_index) const noexcept
{
    return static_cast<uint32_t>(_throttlers[throttler_index]->getThrottlePolicy().getMaxWindowSize());
}

// Extremely simple test that just checks that (min|max)_merges_per_node
// under the stor-server config gets propagated to all the nodes
TEST_F(MergeThrottlerTest, merges_config) {
    for (int i = 0; i < _storageNodeCount; ++i) {
        EXPECT_EQ(25, throttler_max_merges_pending(i));
        EXPECT_EQ(20, _throttlers[i]->getMaxQueueSize());
    }
}

// Test that a distributor sending a merge to the lowest-index storage
// node correctly invokes a merge forwarding chain and subsequent unwind.
TEST_F(MergeThrottlerTest, chain) {
    uint16_t indices[_storageNodeCount];
    for (int i = 0; i < _storageNodeCount; ++i) {
        indices[i] = i;
        _servers[i]->setClusterState(lib::ClusterState("distributor:100 storage:100 version:123"));
    }

    Bucket bucket(makeDocumentBucket(BucketId(14, 0x1337)));

    // Use different node permutations to ensure it works no matter which node is
    // set as the executor. More specifically, _all_ permutations.
    do {
        uint16_t lastNodeIdx = _storageNodeCount - 1;
        uint16_t executorNode = indices[0];

        std::vector<MergeBucketCommand::Node> nodes;
        for (int i = 0; i < _storageNodeCount; ++i) {
            nodes.push_back(MergeBucketCommand::Node(indices[i], (i + executorNode) % 2 == 0));
        }
        auto cmd = std::make_shared<MergeBucketCommand>(bucket, nodes, UINT_MAX, 123);
        cmd->setPriority(7);
        cmd->setTimeout(54321ms);
        cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 0));
        const uint16_t distributorIndex = 123;
        cmd->setSourceIndex(distributorIndex); // Dummy distributor index that must be forwarded
        cmd->set_estimated_memory_footprint(456'789);

        StorageMessage::SP fwd = cmd;
        StorageMessage::SP fwdToExec;

        // TODO: make generic wrt. _storageNodeCount

        for (int i = 0; i < _storageNodeCount - 1; ++i) {
            if (i == executorNode) {
                fwdToExec = fwd;
            }
            ASSERT_EQ(i, _servers[i]->getIndex());
            // No matter the node order, command is always sent to node 0 -> 1 -> 2 etc
            _topLinks[i]->sendDown(fwd);
            _topLinks[i]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

            // Forwarded merge should not be sent down. Should not be necessary
            // to lock throttler here, since it should be sleeping like a champion
            ASSERT_EQ(0, _bottomLinks[i]->getNumCommands());
            ASSERT_EQ(1, _topLinks[i]->getNumReplies());
            ASSERT_EQ(1, _throttlers[i]->getActiveMerges().size());

            fwd = _topLinks[i]->getAndRemoveMessage(MessageType::MERGEBUCKET);
            ASSERT_EQ(i + 1, fwd->getAddress()->getIndex());
            ASSERT_EQ(distributorIndex, dynamic_cast<const StorageCommand&>(*fwd).getSourceIndex());
            {
                std::vector<uint16_t> chain;
                for (int j = 0; j <= i; ++j) {
                    chain.push_back(j);
                }
                EXPECT_TRUE(checkChain(fwd, chain.begin(), chain.end()));
            }
            // Ensure operation properties are forwarded as expected
            EXPECT_EQ(7, static_cast<int>(fwd->getPriority()));
            auto& as_merge = dynamic_cast<const MergeBucketCommand&>(*fwd);
            EXPECT_EQ(as_merge.getClusterStateVersion(), 123);
            EXPECT_EQ(as_merge.getTimeout(), 54321ms);
            EXPECT_EQ(as_merge.estimated_memory_footprint(), 456'789);
        }

        _topLinks[lastNodeIdx]->sendDown(fwd);

        // If node 2 is the first in the node list, it should immediately execute
        // the merge. Otherwise, a cycle with the first node should be formed.
        if (executorNode != lastNodeIdx) {
            _topLinks[lastNodeIdx]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
            // Forwarded merge should not be sent down
            ASSERT_EQ(0, _bottomLinks[lastNodeIdx]->getNumCommands());
            ASSERT_EQ(1, _topLinks[lastNodeIdx]->getNumReplies());
            ASSERT_EQ(1, _throttlers[lastNodeIdx]->getActiveMerges().size());

            fwd = _topLinks[lastNodeIdx]->getAndRemoveMessage(MessageType::MERGEBUCKET);
            ASSERT_EQ(executorNode, fwd->getAddress()->getIndex());
            ASSERT_EQ(distributorIndex, dynamic_cast<const StorageCommand&>(*fwd).getSourceIndex());
            {
                std::vector<uint16_t> chain;
                for (int j = 0; j < _storageNodeCount; ++j) {
                    chain.push_back(j);
                }
                EXPECT_TRUE(checkChain(fwd, chain.begin(), chain.end()));
            }
            EXPECT_EQ(7, static_cast<int>(fwd->getPriority()));
            EXPECT_EQ(123, dynamic_cast<const MergeBucketCommand&>(*fwd).getClusterStateVersion());
            EXPECT_EQ(54321ms, dynamic_cast<const StorageCommand&>(*fwd).getTimeout());

            _topLinks[executorNode]->sendDown(fwd);
        }

        _bottomLinks[executorNode]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

        // Forwarded merge has now been sent down to persistence layer
        ASSERT_EQ(1, _bottomLinks[executorNode]->getNumCommands());
        ASSERT_EQ(0, _topLinks[executorNode]->getNumReplies()); // No reply sent yet
        ASSERT_EQ(1, _throttlers[executorNode]->getActiveMerges().size()); // no re-registering merge

        if (executorNode != lastNodeIdx) {
            // The MergeBucketCommand that is kept in the executor node should
            // be the one from the node it initially got it from, NOT the one
            // from the last node, since the chain has looped
            ASSERT_TRUE(_throttlers[executorNode]->getActiveMerges().find(bucket)
                        != _throttlers[executorNode]->getActiveMerges().end());
            ASSERT_EQ(static_cast<StorageMessage*>(fwdToExec.get()),
                      _throttlers[executorNode]->getActiveMerges().find(bucket)->second.getMergeCmd().get());
        }

        // Send reply up from persistence layer to simulate a completed
        // merge operation. Chain should now unwind properly
        fwd = _bottomLinks[executorNode]->getAndRemoveMessage(MessageType::MERGEBUCKET);
        EXPECT_EQ(7, static_cast<int>(fwd->getPriority()));
        EXPECT_EQ(123, dynamic_cast<const MergeBucketCommand&>(*fwd).getClusterStateVersion());
        EXPECT_EQ(54321ms, dynamic_cast<const StorageCommand&>(*fwd).getTimeout());

        auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
        reply->setResult(ReturnCode(ReturnCode::OK, "Great success! :D-|-<"));
        _bottomLinks[executorNode]->sendUp(reply);

        _topLinks[executorNode]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

        if (executorNode != lastNodeIdx) {
            // Merge should not be removed yet from executor, since it's pending an unwind
            ASSERT_EQ(1, _throttlers[executorNode]->getActiveMerges().size());
            ASSERT_EQ(static_cast<StorageMessage*>(fwdToExec.get()),
                      _throttlers[executorNode]->getActiveMerges().find(bucket)->second.getMergeCmd().get());
        }
        // MergeBucketReply waiting to be sent back to node 2. NOTE: we don't have any
        // transport context stuff set up here to perform the reply mapping, so we
        // have to emulate it
        ASSERT_EQ(1, _topLinks[executorNode]->getNumReplies());

        auto unwind = _topLinks[executorNode]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
        ASSERT_EQ(executorNode, unwind->getAddress()->getIndex());

        // eg: 0 -> 2 -> 1 -> 0. Or: 2 -> 1 -> 0 if no cycle
        for (int i = (executorNode != lastNodeIdx ? _storageNodeCount - 1 : _storageNodeCount - 2); i >= 0; --i) {
            _topLinks[i]->sendDown(unwind);
            _topLinks[i]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

            ASSERT_EQ(0, _bottomLinks[i]->getNumCommands());
            ASSERT_EQ(1, _topLinks[i]->getNumReplies());
            ASSERT_EQ(0, _throttlers[i]->getActiveMerges().size());

            unwind = _topLinks[i]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
            ASSERT_EQ(uint16_t(i), unwind->getAddress()->getIndex());
        }

        const MergeBucketReply& mbr = dynamic_cast<const MergeBucketReply&>(*unwind);

        EXPECT_EQ(ReturnCode::OK, mbr.getResult().getResult());
        EXPECT_EQ(vespalib::string("Great success! :D-|-<"), mbr.getResult().getMessage());
        EXPECT_EQ(bucket, mbr.getBucket());

    } while (std::next_permutation(indices, indices + _storageNodeCount));
}

TEST_F(MergeThrottlerTest, with_source_only_node) {
    BucketId bid(14, 0x1337);

    std::vector<MergeBucketCommand::Node> nodes({{0}, {2}, {1, true}});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, UINT_MAX, 123);

    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 0));
    _topLinks[0]->sendDown(cmd);

    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    StorageMessage::SP fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    ASSERT_EQ(1, fwd->getAddress()->getIndex());

    _topLinks[1]->sendDown(fwd);

    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    fwd = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    ASSERT_EQ(2, fwd->getAddress()->getIndex());

    _topLinks[2]->sendDown(fwd);

    _topLinks[2]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    fwd = _topLinks[2]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    ASSERT_EQ(0, fwd->getAddress()->getIndex());

    _topLinks[0]->sendDown(fwd);
    _bottomLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    _bottomLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Great success! :D-|-<"));
    _bottomLinks[0]->sendUp(reply);

    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    ASSERT_EQ(0, fwd->getAddress()->getIndex());

    // Assume everything's fine from here on out
}

// 4.2 distributors don't guarantee they'll send to lowest node
// index, so we must detect such situations and execute the merge
// immediately rather than attempt to chain it. Test that this
// is done correctly.
// TODO remove functionality and test
TEST_F(MergeThrottlerTest, legacy_42_distributor_behavior) {
    BucketId bid(32, 0xfeef00);

    std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, 1234);

    // Send to node 1, which is not the lowest index
    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 1));
    _topLinks[1]->sendDown(cmd);
    _bottomLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    // Should now have been sent to persistence layer
    ASSERT_EQ(1, _bottomLinks[1]->getNumCommands());
    ASSERT_EQ(0, _topLinks[1]->getNumReplies()); // No reply sent yet
    ASSERT_EQ(1, _throttlers[1]->getActiveMerges().size());

    // Send reply up from persistence layer to simulate a completed
    // merge operation. Merge should be removed from state.
    _bottomLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*cmd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Tonight we dine on turtle soup!"));
    _bottomLinks[1]->sendUp(reply);
    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    ASSERT_EQ(0, _bottomLinks[1]->getNumCommands());
    ASSERT_EQ(1, _topLinks[1]->getNumReplies());
    ASSERT_EQ(0, _throttlers[1]->getActiveMerges().size());

    EXPECT_EQ(uint64_t(1), _throttlers[1]->getMetrics().local.ok.getValue());
}

// Test that we don't take ownership of the merge command when we're
// just passing it through to the persistence layer when receiving
// a merge command that presumably comes form a 4.2 distributor
// TODO remove functionality and test
TEST_F(MergeThrottlerTest, legacy_42_distributor_behavior_does_not_take_ownership) {
    BucketId bid(32, 0xfeef00);

    std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, 1234);

    // Send to node 1, which is not the lowest index
    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 1));
    _topLinks[1]->sendDown(cmd);
    _bottomLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    // Should now have been sent to persistence layer
    ASSERT_EQ(1, _bottomLinks[1]->getNumCommands());
    ASSERT_EQ(0, _topLinks[1]->getNumReplies()); // No reply sent yet
    ASSERT_EQ(1, _throttlers[1]->getActiveMerges().size());

    _bottomLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // To ensure we don't try to deref any non-owned messages
    framework::HttpUrlPath path("?xml");
    std::ostringstream ss;
    _throttlers[1]->reportStatus(ss, path);

    // Flush throttler (synchronously). Should NOT generate a reply
    // for the merge command, as it is not owned by the throttler
    _throttlers[1]->onFlush(true);

    ASSERT_EQ(0, _bottomLinks[1]->getNumCommands());
    ASSERT_EQ(0, _topLinks[1]->getNumReplies());
    ASSERT_EQ(0, _throttlers[1]->getActiveMerges().size());

    // Send a belated reply from persistence up just to ensure the
    // throttler doesn't throw a fit if it receives an unknown merge
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*cmd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Tonight we dine on turtle soup!"));
    _bottomLinks[1]->sendUp(reply);
    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    ASSERT_EQ(0, _bottomLinks[1]->getNumCommands());
    ASSERT_EQ(1, _topLinks[1]->getNumReplies());
    ASSERT_EQ(0, _throttlers[1]->getActiveMerges().size());
}

// Test that we don't take ownership of the merge command when we're
// just passing it through to the persistence layer when we're at the
// the end of the chain and also the designated executor
TEST_F(MergeThrottlerTest, end_of_chain_execution_does_not_take_ownership) {
    BucketId bid(32, 0xfeef00);

    std::vector<MergeBucketCommand::Node> nodes({{2}, {1}, {0}});
    std::vector<uint16_t> chain({0, 1});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, 1234, 1, chain);

    // Send to last node, which is not the lowest index
    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 3));
    _topLinks[2]->sendDown(cmd);
    _bottomLinks[2]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    // Should now have been sent to persistence layer
    ASSERT_EQ(1, _bottomLinks[2]->getNumCommands());
    ASSERT_EQ(0, _topLinks[2]->getNumReplies()); // No reply sent yet
    ASSERT_EQ(1, _throttlers[2]->getActiveMerges().size());

    _bottomLinks[2]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // To ensure we don't try to deref any non-owned messages
    framework::HttpUrlPath path("");
    std::ostringstream ss;
    _throttlers[2]->reportStatus(ss, path);

    // Flush throttler (synchronously). Should NOT generate a reply
    // for the merge command, as it is not owned by the throttler
    _throttlers[2]->onFlush(true);

    ASSERT_EQ(0, _bottomLinks[2]->getNumCommands());
    ASSERT_EQ(0, _topLinks[2]->getNumReplies());
    ASSERT_EQ(0, _throttlers[2]->getActiveMerges().size());

    // Send a belated reply from persistence up just to ensure the
    // throttler doesn't throw a fit if it receives an unknown merge
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*cmd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Tonight we dine on turtle soup!"));
    _bottomLinks[2]->sendUp(reply);
    _topLinks[2]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    ASSERT_EQ(0, _bottomLinks[2]->getNumCommands());
    ASSERT_EQ(1, _topLinks[2]->getNumReplies());
    ASSERT_EQ(0, _throttlers[2]->getActiveMerges().size());
}

// Test that nodes resending a merge command won't lead to duplicate
// state registration/forwarding or erasing the already present state
// information.
TEST_F(MergeThrottlerTest, resend_handling) {
    BucketId bid(32, 0xbadbed);

    std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, 1234);

    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 1));
    _topLinks[0]->sendDown(cmd);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    StorageMessage::SP fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Resend from "distributor". Just use same message, as that won't matter here
    _topLinks[0]->sendDown(cmd);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    // Reply should be BUSY
    StorageMessage::SP reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);

    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
              ReturnCode::BUSY);

    _topLinks[1]->sendDown(fwd);
    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    fwd = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    _topLinks[2]->sendDown(fwd);
    _topLinks[2]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    _topLinks[2]->sendDown(fwd);
    _topLinks[2]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    // Reply should be BUSY
    reply = _topLinks[2]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
              ReturnCode::BUSY);

    fwd = _topLinks[2]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    _topLinks[0]->sendDown(fwd);
    _bottomLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    _topLinks[0]->sendDown(fwd);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
              ReturnCode::BUSY);
}

TEST_F(MergeThrottlerTest, priority_queuing) {
    // Fill up all active merges
    size_t maxPending = throttler_max_merges_pending(0);
    std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
    ASSERT_GE(maxPending, 4u);
    for (size_t i = 0; i < maxPending; ++i) {
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234);
        cmd->setPriority(100);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 0 queued
    _topLinks[0]->waitForMessages(maxPending, 5);
    waitUntilMergeQueueIs(*_throttlers[0], 0, _messageWaitTime);

    // Queue up some merges with different priorities
    int priorities[4] = { 200, 150, 120, 240 };
    int sortedPris[4] = { 120, 150, 200, 240 };
    for (int i = 0; i < 4; ++i) {
        auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(BucketId(32, i)), nodes, 1234);
        cmd->setPriority(priorities[i]);
        _topLinks[0]->sendDown(cmd);
    }

    waitUntilMergeQueueIs(*_throttlers[0], 4, _messageWaitTime);

    // Remove all but 4 forwarded merges
    for (size_t i = 0; i < maxPending - 4; ++i) {
        _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    }
    ASSERT_EQ(0, _topLinks[0]->getNumCommands());
    ASSERT_EQ(4, _topLinks[0]->getNumReplies());

    // Now when we start replying to merges, queued merges should be
    // processed in priority order
    for (int i = 0; i < 4; ++i) {
        StorageMessage::SP replyTo = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
        auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*replyTo));
        reply->setResult(ReturnCode(ReturnCode::OK, "whee"));
        _topLinks[0]->sendDown(reply);
    }

    _topLinks[0]->waitForMessages(8, _messageWaitTime); // 4 merges, 4 replies
    waitUntilMergeQueueIs(*_throttlers[0], 0, _messageWaitTime);

    for (int i = 0; i < 4; ++i) {
        StorageMessage::SP cmd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
        EXPECT_EQ(uint8_t(sortedPris[i]), cmd->getPriority());
    }
}

// Test that we can detect and reject merges that due to resending
// and potential priority queue sneaking etc may end up with duplicates
// in the queue for a merge that is already known.
TEST_F(MergeThrottlerTest, command_in_queue_duplicate_of_known_merge) {
    // Fill up all active merges and 1 queued one
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    for (size_t i = 0; i < maxPending + 1; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {uint16_t(2 + i)}, {uint16_t(5 + i)}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234);
        cmd->setPriority(100 - i);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 3 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 1, _messageWaitTime);

    // Add a merge for the same bucket twice to the queue
    {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {12}, {123}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf000feee)), nodes, 1234);
        _topLinks[0]->sendDown(cmd);
    }
    {
        // Different node set doesn't matter
        std::vector<MergeBucketCommand::Node> nodes({{0}, {124}, {14}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf000feee)), nodes, 1234);
        _topLinks[0]->sendDown(cmd);
    }

    waitUntilMergeQueueIs(*_throttlers[0], 3, _messageWaitTime);

    auto fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Remove and success-reply for 2 merges. This will give enough room
    // for the 2 first queued merges to be processed, the last one having a
    // duplicate in the queue.
    for (int i = 0; i < 2; ++i) {
        StorageMessage::SP fwd2 = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
        auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd2));
        reply->setResult(ReturnCode(ReturnCode::OK, ""));
        _topLinks[0]->sendDown(reply);
    }

    _topLinks[0]->waitForMessages(maxPending + 1, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 1, _messageWaitTime);

    // Remove all current merge commands/replies so we can work with a clean slate
    _topLinks[0]->getRepliesOnce();
    // Send a success-reply for fwd, allowing the duplicate from the queue
    // to have its moment to shine only to then be struck down mercilessly
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
    reply->setResult(ReturnCode(ReturnCode::OK, ""));
    _topLinks[0]->sendDown(reply);

    _topLinks[0]->waitForMessages(2, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 0, _messageWaitTime);

    // First reply is the successful merge reply
    auto reply2 = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply2).getResult().getResult(),
              ReturnCode::OK);

    // Second reply should be the BUSY-rejected duplicate
    auto reply1 = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply1).getResult().getResult(),
              ReturnCode::BUSY);
    EXPECT_TRUE(static_cast<MergeBucketReply&>(*reply1).getResult()
                .getMessage().find("out of date;") != std::string::npos);
}

// Test that sending a merge command to a node not in the set of
// to-be-merged nodes is handled gracefully.
// This is not a scenario that should ever actually happen, but for
// the sake of robustness, include it anyway.
TEST_F(MergeThrottlerTest, invalid_receiver_node) {
    std::vector<MergeBucketCommand::Node> nodes({{1}, {5}, {9}});
    auto cmd = std::make_shared<MergeBucketCommand>(
            makeDocumentBucket(BucketId(32, 0xf00baaaa)), nodes, 1234);

    // Send to node with index 0
    _topLinks[0]->sendDown(cmd);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
              ReturnCode::REJECTED);
    EXPECT_TRUE(static_cast<MergeBucketReply&>(*reply).getResult()
                .getMessage().find("which is not in its forwarding chain") != std::string::npos);
}

// Test that the throttling policy kicks in after a certain number of
// merges are forwarded and that the rest are queued in a prioritized
// order.
TEST_F(MergeThrottlerTest, forward_queued_merge) {
    // Fill up all active merges and then 3 queued ones
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    for (size_t i = 0; i < maxPending + 3; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {uint16_t(2 + i)}, {uint16_t(5 + i)}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234);
        cmd->setPriority(100 - i);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 3 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 3, _messageWaitTime);

    // Merge queue state should not be touched by worker thread now
    auto nextMerge = peek_throttler_queue_top(0);

    auto fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Remove all the rest of the active merges
    while (!_topLinks[0]->getReplies().empty()) {
        _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    }

    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Celebrate good times come on"));
    _topLinks[0]->sendDown(reply);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime); // Success rewind reply

    // Remove reply bound for distributor
    auto distReply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*distReply).getResult().getResult(),
              ReturnCode::OK);

    waitUntilMergeQueueIs(*_throttlers[0], 2, _messageWaitTime);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    ASSERT_EQ(0, _topLinks[0]->getNumCommands());
    ASSERT_EQ(1, _topLinks[0]->getNumReplies());

    // First queued merge should now have been registered and forwarded
    fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    ASSERT_EQ(static_cast<const MergeBucketCommand&>(*fwd).getBucketId(),
              nextMerge->getBucketId());

    ASSERT_TRUE(static_cast<const MergeBucketCommand&>(*fwd).getNodes() == nextMerge->getNodes());

    // Ensure forwarded merge has a higher priority than the next queued one
    EXPECT_LT(fwd->getPriority(), peek_throttler_queue_top(0)->getPriority());

    EXPECT_EQ(uint64_t(1), _throttlers[0]->getMetrics().chaining.ok.getValue());
}

TEST_F(MergeThrottlerTest, execute_queued_merge) {
    MergeThrottler& throttler(*_throttlers[1]);
    DummyStorageLink& topLink(*_topLinks[1]);
    DummyStorageLink& bottomLink(*_bottomLinks[1]);

    // Fill up all active merges and then 3 queued ones
    size_t maxPending = throttler_max_merges_pending(1);
    ASSERT_LT(maxPending, 100);
    for (size_t i = 0; i < maxPending + 3; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{1}, {uint16_t(5 + i)}, {uint16_t(7 + i)}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234, 1);
        cmd->setPriority(250 - i + 5);
        topLink.sendDown(cmd);
    }

    // Wait till we have maxPending replies and 3 queued
    topLink.waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(throttler, 3, _messageWaitTime);

    // Sneak in a higher priority message that is bound to be executed
    // on the given node
    {
        std::vector<MergeBucketCommand::Node> nodes({{1}, {0}});
        std::vector<uint16_t> chain({0});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0x1337)), nodes, 1234, 1, chain);
        cmd->setPriority(0);
        topLink.sendDown(cmd);
    }

    waitUntilMergeQueueIs(throttler, 4, _messageWaitTime);

    // Merge queue state should not be touched by worker thread now
    auto nextMerge = peek_throttler_queue_top(1);
    ASSERT_EQ(BucketId(32, 0x1337), nextMerge->getBucketId());

    auto fwd = topLink.getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Remove all the rest of the active merges
    while (!topLink.getReplies().empty()) {
        topLink.getAndRemoveMessage(MessageType::MERGEBUCKET);
    }

    // Free up a merge slot
    auto reply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
    reply->setResult(ReturnCode(ReturnCode::OK, "Celebrate good times come on"));
    topLink.sendDown(reply);

    topLink.waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    // Remove chain reply
    auto distReply = topLink.getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    ASSERT_EQ(static_cast<MergeBucketReply&>(*distReply).getResult().getResult(),
              ReturnCode::OK);

    waitUntilMergeQueueIs(throttler, 3, _messageWaitTime);
    bottomLink.waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    ASSERT_EQ(0, topLink.getNumCommands());
    ASSERT_EQ(0, topLink.getNumReplies());
    ASSERT_EQ(1, bottomLink.getNumCommands());

    // First queued merge should now have been registered and sent down
    auto cmd = bottomLink.getAndRemoveMessage(MessageType::MERGEBUCKET);

    ASSERT_EQ(static_cast<const MergeBucketCommand&>(*cmd).getBucketId(),
              static_cast<const MergeBucketCommand&>(*nextMerge).getBucketId());

    ASSERT_TRUE(static_cast<const MergeBucketCommand&>(*cmd).getNodes()
                == static_cast<const MergeBucketCommand&>(*nextMerge).getNodes());
}

TEST_F(MergeThrottlerTest, flush) {
    // Fill up all active merges and then 3 queued ones
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    for (size_t i = 0; i < maxPending + 3; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234, 1);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 3 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 3, _messageWaitTime);

    // Remove all forwarded commands
    uint32_t removed = _topLinks[0]->getRepliesOnce().size();
    ASSERT_GE(removed, 5);

    // Flush the storage link, triggering an abort of all commands
    // no matter what their current state is.
    _topLinks[0]->close();
    _topLinks[0]->flush();
    _topLinks[0]->waitForMessages(maxPending + 3 - removed, _messageWaitTime);

    while (!_topLinks[0]->getReplies().empty()) {
        StorageMessage::SP reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
        ASSERT_EQ(ReturnCode::ABORTED,
                  static_cast<const MergeBucketReply&>(*reply).getResult().getResult());
    }
    // NOTE: merges that have been immediately executed (i.e. not cycled)
    // on the node should _not_ be replied to, since they're not owned
    // by the throttler at that point in time
}

// If a node goes down and another node has a merge chained through it in
// its queue, the original node can receive a final chain hop forwarding
// it knows nothing about when it comes back up. If this is not handled
// properly, it will attempt to forward this node again with a bogus
// index. This should be implicitly handled by checking for a full node
TEST_F(MergeThrottlerTest, unseen_merge_with_node_in_chain) {
    std::vector<MergeBucketCommand::Node> nodes({{0}, {5}, {9}});
    std::vector<uint16_t> chain({0, 5, 9});
    auto cmd = std::make_shared<MergeBucketCommand>(
            makeDocumentBucket(BucketId(32, 0xdeadbeef)), nodes, 1234, 1, chain);


    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 9));
    _topLinks[0]->sendDown(cmd);

    // First, test that we get rejected when processing merge immediately
    // Should get a rejection in return
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    ASSERT_EQ(ReturnCode::REJECTED,
              dynamic_cast<const MergeBucketReply&>(*reply).getResult().getResult());

    // Second, test that we get rejected before queueing up. This is to
    // avoid a hypothetical deadlock scenario.
    // Fill up all active merges
    size_t maxPending = throttler_max_merges_pending(0);
    for (size_t i = 0; i < maxPending; ++i) {
        auto fillCmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234);
        _topLinks[0]->sendDown(fillCmd);
    }

    _topLinks[0]->sendDown(cmd);

    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    ASSERT_EQ(ReturnCode::REJECTED,
              dynamic_cast<const MergeBucketReply&>(*reply).getResult().getResult());
}

TEST_F(MergeThrottlerTest, merge_with_newer_cluster_state_flushes_outdated_queued){
    // Fill up all active merges and then 3 queued ones with the same
    // system state
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    std::vector<api::StorageMessage::Id> ids;
    for (size_t i = 0; i < maxPending + 3; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234, 1);
        ids.push_back(cmd->getMsgId());
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 3 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 3, _messageWaitTime);

    // Send down merge with newer system state
    {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0x12345678)), nodes, 1234, 2);
        ids.push_back(cmd->getMsgId());
        _topLinks[0]->sendDown(cmd);
    }

    // Queue should now be flushed with all messages being returned with
    // WRONG_DISTRIBUTION
    _topLinks[0]->waitForMessages(maxPending + 3, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 1, _messageWaitTime);

    for (int i = 0; i < 3; ++i) {
        StorageMessage::SP reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
        ASSERT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
                  ReturnCode::WRONG_DISTRIBUTION);
        ASSERT_EQ(1u, static_cast<MergeBucketReply&>(*reply).getClusterStateVersion());
        ASSERT_EQ(ids[maxPending + i], reply->getMsgId());
    }

    EXPECT_EQ(uint64_t(3), _throttlers[0]->getMetrics().chaining.failures.wrongdistribution.getValue());
}

TEST_F(MergeThrottlerTest, updated_cluster_state_flushes_outdated_queued) {
    // State is version 1. Send down several merges with state version 2.
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    std::vector<api::StorageMessage::Id> ids;
    for (size_t i = 0; i < maxPending + 3; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234, 2);
        ids.push_back(cmd->getMsgId());
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 4 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 3, _messageWaitTime);

    // Send down new system state (also set it explicitly)
    _servers[0]->setClusterState(lib::ClusterState("distributor:100 storage:100 version:3"));
    auto stateCmd = std::make_shared<api::SetSystemStateCommand>(
            lib::ClusterState("distributor:100 storage:100 version:3"));
    _topLinks[0]->sendDown(stateCmd);

    // Queue should now be flushed with all being replied to with WRONG_DISTRIBUTION
    waitUntilMergeQueueIs(*_throttlers[0], 0, _messageWaitTime);
    _topLinks[0]->waitForMessages(maxPending + 3, 5);

    for (int i = 0; i < 3; ++i) {
        StorageMessage::SP reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
        ASSERT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
                  ReturnCode::WRONG_DISTRIBUTION);
        ASSERT_EQ(2u, static_cast<MergeBucketReply&>(*reply).getClusterStateVersion());
        ASSERT_EQ(ids[maxPending + i], reply->getMsgId());
    }

    EXPECT_EQ(uint64_t(3), _throttlers[0]->getMetrics().chaining.failures.wrongdistribution.getValue());
}

// TODO remove functionality and test
TEST_F(MergeThrottlerTest, legacy_42_merges_do_not_trigger_flush) {
    // Fill up all active merges and then 1 queued one
    size_t maxPending = throttler_max_merges_pending(0);
    ASSERT_LT(maxPending, 100);
    for (size_t i = 0; i < maxPending + 1; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00baa00 + i)), nodes, 1234, 1);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and 1 queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], 1, _messageWaitTime);

    auto fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Remove all the rest of the active merges
    while (!_topLinks[0]->getReplies().empty()) {
        _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    }

    // Send down a merge with a cluster state version of 0, which should
    // be ignored and queued as usual
    {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xbaaadbed)), nodes, 1234, 0);
        _topLinks[0]->sendDown(cmd);
    }

    waitUntilMergeQueueIs(*_throttlers[0], 2, _messageWaitTime);

    ASSERT_EQ(0, _topLinks[0]->getNumCommands());
    ASSERT_EQ(0, _topLinks[0]->getNumReplies());

    EXPECT_EQ(uint64_t(0), _throttlers[0]->getMetrics().local.failures.wrongdistribution.getValue());
}

// Test that a merge that arrive with a state version that is less than
// that of the node is rejected immediately
TEST_F(MergeThrottlerTest, outdated_cluster_state_merges_are_rejected_on_arrival) {
    _servers[0]->setClusterState(lib::ClusterState("distributor:100 storage:100 version:10"));

    // Send down a merge with a cluster state version of 9, which should
    // be rejected
    {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xfeef00)), nodes, 1234, 9);
        _topLinks[0]->sendDown(cmd);
    }

    _topLinks[0]->waitForMessages(1, _messageWaitTime);

    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(static_cast<MergeBucketReply&>(*reply).getResult().getResult(),
              ReturnCode::WRONG_DISTRIBUTION);

    EXPECT_EQ(uint64_t(1), _throttlers[0]->getMetrics().chaining.failures.wrongdistribution.getValue());
}

// Test erroneous case where node receives merge where the merge does
// not exist in the state, but it exists in the chain without the chain
// being full. This is something that shouldn't happen, but must still
// not crash the node
TEST_F(MergeThrottlerTest, unknown_merge_with_self_in_chain) {
    BucketId bid(32, 0xbadbed);

    std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
    std::vector<uint16_t> chain({0});
    auto cmd = std::make_shared<MergeBucketCommand>(makeDocumentBucket(bid), nodes, 1234, 1, chain);

    cmd->setAddress(StorageMessageAddress::create(&_storage, lib::NodeType::STORAGE, 1));
    _topLinks[0]->sendDown(cmd);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);

    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);

    EXPECT_EQ(ReturnCode::REJECTED,
              static_cast<MergeBucketReply&>(*reply).getResult().getResult());
}

TEST_F(MergeThrottlerTest, busy_returned_on_full_queue_for_merges_sent_from_distributors) {
    size_t maxPending = throttler_max_merges_pending(0);
    size_t maxQueue = _throttlers[0]->getMaxQueueSize();
    ASSERT_EQ(20, maxQueue);
    ASSERT_LT(maxPending, 100);

    EXPECT_EQ(_throttlers[0]->getMetrics().active_window_size.getLast(), 0);
    EXPECT_EQ(_throttlers[0]->getMetrics().queueSize.getLast(), 0);

    for (size_t i = 0; i < maxPending + maxQueue; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        // No chain set, i.e. merge command is freshly squeezed from a distributor.
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00000 + i)), nodes, 1234, 1);
        _topLinks[0]->sendDown(cmd);
    }

    // Wait till we have maxPending replies and maxQueue queued
    _topLinks[0]->waitForMessages(maxPending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[0], maxQueue, _messageWaitTime);
    EXPECT_EQ(_throttlers[0]->getMetrics().active_window_size.getLast(), maxPending);
    EXPECT_EQ(maxQueue, _throttlers[0]->getMetrics().queueSize.getMaximum());

    // Clear all forwarded merges
    _topLinks[0]->getRepliesOnce();
    // Send down another merge which should be immediately busy-returned
    {
        std::vector<MergeBucketCommand::Node> nodes({{0}, {1}, {2}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf000baaa)), nodes, 1234, 1);
        _topLinks[0]->sendDown(cmd);
    }
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);

    EXPECT_EQ(BucketId(32, 0xf000baaa),
              static_cast<MergeBucketReply&>(*reply).getBucketId());

    EXPECT_EQ(ReturnCode::BUSY,
              static_cast<MergeBucketReply&>(*reply).getResult().getResult());

    EXPECT_EQ(0, _throttlers[0]->getMetrics().chaining.failures.busy.getValue());
    EXPECT_EQ(1, _throttlers[0]->getMetrics().local.failures.busy.getValue());
}

void
MergeThrottlerTest::receive_chained_merge_with_full_queue(bool disable_queue_limits, bool unordered_fwd)
{
    // Note: uses node with index 1 to not be the first node in chain
    _throttlers[1]->set_disable_queue_limits_for_chained_merges_locking(disable_queue_limits);
    size_t max_pending = throttler_max_merges_pending(1);
    size_t max_enqueued = _throttlers[1]->getMaxQueueSize();
    for (size_t i = 0; i < max_pending + max_enqueued; ++i) {
        std::vector<MergeBucketCommand::Node> nodes({{1}, {2}, {3}});
        // No chain set, i.e. merge command is freshly squeezed from a distributor.
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf00000 + i)), nodes, 1234, 1);
        _topLinks[1]->sendDown(cmd);
    }
    _topLinks[1]->waitForMessages(max_pending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[1], max_enqueued, _messageWaitTime);

    // Clear all forwarded merges
    _topLinks[1]->getRepliesOnce();
    // Send down another merge with non-empty chain. It should _not_ be busy bounced
    // (if limits disabled) as it has already been accepted into another node's merge window.
    {
        std::vector<MergeBucketCommand::Node> nodes({{2}, {1}, {0}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf000baaa)), nodes, 1234, 1);
        if (!unordered_fwd) {
            cmd->setChain(std::vector<uint16_t>({0})); // Forwarded from node 0
        } else {
            cmd->setChain(std::vector<uint16_t>({2})); // Forwarded from node 2, i.e. _not_ the lowest index
        }
        cmd->set_use_unordered_forwarding(unordered_fwd);
        _topLinks[1]->sendDown(cmd);
    }
}

TEST_F(MergeThrottlerTest, forwarded_merges_not_busy_bounced_even_if_queue_is_full_if_chained_limits_disabled) {
    ASSERT_NO_FATAL_FAILURE(receive_chained_merge_with_full_queue(true));
    size_t max_enqueued = _throttlers[1]->getMaxQueueSize();
    waitUntilMergeQueueIs(*_throttlers[1], max_enqueued + 1, _messageWaitTime);
}

TEST_F(MergeThrottlerTest, forwarded_merges_busy_bounced_if_queue_is_full_and_chained_limits_enforced) {
    ASSERT_NO_FATAL_FAILURE(receive_chained_merge_with_full_queue(false));

    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    auto reply = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(ReturnCode::BUSY, static_cast<MergeBucketReply&>(*reply).getResult().getResult());
}

TEST_F(MergeThrottlerTest, forwarded_merge_has_higher_pri_when_chain_limits_disabled) {
    ASSERT_NO_FATAL_FAILURE(receive_chained_merge_with_full_queue(true));
    size_t max_enqueued = _throttlers[1]->getMaxQueueSize();
    waitUntilMergeQueueIs(*_throttlers[1], max_enqueued + 1, _messageWaitTime);

    auto highest_pri_merge = peek_throttler_queue_top(1);
    EXPECT_FALSE(highest_pri_merge->getChain().empty()); // Should be the forwarded merge
}

TEST_F(MergeThrottlerTest, forwarded_unordered_merge_is_directly_accepted_into_active_window) {
    // Unordered forwarding is orthogonal to disabled chain limits config, so we implicitly test that too.
    ASSERT_NO_FATAL_FAILURE(receive_chained_merge_with_full_queue(true, true));

    // Unordered merge is immediately forwarded to the next node
    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    auto fwd = std::dynamic_pointer_cast<api::MergeBucketCommand>(
            _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET));
    ASSERT_TRUE(fwd);
    EXPECT_TRUE(fwd->use_unordered_forwarding());
    EXPECT_EQ(fwd->getChain(), std::vector<uint16_t>({2, 1}));
}

TEST_F(MergeThrottlerTest, non_forwarded_unordered_merge_is_enqueued_if_active_window_full)
{
    fill_throttler_queue_with_n_commands(1, 0); // Fill active window entirely
    {
        std::vector<MergeBucketCommand::Node> nodes({{1}, {2}, {0}});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xf000baaa)), nodes, 1234, 1);
        cmd->set_use_unordered_forwarding(true);
        _topLinks[1]->sendDown(cmd);
    }
    waitUntilMergeQueueIs(*_throttlers[1], 1, _messageWaitTime); // Should be in queue, not active window
}

TEST_F(MergeThrottlerTest, broken_cycle) {
    std::vector<MergeBucketCommand::Node> nodes({1, 0, 2});
    {
        std::vector<uint16_t> chain;
        chain.push_back(0);
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xfeef00)), nodes, 1234, 1, chain);
        _topLinks[1]->sendDown(cmd);
    }

    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    auto fwd = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    ASSERT_EQ(2, fwd->getAddress()->getIndex());

    // Send cycled merge which will be executed
    {
        std::vector<uint16_t> chain({0, 1, 2});
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xfeef00)), nodes, 1234, 1, chain);
        _topLinks[1]->sendDown(cmd);
    }

    _bottomLinks[1]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    auto cycled = _bottomLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    // Now, node 2 goes down, auto sending back a failed merge
    auto nodeDownReply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*fwd));
    nodeDownReply->setResult(ReturnCode(ReturnCode::NOT_CONNECTED, "Node went sightseeing"));

    _topLinks[1]->sendDown(nodeDownReply);
    // Merge reply also arrives from persistence
    auto persistenceReply = std::make_shared<MergeBucketReply>(dynamic_cast<const MergeBucketCommand&>(*cycled));
    persistenceReply->setResult(ReturnCode(ReturnCode::ABORTED, "Oh dear"));
    _bottomLinks[1]->sendUp(persistenceReply);

    // Should now be two replies from node 1, one to node 2 and one to node 0
    // since we must handle broken chains
    _topLinks[1]->waitForMessages(2, _messageWaitTime);
    // Unwind reply shares the result of the persistence reply
    for (int i = 0; i < 2; ++i) {
        StorageMessage::SP reply = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
        ASSERT_EQ(api::ReturnCode(ReturnCode::ABORTED, "Oh dear"),
                  static_cast<MergeBucketReply&>(*reply).getResult());
    }

    // Make sure it has been removed from the internal state so we can
    // send new merges for the bucket
    {
        std::vector<uint16_t> chain;
        chain.push_back(0);
        auto cmd = std::make_shared<MergeBucketCommand>(
                makeDocumentBucket(BucketId(32, 0xfeef00)), nodes, 1234, 1, chain);
        _topLinks[1]->sendDown(cmd);
    }

    _topLinks[1]->waitForMessage(MessageType::MERGEBUCKET, 5);
    fwd = _topLinks[1]->getAndRemoveMessage(MessageType::MERGEBUCKET);
    ASSERT_EQ(2, fwd->getAddress()->getIndex());
}

void
MergeThrottlerTest::send_and_expect_reply(
        const std::shared_ptr<api::StorageMessage>& msg,
        const api::MessageType& expectedReplyType,
        api::ReturnCode::Result expectedResultCode)
{
    _topLinks[0]->sendDown(msg);
    _topLinks[0]->waitForMessage(expectedReplyType, _messageWaitTime);
    auto reply = _topLinks[0]->getAndRemoveMessage(expectedReplyType);
    auto& storageReply = dynamic_cast<api::StorageReply&>(*reply);
    ASSERT_EQ(expectedResultCode, storageReply.getResult().getResult());
}

std::shared_ptr<api::StorageMessage>
MergeThrottlerTest::send_and_expect_forwarding(const std::shared_ptr<api::StorageMessage>& msg)
{
    _topLinks[0]->sendDown(msg);
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    return _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
}

TEST_F(MergeThrottlerTest, get_bucket_diff_command_not_in_active_set_is_rejected) {
    document::BucketId bucket(16, 1234);
    std::vector<api::GetBucketDiffCommand::Node> nodes;
    auto getDiffCmd = std::make_shared<api::GetBucketDiffCommand>(
            makeDocumentBucket(bucket), nodes, api::Timestamp(1234));

    ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
            getDiffCmd,
            api::MessageType::GETBUCKETDIFF_REPLY,
            api::ReturnCode::ABORTED));
    ASSERT_EQ(0, _bottomLinks[0]->getNumCommands());
}

TEST_F(MergeThrottlerTest, apply_bucket_diff_command_not_in_active_set_is_rejected) {
    document::BucketId bucket(16, 1234);
    std::vector<api::GetBucketDiffCommand::Node> nodes;
    auto applyDiffCmd = std::make_shared<api::ApplyBucketDiffCommand>(makeDocumentBucket(bucket), nodes);

    ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
            applyDiffCmd,
            api::MessageType::APPLYBUCKETDIFF_REPLY,
            api::ReturnCode::ABORTED));
    ASSERT_EQ(0, _bottomLinks[0]->getNumCommands());
}

api::MergeBucketCommand::SP
MergeThrottlerTest::sendMerge(const MergeBuilder& builder)
{
    api::MergeBucketCommand::SP cmd(builder.create());
    _topLinks[builder._nodes[0]]->sendDown(cmd);
    return cmd;
}

TEST_F(MergeThrottlerTest, new_cluster_state_aborts_all_outdated_active_merges) {
    document::BucketId bucket(16, 6789);
    _throttlers[0]->getThrottlePolicy().setMaxPendingCount(1);

    // Merge will be forwarded (i.e. active).
    sendMerge(MergeBuilder(bucket).clusterStateVersion(10));
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);
    auto fwd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);

    _topLinks[0]->sendDown(makeSystemStateCmd("version:11 distributor:100 storage:100"));
    // Cannot send reply until we're unwinding
    ASSERT_EQ(0, _topLinks[0]->getNumReplies());

    // Trying to diff the bucket should now fail
    {
        auto getDiffCmd = std::make_shared<api::GetBucketDiffCommand>(
                makeDocumentBucket(bucket), std::vector<api::GetBucketDiffCommand::Node>(), api::Timestamp(123));

        ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
                getDiffCmd,
                api::MessageType::GETBUCKETDIFF_REPLY,
                api::ReturnCode::ABORTED));
    }
}

TEST_F(MergeThrottlerTest, backpressure_busy_bounces_merges_for_configured_duration) {
    _servers[0]->getClock().setAbsoluteTimeInSeconds(1000);

    EXPECT_FALSE(_throttlers[0]->backpressure_mode_active());
    _throttlers[0]->apply_timed_backpressure();
    EXPECT_TRUE(_throttlers[0]->backpressure_mode_active());
    document::BucketId bucket(16, 6789);

    EXPECT_EQ(0, _throttlers[0]->getMetrics().bounced_due_to_back_pressure.getValue());
    EXPECT_EQ(uint64_t(0), _throttlers[0]->getMetrics().local.failures.busy.getValue());

    ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
            MergeBuilder(bucket).create(),
            api::MessageType::MERGEBUCKET_REPLY,
            api::ReturnCode::BUSY));

    EXPECT_EQ(1, _throttlers[0]->getMetrics().bounced_due_to_back_pressure.getValue());
    EXPECT_EQ(1, _throttlers[0]->getMetrics().local.failures.busy.getValue());

    _servers[0]->getClock().addSecondsToTime(15); // Test-config has duration set to 15 seconds
    // Backpressure has now been lifted. New merges should be forwarded
    // to next node in chain as expected instead of being bounced with a reply.
    sendMerge(MergeBuilder(bucket));
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    EXPECT_FALSE(_throttlers[0]->backpressure_mode_active());
    EXPECT_EQ(1, _throttlers[0]->getMetrics().bounced_due_to_back_pressure.getValue());
}

TEST_F(MergeThrottlerTest, source_only_merges_are_not_affected_by_backpressure) {
    _servers[2]->getClock().setAbsoluteTimeInSeconds(1000);
    _throttlers[2]->apply_timed_backpressure();
    document::BucketId bucket(16, 6789);

    _topLinks[2]->sendDown(MergeBuilder(bucket).chain(0, 1).source_only(2).create());
    _topLinks[2]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime);

    EXPECT_EQ(0, _throttlers[0]->getMetrics().bounced_due_to_back_pressure.getValue());
}

void MergeThrottlerTest::fill_throttler_queue_with_n_commands(uint16_t throttler_index, size_t queued_count) {
    size_t max_pending = throttler_max_merges_pending(throttler_index);
    for (size_t i = 0; i < max_pending + queued_count; ++i) {
        _topLinks[throttler_index]->sendDown(MergeBuilder(document::BucketId(16, i))
                .nodes(throttler_index, throttler_index + 1)
                .create());
    }
    // Wait till we have max_pending merge forwards and queued_count enqueued.
    _topLinks[throttler_index]->waitForMessages(max_pending, _messageWaitTime);
    waitUntilMergeQueueIs(*_throttlers[throttler_index], queued_count, _messageWaitTime);
}

TEST_F(MergeThrottlerTest, backpressure_evicts_all_queued_merges) {
    _servers[0]->getClock().setAbsoluteTimeInSeconds(1000);

    fill_throttler_queue_with_n_commands(0, 1);
    _topLinks[0]->getRepliesOnce(); // Clear all forwarded merges
    _throttlers[0]->apply_timed_backpressure();

    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET_REPLY, _messageWaitTime);
    auto reply = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET_REPLY);
    EXPECT_EQ(ReturnCode::BUSY, dynamic_cast<const MergeBucketReply&>(*reply).getResult().getResult());
}

TEST_F(MergeThrottlerTest, exceeding_memory_soft_limit_rejects_merges_even_with_available_active_window_slots) {
    ASSERT_GT(throttler_max_merges_pending(0), 1); // Sanity check for the test itself

    throttler(0).set_max_merge_memory_usage_bytes_locking(10_Mi);

    ASSERT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 0);

    std::shared_ptr<api::StorageMessage> fwd_cmd;
    ASSERT_NO_FATAL_FAILURE(fwd_cmd = send_and_expect_forwarding(
            MergeBuilder(document::BucketId(16, 0)).nodes(0, 1, 2).memory_usage(5_Mi).create()));

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 5_Mi);

    // Accepting this merge would exceed memory limits. It is sent as part of a forwarded unordered
    // merge and can therefore NOT be enqueued; it must be bounced immediately.
    ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
            MergeBuilder(document::BucketId(16, 1))
                    .nodes(2, 1, 0).chain(2, 1).unordered(true)
                    .memory_usage(8_Mi).create(),
            MessageType::MERGEBUCKET_REPLY, ReturnCode::BUSY));

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 5_Mi); // Unchanged

    // Fail the forwarded merge. This shall immediately free up the memory usage, allowing a new merge in.
    auto fwd_reply = dynamic_cast<api::MergeBucketCommand&>(*fwd_cmd).makeReply();
    fwd_reply->setResult(ReturnCode(ReturnCode::ABORTED, "node stumbled into a ravine"));

    ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
            std::shared_ptr<api::StorageReply>(std::move(fwd_reply)),
            MessageType::MERGEBUCKET_REPLY, ReturnCode::ABORTED)); // Unwind reply for failed merge

    ASSERT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 0);

    // New merge is accepted and forwarded
    ASSERT_NO_FATAL_FAILURE(send_and_expect_forwarding(
            MergeBuilder(document::BucketId(16, 2)).nodes(0, 1, 2).unordered(true).memory_usage(9_Mi).create()));

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 9_Mi);
}

TEST_F(MergeThrottlerTest, exceeding_memory_soft_limit_can_enqueue_unordered_merge_sent_directly_from_distributor) {
    throttler(0).set_max_merge_memory_usage_bytes_locking(10_Mi);

    ASSERT_NO_FATAL_FAILURE(send_and_expect_forwarding(
            MergeBuilder(document::BucketId(16, 0)).nodes(0, 1, 2).memory_usage(5_Mi).create()));

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 5_Mi);

    // Accepting this merge would exceed memory limits. It is sent directly from a distributor and
    // can therefore be enqueued.
    _topLinks[0]->sendDown(MergeBuilder(document::BucketId(16, 1)).nodes(0, 1, 2).unordered(true).memory_usage(8_Mi).create());
    waitUntilMergeQueueIs(throttler(0), 1, _messageWaitTime); // Should end up in queue

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 5_Mi); // Unchanged
}

TEST_F(MergeThrottlerTest, at_least_one_merge_is_accepted_even_if_exceeding_memory_soft_limit) {
    throttler(0).set_max_merge_memory_usage_bytes_locking(5_Mi);

    _topLinks[0]->sendDown(MergeBuilder(document::BucketId(16, 0)).nodes(0, 1, 2).unordered(true).memory_usage(100_Mi).create());
    _topLinks[0]->waitForMessage(MessageType::MERGEBUCKET, _messageWaitTime); // Forwarded, _not_ bounced

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 100_Mi);
}

TEST_F(MergeThrottlerTest, queued_merges_are_not_counted_towards_memory_usage) {
    // Our utility function for filling queues uses bucket IDs {16, x} where x is increasing
    // from 0 to the max pending. Ensure we don't accidentally overlap with the bucket ID we
    // send for below in the test code.
    ASSERT_LT(throttler_max_merges_pending(0), 1000);

    throttler(0).set_max_merge_memory_usage_bytes_locking(50_Mi);
    // Fill up active window on node 0. Note: these merges do not have any associated memory cost.
    fill_throttler_queue_with_n_commands(0, 0);

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 0_Mi);

    _topLinks[0]->sendDown(MergeBuilder(document::BucketId(16, 1000)).nodes(0, 1, 2).unordered(true).memory_usage(10_Mi).create());
    waitUntilMergeQueueIs(throttler(0), 1, _messageWaitTime); // Should end up in queue

    EXPECT_EQ(throttler(0).getMetrics().estimated_merge_memory_usage.getLast(), 0_Mi);
}

TEST_F(MergeThrottlerTest, enqueued_merge_not_started_if_insufficient_memory_available) {
    // See `queued_merges_are_not_counted_towards_memory_usage` test for magic number rationale
    const auto max_pending = throttler_max_merges_pending(0);
    ASSERT_LT(max_pending, 1000);
    ASSERT_GT(max_pending, 1);
    throttler(0).set_max_merge_memory_usage_bytes_locking(10_Mi);

    // Fill up entire active window and enqueue a single merge
    fill_throttler_queue_with_n_commands(0, 0);
    _topLinks[0]->sendDown(MergeBuilder(document::BucketId(16, 1000)).nodes(0, 1, 2).unordered(true).memory_usage(11_Mi).create());
    waitUntilMergeQueueIs(throttler(0), 1, _messageWaitTime); // Should end up in queue

    // Drain all active merges. As long as we have other active merges, the enqueued merge should not
    // be allowed through since it's too large. Eventually it will hit the "at least one merge must
    // be allowed at any time regardless of size" exception and is dequeued.
    for (uint32_t i = 0; i < max_pending; ++i) {
        auto fwd_cmd = _topLinks[0]->getAndRemoveMessage(MessageType::MERGEBUCKET);
        auto fwd_reply = dynamic_cast<api::MergeBucketCommand&>(*fwd_cmd).makeReply();

        ASSERT_NO_FATAL_FAILURE(send_and_expect_reply(
                std::shared_ptr<api::StorageReply>(std::move(fwd_reply)),
                MessageType::MERGEBUCKET_REPLY, ReturnCode::OK)); // Unwind reply for completed merge

        if (i < max_pending - 1) {
            // Merge should still be in the queue, as it requires 11 MiB, and we only have 10 MiB.
            // It will eventually be executed when the window is empty (see below).
            waitUntilMergeQueueIs(throttler(0), 1, _messageWaitTime);
        }
    }
    // We've freed up the entire send window, so the over-sized merge can finally squeeze through.
    waitUntilMergeQueueIs(throttler(0), 0, _messageWaitTime);
}

namespace {

vespalib::HwInfo make_mem_info(uint64_t mem_size) {
    return {{0, false, false}, {mem_size}, {1}};
}

}

TEST_F(MergeThrottlerTest, memory_limit_can_be_auto_deduced_from_hw_info) {
    StorServerConfigBuilder cfg(*default_server_config());
    auto& cfg_limit = cfg.mergeThrottlingMemoryLimit;
    auto& mt = throttler(0);

    // Enable auto-deduction of limits
    cfg_limit.maxUsageBytes = 0;

    cfg_limit.autoLowerBoundBytes = 100'000;
    cfg_limit.autoUpperBoundBytes = 750'000;
    cfg_limit.autoPhysMemScaleFactor = 0.5;

    mt.set_hw_info_locking(make_mem_info(1'000'000));
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 500'000);
    EXPECT_EQ(throttler(0).getMetrics().merge_memory_limit.getLast(), 500'000);

    cfg_limit.autoPhysMemScaleFactor = 0.75;
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 750'000);
    EXPECT_EQ(throttler(0).getMetrics().merge_memory_limit.getLast(), 750'000);

    cfg_limit.autoPhysMemScaleFactor = 0.25;
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 250'000);

    // Min-capped
    cfg_limit.autoPhysMemScaleFactor = 0.05;
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 100'000);

    // Max-capped
    cfg_limit.autoPhysMemScaleFactor = 0.90;
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 750'000);
}

TEST_F(MergeThrottlerTest, memory_limit_can_be_set_explicitly) {
    StorServerConfigBuilder cfg(*default_server_config());
    auto& cfg_limit = cfg.mergeThrottlingMemoryLimit;
    auto& mt = throttler(0);

    cfg_limit.maxUsageBytes = 1'234'567;
    mt.set_hw_info_locking(make_mem_info(1'000'000));
    mt.on_configure(cfg);
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 1'234'567);
    EXPECT_EQ(throttler(0).getMetrics().merge_memory_limit.getLast(), 1'234'567);
}

TEST_F(MergeThrottlerTest, memory_limit_can_be_set_to_unlimited) {
    StorServerConfigBuilder cfg(*default_server_config());
    auto& cfg_limit = cfg.mergeThrottlingMemoryLimit;
    auto& mt = throttler(0);

    cfg_limit.maxUsageBytes = -1;
    mt.set_hw_info_locking(make_mem_info(1'000'000));
    mt.on_configure(cfg);
    // Zero implies infinity
    EXPECT_EQ(mt.max_merge_memory_usage_bytes_locking(), 0);
    EXPECT_EQ(throttler(0).getMetrics().merge_memory_limit.getLast(), 0);
}

// TODO test message queue aborting (use rendezvous functionality--make guard)

} // namespace storage
