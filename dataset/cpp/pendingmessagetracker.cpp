// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "pendingmessagetracker.h"
#include <vespa/storageframework/generic/clock/clock.h>
#include <vespa/vespalib/stllike/asciistream.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <map>

#include <vespa/log/log.h>
LOG_SETUP(".pendingmessages");

namespace storage::distributor {

PendingMessageTracker::PendingMessageTracker(framework::ComponentRegister& cr, uint32_t stripe_index)
    : framework::HtmlStatusReporter(vespalib::make_string("pendingmessages%u", stripe_index),
                                    vespalib::make_string("Pending messages to storage nodes (stripe %u)", stripe_index)),
      _component(cr, "pendingmessagetracker"),
      _nodeInfo(_component.getClock()),
      _nodeBusyDuration(60s),
      _deferred_read_tasks(),
      _trackTime(false),
      _lock()
{
    _component.registerStatusPage(*this);
}

PendingMessageTracker::~PendingMessageTracker() = default;

PendingMessageTracker::MessageEntry::MessageEntry(TimePoint timeStamp_, uint32_t msgType_, uint32_t priority_,
                                                  uint64_t msgId_, document::Bucket bucket_, uint16_t nodeIdx_) noexcept
    : timeStamp(timeStamp_),
      msgType(msgType_),
      priority(priority_),
      msgId(msgId_),
      bucket(bucket_),
      nodeIdx(nodeIdx_)
{ }

vespalib::string
PendingMessageTracker::MessageEntry::toHtml() const {
    vespalib::asciistream ss;
    ss << "<li><i>Node " << nodeIdx << "</i>: "
       << "<b>" << vespalib::to_string(timeStamp) << "</b> "
       << api::MessageType::get(api::MessageType::Id(msgType)).getName()
       << "(" <<  bucket.getBucketId() <<  ", priority=" << priority << ")</li>\n";
    return ss.str();
}

PendingMessageTracker::TimePoint
PendingMessageTracker::currentTime() const
{
    return _component.getClock().getSystemTime();
}

namespace {

template <typename Pair>
struct PairAsRange {
    Pair _pair;
    explicit PairAsRange(Pair pair) : _pair(std::move(pair)) {}

    auto begin() { return _pair.first; }
    auto end() { return _pair.second; }
    auto begin() const { return _pair.first; }
    auto end() const { return _pair.second; }
};

template <typename Pair>
PairAsRange<Pair>
pairAsRange(Pair pair)
{
    return PairAsRange<Pair>(std::move(pair));
}

document::Bucket
getBucket(const api::StorageMessage & msg) {
    return (msg.getType() != api::MessageType::REQUESTBUCKETINFO)
           ? msg.getBucket()
           : document::Bucket(msg.getBucket().getBucketSpace(), dynamic_cast<const api::RequestBucketInfoCommand&>(msg).super_bucket_id());
}

}

std::vector<uint64_t>
PendingMessageTracker::clearMessagesForNode(uint16_t node)
{
    std::lock_guard guard(_lock);
    auto& idx = boost::multi_index::get<IndexByNodeAndBucket>(_messages);
    auto range = pairAsRange(idx.equal_range(boost::make_tuple(node)));

    std::vector<uint64_t> erasedIds;
    for (auto& entry : range) {
        erasedIds.push_back(entry.msgId);
    }
    idx.erase(std::begin(range), std::end(range));

    _nodeInfo.clearPending(node);
    return erasedIds;
}

void
PendingMessageTracker::enumerate_matching_pending_bucket_ops(
        const std::function<bool(const document::Bucket&)>& bucket_predicate,
        const std::function<void(uint64_t)>& msg_id_callback) const
{
    std::lock_guard guard(_lock);
    const auto& idx = boost::multi_index::get<IndexByBucketAndType>(_messages);
    auto iter = idx.begin();
    const auto last = idx.end();
    while (iter != last) {
        const auto check_bucket = iter->bucket;
        const bool match = bucket_predicate(check_bucket);
        do {
            if (match) {
                msg_id_callback(iter->msgId);
            }
            ++iter;
        } while ((iter != last) && (iter->bucket == check_bucket));
    }
}

void
PendingMessageTracker::insert(const std::shared_ptr<api::StorageMessage>& msg)
{
    if (msg->getAddress()) {
        // TODO STRIPE reevaluate if getBucket() on RequestBucketInfo msgs should transparently return superbucket..!
        document::Bucket bucket = getBucket(*msg);
        {
            // We will not start tracking time until we have been asked for html at least once.
            // Time tracking is only used for presenting pending messages for debugging.
            TimePoint now = (_trackTime.load(std::memory_order_relaxed)) ? currentTime() : TimePoint();
            std::lock_guard guard(_lock);
            _messages.emplace(now, msg->getType().getId(), msg->getPriority(), msg->getMsgId(),
                              bucket, msg->getAddress()->getIndex());

            _nodeInfo.incPending(msg->getAddress()->getIndex());
        }

        LOG(debug, "Sending message %s with id %" PRIu64 " to %s",
            msg->toString().c_str(), msg->getMsgId(), msg->getAddress()->toString().c_str());
    }
}

document::Bucket
PendingMessageTracker::reply(const api::StorageReply& r)
{
    document::Bucket bucket;
    LOG(debug, "Got reply: %s", r.toString().c_str());
    uint64_t msgId = r.getMsgId();

    std::unique_lock guard(_lock);
    auto& msgs = boost::multi_index::get<IndexByMessageId>(_messages);
    auto iter = msgs.find(msgId);
    if (iter != msgs.end()) {
        bucket = iter->bucket;
        _nodeInfo.decPending(r.getAddress()->getIndex());
        api::ReturnCode::Result code = r.getResult().getResult();
        if (code == api::ReturnCode::BUSY || code == api::ReturnCode::TIMEOUT) {
            _nodeInfo.setBusy(r.getAddress()->getIndex(), _nodeBusyDuration);
        }
        msgs.erase(msgId);
        auto deferred_tasks = get_deferred_ops_if_bucket_writes_drained(bucket);
        // Deferred tasks may try to send messages, which in turn will invoke the PendingMessageTracker.
        // To avoid deadlocking, we run the tasks outside the lock.
        // TODO remove locking entirely... Only exists for status pages!
        guard.unlock();
        // We expect this to be "effectively noexcept", i.e. any tasks throwing an
        // exception will end up nuking the distributor process from the unwind.
        for (auto& task : deferred_tasks) {
            task->run(TaskRunState::OK);
        }
        LOG(debug, "Erased message with id %" PRIu64 " for bucket %s", msgId, bucket.toString().c_str());
    }

    return bucket;
}

namespace {

template <typename Range>
bool is_empty_range(const Range& range) noexcept {
    return (range.first == range.second);
}

template <typename Range>
bool range_is_empty_or_only_has_read_ops(const Range& range) noexcept {
    if (is_empty_range(range)) {
        return true;
    }
    // Number of ops to check is expected to be small in the common case
    for (auto iter = range.first; iter != range.second; ++iter) {
        switch (iter->msgType) {
        case api::MessageType::GET_ID:
        case api::MessageType::STAT_ID:
        case api::MessageType::VISITOR_CREATE_ID:
        case api::MessageType::VISITOR_DESTROY_ID:
            continue;
        default:
            return false;
        }
    }
    return true;
}

}

bool
PendingMessageTracker::bucket_has_no_pending_write_ops(const document::Bucket& bucket) const noexcept
{
    auto& bucket_idx = boost::multi_index::get<IndexByBucketAndType>(_messages);
    auto pending_tasks_for_bucket = bucket_idx.equal_range(bucket);
    return range_is_empty_or_only_has_read_ops(pending_tasks_for_bucket);
}

std::vector<std::unique_ptr<DeferredTask>>
PendingMessageTracker::get_deferred_ops_if_bucket_writes_drained(const document::Bucket& bucket)
{
    if (_deferred_read_tasks.empty()) {
        return {};
    }
    std::vector<std::unique_ptr<DeferredTask>> tasks;
    if (bucket_has_no_pending_write_ops(bucket)) {
        auto waiting_tasks = _deferred_read_tasks.equal_range(bucket);
        for (auto task_iter = waiting_tasks.first; task_iter != waiting_tasks.second; ++task_iter) {
            tasks.emplace_back(std::move(task_iter->second));
        }
        _deferred_read_tasks.erase(waiting_tasks.first, waiting_tasks.second);
    }
    return tasks;
}

void
PendingMessageTracker::run_once_no_pending_for_bucket(const document::Bucket& bucket, std::unique_ptr<DeferredTask> task)
{
    std::unique_lock guard(_lock);
    if (bucket_has_no_pending_write_ops(bucket)) {
        guard.unlock(); // Must not be held whilst running task, or else recursive sends will deadlock.
        task->run(TaskRunState::OK); // Nothing pending, run immediately.
    } else {
        _deferred_read_tasks.emplace(bucket, std::move(task));
    }
}

void
PendingMessageTracker::abort_deferred_tasks()
{
    std::lock_guard guard(_lock);
    for (auto& task : _deferred_read_tasks) {
        task.second->run(TaskRunState::Aborted);
    }
}

namespace {

template <typename Range>
void
runCheckerOnRange(PendingMessageTracker::Checker& checker, const Range& range)
{
    for (auto& e : range) {
        if (!checker.check(e.msgType, e.nodeIdx, e.priority)) {
            break;
        }
    }
}

}

void
PendingMessageTracker::checkPendingMessages(uint16_t node, const document::Bucket& bucket, Checker& checker) const
{
    std::lock_guard guard(_lock);
    const auto& msgs = boost::multi_index::get<IndexByNodeAndBucket>(_messages);

    auto range = pairAsRange(msgs.equal_range(boost::make_tuple(node, bucket)));
    runCheckerOnRange(checker, range);
}

void
PendingMessageTracker::checkPendingMessages(const document::Bucket& bucket, Checker& checker) const
{
    std::lock_guard guard(_lock);
    const auto& msgs = boost::multi_index::get<IndexByBucketAndType>(_messages);

    auto range = pairAsRange(msgs.equal_range(boost::make_tuple(bucket)));
    runCheckerOnRange(checker, range);
}

bool
PendingMessageTracker::hasPendingMessage(uint16_t node, const document::Bucket& bucket, uint32_t messageType) const
{
    std::lock_guard guard(_lock);
    const auto& msgs = boost::multi_index::get<IndexByNodeAndBucket>(_messages);

    auto range = msgs.equal_range(boost::make_tuple(node, bucket, messageType));
    return (range.first != range.second);
}

void
PendingMessageTracker::getStatusStartPage(std::ostream& out) const
{
    out << "View:\n<ul>\n<li><a href=\"?order=bucket\">Group by bucket</a></li>"
           "<li><a href=\"?order=node\">Group by node</a></li>\n";
}

void
PendingMessageTracker::getStatusPerBucket(std::ostream& out) const
{
    std::lock_guard guard(_lock);
    const auto& msgs = boost::multi_index::get<IndexByNodeAndBucket>(_messages);
    using BucketMap = std::map<document::Bucket, std::vector<vespalib::string>>;
    BucketMap perBucketMsgs;
    for (const auto& msg : msgs) {
        perBucketMsgs[msg.bucket].emplace_back(msg.toHtml());
    }

    bool first = true;
    for (auto& bucket : perBucketMsgs) {
        if (!first) {
            out << "</ul>\n";
        }
        out << "<b>" << bucket.first.toString() << "</b>\n";
        out << "<ul>\n";
        first = false;
        for (auto& msgDesc : bucket.second) {
            out << msgDesc;
        }
    }

    if (!first) {
        out << "</ul>\n";
    }
}

void
PendingMessageTracker::getStatusPerNode(std::ostream& out) const
{
    std::lock_guard guard(_lock);
    const auto& msgs = boost::multi_index::get<IndexByNodeAndBucket>(_messages);
    int lastNode = -1;
    for (const auto& node : msgs) {
        if (node.nodeIdx != lastNode) {
            if (lastNode != -1) {
                out << "</ul>\n";
            }

            out << "<b>Node " << node.nodeIdx << " (pending count: "
                << _nodeInfo.getPendingCount(node.nodeIdx) << ")</b>\n<ul>\n";
            lastNode = node.nodeIdx;
        }

        out << node.toHtml();
    }

    if (lastNode != -1) {
        out << "</ul>\n";
    }
}

void
PendingMessageTracker::reportHtmlStatus(std::ostream& out, const framework::HttpUrlPath& path) const
{
    _trackTime.store(true, std::memory_order_relaxed);
    if (!path.hasAttribute("order")) {
        getStatusStartPage(out);
    } else if (path.getAttribute("order") == "bucket") {
        getStatusPerBucket(out);
    } else if (path.getAttribute("order") == "node") {
        getStatusPerNode(out);
    }
}

void
PendingMessageTracker::print(std::ostream&, bool, const std::string&) const { }

}
