// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "modifiedbucketchecker.h"
#include "filestormanager.h"
#include <vespa/storageframework/generic/thread/thread.h>
#include <vespa/persistence/spi/persistenceprovider.h>
#include <vespa/config/common/exceptions.h>
#include <vespa/config/subscription/configuri.h>
#include <vespa/config/helper/configfetcher.hpp>
#include <algorithm>
#include <unistd.h>

#include <vespa/log/log.h>
LOG_SETUP(".persistence.filestor.modifiedbucketchecker");

using document::BucketSpace;

namespace storage {

ModifiedBucketChecker::CyclicBucketSpaceIterator::
CyclicBucketSpaceIterator(ContentBucketSpaceRepo::BucketSpaces bucketSpaces)
    : _bucketSpaces(std::move(bucketSpaces)),
      _idx(0)
{
    std::sort(_bucketSpaces.begin(), _bucketSpaces.end());
}

ModifiedBucketChecker::BucketIdListResult::BucketIdListResult()
    : _bucketSpace(document::BucketSpace::invalid()),
      _buckets()
{
}

void
ModifiedBucketChecker::BucketIdListResult::reset(document::BucketSpace bucketSpace,
                                                 document::bucket::BucketIdList &buckets)
{
    _bucketSpace = bucketSpace;
    assert(_buckets.empty());
    _buckets.swap(buckets);
    // We pick chunks from the end of the list, so reverse it to get
    // the same send order as order received.
    std::reverse(_buckets.begin(), _buckets.end());
}

ModifiedBucketChecker::ModifiedBucketChecker(
        ServiceLayerComponentRegister& compReg,
        spi::PersistenceProvider& provider,
        const StorServerConfig& bootstrap_config)
    : StorageLink("Modified bucket checker"),
      _provider(provider),
      _component(),
      _thread(),
      _monitor(),
      _stateLock(),
      _bucketSpaces(),
      _rechecksNotStarted(),
      _pendingRequests(0),
      _maxPendingChunkSize(100),
      _singleThreadMode(false)
{
    on_configure(bootstrap_config);

    std::ostringstream threadName;
    threadName << "Modified bucket checker " << static_cast<void*>(this);
    _component = std::make_unique<ServiceLayerComponent>(compReg, threadName.str());
    _bucketSpaces = std::make_unique<CyclicBucketSpaceIterator>(_component->getBucketSpaceRepo().getBucketSpaces());
}

ModifiedBucketChecker::~ModifiedBucketChecker()
{
    assert(!_thread);
}

void
ModifiedBucketChecker::on_configure(const vespa::config::content::core::StorServerConfig& newConfig)
{
    std::lock_guard lock(_stateLock);
    if (newConfig.bucketRecheckingChunkSize < 1) {
        throw config::InvalidConfigException(
                "Cannot have bucket rechecking chunk size of less than 1");
    }
    _maxPendingChunkSize = newConfig.bucketRecheckingChunkSize;
}


void
ModifiedBucketChecker::onOpen()
{
    if (!_singleThreadMode) {
        _thread = _component->startThread(*this, 60s, 1s);
    }
}

void
ModifiedBucketChecker::onClose()
{
    if (_singleThreadMode) {
        return;
    }
    if (!_thread) {
        return; // Aborted startup; onOpen() was never called so there's nothing to close.
    }
    LOG(debug, "Interrupting modified bucket checker thread");
    _thread->interrupt();
    _cond.notify_one();
    LOG(debug, "Joining modified bucket checker thread");
    _thread->join();
    LOG(debug, "Modified bucket checker thread joined");
    _thread.reset();
}

void
ModifiedBucketChecker::run(framework::ThreadHandle& thread)
{
    LOG(debug, "Started modified bucket checker thread with pid %d", getpid());

    while (!thread.interrupted()) {
        thread.registerTick(framework::UNKNOWN_CYCLE);

        bool ok = tick();

        std::unique_lock guard(_monitor);
        if (ok) {
            _cond.wait_for(guard, 50ms);
        } else {
            _cond.wait_for(guard, 100ms);
        }
    }
}

bool
ModifiedBucketChecker::onInternalReply(const std::shared_ptr<api::InternalReply>& r)
{
    if (r->getType() == RecheckBucketInfoReply::ID) {
        std::lock_guard guard(_stateLock);
        assert(_pendingRequests > 0);
        --_pendingRequests;
        if (_pendingRequests == 0 && moreChunksRemaining()) {
            _cond.notify_one();
        }
        return true;
    }
    return false;
}

bool
ModifiedBucketChecker::requestModifiedBucketsFromProvider(document::BucketSpace bucketSpace)
{
    spi::BucketIdListResult result(_provider.getModifiedBuckets(bucketSpace));
    if (result.hasError()) {
        LOG(debug, "getModifiedBuckets() failed: %s",
            result.toString().c_str());
        return false;
    }
    {
        std::lock_guard guard(_stateLock);
        _rechecksNotStarted.reset(bucketSpace, result.getList());
    }
    return true;
}

void
ModifiedBucketChecker::nextRecheckChunk(
        std::vector<RecheckBucketInfoCommand::SP>& commandsToSend)
{
    assert(_pendingRequests == 0);
    assert(commandsToSend.empty());
    size_t n = std::min(_maxPendingChunkSize, _rechecksNotStarted.size());

    for (size_t i = 0; i < n; ++i) {
        document::Bucket bucket(_rechecksNotStarted.bucketSpace(), _rechecksNotStarted.back());
        commandsToSend.emplace_back(new RecheckBucketInfoCommand(bucket));
        _rechecksNotStarted.pop_back();
    }
    _pendingRequests = n;
    LOG(spam, "Prepared new recheck chunk with %zu commands", n);
}

void
ModifiedBucketChecker::dispatchAllToPersistenceQueues(
        const std::vector<RecheckBucketInfoCommand::SP>& commandsToSend)
{
    for (auto& cmd : commandsToSend) {
        // We assume sendDown doesn't throw, but that it may send a reply
        // up synchronously, so we cannot hold lock around it. We also make
        // the assumption that recheck commands are only discared if their
        // bucket no longer exists, so it's safe to not retry them.
        sendDown(cmd);
    }
}

bool
ModifiedBucketChecker::tick()
{
    // Do two phases of locking, as we want tick() to both fetch modified
    // buckets and send the first chunk for these in a single call. However,
    // we want getModifiedBuckets() to called outside the lock.
    bool shouldRequestFromProvider;
    {
        std::lock_guard guard(_stateLock);
        if (!currentChunkFinished()) {
            return true;
        }
        shouldRequestFromProvider = !moreChunksRemaining();
    }
    if (shouldRequestFromProvider) {
        if (!requestModifiedBucketsFromProvider(_bucketSpaces->next())) {
            return false;
        }
    }

    std::vector<RecheckBucketInfoCommand::SP> commandsToSend;
    {
        std::lock_guard guard(_stateLock);
        if (moreChunksRemaining()) {
            nextRecheckChunk(commandsToSend);
        } 
    }
    // Sending must be done outside the lock.
    if (!commandsToSend.empty()) {
        dispatchAllToPersistenceQueues(commandsToSend);
    } 
    return true;
}

} // ns storage
