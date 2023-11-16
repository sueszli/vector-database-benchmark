// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "persistenceengine.h"
#include "ipersistenceengineowner.h"
#include "transport_latch.h"
#include <vespa/persistence/spi/bucketexecutor.h>
#include <vespa/persistence/spi/doctype_gid_and_timestamp.h>
#include <vespa/persistence/spi/catchresult.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/datatype/documenttype.h>
#include <vespa/document/update/documentupdate.h>
#include <vespa/document/util/feed_reject_helper.h>
#include <vespa/document/base/exceptions.h>
#include <thread>

#include <vespa/log/log.h>
LOG_SETUP(".proton.persistenceengine.persistenceengine");

using document::Document;
using document::DocumentId;
using storage::spi::BucketChecksum;
using storage::spi::BucketExecutor;
using storage::spi::BucketTask;
using storage::spi::BucketIdListResult;
using storage::spi::BucketInfo;
using storage::spi::BucketInfoResult;
using storage::spi::IncludedVersions;
using storage::spi::Result;
using storage::spi::OperationComplete;
using vespalib::IllegalStateException;
using vespalib::Sequence;
using vespalib::make_string_short::fmt;
using std::make_unique;

using namespace std::chrono_literals;

namespace proton {

namespace {

class ResultHandlerBase {
private:
    virtual Result::UP createResult() const = 0;
protected:
    std::mutex             _lock;
    std::atomic<uint32_t>  _countDown;
    OperationComplete::UP  _onComplete;
    void countDown();
    void complete() {
        _onComplete->onComplete(createResult());
    }
public:
    ResultHandlerBase(uint32_t waitCnt, OperationComplete::UP onComplete);
    virtual ~ResultHandlerBase();
};

ResultHandlerBase::ResultHandlerBase(uint32_t waitCnt, OperationComplete::UP onComplete)
    : _lock(),
      _countDown(waitCnt),
      _onComplete(std::move(onComplete))
{}
ResultHandlerBase::~ResultHandlerBase() = default;

void
ResultHandlerBase::countDown() {
    if (_countDown.fetch_sub(1) == 1) {
        complete();
    }
}

class GenericResultHandler final : public ResultHandlerBase, public IGenericResultHandler {
private:
    Result _result;
public:
    GenericResultHandler(uint32_t waitCnt, OperationComplete::UP onComplete) :
        ResultHandlerBase(waitCnt, std::move(onComplete)),
        _result()
    {
        if (waitCnt == 0) {
            complete();
        }
    }
    ~GenericResultHandler() override;
    void handle(const Result &result) override {
        if (result.hasError()) {
            std::lock_guard<std::mutex> guard(_lock);
            if (_result.hasError()) {
                _result = TransportMerger::mergeErrorResults(_result, result);
            } else {
                _result = result;
            }
        }
        countDown();
    }
    Result::UP createResult() const override { return std::make_unique<Result>(_result); }
};

GenericResultHandler::~GenericResultHandler() = default;

class BucketIdListResultHandler : public IBucketIdListResultHandler
{
private:
    using BucketIdSet = vespalib::hash_set<document::BucketId, document::BucketId::hash>;
    BucketIdSet _bucketSet;
public:
    BucketIdListResultHandler()
        : _bucketSet()
    { }
    ~BucketIdListResultHandler() override;
    void
    handle(BucketIdListResult result) override {
        const BucketIdListResult::List &buckets = result.getList();
        for (size_t i = 0; i < buckets.size(); ++i) {
            _bucketSet.insert(buckets[i]);
        }
    }
    std::unique_ptr<BucketIdListResult>
    getResult() const {
        BucketIdListResult::List buckets;
        buckets.reserve(_bucketSet.size());
        for (document::BucketId bucketId : _bucketSet) {
            buckets.push_back(bucketId);
        }
        return std::make_unique<BucketIdListResult>(std::move(buckets));
    }
};


BucketIdListResultHandler::~BucketIdListResultHandler() = default;

class SynchronizedBucketIdListResultHandler : public ResultHandlerBase,
                                              public BucketIdListResultHandler
{
public:
    SynchronizedBucketIdListResultHandler(uint32_t waitCnt, OperationComplete::UP onComplete)
        : ResultHandlerBase(waitCnt, std::move(onComplete)),
          BucketIdListResultHandler()
    {
        if (waitCnt == 0) {
            complete();
        }
    }
    ~SynchronizedBucketIdListResultHandler() override;
    void handle(BucketIdListResult result) override {
        {
            std::lock_guard<std::mutex> guard(_lock);
            BucketIdListResultHandler::handle(std::move(result));
        }
        countDown();
    }
    Result::UP createResult() const override {
        return getResult();
    }
};

SynchronizedBucketIdListResultHandler::~SynchronizedBucketIdListResultHandler() = default;

class BucketInfoResultHandler : public IBucketInfoResultHandler {
private:
    BucketInfoResult _result;
    bool             _first;
public:
    BucketInfoResultHandler() :
        _result(BucketInfo()),
        _first(true)
    {
    }
    ~BucketInfoResultHandler() override;
    void handle(const BucketInfoResult &result) override {
        if (_first) {
            _result = result;
            _first = false;
        } else {
            BucketInfo b1 = _result.getBucketInfo();
            BucketInfo b2 = result.getBucketInfo();
            BucketInfo::ReadyState ready =
                    (b1.getReady() == b2.getReady() ? b1.getReady() : BucketInfo::NOT_READY);
            BucketInfo::ActiveState active =
                    (b1.getActive() == b2.getActive() ? b1.getActive() : BucketInfo::NOT_ACTIVE);
            _result = BucketInfoResult(
                    BucketInfo(BucketChecksum(b1.getChecksum() + b2.getChecksum()),
                            b1.getDocumentCount() + b2.getDocumentCount(),
                            b1.getDocumentSize() + b2.getDocumentSize(),
                            b1.getEntryCount() + b2.getEntryCount(),
                            b1.getUsedSize() + b2.getUsedSize(),
                            ready, active));
        }
    }
    const BucketInfoResult &getResult() const { return _result; }
};

BucketInfoResultHandler::~BucketInfoResultHandler() = default;

}

PersistenceEngine::HandlerSnapshot
PersistenceEngine::getHandlerSnapshot(const WriteGuard &) const
{
    return _handlers.getHandlerSnapshot();
}

PersistenceEngine::HandlerSnapshot
PersistenceEngine::getSafeHandlerSnapshot(const ReadGuard &, document::BucketSpace bucketSpace) const
{
    return _handlers.getHandlerSnapshot(bucketSpace);
}

PersistenceEngine::UnsafeHandlerSnapshot
PersistenceEngine::getHandlerSnapshot(const ReadGuard &, document::BucketSpace bucketSpace) const {
    return _handlers.getUnsafeHandlerSnapshot(bucketSpace);
}

PersistenceEngine::HandlerSnapshot
PersistenceEngine::getHandlerSnapshot(const WriteGuard &, document::BucketSpace bucketSpace) const
{
    return _handlers.getHandlerSnapshot(bucketSpace);
}

PersistenceEngine::PersistenceEngine(IPersistenceEngineOwner &owner, const IResourceWriteFilter &writeFilter, IDiskMemUsageNotifier& disk_mem_usage_notifier,
                                     ssize_t defaultSerializedSize, bool ignoreMaxBytes)
    : AbstractPersistenceProvider(),
      _defaultSerializedSize(defaultSerializedSize),
      _ignoreMaxBytes(ignoreMaxBytes),
      _handlers(),
      _lock(),
      _iterators(),
      _iterators_lock(),
      _owner(owner),
      _writeFilter(writeFilter),
      _clusterStates(),
      _extraModifiedBuckets(),
      _rwMutex(),
      _resource_usage_tracker(std::make_shared<ResourceUsageTracker>(disk_mem_usage_notifier))
{
}


PersistenceEngine::~PersistenceEngine()
{
    destroyIterators();
}


IPersistenceHandler::SP
PersistenceEngine::putHandler(const WriteGuard &, document::BucketSpace bucketSpace, const DocTypeName &docType,const IPersistenceHandler::SP &handler)
{
    return _handlers.putHandler(bucketSpace, docType, handler);
}


IPersistenceHandler *
PersistenceEngine::getHandler(const ReadGuard &, document::BucketSpace bucketSpace, const DocTypeName &docType) const
{
    return _handlers.getHandler(bucketSpace, docType);
}


IPersistenceHandler::SP
PersistenceEngine::removeHandler(const WriteGuard &, document::BucketSpace bucketSpace, const DocTypeName &docType)
{
    // TODO: Grab bucket list and treat them as modified
    return _handlers.removeHandler(bucketSpace, docType);
}


Result
PersistenceEngine::initialize()
{
    WriteGuard wguard(getWLock());
    LOG(debug, "Begin initializing persistence handlers");
    HandlerSnapshot snap = getHandlerSnapshot(wguard);
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->initialize();
    }
    LOG(debug, "Done initializing persistence handlers");
    return Result();
}


BucketIdListResult
PersistenceEngine::listBuckets(BucketSpace bucketSpace) const
{
    // Runs in SPI thread.
    // No handover to write threads in persistence handlers.
    ReadGuard rguard(_rwMutex);
    auto snap = getHandlerSnapshot(rguard, bucketSpace);
    BucketIdListResultHandler resultHandler;
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleListBuckets(resultHandler);
    }
    return std::move(*resultHandler.getResult());
}


Result
PersistenceEngine::setClusterState(BucketSpace bucketSpace, const ClusterState &calc)
{
    ReadGuard rguard(_rwMutex);
    saveClusterState(bucketSpace, calc);
    auto snap = getHandlerSnapshot(rguard, bucketSpace);
    auto catchResult = std::make_unique<storage::spi::CatchResult>();
    auto futureResult = catchResult->future_result();
    GenericResultHandler resultHandler(snap.size(), std::move(catchResult));
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleSetClusterState(calc, resultHandler);
    }
    Result::UP result = futureResult.get();
    _owner.setClusterState(bucketSpace, calc);
    return *result;
}


void
PersistenceEngine::setActiveStateAsync(const Bucket & bucket, BucketInfo::ActiveState newState, OperationComplete::UP onComplete)
{
    ReadGuard rguard(_rwMutex);
    auto snap = getHandlerSnapshot(rguard, bucket.getBucketSpace());
    auto resultHandler = std::make_shared<GenericResultHandler>(snap.size(), std::move(onComplete));
    while (snap.handlers().valid()) {
        IPersistenceHandler *handler = snap.handlers().get();
        snap.handlers().next();
        if (snap.handlers().valid()) {
            handler->handleSetActiveState(bucket, newState, resultHandler);
        } else {
            handler->handleSetActiveState(bucket, newState, std::move(resultHandler));
        }
    }
}


BucketInfoResult
PersistenceEngine::getBucketInfo(const Bucket& b) const
{
    // Runs in SPI thread.
    // No handover to write threads in persistence handlers.
    ReadGuard rguard(_rwMutex);
    auto snap = getHandlerSnapshot(rguard, b.getBucketSpace());
    BucketInfoResultHandler resultHandler;
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleGetBucketInfo(b, resultHandler);
    }
    return resultHandler.getResult();
}


void
PersistenceEngine::putAsync(const Bucket &bucket, Timestamp ts, storage::spi::DocumentSP doc, OperationComplete::UP onComplete)
{
    if (!_writeFilter.acceptWriteOperation()) {
        IResourceWriteFilter::State state = _writeFilter.getAcceptState();
        if (!state.acceptWriteOperation()) {
            return onComplete->onComplete(std::make_unique<Result>(Result::ErrorType::RESOURCE_EXHAUSTED,
                    fmt("Put operation rejected for document '%s': '%s'", doc->getId().toString().c_str(), state.message().c_str())));
        }
    }
    ReadGuard rguard(_rwMutex);
    DocTypeName docType(doc->getType());
    LOG(spam, "putAsync(%s, %" PRIu64 ", (\"%s\", \"%s\"))", bucket.toString().c_str(), static_cast<uint64_t>(ts.getValue()),
        docType.toString().c_str(), doc->getId().toString().c_str());
    if (!doc->getId().hasDocType()) {
        return onComplete->onComplete(std::make_unique<Result>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("Old id scheme not supported in elastic mode (%s)", doc->getId().toString().c_str())));
    }
    IPersistenceHandler * handler = getHandler(rguard, bucket.getBucketSpace(), docType);
    if (!handler) {
        return onComplete->onComplete(std::make_unique<Result>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("No handler for document type '%s'", docType.toString().c_str())));
    }
    auto transportContext = std::make_shared<AsyncTransportContext>(1, std::move(onComplete));
    handler->handlePut(feedtoken::make(std::move(transportContext)), bucket, ts, std::move(doc));
}

void
PersistenceEngine::removeAsync(const Bucket& b, std::vector<storage::spi::IdAndTimestamp> ids, OperationComplete::UP onComplete)
{
    if (ids.size() == 1) {
        removeAsyncSingle(b, ids[0].timestamp, ids[0].id, std::move(onComplete));
    } else {
        removeAsyncMulti(b, std::move(ids), std::move(onComplete));
    }
}

void
PersistenceEngine::removeAsyncMulti(const Bucket& b, std::vector<storage::spi::IdAndTimestamp> ids, OperationComplete::UP onComplete) {
    ReadGuard rguard(_rwMutex);
    //TODO Group per document type/handler and handle in one go.
    for (const auto & stampedId : ids) {
        const document::DocumentId & id = stampedId.id;
        if (!id.hasDocType()) {
            return onComplete->onComplete(
                    std::make_unique<RemoveResult>(Result::ErrorType::PERMANENT_ERROR,
                                                   fmt("Old id scheme not supported in elastic mode (%s)", id.toString().c_str())));
        }
        DocTypeName docType(id.getDocType());
        IPersistenceHandler *handler = getHandler(rguard, b.getBucketSpace(), docType);
        if (!handler) {
            return onComplete->onComplete(std::make_unique<RemoveResult>(Result::ErrorType::PERMANENT_ERROR,
                                                                         fmt("No handler for document type '%s'",
                                                                             docType.toString().c_str())));
        }
    }
    auto transportContext = std::make_shared<AsyncRemoveTransportContext>(ids.size(), std::move(onComplete));
    for (const auto & stampedId : ids) {
        const document::DocumentId & id = stampedId.id;
        DocTypeName docType(id.getDocType());
        IPersistenceHandler *handler = getHandler(rguard, b.getBucketSpace(), docType);
        handler->handleRemove(feedtoken::make(transportContext), b, stampedId.timestamp, id);
    }
}

void
PersistenceEngine::removeAsyncSingle(const Bucket& b, Timestamp t, const DocumentId& id, OperationComplete::UP onComplete)
{
    ReadGuard rguard(_rwMutex);
    LOG(spam, "remove(%s, %" PRIu64 ", \"%s\")", b.toString().c_str(),
        static_cast<uint64_t>(t.getValue()), id.toString().c_str());
    if (!id.hasDocType()) {
        return onComplete->onComplete(std::make_unique<RemoveResult>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("Old id scheme not supported in elastic mode (%s)", id.toString().c_str())));
    }
    DocTypeName docType(id.getDocType());
    IPersistenceHandler * handler = getHandler(rguard, b.getBucketSpace(), docType);
    if (!handler) {
        return onComplete->onComplete(std::make_unique<RemoveResult>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("No handler for document type '%s'", docType.toString().c_str())));
    }
    auto transportContext = std::make_shared<AsyncTransportContext>(1, std::move(onComplete));
    handler->handleRemove(feedtoken::make(std::move(transportContext)), b, t, id);
}

void
PersistenceEngine::removeByGidAsync(const Bucket& b, std::vector<storage::spi::DocTypeGidAndTimestamp> ids, std::unique_ptr<OperationComplete> onComplete)
{
    ReadGuard rguard(_rwMutex);
    for (const auto & dt_gid_ts : ids) {
        DocTypeName doc_type(dt_gid_ts.doc_type);
        IPersistenceHandler *handler = getHandler(rguard, b.getBucketSpace(), doc_type);
        if (!handler) {
            return onComplete->onComplete(std::make_unique<RemoveResult>(Result::ErrorType::PERMANENT_ERROR,
                                                                         fmt("No handler for document type '%s'",
                                                                             doc_type.toString().c_str())));
        }
    }
    auto transportContext = std::make_shared<AsyncRemoveTransportContext>(ids.size(), std::move(onComplete));
    for (const auto & dt_gid_ts : ids) {
        DocTypeName doc_type(dt_gid_ts.doc_type);
        IPersistenceHandler *handler = getHandler(rguard, b.getBucketSpace(), doc_type);
        handler->handleRemoveByGid(feedtoken::make(transportContext), b, dt_gid_ts.timestamp, dt_gid_ts.doc_type, dt_gid_ts.gid);
    }
}

void
PersistenceEngine::updateAsync(const Bucket& b, Timestamp t, DocumentUpdate::SP upd, OperationComplete::UP onComplete)
{
    if (!_writeFilter.acceptWriteOperation()) {
        IResourceWriteFilter::State state = _writeFilter.getAcceptState();
        if (!state.acceptWriteOperation() && document::FeedRejectHelper::mustReject(*upd)) {
            return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::RESOURCE_EXHAUSTED,
                    fmt("Update operation rejected for document '%s': '%s'", upd->getId().toString().c_str(), state.message().c_str())));
        }
    }
    try {
        upd->eagerDeserialize();
    } catch (document::FieldNotFoundException & e) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::TRANSIENT_ERROR,
                    fmt("Update operation rejected for document '%s' of type '%s': 'Field not found'",
                        upd->getId().toString().c_str(), upd->getType().getName().c_str())));
    } catch (document::DocumentTypeNotFoundException & e) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::TRANSIENT_ERROR,
                    fmt("Update operation rejected for document '%s' of type '%s'.",
                        upd->getId().toString().c_str(), e.getDocumentTypeName().c_str())));

    } catch (document::WrongTensorTypeException &e) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::TRANSIENT_ERROR,
                    fmt("Update operation rejected for document '%s' of type '%s': 'Wrong tensor type: %s'",
                        upd->getId().toString().c_str(), upd->getType().getName().c_str(), e.getMessage().c_str())));
    }
    ReadGuard rguard(_rwMutex);
    DocTypeName docType(upd->getType());
    LOG(spam, "update(%s, %" PRIu64 ", (\"%s\", \"%s\"), createIfNonExistent='%s')",
        b.toString().c_str(), static_cast<uint64_t>(t.getValue()), docType.toString().c_str(),
        upd->getId().toString().c_str(), (upd->getCreateIfNonExistent() ? "true" : "false"));
    if (!upd->getId().hasDocType()) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("Old id scheme not supported in elastic mode (%s)", upd->getId().toString().c_str())));
    }
    if (upd->getId().getDocType() != docType.getName()) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("Update operation rejected due to bad id (%s, %s)", upd->getId().toString().c_str(), docType.getName().c_str())));
    }
    IPersistenceHandler * handler = getHandler(rguard, b.getBucketSpace(), docType);

    if (handler == nullptr) {
        return onComplete->onComplete(std::make_unique<UpdateResult>(Result::ErrorType::PERMANENT_ERROR,
                    fmt("No handler for document type '%s'", docType.toString().c_str())));
    }
    auto transportContext = std::make_shared<AsyncTransportContext>(1, std::move(onComplete));
    handler->handleUpdate(feedtoken::make(std::move(transportContext)), b, t, std::move(upd));
}


PersistenceEngine::GetResult
PersistenceEngine::get(const Bucket& b, const document::FieldSet& fields, const DocumentId& did, Context& context) const
{
    ReadGuard rguard(_rwMutex);
    auto snap = getHandlerSnapshot(rguard, b.getBucketSpace());

    for (;snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler::RetrieversSP retrievers = snap.handlers().get()->getDocumentRetrievers(context.getReadConsistency());
        for (size_t i = 0; i < retrievers->size(); ++i) {
            IDocumentRetriever &retriever = *(*retrievers)[i];
            search::DocumentMetaData meta = retriever.getDocumentMetaData(did);
            storage::spi::Timestamp timestamp(meta.timestamp);
            if (timestamp != 0 && meta.bucketId == b.getBucketId()) {
                if (meta.removed) {
                    return GetResult::make_for_tombstone(timestamp);
                }
                if (document::FieldSet::Type::NONE == fields.getType()) {
                    return GetResult::make_for_metadata_only(timestamp);
                }
                document::Document::UP doc = retriever.getPartialDocument(meta.lid, did, fields);
                if (!doc || doc->getId().getGlobalId() != meta.gid) {
                    return GetResult();
                }
                return GetResult(std::move(doc), timestamp);
            }
        }
    }
    return GetResult();
}


PersistenceEngine::CreateIteratorResult
PersistenceEngine::createIterator(const Bucket &bucket, FieldSetSP fields, const Selection &selection,
                                  IncludedVersions versions, Context &context)
{
    ReadGuard rguard(_rwMutex);
    auto snap = getSafeHandlerSnapshot(rguard, bucket.getBucketSpace());

    auto entry = std::make_unique<IteratorEntry>(context.getReadConsistency(), bucket, std::move(fields), selection,
                                                 versions, _defaultSerializedSize, _ignoreMaxBytes);
    for (; snap.handlers().valid(); snap.handlers().next()) {
        auto *handler = snap.handlers().get();
        IPersistenceHandler::RetrieversSP retrievers = handler->getDocumentRetrievers(context.getReadConsistency());
        for (const auto & retriever : *retrievers) {
            entry->it.add(handler->doc_type_name(), retriever);
        }
    }
    entry->handler_sequence = HandlerSnapshot::release(std::move(snap));

    std::lock_guard<std::mutex> guard(_iterators_lock);
    static std::atomic<IteratorId::Type> id_counter(0);
    IteratorId id(++id_counter);
    _iterators[id] = entry.release();
    return CreateIteratorResult(id);
}


PersistenceEngine::IterateResult
PersistenceEngine::iterate(IteratorId id, uint64_t maxByteSize) const
{
    ReadGuard rguard(_rwMutex);
    IteratorEntry *iteratorEntry;
    {
        std::lock_guard<std::mutex> guard(_iterators_lock);
        auto it = _iterators.find(id);
        if (it == _iterators.end()) {
            return IterateResult(Result::ErrorType::PERMANENT_ERROR, fmt("Unknown iterator with id %" PRIu64, id.getValue()));
        }
        iteratorEntry = it->second;
        if (iteratorEntry->in_use) {
            return IterateResult(Result::ErrorType::TRANSIENT_ERROR, fmt("Iterator with id %" PRIu64 " is already in use", id.getValue()));
        }
        iteratorEntry->in_use = true;
    }

    DocumentIterator &iterator = iteratorEntry->it;
    try {
        IterateResult result = iterator.iterate(maxByteSize);
        std::lock_guard<std::mutex> guard(_iterators_lock);
        iteratorEntry->in_use = false;
        return result;
    } catch (const std::exception & e) {
        IterateResult result(Result::ErrorType::PERMANENT_ERROR, fmt("Caught exception during visitor iterator.iterate() = '%s'", e.what()));
        LOG(warning, "Caught exception during visitor iterator.iterate() = '%s'", e.what());
        std::lock_guard<std::mutex> guard(_iterators_lock);
        iteratorEntry->in_use = false;
        return result;
    }
}


Result
PersistenceEngine::destroyIterator(IteratorId id)
{
    ReadGuard rguard(_rwMutex);
    std::lock_guard<std::mutex> guard(_iterators_lock);
    auto it = _iterators.find(id);
    if (it == _iterators.end()) {
        return Result();
    }
    if (it->second->in_use) {
        return Result(Result::ErrorType::TRANSIENT_ERROR, fmt("Iterator with id %" PRIu64 " is currently in use", id.getValue()));
    }
    delete it->second;
    _iterators.erase(it);
    return Result();
}


void
PersistenceEngine::createBucketAsync(const Bucket &b, OperationComplete::UP onComplete) noexcept
{
    ReadGuard rguard(_rwMutex);
    LOG(spam, "createBucket(%s)", b.toString().c_str());
    auto snap = getHandlerSnapshot(rguard, b.getBucketSpace());

    auto transportContext = std::make_shared<AsyncTransportContext>(snap.size(), std::move(onComplete));
    while (snap.handlers().valid()) {
        IPersistenceHandler *handler = snap.handlers().get();
        snap.handlers().next();
        if (snap.handlers().valid()) {
            handler->handleCreateBucket(feedtoken::make(transportContext), b);
        } else {
            handler->handleCreateBucket(feedtoken::make(std::move(transportContext)), b);
        }
    }
}


void
PersistenceEngine::deleteBucketAsync(const Bucket& b, OperationComplete::UP onComplete) noexcept
{
    ReadGuard rguard(_rwMutex);
    LOG(spam, "deleteBucket(%s)", b.toString().c_str());
    auto snap = getHandlerSnapshot(rguard, b.getBucketSpace());

    auto transportContext = std::make_shared<AsyncTransportContext>(snap.size(), std::move(onComplete));
    while (snap.handlers().valid()) {
        IPersistenceHandler *handler = snap.handlers().get();
        snap.handlers().next();
        if (snap.handlers().valid()) {
            handler->handleDeleteBucket(feedtoken::make(transportContext), b);
        } else {
            handler->handleDeleteBucket(feedtoken::make(std::move(transportContext)), b);
        }
    }
}


BucketIdListResult
PersistenceEngine::getModifiedBuckets(BucketSpace bucketSpace) const
{
    ReadGuard rguard(_rwMutex);
    using MBV = BucketIdListResultV;
    MBV extraModifiedBuckets;
    {
        std::lock_guard<std::mutex> guard(_lock);
        extraModifiedBuckets.swap(_extraModifiedBuckets[bucketSpace]);
    }
    auto snap = getHandlerSnapshot(rguard, bucketSpace);
    auto catchResult = std::make_unique<storage::spi::CatchResult>();
    auto futureResult = catchResult->future_result();
    SynchronizedBucketIdListResultHandler resultHandler(snap.size() + extraModifiedBuckets.size(), std::move(catchResult));
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleGetModifiedBuckets(resultHandler);
    }
    for (auto & item : extraModifiedBuckets) {
        resultHandler.handle(std::move(*item));
    }
    return std::move(dynamic_cast<BucketIdListResult &>(*futureResult.get()));
}


Result
PersistenceEngine::split(const Bucket& source, const Bucket& target1, const Bucket& target2)
{
    ReadGuard rguard(_rwMutex);
    LOG(spam, "split(%s, %s, %s)", source.toString().c_str(), target1.toString().c_str(), target2.toString().c_str());
    assert(source.getBucketSpace() == target1.getBucketSpace());
    assert(source.getBucketSpace() == target2.getBucketSpace());
    auto snap = getHandlerSnapshot(rguard, source.getBucketSpace());
    TransportLatch latch(snap.size());
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleSplit(feedtoken::make(latch), source, target1, target2);
    }
    latch.await();
    return latch.getResult();
}


Result
PersistenceEngine::join(const Bucket& source1, const Bucket& source2, const Bucket& target)
{
    ReadGuard rguard(_rwMutex);
    LOG(spam, "join(%s, %s, %s)", source1.toString().c_str(), source2.toString().c_str(), target.toString().c_str());
    assert(source1.getBucketSpace() == target.getBucketSpace());
    assert(source2.getBucketSpace() == target.getBucketSpace());
    auto snap = getHandlerSnapshot(rguard, target.getBucketSpace());
    TransportLatch latch(snap.size());
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleJoin(feedtoken::make(latch), source1, source2, target);
    }
    latch.await();
    return latch.getResult();
}

std::unique_ptr<vespalib::IDestructorCallback>
PersistenceEngine::register_resource_usage_listener(IResourceUsageListener& listener)
{
    return _resource_usage_tracker->set_listener(listener);
}

void
PersistenceEngine::destroyIterators()
{
    for (;;) {
        IteratorId id;
        {
            std::lock_guard<std::mutex> guard(_iterators_lock);
            if (_iterators.empty())
                break;
            id = _iterators.begin()->first;
        }
        Result res(destroyIterator(id));
        if (res.hasError()) {
            LOG(debug, "%zu iterator left. Can not destroy iterator '%" PRIu64 "'. Reason='%s'", _iterators.size(), id.getValue(), res.toString().c_str());
            std::this_thread::sleep_for(100ms);
        }
    }
}


void
PersistenceEngine::saveClusterState(BucketSpace bucketSpace, const ClusterState &calc)
{
    auto clusterState = std::make_shared<ClusterState>(calc);
    {
        std::lock_guard<std::mutex> guard(_lock);
        clusterState.swap(_clusterStates[bucketSpace]);
    }
}

PersistenceEngine::ClusterState::SP
PersistenceEngine::savedClusterState(BucketSpace bucketSpace) const
{
    std::lock_guard<std::mutex> guard(_lock);
    auto itr(_clusterStates.find(bucketSpace));
    return ((itr != _clusterStates.end()) ? itr->second : ClusterState::SP());
}

void
PersistenceEngine::propagateSavedClusterState(BucketSpace bucketSpace, IPersistenceHandler &handler)
{
    ClusterState::SP clusterState(savedClusterState(bucketSpace));
    if (!clusterState)
        return;
    // Propagate saved cluster state.
    // TODO: Fix race with new cluster state setting.
    auto catchResult = std::make_unique<storage::spi::CatchResult>();
    auto futureResult = catchResult->future_result();
    GenericResultHandler resultHandler(1, std::move(catchResult));
    handler.handleSetClusterState(*clusterState, resultHandler);
    futureResult.get();
}

void
PersistenceEngine::grabExtraModifiedBuckets(BucketSpace bucketSpace, IPersistenceHandler &handler)
{
    BucketIdListResultHandler resultHandler;
    handler.handleListBuckets(resultHandler);
    std::shared_ptr<BucketIdListResult> result(resultHandler.getResult());
    std::lock_guard<std::mutex> guard(_lock);
    _extraModifiedBuckets[bucketSpace].push_back(result);
}


class ActiveBucketIdListResultHandler : public IBucketIdListResultHandler
{
private:
    using BucketIdMap = std::map<document::BucketId, size_t>;
    using IR = std::pair<BucketIdMap::iterator, bool>;
    BucketIdMap _bucketMap;
public:
    ActiveBucketIdListResultHandler() : _bucketMap() { }

    void handle(BucketIdListResult result) override {
        const BucketIdListResult::List &buckets = result.getList();
        for (size_t i = 0; i < buckets.size(); ++i) {
            IR ir(_bucketMap.insert(std::make_pair(buckets[i], 1u)));
            if (!ir.second) {
                ++(ir.first->second);
            }
        }
    }

    const BucketIdMap & getBucketMap() const { return _bucketMap; }
};

void
PersistenceEngine::populateInitialBucketDB(const WriteGuard & guard, BucketSpace bucketSpace, IPersistenceHandler &targetHandler)
{
    HandlerSnapshot snap = getHandlerSnapshot(guard, bucketSpace);
    
    size_t snapSize(snap.size());
    size_t flawed = 0;

    // handleListActiveBuckets() runs in SPI thread.
    // No handover to write threads in persistence handlers.
    ActiveBucketIdListResultHandler resultHandler;
    for (; snap.handlers().valid(); snap.handlers().next()) {
        IPersistenceHandler *handler = snap.handlers().get();
        handler->handleListActiveBuckets(resultHandler);
    }
    using BucketIdMap = std::map<document::BucketId, size_t>;
    document::BucketId::List buckets;
    const BucketIdMap &bucketMap(resultHandler.getBucketMap());

    for (const auto & item : bucketMap) {
        if (item.second != snapSize) {
            ++flawed;
        }
        buckets.push_back(item.first);
    }
    LOG(info, "Adding %zu active buckets (%zu flawed) to new bucket db", buckets.size(), flawed);
    auto catchResult = std::make_unique<storage::spi::CatchResult>();
    auto futureResult = catchResult->future_result();
    GenericResultHandler trHandler(1, std::move(catchResult));
    targetHandler.handlePopulateActiveBuckets(std::move(buckets), trHandler);
    futureResult.get();
}

std::unique_lock<std::shared_mutex>
PersistenceEngine::getWLock() const
{
    return WriteGuard(_rwMutex);
}

namespace {

class ExecutorRegistration : public vespalib::IDestructorCallback {
public:
    explicit ExecutorRegistration(std::shared_ptr<BucketExecutor> executor) : _executor(std::move(executor)) { }
    ~ExecutorRegistration() override = default;
private:
    std::shared_ptr<BucketExecutor> _executor;
};

}

std::unique_ptr<vespalib::IDestructorCallback>
PersistenceEngine::register_executor(std::shared_ptr<BucketExecutor> executor)
{
    assert(_bucket_executor.expired());
    _bucket_executor = executor;
    return std::make_unique<ExecutorRegistration>(executor);
}

void
PersistenceEngine::execute(const storage::spi::Bucket &bucket, std::unique_ptr<BucketTask> task) {
    auto bucketExecutor = get_bucket_executor();
    if (bucketExecutor) {
        bucketExecutor->execute(bucket, std::move(task));
    } else {
        return task->fail(bucket);
    }
}

} // storage
