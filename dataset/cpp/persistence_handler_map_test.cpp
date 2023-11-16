// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/searchcore/proton/persistenceengine/ipersistencehandler.h>
#include <vespa/searchcore/proton/persistenceengine/persistence_handler_map.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/update/documentupdate.h>

#include <vespa/vespalib/testkit/testapp.h>

using namespace document;
using namespace proton;

using HandlerSnapshot = PersistenceHandlerMap::HandlerSnapshot;

struct DummyPersistenceHandler : public IPersistenceHandler {
    using SP = std::shared_ptr<DummyPersistenceHandler>;
    void initialize() override {}
    void handlePut(FeedToken, const storage::spi::Bucket &, storage::spi::Timestamp, DocumentSP) override {}
    void handleUpdate(FeedToken, const storage::spi::Bucket &, storage::spi::Timestamp, DocumentUpdateSP) override {}
    void handleRemove(FeedToken, const storage::spi::Bucket &, storage::spi::Timestamp, const document::DocumentId &) override {}
    void handleRemoveByGid(FeedToken, const storage::spi::Bucket&, storage::spi::Timestamp, vespalib::stringref, const GlobalId&) override { }
    void handleListBuckets(IBucketIdListResultHandler &) override {}
    void handleSetClusterState(const storage::spi::ClusterState &, IGenericResultHandler &) override {}
    void handleSetActiveState(const storage::spi::Bucket &, storage::spi::BucketInfo::ActiveState, std::shared_ptr<IGenericResultHandler>) override {}
    void handleGetBucketInfo(const storage::spi::Bucket &, IBucketInfoResultHandler &) override {}
    void handleCreateBucket(FeedToken, const storage::spi::Bucket &) override {}
    void handleDeleteBucket(FeedToken, const storage::spi::Bucket &) override {}
    void handleGetModifiedBuckets(IBucketIdListResultHandler &) override {}
    void handleSplit(FeedToken, const storage::spi::Bucket &, const storage::spi::Bucket &, const storage::spi::Bucket &) override {}
    void handleJoin(FeedToken, const storage::spi::Bucket &, const storage::spi::Bucket &, const storage::spi::Bucket &) override {}

    RetrieversSP getDocumentRetrievers(storage::spi::ReadConsistency) override { return RetrieversSP(); }
    void handleListActiveBuckets(IBucketIdListResultHandler &) override {}
    void handlePopulateActiveBuckets(document::BucketId::List, IGenericResultHandler &) override {}
    const DocTypeName & doc_type_name() const noexcept override { abort(); }
};

BucketSpace space_1(1);
BucketSpace space_2(2);
BucketSpace space_null(3);
DocTypeName type_a("a");
DocTypeName type_b("b");
DocTypeName type_c("c");
DummyPersistenceHandler::SP handler_a(std::make_shared<DummyPersistenceHandler>());
DummyPersistenceHandler::SP handler_b(std::make_shared<DummyPersistenceHandler>());
DummyPersistenceHandler::SP handler_c(std::make_shared<DummyPersistenceHandler>());
DummyPersistenceHandler::SP handler_a_new(std::make_shared<DummyPersistenceHandler>());

void
assertHandler(const IPersistenceHandler::SP & lhs, const IPersistenceHandler * rhs)
{
    EXPECT_EQUAL(lhs.get(), rhs);
}

void
assertHandler(const IPersistenceHandler::SP &lhs, const IPersistenceHandler::SP &rhs)
{
    EXPECT_EQUAL(lhs.get(), rhs.get());
}

template <typename T>
void
assertNullHandler(const T & handler)
{
    EXPECT_TRUE(! handler);
}

void
assertSnapshot(const std::vector<IPersistenceHandler::SP> &exp, HandlerSnapshot snapshot)
{
    EXPECT_EQUAL(exp.size(), snapshot.size());
    auto &sequence = snapshot.handlers();
    for (size_t i = 0; i < exp.size() && sequence.valid(); ++i, sequence.next()) {
        EXPECT_EQUAL(exp[i].get(), sequence.get());
    }
}

struct Fixture {
    PersistenceHandlerMap map;
    Fixture() {
        TEST_DO(assertNullHandler(map.putHandler(space_1, type_a, handler_a)));
        TEST_DO(assertNullHandler(map.putHandler(space_1, type_b, handler_b)));
        TEST_DO(assertNullHandler(map.putHandler(space_2, type_c, handler_c)));
    }
};

TEST_F("require that handlers can be retrieved", Fixture)
{
    TEST_DO(assertHandler(handler_a, f.map.getHandler(space_1, type_a)));
    TEST_DO(assertHandler(handler_b, f.map.getHandler(space_1, type_b)));
    TEST_DO(assertHandler(handler_c, f.map.getHandler(space_2, type_c)));
    TEST_DO(assertNullHandler(f.map.getHandler(space_1, type_c)));
    TEST_DO(assertNullHandler(f.map.getHandler(space_null, type_a)));
}

TEST_F("require that old handler is returned if replaced by new handler", Fixture)
{
    TEST_DO(assertHandler(handler_a, f.map.putHandler(space_1, type_a, handler_a_new)));
    TEST_DO(assertHandler(handler_a_new, f.map.getHandler(space_1, type_a)));
}

TEST_F("require that handler can be removed (and old handler returned)", Fixture)
{
    TEST_DO(assertHandler(handler_a, f.map.removeHandler(space_1, type_a)));
    TEST_DO(assertNullHandler(f.map.getHandler(space_1, type_a)));
    TEST_DO(assertNullHandler(f.map.removeHandler(space_1, type_c)));
}

TEST_F("require that handler snapshot can be retrieved for all handlers", Fixture)
{
    TEST_DO(assertSnapshot({handler_c, handler_a, handler_b}, f.map.getHandlerSnapshot()));
}

TEST_F("require that handler snapshot can be retrieved for given bucket space", Fixture)
{
    TEST_DO(assertSnapshot({handler_a, handler_b}, f.map.getHandlerSnapshot(space_1)));
    TEST_DO(assertSnapshot({handler_c}, f.map.getHandlerSnapshot(space_2)));
    TEST_DO(assertSnapshot({}, f.map.getHandlerSnapshot(space_null)));
}

TEST_MAIN()
{
    TEST_RUN_ALL();
}

