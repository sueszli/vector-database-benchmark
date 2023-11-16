// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/vespalib/testkit/testapp.h>
#include <vespa/vespalib/stllike/string.h>
#include <vespa/document/base/documentid.h>
#include <vespa/searchcore/proton/reference/gid_to_lid_change_listener.h>
#include <vespa/searchlib/common/i_gid_to_lid_mapper_factory.h>
#include <vespa/vespalib/util/destructor_callbacks.h>
#include <vespa/vespalib/util/gate.h>
#include <vespa/vespalib/util/monitored_refcount.h>
#include <vespa/vespalib/util/sequencedtaskexecutor.h>
#include <vespa/searchlib/test/mock_gid_to_lid_mapping.h>
#include <map>
#include <vespa/log/log.h>
LOG_SETUP("gid_to_lid_change_listener_test");

using document::GlobalId;
using document::BucketId;
using document::DocumentId;
using search::attribute::Config;
using search::attribute::BasicType;
using search::attribute::Reference;
using search::attribute::ReferenceAttribute;
using search::attribute::test::MockGidToLidMapperFactory;
using vespalib::MonitoredRefCount;
using vespalib::GenerationHandler;

namespace proton {

namespace {

GlobalId toGid(vespalib::stringref docId) {
    return DocumentId(docId).getGlobalId();
}

vespalib::string doc1("id:test:music::1");
vespalib::string doc2("id:test:music::2");
vespalib::string doc3("id:test:music::3");
VESPA_THREAD_STACK_TAG(test_executor)

struct MyGidToLidMapperFactory : public MockGidToLidMapperFactory
{
    MyGidToLidMapperFactory()
        : MockGidToLidMapperFactory()
    {
        _map.insert({toGid(doc1), 10});
        _map.insert({toGid(doc2), 17});
    }
};

}

struct Fixture
{
    std::shared_ptr<ReferenceAttribute> _attr;
    std::unique_ptr<vespalib::ISequencedTaskExecutor> _writer;
    MonitoredRefCount _refCount;
    std::unique_ptr<GidToLidChangeListener>  _listener;

    Fixture()
        : _attr(std::make_shared<ReferenceAttribute>("test")),
          _writer(vespalib::SequencedTaskExecutor::create(test_executor, 1)),
          _refCount(),
          _listener()
    {
    }

    void ensureDocIdLimit(uint32_t docIdLimit) {
        while (_attr->getNumDocs() < docIdLimit) {
            uint32_t newDocId = 0u;
            _attr->addDoc(newDocId);
            _attr->commit();
        }
    }

    void set(uint32_t doc, const GlobalId &gid) {
        _attr->update(doc, gid);
    }

    void commit() { _attr->commit(); }

    const Reference *getRef(uint32_t doc) {
        return _attr->getReference(doc);
    }

    void assertTargetLid(uint32_t expLid, uint32_t doc) {
        auto ref = getRef(doc);
        EXPECT_TRUE(ref != nullptr);
        EXPECT_EQUAL(expLid, ref->lid());
    }

    void assertNoTargetLid(uint32_t doc) {
        auto ref = getRef(doc);
        EXPECT_TRUE(ref == nullptr);
    }

    void allocListener() {
        _listener = std::make_unique<GidToLidChangeListener>(*_writer, _attr, _refCount, "test", "testdoc");
    }

    void notifyPutDone(const GlobalId &gid, uint32_t referencedDoc) {
        vespalib::Gate gate;
        _listener->notifyPutDone(std::make_shared<vespalib::GateCallback>(gate), gid, referencedDoc);
        gate.await();
    }

    void notifyListenerRegistered(const std::vector<GlobalId>& removes) {
        _listener->notifyRegistered(removes);
    }
};

TEST_F("Test that we can use gid to lid change listener", Fixture)
{
    f.ensureDocIdLimit(4);
    f.set(1, toGid(doc1));
    f.set(2, toGid(doc2));
    f.set(3, toGid(doc1));
    f.commit();
    TEST_DO(f.assertTargetLid(0, 1));
    TEST_DO(f.assertTargetLid(0, 2));
    TEST_DO(f.assertTargetLid(0, 3));
    f.allocListener();
    f.notifyPutDone(toGid(doc1), 10);
    f.notifyPutDone(toGid(doc2), 20);
    f.notifyPutDone(toGid(doc3), 30);
    TEST_DO(f.assertTargetLid(10, 1));
    TEST_DO(f.assertTargetLid(20, 2));
    TEST_DO(f.assertTargetLid(10, 3));
}

TEST_F("Test that target lids are populated when listener is registered", Fixture)
{
    f.ensureDocIdLimit(6);
    f.set(1, toGid(doc1));
    f.set(2, toGid(doc2));
    f.set(3, toGid(doc1));
    f.set(4, toGid(doc3));
    f.commit();
    TEST_DO(f.assertTargetLid(0, 1));
    TEST_DO(f.assertTargetLid(0, 2));
    TEST_DO(f.assertTargetLid(0, 3));
    TEST_DO(f.assertTargetLid(0, 4));
    TEST_DO(f.assertNoTargetLid(5));
    std::shared_ptr<search::IGidToLidMapperFactory> factory =
        std::make_shared<MyGidToLidMapperFactory>();
    f._attr->setGidToLidMapperFactory(factory);
    f.allocListener();
    f.notifyListenerRegistered({});
    TEST_DO(f.assertTargetLid(10, 1));
    TEST_DO(f.assertTargetLid(17, 2));
    TEST_DO(f.assertTargetLid(10, 3));
    TEST_DO(f.assertTargetLid(0, 4));
    TEST_DO(f.assertNoTargetLid(5));
}

TEST_F("Test that removed target lids are pruned when listener is registered", Fixture)
{
    f.ensureDocIdLimit(6);
    f.set(1, toGid(doc1));
    f.set(2, toGid(doc2));
    f.set(3, toGid(doc1));
    f.set(4, toGid(doc3));
    f.commit();
    TEST_DO(f.assertTargetLid(0, 1));
    TEST_DO(f.assertTargetLid(0, 2));
    TEST_DO(f.assertTargetLid(0, 3));
    TEST_DO(f.assertTargetLid(0, 4));
    TEST_DO(f.assertNoTargetLid(5));
    std::shared_ptr<search::IGidToLidMapperFactory> factory =
        std::make_shared<MyGidToLidMapperFactory>();
    f._attr->setGidToLidMapperFactory(factory);
    f.allocListener();
    f.notifyListenerRegistered({ toGid(doc1) });
    TEST_DO(f.assertTargetLid(0, 1));
    TEST_DO(f.assertTargetLid(17, 2));
    TEST_DO(f.assertTargetLid(0, 3));
    TEST_DO(f.assertTargetLid(0, 4));
    TEST_DO(f.assertNoTargetLid(5));
}

}

TEST_MAIN()
{
    TEST_RUN_ALL();
}
