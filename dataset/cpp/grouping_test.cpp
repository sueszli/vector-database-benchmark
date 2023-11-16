// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/searchlib/aggregation/grouping.h>
#include <vespa/searchlib/aggregation/sumaggregationresult.h>
#include <vespa/searchcommon/attribute/iattributevector.h>
#include <vespa/searchlib/expression/attributenode.h>
#include <vespa/searchlib/expression/integerresultnode.h>
#include <vespa/searchlib/attribute/extendableattributes.h>
#include <vespa/searchcore/grouping/groupingcontext.h>
#include <vespa/searchcore/grouping/groupingmanager.h>
#include <vespa/searchcore/grouping/groupingsession.h>
#include <vespa/searchcore/proton/matching/sessionmanager.h>
#include <vespa/searchlib/common/allocatedbitvector.h>
#include <vespa/searchlib/test/mock_attribute_context.h>
#include <vespa/vespalib/util/testclock.h>
#include <iostream>
#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/log/log.h>
LOG_SETUP("grouping_test");

using namespace search::attribute;
using namespace search::aggregation;
using namespace search::expression;
using namespace search::grouping;
using namespace search;
using search::attribute::test::MockAttributeContext;
using proton::matching::SessionManager;
using vespalib::steady_time;
using vespalib::duration;

//-----------------------------------------------------------------------------

const uint32_t NUM_DOCS = 1000;

//-----------------------------------------------------------------------------

struct MyWorld {
    MockAttributeContext attributeContext;
    search::AllocatedBitVector bv;

    MyWorld()
        : attributeContext(),
          bv(NUM_DOCS+1)
    {
        bv.setInterval(0, NUM_DOCS);
        // attribute context
        {
            SingleInt32ExtAttribute *attr = new SingleInt32ExtAttribute("attr0");
            AttributeVector::DocId docid;
            for (uint32_t i = 0; i < NUM_DOCS; ++i) {
                attr->addDoc(docid);
                attr->add(i, docid); // value = docid
            }
            assert(docid + 1 == NUM_DOCS);
            attributeContext.add(attr);
        }
        {
            SingleInt32ExtAttribute *attr = new SingleInt32ExtAttribute("attr1");
            AttributeVector::DocId docid;
            for (uint32_t i = 0; i < NUM_DOCS; ++i) {
                attr->addDoc(docid);
                attr->add(i * 2, docid); // value = docid * 2
            }
            assert(docid + 1 == NUM_DOCS);
            attributeContext.add(attr);
        }
        {
            SingleInt32ExtAttribute *attr = new SingleInt32ExtAttribute("attr2");
            AttributeVector::DocId docid;
            for (uint32_t i = 0; i < NUM_DOCS; ++i) {
                attr->addDoc(docid);
                attr->add(i * 3, docid); // value = docid * 3
            }
            assert(docid + 1 == NUM_DOCS);
            attributeContext.add(attr);
        }
        {
            SingleInt32ExtAttribute *attr = new SingleInt32ExtAttribute("attr3");
            AttributeVector::DocId docid;
            for (uint32_t i = 0; i < NUM_DOCS; ++i) {
                attr->addDoc(docid);
                attr->add(i * 4, docid); // value = docid * 4
            }
            assert(docid + 1 == NUM_DOCS);
            attributeContext.add(attr);
        }

    }
};

//-----------------------------------------------------------------------------

using GroupingList = GroupingContext::GroupingList;

SessionId createSessionId(const std::string & s) {
    std::vector<char> vec;
    for (size_t i = 0; i < s.size(); i++) {
        vec.push_back(s[i]);
    }
    return SessionId(&vec[0], vec.size());
}

class CheckAttributeReferences : public vespalib::ObjectOperation, public vespalib::ObjectPredicate
{
public:
    CheckAttributeReferences(bool log=false) : _log(log), _numrefs(0) { }
    bool     _log;
    uint32_t _numrefs;
private:
    void execute(vespalib::Identifiable &obj) override {
        if (_log) {
            std::cerr << _numrefs << ": " << &obj << " = " << obj.asString() << std::endl;
        }
        if (static_cast<AttributeNode &>(obj).getAttribute() != nullptr) {
            _numrefs++;
        }
    }
    bool check(const vespalib::Identifiable &obj) const override { return obj.inherits(AttributeNode::classId); }
};

struct DoomFixture {
    vespalib::TestClock clock;
    steady_time timeOfDoom;
    DoomFixture() : clock(), timeOfDoom(steady_time::max()) {}
};

//-----------------------------------------------------------------------------

TEST("testSessionId") {
    SessionId id1;
    ASSERT_TRUE(id1.empty());

    SessionId id2(createSessionId("foo"));
    SessionId id3(createSessionId("bar"));

    ASSERT_TRUE(!id2.empty());
    ASSERT_TRUE(!id3.empty());
    ASSERT_TRUE(id3 < id2);
    EXPECT_EQUAL(id2, id2);
}

#define MU std::make_unique

GroupingLevel
createGL(ExpressionNode::UP expr, ExpressionNode::UP result) {
    GroupingLevel l;
    l.setExpression(std::move(expr));
    l.addResult(SumAggregationResult().setExpression(std::move(result)));
    return l;
}

GroupingLevel
createGL(ExpressionNode::UP expr, ExpressionNode::UP resultExpr, ResultNode::UP result) {
    GroupingLevel l;
    l.setExpression(std::move(expr));
    l.addResult(SumAggregationResult().setExpression(std::move(resultExpr)).setResult(result.release()));
    return l;
}
GroupingLevel
createGL(size_t maxGroups, ExpressionNode::UP expr) {
    GroupingLevel l;
    l.setMaxGroups(maxGroups);
    l.setExpression(std::move(expr));
    return l;
}

TEST_F("testGroupingContextInitialization", DoomFixture()) {
    vespalib::nbostream os;
    Grouping baseRequest;
    baseRequest.setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")))
            .addLevel(createGL(MU<AttributeNode>("attr3"), MU<AttributeNode>("attr1")));

    vespalib::NBOSerializer nos(os);
    nos << (uint32_t)1;
    baseRequest.serialize(nos);

    AllocatedBitVector bv(1);
    GroupingContext context(bv, f1.clock.clock(), f1.timeOfDoom, os.data(), os.size(), true);
    ASSERT_TRUE(!context.empty());
    GroupingContext::GroupingList list = context.getGroupingList();
    ASSERT_TRUE(list.size() == 1);
    EXPECT_EQUAL(list[0]->asString(), baseRequest.asString());
    context.reset();
    ASSERT_TRUE(context.empty());
}

TEST_F("testGroupingContextUsage", DoomFixture()) {
    vespalib::nbostream os;
    Grouping request1;
    request1.setFirstLevel(0)
            .setLastLevel(0)
            .setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")))
            .addLevel(createGL(MU<AttributeNode>("attr3"), MU<AttributeNode>("attr1")));

    Grouping request2;
    request2.setFirstLevel(0)
            .setLastLevel(3)
            .setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")))
            .addLevel(createGL(MU<AttributeNode>("attr3"), MU<AttributeNode>("attr1")));


    auto r1 = std::make_shared<Grouping>(request1);
    auto r2 = std::make_shared<Grouping>(request2);
    AllocatedBitVector bv(1);
    GroupingContext context(bv, f1.clock.clock(), f1.timeOfDoom);
    ASSERT_TRUE(context.empty());
    context.addGrouping(r1);
    ASSERT_TRUE(context.getGroupingList().size() == 1);
    context.addGrouping(r2);
    ASSERT_TRUE(context.getGroupingList().size() == 2);
    context.reset();
    ASSERT_TRUE(context.empty());
}

TEST_F("testGroupingContextSerializing", DoomFixture()) {
    Grouping baseRequest;
    baseRequest.setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")))
            .addLevel(createGL(MU<AttributeNode>("attr3"), MU<AttributeNode>("attr1")));

    vespalib::nbostream os;
    vespalib::NBOSerializer nos(os);
    nos << (uint32_t)1;
    baseRequest.serialize(nos);

    AllocatedBitVector bv(1);
    GroupingContext context(bv, f1.clock.clock(), f1.timeOfDoom);
    auto bp = std::make_shared<Grouping>(baseRequest);
    context.addGrouping(bp);
    context.serialize();
    vespalib::nbostream & res(context.getResult());
    EXPECT_EQUAL(res.size(), os.size());
    ASSERT_TRUE(memcmp(res.data(), os.data(), res.size()) == 0);
}

TEST_F("testGroupingManager", DoomFixture()) {
    vespalib::nbostream os;
    Grouping request1;
    request1.setFirstLevel(0)
            .setLastLevel(0)
            .setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")));

    AllocatedBitVector bv(1);
    GroupingContext context(bv, f1.clock.clock(), f1.timeOfDoom);
    auto bp = std::make_shared<Grouping>(request1);
    context.addGrouping(bp);
    GroupingManager manager(context);
    ASSERT_TRUE(!manager.empty());
}

TEST_F("testGroupingSession", DoomFixture()) {
    MyWorld world;
    vespalib::nbostream os;
    Grouping request1;
    request1.setId(0)
            .setFirstLevel(0)
            .setLastLevel(0)
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")));

    Grouping request2;
    request2.setId(1)
            .setFirstLevel(0)
            .setLastLevel(3)
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")))
            .addLevel(createGL(MU<AttributeNode>("attr3"), MU<AttributeNode>("attr1")));


    CheckAttributeReferences attrCheck;
    request1.select(attrCheck, attrCheck);
    EXPECT_EQUAL(0u, attrCheck._numrefs);
    request2.select(attrCheck, attrCheck);
    EXPECT_EQUAL(0u, attrCheck._numrefs);

    auto r1 = std::make_shared<Grouping>(request1);
    auto r2 = std::make_shared<Grouping>(request2);
    GroupingContext initContext(world.bv, f1.clock.clock(), f1.timeOfDoom);
    initContext.addGrouping(r1);
    initContext.addGrouping(r2);
    SessionId id("foo");

    // Test initialization phase
    GroupingSession session(id, initContext, world.attributeContext);
    CheckAttributeReferences attrCheck2;
    EXPECT_EQUAL(2u, initContext.getGroupingList().size());
    for (const auto & g : initContext.getGroupingList()) {
        g->select(attrCheck2, attrCheck2);
    }
    EXPECT_EQUAL(8u, attrCheck2._numrefs);
    RankedHit hit;
    hit._docId = 0;
    GroupingManager &manager(session.getGroupingManager());
    manager.groupInRelevanceOrder(&hit, 1);
    CheckAttributeReferences attrCheck_after;
    GroupingList &gl3(initContext.getGroupingList());
    for (unsigned int i = 0; i < gl3.size(); i++) {
        gl3[i]->select(attrCheck_after, attrCheck_after);
    }
    EXPECT_EQUAL(attrCheck_after._numrefs, 0u);
    {
        EXPECT_EQUAL(id, session.getSessionId());
        ASSERT_TRUE(!session.getGroupingManager().empty());
        ASSERT_TRUE(!session.finished());
        session.continueExecution(initContext);
        ASSERT_TRUE(!session.finished());
    }
    // Test second pass
    {
        GroupingContext context(world.bv, f1.clock.clock(), f1.timeOfDoom);
        auto r = std::make_shared<Grouping>(request1);
        r->setFirstLevel(1);
        r->setLastLevel(1);
        context.addGrouping(r);

        session.continueExecution(context);
        ASSERT_TRUE(!session.finished());
    }
    // Test last pass. Session should be marked as finished
    {
        GroupingContext context(world.bv, f1.clock.clock(), f1.timeOfDoom);
        auto r = std::make_shared<Grouping>(request1);
        r->setFirstLevel(2);
        r->setLastLevel(2);
        context.addGrouping(r);

        session.continueExecution(context);
        ASSERT_TRUE(session.finished());
    }

}

TEST_F("testEmptySessionId", DoomFixture()) {
    MyWorld world;
    vespalib::nbostream os;
    Grouping request1;
    request1.setId(0)
            .setFirstLevel(0)
            .setLastLevel(0)
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2")))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3")));

    auto r1 = std::make_shared<Grouping>(request1);
    GroupingContext initContext(world.bv, f1.clock.clock(), f1.timeOfDoom);
    initContext.addGrouping(r1);
    SessionId id;

    // Test initialization phase
    GroupingSession session(id, initContext, world.attributeContext);
    RankedHit hit;
    hit._docId = 0;
    GroupingManager &manager(session.getGroupingManager());
    manager.groupInRelevanceOrder(&hit, 1);
    EXPECT_EQUAL(id, session.getSessionId());
    ASSERT_TRUE(!session.getGroupingManager().empty());
    ASSERT_TRUE(session.finished() && session.getSessionId().empty());
    session.continueExecution(initContext);
    ASSERT_TRUE(session.finished());
    ASSERT_TRUE(r1->getRoot().getChildrenSize() > 0);
}

TEST_F("testSessionManager", DoomFixture()) {
    MyWorld world;
    vespalib::nbostream os;
    Grouping request1;
    request1.setId(0)
            .setFirstLevel(0)
            .setLastLevel(0)
            .addLevel(createGL(MU<AttributeNode>("attr1"), MU<AttributeNode>("attr2"), MU<Int64ResultNode>(0)))
            .addLevel(createGL(MU<AttributeNode>("attr2"), MU<AttributeNode>("attr3"), MU<Int64ResultNode>(0)))
            .setRoot(Group().addResult(SumAggregationResult()
                                               .setExpression(MU<AttributeNode>("attr0"))
                                               .setResult(Int64ResultNode(0))));

    auto r1 = std::make_shared<Grouping>(request1);
    GroupingContext initContext(world.bv, f1.clock.clock(), f1.timeOfDoom);
    initContext.addGrouping(r1);

    SessionManager mgr(2);
    SessionId id1("foo");
    SessionId id2("bar");
    SessionId id3("baz");
    auto s1 = std::make_unique<GroupingSession>(id1, initContext, world.attributeContext);
    auto s2 = std::make_unique<GroupingSession>(id2, initContext, world.attributeContext);
    auto s3 = std::make_unique<GroupingSession>(id3, initContext, world.attributeContext);

    ASSERT_EQUAL(f1.timeOfDoom, s1->getTimeOfDoom());
    mgr.insert(std::move(s1));
    s1 = mgr.pickGrouping(id1);
    ASSERT_TRUE(s1.get());
    EXPECT_EQUAL(id1, s1->getSessionId());

    mgr.insert(std::move(s1));
    mgr.insert(std::move(s2));
    mgr.insert(std::move(s3));
    s1 = mgr.pickGrouping(id1);
    s2 = mgr.pickGrouping(id2);
    s3 = mgr.pickGrouping(id3);
    ASSERT_FALSE(s1);
    ASSERT_TRUE(s2);
    ASSERT_TRUE(s3);
    EXPECT_EQUAL(id2, s2->getSessionId());
    EXPECT_EQUAL(id3, s3->getSessionId());
    SessionManager::Stats stats = mgr.getGroupingStats();
    EXPECT_EQUAL(4u, stats.numInsert);
    EXPECT_EQUAL(3u, stats.numPick);
    EXPECT_EQUAL(1u, stats.numDropped);
}

void doGrouping(GroupingContext &ctx,
                uint32_t doc1, double rank1,
                uint32_t doc2, double rank2,
                uint32_t doc3, double rank3)
{
    GroupingManager man(ctx);
    std::vector<RankedHit> hits;
    hits.push_back(RankedHit(doc1, rank1));
    hits.push_back(RankedHit(doc2, rank2));
    hits.push_back(RankedHit(doc3, rank3));
    man.groupInRelevanceOrder(&hits[0], 3);
}

TEST_F("test grouping fork/join", DoomFixture()) {
    MyWorld world;

    Grouping request;
    request.setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0"))))
           .addLevel(createGL(3, MU<AttributeNode>("attr0")))
           .setFirstLevel(0)
           .setLastLevel(1);

    auto g1 = std::make_shared<Grouping>(request);
    GroupingContext context(world.bv, f1.clock.clock(), f1.timeOfDoom);
    context.addGrouping(g1);
    GroupingSession session(SessionId(), context, world.attributeContext);
    session.prepareThreadContextCreation(4);

    GroupingContext::UP ctx0 = session.createThreadContext(0, world.attributeContext);
    GroupingContext::UP ctx1 = session.createThreadContext(1, world.attributeContext);
    GroupingContext::UP ctx2 = session.createThreadContext(2, world.attributeContext);
    GroupingContext::UP ctx3 = session.createThreadContext(3, world.attributeContext);
    doGrouping(*ctx0, 12, 30.0, 11, 20.0, 10, 10.0);
    doGrouping(*ctx1, 22, 150.0, 21, 40.0, 20, 25.0);
    doGrouping(*ctx2, 32, 100.0, 31, 15.0, 30, 5.0);
    doGrouping(*ctx3, 42, 4.0, 41, 3.0, 40, 2.0); // not merged (verify independent contexts)
    {
        GroupingManager man(*ctx0);
        man.merge(*ctx1);
        man.merge(*ctx2);
        man.prune();
    }

    Grouping expect;
    expect.setRoot(Group().addResult(SumAggregationResult().setExpression(MU<AttributeNode>("attr0")).setResult(Int64ResultNode(189)))
                           .addChild(Group().setId(Int64ResultNode(21)).setRank(40.0))
                           .addChild(Group().setId(Int64ResultNode(22)).setRank(150.0))
                           .addChild(Group().setId(Int64ResultNode(32)).setRank(100.0)))
            .addLevel(createGL(3, MU<AttributeNode>("attr0")))
            .setFirstLevel(0)
            .setLastLevel(1);

    session.continueExecution(context);
    GroupingContext::GroupingList list = context.getGroupingList();
    ASSERT_TRUE(list.size() == 1);
    EXPECT_EQUAL(expect.asString(), list[0]->asString());
}

TEST_F("test session timeout", DoomFixture()) {
    MyWorld world;
    SessionManager mgr(2);
    SessionId id1("foo");
    SessionId id2("bar");

    GroupingContext initContext1(world.bv, f1.clock.clock(), steady_time(duration(10)));
    GroupingContext initContext2(world.bv, f1.clock.clock(), steady_time(duration(20)));
    auto s1 = std::make_unique<GroupingSession>(id1, initContext1, world.attributeContext);
    auto s2 = std::make_unique<GroupingSession>(id2, initContext2, world.attributeContext);
    mgr.insert(std::move(s1));
    mgr.insert(std::move(s2));
    mgr.pruneTimedOutSessions(steady_time(5ns));
    ASSERT_EQUAL(2u, mgr.getGroupingStats().numCached);
    mgr.pruneTimedOutSessions(steady_time(10ns));
    ASSERT_EQUAL(2u, mgr.getGroupingStats().numCached);

    mgr.pruneTimedOutSessions(steady_time(11ns));
    ASSERT_EQUAL(1u, mgr.getGroupingStats().numCached);

    mgr.pruneTimedOutSessions(steady_time(21ns));
    ASSERT_EQUAL(0u, mgr.getGroupingStats().numCached);
}

TEST_MAIN() { TEST_RUN_ALL(); }
