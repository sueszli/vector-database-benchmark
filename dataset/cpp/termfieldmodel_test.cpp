// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/searchlib/fef/fef.h>
#include <vespa/searchlib/queryeval/searchiterator.h>

#include <algorithm>

using namespace search::fef;

struct State {
    SimpleTermData          term;
    MatchData::UP           md;
    TermFieldMatchData     *f3;
    TermFieldMatchData     *f5;
    TermFieldMatchData     *f7;
    TermFieldMatchDataArray array;

    State();
    ~State();

    void setArray(TermFieldMatchDataArray value) {
        array = std::move(value);
    }
};

State::State() : term(), md(), f3(nullptr), f5(nullptr), f7(nullptr), array() {}
State::~State() = default;

/**
 * convenience adapter for easy iteration
 **/
class SimpleTermFieldRangeAdapter
{
    SimpleTermData& _ref;
    size_t _idx;
    size_t _lim;
public:
    explicit SimpleTermFieldRangeAdapter(SimpleTermData& ref)
            : _ref(ref), _idx(0), _lim(ref.numFields())
    {}

    [[nodiscard]] bool valid() const { return (_idx < _lim); }

    [[nodiscard]] SimpleTermFieldData& get() const  { return _ref.field(_idx); }

    void next() { assert(valid()); ++_idx; }
};

void testInvalidId() {
    const TermFieldMatchData empty;
    using search::queryeval::SearchIterator;

    EXPECT_EQUAL(TermFieldMatchData::invalidId(), empty.getDocId());
    EXPECT_TRUE(TermFieldMatchData::invalidId() < (SearchIterator::beginId() + 1 ) ||
               TermFieldMatchData::invalidId() > (search::endDocId - 1));
}

void testSetup(State &state) {
    MatchDataLayout layout;

    state.term.addField(3); // docfreq = 1
    state.term.addField(7); // docfreq = 2
    state.term.addField(5); // docfreq = 3

    using FRA = search::fef::ITermFieldRangeAdapter;
    using SFR = SimpleTermFieldRangeAdapter;

    // lookup terms
    {
        int i = 1;
        for (SFR iter(state.term); iter.valid(); iter.next()) {
            iter.get().setDocFreq(25 * i++, 100);
        }
    }

    // reserve handles
    {
        for (SFR iter(state.term); iter.valid(); iter.next()) {
            iter.get().setHandle(layout.allocTermField(iter.get().getFieldId()));
        }
    }

    state.md = layout.createMatchData();

    // init match data
    {
        for (FRA iter(state.term); iter.valid(); iter.next()) {
            const ITermFieldData& tfd = iter.get();

            TermFieldHandle handle = tfd.getHandle();
            TermFieldMatchData *data = state.md->resolveTermField(handle);
            switch (tfd.getFieldId()) {
            case 3:
                state.f3 = data;
                break;
            case 5:
                state.f5 = data;
                break;
            case 7:
                state.f7 = data;
                break;
            default:
                EXPECT_TRUE(false);
            }
        }
        EXPECT_EQUAL(3u, state.f3->getFieldId());
        EXPECT_EQUAL(5u, state.f5->getFieldId());
        EXPECT_EQUAL(7u, state.f7->getFieldId());
    }

    // test that we can setup array
    EXPECT_EQUAL(false, state.array.valid());
    state.setArray(TermFieldMatchDataArray().add(state.f3).add(state.f5).add(state.f7));
    EXPECT_EQUAL(true, state.array.valid());
}

void testGenerate(State &state) {
    // verify array
    EXPECT_EQUAL(3u, state.array.size());
    EXPECT_EQUAL(state.f3, state.array[0]);
    EXPECT_EQUAL(state.f5, state.array[1]);
    EXPECT_EQUAL(state.f7, state.array[2]);

    // stale unpacked data
    state.f5->reset(5);
    EXPECT_EQUAL(5u, state.f5->getDocId());
    {
        TermFieldMatchDataPosition pos;
        pos.setPosition(3);
        pos.setElementId(0);
        pos.setElementLen(10);
        state.f5->appendPosition(pos);
        EXPECT_EQUAL(1u, state.f5->getIterator().size());
        EXPECT_EQUAL(10u, state.f5->getIterator().getFieldLength());
    }
    state.f5->reset(6);
    EXPECT_EQUAL(6u, state.f5->getDocId());
    EXPECT_EQUAL(FieldPositionsIterator::UNKNOWN_LENGTH,
               state.f5->getIterator().getFieldLength());
    EXPECT_EQUAL(0u, state.f5->getIterator().size());


    // fresh unpacked data
    state.f3->reset(10);
    {
        TermFieldMatchDataPosition pos;
        pos.setPosition(3);
        pos.setElementId(0);
        pos.setElementLen(10);
        EXPECT_EQUAL(FieldPositionsIterator::UNKNOWN_LENGTH,
                   state.f3->getIterator().getFieldLength());
        state.f3->appendPosition(pos);
        EXPECT_EQUAL(10u, state.f3->getIterator().getFieldLength());
    }
    {
        TermFieldMatchDataPosition pos;
        pos.setPosition(15);
        pos.setElementId(1);
        pos.setElementLen(20);
        state.f3->appendPosition(pos);
        EXPECT_EQUAL(20u, state.f3->getIterator().getFieldLength());
    }
    {
        TermFieldMatchDataPosition pos;
        pos.setPosition(1);
        pos.setElementId(2);
        pos.setElementLen(5);
        state.f3->appendPosition(pos);
        EXPECT_EQUAL(20u, state.f3->getIterator().getFieldLength());
    }

    // raw score
    state.f7->setRawScore(10, 5.0);
}

void testAnalyze(State &state) {
    EXPECT_EQUAL(10u, state.f3->getDocId());
    EXPECT_NOT_EQUAL(10u, state.f5->getDocId());
    EXPECT_EQUAL(10u, state.f7->getDocId());

    FieldPositionsIterator it = state.f3->getIterator();
    EXPECT_EQUAL(20u, it.getFieldLength());
    EXPECT_EQUAL(3u, it.size());
    EXPECT_TRUE(it.valid());
    EXPECT_EQUAL(3u, it.getPosition());
    EXPECT_EQUAL(0u, it.getElementId());
    EXPECT_EQUAL(10u, it.getElementLen());
    it.next();
    EXPECT_TRUE(it.valid());
    EXPECT_EQUAL(15u, it.getPosition());
    EXPECT_EQUAL(1u, it.getElementId());
    EXPECT_EQUAL(20u, it.getElementLen());
    it.next();
    EXPECT_TRUE(it.valid());
    EXPECT_EQUAL(1u, it.getPosition());
    EXPECT_EQUAL(2u, it.getElementId());
    EXPECT_EQUAL(5u, it.getElementLen());
    it.next();
    EXPECT_TRUE(!it.valid());

    EXPECT_EQUAL(0.0, state.f3->getRawScore());
    EXPECT_EQUAL(0.0, state.f5->getRawScore());
    EXPECT_EQUAL(5.0, state.f7->getRawScore());
}

TEST("term field model") {
    State state;
    testSetup(state);
    testGenerate(state);
    testAnalyze(state);
    testInvalidId();
}

TEST("append positions") {
    TermFieldMatchData tfmd;
    tfmd.setFieldId(123);
    EXPECT_EQUAL(0u, tfmd.size());
    EXPECT_EQUAL(1u, tfmd.capacity());
    tfmd.reset(7);
    EXPECT_EQUAL(0u, tfmd.size());
    EXPECT_EQUAL(1u, tfmd.capacity());
    TermFieldMatchDataPosition pos(0x01020304, 0x10203040, 0x11223344, 0x12345678);
    tfmd.appendPosition(pos);
    EXPECT_EQUAL(1u, tfmd.size());
    EXPECT_EQUAL(1u, tfmd.capacity());
    EXPECT_EQUAL(0x01020304u, tfmd.begin()->getElementId());
    EXPECT_EQUAL(0x10203040u, tfmd.begin()->getPosition());
    EXPECT_EQUAL(0x11223344, tfmd.begin()->getElementWeight());
    EXPECT_EQUAL(0x12345678u, tfmd.begin()->getElementLen());
    tfmd.reset(11);
    EXPECT_EQUAL(0u, tfmd.size());
    EXPECT_EQUAL(1u, tfmd.capacity());
    TermFieldMatchDataPosition pos2(0x21020304, 0x20203040, 0x21223344, 0x22345678);
    tfmd.appendPosition(pos);
    tfmd.appendPosition(pos2);
    EXPECT_EQUAL(2u, tfmd.size());
    EXPECT_EQUAL(42u, tfmd.capacity());
    TermFieldMatchDataPosition pos3(0x31020304, 0x30203040, 0x31223344, 0x32345678);
    tfmd.appendPosition(pos3);
    EXPECT_EQUAL(3u, tfmd.size());
    EXPECT_EQUAL(42u, tfmd.capacity());
    EXPECT_EQUAL(0x01020304u, tfmd.begin()->getElementId());
    EXPECT_EQUAL(0x10203040u, tfmd.begin()->getPosition());
    EXPECT_EQUAL(0x11223344, tfmd.begin()->getElementWeight());
    EXPECT_EQUAL(0x12345678u, tfmd.begin()->getElementLen());

    EXPECT_EQUAL(0x21020304u, tfmd.begin()[1].getElementId());
    EXPECT_EQUAL(0x20203040u, tfmd.begin()[1].getPosition());
    EXPECT_EQUAL(0x21223344, tfmd.begin()[1].getElementWeight());
    EXPECT_EQUAL(0x22345678u, tfmd.begin()[1].getElementLen());

    EXPECT_EQUAL(0x31020304u, tfmd.begin()[2].getElementId());
    EXPECT_EQUAL(0x30203040u, tfmd.begin()[2].getPosition());
    EXPECT_EQUAL(0x31223344, tfmd.begin()[2].getElementWeight());
    EXPECT_EQUAL(0x32345678u, tfmd.begin()[2].getElementLen());
}

TEST("Access subqueries") {
    State state;
    testSetup(state);
    state.f3->reset(10);
    state.f3->setSubqueries(10, 42);
    EXPECT_EQUAL(42ULL, state.f3->getSubqueries());
    state.f3->enableRawScore();
    EXPECT_EQUAL(0ULL, state.f3->getSubqueries());

    state.f3->reset(11);
    state.f3->appendPosition(TermFieldMatchDataPosition());
    state.f3->setSubqueries(11, 42);
    EXPECT_EQUAL(0ULL, state.f3->getSubqueries());
}

TEST("require that TermFieldMatchData can be tagged as needed or not") {
    TermFieldMatchData tfmd;
    tfmd.setFieldId(123);
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(!tfmd.isNotNeeded());
    EXPECT_TRUE(tfmd.needs_normal_features());
    EXPECT_TRUE(tfmd.needs_interleaved_features());
    tfmd.tagAsNotNeeded();
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(tfmd.isNotNeeded());
    EXPECT_TRUE(!tfmd.needs_normal_features());
    EXPECT_TRUE(!tfmd.needs_interleaved_features());
    tfmd.setNeedNormalFeatures(true);
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(!tfmd.isNotNeeded());
    EXPECT_TRUE(tfmd.needs_normal_features());
    EXPECT_TRUE(!tfmd.needs_interleaved_features());
    tfmd.setNeedInterleavedFeatures(true);
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(!tfmd.isNotNeeded());
    EXPECT_TRUE(tfmd.needs_normal_features());
    EXPECT_TRUE(tfmd.needs_interleaved_features());
    tfmd.setNeedNormalFeatures(false);
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(!tfmd.isNotNeeded());
    EXPECT_TRUE(!tfmd.needs_normal_features());
    EXPECT_TRUE(tfmd.needs_interleaved_features());
    tfmd.setNeedInterleavedFeatures(false);
    EXPECT_EQUAL(tfmd.getFieldId(),123u);
    EXPECT_TRUE(tfmd.isNotNeeded());
    EXPECT_TRUE(!tfmd.needs_normal_features());
    EXPECT_TRUE(!tfmd.needs_interleaved_features());
}

TEST("require that MatchData soft_reset retains appropriate state") {    
    auto md = MatchData::makeTestInstance(10, 10);
    md->set_termwise_limit(0.5);
    auto *old_term = md->resolveTermField(7); 
    old_term->tagAsNotNeeded();
    old_term->populate_fixed()->setElementWeight(21);
    old_term->resetOnlyDocId(42);
    EXPECT_EQUAL(md->get_termwise_limit(), 0.5);
    EXPECT_TRUE(old_term->isNotNeeded());
    EXPECT_EQUAL(old_term->getFieldId(), 7u);
    EXPECT_EQUAL(old_term->getWeight(), 21);
    EXPECT_EQUAL(old_term->getDocId(), 42u);
    md->soft_reset();
    auto *new_term = md->resolveTermField(7);
    EXPECT_EQUAL(new_term, old_term);
    EXPECT_EQUAL(md->get_termwise_limit(), 1.0);
    EXPECT_TRUE(new_term->isNotNeeded());
    EXPECT_EQUAL(new_term->getFieldId(), 7u);
    EXPECT_EQUAL(new_term->getWeight(), 21);
    EXPECT_EQUAL(new_term->getDocId(), TermFieldMatchData::invalidId());
}

TEST("require that compareWithExactness implements a strict weak ordering") {
   TermFieldMatchDataPosition a(0, 1, 100, 1);
   TermFieldMatchDataPosition b(0, 2, 100, 1);
   TermFieldMatchDataPosition c(0, 2, 100, 1);
   TermFieldMatchDataPosition d(0, 3, 100, 3);
   TermFieldMatchDataPosition e(0, 3, 100, 3);
   TermFieldMatchDataPosition f(0, 4, 100, 1);

   d.setMatchExactness(0.75);
   e.setMatchExactness(0.5);

   bool (*cmp)(const TermFieldMatchDataPosition &a,
               const TermFieldMatchDataPosition &b) = TermFieldMatchDataPosition::compareWithExactness;

   EXPECT_EQUAL(true, cmp(a, b));
   EXPECT_EQUAL(false, cmp(b, c));
   EXPECT_EQUAL(true, cmp(c, d));
   EXPECT_EQUAL(true, cmp(d, e));
   EXPECT_EQUAL(true, cmp(e, f));

   EXPECT_EQUAL(false, cmp(b, a));
   EXPECT_EQUAL(false, cmp(c, b));
   EXPECT_EQUAL(false, cmp(d, c));
   EXPECT_EQUAL(false, cmp(e, d));
   EXPECT_EQUAL(false, cmp(f, e));
}


TEST_MAIN() { TEST_RUN_ALL(); }
