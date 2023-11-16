// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/searchcore/proton/flushengine/active_flush_stats.h>
#include <vespa/searchcore/proton/flushengine/cachedflushtarget.h>
#include <vespa/searchcore/proton/flushengine/flush_engine_explorer.h>
#include <vespa/searchcore/proton/flushengine/flushengine.h>
#include <vespa/searchcore/proton/flushengine/i_tls_stats_factory.h>
#include <vespa/searchcore/proton/flushengine/threadedflushtarget.h>
#include <vespa/searchcore/proton/flushengine/tls_stats_map.h>
#include <vespa/searchcore/proton/server/igetserialnum.h>
#include <vespa/searchcore/proton/test/dummy_flush_handler.h>
#include <vespa/searchcore/proton/test/dummy_flush_target.h>
#include <vespa/searchlib/common/flush_token.h>
#include <vespa/vespalib/data/slime/slime.h>
#include <vespa/vespalib/test/insertion_operators.h>
#include <vespa/vespalib/testkit/testapp.h>
#include <mutex>
#include <thread>

#include <vespa/log/log.h>
LOG_SETUP("flushengine_test");

using namespace proton;
using namespace proton::flushengine;
using namespace vespalib::slime;
using searchcorespi::IFlushTarget;
using searchcorespi::FlushTask;
using vespalib::Slime;

constexpr vespalib::duration LONG_TIMEOUT = 66666ms;
constexpr vespalib::duration SHORT_TIMEOUT = 1ms;
constexpr vespalib::duration IINTERVAL = 1s;

class SimpleExecutor : public vespalib::Executor {
public:
    vespalib::Gate _done;

public:
    SimpleExecutor()
        : _done()
    { }

    Task::UP
    execute(Task::UP task) override
    {
        task->run();
        _done.countDown();
        return Task::UP();
    }
    void wakeup() override { }
};

class SimpleGetSerialNum : public IGetSerialNum
{
    search::SerialNum getSerialNum() const override {
        return 0u;
    }
};

class SimpleTlsStatsFactory : public flushengine::ITlsStatsFactory
{
    flushengine::TlsStatsMap create() override {
        vespalib::hash_map<vespalib::string, flushengine::TlsStats> map;
        return flushengine::TlsStatsMap(std::move(map));
    }
};

class SimpleHandler;

class WrappedFlushTask : public searchcorespi::FlushTask
{
    searchcorespi::FlushTask::UP _task;
    SimpleHandler &_handler;

public:
    void run() override;
    WrappedFlushTask(searchcorespi::FlushTask::UP task,
                     SimpleHandler &handler)
        : _task(std::move(task)),
          _handler(handler)
    { }

    search::SerialNum getFlushSerial() const override {
        return _task->getFlushSerial();
    }
};

class WrappedFlushTarget : public FlushTargetProxy
{
    SimpleHandler &_handler;
public:
    WrappedFlushTarget(const IFlushTarget::SP &target, SimpleHandler &handler)
        : FlushTargetProxy(target),
          _handler(handler)
    { }

    Task::UP initFlush(SerialNum currentSerial, std::shared_ptr<search::IFlushToken> flush_token) override {
        Task::UP task(_target->initFlush(currentSerial, std::move(flush_token)));
        if (task) {
            return std::make_unique<WrappedFlushTask>(std::move(task), _handler);
        }
        return task;
    }
};

using Targets = std::vector<IFlushTarget::SP>;

using FlushDoneHistory = std::vector<search::SerialNum>;

class SimpleHandler : public test::DummyFlushHandler {
public:
    Targets                   _targets;
    search::SerialNum         _oldestSerial;
    search::SerialNum         _currentSerial;
    uint32_t                  _pendingDone;
    uint32_t                  _taskDone;
    mutable std::mutex        _lock;
    vespalib::CountDownLatch  _done;
    FlushDoneHistory          _flushDoneHistory;

public:
    using SP = std::shared_ptr<SimpleHandler>;

    SimpleHandler(const Targets &targets, const std::string &name = "anon",
                  search::SerialNum currentSerial = -1)
        : test::DummyFlushHandler(name),
          _targets(targets),
          _oldestSerial(0),
          _currentSerial(currentSerial),
          _pendingDone(0u),
          _taskDone(0u),
          _lock(),
          _done(targets.size()),
          _flushDoneHistory()
    { }

    search::SerialNum getCurrentSerialNumber() const override {
        LOG(info, "SimpleHandler(%s)::getCurrentSerialNumber()", getName().c_str());
        return _currentSerial;
    }

    std::vector<IFlushTarget::SP>
    getFlushTargets() override {
        {
            std::lock_guard guard(_lock);
            _pendingDone += _taskDone;
            _taskDone = 0;
        }
        LOG(info, "SimpleHandler(%s)::getFlushTargets()", getName().c_str());
        std::vector<IFlushTarget::SP> wrappedTargets;
        for (const auto &target : _targets) {
            wrappedTargets.push_back(std::make_shared<WrappedFlushTarget>(target, *this));
        }
        return wrappedTargets;
    }

    // Called once by flush engine thread for each task done
    void taskDone() {
        std::lock_guard guard(_lock);
        ++_taskDone;
    }

    // Called by flush engine master thread after flush handler is
    // added to flush engine and when one or more flush tasks related
    // to flush handler have completed.
    void flushDone(search::SerialNum oldestSerial) override {
        std::lock_guard guard(_lock);
        LOG(info, "SimpleHandler(%s)::flushDone(%" PRIu64 ")", getName().c_str(), oldestSerial);
        _oldestSerial = std::max(_oldestSerial, oldestSerial);
        _flushDoneHistory.push_back(oldestSerial);
        while (_pendingDone > 0) {
            --_pendingDone;
            _done.countDown();
        }
    }

    FlushDoneHistory getFlushDoneHistory() {
        std::lock_guard guard(_lock);
        return _flushDoneHistory;
    }

    [[nodiscard]] search::SerialNum oldest_serial() const noexcept {
        std::lock_guard guard(_lock);
        return _oldestSerial;
    }
};

void WrappedFlushTask::run()
{
    _task->run();
    _handler.taskDone();
}

class SimpleTask : public searchcorespi::FlushTask {
    std::atomic<search::SerialNum> &_flushedSerial;
    search::SerialNum &_currentSerial;
public:
    vespalib::Gate &_start;
    vespalib::Gate &_done;
    vespalib::Gate *_proceed;

public:
    SimpleTask(vespalib::Gate &start,
               vespalib::Gate &done,
               vespalib::Gate *proceed,
               std::atomic<search::SerialNum> &flushedSerial,
               search::SerialNum &currentSerial)
        : _flushedSerial(flushedSerial), _currentSerial(currentSerial),
          _start(start), _done(done), _proceed(proceed)
    { }

    void run() override {
        _start.countDown();
        if (_proceed != nullptr) {
            _proceed->await();
        }
        _flushedSerial.store(_currentSerial, std::memory_order_relaxed);
        _done.countDown();
    }

    search::SerialNum getFlushSerial() const override { return 0u; }
};

class SimpleTarget : public test::DummyFlushTarget {
public:
    std::atomic<search::SerialNum> _flushedSerial;
    search::SerialNum _currentSerial;
    vespalib::Gate    _proceed;
    vespalib::Gate    _initDone;
    vespalib::Gate    _taskStart;
    vespalib::Gate    _taskDone;
    Task::UP          _task;

protected:
    SimpleTarget(const std::string &name, const Type &type, search::SerialNum flushedSerial = 0, bool proceedImmediately = true) :
        test::DummyFlushTarget(name, type, Component::OTHER),
        _flushedSerial(flushedSerial),
        _currentSerial(0),
        _proceed(),
        _initDone(),
        _taskStart(),
        _taskDone(),
        _task(std::make_unique<SimpleTask>(_taskStart, _taskDone, &_proceed,
                                           _flushedSerial, _currentSerial))
    {
        if (proceedImmediately) {
            _proceed.countDown();
        }
    }

public:
    using SP = std::shared_ptr<SimpleTarget>;

    SimpleTarget(Task::UP task, const std::string &name) noexcept :
        test::DummyFlushTarget(name),
        _flushedSerial(0),
        _currentSerial(0),
        _proceed(),
        _initDone(),
        _taskStart(),
        _taskDone(),
        _task(std::move(task))
    { }

    SimpleTarget(search::SerialNum flushedSerial = 0, bool proceedImmediately = true)
        : SimpleTarget("anon", flushedSerial, proceedImmediately)
    { }

    SimpleTarget(const std::string &name, search::SerialNum flushedSerial = 0, bool proceedImmediately = true)
        : SimpleTarget(name, Type::OTHER, flushedSerial, proceedImmediately)
    { }

    Time getLastFlushTime() const override { return vespalib::system_clock::now(); }

    SerialNum getFlushedSerialNum() const override {
        LOG(info, "SimpleTarget(%s)::getFlushedSerialNum() = %" PRIu64, getName().c_str(), _flushedSerial.load(std::memory_order_relaxed));
        return _flushedSerial.load(std::memory_order_relaxed);
    }

    Task::UP initFlush(SerialNum currentSerial, std::shared_ptr<search::IFlushToken>) override {
        LOG(info, "SimpleTarget(%s)::initFlush(%" PRIu64 ")", getName().c_str(), currentSerial);
        _currentSerial = currentSerial;
        _initDone.countDown();
        return std::move(_task);
    }

};

class GCTarget : public SimpleTarget {
public:
    GCTarget(const vespalib::string &name, search::SerialNum flushedSerial)
        : SimpleTarget(name, Type::GC, flushedSerial)
    {}
};

class HighPriorityTarget : public SimpleTarget {
public:
    HighPriorityTarget(const vespalib::string &name, search::SerialNum flushedSerial, bool proceed)
        : SimpleTarget(name, Type::OTHER, flushedSerial, proceed)
    {}

    Priority getPriority() const override {
        return Priority::HIGH;
    }
};

class AssertedTarget : public SimpleTarget {
public:
    mutable bool _mgain;
    mutable bool _serial;

public:
    using SP = std::shared_ptr<AssertedTarget>;

    AssertedTarget()
        : SimpleTarget("anon"),
          _mgain(false),
          _serial(false)
    { }

    MemoryGain getApproxMemoryGain() const override {
        LOG_ASSERT(_mgain == false);
        _mgain = true;
        return SimpleTarget::getApproxMemoryGain();
    }

    search::SerialNum getFlushedSerialNum() const override {
        LOG_ASSERT(_serial == false);
        _serial = true;
        return SimpleTarget::getFlushedSerialNum();
    }
};

class SimpleStrategy : public IFlushStrategy {
public:
    using SP = std::shared_ptr<SimpleStrategy>;
    enum class OrderBy {INDEX_OF, SERIAL};
    std::vector<IFlushTarget::SP> _targets;
    OrderBy                       _orderBy;

    struct CompareIndexOf {
        CompareIndexOf(const SimpleStrategy &flush) : _flush(flush) { }
        bool operator () (const FlushContext::SP &lhs, const FlushContext::SP &rhs) const {
            return _flush.compare(lhs->getTarget(), rhs->getTarget());
        }
        const SimpleStrategy &_flush;
    };

    FlushContext::List getFlushTargets(const FlushContext::List& targetList,
                                       const flushengine::TlsStatsMap&,
                                       const flushengine::ActiveFlushStats&) const override {
        FlushContext::List fv(targetList);
        if (_orderBy == OrderBy::INDEX_OF) {
            std::sort(fv.begin(), fv.end(), CompareIndexOf(*this));
        } else {
            std::sort(fv.begin(), fv.end(), [](const auto & a, const auto & b) {
                return a->getTarget()->getFlushedSerialNum() < b->getTarget()->getFlushedSerialNum(); }
            );
        }
        return fv;
    }

    bool
    compare(const IFlushTarget::SP &lhs, const IFlushTarget::SP &rhs) const
    {
        LOG(info, "SimpleStrategy::compare(%p, %p)", lhs.get(), rhs.get());
        return indexOf(lhs) < indexOf(rhs);
    }

    SimpleStrategy(OrderBy orderBy) noexcept : _targets(), _orderBy(orderBy) {}

    uint32_t
    indexOf(const IFlushTarget::SP &target) const
    {
        IFlushTarget *raw = target.get();
        CachedFlushTarget *cached = dynamic_cast<CachedFlushTarget*>(raw);
        if (cached != nullptr) {
            raw = cached->getFlushTarget().get();
        }
        WrappedFlushTarget *wrapped = dynamic_cast<WrappedFlushTarget *>(raw);
        if (wrapped != nullptr) {
            raw = wrapped->getFlushTarget().get();
        }
        for (uint32_t i = 0, len = _targets.size(); i < len; ++i) {
            if (raw == _targets[i].get()) {
                LOG(info, "Index of target %p is %d.", raw, i);
                return i;
            }
        }
        LOG(info, "Target %p not found.", raw);
        return -1;
    }
};

class NoFlushStrategy : public SimpleStrategy
{
public:
    NoFlushStrategy() noexcept : SimpleStrategy(OrderBy::INDEX_OF) {}
    FlushContext::List getFlushTargets(const FlushContext::List &, const flushengine::TlsStatsMap &, const flushengine::ActiveFlushStats&) const override {
        return {};
    }
};

// --------------------------------------------------------------------------------
//
// Tests.
//
// --------------------------------------------------------------------------------

class AppendTask : public FlushTask
{
public:
    AppendTask(const vespalib::string & name, std::vector<vespalib::string> & list, vespalib::Gate & done) :
        _list(list),
        _done(done),
        _name(name)
    { }
    void run() override {
        _list.push_back(_name);
        _done.countDown();
    }
    search::SerialNum getFlushSerial() const override { return 0u; }
    std::vector<vespalib::string> & _list;
    vespalib::Gate    & _done;
    vespalib::string    _name;
};


struct Fixture
{
    std::shared_ptr<flushengine::ITlsStatsFactory> tlsStatsFactory;
    SimpleStrategy::SP strategy;
    FlushEngine engine;

    Fixture(uint32_t numThreads, vespalib::duration idleInterval, SimpleStrategy::SP strategy_)
        : tlsStatsFactory(std::make_shared<SimpleTlsStatsFactory>()),
          strategy(strategy_),
          engine(tlsStatsFactory, strategy, numThreads, idleInterval)
    { }

    Fixture(uint32_t numThreads, vespalib::duration idleInterval)
        : Fixture(numThreads, idleInterval, std::make_shared<SimpleStrategy>(SimpleStrategy::OrderBy::INDEX_OF))
    { }

    void putFlushHandler(const vespalib::string &docTypeName, IFlushHandler::SP handler) {
        engine.putFlushHandler(DocTypeName(docTypeName), handler);
    }

    void addTargetToStrategy(IFlushTarget::SP target) {
        strategy->_targets.push_back(std::move(target));
    }

    std::shared_ptr<SimpleHandler> addSimpleHandler(Targets targets) {
        auto handler = std::make_shared<SimpleHandler>(targets, "handler", 20);
        engine.putFlushHandler(DocTypeName("handler"), handler);
        engine.start();
        return handler;
    }

    void assertOldestSerial(SimpleHandler &handler, search::SerialNum expOldestSerial) {
        using namespace std::chrono_literals;
        for (int pass = 0; pass < 600; ++pass) {
            std::this_thread::sleep_for(100ms);
            if (handler.oldest_serial() == expOldestSerial) {
                break;
            }
        }
        EXPECT_EQUAL(expOldestSerial, handler.oldest_serial());
    }
};

TEST("require that leaf defaults are sane") {
    test::DummyFlushTarget leaf("dummy");
    EXPECT_FALSE(leaf.needUrgentFlush());
    EXPECT_EQUAL(0.0, leaf.get_replay_operation_cost());
    EXPECT_TRUE(IFlushTarget::Priority::NORMAL == leaf.getPriority());
    EXPECT_TRUE(50 == static_cast<int>(IFlushTarget::Priority::NORMAL));
    EXPECT_TRUE(100 == static_cast<int>(IFlushTarget::Priority::HIGH));
    EXPECT_TRUE(IFlushTarget::Priority::NORMAL < IFlushTarget::Priority::HIGH);
    EXPECT_TRUE(IFlushTarget::Priority::HIGH > IFlushTarget::Priority::NORMAL);
}

TEST_F("require that strategy controls flush target", Fixture(1, IINTERVAL))
{
    vespalib::Gate fooG, barG;
    std::vector<vespalib::string> order;
    auto foo = std::make_shared<SimpleTarget>(std::make_unique<AppendTask>("foo", order, fooG), "foo");
    auto bar = std::make_shared<SimpleTarget>(std::make_unique<AppendTask>("bar", order, barG), "bar");
    f.addTargetToStrategy(foo);
    f.addTargetToStrategy(bar);

    auto handler = std::make_shared<SimpleHandler>(Targets({bar, foo}), "anon");
    f.putFlushHandler("anon", handler);
    f.engine.start();

    EXPECT_TRUE(fooG.await(LONG_TIMEOUT));
    EXPECT_TRUE(barG.await(LONG_TIMEOUT));
    EXPECT_EQUAL(2u, order.size());
    EXPECT_EQUAL("foo", order[0]);
    EXPECT_EQUAL("bar", order[1]);
}

TEST_F("require that zero handlers does not core", Fixture(2, 50ms))
{
    f.engine.start();
}

TEST_F("require that zero targets does not core", Fixture(2, 50ms))
{
    f.putFlushHandler("foo", std::make_shared<SimpleHandler>(Targets(), "foo"));
    f.putFlushHandler("bar", std::make_shared<SimpleHandler>(Targets(), "bar"));
    f.engine.start();
}

TEST_F("require that oldest serial is found", Fixture(1, IINTERVAL))
{
    auto foo = std::make_shared<SimpleTarget>("foo", 10);
    auto bar = std::make_shared<SimpleTarget>("bar", 20);
    f.addTargetToStrategy(foo);
    f.addTargetToStrategy(bar);

    auto handler = std::make_shared<SimpleHandler>(Targets({foo, bar}), "anon", 25);
    f.putFlushHandler("anon", handler);
    f.engine.start();

    EXPECT_TRUE(handler->_done.await(LONG_TIMEOUT));
    EXPECT_EQUAL(25ul, handler->_oldestSerial);
    FlushDoneHistory handlerFlushDoneHistory(handler->getFlushDoneHistory());
    if (handlerFlushDoneHistory.size() == 2u) {
        // Lost sample of oldest serial might happen when system load is high
        EXPECT_EQUAL(FlushDoneHistory({ 10, 25 }), handlerFlushDoneHistory);
    } else {
        EXPECT_EQUAL(FlushDoneHistory({ 10, 20, 25 }), handlerFlushDoneHistory);
    }
}

TEST_F("require that GC targets are not considered when oldest serial is found", Fixture(1, IINTERVAL))
{
    auto foo = std::make_shared<SimpleTarget>("foo", 5);
    auto bar = std::make_shared<GCTarget>("bar", 10);
    auto baz = std::make_shared<SimpleTarget>("baz", 20);
    f.addTargetToStrategy(foo);
    f.addTargetToStrategy(bar);
    f.addTargetToStrategy(baz);

    auto handler = std::make_shared<SimpleHandler>(Targets({foo, bar, baz}), "handler", 25);
    f.putFlushHandler("handler", handler);
    f.engine.start();

    // The targets are flushed in sequence: 'foo', 'bar', 'baz'
    EXPECT_TRUE(handler->_done.await(LONG_TIMEOUT));
    EXPECT_EQUAL(25ul, handler->_oldestSerial);

    // Before anything is flushed the oldest serial is 5.
    // After 'foo' has been flushed the oldest serial is 20 as GC target 'bar' is not considered.
    FlushDoneHistory history = handler->getFlushDoneHistory();
    EXPECT_TRUE(history.end() == std::find(history.begin(), history.end(), 10));
    auto last_unique = std::unique(history.begin(), history.end());
    history.erase(last_unique, history.end());
    EXPECT_EQUAL(FlushDoneHistory({ 5, 20, 25 }), history);
}

TEST_F("require that oldest serial is found in group", Fixture(2, IINTERVAL))
{
    auto fooT1 = std::make_shared<SimpleTarget>("fooT1", 10);
    auto fooT2 = std::make_shared<SimpleTarget>("fooT2", 20);
    auto barT1 = std::make_shared<SimpleTarget>("barT1",  5);
    auto barT2 = std::make_shared<SimpleTarget>("barT2", 15);
    f.addTargetToStrategy(fooT1);
    f.addTargetToStrategy(fooT2);
    f.addTargetToStrategy(barT1);
    f.addTargetToStrategy(barT2);

    auto fooH = std::make_shared<SimpleHandler>(Targets({fooT1, fooT2}), "fooH", 25);
    f.putFlushHandler("foo", fooH);

    auto barH = std::make_shared<SimpleHandler>(Targets({barT1, barT2}), "barH", 20);
    f.putFlushHandler("bar", barH);

    f.engine.start();

    EXPECT_TRUE(fooH->_done.await(LONG_TIMEOUT));
    EXPECT_EQUAL(25ul, fooH->_oldestSerial);
    // [ 10, 25 ], [10, 10, 25], [ 10, 25, 25 ] and [ 10, 20, 25 ] are
    // legal histories
    FlushDoneHistory fooHFlushDoneHistory(fooH->getFlushDoneHistory());
    if (fooHFlushDoneHistory != FlushDoneHistory({ 10, 25 }) &&
        fooHFlushDoneHistory != FlushDoneHistory({ 10, 10, 25 }) &&
        fooHFlushDoneHistory != FlushDoneHistory({ 10, 25, 25 })) {
        EXPECT_EQUAL(FlushDoneHistory({ 10, 20, 25 }), fooHFlushDoneHistory);
    }
    EXPECT_TRUE(barH->_done.await(LONG_TIMEOUT));
    EXPECT_EQUAL(20ul, barH->_oldestSerial);
    // [ 5, 20 ], [5, 5, 20], [ 5, 20, 20 ] and [ 5, 15, 20 ] are
    // legal histories
    FlushDoneHistory barHFlushDoneHistory(barH->getFlushDoneHistory());
    if (barHFlushDoneHistory != FlushDoneHistory({ 5, 20 }) &&
        barHFlushDoneHistory != FlushDoneHistory({ 5, 5, 20 }) &&
        barHFlushDoneHistory != FlushDoneHistory({ 5, 20, 20 })) {
        EXPECT_EQUAL(FlushDoneHistory({ 5, 15, 20 }), barHFlushDoneHistory);
    }
}

TEST_F("require that target can refuse flush", Fixture(2, IINTERVAL))
{
    auto target = std::make_shared<SimpleTarget>();
    auto handler = std::make_shared<SimpleHandler>(Targets({target}));
    target->_task = searchcorespi::FlushTask::UP();
    f.putFlushHandler("anon", handler);
    f.engine.start();

    EXPECT_TRUE(target->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(!target->_taskDone.await(SHORT_TIMEOUT));
    EXPECT_TRUE(!handler->_done.await(SHORT_TIMEOUT));
}

TEST_F("require that targets are flushed when nothing new to flush", Fixture(2, IINTERVAL))
{
    auto target = std::make_shared<SimpleTarget>("anon", 5); // oldest unflushed serial num = 5
    auto handler = std::make_shared<SimpleHandler>(Targets({target}), "anon", 4); // current serial num = 4
    f.putFlushHandler("anon", handler);
    f.engine.start();

    EXPECT_TRUE(target->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target->_taskDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(handler->_done.await(LONG_TIMEOUT));
}

TEST_F("require that flushing targets are skipped", Fixture(2, IINTERVAL))
{
    auto foo = std::make_shared<SimpleTarget>("foo");
    auto bar = std::make_shared<SimpleTarget>("bar");
    f.addTargetToStrategy(foo);
    f.addTargetToStrategy(bar);

    auto handler = std::make_shared<SimpleHandler>(Targets({bar, foo}));
    f.putFlushHandler("anon", handler);
    f.engine.start();

    EXPECT_TRUE(foo->_taskDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(bar->_taskDone.await(LONG_TIMEOUT)); /* this is the key check */
}

TEST_F("require that updated targets are not skipped", Fixture(2, IINTERVAL))
{
    auto target = std::make_shared<SimpleTarget>("target", 1);
    f.addTargetToStrategy(target);

    auto handler = std::make_shared<SimpleHandler>(Targets({target}), "handler", 0);
    f.putFlushHandler("handler", handler);
    f.engine.start();

    EXPECT_TRUE(target->_taskDone.await(LONG_TIMEOUT));
}

TEST("require that threaded target works")
{
    SimpleExecutor executor;
    SimpleGetSerialNum getSerialNum;
    auto target = std::make_shared<ThreadedFlushTarget>(executor, getSerialNum, std::make_shared<SimpleTarget>());

    EXPECT_FALSE(executor._done.await(SHORT_TIMEOUT));
    EXPECT_TRUE(target->initFlush(0, std::make_shared<search::FlushToken>()));
    EXPECT_TRUE(executor._done.await(LONG_TIMEOUT));
}

TEST("require that cached target works")
{
    auto target = std::make_shared<CachedFlushTarget>(std::make_shared<AssertedTarget>());
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQUAL(0l, target->getApproxMemoryGain().getBefore());
        EXPECT_EQUAL(0l, target->getApproxMemoryGain().getAfter());
        EXPECT_EQUAL(0ul, target->getFlushedSerialNum());
    }
}

TEST_F("require that trigger flush works", Fixture(2, IINTERVAL))
{
    auto target = std::make_shared<SimpleTarget>("target", 1);
    f.addTargetToStrategy(target);

    auto handler = std::make_shared<SimpleHandler>(Targets({target}), "handler", 9);
    f.putFlushHandler("handler", handler);
    f.engine.start();
    f.engine.triggerFlush();
    EXPECT_TRUE(target->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target->_taskDone.await(LONG_TIMEOUT));
}

bool
asserCorrectHandlers(const FlushEngine::FlushMetaSet & current1, const std::vector<const char *> & targets)
{
    bool retval(targets.size() == current1.size());
    auto curr = current1.begin();
    if (retval) {
        for (const char * target : targets) {
            if (target != (curr++)->getName()) {
                return false;
            }
        }
    }
    return retval;
}

void
assertThatHandlersInCurrentSet(FlushEngine & engine, const std::vector<const char *> & targets)
{
    FlushEngine::FlushMetaSet current1 = engine.getCurrentlyFlushingSet();
    while ((current1.size() < targets.size()) || !asserCorrectHandlers(current1, targets)) {
        std::this_thread::sleep_for(1ms);
        current1 = engine.getCurrentlyFlushingSet();
    }
}

TEST_F("require that concurrency works", Fixture(2, 1ms))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 1, false);
    auto target2 = std::make_shared<SimpleTarget>("target2", 2, false);
    auto target3 = std::make_shared<SimpleTarget>("target3", 3, false);
    auto handler = std::make_shared<SimpleHandler>(Targets({target1, target2, target3}), "handler", 9);
    f.putFlushHandler("handler", handler);
    f.engine.start();

    EXPECT_TRUE(target1->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target2->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(!target3->_initDone.await(SHORT_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target1", "handler.target2"});
    EXPECT_TRUE(!target3->_initDone.await(SHORT_TIMEOUT));
    target1->_proceed.countDown();
    EXPECT_TRUE(target1->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target2", "handler.target3"});
    target3->_proceed.countDown();
    target2->_proceed.countDown();
}

TEST_F("require that there is room for one and only one high pri target",
       Fixture(2, 1ms, std::make_unique<SimpleStrategy>(SimpleStrategy::OrderBy::SERIAL)))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 1, false);
    auto target2 = std::make_shared<SimpleTarget>("target2", 2, false);
    auto target3 = std::make_shared<HighPriorityTarget>("target3", 3, false);
    auto target4 = std::make_shared<HighPriorityTarget>("target4", 4, false);
    auto handler = std::make_shared<SimpleHandler>(Targets({target1, target2, target3, target4}), "handler", 9);
    f.putFlushHandler("handler", handler);
    f.engine.start();
    EXPECT_EQUAL(2u, f.engine.maxConcurrentNormal());
    EXPECT_EQUAL(3u, f.engine.maxConcurrentTotal());
    EXPECT_EQUAL(f.engine.maxConcurrentTotal(), f.engine.get_executor().getNumThreads());

    EXPECT_TRUE(target1->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target2->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target3->_initDone.await(LONG_TIMEOUT));
    EXPECT_FALSE(target4->_initDone.await(SHORT_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target1", "handler.target2", "handler.target3"});
    target1->_proceed.countDown();
    EXPECT_TRUE(target1->_taskDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target4->_initDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target2", "handler.target3", "handler.target4"});
    target2->_proceed.countDown();
    EXPECT_TRUE(target2->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target3", "handler.target4"});
    target3->_proceed.countDown();
    EXPECT_TRUE(target3->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target4"});
    target4->_proceed.countDown();
    EXPECT_TRUE(target4->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {});
}

TEST_F("require that high priority does not jump the queue",
       Fixture(2, 1ms, std::make_unique<SimpleStrategy>(SimpleStrategy::OrderBy::SERIAL)))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 1, false);
    auto target2 = std::make_shared<SimpleTarget>("target2", 2, false);
    auto target3 = std::make_shared<SimpleTarget>("target3", 3, false);
    auto target4 = std::make_shared<HighPriorityTarget>("target4", 4, false);
    auto handler = std::make_shared<SimpleHandler>(Targets({target1, target2, target3, target4}), "handler", 9);
    f.putFlushHandler("handler", handler);
    f.engine.start();
    EXPECT_EQUAL(2u, f.engine.maxConcurrentNormal());
    EXPECT_EQUAL(3u, f.engine.maxConcurrentTotal());
    EXPECT_EQUAL(f.engine.maxConcurrentTotal(), f.engine.get_executor().getNumThreads());

    EXPECT_TRUE(target1->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target2->_initDone.await(LONG_TIMEOUT));
    EXPECT_FALSE(target3->_initDone.await(SHORT_TIMEOUT));
    EXPECT_FALSE(target4->_initDone.await(SHORT_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target1", "handler.target2"});
    target1->_proceed.countDown();
    EXPECT_TRUE(target1->_taskDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target3->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target4->_initDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target2", "handler.target3", "handler.target4"});
    target2->_proceed.countDown();
    EXPECT_TRUE(target2->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target3", "handler.target4"});
    target3->_proceed.countDown();
    EXPECT_TRUE(target3->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target4"});
    target4->_proceed.countDown();
    EXPECT_TRUE(target4->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {});
}

TEST_F("require that concurrency works with triggerFlush", Fixture(2, 1ms))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 1, false);
    auto target2 = std::make_shared<SimpleTarget>("target2", 2, false);
    auto target3 = std::make_shared<SimpleTarget>("target3", 3, false);
    auto handler = std::make_shared<SimpleHandler>(Targets({target1, target2, target3}), "handler", 9);
    f.putFlushHandler("handler", handler);
    std::thread thread([this]() { f.engine.triggerFlush(); });
    std::this_thread::sleep_for(1s);
    f.engine.start();
    
    EXPECT_TRUE(target1->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(target2->_initDone.await(LONG_TIMEOUT));
    EXPECT_TRUE(!target3->_initDone.await(SHORT_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target1", "handler.target2"});
    EXPECT_TRUE(!target3->_initDone.await(SHORT_TIMEOUT));
    target1->_proceed.countDown();
    EXPECT_TRUE(target1->_taskDone.await(LONG_TIMEOUT));
    assertThatHandlersInCurrentSet(f.engine, {"handler.target2", "handler.target3"});
    target3->_proceed.countDown();
    target2->_proceed.countDown();
    thread.join();
}

TEST_F("require that state explorer can list flush targets", Fixture(1, 1ms))
{
    auto target = std::make_shared<SimpleTarget>("target1", 100, false);
    f.putFlushHandler("handler",
                      std::make_shared<SimpleHandler>(
                              Targets({target, std::make_shared<SimpleTarget>("target2", 50, true)}),
                              "handler", 9));
    f.engine.start();
    target->_initDone.await(LONG_TIMEOUT);
    target->_taskStart.await(LONG_TIMEOUT);

    FlushEngineExplorer explorer(f.engine);
    Slime state;
    SlimeInserter inserter(state);
    explorer.get_state(inserter, true);

    Inspector &all = state.get()["allTargets"];
    EXPECT_EQUAL(2u, all.children());
    EXPECT_EQUAL("handler.target2", all[0]["name"].asString().make_string());
    EXPECT_EQUAL(50, all[0]["flushedSerialNum"].asLong());
    EXPECT_EQUAL("handler.target1", all[1]["name"].asString().make_string());
    EXPECT_EQUAL(100, all[1]["flushedSerialNum"].asLong());

    Inspector &flushing = state.get()["flushingTargets"];
    EXPECT_EQUAL(1u, flushing.children());
    EXPECT_EQUAL("handler.target1", flushing[0]["name"].asString().make_string());

    target->_proceed.countDown();
    target->_taskDone.await(LONG_TIMEOUT);
}

TEST_F("require that oldest serial is updated when closing engine", Fixture(1, 100ms))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 10, false);
    auto handler = f.addSimpleHandler({ target1 });
    TEST_DO(f.assertOldestSerial(*handler, 10));
    target1->_proceed.countDown();
    f.engine.close();
    EXPECT_EQUAL(20u, handler->_oldestSerial);
}

TEST_F("require that oldest serial is updated when finishing priority flush strategy", Fixture(1, 100ms, std::make_shared<NoFlushStrategy>()))
{
    auto target1 = std::make_shared<SimpleTarget>("target1", 10, true);
    auto handler = f.addSimpleHandler({ target1 });
    TEST_DO(f.assertOldestSerial(*handler, 10));
    f.engine.setStrategy(std::make_shared<SimpleStrategy>(SimpleStrategy::OrderBy::INDEX_OF));
    EXPECT_EQUAL(20u, handler->_oldestSerial);
}

TEST("the oldest start time is tracked per flush handler in ActiveFlushStats")
{
    using seconds = std::chrono::seconds;
    using vespalib::system_time;
    system_time now = vespalib::system_clock::now();
    system_time t1 = now + seconds(1);
    system_time t2 = now + seconds(2);
    system_time t3 = now + seconds(3);
    system_time t4 = now + seconds(4);
    ActiveFlushStats stats;
    EXPECT_FALSE(stats.oldest_start_time("h1").has_value());
    stats.set_start_time("h1", t2);
    stats.set_start_time("h2", t4);
    EXPECT_EQUAL(t2, stats.oldest_start_time("h1").value());
    EXPECT_EQUAL(t4, stats.oldest_start_time("h2").value());

    stats.set_start_time("h1", t1);
    EXPECT_EQUAL(t1, stats.oldest_start_time("h1").value());
    stats.set_start_time("h1", t3);
    EXPECT_EQUAL(t1, stats.oldest_start_time("h1").value());
}


TEST_MAIN()
{
    TEST_RUN_ALL();
}
