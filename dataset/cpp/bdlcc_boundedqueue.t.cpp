// bdlcc_boundedqueue.t.cpp                                           -*-C++-*-

#include <bdlcc_boundedqueue.h>

#include <bslim_testutil.h>

#include <bdlf_bind.h>

#include <bslma_allocator.h>
#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bslma_testallocator.h>
#include <bslma_testallocatormonitor.h>

#include <bslmt_barrier.h>
#include <bslmt_condition.h>
#include <bslmt_lockguard.h>
#include <bslmt_mutex.h>
#include <bslmt_threadgroup.h>
#include <bslmt_threadutil.h>

#include <bsls_assert.h>
#include <bsls_asserttest.h>
#include <bsls_atomic.h>
#include <bsls_atomicoperations.h>
#include <bsls_systemtime.h>
#include <bsls_timeinterval.h>
#include <bsls_types.h>

#include <bsltf_moveonlyalloctesttype.h>
#include <bsltf_movablealloctesttype.h>

#include <bsl_cstring.h>
#include <bsl_cstdlib.h>
#include <bsl_iostream.h>
#include <bsl_ostream.h>
#include <bsl_string.h>
#include <bsl_unordered_map.h>
#include <bsl_vector.h>

using namespace BloombergLP;
using namespace bsl;

// ============================================================================
//                             TEST PLAN
// ----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test implements a concurrent FIFO queue container with
// bounded capacity.  The primary manipulators are the methods for adding
// elements ('pushBack') and emptying the queue ('removeAll').  The provided
// basic accessors are the methods for obtaining the allocator ('allocator')
// and the number of elements in the queue ('numElements').  The manipulator
// 'popFront' will be used extensively to verify the value of resultant queues.
// The basic functionality of the queue will be verified initially with a
// single thread of execution, and then concurrency concerns will be addressed.
// Effort is made to use only the primary manipulators, basic accessors, and
// 'popFront' whenever possible, thus making every test case independent.
//
// Note that the move-semantics testing types are not universally supported.
// Move-only type is supported in C++11 mode.  Allocating move-only type is
// supported in C++11 mode.  'IncorrectlyMatchingMoveConstructorTestType' works
// in C++03 mode.
//
// Global Concerns:
//: o The test driver is robust w.r.t. reuse in other, similar components.
//: o ACCESSOR methods are declared 'const'.
//: o CREATOR & MANIPULATOR pointer/reference parameters are declared 'const'.
//: o No memory is ever allocated from the global allocator.
//: o Any allocated memory is always from the object allocator.
//: o An object's value is independent of the allocator used to supply memory.
//: o Injected exceptions are safely propagated during memory allocation.
//: o Precondition violations are detected in appropriate build modes.
//
// Global Assumptions:
//: o All explicit memory allocations are presumed to use the global, default,
//:   or object allocator.
//: o ACCESSOR methods are 'const' thread-safe.
// ----------------------------------------------------------------------------
// [ 2] BoundedQueue(bsl::size_t capacity, bslma::Allocator bA = 0);
// [ 2] ~BoundedQueue();
// [ 2] int popFront(TYPE *value);
// [ 2] int pushBack(const TYPE& value);
// [ 9] int pushBack(bslmf::MovableRef<TYPE> value);
// [ 2] void removeAll();
// [ 7] int tryPopFront(TYPE *value);
// [ 6] int tryPushBack(const TYPE& value);
// [ 9] int tryPushBack(bslmf::MovableRef<TYPE> value);
// [ 5] void disablePopFront();
// [ 5] void disablePushBack();
// [ 5] void enablePopFront();
// [ 5] void enablePushBack();
// [ 4] bsl::size_t capacity() const;
// [ 4] bool isEmpty() const;
// [ 4] bool isFull() const;
// [ 5] bool isPopFrontDisabled() const;
// [ 5] bool isPushBackDisabled() const;
// [ 4] bsl::size_t numElements() const;
// [ 8] int waitUntilEmpty() const;
// [ 4] bslma::Allocator *allocator() const;
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [16] USAGE EXAMPLE
// [ 3] Obj& gg(Obj *object, const char *spec);
// [ 3] int ggg(Obj *object, const char *spec);
// [ 2] CONCERN: 0 == e_SUCCESS
// [ 6] CONCERN: 'tryPushBack(MovableRef<TYPE>)' basic functionality
// [ 9] CONCERN: 'popFront' and 'tryPopFront' honor move-semantics
// [10] CONCERN: template requirements
// [11] CONCERN: ordering guarantee
// [12] DRQS 153332608: 'waitUntilEmpty' RACE WITH 'popFront'
// [13] DRQS 164984269: 'removeAll' STARTED/FINISHED ISSUE
// [14] DRQS 153332608: 'pushBack', 'pushBack', 'waitUntilEmpty'
// [15] DRQS 168011541: 'waitUntilEmpty' RACE WITH 'disablePopFront'
// ----------------------------------------------------------------------------

// ============================================================================
//                     STANDARD BDE ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        cout << "Error " __FILE__ "(" << line << "): " << message
             << "    (failed)" << endl;

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace

// ============================================================================
//               STANDARD BDE TEST DRIVER MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT       BSLIM_TESTUTIL_ASSERT
#define ASSERTV      BSLIM_TESTUTIL_ASSERTV

#define LOOP_ASSERT  BSLIM_TESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLIM_TESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLIM_TESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLIM_TESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLIM_TESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLIM_TESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLIM_TESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLIM_TESTUTIL_LOOP6_ASSERT

#define Q            BSLIM_TESTUTIL_Q   // Quote identifier literally.
#define P            BSLIM_TESTUTIL_P   // Print identifier and value.
#define P_           BSLIM_TESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLIM_TESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLIM_TESTUTIL_L_  // current Line number

// ============================================================================
//                        GLOBAL MACROS FOR TESTING
// ----------------------------------------------------------------------------

#if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES) \
         && defined(BSLS_COMPILERFEATURES_SUPPORT_ALIAS_TEMPLATES)
#    define BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
    // This macro indicates whether the component uses C++11 r-value references
    // to implement 'bslmf::MovableRef<TYPE>'.  It will evaluate to 'false' for
    // C++03 implementations and to 'true' for proper C++11 implementations.
    // For partial C++11 implementations it may evaluate to 'false' because
    // both r-value reference and alias templates need to be supported.
#endif

// ============================================================================
//                   GLOBAL STRUCTS/FUNCTIONS FOR TESTING
// ----------------------------------------------------------------------------

                           // ====================
                           // AllocExceptionHelper
                           // ====================

class AllocExceptionHelper {
    // DATA
    void             *d_memory_p;
    bslma::Allocator *d_allocator_p;

  public:
    // TRAITS
    BSLMF_NESTED_TRAIT_DECLARATION(AllocExceptionHelper,
                                   bslma::UsesBslmaAllocator);

    // CREATORS
    explicit
    AllocExceptionHelper(bslma::Allocator *allocator)
        // Create an 'AllocExceptionHelper' object using the specified
        // 'allocator' to obtain memory.
    : d_allocator_p(allocator)
    {
        d_memory_p = d_allocator_p->allocate(1);
    }

    AllocExceptionHelper(const AllocExceptionHelper&  obj,
                         bslma::Allocator            *allocator)
        // Create an 'AllocExceptionHelper' object having the value of the
        // specified 'obj', using the specified 'allocator' to obtain memory.
    : d_allocator_p(allocator)
    {
        (void)obj;

        d_memory_p = d_allocator_p->allocate(1);
    }

    ~AllocExceptionHelper()
        // Destroy this object.
    {
        d_allocator_p->deallocate(d_memory_p);
    }

    // MANIPULATORS
    AllocExceptionHelper& operator=(const AllocExceptionHelper& rhs)
        // Assign to this object the value of the specified 'rhs' object, and
        // return a reference providing modifiable access to this object.
    {
        (void)rhs;

        void *memory = d_allocator_p->allocate(1);

        d_allocator_p->deallocate(d_memory_p);
        d_memory_p = memory;

        return *this;
    }

    // ACCESSORS

                                  // Aspects

    bslma::Allocator *allocator() const
        // Return the allocator used by this object to supply memory.
    {
        return d_allocator_p;
    }
};

                                // ==========
                                // MoveTester
                                // ==========

class MoveTester {
    // DATA
    bool  d_moved;
    int  *d_moveCounter_p;
    int   d_value;

    // NOT IMPLEMENTED
    MoveTester(const MoveTester& other);
    MoveTester& operator=(const MoveTester& other);

  public:
    // CREATORS
    explicit MoveTester(int value, int *moveCounter = 0);
        // Construct a new 'MoveTester' object with the specified 'value'.
        // Optionally specify a 'moveCounter' that, if specified, will be
        // incremented when this object is moved from.

    explicit MoveTester(bslmf::MovableRef<MoveTester> other);
        // Move-construct a new 'MoveTester' object from the specified 'other'.

    // MANIPULATORS
    MoveTester& operator=(bslmf::MovableRef<MoveTester> other);
        // Move-assign the value of the specified 'other' object to this one.
        // Set 'isMoved' to 'true', and if non-null value is specified at
        // construction time also set '*d_moveCounter_p' to 'true'

    void reset();
        // Reset 'isMoved' to false.

    // ACCESSORS
    bool isMoved() const;
        // Return the value of the moved non-salient attribute.

    int value() const;
        // Return the value of this object.
};

                                // ----------
                                // MoveTester
                                // ----------

// CREATORS
MoveTester::MoveTester(int value, int *moveCounter)
: d_moved(false)
, d_moveCounter_p(moveCounter)
, d_value(value)
{
}

MoveTester::MoveTester(bslmf::MovableRef<MoveTester> other)
{
    typedef bslmf::MovableRefUtil MoveUtil;

    d_value         = MoveUtil::access(other).d_value;
    d_moved         = false;
    d_moveCounter_p = MoveUtil::access(other).d_moveCounter_p;

    MoveUtil::access(other).d_moved = true;
    if (MoveUtil::access(other).d_moveCounter_p) {
        ++(*MoveUtil::access(other).d_moveCounter_p);
    }
}

// MANIPULATORS
MoveTester& MoveTester::operator=(bslmf::MovableRef<MoveTester> other)
{
    typedef bslmf::MovableRefUtil MoveUtil;

    d_value = MoveUtil::access(other).d_value;
    MoveUtil::access(other).d_moved = true;
    if (MoveUtil::access(other).d_moveCounter_p) {
        ++(*MoveUtil::access(other).d_moveCounter_p);
    }

    return *this;
}

// ACCESSORS
bool MoveTester::isMoved() const
{
    return d_moved;
}

int MoveTester::value() const
{
    return d_value;
}

             // ================================================
             // class IncorrectlyMatchingMoveConstructorTestType
             // ================================================

class IncorrectlyMatchingMoveConstructorTestType {
    // This class is convertible from any type that has a 'data' method
    // returning a 'int' value.  It is used to facilitate testing that the
    // implementation of 'bdlcc::BoundedQueue' does not pass a
    // 'bslmf::MovableRef<T>' object to a class whose interface does not
    // support it (in C++03 mode).

    // DATA
    int d_data;  // value not meaningful

  public:
    // CREATORS
    IncorrectlyMatchingMoveConstructorTestType();
        // Create a 'IncorrectlyMatchingMoveConstructorTestType' object having
        // the default value.

    explicit IncorrectlyMatchingMoveConstructorTestType(int data);
        // Create a 'IncorrectlyMatchingMoveConstructorTestType' object having
        // the specified 'data' value.

    IncorrectlyMatchingMoveConstructorTestType(
                   const IncorrectlyMatchingMoveConstructorTestType& original);
        // Create a 'IncorrectlyMatchingMoveConstructorTestType' object having
        // the same value as the specified 'original' object.

    template <class TYPE>
    IncorrectlyMatchingMoveConstructorTestType(const TYPE& other);  // IMPLICIT
        // Create a 'IncorrectlyMatchingMoveConstructorTestType' object having
        // the same value as the specified 'other' object of (template
        // parameter) 'TYPE'.

    // MANIPULATORS
    IncorrectlyMatchingMoveConstructorTestType& operator=(
                        const IncorrectlyMatchingMoveConstructorTestType& rhs);
        // Assign to this object the value of the specified 'rhs' object, and
        // return a non-'const' reference to this object.

    template <class TYPE>
    IncorrectlyMatchingMoveConstructorTestType& operator=(const TYPE& rhs);
        // Assign to this object the value of the specified 'rhs' object of
        // (template parameter) 'TYPE', and return a non-'const' reference to
        // this object.

    // ACCESSORS
    int data() const;
        // Return the (meaningless) value held by this object.
};

             // ------------------------------------------------
             // class IncorrectlyMatchingMoveConstructorTestType
             // ------------------------------------------------

// CREATORS
IncorrectlyMatchingMoveConstructorTestType::
IncorrectlyMatchingMoveConstructorTestType()
: d_data(0)
{
}

IncorrectlyMatchingMoveConstructorTestType::
IncorrectlyMatchingMoveConstructorTestType(int data)
: d_data(data)
{
}

IncorrectlyMatchingMoveConstructorTestType::
IncorrectlyMatchingMoveConstructorTestType(
                    const IncorrectlyMatchingMoveConstructorTestType& original)
: d_data(original.d_data)
{
}

template <class TYPE>
IncorrectlyMatchingMoveConstructorTestType::
IncorrectlyMatchingMoveConstructorTestType(const TYPE& other)
: d_data(other.data())
{
}

// MANIPULATORS
IncorrectlyMatchingMoveConstructorTestType&
IncorrectlyMatchingMoveConstructorTestType::operator=(
                         const IncorrectlyMatchingMoveConstructorTestType& rhs)
{
    d_data = rhs.d_data;
    return *this;
}

template <class TYPE>
IncorrectlyMatchingMoveConstructorTestType&
IncorrectlyMatchingMoveConstructorTestType::operator=(const TYPE& rhs)
{
    d_data = rhs.data();
    return *this;
}

// ACCESSORS
int IncorrectlyMatchingMoveConstructorTestType::data() const
{
    return d_data;
}

                    // ---------------------------------
                    // class TemplateRequirementTestType
                    // ---------------------------------

class TemplateRequirementTestType {
    int *d_arg_p;

  public:
    // CREATORS
    explicit
    TemplateRequirementTestType(int *arg1)
        // Create a 'TemplateRequirementTestType' object using the specified
        // 'arg1'.
    : d_arg_p(arg1)
    {
        ++(*d_arg_p);
    }

    TemplateRequirementTestType(const TemplateRequirementTestType& obj)
        // Create an 'TemplateRequirementTestType' object having the value of
        // the specified 'obj'.
    : d_arg_p(obj.d_arg_p)
    {
        ++(*d_arg_p);
    }

    ~TemplateRequirementTestType()
        // Destroy this object.
    {
        --(*d_arg_p);
    }

    // MANIPULATORS
    TemplateRequirementTestType& operator=(
                                        const TemplateRequirementTestType& rhs)
        // Assign to this object the value of the specified 'rhs' object, and
        // return a reference providing modifiable access to this object.
    {
        --(*d_arg_p);
        d_arg_p = rhs.d_arg_p;
        ++(*d_arg_p);
        return *this;
    }
};

// ============================================================================
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

typedef bdlcc::BoundedQueue<int>          Obj;

typedef bdlcc::BoundedQueue<bsl::string>  AllocObj;

const int e_SUCCESS  = Obj::e_SUCCESS;
const int e_EMPTY    = Obj::e_EMPTY;
const int e_FULL     = Obj::e_FULL;
const int e_DISABLED = Obj::e_DISABLED;

const int k_DECISECOND = 100000;  // microseconds in 0.1 seconds

// ============================================================================
//                   GLOBAL METHODS FOR TESTING
// ----------------------------------------------------------------------------

static bsls::AtomicInt  s_continue;
static bslmt::Barrier  *s_barrier_p;

extern "C" void *deferredDisablePopFront(void *arg)
{
    Obj& mX = *static_cast<Obj *>(arg);

    bslmt::ThreadUtil::microSleep(0, 1);

    mX.disablePopFront();

    return 0;
}

static bsls::TimeInterval s_deferredPopFrontInterval;

extern "C" void *deferredPopFront(void *arg)
{
    s_deferredPopFrontInterval =
                      bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME);

    Obj& mX = *static_cast<Obj *>(arg);

    bslmt::ThreadUtil::microSleep(0, 1);

    Obj::value_type value;

    mX.popFront(&value);

    s_deferredPopFrontInterval =
                       bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME)
                     - s_deferredPopFrontInterval;

    return 0;
}

struct OrderingValue {
    bsls::Types::Uint64 d_pushThreadId;
    bsls::Types::Uint64 d_sequenceNumber;
};

typedef bdlcc::BoundedQueue<OrderingValue> OrderingObj;

struct OrderingPopData {
    OrderingObj                                              *d_obj_p;
    bsl::unordered_map<bsls::Types::Uint64, bsls::Types::Uint64>
                                                              d_sequenceNumber;
    bool                                                      d_isStrongTest;
};

extern "C" void *orderingPop(void *arg)
{
    OrderingPopData *data = static_cast<OrderingPopData *>(arg);
    OrderingObj&     mX   = *data->d_obj_p;

    OrderingObj::value_type value;

    while (1 < s_continue) {
        if (0 == mX.popFront(&value)) {
            bsls::Types::Uint64 pushThreadId   = value.d_pushThreadId;
            bsls::Types::Uint64 sequenceNumber = value.d_sequenceNumber;

            bsls::Types::Uint64& lastSequenceNumber =
                                          data->d_sequenceNumber[pushThreadId];

            if (data->d_isStrongTest) {
                ASSERTV(pushThreadId,
                        lastSequenceNumber,
                        sequenceNumber,
                        lastSequenceNumber + 1 == sequenceNumber);
            }
            else {
                ASSERTV(pushThreadId,
                        lastSequenceNumber,
                        sequenceNumber,
                        lastSequenceNumber < sequenceNumber);
            }

            lastSequenceNumber = sequenceNumber;
        }
    }

    return 0;
}

extern "C" void *orderingPush(void *arg)
{
    OrderingObj& mX = *static_cast<OrderingObj *>(arg);

    OrderingObj::value_type value;

    value.d_pushThreadId   = bslmt::ThreadUtil::selfIdAsUint64();
    value.d_sequenceNumber = 1;

    while (2 < s_continue) {
        if (0 == mX.pushBack(value)) {
            value.d_sequenceNumber += 1;
        }
    }

    return 0;
}

extern "C" void *orderingState(void *arg)
{
    OrderingObj& mX = *static_cast<OrderingObj *>(arg);

    int count = 0;

    while (3 < s_continue) {
        bslmt::ThreadUtil::microSleep(k_DECISECOND);

        if (3 == count % 10) {
            mX.disablePopFront();
        }
        else if (4 == count % 10) {
            mX.enablePopFront();
        }
        else if (8 == count % 10) {
            mX.disablePushBack();
        }
        else if (9 == count % 10) {
            mX.enablePushBack();
        }

        ++count;
    }

    return 0;
}

static char s_watchdogText[128];

void setWatchdogText(const char *value)
    // Assign the specified 'value' to be displayed if the watchdog expires.
{
    memcpy(s_watchdogText, value, strlen(value) + 1);
}

extern "C" void *watchdog(void *arg)
    // Watchdog function used to determine when a timeout should occur.  This
    // function returns without expiration if '0 == s_continue' before one
    // second elapses.  Upon expiration, 's_watchdogText' is displayed and the
    // program is aborted.
{
    if (arg) {
        setWatchdogText(static_cast<const char *>(arg));
    }

    const int MAX = 100;  // one iteration is a deci-second

    int count = 0;

    while (s_continue) {
        bslmt::ThreadUtil::microSleep(k_DECISECOND);
        ++count;

        ASSERTV(s_watchdogText, count < MAX);

        if (MAX == count && s_continue) {
            abort();
        }
    }

    return 0;
}

void orderingGuaranteeTest(const int numPushThread, const int numPopThread)
    // Exercise an 'OrderingObj' to verify, for the specified 'numPushThread'
    // and 'numPopThread', the set of elements enqueued by a particular thread
    // and dequeued by a particular thread, the order in which the elements of
    // this set are dequeued match the order these elements were enqueued.
{
    bslmt::ThreadUtil::Handle              watchdogHandle;
    bslmt::ThreadUtil::Handle              stateHandle;
    bsl::vector<bslmt::ThreadUtil::Handle> pushHandle(numPushThread);
    bsl::vector<bslmt::ThreadUtil::Handle> popHandle(numPopThread);
    bsl::vector<OrderingPopData>           orderingPopData(numPopThread);

    s_continue = 4;

    OrderingObj mX(32);  const OrderingObj& X = mX;

    setWatchdogText("ordering guarantee");
    bslmt::ThreadUtil::create(&watchdogHandle, watchdog, 0);

    bslmt::ThreadUtil::create(&stateHandle, orderingState, &mX);

    {
        for (int i = 0; i < numPushThread; ++i) {
            bslmt::ThreadUtil::create(&pushHandle[i], orderingPush, &mX);
        }
        for (int i = 0; i < numPopThread; ++i) {
            orderingPopData[i].d_obj_p = &mX;
            orderingPopData[i].d_isStrongTest = (   1 == numPushThread
                                                 && 1 == numPopThread);
            bslmt::ThreadUtil::create(&popHandle[i],
                                      orderingPop,
                                      &orderingPopData[i]);
        }
    }

    bslmt::ThreadUtil::microSleep(0, 1);
    s_continue = 3;

    setWatchdogText("ordering guarantee: join state");
    bslmt::ThreadUtil::join(stateHandle);

    mX.enablePopFront();
    mX.enablePushBack();

    bslmt::ThreadUtil::microSleep(0, 1);
    s_continue = 2;

    mX.disablePushBack();

    setWatchdogText("ordering guarantee: join push");
    for (int i = 0; i < numPushThread; ++i){
        bslmt::ThreadUtil::join(pushHandle[i]);
    }

    setWatchdogText("ordering guarantee: wait until empty");
    int rv = X.waitUntilEmpty();
    ASSERT(0 == rv);
    ASSERT(0 == X.numElements());

    s_continue = 1;

    mX.disablePopFront();

    setWatchdogText("ordering guarantee: join pop");
    for (int i = 0; i < numPopThread; ++i){
        bslmt::ThreadUtil::join(popHandle[i]);
    }

    s_continue = 0;

    setWatchdogText("ordering guarantee: join watchdog");
    bslmt::ThreadUtil::join(watchdogHandle);
}

const int k_PUSHPUSHWAIT_COUNT = 100000;

extern "C" void *pushPushWait(void *arg)
{
    Obj& mX = *static_cast<Obj *>(arg);

    Obj::value_type value = 0;

    for (int i = 0; i < k_PUSHPUSHWAIT_COUNT; ++i) {
        mX.pushBack(value);

        mX.pushBack(value);

        mX.waitUntilEmpty();

        ASSERT(0 == mX.numElements());
    }

    return 0;
}

extern "C" void *pushWaitDisable(void *arg)
{
    Obj& mX = *static_cast<Obj *>(arg);

    Obj::value_type value = 0;

    mX.pushBack(value);

    mX.waitUntilEmpty();

    mX.disablePopFront();

    return 0;
}

extern "C" void *popLoop(void *arg)
{
    Obj& mX = *static_cast<Obj *>(arg);
    int value;

    while (0 < s_continue) {
        s_barrier_p->wait();
        while (Obj::e_DISABLED != mX.popFront(&value));
        s_barrier_p->wait();
    }

    return 0;
}

class LongDestructor {
    int d_value;

  public:
    LongDestructor() : d_value(3) {}

    ~LongDestructor()
    {
        BSLS_ASSERT(3 == d_value);

        d_value = 1;

        bslmt::ThreadUtil::yield();
    }
};

struct Case13Data {
    bslmt::Barrier                      *d_barrier_p;
    bdlcc::BoundedQueue<LongDestructor> *d_queue_p;
    bsls::AtomicInt                     *d_count_p;
};

extern "C" void *case13_removeAll(void *arg)
{
    bdlcc::BoundedQueue<LongDestructor> *queue =
                                     static_cast<Case13Data *>(arg)->d_queue_p;

    bslmt::Barrier  *barrier = static_cast<Case13Data *>(arg)->d_barrier_p;
    bsls::AtomicInt *count   = static_cast<Case13Data *>(arg)->d_count_p;

    barrier->wait();

    while (*count > 0) {
        queue->removeAll();
        bslmt::ThreadUtil::microSleep(1, 0);
        (*count)--;
    }

    barrier->wait();

    return 0;
}

extern "C" void *case13_tryPushBack(void *arg)
{
    bdlcc::BoundedQueue<LongDestructor> *queue =
                                     static_cast<Case13Data *>(arg)->d_queue_p;

    bslmt::Barrier  *barrier = static_cast<Case13Data *>(arg)->d_barrier_p;
    bsls::AtomicInt *count   = static_cast<Case13Data *>(arg)->d_count_p;

    barrier->wait();

    LongDestructor record;

    while (*count > 0) {
        queue->tryPushBack(record);
    }

    barrier->wait();

    return 0;
}

extern "C" void *case13_popFront(void *arg)
{
    bdlcc::BoundedQueue<LongDestructor> *queue =
                                     static_cast<Case13Data *>(arg)->d_queue_p;

    while (true) {
        LongDestructor record;

        int rc = queue->popFront(&record);

        if (bdlcc::BoundedQueue<LongDestructor>::e_DISABLED == rc) {
            break;
        }

        bslmt::ThreadUtil::yield();
    }

    return 0;
}

// ============================================================================
//               GENERATOR FUNCTIONS 'gg' AND 'ggg' FOR TESTING
// ----------------------------------------------------------------------------
// The following functions interpret the given 'spec' in order from left to
// right to configure the object according to a custom language.
//
// LANGUAGE SPECIFICATION:
// -----------------------
//
// <SPEC> ::= <EMPTY>   | <LIST>
//
// <EMPTY> ::=
//
// <LIST> ::= <ITEM>    | <ITEM><LIST>
//
// <ITEM> ::= <ELEMENT> | '~'
//
// <ELEMENT> ::= 'a' | 'b' | 'c' | 'd' | 'e'

int getValue(int *value, char specChar, int verboseFlag);
    // Place into the specified 'value' the value corresponding to the
    // specified 'specChar' and display errors to 'cerr' if the specified
    // 'verboseFlag' is set.  Return 0 if operation successful, return non-zero
    // otherwise.

enum { e_SUCCESS_PUSHBACK = -1, e_SUCCESS_REMOVEALL = -2 };

int getValue(int *value, char specChar, int verboseFlag)
{
    static int values[] = { 0, 1, 2, 3, 4 };

    if ('~' == specChar) {
        return e_SUCCESS_REMOVEALL;                                   // RETURN
    }

    if ('a' <= specChar && 'e' >= specChar) {
        *value = values[specChar - 'a'];
        return e_SUCCESS_PUSHBACK;                                    // RETURN
    }

    if (verboseFlag) {
        cerr << "\t\tERROR!" << endl;
        cerr << specChar << " not recognized." << endl;
    }

    return 1;
}

template <class OBJ>
int ggg(OBJ *object, const char *spec, int verboseFlag = 1)
    // Configure the specified 'object' according to the specified 'spec',
    // using only the primary manipulator functions 'pushBack' and 'removeAll'.
    // Optionally specify a zero 'verboseFlag' to suppress 'spec' syntax error
    // messages.  Return the index of the first invalid character, and a
    // negative value otherwise.  Note that this function is used to implement
    // 'gg' as well as allow for verification of syntax error detection.
{
    enum { e_SUCCESS = -1 };

    typename OBJ::value_type v;

    for (int i = 0; spec[i]; ++i) {
        int rv = getValue(&v, spec[i], verboseFlag);
        if (rv > -1) {
            return i;                                                 // RETURN
        }
        else if (e_SUCCESS_PUSHBACK  == rv) {
            object->pushBack(v);
        }
        else if (e_SUCCESS_REMOVEALL == rv) {
            object->removeAll();
        }
    }

    return e_SUCCESS;
}

template <class OBJ>
OBJ& gg(OBJ *object, const char *spec)
    // Return, by reference, the specified 'object' with its value adjusted
    // according to the specified 'spec'.
{
    ASSERT(ggg(object, spec) < 0);
    return *object;
}

// ============================================================================
//                                USAGE EXAMPLE
// ----------------------------------------------------------------------------

///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: A Simple Thread Pool
///- - - - - - - - - - - - - - - -
// In the following example a 'bdlcc::BoundedQueue' is used to communicate
// between a single "producer" thread and multiple "consumer" threads.  The
// "producer" will push work requests onto the queue, and each "consumer" will
// iteratively take a work request from the queue and service the request.
// This example shows a partial, simplified implementation of the
// 'bdlmt::FixedThreadPool' class.  See component 'bdlmt_fixedthreadpool' for
// more information.
//
// First, we define a utility classes that handles a simple "work item":
//..
    struct my_WorkData {
        // Work data...
    };

    struct my_WorkRequest {
        enum RequestType {
            e_WORK = 1,
            e_STOP = 2
        };

        RequestType d_type;
        my_WorkData d_data;
        // Work data...
    };
//..
// Next, we provide a simple function to service an individual work item.  The
// details are unimportant for this example:
//..
    void myDoWork(const my_WorkData& data)
        // Do some work based upon the specified 'data'.
    {
        // do some stuff...
        (void)data;
    }
//..
// Then, we define a 'myConsumer' function that will pop elements off the queue
// and process them.  Note that the call to 'queue->popFront()' will block
// until there is an element available on the queue.  This function will be
// executed in multiple threads, so that each thread waits in
// 'queue->popFront()', and 'bdlcc::BoundedQueue' guarantees that each thread
// gets a unique element from the queue:
//..
    void myConsumer(bdlcc::BoundedQueue<my_WorkRequest> *queue)
        // Pop elements from the specified 'queue'.
    {
        while (1) {
            // 'popFront()' will wait for a 'my_WorkRequest' until available.

            my_WorkRequest item;
            item.d_type = my_WorkRequest::e_WORK;

            ASSERT(0 == queue->popFront(&item));

            if (item.d_type == my_WorkRequest::e_STOP) { break; }
            myDoWork(item.d_data);
        }
    }
//..
// Finally, we define a 'myProducer' function that serves multiple roles: it
// creates the 'bdlcc::BoundedQueue', starts the consumer threads, and then
// produces and enqueues work items.  When work requests are exhausted, this
// function enqueues one 'e_STOP' item for each consumer queue.  This 'e_STOP'
// item indicates to the consumer thread to terminate its thread-handling
// function.
//
// Note that, although the producer cannot control which thread 'pop's a
// particular work item, it can rely on the knowledge that each consumer thread
// will read a single 'e_STOP' item and then terminate.
//..
    void myProducer(int numThreads)
        // Create a queue, start the specified 'numThreads' consumer threads,
        // produce and enqueue work.
    {
        enum {
            k_MAX_QUEUE_LENGTH = 100,
            k_NUM_WORK_ITEMS   = 1000
        };

        bdlcc::BoundedQueue<my_WorkRequest> queue(k_MAX_QUEUE_LENGTH);

        bslmt::ThreadGroup consumerThreads;
        consumerThreads.addThreads(bdlf::BindUtil::bind(&myConsumer, &queue),
                                   numThreads);

        for (int i = 0; i < k_NUM_WORK_ITEMS; ++i) {
            my_WorkRequest item;
            item.d_type = my_WorkRequest::e_WORK;
            item.d_data = my_WorkData(); // some stuff to do
            queue.pushBack(item);
        }

        for (int i = 0; i < numThreads; ++i) {
            my_WorkRequest item;
            item.d_type = my_WorkRequest::e_STOP;
            queue.pushBack(item);
        }

        consumerThreads.joinAll();
    }
//..

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int                 test = argc > 1 ? atoi(argv[1]) : 0;
    bool             verbose = argc > 2;
    bool         veryVerbose = argc > 3;
    bool     veryVeryVerbose = argc > 4;
    bool veryVeryVeryVerbose = argc > 5;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    // CONCERN: In no case does memory come from the global allocator.

    bslma::TestAllocator globalAllocator("global", veryVeryVeryVerbose);
    bslma::Default::setGlobalAllocator(&globalAllocator);

    bslma::TestAllocator defaultAllocator("default", veryVeryVeryVerbose);
    ASSERT(0 == bslma::Default::setDefaultAllocator(&defaultAllocator));

    switch (test) { case 0:  // Zero is always the leading case.
      case 16: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //   Extracted from component header file.
        //
        // Concerns:
        //: 1 The usage example provided in the component header file compiles,
        //:   links, and runs as shown.
        //
        // Plan:
        //: 1 Incorporate usage example from header into test driver, remove
        //:   leading comment characters, and replace 'assert' with 'ASSERT'.
        //:   (C-1)
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "USAGE EXAMPLE" << endl
                          << "=============" << endl;

        bslmt::ThreadUtil::Handle watchdogHandle;

        s_continue = 1;

        bslmt::ThreadUtil::create(&watchdogHandle,
                                  watchdog,
                                  const_cast<char *>("usage example"));

        enum { k_NUM_THREADS = 4 };

        myProducer(k_NUM_THREADS);

        s_continue = 0;

        bslmt::ThreadUtil::join(watchdogHandle);
      } break;
      case 15: {
        // --------------------------------------------------------------------
        // DRQS 168011541: 'waitUntilEmpty' RACE WITH 'disablePopFront'
        //
        // Concerns:
        //: 1 Given other threads running 'popFront' in a loop, a thread
        //:   executing 'pushBack' operations, 'waitUntilEmpty', and
        //:   'disablePopFront' should always then observe the queue as empty.
        //
        // Plan:
        //: 1 Create a number of threads that execute 'popFront' and
        //:   synchronize with the main thread using a barrier.  The main
        //:   thread executes a 'pushBack' per background thread,
        //:   'waitUntilEmpty', 'disablePopFront', and then verifies the queue
        //:   is empty.
        //
        // Testing:
        //   DRQS 168011541: 'waitUntilEmpty' RACE WITH 'disablePopFront'
        // --------------------------------------------------------------------

        if (verbose) {
            cout
              << "DRQS 168011541: 'waitUntilEmpty' RACE WITH 'disablePopFront'"
              << endl
              << "============================================================"
              << endl;
        }

        Obj mX(32);

        const int numThreads = 4;

        bslmt::Barrier barrier(numThreads + 1);

        s_continue  = 1;
        s_barrier_p = &barrier;

        bsl::vector<bslmt::ThreadUtil::Handle> handles(numThreads);

        for (int i = 0; i < numThreads; ++i) {
            bslmt::ThreadUtil::create(&handles[i], popLoop, &mX);
        }

        bslmt::ThreadUtil::Handle watchdogHandle;

        bslmt::ThreadUtil::create(&watchdogHandle,
                                  watchdog,
                                  const_cast<char *>(
                              "'waitUntilEmpty' race with 'disablePopFront'"));

        // Run for 10000 iterations or 3 seconds.

        int                completed = 0;
        bsls::TimeInterval start     = bsls::SystemTime::nowMonotonicClock();
        bsls::TimeInterval now       = start;

        while (completed < 10000 && now - start < bsls::TimeInterval(3, 0)) {
            mX.enablePopFront();
            barrier.wait();

            for (int j = 0; j < numThreads; ++j) {
                mX.pushBack(j);
            }

            ASSERT(Obj::e_SUCCESS == mX.waitUntilEmpty());
            mX.disablePopFront();

            barrier.wait();

            ASSERT(mX.isEmpty());

            ++completed;

            now = bsls::SystemTime::nowMonotonicClock();
        }

        {
            mX.enablePopFront();
            barrier.wait();
            s_continue = 0;
            mX.disablePopFront();
            barrier.wait();
        }

        for (int i = 0; i < numThreads; ++i){
            bslmt::ThreadUtil::join(handles[i]);
        }

        bslmt::ThreadUtil::join(watchdogHandle);
      } break;
      case 14: {
        // --------------------------------------------------------------------
        // DRQS 153332608: 'pushBack', 'pushBack', 'waitUntilEmpty'
        //
        // Concerns:
        //: 1 If one thread executes 'pushBack', 'pushBack', and
        //:   'waitUntilEmpty' on an empty queue, another thread executing two
        //:   'popFront' should never cause 'waitUntilEmpty' to return after
        //:   the first 'popFront' (detectible with '0 != numElements()').
        //
        // Plan:
        //: 1 Create a thread that executes a known number of 'pushBack',
        //:   'pushBack', and 'waitUntilEmpty' sequences.  Concurrently,
        //:   another thread executes the same number of 'popFront' and
        //:   'popFront' sequences.  After every 'waitUntilEmpty', verify
        //:   '0 == numElements()'.
        //
        // Testing:
        //   DRQS 153332608: 'pushBack', 'pushBack', 'waitUntilEmpty'
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "DRQS 153332608: 'pushBack', 'pushBack', 'waitUntilEmpty'"
                 << endl
                 << "========================================================"
                 << endl;
        }

        bslmt::ThreadUtil::Handle              handle;

        Obj mX(128);

        bslmt::ThreadUtil::create(&handle, pushPushWait, &mX);

        Obj::value_type value;

        for (int i = 0; i < k_PUSHPUSHWAIT_COUNT; ++i) {
            ASSERT(0 == mX.popFront(&value));
            ASSERT(0 == mX.popFront(&value));
        }

        bslmt::ThreadUtil::join(handle);
      } break;
      case 13: {
        // --------------------------------------------------------------------
        // DRQS 164984269: 'removeAll' STARTED/FINISHED ISSUE
        //
        // Concerns:
        //: 1 The method 'removeAll' correctly makes entries available for
        //:   reuse.
        //
        // Plan:
        //: 1 Create threads that will execute 'removeAll', 'popFront', and
        //:   'pushBack' concurrently on a queue with an element type,
        //:   'LongDestructor', that verifies the destructor is not called
        //:   twice upon the same element and causes a 'yield' to other
        //:   threads (to frequently recreate the issue seen in the DRQS).
        //:   Note that the issue is exposed after 'pushBack' gains access to
        //:   a not-yet-destructed element, the element gets destructed after
        //:   'pushBack' completes, and the element is removed (destructed a
        //:   second time).
        //
        // Testing:
        //   DRQS 164984269: 'removeAll' STARTED/FINISHED ISSUE
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "DRQS 164984269: 'removeAll' STARTED/FINISHED ISSUE\n"
                 << "==================================================\n";
        }

        bdlcc::BoundedQueue<LongDestructor> queue(8192);

        bslmt::Barrier  barrier(3);
        bsls::AtomicInt count(5000);

        Case13Data data;

        data.d_barrier_p = &barrier;
        data.d_queue_p   = &queue;
        data.d_count_p   = &count;

        bslmt::ThreadUtil::Handle removeThread, pushThread, popThread;

        bslmt::ThreadUtil::create(&removeThread, case13_removeAll, &data);
        bslmt::ThreadUtil::create(&pushThread, case13_tryPushBack, &data);
        bslmt::ThreadUtil::create(&popThread, case13_popFront, &data);

        barrier.wait();  // Start of the test.
        barrier.wait();  // end of the test

        bslmt::ThreadUtil::join(removeThread);
        bslmt::ThreadUtil::join(pushThread);

        queue.disablePopFront();  // exit the popFront loops
        bslmt::ThreadUtil::join(popThread);
      } break;
      case 12: {
        // --------------------------------------------------------------------
        // DRQS 153332608: 'waitUntilEmpty' RACE WITH 'popFront'
        //
        // Concerns:
        //: 1 The method 'waitUntilEmpty' can not return before the 'popFront'
        //:   which makes the queue empty is guaranteed to complete the pop.
        //
        // Plan:
        //: 1 Create a thread that will push an element onto the queue,
        //:   'waitUntilEmpty', and then 'disablePopFront' while the main
        //:   thread executes 'popFront'.  Verify the pop was successful
        //:   (i.e., did not return due to disablement).  (C-1)
        //
        // Testing:
        //   DRQS 153332608: 'waitUntilEmpty' RACE WITH 'popFront'
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "DRQS 153332608: 'waitUntilEmpty' RACE WITH 'popFront'\n"
                 << "=====================================================\n";
        }

        bslmt::ThreadUtil::Handle              handle;

        Obj mX(128);

        bslmt::ThreadUtil::create(&handle, pushWaitDisable, &mX);

        Obj::value_type value;

        ASSERT(0 == mX.popFront(&value));

        bslmt::ThreadUtil::join(handle);
      } break;
      case 11: {
        // ---------------------------------------------------------
        // ORDERING GUARANTEE TEST
        //   The queue should provide the minimal ordering guarantee for
        //   concurrent queues.  Specifically, for the set of elements enqueued
        //   by a particular thread and dequeued by a particular thread, the
        //   order in which the elements of this set are dequeued must match
        //   the order these elements were enqueued.
        //
        // Concerns:
        //: 1 If the queue has a single consumer and a single consumer, the
        //:   strong form of the ordering guarantee is provided.
        //:
        //: 2 Under normal operation, the weak form of the ordering guarantee
        //:   is provided.
        //
        // Plan:
        //: 1 Using elements that store a sequence number, ensure the dequeued
        //:   elements' sequence number increases by one.  (C-1)
        //:
        //: 2 Using elements that store a sequence number, ensure the dequeued
        //:   elements' sequence number for the source thread is increasing.
        //:   (C-2)
        //
        // Testing:
        //   CONCERN: ordering guarantee
        // ---------------------------------------------------------

        if (verbose) cout << endl
                          << "ORDERING GUARANTEE TEST" << endl
                          << "=======================" << endl;

        orderingGuaranteeTest(1, 1);  // single producer, single consumer
        orderingGuaranteeTest(4, 4);  // multiple producers, multiple consumers
      } break;
      case 10: {
        // ---------------------------------------------------------
        // TEMPLATE REQUIREMENTS TEST
        //   The queue should work for types that have no default
        //   constructor and a 1-arg copy constructor.
        //
        // Concerns:
        //: 1 The queue should work for types that have no default
        //:   constructor and a 1-arg copy constructor.
        //
        // Plan:
        //: 1 Create and exercise a queue using 'TemplateRequirementTestType'
        //:   (C-1)
        //
        // Testing:
        //   CONCERN: template requirements
        // ---------------------------------------------------------

        if (verbose) cout << endl
                          << "TEMPLATE REQUIREMENTS TEST" << endl
                          << "==========================" << endl;

        bdlcc::BoundedQueue<TemplateRequirementTestType> queue(8);

        int count = 0;
        {
            TemplateRequirementTestType t(&count);
            queue.pushBack(t);

            TemplateRequirementTestType t2(&count);
            queue.popFront(&t2);
        }
        ASSERT(0 == count);
      } break;
      case 9: {
        // --------------------------------------------------------------------
        // MOVING TESTS
        //   Ensure move-semantics are honored.  Note that the testing types
        //   are not universally supported.  Move-only type is supported in
        //   C++11 mode.  Allocating move-only type is supported in C++11 mode.
        //   'IncorrectlyMatchingMoveConstructorTestType' works in C++03 mode.
        //
        // Concerns:
        //: 1 The manipulators 'pushBack', 'tryPushBack' 'popFront', and
        //:   'tryPopFront' honor move-semantics.
        //
        // Plan:
        //: 1 After pushing back a value, verify the value is in moved-from
        //:   state.  After popping a value, verify the global moved-from
        //:   counter has increased.  (C-1)
        //
        // Testing:
        //   int pushBack(bslmf::MovableRef<TYPE> value);
        //   int tryPushBack(bslmf::MovableRef<TYPE> value);
        //   CONCERN: 'popFront' and 'tryPopFront' honor move-semantics
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "MOVING TESTS" << endl
                          << "============" << endl;

        typedef bslmf::MovableRefUtil MoveUtil;

        if (veryVerbose) cout << "Move-only type" << endl;
#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
        // Move-only types are not supported in C++03 mode, only C++11 and
        // higher.  See internal bug report 99039150.
        {
            bslma::TestAllocator ta(veryVeryVerbose);

            bdlcc::BoundedQueue<MoveTester> queue(42, &ta);

            int moveCtr = 0;

            queue.pushBack(MoveTester(1, &moveCtr));

            ASSERT(1 == moveCtr);

            moveCtr = 1;

            MoveTester moveTester2(2, &moveCtr);
            queue.pushBack(MoveUtil::move(moveTester2));

            ASSERT(moveTester2.isMoved());

            ASSERT(queue.tryPushBack(MoveTester(3, &moveCtr)) == 0);

            ASSERT(3 == moveCtr);
            moveCtr = 3;

            MoveTester moveTester4(4, &moveCtr);
            ASSERT(queue.tryPushBack(MoveUtil::move(moveTester4)) == 0);

            ASSERT(moveTester4.isMoved());

            ASSERT(4 == moveCtr);

            // Test 'popFront', 'tryPopFront'

            MoveTester popped(42);
            queue.popFront(&popped);

            ASSERT(5 == moveCtr);

            ASSERT(queue.tryPopFront(&popped) == 0);
            ASSERT(6 == moveCtr);
        }

        if (veryVerbose) cout << "Move-only allocating type" << endl;
        {
            typedef bsltf::MoveOnlyAllocTestType ValueType;

            bslma::TestAllocator ta(veryVeryVerbose);

            bdlcc::BoundedQueue<ValueType> queue(42, &ta);

            bslma::TestAllocatorMonitor tam(&ta);

            queue.pushBack(ValueType(1, &ta));

            ASSERT(tam.isInUseUp());
            tam.reset();

            ValueType popped1(&ta);
            queue.popFront(&popped1);
            ASSERT(tam.isInUseSame());
            ASSERT(popped1.data() == 1);
            ASSERT(popped1.allocator() == &ta);
            tam.reset();

            ValueType value2(2, &ta);
            tam.reset();
            queue.pushBack(MoveUtil::move(value2));
            ASSERT(tam.isInUseSame());

            ValueType popped2(&ta);
            tam.reset();
            queue.popFront(&popped2);
            ASSERT(tam.isInUseDown());
            ASSERT(popped2.data() == 2);
            ASSERT(popped2.allocator() == &ta);

            ASSERT(queue.tryPushBack(ValueType(3, &ta)) == 0);

            ASSERT(tam.isInUseSame());
            tam.reset();

            ValueType popped3(&ta);
            ASSERT(queue.tryPopFront(&popped3) == 0);
            ASSERT(tam.isInUseSame());
            ASSERT(popped3.data() == 3);
            ASSERT(popped3.allocator() == &ta);
            tam.reset();

            ValueType value4(4, &ta);
            tam.reset();
            ASSERT(queue.tryPushBack(MoveUtil::move(value4)) == 0);
            ASSERT(tam.isInUseSame());

            ValueType popped4(&ta);
            queue.popFront(&popped4);
            ASSERT(tam.isInUseSame());
            ASSERT(popped4.data() == 4);
            ASSERT(popped4.allocator() == &ta);

            // Test with different allocator for the value

            bslma::TestAllocator        tb(veryVeryVerbose);
            bslma::TestAllocatorMonitor tbm(&tb);

            ASSERT(queue.tryPushBack(ValueType(5, &tb)) == 0);

            ASSERT(tbm.isInUseSame());
            tam.reset();
            tbm.reset();

            ValueType popped5(&tb);
            ASSERT(queue.tryPopFront(&popped5) == 0);
            ASSERT(tam.isInUseDown()); // the element in the queue is deleted
            ASSERT(popped5.data() == 5);
            ASSERT(popped5.allocator() == &tb);
            tam.reset();

            ValueType value6(6, &ta);
            tam.reset();
            ASSERT(queue.tryPushBack(MoveUtil::move(value6)) == 0);
            ASSERT(tam.isInUseSame());

            ValueType popped6(&ta);
            queue.popFront(&popped6);
            ASSERT(tam.isInUseSame());
            ASSERT(popped6.data() == 6);
            ASSERT(popped6.allocator() == &ta);
        }
#endif

        if (veryVerbose) cout << "Movable allocating type" << endl;
        {
            typedef bsltf::MovableAllocTestType ValueType;

            bslma::TestAllocator ta(veryVeryVerbose);

            bdlcc::BoundedQueue<ValueType> queue(42, &ta);

            bslma::TestAllocatorMonitor tam(&ta);

#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
            queue.pushBack(ValueType(1, &ta));

            ASSERT(tam.isInUseUp());
            tam.reset();

            ValueType popped1(&ta);
            queue.popFront(&popped1);
            ASSERT(tam.isInUseSame());
            ASSERT(popped1.data() == 1);
            ASSERT(popped1.allocator() == &ta);
            tam.reset();
#endif
            ValueType value2(2, &ta);
            tam.reset();
            queue.pushBack(MoveUtil::move(value2));
            ASSERT(tam.isInUseSame());

            ValueType popped2(&ta);
            tam.reset();
            queue.popFront(&popped2);
            ASSERT(tam.isInUseDown());
            ASSERT(popped2.data() == 2);
            ASSERT(popped2.allocator() == &ta);

#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
            ASSERT(queue.tryPushBack(ValueType(3, &ta)) == 0);

            ASSERT(tam.isInUseSame());
            tam.reset();

            ValueType popped3(&ta);
            ASSERT(queue.tryPopFront(&popped3) == 0);
            ASSERT(tam.isInUseSame());
            ASSERT(popped3.data() == 3);
            ASSERT(popped3.allocator() == &ta);
            tam.reset();
#endif
            ValueType value4(4, &ta);
            tam.reset();
            ASSERT(queue.tryPushBack(MoveUtil::move(value4)) == 0);
            ASSERT(tam.isInUseSame());

            ValueType popped4(&ta);
            queue.popFront(&popped4);
            ASSERT(tam.isInUseSame());
            ASSERT(popped4.data() == 4);
            ASSERT(popped4.allocator() == &ta);

            // Test with different allocator for the value

            bslma::TestAllocator        tb(veryVeryVerbose);
            bslma::TestAllocatorMonitor tbm(&tb);

            ASSERT(queue.tryPushBack(ValueType(5, &tb)) == 0);

            ASSERT(tbm.isInUseSame());
            tam.reset();
            tbm.reset();

            ValueType popped5(&tb);
            ASSERT(queue.tryPopFront(&popped5) == 0);
            ASSERT(tam.isInUseDown()); // the element in the queue is deleted
            ASSERT(popped5.data() == 5);
            ASSERT(popped5.allocator() == &tb);
            tam.reset();

            ValueType value6(6, &ta);
            tam.reset();
            ASSERT(queue.tryPushBack(MoveUtil::move(value6)) == 0);
            ASSERT(tam.isInUseSame());

            ValueType popped6(&ta);
            queue.popFront(&popped6);
            ASSERT(tam.isInUseSame());
            ASSERT(popped6.data() == 6);
            ASSERT(popped6.allocator() == &ta);
        }

        if (veryVerbose) cout << "IncorrectlyMatchingMoveConstructorTestType"
                              << endl;
        {
            typedef IncorrectlyMatchingMoveConstructorTestType ValueType;

            bslma::TestAllocator ta(veryVeryVerbose);

            bdlcc::BoundedQueue<ValueType> queue(42, &ta);

            bslma::TestAllocatorMonitor tam(&ta);

#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
            queue.pushBack(ValueType(1));

            ASSERT(tam.isInUseSame());
            tam.reset();

            ValueType popped1;
            queue.popFront(&popped1);
            ASSERT(tam.isInUseSame());
            ASSERT(popped1.data() == 1);
            tam.reset();
#endif
            ValueType value2(2);
            tam.reset();
            queue.pushBack(value2);
            ASSERT(tam.isInUseSame());

            ValueType popped2;
            queue.popFront(&popped2);
            ASSERT(tam.isInUseSame());
            ASSERT(popped2.data() == 2);

#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
            ASSERT(queue.tryPushBack(ValueType(3)) == 0);

            ASSERT(tam.isInUseSame());
            tam.reset();

            ValueType popped3;
            ASSERT(queue.tryPopFront(&popped3) == 0);
            ASSERT(tam.isInUseSame());
            ASSERT(popped3.data() == 3);
            tam.reset();
#endif
            ValueType value4(4);
            tam.reset();
            ASSERT(queue.tryPushBack(value4) == 0);
            ASSERT(tam.isInUseSame());

            ValueType popped4;
            queue.popFront(&popped4);
            ASSERT(tam.isInUseSame());
            ASSERT(popped4.data() == 4);
        }
      } break;
      case 8: {
        // --------------------------------------------------------------------
        // TESTING 'waitUntilEmpty'
        //   Ensure the accessor functions as expected.
        //
        // Concerns:
        //: 1 The method 'waitUntilEmpty' returns at the appropriate time with
        //:   the expected value.
        //:
        //: 2 The accessor method is declared 'const'.
        //:
        //: 3 The accessor does not allocate any memory.
        //
        // Plan:
        //: 1 Create objects and set their state using 'pushBack'.  Schedule a
        //:   deferred 'popFront' or 'disablePopFront' on a created thread.
        //:   Invoke 'waitUntilEmpty', verify when the accessor returns and
        //:   the returned value.  (C-1)
        //:
        //: 2 The accessor will only be accessed from a 'const' reference to
        //:   the created object.  (C-2)
        //:
        //: 3 The default allocator will be used for all created objects
        //:   (excluding those used to test 'allocator') and the number of
        //:   allocation will be verified to ensure that no memory was
        //:   allocated during use of the accessor.  (C-3)
        //
        // Testing:
        //   int waitUntilEmpty() const;
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'waitUntilEmpty'" << endl
                          << "========================" << endl;

        if (verbose) cout << "\nTesting 'waitUntilEmpty'." << endl;
        {
            bslmt::ThreadUtil::Handle watchdogHandle;

            s_continue = 1;

            bslmt::ThreadUtil::create(&watchdogHandle,
                                      watchdog,
                                      const_cast<char *>("waitUntilEmpty 1"));

            Obj mX(8);  const Obj& X = mX;

            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            int rv = X.waitUntilEmpty();

            ASSERT(e_SUCCESS   == rv);
            ASSERT(allocations == defaultAllocator.numAllocations());

            s_continue = 0;

            bslmt::ThreadUtil::join(watchdogHandle);
        }
        {
            bslmt::ThreadUtil::Handle watchdogHandle;

            s_continue = 1;

            bslmt::ThreadUtil::create(&watchdogHandle,
                                      watchdog,
                                      const_cast<char *>("waitUntilEmpty 2"));

            Obj mX(8);  const Obj& X = mX;

            bslmt::ThreadUtil::Handle handle;

            bslmt::ThreadUtil::create(&handle, deferredPopFront, &mX);

            mX.pushBack(0);

            bsls::TimeInterval interval =
                      bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME);

            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            int rv = X.waitUntilEmpty();

            ASSERT(e_SUCCESS   == rv);
            ASSERT(allocations == defaultAllocator.numAllocations());

            interval = bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME)
                     - interval;

            bslmt::ThreadUtil::join(handle);

            ASSERT(   s_deferredPopFrontInterval.totalSecondsAsDouble() * 0.8
                                           <= interval.totalSecondsAsDouble()
                   && s_deferredPopFrontInterval.totalSecondsAsDouble() * 1.5
                                           >= interval.totalSecondsAsDouble());

            s_continue = 0;

            bslmt::ThreadUtil::join(watchdogHandle);
        }
        {
            bslmt::ThreadUtil::Handle watchdogHandle;

            s_continue = 1;

            bslmt::ThreadUtil::create(&watchdogHandle,
                                      watchdog,
                                      const_cast<char *>("waitUntilEmpty 3"));

            Obj mX(8);  const Obj& X = mX;

            bslmt::ThreadUtil::Handle handle;

            bslmt::ThreadUtil::create(&handle, deferredDisablePopFront, &mX);

            mX.pushBack(0);

            bsls::TimeInterval interval =
                      bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME);

            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            int rv = X.waitUntilEmpty();

            ASSERT(e_DISABLED  == rv);
            ASSERT(allocations == defaultAllocator.numAllocations());

            interval = bsls::SystemTime::now(bsls::SystemClockType::e_REALTIME)
                     - interval;

            ASSERT(   bsls::TimeInterval(0.8) <= interval
                   && bsls::TimeInterval(1.5) >= interval);

            bslmt::ThreadUtil::join(handle);

            s_continue = 0;

            bslmt::ThreadUtil::join(watchdogHandle);
        }
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // TESTING 'tryPopFront'
        //   Ensure the manipulator functions as expected.
        //
        // Concerns:
        //: 1 The method 'tryPopFront' produces the expected value, does not
        //:   affect allocated memory, and is exception neutral with respect to
        //:   memory allocation.
        //:
        //: 2 When the queue is empty, the method 'tryPopFront' does not block,
        //:   the argument is not modified, and the return value is 'e_EMPTY'.
        //:
        //: 3 Memory is not leaked by any method and the destructor properly
        //:   deallocates the residual allocated memory.
        //
        // Plan:
        //: 1 Create objects using the 'bslma::TestAllocator', use 'pushBack'
        //:   to obtain various states, use 'tryPopFront', verify the returned
        //:   value, and that objects have a reduced length but unchanged
        //:   capacity.  Also vary the test allocator's allocation limit to
        //:   verify behavior in the presence of exceptions.  (C-1)
        //:
        //: 2 Use 'tryPopFront' on empty objects, verify the argument is
        //:   unmodified and the return value indicates failure.  (C-2)
        //:
        //: 3 Use a supplied 'bslma::TestAllocator' that goes out-of-scope
        //:   at the conclusion of each test to ensure all memory is returned
        //:   to the allocator.  (C-3)
        //
        // Testing:
        //   int tryPopFront(TYPE *value);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'tryPopFront'" << endl
                          << "=====================" << endl;

        if (verbose) cout << "\nTesting 'tryPopFront'." << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;

            int value;
            int rv;

            value = 7;
            rv = mX.tryPopFront(&value);
            ASSERT(e_EMPTY == rv);
            ASSERT(      0 == X.numElements());
            ASSERT(      7 == value);

            value = 3;
            rv = mX.tryPopFront(&value);
            ASSERT(e_EMPTY == rv);
            ASSERT(      0 == X.numElements());
            ASSERT(      3 == value);

            mX.pushBack(0);
            mX.pushBack(1);
            mX.pushBack(2);
            ASSERT(3 == X.numElements());

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            rv = mX.tryPopFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        2 == X.numElements());
            ASSERT(        0 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            rv = mX.tryPopFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());
            ASSERT(        1 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            rv = mX.tryPopFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        0 == X.numElements());
            ASSERT(        2 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            rv = mX.tryPopFront(&value);
            ASSERT(e_EMPTY == rv);
            ASSERT(      0 == X.numElements());
            ASSERT(      2 == value);
            ASSERT(     na == sa.numAllocations());
            ASSERT(     nd == sa.numDeallocations());

            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            ASSERT( 3 == X.numElements());
            ASSERT(na == sa.numAllocations());
            ASSERT(nd == sa.numDeallocations());

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#ifdef BDE_BUILD_TARGET_EXC
        {
            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<AllocExceptionHelper>        mX(8, &sa);
            const bdlcc::BoundedQueue<AllocExceptionHelper>& X = mX;

            AllocExceptionHelper value(&sa);

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.pushBack(value);
            mX.pushBack(value);

            ASSERT(     2 == X.numElements());
            ASSERT(na + 2 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            int numException = 0;

            sa.setAllocationLimit(0);
            try {
                mX.tryPopFront(&value);
            } catch (BloombergLP::bslma::TestAllocatorException& e) {
                ++numException;
            }
            sa.setAllocationLimit(-1);

            ASSERT(     1 == numException);
            ASSERT(     1 == X.numElements());

            // The test allocator increments the number of allocations and then
            // throws the exception.

            ASSERT(na + 3 == sa.numAllocations());
            ASSERT(nd + 1 == sa.numDeallocations());

            mX.tryPopFront(&value);

            ASSERT(na + 4 == sa.numAllocations());
            ASSERT(nd + 3 == sa.numDeallocations());

            ASSERT(     0 == X.numElements());

            mX.pushBack(value);
            ASSERT(     1 == X.numElements());

            // Since the queue is empty, only the element should have
            // allocated.

            ASSERT(na + 5 == sa.numAllocations());
            ASSERT(nd + 3 == sa.numDeallocations());
        }
#endif
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING 'tryPushBack'
        //   Ensure the manipulators function as expected.
        //
        // Concerns:
        //: 1 The methods 'tryPushBack' produce the expected value, do not
        //:   affect allocated memory, and are exception neutral with respect
        //:   to memory allocation.
        //:
        //: 2 When the queue is full, the method 'tryPushBack' does not block
        //:   and the return value is 'e_FULL'.
        //:
        //: 3 The methods return the correct value when the queue is push
        //:   disabled.
        //:
        //: 4 Memory is not leaked by any method and the destructor properly
        //:   deallocates the residual allocated memory.
        //
        // Plan:
        //: 1 Create an object and use 'tryPushBack' to modify the object's
        //:   value.  Verify the state of the object using the basic accessors
        //:   and 'popFront'.  Also vary the test allocator's allocation limit
        //:   to verify behavior in the presence of exceptions.  (C-1)
        //:
        //: 2 Use 'tryPushBack' on full objects, verify the return value
        //:   indicates failure.  (C-2)
        //:
        //: 3 Verify the return value of 'tryPushBack' before and after use of
        //:   'disablePushBack'.  (C-3)
        //:
        //: 4 Use a supplied 'bslma::TestAllocator' that goes out-of-scope
        //:   at the conclusion of each test to ensure all memory is returned
        //:   to the allocator.  (C-4)
        //
        // Testing:
        //   int tryPushBack(const TYPE& value);
        //   CONCERN: 'tryPushBack(MovableRef<TYPE>)' basic functionality
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'tryPushBack'" << endl
                          << "=====================" << endl;

        if (verbose) cout << "\nTesting 'tryPushBack(const TYPE&)'." << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;

            ASSERT(0 == X.numElements());

            int rv;

            rv = mX.tryPushBack(0);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());

            rv = mX.tryPushBack(1);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        2 == X.numElements());

            int value;

            mX.popFront(&value);
            ASSERT(0 == value);
            ASSERT(1 == X.numElements());

            mX.popFront(&value);
            ASSERT(1 == value);
            ASSERT(0 == X.numElements());

            mX.disablePushBack();

            rv = mX.tryPushBack(0);
            ASSERT(e_DISABLED == rv);
            ASSERT(         0 == X.numElements());

            mX.enablePushBack();

            rv = mX.tryPushBack(0);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());

            mX.tryPushBack(0);
            mX.tryPushBack(0);
            mX.tryPushBack(0);
            mX.tryPushBack(0);
            mX.tryPushBack(0);
            mX.tryPushBack(0);
            mX.tryPushBack(0);

            ASSERT(8 == X.numElements());

            rv = mX.tryPushBack(1);
            ASSERT(e_FULL == rv);
            ASSERT(     8 == X.numElements());

            mX.disablePushBack();

            rv = mX.tryPushBack(0);
            ASSERT(e_DISABLED == rv);
            ASSERT(         8 == X.numElements());

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#ifdef BDE_BUILD_TARGET_EXC
        {
            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<AllocExceptionHelper>        mX(8, &sa);
            const bdlcc::BoundedQueue<AllocExceptionHelper>& X = mX;

            AllocExceptionHelper value(&sa);

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.pushBack(value);
            mX.pushBack(value);

            ASSERT(     2 == X.numElements());
            ASSERT(na + 2 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            int numException = 0;

            sa.setAllocationLimit(0);
            try {
                mX.tryPushBack(value);
            } catch (BloombergLP::bslma::TestAllocatorException& e) {
                ++numException;
            }
            sa.setAllocationLimit(-1);

            ASSERT(     1 == numException);
            ASSERT(     2 == X.numElements());

            // The test allocator increments the number of allocations and then
            // throws the exception.

            ASSERT(na + 3 == sa.numAllocations());
            ASSERT(nd + 0 == sa.numDeallocations());

            mX.tryPushBack(value);

            ASSERT(     3 == X.numElements());
            ASSERT(na + 4 == sa.numAllocations());
            ASSERT(nd + 0 == sa.numDeallocations());

            mX.pushBack(value);
            ASSERT(     4 == X.numElements());
            ASSERT(na + 5 == sa.numAllocations());
            ASSERT(nd + 0 == sa.numDeallocations());
        }
#endif

        if (verbose) {
            cout << "\nTesting 'tryPushBack(bslmf::MovableRef<TYPE>)'."
                 << endl;
        }
#ifdef BSLMF_MOVABLEREF_USES_RVALUE_REFERENCES
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<MoveTester>        mX(8, &sa);
            const bdlcc::BoundedQueue<MoveTester>& X = mX;

            ASSERT(0 == X.numElements());

            int rv;
            int moveCtr = 0;

            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());
            ASSERT(        1 == moveCtr);

            rv = mX.tryPushBack(MoveTester(1, &moveCtr));
            ASSERT(e_SUCCESS == rv);
            ASSERT(        2 == X.numElements());
            ASSERT(        2 == moveCtr);

            MoveTester value(-1);

            mX.popFront(&value);
            ASSERT(0 == value.value());
            ASSERT(3 == moveCtr);

            mX.popFront(&value);
            ASSERT(1 == value.value());
            ASSERT(4 == moveCtr);

            mX.disablePushBack();

            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            ASSERT(e_DISABLED == rv);
            ASSERT(         0 == X.numElements());
            ASSERT(         4 == moveCtr);

            mX.enablePushBack();

            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());
            ASSERT(        5 == moveCtr);

            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            rv = mX.tryPushBack(MoveTester(0, &moveCtr));

            ASSERT(        8 == X.numElements());
            ASSERT(       12 == moveCtr);

            rv = mX.tryPushBack(MoveTester(1, &moveCtr));
            ASSERT(e_FULL == rv);
            ASSERT(     8 == X.numElements());

            mX.disablePushBack();

            rv = mX.tryPushBack(MoveTester(0, &moveCtr));
            ASSERT(e_DISABLED == rv);
            ASSERT(         8 == X.numElements());

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#endif
      } break;
      case 5: {
        // --------------------------------------------------------------------
        // ENABLE/DISABLE ENQUEUE/DEQUEUE
        //   Ensure the manipulators and associated accessors for enabling and
        //   disabling enqueueing and dequeueing work as expected.
        //
        // Concerns:
        //: 1 Each manipulator appropriated changes the state of the object.
        //:
        //: 2 Each accessor returns the value of the corresponding attribute of
        //:   the object.
        //:
        //: 3 Each accessor method is declared 'const'.
        //:
        //: 4 No accessor allocates any memory.
        //:
        //: 5 The behavior of the associated enqueueing and dequeueing methods
        //:   is correctly modified.
        //:
        //: 6 Blocked threads are released as appropriate.
        //
        // Plan:
        //: 1 Create an object and directly modify the state.  Verify the state
        //:   changes through use of the accessors.  (C-1,2)
        //:
        //: 2 The accessors will only be accessed from a 'const' reference to
        //:   the created object.  (C-3)
        //:
        //: 3 The default allocator will be used for all created objects
        //:   (excluding those used to test 'allocator') and the number of
        //:   allocation will be verified to ensure that no memory was
        //:   allocated during use of the accessors.  (C-4)
        //:
        //: 4 Create an object and directly verify the effect on the methods
        //:   'waitUntilEmpty', 'popFront', and 'pushBack', as well as their
        //:   return values.  (C-5)
        //:
        //: 5 Create objects and cause threads to block on the 'waitUntilEmpty'
        //:   and 'popFront' methods.  Verify the threads are released, with
        //:   appropriate return value, upon disabling the relevant
        //:   functionality.  (C-6)
        //
        // Testing:
        //   void disablePopFront();
        //   void disablePushBack();
        //   void enablePopFront();
        //   void enablePushBack();
        //   bool isPopFrontDisabled() const;
        //   bool isPushBackDisabled() const;
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "ENABLE/DISABLE ENQUEUE/DEQUEUE" << endl
                          << "==============================" << endl;

        if (verbose) cout << "\nTesting basic functionality." << endl;
        {
            Obj mX(8);  const Obj& X = mX;
            ASSERT(&defaultAllocator == X.allocator());

            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            ASSERT(!X.isPopFrontDisabled());
            ASSERT(!X.isPushBackDisabled());

            mX.disablePopFront();

            ASSERT( X.isPopFrontDisabled());
            ASSERT(!X.isPushBackDisabled());

            mX.disablePushBack();

            ASSERT( X.isPopFrontDisabled());
            ASSERT( X.isPushBackDisabled());

            mX.enablePopFront();

            ASSERT(!X.isPopFrontDisabled());
            ASSERT( X.isPushBackDisabled());

            mX.enablePushBack();

            ASSERT(!X.isPopFrontDisabled());
            ASSERT(!X.isPushBackDisabled());

            ASSERT(defaultAllocator.numAllocations() == allocations);
        }

        if (verbose) cout << "\nTesting effect on other methods." << endl;
        {
            Obj mX(8);  const Obj& X = mX;

            int rv;

            rv = mX.pushBack(0);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        1 == X.numElements());

            mX.disablePopFront();
            mX.disablePushBack();

            rv = X.waitUntilEmpty();
            ASSERT(e_DISABLED == rv);
            ASSERT(         1 == X.numElements());

            int value = 7;
            rv = mX.popFront(&value);
            ASSERT(e_DISABLED == rv);
            ASSERT(         7 == value);
            ASSERT(         1 == X.numElements());

            rv = mX.pushBack(0);
            ASSERT(e_DISABLED == rv);
            ASSERT(         1 == X.numElements());
        }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // BASIC ACCESSORS
        //   Ensure each basic accessor properly interprets object state.
        //
        // Concerns:
        //: 1 Each accessor returns the value of the corresponding attribute of
        //:   the object.
        //:
        //: 2 Each accessor method is declared 'const'.
        //:
        //: 3 No accessor allocates any memory.
        //
        // Plan:
        //: 1 To test 'allocator', create object with various allocators and
        //:   ensure the returned value matches the supplied allocator.
        //:
        //: 2 To test 'capacity', create object with various initial capacities
        //:   and ensure the returned value matches the expected value.
        //:
        //: 3 Use the generator function to produce objects of arbitrary state
        //:   and verify the accessor return value against expected values.
        //:   (C-1)
        //:
        //: 4 The accessors will only be accessed from a 'const' reference to
        //:   the created object.  (C-2)
        //:
        //: 5 The default allocator will be used for all created objects
        //:   (excluding those used to test 'allocator') and the number of
        //:   allocation will be verified to ensure that no memory was
        //:   allocated during use of the accessors.  (C-3)
        //
        // Testing:
        //   bslma::Allocator *allocator() const;
        //   bsl::size_t capacity() const;
        //   bool isEmpty() const;
        //   bool isFull() const;
        //   bsl::size_t numElements() const;
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "BASIC ACCESSORS" << endl
                          << "===============" << endl;

        if (verbose) cout << "\nTesting 'allocator'." << endl;
        {
            Obj mX(8);  const Obj& X = mX;
            ASSERT(&defaultAllocator == X.allocator());
        }
        {
            Obj        mX(8, reinterpret_cast<bslma::TestAllocator *>(0));
            const Obj& X = mX;
            ASSERT(&defaultAllocator == X.allocator());
        }
        {
            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;
            ASSERT(&sa == X.allocator());
        }

        if (verbose) cout << "\nTesting 'capacity'." << endl;
        {
            Obj mX(8);  const Obj& X = mX;
            ASSERT(8 == X.capacity());
        }
        {
            Obj mX(16);  const Obj& X = mX;
            ASSERT(16 == X.capacity());
        }
        {
            Obj mX(17);  const Obj& X = mX;
            ASSERT(17 == X.capacity());
        }

        if (verbose) cout << "\nTesting residual accessors." << endl;
        {
            static const struct {
                int          d_lineNum;     // source line number
                const char  *d_spec_p;      // specification string
                bsl::size_t  d_numElement;  // expected length
            } DATA[] = {
                //line  spec                 ne
                //----  -------------------  --
                { L_,                    "",  0 },
                { L_,                   "a",  1 },
                { L_,                  "aa",  2 },
                { L_,                 "aaa",  3 },
                { L_,                   "b",  1 },
                { L_,                  "bb",  2 },
                { L_,                 "bbb",  3 },
                { L_,                "aaab",  4 },
                { L_,                "aaba",  4 },
                { L_,                "abaa",  4 },
                { L_,                "baaa",  4 },
                { L_,           "abcabcabc",  9 },
                { L_,           "cbaabcbac",  9 },

                { L_,                  "a~",  0 },
                { L_,                "aa~~",  0 },
                { L_,                "aab~",  0 },
                { L_,               "aba~a",  1 },
                { L_,              "aba~ab",  2 },
                { L_,             "aba~aba",  3 },
                { L_,            "aba~aba~",  0 },
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int         LINE = DATA[ti].d_lineNum;
                const char *const SPEC = DATA[ti].d_spec_p;
                const bsl::size_t NE   = DATA[ti].d_numElement;

                if (veryVerbose) {
                    T_ P_(LINE) P_(SPEC) P(NE);
                }

                Obj mX(16);  const Obj& X = gg(&mX, SPEC);

                bsls::Types::Int64 allocations =
                                             defaultAllocator.numAllocations();

                LOOP_ASSERT(LINE, NE        == X.numElements());
                LOOP_ASSERT(LINE, (0 == NE) == X.isEmpty());
                LOOP_ASSERT(LINE, false     == X.isFull());

                LOOP_ASSERT(LINE,
                            defaultAllocator.numAllocations() == allocations);
            }
        }
        {
            // test 'isFull'

            Obj mX(4);  const Obj& X = mX;

            ASSERT(false == X.isFull());

            mX.pushBack(0);

            ASSERT(false == X.isFull());

            mX.pushBack(0);

            ASSERT(false == X.isFull());

            mX.pushBack(0);

            ASSERT(false == X.isFull());

            mX.pushBack(0);

            ASSERT(true  == X.isFull());

            int value;
            mX.popFront(&value);

            ASSERT(false == X.isFull());

            mX.pushBack(0);

            ASSERT(true  == X.isFull());
        }
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // PRIMITIVE GENERATOR FUNCTIONS
        //   Ensure that the generator functions are able to create an object
        //   in any state.
        //
        // Concerns:
        //: 1 Valid syntax produces the expected results.
        //:
        //: 2 Invalid syntax is detected and reported.
        //
        // Plan:
        //: 1 Evaluate a series of test strings of increasing complexity to
        //:   set the state of a newly created object and verify the returned
        //:   object using basic accessors and 'popFront'.  (C-1)
        //:
        //: 2 Evaluate the 'ggg' function with a series of test strings of
        //:   increasing complexity and verify its return value.  (C-2)
        //
        // Testing:
        //    Obj& gg(Obj *object, const char *spec);
        //    int ggg(Obj *object, const char *spec);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "PRIMITIVE GENERATOR FUNCTIONS" << endl
                          << "=============================" << endl;

        if (verbose) cout << "\nTesting generator on valid specs." << endl;
        {
            static const struct {
                int          d_lineNum;          // source line number
                const char  *d_spec_p;           // specification string
                const char  *d_exp_p;            // expected value string
            } DATA[] = {
                //line  spec                 expected
                //----  -------------------  --------------
                { L_,                    "",             "" },
                { L_,                   "a",            "a" },
                { L_,                  "aa",           "aa" },
                { L_,                 "aaa",          "aaa" },
                { L_,                   "b",            "b" },
                { L_,                  "bb",           "bb" },
                { L_,                 "bbb",          "bbb" },
                { L_,                "aaab",         "aaab" },
                { L_,                "aaba",         "aaba" },
                { L_,                "abaa",         "abaa" },
                { L_,                "baaa",         "baaa" },
                { L_,           "abcabcabc",    "abcabcabc" },
                { L_,           "cbaabcbac",    "cbaabcbac" },

                { L_,                  "a~",             "" },
                { L_,                "aa~~",             "" },
                { L_,                "aab~",             "" },
                { L_,               "aba~a",            "a" },
                { L_,              "aba~ab",           "ab" },
                { L_,             "aba~aba",          "aba" },
                { L_,            "aba~aba~",             "" }
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int         LINE = DATA[ti].d_lineNum;
                const char *const SPEC = DATA[ti].d_spec_p;
                const char *const EXP  = DATA[ti].d_exp_p;
                const bsl::size_t LEN  = strlen(EXP);

                if (veryVerbose) { P_(LINE) P_(SPEC) P(EXP) }

                Obj mX(16);  const Obj& X = gg(&mX, SPEC);  // original spec

                LOOP_ASSERT(LINE, LEN == X.numElements());

                for (bsl::size_t i = 0; i < LEN; i++) {
                    Obj::value_type value;

                    int rv = mX.popFront(&value);
                    ASSERT(e_SUCCESS == rv);

                    if (veryVerbose) { T_ P_(EXP[i]) P(value) }

                    Obj::value_type expValue;
                    getValue(&expValue, EXP[i], 0);

                    LOOP_ASSERT(LINE, expValue == value);
                }
            }
        }

        if (verbose) cout << "\nTesting generator on invalid specs." << endl;
        {
            static const struct {
                int         d_lineNum;  // source line number
                const char *d_spec_p;   // specification string
                int         d_index;    // offending character index
            } DATA[] = {
                //line  spec            index
                //----  -------------   -----
                { L_,   "",             -1,     }, // control

                { L_,   "a",            -1,     }, // control
                { L_,   " ",             0,     },
                { L_,   ".",             0,     },
                { L_,   "2",             0,     },

                { L_,   "ab",           -1,     }, // control
                { L_,   " a",            0,     },
                { L_,   ".a",            0,     },
                { L_,   "2a",            0,     },
                { L_,   "a ",            1,     },
                { L_,   "a.",            1,     },
                { L_,   "a2",            1,     },

                { L_,   "abc",          -1,     }, // control
                { L_,   " bc",           0,     },
                { L_,   ".bc",           0,     },
                { L_,   "2bc",           0,     },
                { L_,   "b c",           1,     },
                { L_,   "b.c",           1,     },
                { L_,   "b2c",           1,     },
                { L_,   "bc ",           2,     },
                { L_,   "bc.",           2,     },
                { L_,   "bc2",           2,     },
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int          LINE  = DATA[ti].d_lineNum;
                const char *const  SPEC  = DATA[ti].d_spec_p;
                const int          INDEX = DATA[ti].d_index;

                if (veryVerbose) { P_(SPEC) P(INDEX) }
                {
                    Obj mX(8);

                    int result = ggg(&mX, SPEC, veryVerbose);
                    LOOP_ASSERT(LINE, INDEX == result);
                }
            }
        }
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // PRIMARY MANIPULATORS TEST
        //   The basic concern is that the default constructor, the destructor,
        //   'popFront', and the primary manipulators:
        //      - pushBack
        //      - removeAll
        //   operate as expected.
        //
        // Concerns:
        //: 1 The default constructor creates the correct initial value and has
        //:   the internal memory management system hooked up properly so that
        //:   *all* internally allocated memory draws from the same
        //:   user-supplied allocator whenever one is specified.
        //:
        //: 2 The method 'pushBack' produces the expected value, increases
        //:   capacity as needed, and is exception neutral with respect to
        //:   memory allocation.
        //:
        //: 3 The method 'removeAll' produces the expected value (empty) and
        //:   does not affect allocated memory.
        //:
        //: 4 The method 'popFront' produces the expected value, does not
        //:   affect allocated memory, and is exception neutral with respect to
        //:   memory allocation.
        //:
        //: 5 Memory is not leaked by any method and the destructor properly
        //:   deallocates the residual allocated memory.
        //
        // Plan:
        //: 1 Create an object using the default constructor with and without
        //:   passing in an allocator, verify the allocator is stored using the
        //:   (untested) 'allocator' accessor, and verifying all allocations
        //:   are done from the allocator in future tests.
        //:
        //: 2 Create objects using the
        //:   'bslma::TestAllocator', use the 'pushBack' method with various
        //:   values, and the (untested) accessors to verify the value of the
        //:   object and that allocation occurred when expected.  Also vary the
        //:   test allocator's allocation limit to verify behavior in the
        //:   presence of exceptions.  (C-1,2)
        //:
        //: 3 Create objects using the 'bslma::TestAllocator', use 'pushBack'
        //:   to obtain various states, use 'removeAll', verify the objects are
        //:   empty, then repopulate the objects and ensure no allocation
        //:   occurs.  (C-3)
        //:
        //: 4 Create objects using the 'bslma::TestAllocator', use 'pushBack'
        //:   to obtain various states, use 'popFront', verify the objects have
        //:   a reduced length but unchanged capacity.  Also vary the test
        //:   allocator's allocation limit to verify behavior in the presence
        //:   of exceptions.  (C-4)
        //:
        //: 5 Use a supplied 'bslma::TestAllocator' that goes out-of-scope
        //:   at the conclusion of each test to ensure all memory is returned
        //:   to the allocator.  (C-5)
        //
        // Testing:
        //   BoundedQueue(bsl::size_t capacity, bslma::Allocator bA = 0);
        //   ~BoundedQueue();
        //   int pushBack(const TYPE& value);
        //   int popFront(TYPE *value);
        //   void removeAll();
        //   CONCERN: 0 == e_SUCCESS
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "PRIMARY MANIPULATORS TEST" << endl
                          << "=========================" << endl;

        ASSERT(0 == Obj::e_SUCCESS);

        bsl::string longString;
        {
            longString = "abc";
            for (bsl::size_t i = 0; i < sizeof(bsl::string); ++i) {
                longString += " ";
            }
        }

        if (verbose) cout << "\nTesting with various allocator configurations."
                          << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            Obj mX(8);  const Obj& X = mX;
            ASSERT(&defaultAllocator == X.allocator());
            ASSERT(0 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(1 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(2 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(3 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());
        }
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            Obj        mX(8, reinterpret_cast<bslma::TestAllocator *>(0));
            const Obj& X = mX;
            ASSERT(&defaultAllocator == X.allocator());
            ASSERT(0 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(1 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(2 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());

            mX.pushBack(0);
            ASSERT(3 == X.numElements());
            ASSERT(allocations + 1 == defaultAllocator.numAllocations());
        }
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;
            ASSERT(&sa == X.allocator());
            ASSERT(0 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(1 == sa.numAllocations());
            ASSERT(8 * sizeof(int) == sa.numBytesTotal());

            mX.pushBack(0);
            ASSERT(1 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(1 == sa.numAllocations());

            mX.pushBack(0);
            ASSERT(2 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(1 == sa.numAllocations());

            mX.pushBack(0);
            ASSERT(3 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(1 == sa.numAllocations());
        }
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            AllocObj mX(8, &sa);  const AllocObj& X = mX;
            ASSERT(&sa == X.allocator());
            ASSERT(0 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(1 == sa.numAllocations());

            mX.pushBack(longString);
            ASSERT(1 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(2 == sa.numAllocations());

            mX.pushBack(longString);
            ASSERT(2 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(3 == sa.numAllocations());

            mX.pushBack(longString);
            ASSERT(3 == X.numElements());
            ASSERT(allocations == defaultAllocator.numAllocations());
            ASSERT(4 == sa.numAllocations());
        }

        if (verbose) cout << "\nTesting 'pushBack'." << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(sa) {
                Obj mX(8, &sa);  const Obj& X = mX;
                ASSERT(0 == X.numElements());

                for (int i = 1; i <= 8; ++i) {
                    if (veryVerbose) {
                        P(i);
                    }

                    mX.pushBack(i);
                    ASSERT(static_cast<bsl::size_t>(i) == X.numElements());
                }
            } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(sa) {
                AllocObj mX(8, &sa);  const AllocObj& X = mX;
                ASSERT(0 == X.numElements());

                for (int i = 1; i <= 8; ++i) {
                    if (veryVerbose) {
                        P(i);
                    }

                    mX.pushBack(longString);
                    ASSERT(static_cast<bsl::size_t>(i) == X.numElements());
                }
            } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#ifdef BDE_BUILD_TARGET_EXC
        {
            // white-box test for when the element copy/move throws

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<AllocExceptionHelper>        mX(8, &sa);
            const bdlcc::BoundedQueue<AllocExceptionHelper>& X = mX;

            AllocExceptionHelper value(&sa);

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.pushBack(value);
            mX.pushBack(value);

            ASSERT(     2 == X.numElements());
            ASSERT(na + 2 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            int numException = 0;

            sa.setAllocationLimit(0);
            try {
                mX.pushBack(value);
            } catch (BloombergLP::bslma::TestAllocatorException& e) {
                ++numException;
            }
            sa.setAllocationLimit(-1);

            ASSERT(     1 == numException);
            ASSERT(     2 == X.numElements());

            // The test allocator increments the number of allocations and then
            // throws the exception.

            ASSERT(     2 == X.numElements());
            ASSERT(na + 3 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            mX.pushBack(value);

            ASSERT(     3 == X.numElements());
            ASSERT(na + 4 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            mX.popFront(&value);

            ASSERT(     2 == X.numElements());
            ASSERT(na + 5 == sa.numAllocations());
            ASSERT(nd + 2 == sa.numDeallocations());

            mX.popFront(&value);

            ASSERT(     1 == X.numElements());
            ASSERT(na + 6 == sa.numAllocations());
            ASSERT(nd + 4 == sa.numDeallocations());

            int rv = mX.tryPopFront(&value);

            ASSERT(e_SUCCESS == rv);
            ASSERT(     0 == X.numElements());
        }
#endif

        if (verbose) cout << "\nTesting 'removeAll'." << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            ASSERT( 8 == X.numElements());

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.removeAll();
            ASSERT( 0 == X.numElements());
            ASSERT(na == sa.numAllocations());
            ASSERT(nd == sa.numDeallocations());

            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            mX.pushBack(0);
            ASSERT( 8 == X.numElements());
            ASSERT(na == sa.numAllocations());
            ASSERT(nd == sa.numDeallocations());

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#ifdef BDE_BUILD_TARGET_EXC
        {
            // white-box test for when the element copy/move throws

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<AllocExceptionHelper>        mX(8, &sa);
            const bdlcc::BoundedQueue<AllocExceptionHelper>& X = mX;

            AllocExceptionHelper value(&sa);

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.pushBack(value);
            mX.pushBack(value);

            ASSERT(      2 == X.numElements());
            ASSERT(na +  2 == sa.numAllocations());
            ASSERT(nd      == sa.numDeallocations());

            int numException = 0;

            sa.setAllocationLimit(0);
            try {
                mX.pushBack(value);
            } catch (BloombergLP::bslma::TestAllocatorException& e) {
                ++numException;
            }
            sa.setAllocationLimit(-1);

            ASSERT(1 == numException);
            ASSERT(2 == X.numElements());

            // The test allocator increments the number of allocations and then
            // throws the exception.

            ASSERT(      2 == X.numElements());
            ASSERT(na +  3 == sa.numAllocations());
            ASSERT(nd      == sa.numDeallocations());

            mX.pushBack(value);

            ASSERT(      3 == X.numElements());
            ASSERT(na +  4 == sa.numAllocations());
            ASSERT(nd      == sa.numDeallocations());

            mX.removeAll();

            ASSERT(      0 == X.numElements());
            ASSERT(na +  4 == sa.numAllocations());
            ASSERT(nd +  3 == sa.numDeallocations());
        }
#endif

        if (verbose) cout << "\nTesting 'popFront'." << endl;
        {
            bsls::Types::Int64 allocations = defaultAllocator.numAllocations();

            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            Obj mX(8, &sa);  const Obj& X = mX;
            mX.pushBack(0);
            mX.pushBack(1);
            mX.pushBack(2);
            mX.pushBack(3);
            mX.pushBack(4);
            mX.pushBack(5);
            mX.pushBack(6);
            mX.pushBack(7);
            ASSERT( 8 == X.numElements());

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            int value;
            int rv;

            rv = mX.popFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        7 == X.numElements());
            ASSERT(        0 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            rv = mX.popFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        6 == X.numElements());
            ASSERT(        1 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            rv = mX.popFront(&value);
            ASSERT(e_SUCCESS == rv);
            ASSERT(        5 == X.numElements());
            ASSERT(        2 == value);
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            mX.pushBack(0);
            mX.pushBack(0);
            ASSERT(        7 == X.numElements());
            ASSERT(       na == sa.numAllocations());
            ASSERT(       nd == sa.numDeallocations());

            ASSERT(allocations == defaultAllocator.numAllocations());
        }
#ifdef BDE_BUILD_TARGET_EXC
        {
            bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

            bdlcc::BoundedQueue<AllocExceptionHelper>        mX(8, &sa);
            const bdlcc::BoundedQueue<AllocExceptionHelper>& X = mX;

            AllocExceptionHelper value(&sa);

            bsls::Types::Int64 na = sa.numAllocations();
            bsls::Types::Int64 nd = sa.numDeallocations();

            mX.pushBack(value);
            mX.pushBack(value);

            ASSERT(     2 == X.numElements());
            ASSERT(na + 2 == sa.numAllocations());
            ASSERT(nd     == sa.numDeallocations());

            int numException = 0;

            sa.setAllocationLimit(0);
            try {
                mX.popFront(&value);
            } catch (BloombergLP::bslma::TestAllocatorException& e) {
                ++numException;
            }
            sa.setAllocationLimit(-1);

            ASSERT(1 == numException);
            ASSERT(1 == X.numElements());

            // The test allocator increments the number of allocations and then
            // throws the exception.

            ASSERT(na + 3 == sa.numAllocations());
            ASSERT(nd + 1 == sa.numDeallocations());

            mX.popFront(&value);

            ASSERT(na + 4 == sa.numAllocations());
            ASSERT(nd + 3 == sa.numDeallocations());

            ASSERT(0 == X.numElements());

            mX.pushBack(value);
            ASSERT(1 == X.numElements());

            // Since the queue is empty, only the element should have
            // allocated.

            ASSERT(na + 5 == sa.numAllocations());
            ASSERT(nd + 3 == sa.numDeallocations());
        }
#endif
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //   This case exercises (but does not fully test) basic functionality.
        //
        // Concerns:
        //: 1 The class is sufficiently functional to enable comprehensive
        //:   testing in subsequent test cases.
        //
        // Plan:
        //: 1 Instantiate an object and verify basic functionality.  (C-1)
        //
        // Testing:
        //   BREATHING TEST
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "BREATHING TEST" << endl
                          << "==============" << endl;

        Obj mX(8);  const Obj& X = mX;

        ASSERT(0 == X.numElements());

        mX.pushBack(1);

        ASSERT(1 == X.numElements());

        mX.pushBack(2);

        ASSERT(2 == X.numElements());

        mX.pushBack(3);

        ASSERT(3 == X.numElements());

        int v;

        mX.popFront(&v);

        ASSERT(1 == v);
        ASSERT(2 == X.numElements());

        mX.popFront(&v);

        ASSERT(2 == v);
        ASSERT(1 == X.numElements());

        mX.popFront(&v);

        ASSERT(3 == v);
        ASSERT(0 == X.numElements());
      } break;
      default: {
        cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    // CONCERN: In no case does memory come from the global allocator.

    LOOP_ASSERT(globalAllocator.numBlocksTotal(),
                0 == globalAllocator.numBlocksTotal());

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2019 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------- END-OF-FILE ----------------------------------
