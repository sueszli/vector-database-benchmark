// balm_integercollector.t.cpp                                        -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <balm_integercollector.h>

#include <bslma_testallocator.h>
#include <bslmt_barrier.h>
#include <bdlmt_fixedthreadpool.h>
#include <bdlf_bind.h>

#include <bsla_maybeunused.h>

#include <bslma_defaultallocatorguard.h>
#include <bslma_testallocator.h>

#include <bsls_types.h>

#include <bsl_functional.h>
#include <bsl_ostream.h>
#include <bsl_cstring.h>
#include <bsl_cstdlib.h>
#include <bsl_iostream.h>
#include <bsl_sstream.h>

#include <bslim_testutil.h>

using namespace BloombergLP;

using bsl::cout;
using bsl::endl;
using bsl::flush;

// ============================================================================
//                                 TEST PLAN
// ----------------------------------------------------------------------------
//                                  Overview
//                                  --------
// The 'balm::IntegerCollector' is a mechanism for collecting and recording
// aggregated metric values.  Ensure values can be accumulated into and read
// out of the container, and that the operations are thread safe.
// ----------------------------------------------------------------------------
// CREATORS
// [ 3]  balm::Collector(const balm::MetricId& metric);
// [ 3]  ~balm::Collector();
//
// MANIPULATORS
// [ 7]  void reset();
// [ 6]  void loadAndReset(balm::MetricRecord *records);
// [ 2]  void update(int value);
// [ 5]  void accumulateCountTotalMinMax(int count,
//                                       int total,
//                                       int min,
//                                       int max);
// [ 4]  void setCountTotalMinMax(int count, int total, int min, int max);
//
// ACCESSORS
// [ 2]  const balm::MetricId& metric() const;
// [ 2]  void load(balm::MetricRecord *record) const;
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [ 8] CONCURRENCY TEST
// [ 9] USAGE EXAMPLE

// ============================================================================
//                      STANDARD BDE ASSERT TEST MACRO
// ----------------------------------------------------------------------------
static int testStatus = 0;

static void aSsErT(int c, const char *s, int i)
{
    if (c) {
        bsl::cout << "Error " << __FILE__ << "(" << i << "): " << s
                  << "    (failed)" << bsl::endl;
        if (0 <= testStatus && testStatus <= 100) ++testStatus;
    }
}

// ============================================================================
//                      STANDARD BDE TEST DRIVER MACROS
// ----------------------------------------------------------------------------

#define ASSERT       BSLIM_TESTUTIL_ASSERT
#define LOOP_ASSERT  BSLIM_TESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLIM_TESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLIM_TESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLIM_TESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLIM_TESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLIM_TESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLIM_TESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLIM_TESTUTIL_LOOP6_ASSERT
#define ASSERTV      BSLIM_TESTUTIL_ASSERTV

#define Q   BSLIM_TESTUTIL_Q   // Quote identifier literally.
#define P   BSLIM_TESTUTIL_P   // Print identifier and value.
#define P_  BSLIM_TESTUTIL_P_  // P(X) without '\n'.
#define T_  BSLIM_TESTUTIL_T_  // Print a tab (w/o newline).
#define L_  BSLIM_TESTUTIL_L_  // current Line number

// ============================================================================
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

typedef balm::IntegerCollector  Obj;
typedef balm::MetricRecord      Rec;
typedef balm::MetricDescription Desc;
typedef balm::MetricId          Id;

// ============================================================================
//                      GLOBAL STUB CLASSES FOR TESTING
// ----------------------------------------------------------------------------

class ConcurrencyTest {
    // Invoke a set of operations operations synchronously.

    // DATA
    bdlmt::FixedThreadPool  d_pool;
    bslmt::Barrier          d_barrier;
    balm::IntegerCollector *d_collector_p;
    bslma::Allocator       *d_allocator_p;

    // PRIVATE MANIPULATORS
    void execute();
        // Execute a single test.

  public:

    // CREATORS
    ConcurrencyTest(int                    numThreads,
                    balm::IntegerCollector *collector,
                    bslma::Allocator      *basicAllocator)
    : d_pool(numThreads, 1000, basicAllocator)
    , d_barrier(numThreads)
    , d_collector_p(collector)
    , d_allocator_p(basicAllocator)
    {
        d_pool.start();
    }

    ~ConcurrencyTest() {}

    //  MANIPULATORS
    void runTest();
        // Run the test.
};

void ConcurrencyTest::execute()
{

    Obj *mX = d_collector_p; const Obj *MX = mX;

    balm::MetricRecord empty(MX->metricId());
    balm::MetricRecord r1(MX->metricId(), 1, 2, 3, 5);
    balm::MetricRecord r2(MX->metricId(), 7, 8, 9, 10);

    // Test simultaneous set and loads, verify values are set/loaded
    // atomically.
    d_barrier.wait();
    for(int i = 0; i < 10; ++i) {
        balm::MetricRecord  result;
        balm::MetricRecord *rec = (0 == i % 2) ? &r1 : &r2;
        mX->setCountTotalMinMax(rec->count(),
                                static_cast<int>(rec->total()),
                                static_cast<int>(rec->min()),
                                static_cast<int>(rec->max()));
        MX->load(&result);
        ASSERT(r1 == result || r2 == result);

    }
    d_barrier.wait();

    mX->reset();

    // Test simultaneous accumulate and loads, verify values are
    // accumulated/loaded atomically.
    d_barrier.wait();
    for(int i = 0; i < 10; ++i) {
        balm::MetricRecord  result;
        mX->accumulateCountTotalMinMax(1, 1, -i, i);
        MX->load(&result);

        ASSERT(result.count() >= i);
        ASSERT(result.total() >= i);
        ASSERT(result.count() == result.total());
        ASSERT(result.min() <= -i);
        ASSERT(result.max() >= i);
        ASSERT(result.min() == -result.max());
    }
    d_barrier.wait();

    mX->reset();

    // Test simultaneous set and loadAndReset, verify values are loaded and
    // reset atomically.
    d_barrier.wait();
    for(int i = 0; i < 10; ++i) {
        balm::MetricRecord  result;
        mX->setCountTotalMinMax(r1.count(),
                                static_cast<int>(r1.total()),
                                static_cast<int>(r1.min()),
                                static_cast<int>(r1.max()));
        mX->loadAndReset(&result);

        ASSERT(result == r1 || result == empty);
    }
    d_barrier.wait();
}

void ConcurrencyTest::runTest()
{
    bsl::function<void()> job = bdlf::BindUtil::bind(&ConcurrencyTest::execute,
                                                      this);
    for (int i = 0; i < d_pool.numThreads(); ++i) {
        d_pool.enqueueJob(job);
    }
    d_pool.drain();
}

// ============================================================================
//                               USAGE EXAMPLE
// ----------------------------------------------------------------------------


// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test = argc > 1 ? bsl::atoi(argv[1]) : 0;

    BSLA_MAYBE_UNUSED int verbose = argc > 2;
    BSLA_MAYBE_UNUSED int veryVerbose = argc > 3;
    BSLA_MAYBE_UNUSED int veryVeryVerbose = argc > 4;

    bsl::cout << "TEST " << __FILE__ << " CASE " << test << bsl::endl;;

    balm::Category cat_A("A", true);
    Desc desc_A(&cat_A, "A"); const Desc *DESC_A = &desc_A;
    Desc desc_B(&cat_A, "B"); const Desc *DESC_B = &desc_B;
    Desc desc_C(&cat_A, "C"); const Desc *DESC_C = &desc_C;
    Desc desc_D(&cat_A, "D"); const Desc *DESC_D = &desc_D;
    Desc desc_E(&cat_A, "E"); const Desc *DESC_E = &desc_E;

    Id metric_A(DESC_A); const Id& METRIC_A = metric_A;
    Id metric_B(DESC_B); const Id& METRIC_B = metric_B;

    switch (test) { case 0:  // Zero is always the leading case.
      case 9: {
        // --------------------------------------------------------------------
        // TESTING USAGE EXAMPLE
        //
        // Concerns:
        //   The usage example provided in the component header file must
        //   compile, link, and run on all platforms as shown.
        //
        // Plan:
        //   Incorporate usage example from header into driver, remove leading
        //   comment characters, and replace 'assert' with 'ASSERT'.
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting Usage Example"
                          << "\n=====================" << endl;

///Usage
///-----
// The following example creates a 'balm::IntegerCollector', modifies its
// values, then collects a 'balm::MetricRecord'.
//
// We start by creating 'balm::MetricId' object by hand; however, in practice
// an id should be obtained from a 'balm::MetricRegistry' object (such as the
// one owned by a 'balm::MetricsManager').
//..
    balm::Category           myCategory("MyCategory");
    balm::MetricDescription  description(&myCategory, "MyMetric");
    balm::MetricId           myMetric(&description);
//..
// Now we create a 'balm::IntegerCollector' object using 'myMetric' and use the
// 'update' method to update its collected value.
//..
    balm::IntegerCollector collector(myMetric);

    collector.update(1);
    collector.update(3);
//..
// The collector accumulated the values 1 and 3.  The result should have a
// count of 2, a total of 4 (3 + 1), a max of 3 (max(3,1)), and a min of 1
// (min(3,1)).
//..
    balm::MetricRecord record;
    collector.loadAndReset(&record);

        ASSERT(myMetric == record.metricId());
        ASSERT(2        == record.count());
        ASSERT(4        == record.total());
        ASSERT(1        == record.min());
        ASSERT(3        == record.max());
//..

      } break;
      case 8: {
        // --------------------------------------------------------------------
        // CONCURRENCY TEST
        //
        // Testing:
        //     Thread-safety of manipulator methods
        //
        // --------------------------------------------------------------------

        if (verbose) cout << endl << "TEST CONCURRENCY" << endl
                                  << "================" << endl;

        bslma::TestAllocator defaultAllocator;
        bslma::DefaultAllocatorGuard guard(&defaultAllocator);

        bslma::TestAllocator testAllocator;
        balm::IntegerCollector mX(METRIC_A);
        {
            ConcurrencyTest tester(10, &mX, &defaultAllocator);
            tester.runTest();
        }
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // TESTING ADDITIONAL MANIPULATOR: reset
        //
        // Concerns:
        //   reset
        //
        // Plan:
        //   Create a table of test values.  Iterate over the test values
        //   set them on the container.  Verify load returns the
        //   correct value, call reset, verify the value is the default value.
        //
        // Testing:
        // void reset()
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting reset." << endl;
        struct {
            const Desc *d_id;
            int         d_count;
            int         d_total;
            int         d_min;
            int         d_max;
        } VALUES [] = {
            {      0,      0,        0,           0,           0 },
            { DESC_A,      1,        1,           1,           1 },
            { DESC_B,      1,        2,           3,           4 },
            { DESC_C,     -1,       -2,          -3,          -4 },
            { DESC_D, 100000,  2000000,     3000000,     4000000 },
            { DESC_E, INT_MAX, INT_MAX, INT_MIN + 1, INT_MAX - 1 },
            { DESC_A, INT_MAX, INT_MIN, INT_MAX - 1, INT_MIN + 1 }
        };
        const int NUM_VALUES = sizeof(VALUES)/sizeof(*VALUES);


        for (int i = 0; i < NUM_VALUES; ++i) {
            balm::MetricRecord r1, r2;
            Obj mX(Id(VALUES[i].d_id)); const Obj& MX = mX;
            mX.setCountTotalMinMax(VALUES[i].d_count,
                                   VALUES[i].d_total,
                                   VALUES[i].d_min,
                                   VALUES[i].d_max);
            MX.load(&r1);
            ASSERT(VALUES[i].d_id    == r1.metricId());
            ASSERT(VALUES[i].d_count == r1.count());
            ASSERT(VALUES[i].d_total == r1.total());
            ASSERT(VALUES[i].d_min   == r1.min());
            ASSERT(VALUES[i].d_max   == r1.max());

            mX.reset();
            MX.load(&r2);
            ASSERT(VALUES[i].d_id    == r2.metricId());
            ASSERT(0                 == r2.count());
            ASSERT(0.0               == r2.total());
            ASSERT(Rec::k_DEFAULT_MIN  == r2.min());
            ASSERT(Rec::k_DEFAULT_MAX  == r2.max());
        }

      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING ADDITIONAL MANIPULATOR: loadAndReset
        //
        // Concerns:
        //   loadAndReset
        //
        // Plan:
        //   Create a table of test values.  Iterate over the test values
        //   set them on the container.  Verify loadAndReset returns the
        //   correct values and that the collector is left in the correct
        //   state.
        //
        // Testing:
        // void loadAndReset(balm::MetricRecord *);
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting loadAndReset." << endl;
        struct {
            const Desc *d_id;
            int         d_count;
            int         d_total;
            int         d_min;
            int         d_max;
        } VALUES [] = {
            {      0,      0,        0,           0,           0 },
            { DESC_A,      1,        1,           1,           1 },
            { DESC_B,      1,        2,           3,           4 },
            { DESC_C,     -1,       -2,          -3,          -4 },
            { DESC_D, 100000,  2000000,     3000000,     4000000 },
            { DESC_E, INT_MAX, INT_MAX, INT_MIN + 1, INT_MAX - 1 },
            { DESC_A, INT_MAX, INT_MIN, INT_MAX - 1, INT_MIN + 1 }
        };
        const int NUM_VALUES = sizeof(VALUES)/sizeof(*VALUES);


        for (int i = 0; i < NUM_VALUES; ++i) {
            balm::MetricRecord r1, r2, r3;
            Obj mX(Id(VALUES[i].d_id)); const Obj& MX = mX;
            mX.setCountTotalMinMax(VALUES[i].d_count,
                                   VALUES[i].d_total,
                                   VALUES[i].d_min,
                                   VALUES[i].d_max);
            MX.load(&r1);
            ASSERT(VALUES[i].d_id    == r1.metricId());
            ASSERT(VALUES[i].d_count == r1.count());
            ASSERT(VALUES[i].d_total == r1.total());
            ASSERT(VALUES[i].d_min   == r1.min());
            ASSERT(VALUES[i].d_max   == r1.max());

            mX.loadAndReset(&r2);
            ASSERT(VALUES[i].d_id    == r2.metricId());
            ASSERT(VALUES[i].d_count == r2.count());
            ASSERT(VALUES[i].d_total == r2.total());
            ASSERT(VALUES[i].d_min   == r2.min());
            ASSERT(VALUES[i].d_max   == r2.max());

            MX.load(&r3);
            ASSERT(VALUES[i].d_id    == r3.metricId());
            ASSERT(0                 == r3.count());
            ASSERT(0.0               == r3.total());
            ASSERT(Rec::k_DEFAULT_MIN  == r3.min());
            ASSERT(Rec::k_DEFAULT_MAX  == r3.max());
        }

      } break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING ADDITIONAL MANIPULATOR: accumulateCountTotalMinMax
        //
        // Concerns:
        //   accumulateCountTotalMinMax
        //
        // Plan:
        //   Create a table of test values.  Iterate over the test values
        //   selecting the first value, then iterate over the test values
        //   starting at that first value and accumulate the values in the
        //   collector.  Maintain a separate count, total, min, and max and
        //   verify the values returned by the collector.
        //
        // Testing:
        // void accumulateCountTotalMinMax(int , int , int , int );
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting accumulateCountTotalMinMax." << endl;

        struct {
            int d_count;
            int d_total;
            int d_min;
            int d_max;
        } VALUES [] = {
            {         0,         0,               0,                0 },
            {         1,         1,               1,                1 },
            {      1210,   INT_MAX,     INT_MIN + 1,      INT_MAX - 1 },
            {         1,         2,               3,                4 },
            {        10,        -2,              -3,               -4 },
            {      123110, INT_MIN,     INT_MAX - 1,      INT_MIN + 1 },
            {     -12311,   231413,           -1123,           410001 },
         };
        const int NUM_VALUES = sizeof(VALUES)/sizeof(*VALUES);

        for (int i = 0; i < NUM_VALUES; ++i) {
            Obj mX(0); const Obj& MX = mX;

            int                count = 0;
            bsls::Types::Int64 total = 0;
            int                min   = VALUES[i].d_min;
            int                max   = VALUES[i].d_max;
            for (int j = 0; j < NUM_VALUES; ++j) {
                int idx = (i + j) % NUM_VALUES;
                balm::MetricRecord r;

                count += VALUES[idx].d_count;
                total += VALUES[idx].d_total;
                min   =  bsl::min(VALUES[idx].d_min, min);
                max   =  bsl::max(VALUES[idx].d_max, max);

                mX.accumulateCountTotalMinMax(VALUES[idx].d_count,
                                              VALUES[idx].d_total,
                                              VALUES[idx].d_min,
                                              VALUES[idx].d_max);
                MX.load(&r);
                ASSERT(0     == r.metricId());
                ASSERT(count == r.count());
                ASSERT(total == static_cast<bsls::Types::Int64>(r.total()));
                ASSERT(min   == r.min());
                ASSERT(max   == r.max());
            }
        }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING ADDITIONAL MANIPULATOR: setCountTotalMinMax
        //
        // Concerns:
        //   setCountTotalMinMax
        //
        // Plan:
        //   Create a table of test values and set them to the object, verify
        //   that values are set correctly.
        //
        // Testing:
        // void setCountTotalMinMax(int , int , int , int );
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting setCountTotalMinMax." << endl;

        struct {
            int d_count;
            int d_total;
            int d_min;
            int d_max;
        } VALUES [] = {
            {       0,           0,           0,           0 },
            {       1,           1,           1,           1 },
            {       1,           2,           3,           4 },
            {      -1,          -2,          -3,          -4 },
            {  100000,     2000000,     3000000,     4000000 },
            { INT_MAX,     INT_MAX, INT_MIN + 1, INT_MAX - 1 },
            { INT_MAX,     INT_MIN, INT_MAX - 1, INT_MIN + 1 }
        };
        const int NUM_VALUES = sizeof(VALUES)/sizeof(*VALUES);

        Obj mX(0); const Obj& MX = mX;
        for (int i = 0; i < NUM_VALUES; ++i) {
            balm::MetricRecord r;
            mX.setCountTotalMinMax(VALUES[i].d_count,
                                   VALUES[i].d_total,
                                   VALUES[i].d_min,
                                   VALUES[i].d_max);
            MX.load(&r);
            ASSERT(0                 == r.metricId());
            ASSERT(VALUES[i].d_count == r.count());
            ASSERT(VALUES[i].d_total == r.total());
            ASSERT(VALUES[i].d_min   == r.min());
            ASSERT(VALUES[i].d_max   == r.max());
        }
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING CONSTRUCTORS
        //
        // Concerns:
        //   Test the constructor arguments
        //
        // Plan:
        //   Verify the constructor by passing a value from a table of values
        //   and verifying the object is initialized with the value..
        //
        //
        // Testing:
        //   balm::Collector(const balm::MetricId^ )
        //   ~balm::Collector()
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting constructor." << endl;

        const Desc *IDS[]  = { 0, DESC_A, DESC_B, DESC_C, DESC_D, DESC_E };
        const int NUM_IDS = sizeof(IDS)/sizeof(*IDS);
        for (int i = 0; i < NUM_IDS; ++i) {
            Obj mX(IDS[i]); const Obj& MX = mX;
            ASSERT(IDS[i] == MX.metricId());
        }

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING BASIC MANIPULATORS and ACCESSORS (BOOTSTRAP):
        //
        // Concerns:
        //   The primary fields must be correctly modifiable and accessible.
        //
        // Plan:
        //   First, verify the constructor by testing the value of the
        //   resulting object.
        //
        //   Next, for a sequence of independent test values, add elements to
        //   the container and used the basic accessors to verify the
        //   modification.
        //
        // Testing:
        //   balm::Collector(const balm::MetricId *)
        //   void update(int );
        //
        //   const balm::MetricId *metric() const;
        //   void load(balm::MetricRecord *record) const;
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting basic mainpulator and accessors."
                          << endl;

        int UPDATES[] = { 0, 12, -1321123, 2131241, 1321,
                            43145, 1, -1, INT_MIN + 1, INT_MAX - 1};
        const int NUM_UPDATES = sizeof(UPDATES)/sizeof(*UPDATES);

        for (int i = 0; i < NUM_UPDATES; ++i) {
            const Desc *METRIC = reinterpret_cast<const Desc *>(
                                      static_cast<bsls::Types::IntPtr>(i + 1));
            Obj mX(METRIC); const Obj& MX = mX;


            balm::MetricRecord r1, r2;
            MX.load(&r1);
            ASSERT(METRIC           == r1.metricId().description());
            ASSERT(0                == r1.count());
            ASSERT(0                == r1.total());
            ASSERT(Rec::k_DEFAULT_MIN == r1.min());
            ASSERT(Rec::k_DEFAULT_MAX == r1.max());

            bsls::Types::Int64 total = 0;
            int                min   = UPDATES[i];
            int                max   = UPDATES[i];
            for (int j = 0; j < NUM_UPDATES; ++j) {
                const int INDEX = (i + j) % NUM_UPDATES;
                mX.update(UPDATES[INDEX]);

                total += UPDATES[INDEX];
                min   = bsl::min(min, UPDATES[INDEX]);
                max   = bsl::max(max, UPDATES[INDEX]);

                MX.load(&r1);
                ASSERT(METRIC == r1.metricId().description());
                ASSERT(j + 1  == r1.count());
                ASSERT(total  == static_cast<bsls::Types::Int64>(r1.total()));
                ASSERT(min    == r1.min());
                LOOP2_ASSERT(max, r1.max(), max    == r1.max());

                MX.load(&r2);
                ASSERT(r1 == r2);
            }
        }
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST:
        //   Developers' Sandbox.
        //
        // Plan:
        //   Perform ad-hoc test of the primary modifiers and accessors.
        //
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "BREATHING TEST" << endl
                          << "==============" << endl;

        Obj mX(METRIC_A); const Obj& MX = mX;
        Obj mY(METRIC_B); const Obj& MY = mY;

        ASSERT(METRIC_A == MX.metricId());
        ASSERT(METRIC_B == MY.metricId());

        balm::MetricRecord r1, r2;

        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(0  == r1.count());
        ASSERT(0  == r1.total());
        ASSERT(Rec::k_DEFAULT_MIN == r1.min());
        ASSERT(Rec::k_DEFAULT_MAX == r1.max());

        MY.load(&r2);
        ASSERT(METRIC_B == r2.metricId());
        ASSERT(0  == r2.count());
        ASSERT(0  == r2.total());
        ASSERT(Rec::k_DEFAULT_MIN == r2.min());
        ASSERT(Rec::k_DEFAULT_MAX == r2.max());

        mX.update(1);
        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(1  == r1.count());
        ASSERT(1  == r1.total());
        ASSERT(1  == r1.min());
        ASSERT(1  == r1.max());

        mX.update(2);
        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(2  == r1.count());
        ASSERT(3  == r1.total());
        ASSERT(1  == r1.min());
        ASSERT(2  == r1.max());

        mX.update(-5);
        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(3  == r1.count());
        ASSERT(-2 == r1.total());
        ASSERT(-5 == r1.min());
        ASSERT(2  == r1.max());

        mX.reset();
        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(0  == r1.count());
        ASSERT(0  == r1.total());
        ASSERT(Rec::k_DEFAULT_MIN == r1.min());
        ASSERT(Rec::k_DEFAULT_MAX == r1.max());

        mX.update(3);
        mX.loadAndReset(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(1  == r1.count());
        ASSERT(3  == r1.total());
        ASSERT(3  == r1.min());
        ASSERT(3 == r1.max());

        MX.load(&r1);
        ASSERT(METRIC_A == r1.metricId());
        ASSERT(0  == r1.count());
        ASSERT(0  == r1.total());
        ASSERT(Rec::k_DEFAULT_MIN == r1.min());
        ASSERT(Rec::k_DEFAULT_MAX == r1.max());
      } break;
      default: {
        bsl::cerr << "WARNING: CASE `" << test << "' NOT FOUND." << bsl::endl;
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        bsl::cerr << "Error, non-zero test status = " << testStatus << "."
                  << bsl::endl;
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2015 Bloomberg Finance L.P.
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
