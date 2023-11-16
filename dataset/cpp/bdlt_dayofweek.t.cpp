// bdlt_dayofweek.t.cpp                                               -*-C++-*-
#include <bdlt_dayofweek.h>

#include <bslim_testutil.h>

#include <bslma_testallocator.h>
#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bsls_assert.h>
#include <bsls_asserttest.h>

#include <bslx_byteinstream.h>
#include <bslx_byteoutstream.h>
#include <bslx_instreamfunctions.h>
#include <bslx_outstreamfunctions.h>
#include <bslx_testinstream.h>
#include <bslx_testinstreamexception.h>
#include <bslx_testoutstream.h>
#include <bslx_versionfunctions.h>

#include <bsl_cstdlib.h>     // 'atoi'
#include <bsl_cstring.h>     // 'strcmp'
#include <bsl_iostream.h>
#include <bsl_set.h>
#include <bsl_sstream.h>

using namespace BloombergLP;
using namespace bsl;

// ============================================================================
//                                TEST PLAN
//-----------------------------------------------------------------------------
//                                Overview
//                                --------
// Standard enumeration test plan.
// ----------------------------------------------------------------------------
// TYPES
// [ 1] enum Enum { ... };
//
// CLASS METHODS
// [ 4] static STREAM& bdexStreamIn(STREAM& s, Enum& variable, int v);
// [ 4] static STREAM& bdexStreamOut(STREAM& s, Enum value, int v) const;
// [ 4] static int maxSupportedBdexVersion(int versionSelector);
// [ 2] ostream& print(ostream&, DayOfWeek::Enum, int, int)
// [ 1] static const char *toAscii(Enum dayOfWeek);
//
// FREE OPERATORS
// [ 3] operator<<(ostream&, DayOfWeek::Enum);
//
// FREE FUNCTIONS
// [ 4] STREAM& bdexStreamIn(STREAM& s, Enum& variable, int v);
// [ 4] STREAM& bdexStreamOut(STREAM& s, const Enum& value, int v) const;
// [ 4] int maxSupportedBdexVersion(const Enum *, int versionSelector);
#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation
// DEPRECATED
// [ 4] static int maxSupportedBdexVersion();
// [ 4] STREAM& streamIn(STREAM& s, Enum& variable, int v);
// [ 4] STREAM& streamOut(STREAM& s, const Enum& value, int v) const;
// [ 4] int maxSupportedVersion(Enum);
#endif // BDE_OPENSOURCE_PUBLICATION -- pending deprecation
// ----------------------------------------------------------------------------
// [ 5] USAGE EXAMPLE

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
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

typedef bdlt::DayOfWeek    Obj;
#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation
typedef Obj::Day           Enum;
#else
typedef Obj::Enum          Enum;
#endif // BDE_OPENSOURCE_PUBLICATION -- pending deprecation

enum { ABOVE_ENUM_RANGE = Obj::e_SAT + 1 };

typedef bslx::TestInStream  In;
typedef bslx::TestOutStream Out;

#define VERSION_SELECTOR 20140601

// ============================================================================
//                              MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const int                 test = argc > 1 ? atoi(argv[1]) : 0;
    const bool             verbose = argc > 2;
    const bool         veryVerbose = argc > 3;
    const bool     veryVeryVerbose = argc > 4;
    const bool veryVeryVeryVerbose = argc > 5;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    bslma::TestAllocator globalAlloc;
    bslma::TestAllocator dfltAlloc;

    bslma::Default::setGlobalAllocator(&globalAlloc);
    bslma::DefaultAllocatorGuard allocatorGuard(&dfltAlloc);

    switch (test) { case 0:
      case 5: {
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

        if (verbose) cout << endl
                          << "TESTING USAGE EXAMPLE" << endl
                          << "=====================" << endl;

///Usage
///-----
// The following snippets of code provide a simple illustration of
// 'bdlt::DayOfWeek' operation.
//
// First, create a variable, 'day', of type 'bdlt::DayOfWeek::Enum', and
// initialize it to the value 'bdlt::DayOfWeek::e_MONDAY':
//..
    bdlt::DayOfWeek::Enum day = bdlt::DayOfWeek::e_MONDAY;
//..
// Next, store its representation in a variable, 'rep', of type 'const char *':
//..
    const char *rep = bdlt::DayOfWeek::toAscii(day);
    ASSERT(0 == bsl::strcmp(rep, "MON"));
//..
// Finally, we print the value of 'day' to 'stdout':
//..

if (verbose) {  // added in test driver

    bsl::cout << day << bsl::endl;

}               // added in test driver
//..
// This statement produces the following output on 'stdout':
//..
//  MON
//..
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING BDEX STREAMING
        //   Verify the BDEX streaming implementation works correctly.
        //   Specific concerns include wire format, handling of stream states
        //   (valid, empty, invalid, incomplete, and corrupted), and exception
        //   neutrality.
        //
        // Concerns:
        //: 1 The class method 'maxSupportedBdexVersion' returns the correct
        //:   version to be used for the specified 'versionSelector'.
        //:
        //: 2 The 'bdexStreamOut' methods are callable on a reference providing
        //:   only non-modifiable access.
        //:
        //: 3 For valid streams, externalization and unexternalization are
        //:   inverse operations.
        //:
        //: 4 For invalid streams, externalization leaves the stream invalid
        //:   and unexternalization does not alter the value of the object and
        //:   leaves the stream invalid.
        //:
        //: 5 Unexternalizing of incomplete, invalid, or corrupted data results
        //:   in a valid object of unspecified value and an invalidated stream.
        //:
        //: 6 The wire format of the object is as expected.
        //:
        //: 7 All methods are exception neutral.
        //:
        //: 8 The 'bdexStreamIn' and 'bdexStreamOut' methods return a reference
        //:   to the provided stream in all situations.
        //:
        //: 9 The initial value of the object has no affect on
        //:   unexternalization.
        //
        // Plan:
        //: 1 Test 'maxSupportedBdexVersion' explicitly.  (C-1)
        //:
        //: 2 All calls to the 'bdexStreamOut' class method will be done with a
        //:   'const' object or reference and all calls to the 'bdexStreamOut'
        //:   free function (provided by 'bslx') will be supplied a 'const'
        //:   object or reference.  (C-2)
        //:
        //: 3 Perform a direct test of the 'bdexStreamOut' and 'bdexStreamIn'
        //:   methods (the rest of the testing will use the free functions
        //:   'bslx::OutStreamFunctions::bdexStreamOut' and
        //:   'bslx::InStreamFunctions::bdexStreamIn').
        //:
        //: 4 Define a set 'S' of test values to be used throughout the test
        //:   case.
        //:
        //: 5 For all '(u, v)' in the cross product 'S X S', stream the value
        //:   of 'u' into (a temporary copy of) 'v', 'T', and assert 'T == u'.
        //:   (C-3, 9)
        //:
        //: 6 For all 'u' in 'S', create a copy of 'u' and attempt to stream
        //:   into it from an invalid stream.  Verify after each attempt that
        //:   the object is unchanged and that the stream is invalid.  (C-4)
        //:
        //: 7 Write 3 distinct objects to an output stream buffer of total
        //:   length 'N'.  For each partial stream length from 0 to 'N - 1',
        //:   construct an input stream and attempt to read into objects
        //:   initialized with distinct values.  Verify values of objects
        //:   that are either successfully modified or left entirely
        //:   unmodified, and that the stream became invalid immediately after
        //:   the first incomplete read.  Finally, ensure that each object
        //:   streamed into is in some valid state.
        //:
        //: 8 Use the underlying stream package to simulate a typical valid
        //:   (control) stream and verify that it can be streamed in
        //:   successfully.  Then for each data field in the stream (beginning
        //:   with the version number), provide one or more similar tests with
        //:   that data field corrupted.  After each test, verify that the
        //:   object is in some valid state after streaming, and that the
        //:   input stream has become invalid.  (C-5)
        //:
        //: 9 Explicitly test the wire format.  (C-6)
        //:
        //:10 In all cases, confirm exception neutrality using the specially
        //:   instrumented 'bslx::TestInStream' and a pair of standard macros,
        //:   'BSLX_TESTINSTREAM_EXCEPTION_TEST_BEGIN' and
        //:   'BSLX_TESTINSTREAM_EXCEPTION_TEST_END', which configure the
        //:   'bslx::TestInStream' object appropriately in a loop.  (C-7)
        //:
        //:11 In all cases, verify the return value of the tested method.
        //:   (C-8)
        //
        // Testing:
        //   static STREAM& bdexStreamIn(STREAM& s, Enum& variable, int v);
        //   static STREAM& bdexStreamOut(STREAM& s, Enum value, int v) const;
        //   static int maxSupportedBdexVersion(int versionSelector);
        //   STREAM& bdexStreamIn(STREAM& s, Enum& variable, int v);
        //   STREAM& bdexStreamOut(STREAM& s, const Enum& value, int v) const;
        //   int maxSupportedBdexVersion(const Enum *, int versionSelector);
#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation
        //   static int maxSupportedBdexVersion();
        //   STREAM& streamIn(STREAM& s, Enum& variable, int v);
        //   STREAM& streamOut(STREAM& s, const Enum& value, int v) const;
        //   int maxSupportedVersion(Enum);
#endif // BDE_OPENSOURCE_PUBLICATION -- pending deprecation
        // --------------------------------------------------------------------

        // Allocator to use instead of the default allocator.
        bslma::TestAllocator allocator("bslx", veryVeryVeryVerbose);

        if (verbose) cout << endl
                          << "TESTING BDEX STREAMING" << endl
                          << "======================" << endl;

        // Scalar object values used in various stream tests.
        const Enum VA(Obj::e_SUN);
        const Enum VB(Obj::e_MON);
        const Enum VC(Obj::e_TUE);
        const Enum VD(Obj::e_WED);
        const Enum VE(Obj::e_THU);
        const Enum VF(Obj::e_FRI);
        const Enum VG(Obj::e_SAT);

        // Array object used in various stream tests.
        const Enum VALUES[]   = { VA, VB, VC, VD, VE, VF, VG };
        const int NUM_VALUES = static_cast<int>(sizeof VALUES
                                                / sizeof *VALUES);

        if (verbose) {
            cout << "\nTesting 'maxSupportedBdexVersion'." << endl;
        }
        {
            ASSERT(1 == Obj::maxSupportedBdexVersion(0));
            ASSERT(1 == Obj::maxSupportedBdexVersion(VERSION_SELECTOR));

            using bslx::VersionFunctions::maxSupportedBdexVersion;

            ASSERT(1 == maxSupportedBdexVersion(reinterpret_cast<Enum *>(0),
                                                0));
            ASSERT(1 == maxSupportedBdexVersion(reinterpret_cast<Enum *>(0),
                                                VERSION_SELECTOR));
        }

        const int VERSION = Obj::maxSupportedBdexVersion(0);

        if (verbose) {
            cout << "\nDirect initial trial of 'bdexStreamOut' and (valid) "
                 << "'bdexStreamIn' functionality." << endl;
        }
        {
            const Enum X(VC);
            Out        out(VERSION_SELECTOR, &allocator);

            Out& rvOut = Obj::bdexStreamOut(out, X, VERSION);
            ASSERT(&out == &rvOut);
            ASSERT(out);

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            In in(OD, LOD);
            ASSERT(in);
            ASSERT(!in.isEmpty());

            Enum mT(VA);  const Enum& T = mT;
            ASSERT(X != T);

            In& rvIn = Obj::bdexStreamIn(in, mT, VERSION);
            ASSERT(&in == &rvIn);
            ASSERT(X == T);
            ASSERT(in);
            ASSERT(in.isEmpty());
        }

        // We will use the stream free functions provided by 'bslx', as opposed
        // to the class member functions, since the 'bslx' implementation gives
        // priority to the free function implementations; we want to test what
        // will be used.  Furthermore, toward making this test case more
        // reusable in other components, from here on we generally use the
        // 'bdexStreamIn' and 'bdexStreamOut' free functions that are defined
        // in the 'bslx' package rather than call the like-named member
        // functions directly.

        if (verbose) {
            cout << "\nThorough test using stream free functions."
                 << endl;
        }
        {
            for (int i = 0; i < NUM_VALUES; ++i) {
                const Enum X(VALUES[i]);

                Out out(VERSION_SELECTOR, &allocator);

                using bslx::OutStreamFunctions::bdexStreamOut;
                using bslx::InStreamFunctions::bdexStreamIn;

                Out& rvOut = bdexStreamOut(out, X, VERSION);
                LOOP_ASSERT(i, &out == &rvOut);
                LOOP_ASSERT(i, out);
                const char *const OD  = out.data();
                const int         LOD = static_cast<int>(out.length());

                // Verify that each new value overwrites every old value and
                // that the input stream is emptied, but remains valid.

                for (int j = 0; j < NUM_VALUES; ++j) {
                    In in(OD, LOD);
                    LOOP2_ASSERT(i, j, in);
                    LOOP2_ASSERT(i, j, !in.isEmpty());

                    Enum mT(VALUES[j]);  const Enum& T = mT;
                    LOOP2_ASSERT(i, j, (X == T) == (i == j));

                    BSLX_TESTINSTREAM_EXCEPTION_TEST_BEGIN(in) {
                        in.reset();
                        In& rvIn = bdexStreamIn(in, mT, VERSION);
                        LOOP2_ASSERT(i, j, &in == &rvIn);
                    } BSLX_TESTINSTREAM_EXCEPTION_TEST_END

                    LOOP2_ASSERT(i, j, X == T);
                    LOOP2_ASSERT(i, j, in);
                    LOOP2_ASSERT(i, j, in.isEmpty());
                }
            }
        }

        if (verbose) {
            cout << "\tOn empty streams and non-empty, invalid streams."
                 << endl;
        }

        // Verify correct behavior for empty streams (valid and invalid).

        {
            Out               out(VERSION_SELECTOR, &allocator);
            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());
            ASSERT(0 == LOD);

            for (int i = 0; i < NUM_VALUES; ++i) {
                In in(OD, LOD);
                LOOP_ASSERT(i, in);
                LOOP_ASSERT(i, in.isEmpty());

                // Ensure that reading from an empty or invalid input stream
                // leaves the stream invalid and the target object unchanged.

                using bslx::InStreamFunctions::bdexStreamIn;

                const Enum  X(VALUES[i]);
                Enum        mT(X);
                const Enum& T = mT;
                LOOP_ASSERT(i, X == T);

                BSLX_TESTINSTREAM_EXCEPTION_TEST_BEGIN(in) {
                    in.reset();

                    // Stream is valid.
                    In& rvIn1 = bdexStreamIn(in, mT, VERSION);
                    LOOP_ASSERT(i, &in == &rvIn1);
                    LOOP_ASSERT(i, !in);
                    LOOP_ASSERT(i, X == T);

                    // Stream is invalid.
                    In& rvIn2 = bdexStreamIn(in, mT, VERSION);
                    LOOP_ASSERT(i, &in == &rvIn2);
                    LOOP_ASSERT(i, !in);
                    LOOP_ASSERT(i, X == T);
                } BSLX_TESTINSTREAM_EXCEPTION_TEST_END
            }
        }

        // Verify correct behavior for non-empty, invalid streams.

        {
            Out               out(VERSION_SELECTOR, &allocator);

            using bslx::OutStreamFunctions::bdexStreamOut;
            Out& rvOut = bdexStreamOut(out, Obj::e_SUN, VERSION);
            ASSERT(&out == &rvOut);
            ASSERT(out);

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());
            ASSERT(0 < LOD);

            for (int i = 0; i < NUM_VALUES; ++i) {
                In in(OD, LOD);
                in.invalidate();
                LOOP_ASSERT(i, !in);
                LOOP_ASSERT(i, !in.isEmpty());

                // Ensure that reading from a non-empty, invalid input stream
                // leaves the stream invalid and the target object unchanged.

                using bslx::InStreamFunctions::bdexStreamIn;

                const Enum  X(VALUES[i]);
                Enum        mT(X);
                const Enum& T = mT;
                LOOP_ASSERT(i, X == T);

                BSLX_TESTINSTREAM_EXCEPTION_TEST_BEGIN(in) {
                    in.reset();
                    in.invalidate();
                    LOOP_ASSERT(i, !in);
                    LOOP_ASSERT(i, !in.isEmpty());

                    In& rvIn = bdexStreamIn(in, mT, VERSION);
                    LOOP_ASSERT(i, &in == &rvIn);
                    LOOP_ASSERT(i, !in);
                    LOOP_ASSERT(i, X == T);
                } BSLX_TESTINSTREAM_EXCEPTION_TEST_END
            }
        }

        if (verbose) {
            cout << "\tOn incomplete (but otherwise valid) data."
                 << endl;
        }
        {
            const Enum W1 = VA, X1 = VB;
            const Enum W2 = VB, X2 = VC;
            const Enum W3 = VC, X3 = VD;

            using bslx::OutStreamFunctions::bdexStreamOut;
            using bslx::InStreamFunctions::bdexStreamIn;

            Out out(VERSION_SELECTOR, &allocator);

            Out& rvOut1 = bdexStreamOut(out, X1, VERSION);
            ASSERT(&out == &rvOut1);
            ASSERT(out);
            const int         LOD1 = static_cast<int>(out.length());

            Out& rvOut2 = bdexStreamOut(out, X2, VERSION);
            ASSERT(&out == &rvOut2);
            ASSERT(out);
            const int         LOD2 = static_cast<int>(out.length());

            Out& rvOut3 = bdexStreamOut(out, X3, VERSION);
            ASSERT(&out == &rvOut3);
            ASSERT(out);
            const int         LOD3 = static_cast<int>(out.length());
            const char *const OD3  = out.data();

            for (int i = 0; i < LOD3; ++i) {
                In in(OD3, i);

                BSLX_TESTINSTREAM_EXCEPTION_TEST_BEGIN(in) {
                    in.reset();
                    LOOP_ASSERT(i, in);
                    LOOP_ASSERT(i, !i == in.isEmpty());

                    Enum mT1(W1);  const Enum& T1 = mT1;
                    Enum mT2(W2);  const Enum& T2 = mT2;
                    Enum mT3(W3);  const Enum& T3 = mT3;

                    if (i < LOD1) {
                        In& rvIn1 = bdexStreamIn(in, mT1, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn1);
                        LOOP_ASSERT(i, !in);
                        if (0 == i) LOOP_ASSERT(i, W1 == T1);
                        In& rvIn2 = bdexStreamIn(in, mT2, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn2);
                        LOOP_ASSERT(i, !in);
                        LOOP_ASSERT(i, W2 == T2);
                        In& rvIn3 = bdexStreamIn(in, mT3, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn3);
                        LOOP_ASSERT(i, !in);
                        LOOP_ASSERT(i, W3 == T3);
                    }
                    else if (i < LOD2) {
                        In& rvIn1 = bdexStreamIn(in, mT1, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn1);
                        LOOP_ASSERT(i,  in);
                        LOOP_ASSERT(i, X1 == T1);
                        In& rvIn2 = bdexStreamIn(in, mT2, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn2);
                        LOOP_ASSERT(i, !in);
                        if (LOD1 <= i) LOOP_ASSERT(i, W2 == T2);
                        In& rvIn3 = bdexStreamIn(in, mT3, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn3);
                        LOOP_ASSERT(i, !in);
                        LOOP_ASSERT(i, W3 == T3);
                    }
                    else {  // 'LOD2 <= i < LOD3'
                        In& rvIn1 = bdexStreamIn(in, mT1, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn1);
                        LOOP_ASSERT(i,  in);
                        LOOP_ASSERT(i, X1 == T1);
                        In& rvIn2 = bdexStreamIn(in, mT2, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn2);
                        LOOP_ASSERT(i,  in);
                        LOOP_ASSERT(i, X2 == T2);
                        In& rvIn3 = bdexStreamIn(in, mT3, VERSION);
                        LOOP_ASSERT(i, &in == &rvIn3);
                        LOOP_ASSERT(i, !in);
                        if (LOD2 <= i) LOOP_ASSERT(i, W3 == T3);
                    }

                    // Verify the objects are in a valid state.

                    LOOP_ASSERT(i, Obj::e_SUN <= T1 && Obj::e_SAT >= T1);
                    LOOP_ASSERT(i, Obj::e_SUN <= T2 && Obj::e_SAT >= T2);
                    LOOP_ASSERT(i, Obj::e_SUN <= T3 && Obj::e_SAT >= T3);

                } BSLX_TESTINSTREAM_EXCEPTION_TEST_END
            }
        }

        if (verbose) {
            cout << "\tOn corrupted data." << endl;
        }

        const Enum W(Obj::e_SUN);       // default value
        const Enum X(Obj::e_MON);       // original (control)
        const Enum Y(Obj::e_TUE);       // new (streamed-out)

        // Verify the three objects are distinct.
        ASSERT(W != X);
        ASSERT(W != Y);
        ASSERT(X != Y);

        const int SERIAL_Y = 3;       // internal rep. of 'Y'

        if (verbose) {
            cout << "\t\tGood stream (for control)." << endl;
        }
        {
            Out out(VERSION_SELECTOR, &allocator);
            out.putInt8(SERIAL_Y);  // Stream out "new" value.
            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            Enum mT(X);  const Enum& T = mT;
            ASSERT(X == T);

            In in(OD, LOD);
            ASSERT(in);

            using bslx::InStreamFunctions::bdexStreamIn;

            In& rvIn = bdexStreamIn(in, mT, VERSION);
            ASSERT(&in == &rvIn);
            ASSERT(in);
            ASSERT(Y == T);
        }

        if (verbose) {
            cout << "\t\tBad version." << endl;
        }
        {
            const char version = 0; // too small ('version' must be >= 1)

            Out out(VERSION_SELECTOR, &allocator);
            out.putInt8(SERIAL_Y);  // Stream out "new" value.

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            Enum mT(X);  const Enum& T = mT;
            ASSERT(X == T);

            In in(OD, LOD);
            ASSERT(in);
            in.setQuiet(!veryVerbose);

            using bslx::InStreamFunctions::bdexStreamIn;

            In& rvIn = bdexStreamIn(in, mT, version);
            ASSERT(&in == &rvIn);
            ASSERT(!in);
            ASSERT(X == T);
        }
        {
            const char version = 2 ; // too large (current version is 1)

            Out out(VERSION_SELECTOR, &allocator);
            out.putInt8(SERIAL_Y);  // Stream out "new" value.

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            Enum mT(X);  const Enum& T = mT;
            ASSERT(X == T);

            In in(OD, LOD);
            ASSERT(in);
            in.setQuiet(!veryVerbose);

            using bslx::InStreamFunctions::bdexStreamIn;

            In& rvIn = bdexStreamIn(in, mT, version);
            ASSERT(&in == &rvIn);
            ASSERT(!in);
            ASSERT(X == T);
        }

        if (verbose) {
            cout << "\t\tValue too small." << endl;
        }
        {
            Out out(VERSION_SELECTOR, &allocator);

            // Stream out "new" value.
            out.putInt8(static_cast<char>(Obj::e_SUN - 1));

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            Enum mT(X);  const Enum& T = mT;
            ASSERT(X == T);

            In in(OD, LOD);
            ASSERT(in);
            in.setQuiet(!veryVerbose);

            using bslx::InStreamFunctions::bdexStreamIn;

            In& rvIn = bdexStreamIn(in, mT, VERSION);
            ASSERT(&in == &rvIn);
            ASSERT(!in);
            ASSERT(X == T);
        }

        if (verbose) {
            cout << "\t\tValue too large." << endl;
        }
        {
            Out out(VERSION_SELECTOR, &allocator);

            // Stream out "new" value.
            out.putInt8(static_cast<char>(Obj::e_SAT + 1));

            const char *const OD  = out.data();
            const int         LOD = static_cast<int>(out.length());

            Enum mT(X);  const Enum& T = mT;
            ASSERT(X == T);

            In in(OD, LOD);
            ASSERT(in);
            in.setQuiet(!veryVerbose);

            using bslx::InStreamFunctions::bdexStreamIn;

            In& rvIn = bdexStreamIn(in, mT, VERSION);
            ASSERT(&in == &rvIn);
            ASSERT(!in);
            ASSERT(X == T);
        }

        if (verbose) {
             cout << "\nWire format direct tests." << endl;
        }
        {
            static const struct {
                int          d_lineNum;      // source line number
                Enum         d_value;        // specification value
                int          d_version;      // version to stream with
                bsl::size_t  d_length;       // expect output length
                const char  *d_fmt_p;        // expected output format
            } DATA[] = {
                //LINE  VALUE       VER  LEN  FORMAT
                //----  ----------  ---  ---  -------
                { L_,   Obj::e_SUN,   1,   1,  "\x01"  },
                { L_,   Obj::e_MON,   1,   1,  "\x02"  },
                { L_,   Obj::e_TUE,   1,   1,  "\x03"  },
                { L_,   Obj::e_WED,   1,   1,  "\x04"  },
                { L_,   Obj::e_THU,   1,   1,  "\x05"  },
                { L_,   Obj::e_FRI,   1,   1,  "\x06"  },
                { L_,   Obj::e_SAT,   1,   1,  "\x07"  }
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int i = 0; i < NUM_DATA; ++i) {
                const int         LINE        = DATA[i].d_lineNum;
                const Enum        VALUE       = DATA[i].d_value;
                const int         VERSION     = DATA[i].d_version;
                const bsl::size_t LEN         = DATA[i].d_length;
                const char *const FMT         = DATA[i].d_fmt_p;

                // Test using class methods.

                {
                    const Enum X(VALUE);

                    bslx::ByteOutStream  out(VERSION_SELECTOR, &allocator);
                    bslx::ByteOutStream& rvOut = Obj::bdexStreamOut(out,
                                                                    X,
                                                                    VERSION);
                    LOOP_ASSERT(LINE, &out == &rvOut);
                    LOOP_ASSERT(LINE, out);

                    LOOP_ASSERT(LINE, LEN == out.length());
                    LOOP_ASSERT(LINE, 0 == memcmp(out.data(), FMT, LEN));

                    if (verbose && memcmp(out.data(), FMT, LEN)) {
                        const char *hex = "0123456789abcdef";
                        P_(LINE);
                        for (bsl::size_t j = 0; j < out.length(); ++j) {
                            cout << "\\x"
                                 << hex[static_cast<unsigned char>
                                            ((*(out.data() + j) >> 4) & 0x0f)]
                                 << hex[static_cast<unsigned char>
                                                   (*(out.data() + j) & 0x0f)];
                        }
                        cout << endl;
                    }

                    Enum mY;  const Enum& Y = mY;

                    bslx::ByteInStream  in(out.data(), out.length());
                    bslx::ByteInStream& rvIn = Obj::bdexStreamIn(in,
                                                                 mY,
                                                                 VERSION);
                    LOOP_ASSERT(LINE, &in == &rvIn);
                    LOOP_ASSERT(LINE, X == Y);
                }

                // Test using free functions.

                {
                    const Enum X(VALUE);

                    using bslx::OutStreamFunctions::bdexStreamOut;

                    bslx::ByteOutStream  out(VERSION_SELECTOR, &allocator);
                    bslx::ByteOutStream& rvOut = bdexStreamOut(out,
                                                               X,
                                                               VERSION);
                    LOOP_ASSERT(LINE, &out == &rvOut);
                    LOOP_ASSERT(LINE, out);

                    LOOP_ASSERT(LINE, LEN == out.length());
                    LOOP_ASSERT(LINE, 0 == memcmp(out.data(), FMT, LEN));

                    if (verbose && memcmp(out.data(), FMT, LEN)) {
                        const char *hex = "0123456789abcdef";
                        P_(LINE);
                        for (bsl::size_t j = 0; j < out.length(); ++j) {
                            cout << "\\x"
                                 << hex[static_cast<unsigned char>
                                            ((*(out.data() + j) >> 4) & 0x0f)]
                                 << hex[static_cast<unsigned char>
                                                   (*(out.data() + j) & 0x0f)];
                        }
                        cout << endl;
                    }

                    Enum mY;  const Enum& Y = mY;

                    using bslx::InStreamFunctions::bdexStreamIn;

                    bslx::ByteInStream  in(out.data(), out.length());
                    bslx::ByteInStream& rvIn = bdexStreamIn(in, mY, VERSION);
                    LOOP_ASSERT(LINE, &in == &rvIn);
                    LOOP_ASSERT(LINE, X == Y);
                }
            }
        }

#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation

        if (verbose) {
            cout << "\nTesting deprecated methods." << endl;
        }
        {
            ASSERT(Obj::maxSupportedBdexVersion()
                                           == Obj::maxSupportedBdexVersion(0));

            using bdex_VersionFunctions::maxSupportedVersion;

            ASSERT(1 == maxSupportedVersion(Obj::e_SUN));

            static const struct {
                int          d_lineNum;      // source line number
                Enum         d_value;        // specification value
                int          d_version;      // version to stream with
                bsl::size_t  d_length;       // expect output length
                const char  *d_fmt_p;        // expected output format
            } DATA[] = {
                //LINE  VALUE       VER  LEN  FORMAT
                //----  ----------  ---  ---  -------
                { L_,   Obj::e_SUN,   1,   1,  "\x01"  },
                { L_,   Obj::e_MON,   1,   1,  "\x02"  },
                { L_,   Obj::e_TUE,   1,   1,  "\x03"  },
                { L_,   Obj::e_WED,   1,   1,  "\x04"  },
                { L_,   Obj::e_THU,   1,   1,  "\x05"  },
                { L_,   Obj::e_FRI,   1,   1,  "\x06"  },
                { L_,   Obj::e_SAT,   1,   1,  "\x07"  }
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int i = 0; i < NUM_DATA; ++i) {
                const int         LINE        = DATA[i].d_lineNum;
                const Enum        VALUE       = DATA[i].d_value;
                const int         VERSION     = DATA[i].d_version;
                const bsl::size_t LEN         = DATA[i].d_length;
                const char *const FMT         = DATA[i].d_fmt_p;

                {
                    const Enum X(VALUE);

                    using bdex_OutStreamFunctions::streamOut;

                    bslx::ByteOutStream  out(VERSION_SELECTOR, &allocator);
                    bslx::ByteOutStream& rvOut = streamOut(out, X, VERSION);
                    LOOP_ASSERT(LINE, &out == &rvOut);
                    LOOP_ASSERT(LINE, out);

                    LOOP_ASSERT(LINE, LEN == out.length());
                    LOOP_ASSERT(LINE, 0 == memcmp(out.data(), FMT, LEN));

                    if (verbose && memcmp(out.data(), FMT, LEN)) {
                        const char *hex = "0123456789abcdef";
                        P_(LINE);
                        for (bsl::size_t j = 0; j < out.length(); ++j) {
                            cout << "\\x"
                                 << hex[static_cast<unsigned char>
                                            ((*(out.data() + j) >> 4) & 0x0f)]
                                 << hex[static_cast<unsigned char>
                                                   (*(out.data() + j) & 0x0f)];
                        }
                        cout << endl;
                    }

                    Enum mY;  const Enum& Y = mY;

                    using bdex_InStreamFunctions::streamIn;

                    bslx::ByteInStream  in(out.data(), out.length());
                    bslx::ByteInStream& rvIn = streamIn(in, mY, VERSION);
                    LOOP_ASSERT(LINE, &in == &rvIn);
                    LOOP_ASSERT(LINE, X == Y);
                }
            }
        }

#endif // BDE_OPENSOURCE_PUBLICATION -- pending deprecation
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING 'operator<<'
        //
        // Concerns:
        //: 1 The '<<' operator writes the output to the specified stream.
        //
        //: 2 The '<<' operator writes the string representation of each
        //:   enumerator in the intended format.
        //
        //: 3 The '<<' operator writes a distinguished string when passed an
        //:   out-of-band value.
        //
        //: 4 The output produced by 'stream << value' is the same as that
        //:   produced by 'Obj::print(stream, value, 0, -1)'.
        //
        //: 5 There is no output when the stream is invalid.
        //
        //: 6 The '<<' operator has the expected signature.
        //
        //: 7 The '<<' operator returns the stream passed to it.
        //
        // Plan:
        //: 1 Verify that the '<<' operator produces the expected results for
        //:   each enumerator and returns a reference to the stream passed to
        //:   it.  (C-1 .. 2, 7)
        //
        //: 2 Verify that the '<<' operator writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //
        //: 3 Verify that 'stream << value' writes the same output as
        //:   'Obj::print(stream, value, 0, -1)'.  (C-4)
        //
        //: 4 Verify that there is no output when the stream is invalid.  (C-5)
        //
        //: 5 Take the address of the '<<' (free) operator and use the result
        //:   to initialize a variable of the appropriate type.  (C-6)
        //
        // Testing:
        //   operator<<(ostream&, DayOfWeek::Enum);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'operator<<'" << endl
                          << "====================" << endl;

#define UNKNOWN_FORMAT "(* Unknown Enumerator *)"

        static const struct {
            int         d_lineNum;  // source line number
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
            //line       enumerator value         expected result
            //----    -----------------------     ---------------
            { L_,     Obj::e_SUN,                 "SUN"              },
            { L_,     Obj::e_MON,                 "MON"              },
            { L_,     Obj::e_TUE,                 "TUE"              },
            { L_,     Obj::e_WED,                 "WED"              },
            { L_,     Obj::e_THU,                 "THU"              },
            { L_,     Obj::e_FRI,                 "FRI"              },
            { L_,     Obj::e_SAT,                 "SAT"              },

            { L_,     Obj::e_SUNDAY,              "SUN"              },
            { L_,     Obj::e_MONDAY,              "MON"              },
            { L_,     Obj::e_TUESDAY,             "TUE"              },
            { L_,     Obj::e_WEDNESDAY,           "WED"              },
            { L_,     Obj::e_THURSDAY,            "THU"              },
            { L_,     Obj::e_FRIDAY,              "FRI"              },
            { L_,     Obj::e_SATURDAY,            "SAT"              },

#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation
            { L_,     Obj::BDET_SUN,              "SUN"              },
            { L_,     Obj::BDET_MON,              "MON"              },
            { L_,     Obj::BDET_TUE,              "TUE"              },
            { L_,     Obj::BDET_WED,              "WED"              },
            { L_,     Obj::BDET_THU,              "THU"              },
            { L_,     Obj::BDET_FRI,              "FRI"              },
            { L_,     Obj::BDET_SAT,              "SAT"              },

            { L_,     Obj::BDET_SUNDAY,           "SUN"              },
            { L_,     Obj::BDET_MONDAY,           "MON"              },
            { L_,     Obj::BDET_TUESDAY,          "TUE"              },
            { L_,     Obj::BDET_WEDNESDAY,        "WED"              },
            { L_,     Obj::BDET_THURSDAY,         "THU"              },
            { L_,     Obj::BDET_FRIDAY,           "FRI"              },
            { L_,     Obj::BDET_SATURDAY,         "SAT"              },
#endif

            { L_,     ABOVE_ENUM_RANGE,           UNKNOWN_FORMAT     },
            { L_,     -1,                         UNKNOWN_FORMAT     },
            { L_,     -5,                         UNKNOWN_FORMAT     },
            { L_,     99,                         UNKNOWN_FORMAT     },
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        if (verbose) cout << "\nTesting '<<' operator." << endl;

        bslma::TestAllocator tmpDfltAlloc;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            ostringstream oss(&tmpDfltAlloc);
            ostream *po = &(oss << VALUE);
            ASSERT(&oss == po);

            bslma::DefaultAllocatorGuard allocatorGuard(&tmpDfltAlloc);

            const bsl::string& str = oss.str();

            if (veryVerbose) cout << "  ACTUAL FORMAT: " << str << endl;

            ASSERTV(LINE, ti, EXP == str);
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            ostringstream oss(&tmpDfltAlloc);
            oss.setstate(ios::badbit);
            oss << VALUE;

            bslma::DefaultAllocatorGuard allocatorGuard(&tmpDfltAlloc);

            ASSERTV(LINE, ti, "" == oss.str());
        }

        if (verbose) cout << "\nVerify '<<' operator signature." << endl;
        {
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum);
            using namespace bdlt;

            const FuncPtr FP = &operator<<;
            (void) FP;
        }

#undef UNKNOWN_FORMAT
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING 'print'
        //
        // Concerns:
        //: 1 The 'print' method writes the output to the specified stream.
        //
        //: 2 The 'print' method writes the string representation of each
        //:   enumerator in the intended format.
        //
        //: 3 The 'print' method writes a distinguished string when passed an
        //:   out-of-band value.
        //
        //: 4 There is no output when the stream is invalid.
        //
        //: 5 The 'print' method has the expected signature.
        //
        //: 6 That 'print' returns a reference to the 'ostream' passed to it.
        //
        // Plan:
        //: 1 Verify that the 'print' method produces the expected results for
        //:   each enumerator, and that 'print' returns a reference to the
        //:   'ostream' passed to it.  (C-1 .. 2, 7)
        //
        //: 2 Verify that the 'print' method writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //
        //: 3 Verify that there is no output when the stream is invalid.  (C-4)
        //
        //: 4 Take the address of the 'print' (class) method and use the
        //:   result to initialize a variable of the appropriate type.  (C-5)
        //
        // Testing:
        //   ostream& print(ostream&, DayOfWeek::Enum, int, int)
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'print'" << endl
                          << "===============" << endl;

#define UNKNOWN_FORMAT "(* Unknown Enumerator *)"

        static const struct {
            int         d_lineNum;  // source line number
            int         d_level;    // level
            int         d_spl;      // spaces per level
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
#define NL "\n"
            //line  level  spl    enumerator value        expected result
            //----  -----  ---  ----------------------   ------------------
            { L_,     0,    4,  Obj::e_SUN,              "SUN" NL           },
            { L_,     0,    4,  Obj::e_MON,              "MON" NL           },
            { L_,     0,    4,  Obj::e_TUE,              "TUE" NL           },
            { L_,     0,    4,  Obj::e_WED,              "WED" NL           },
            { L_,     0,    4,  Obj::e_THU,              "THU" NL           },
            { L_,     0,    4,  Obj::e_FRI,              "FRI" NL           },
            { L_,     0,    4,  Obj::e_SAT,              "SAT" NL           },

            { L_,     0,    4,  Obj::e_SUNDAY,           "SUN" NL           },
            { L_,     0,    4,  Obj::e_MONDAY,           "MON" NL           },
            { L_,     0,    4,  Obj::e_TUESDAY,          "TUE" NL           },
            { L_,     0,    4,  Obj::e_WEDNESDAY,        "WED" NL           },
            { L_,     0,    4,  Obj::e_THURSDAY,         "THU" NL           },
            { L_,     0,    4,  Obj::e_FRIDAY,           "FRI" NL           },
            { L_,     0,    4,  Obj::e_SATURDAY,         "SAT" NL           },

#ifndef BDE_OPENSOURCE_PUBLICATION  // pending deprecation
            { L_,     0,    4,  Obj::BDET_SUN,           "SUN" NL           },
            { L_,     0,    4,  Obj::BDET_MON,           "MON" NL           },
            { L_,     0,    4,  Obj::BDET_TUE,           "TUE" NL           },
            { L_,     0,    4,  Obj::BDET_WED,           "WED" NL           },
            { L_,     0,    4,  Obj::BDET_THU,           "THU" NL           },
            { L_,     0,    4,  Obj::BDET_FRI,           "FRI" NL           },
            { L_,     0,    4,  Obj::BDET_SAT,           "SAT" NL           },
#endif

#if !defined(BSLS_ASSERT_SAFE_IS_ACTIVE)
            { L_,     0,    4,  ABOVE_ENUM_RANGE,        UNKNOWN_FORMAT NL  },
            { L_,     0,    4,  -1,                      UNKNOWN_FORMAT NL  },
            { L_,     0,    4,  -5,                      UNKNOWN_FORMAT NL  },
            { L_,     0,    4,  99,                      UNKNOWN_FORMAT NL  },
#endif

            { L_,     0,   -1,  Obj::e_WED,              "WED"              },
            { L_,     0,    0,  Obj::e_WED,              "WED" NL           },
            { L_,     0,    2,  Obj::e_WED,              "WED" NL           },
            { L_,     1,    1,  Obj::e_WED,              " WED" NL          },

            { L_,     1,    2,  Obj::e_SUN,              "  SUN" NL         },
            { L_,     1,    2,  Obj::e_MON,              "  MON" NL         },
            { L_,     1,    2,  Obj::e_TUE,              "  TUE" NL         },
            { L_,     1,    2,  Obj::e_WED,              "  WED" NL         },
            { L_,     1,    2,  Obj::e_THU,              "  THU" NL         },
            { L_,     1,    2,  Obj::e_FRI,              "  FRI" NL         },
            { L_,     1,    2,  Obj::e_SAT,              "  SAT" NL         },

            { L_,    -1,    2,  Obj::e_WED,              "WED" NL           },
            { L_,    -2,    1,  Obj::e_WED,              "WED" NL           },
            { L_,     2,    1,  Obj::e_WED,              "  WED" NL         },
            { L_,     1,    3,  Obj::e_WED,              "   WED" NL        },
#undef NL
#undef UNKNOWN_FORMAT
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        if (verbose) cout << "\nTesting 'print'." << endl;

        bslma::TestAllocator tmpDfltAlloc;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            ostringstream oss(&tmpDfltAlloc);
            Obj::print(oss, VALUE, LEVEL, SPL);

            {
                bslma::DefaultAllocatorGuard allocatorGuard(&tmpDfltAlloc);

                const bsl::string& str = oss.str();

                if (veryVerbose) cout << "  ACTUAL FORMAT: " << str << endl;

                ASSERTV(LINE, EXP == str);
            }

            if (0 == LEVEL && 4 == SPL) {
                if (veryVerbose)
                    cout << "\tRepeat for 'print' default arguments." << endl;

                oss.str("");
                ostream *po = &Obj::print(oss, VALUE);
                ASSERT(&oss == po);

                bslma::DefaultAllocatorGuard allocatorGuard(&tmpDfltAlloc);

                const bsl::string& str2 = oss.str();

                if (veryVerbose) cout << "  ACTUAL FORMAT: " << str2;

                ASSERTV(LINE, EXP, str2, EXP == str2);
            }
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
//          const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            ostringstream oss(&tmpDfltAlloc);
            oss.setstate(ios::badbit);
            Obj::print(oss, VALUE, LEVEL, SPL);

            bslma::DefaultAllocatorGuard allocatorGuard(&tmpDfltAlloc);

            ASSERTV(LINE, ti, "" == oss.str());
        }

        if (verbose) cout << "\nVerify 'print' signature." << endl;

        {
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum, int, int);

            const FuncPtr FP = &Obj::print;
            (void) FP;
        }

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // TESTING 'Enum' AND 'toAscii'
        //
        // Concerns:
        //: 1 The enumerator values are sequential, starting from 1.  (P-1)
        //: 2 The 'toAscii' method returns the expected string representation
        //:   for each enumerator.  (P-2)
        //: 3 The 'toAscii' method returns a distinguished string when passed
        //:   an out-of-band value.  (P-3)
        //: 4 The string returned by 'toAscii' is non-modifiable.  (P-4)
        //: 5 The 'toAscii' method has the expected signature.  (P-4)
        //
        // Plan:
        //: 1 Verify that the enumerator values are sequential, starting from
        //:   1.  (C-1)
        //: 2 Verify that the 'toAscii' method returns the expected string
        //:   representation for each enumerator.  (C-2)
        //: 3 Verify that the 'toAscii' method returns a distinguished string
        //:   when passed an out-of-band value.  (C-3)
        //: 4 Take the address of the 'toAscii' (class) method and use the
        //:   result to initialize a variable of the appropriate type.
        //:   (C-4, C-5)
        //
        // Testing:
        //   enum Enum { ... };
        //   static const char *toAscii(Enum dayOfWeek);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING 'Enum' AND 'toAscii'" << endl
                          << "============================" << endl;

        static const struct {
            int         d_line;    // line number
            int         d_intDay;  // enumerated Value as int
            int         d_valid;   // is a valid enum valid
            const char *d_exp;     // expected String Rep.
        } DATA[] = {
            // line    Enumerated Value      Valid   Expected output
            // ----    ----------------      -----   ---------------
            {  L_,     Obj::e_SUNDAY,        1, "SUN"                      },
            {  L_,     Obj::e_SUN,           1, "SUN"                      },
            {  L_,     Obj::e_MONDAY,        1, "MON"                      },
            {  L_,     Obj::e_MON,           1, "MON"                      },
            {  L_,     Obj::e_TUESDAY,       1, "TUE"                      },
            {  L_,     Obj::e_TUE,           1, "TUE"                      },
            {  L_,     Obj::e_WEDNESDAY,     1, "WED"                      },
            {  L_,     Obj::e_WED,           1, "WED"                      },
            {  L_,     Obj::e_THURSDAY,      1, "THU"                      },
            {  L_,     Obj::e_THU,           1, "THU"                      },
            {  L_,     Obj::e_FRIDAY,        1, "FRI"                      },
            {  L_,     Obj::e_FRI,           1, "FRI"                      },
            {  L_,     Obj::e_SATURDAY,      1, "SAT"                      },
            {  L_,     Obj::e_SAT,           1, "SAT"                      },
#if !defined(BSLS_ASSERT_SAFE_IS_ACTIVE)
            {  L_,     0,                    0, "(* Unknown Enumerator *)" },
            {  L_,     Obj::e_SAT + 1,       0, "(* Unknown Enumerator *)" },
            {  L_,     19,                   0, "(* Unknown Enumerator *)" },
#endif
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        if (verbose) cout << "\nVerify enumerator values are sequential\n"
                          <<   "beginning at 1.\n";
        {
            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int  DAY   = DATA[ti].d_intDay;
                const bool VALID = DATA[ti].d_valid;

                if (veryVerbose) { P_(DAY); }

                if (VALID) {
                    ASSERT(DAY == ti / 2 + 1);
                }
            }
        }

        if (verbose) cout << "Testing 'toAscii'." << endl;
        {
            for (int i = 0 ; i < NUM_DATA; ++i) {
                const int   LINE = DATA[i].d_line;
                const Enum  DAY  = (Enum) DATA[i].d_intDay;
                const char *EXP  = DATA[i].d_exp;

                if (veryVerbose) { P_(DAY); P(EXP); }

                const char *res = Obj::toAscii(DAY);
                if (veryVerbose) { cout << '\t'; P_(i); P_(DAY); P(res); }
                ASSERTV(LINE, i, 0 == strcmp(EXP, res));
            }

            typedef const char *(*FuncPtr)(Enum);
            FuncPtr fp = &Obj::toAscii;
            (void) fp;
        }
      } break;
      default: {
          cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
          testStatus = -1;
      }
    }

    ASSERT(0 == globalAlloc.numBlocksTotal());
    ASSERT(0 == dfltAlloc.  numBlocksTotal());

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2014 Bloomberg Finance L.P.
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
