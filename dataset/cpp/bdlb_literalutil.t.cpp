// bdlb_literalutil.t.cpp                                             -*-C++-*-
#include <bdlb_literalutil.h>

#include <bsl_iostream.h>

#include <bslma_default.h>
#include <bslma_testallocator.h>            // to verify that we do not
#include <bslma_testallocatormonitor.h>     // allocate any memory
#include <bsls_asserttest.h>
#include <bsls_libraryfeatures.h>

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
#include <memory_resource>  // 'std::pmr:polymorphic_allocator'
#endif  // BSLS_LIBRARYFEATURES_HAS_CPP17_PMR

using namespace BloombergLP;
using namespace bsl;

// ============================================================================
//                                  TEST PLAN
// ----------------------------------------------------------------------------
//                                   Overview
//                                   --------
// TBD
//
// Primary Manipulators:
//: o TBD
//
// Basic Accessors:
//: o TBD
//
// Global Concerns:
//: o No memory is allocated.
//: o TBD
//
// Global Assumptions:
//: o ACCESSOR methods are 'const' thread-safe.
//: o TBD
// ----------------------------------------------------------------------------
// CLASS METHODS
// [  ] TBD
//
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [  ] USAGE EXAMPLE
// [ *] CONCERN: DOES NOT ALLOCATE MEMORY
// [  ] TEST APPARATUS: TBD
// [ 5] PRINTING: TBD

// ============================================================================
//                        STANDARD BDE ASSERT TEST MACRO
// ----------------------------------------------------------------------------

static int testStatus = 0;

static void aSsErT(int c, const char *s, int i)
{
    if (c) {
        cout << "Error " << __FILE__ << "(" << i << "): " << s
             << "    (failed)" << endl;
        if (0 <= testStatus && 100 >= testStatus) {
            ++testStatus;
        }
    }
}
# define ASSERT(expX) { aSsErT(!(expX), #expX, __LINE__); }

// ============================================================================
//                  STANDARD BDE LOOP-ASSERT TEST MACROS
// ----------------------------------------------------------------------------
#define LOOP_ASSERT(expI, expX) {                                             \
    if (!(expX)) { aSsErT(1, #expX, __LINE__);                                \
                cout << "\t"#expI": " << expI << "\n";}}

#define LOOP2_ASSERT(expI, expJ, expX) {                                      \
    if (!(expX)) { aSsErT(1, #expX, __LINE__);                                \
                   cout <<   "\t"#expI": " << expI <<                         \
                           "\n\t"#expJ": " << expJ << "\n";}}

#define LOOP3_ASSERT(expI, expJ, expK, expX) {                                \
    if (!(expX)) { aSsErT(1, #expX, __LINE__);                                \
                   cout <<   "\t"#expI": " << expI <<                         \
                           "\n\t"#expJ": " << expJ <<                         \
                           "\n\t"#expK": " << expK << "\n"; } }

#define LOOP4_ASSERT(expI, expJ, expK, expL, expX) {                          \
    if (!(expX)) { aSsErT(1, #expX, __LINE__);                                \
                   cout <<   "\t"#expI": " << expI <<                         \
                           "\n\t"#expJ": " << expJ <<                         \
                           "\n\t"#expK": " << expK <<                         \
                           "\n\t"#expL": " << expL << "\n"; } }

#define LOOP5_ASSERT(expI, expJ, expK, expL, expM, expX) {                    \
    if (!(expX)) { aSsErT(1, #expX, __LINE__);                                \
                   cout <<   "\t"#expI": " << expI <<                         \
                           "\n\t"#expJ": " << expJ <<                         \
                           "\n\t"#expK": " << expK <<                         \
                           "\n\t"#expL": " << expL <<                         \
                           "\n\t"#expM": " << expM << "\n"; } }

#define LOOP0_ASSERT ASSERT
#define LOOP1_ASSERT LOOP_ASSERT

// ============================================================================
//                   STANDARD BDE VARIADIC ASSERT TEST MACROS
// ----------------------------------------------------------------------------

#define NUM_ARGS_IMPL(X5, X4, X3, X2, X1, X0, N, ...)   N
#define NUM_ARGS(...) NUM_ARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1, 0, "")

#define LOOPN_ASSERT_IMPL(N, ...) LOOP ## N ## _ASSERT(__VA_ARGS__)
#define LOOPN_ASSERT(N, ...)      LOOPN_ASSERT_IMPL(N, __VA_ARGS__)

#define ASSERTV(...) LOOPN_ASSERT(NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

// ============================================================================
//                  SEMI-STANDARD TEST OUTPUT MACROS
// ----------------------------------------------------------------------------

#define P(expX) cout << #expX " = " << (expX) << endl;
    // Print expression and value.

#define Q(expX) cout << "<| " #expX " |>" << endl;
    // Quote expression literally.

#define P_(expX) cout << #expX " = " << (expX) << ", " << flush;
    // 'P(expX)' without '\n'

#define T_ cout << "\t" << flush;             // Print tab w/o newline.
#define L_ __LINE__                           // current Line number

// ============================================================================
//                      NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_FAIL(expr) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(expr)
#define ASSERT_SAFE_PASS(expr) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(expr)

// ============================================================================
//                                USEFUL MACROS
// ----------------------------------------------------------------------------

// The following macros may be used to print an expression 'X' at different
// levels of verbosity.  Note that 'X' is not surrounded with parentheses so
// that expressions containing output stream operations can be supported.

#define PV(X)   if         (verbose) cout << endl << X << endl;
#define PVV(X)  if     (veryVerbose) cout << endl << X << endl;
#define PVVV(X) if (veryVeryVerbose) cout << endl << X << endl;

// ============================================================================
//                    GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

// ============================================================================
//                                TEST APPARATUS
// ----------------------------------------------------------------------------

// ============================================================================
//                                 MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int                 test = argc > 1 ? atoi(argv[1]) : 0;
    bool             verbose = argc > 2;
    bool         veryVerbose = argc > 3;
    bool     veryVeryVerbose = argc > 4;
    bool veryVeryVeryVerbose = argc > 5;

    (void)veryVerbose;
    (void)veryVeryVerbose;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    // CONCERN: DOES NOT ALLOCATE MEMORY

    bslma::TestAllocator ga("global", veryVeryVeryVerbose);
    bslma::Default::setGlobalAllocator(&ga);

    bslma::TestAllocator da("default", veryVeryVeryVerbose);
    ASSERT(0 == bslma::Default::setDefaultAllocator(&da));

    bslma::TestAllocatorMonitor gam(&ga), dam(&da);

    switch (test) { case 0:
      case 1: {
        //---------------------------------------------------------------------
        // BREATHING TEST:
        //   This case exercises (but does not fully test) basic functionality.
        //
        // Concerns:
        //:
        //: 1 The class is sufficiently functional to enable comprehensive
        //:   testing in subsequent test cases
        //
        // Plan:
        //:
        //: 1 TBD
        //
        // Testing:
        //   BREATHING TEST
        //---------------------------------------------------------------------

        if (verbose) cout << endl << "BREATHING TEST" << endl
                                  << "==============" << endl;

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        bslma::TestAllocator ta("test", veryVeryVeryVerbose);

        static const char inArray[] =
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "~`@#$%^&*()_-+={[}]|\\;:'\"<>,.?/"
            "\0'\"\a\b\f\n\r\t\v\x7f\xe1";

        const bsl::string_view oracle(
            "\"0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "~`@#$%^&*()_-+={[}]|\\\\;:'\\\"<>,.?/"
            "\\000'\\\"\\a\\b\\f\\n\\r\\t\\v\\177\\341\"");

        bsl::string      bslResult("garbage", &ta);
        std::string      stdResult("garbage");
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        std::pmr::string pmrResult("garbage");
#endif

        // 'bsl::string_view' input
        {
            const bsl::string_view  inVue(inArray, sizeof(inArray) - 1);

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inVue);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inVue);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inVue);
#endif

            ASSERTV(bslResult, oracle, bslResult == oracle);
            ASSERTV(stdResult, oracle, stdResult == oracle);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            ASSERTV(pmrResult, oracle, pmrResult == oracle);
#endif
        }

        // 'bslstl::StringRef' input
        {
            const bslstl::StringRef inRef(inArray, sizeof(inArray) - 1);

            bslResult = "garbage";
            stdResult = "garbage";
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            pmrResult = "garbage";
#endif

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inRef);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inRef);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inRef);
#endif

            ASSERTV(bslResult, oracle, bslResult == oracle);
            ASSERTV(stdResult, oracle, stdResult == oracle);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            ASSERTV(pmrResult, oracle, pmrResult == oracle);
#endif
        }

        // 'bsl::string' input
        {
            const bsl::string inBsl(inArray, sizeof(inArray) - 1, &ta);

            bslResult = "garbage";
            stdResult = "garbage";
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            pmrResult = "garbage";
#endif

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inBsl);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inBsl);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inBsl);
#endif

            ASSERTV(bslResult, oracle, bslResult == oracle);
            ASSERTV(stdResult, oracle, stdResult == oracle);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            ASSERTV(pmrResult, oracle, pmrResult == oracle);
#endif
        }

        // 'std::string' input
        {
            const std::string inStd(inArray, sizeof(inArray) - 1);

            bslResult = "garbage";
            stdResult = "garbage";
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            pmrResult = "garbage";
#endif

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inStd);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inStd);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inStd);
#endif

            ASSERTV(bslResult, oracle, bslResult == oracle);
            ASSERTV(stdResult, oracle, stdResult == oracle);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            ASSERTV(pmrResult, oracle, pmrResult == oracle);
#endif
        }

        // 'std::pmr::string' input
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        {
            const std::pmr::string inPmr(inArray, sizeof(inArray) - 1);

            bslResult = "garbage";
            stdResult = "garbage";
            pmrResult = "garbage";

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inPmr);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inPmr);
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inPmr);

            ASSERTV(bslResult, oracle, bslResult == oracle);
            ASSERTV(stdResult, oracle, stdResult == oracle);
            ASSERTV(pmrResult, oracle, pmrResult == oracle);
        }
#endif

        // C string input
        {
            const char *inPtr = inArray;       // ends at the first '\0' char

            const bsl::string_view ptrOracle(  // so we need another oracle
                "\"0123456789"
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "~`@#$%^&*()_-+={[}]|\\\\;:'\\\"<>,.?/\"");

            bslResult = "garbage";
            stdResult = "garbage";
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            pmrResult = "garbage";
#endif

            bdlb::LiteralUtil::createQuotedEscapedCString(&bslResult, inPtr);
            bdlb::LiteralUtil::createQuotedEscapedCString(&stdResult, inPtr);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            bdlb::LiteralUtil::createQuotedEscapedCString(&pmrResult, inPtr);
#endif

            ASSERTV(bslResult, ptrOracle, bslResult == ptrOracle);
            ASSERTV(stdResult, ptrOracle, stdResult == ptrOracle);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            ASSERTV(pmrResult, ptrOracle, pmrResult == ptrOracle);
#endif
        }
      } break;
      default: {
        cerr << "WARNING: CASE '" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    // CONCERN: DOES NOT ALLOCATE MEMORY

    ASSERTV(gam.isTotalSame());
    ASSERTV(dam.isTotalSame());

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2020 Bloomberg Finance L.P.
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
