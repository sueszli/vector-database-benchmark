// bslstl_inplace.t.cpp                                               -*-C++-*-

#include <bslstl_inplace.h>

#include <bslmf_assert.h>
#include <bslmf_issame.h>

#include <bsls_bsltestutil.h>
#include <bsls_platform.h>

#include <stdio.h>
#include <stdlib.h>

using namespace BloombergLP;
using namespace bsl;

// ============================================================================
//                             TEST PLAN
// ----------------------------------------------------------------------------
//                             Overview
//                             --------
// The types under test are 'bsl::in_place_t', 'in_place_type_t', and
// 'in_place_index_t, trivial tag types whose interface and contract is
// dictated by the C++ standard.  For each type, if  'std::tag_name' is
// available, we need to check that 'bsl::tag_name' is a typedef to the
// standard's tag type.  If 'std::tag_name' isn't available, we need to check
// that 'bsl::tag_name' satisfies the interface and contract of
// 'std::tag_name'.
//
//
// ----------------------------------------------------------------------------
// TYPES:
// [ 2] bsl::in_place_t
// [ 2] bsl::in_place_type_t
// [ 2] bsl::in_place_index_t
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST

// ============================================================================
//                     STANDARD BSL ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        printf("Error " __FILE__ "(%d): %s    (failed)\n", line, message);

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace


// ============================================================================
//               STANDARD BSL TEST DRIVER MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT       BSLS_BSLTESTUTIL_ASSERT
#define ASSERTV      BSLS_BSLTESTUTIL_ASSERTV

#define LOOP_ASSERT  BSLS_BSLTESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLS_BSLTESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLS_BSLTESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLS_BSLTESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLS_BSLTESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLS_BSLTESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLS_BSLTESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLS_BSLTESTUTIL_LOOP6_ASSERT

#define Q            BSLS_BSLTESTUTIL_Q   // Quote identifier literally.
#define P            BSLS_BSLTESTUTIL_P   // Print identifier and value.
#define P_           BSLS_BSLTESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLS_BSLTESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLS_BSLTESTUTIL_L_  // current Line number

// ============================================================================
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)


//=============================================================================
//                         GLOBAL FUNCTIONS FOR TESTING
//-----------------------------------------------------------------------------

// Return 'true' if invoked with an object of type 'bsl::in_place_type_t' that
// matches the expected deduced type, and 'false' otherwise.
template <class EXPECTED, class TYPE>
bool isInplaceTypeTagType(const bsl::in_place_type_t<TYPE>&)
{
    return bsl::is_same<EXPECTED, TYPE>::value;
}
template <class TYPE>
BSLS_KEYWORD_CONSTEXPR bool isInplaceTypeTagType(const TYPE&)
{
    return false;
}

// Return 'true' if invoked with an object of type 'bsl::in_place_index_t' that
// matches the expected index, and 'false' otherwise.
template <size_t EXPECTED, size_t INDEX>
bool isInplaceIndexTagType(const bsl::in_place_index_t<INDEX>&)
{
    return EXPECTED == INDEX;
}
template <class TYPE>
bool isInplaceIndexTagType(const TYPE&)
{
    return false;
}

// Return 'true' if invoked with an object of type 'bsl::in_place_t', and
// 'false' otherwise.
bool isInplaceTagType(const bsl::in_place_t&)
{
    return true;
}
template <class TYPE>
bool isInplaceTagType(const TYPE&)
{
    return false;
}

//=============================================================================
//                             USAGE EXAMPLE
//-----------------------------------------------------------------------------

//=============================================================================
//                              MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int  test    = argc > 1 ? atoi(argv[1]) : 0;
    bool verbose = argc > 2;
    //  bool veryVerbose         = argc > 3;
    //  bool veryVeryVerbose     = argc > 4;
    //  bool veryVeryVeryVerbose = argc > 5;

    printf("TEST " __FILE__ " CASE %d\n", test);

    switch (test) { case 0:
      case 2: {
        // --------------------------------------------------------------------
        // 'bsl::in_place_t' TYPEDEF
        //
        // Concerns:
        //: 1 The 'bsl::in_place_t' is a typedef for 'std::in_place_t' if
        //:   'std::in_place_t' is available.
        //: 2 'bsl::in_place_type_t<TYPE>' is an alias for
        //:   'std::in_place_type_t<TYPE>' if 'std::in_place_type_t<TYPE>' is
        //:    available.
        //: 3 'bsl::in_place_index<INDEX>' is an alias for
        //:   'std::in_place_index_t<INDEX>' if 'std::in_place_index_t<INDEX>'
        //:    is available.
        //
        // Plan:
        //: 1 Check that 'bsl::in_place_t' is the same type as
        //:   'std::in_place_t' using 'bsl::is_same' if CPP17 library is
        //:   available (C-1).
        //: 2 Check that 'bsl::in_place_type_t<int>' is the same type as
        //:   'std::in_place_type<int>' using 'bsl::is_same' if CPP17 library
        //:   is available (C-2).
        //: 3 Check that 'bsl::in_place_index_t<1>' is the same type as
        //:   'std::in_place_index_t<1>' using 'bsl::is_same' if CPP17 library
        //:   is available (C-3).
        //
        // Testing:
        //   bsl::in_place_t
        //   bsl::in_place_type_t
        //   bsl::in_place_index_t
        // --------------------------------------------------------------------
        if (verbose) printf("\n'bsl::in_placeX' TYPEDEF"
                            "\n========================\n");

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_BASELINE_LIBRARY
        BSLMF_ASSERT((bsl::is_same<bsl::in_place_t, std::in_place_t>::value));
        BSLMF_ASSERT((bsl::is_same<bsl::in_place_type_t<int>,
                                   std::in_place_type_t<int> >::value));
        BSLMF_ASSERT((bsl::is_same<bsl::in_place_index_t<1>,
                                   std::in_place_index_t<1> >::value));
#endif  //BSLS_LIBRARYFEATURES_HAS_CPP17_BASELINE_LIBRARY
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //   This case exercises basic functionality.
        //
        // Concerns:
        //: 1 The class 'bsl::in_place_t' is sufficiently functional and
        //:   'bsl::in_place' variable exists.
        //: 2 The class 'bsl::in_place_type_t' is sufficiently functional.
        //: 3 The class 'bsl::in_place_index_t' is sufficiently functional.
        //: 4 If variable templates are supported, that
        //:   bsl::in_place_type<TYPE> template variable exists.
        //: 5 If variable templates are supported, that
        //:   bsl::in_place_index<INDEX> template variable exists.
        //
        // Plan:
        //: 1 Perform and ad-hoc test of the primary modifiers and accessors.
        //
        // Testing:
        //   BREATHING TEST
        // --------------------------------------------------------------------

        if (verbose) printf("\nBREATHING TEST"
                            "\n==============\n");

        bsl::in_place_t b;

        ASSERT(isInplaceTagType(b));
        ASSERT(isInplaceTagType(bsl::in_place));

        bsl::in_place_type_t<int> bt;
        ASSERT(isInplaceTypeTagType<int>(bt));
#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
        ASSERT(isInplaceTypeTagType<int>(bsl::in_place_type<int>));
#endif  //BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES

        bsl::in_place_index_t<1> bi;
        ASSERT(isInplaceIndexTagType<1>(bi));
#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
        ASSERT(isInplaceIndexTagType<1>(bsl::in_place_index<1>));
#endif  //BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES

      } break;
      default: {
        fprintf(stderr, "WARNING: CASE `%d' NOT FOUND.\n", test);
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        fprintf(stderr, "Error, non-zero test status = %d.\n", testStatus);
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
