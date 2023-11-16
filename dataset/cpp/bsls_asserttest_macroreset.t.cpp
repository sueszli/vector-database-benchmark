// bsls_asserttest_macroreset.t.cpp                                   -*-C++-*-

#include <bsls_asserttest_macroreset.h>

#include <bsls_bsltestutil.h>
#include <cstdio>    // 'fprintf'
#include <cstdlib>   // 'atoi'

using namespace BloombergLP;
using namespace std;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// This component undefines macros from 'bsls_asserttest.h', 'bsls_assert.h',
// and 'bsls_review.h'.  We will validate that those macros are not defined,
// then define them, then include the header and validate again that they are
// not defined.
//-----------------------------------------------------------------------------
// [1] bsls_asserttest_macroreset.h
//-----------------------------------------------------------------------------

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

//=============================================================================
//                    GLOBAL CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

bool globalVerbose         = false;
bool globalVeryVerbose     = false;
bool globalVeryVeryVerbose = false;

// Forward declaration of function that appears after 'main' to minimize macro
// and header re-inclusion impact on code.
void testMacroHandling();
    // Assert test failures if any component macros are defined or if any do
    // not get undefined by including 'bsls_asserttest_macroreset.h'.

int main(int argc, char *argv[])
{
    int            test = argc > 1 ? atoi(argv[1]) : 0;
    int         verbose = argc > 2;
    int     veryVerbose = argc > 3;
    int veryVeryVerbose = argc > 4;

            globalVerbose =         verbose;
        globalVeryVerbose =     veryVerbose;
    globalVeryVeryVerbose = veryVeryVerbose;

    printf( "TEST %s CASE %d\n", __FILE__, test);

    switch (test) { case 0:  // zero is always the leading case
      case 1: {
        // --------------------------------------------------------------------
        // MACRO TEST
        //
        // Concerns:
        //: 1 The macros that 'bsls_asserttest_macroreset.h' purports to
        //:   undefine are indeed undefined immediately following its
        //:   inclusion.
        //
        //: 2 These macros should include all macros that are defined and leak
        //:   out of 'bsls_asserttest.h', 'bsls_review.h' and 'bsls_assert.h'.
        //
        // Plan:
        //: 1 Use a script to generate the list of macros that either leak out
        //:   of 'bsls_asserttest.h', 'bsls_review.h', and 'bsls_assert.h' or
        //:   that are undefined by 'bsls_asserttest.h',
        //:   'bsls_review_macroreset.h', or 'bsls_assert_macroreset.h'.
        //:
        //: 2 Call a function defined at the end of this file that contains the
        //:   generated code to do the remaining steps of this plan.
        //:
        //: 3 Check that all component macros are not defined.
        //:
        //: 4 Define all component macros with a fixed value.
        //:
        //: 5 *Re*-include 'bsls_asserttest_macroreset.h'.
        //:
        //: 6 Check that all component macros are not defined again.
        //
        // Testing:
        //   bsls_asserttest_macroreset.h
        // --------------------------------------------------------------------

        if (verbose) printf( "\nMACRO TEST"
                             "\n==========\n" );

        testMacroHandling();

      } break;
      default: {
          fprintf( stderr, "WARNING: CASE `%d` NOT FOUND.\n" , test);
          testStatus = -1;
      }
    }

    if (testStatus > 0) {
        fprintf( stderr, "Error, non-zero test status = %d.\n", testStatus );
    }

    return testStatus;
}


//--------------------------------------------------------------------GENERATOR
// The following 'bash' script generates all of the checks for the macros that
// are defined in 'bsls_asserttest.h', 'bsls_assert.h', and 'bsls_review.h' or
// undefined in 'bsls_asserttest_macroreset.h', 'bsls_assert_macroreset.h', and
// 'bsls_review_macroreset.h'.
//..
//  #!/bin/bash
//
//  HEADERS=( bsls_asserttest.h bsls_assert.h bsls_review.h )
//  RESETHEADERS=( bsls_asserttest_macroreset.h )
//  RESETHEADERS+=( bsls_assert_macroreset.h bsls_review_macroreset.h )
//
//  UNDEFINED=( $(cat ${HEADERS[@]} | grep "^\\s*#\\s*undef" |
//                    sed -e "s/.*undef //") )
//
//  DEFINED=( $(cat ${HEADERS[@]} ${RESETHEADERS} |
//            egrep "^\\s*#\\s*(define|undef)" |
//            sed -e "s/.*define //" -e "s/.*undef //" -e "s/[( ].*//" |
//            grep -v "TESTDRIVER_GUARD" |
//            egrep -v "(^BDE_ASSERT)|(^BSL_ASSERT)|(MACRORESET)|(CONTRACTS)" |
//            sort -u) )
//
//  MACROS=( $(for M in ${DEFINED[@]} ${UNDEFINED[@]} ; do echo "${M}" ; done |
//                 sort | uniq -u ) )
//
//  for M in "${MACROS[@]}" ; do
//      cat <<EOF
//      #ifdef ${M}
//          ASSERT(!"${M} is defined!");
//      #endif
//  EOF
//  done
//
//  echo
//  for M in "${MACROS[@]}" ; do
//      echo "    #define ${M} 17"
//  done
//
//  cat <<EOF
//
//      #undef INCLUDED_BSLS_ASSERTTEST_MACRORESET
//      #include <bsls_asserttest_macroreset.h>
//
//  EOF
//
//  for M in "${MACROS[@]}" ; do
//      cat <<EOF
//      #ifdef ${M}
//          ASSERT(!"${M} is still defined!");
//      #endif
//  EOF
//  done
//..
//----------------------------------------------------------------END GENERATOR

void testMacroHandling()
{
    // All asserts will be compiled out if tests pass - call this here to avoid
    // unused static function warnings.
    ASSERT(true);

    // smoke test
    #ifndef INCLUDED_BSLS_ASSERTTEST_MACRORESET
        ASSERT(!"INCLUDED_BSLS_ASSERTTEST_MACRORESET is NOT defined!");
    #endif

//--------------------------------------------------------------------GENERATED
    #ifdef BDE_BUILD_TARGET_DBG
        ASSERT(!"BDE_BUILD_TARGET_DBG is defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_OPT
        ASSERT(!"BDE_BUILD_TARGET_OPT is defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_SAFE
        ASSERT(!"BDE_BUILD_TARGET_SAFE is defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_SAFE_2
        ASSERT(!"BDE_BUILD_TARGET_SAFE_2 is defined!");
    #endif
    #ifdef BSLS_ASSERT
        ASSERT(!"BSLS_ASSERT is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_ACTIVE_FLAG is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_FAIL is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_FAIL_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_PASS is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_PASS_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_ACTIVE_FLAG is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_FAIL is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_PASS is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_PASS is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_PASS_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_ACTIVE_FLAG is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_FAIL is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_PASS is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_BRUTE_FORCE_IMP
        ASSERT(!"BSLS_ASSERTTEST_BRUTE_FORCE_IMP is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_BRUTE_FORCE_IMP_RAW
        ASSERT(!"BSLS_ASSERTTEST_BRUTE_FORCE_IMP_RAW is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CAN_CHECK_LEVELS
        ASSERT(!"BSLS_ASSERTTEST_CAN_CHECK_LEVELS is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CHECK_LEVEL
        ASSERT(!"BSLS_ASSERTTEST_CHECK_LEVEL is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CHECK_LEVEL_ARG
        ASSERT(!"BSLS_ASSERTTEST_CHECK_LEVEL_ARG is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_DISABLED_IMP
        ASSERT(!"BSLS_ASSERTTEST_DISABLED_IMP is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_IS_ACTIVE
        ASSERT(!"BSLS_ASSERTTEST_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_SAFE_2_BUILD_FLAG
        ASSERT(!"BSLS_ASSERTTEST_SAFE_2_BUILD_FLAG is defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_ASSERTTEST_VALIDATE_DISABLED_MACROS is defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSERT
        ASSERT(!"BSLS_ASSERT_ASSERT is defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSERT_IMP
        ASSERT(!"BSLS_ASSERT_ASSERT_IMP is defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSUME_IMP
        ASSERT(!"BSLS_ASSERT_ASSUME_IMP is defined!");
    #endif
    #ifdef BSLS_ASSERT_DISABLED_IMP
        ASSERT(!"BSLS_ASSERT_DISABLED_IMP is defined!");
    #endif
    #ifdef BSLS_ASSERT_INVOKE
        ASSERT(!"BSLS_ASSERT_INVOKE is defined!");
    #endif
    #ifdef BSLS_ASSERT_INVOKE_NORETURN
        ASSERT(!"BSLS_ASSERT_INVOKE_NORETURN is defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_IS_ASSUMED is defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_IS_REVIEW is defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_USED
        ASSERT(!"BSLS_ASSERT_IS_USED is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT_OPT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT_OPT is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT_SAFE
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT_SAFE is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_ASSERT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_ASSERT is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_OPT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_OPT is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_SAFE
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_SAFE is defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_NONE
        ASSERT(!"BSLS_ASSERT_LEVEL_NONE is defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT
        ASSERT(!"BSLS_ASSERT_OPT is defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_OPT_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_OPT_IS_ASSUMED is defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_OPT_IS_REVIEW is defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_USED
        ASSERT(!"BSLS_ASSERT_OPT_IS_USED is defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE
        ASSERT(!"BSLS_ASSERT_SAFE is defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_SAFE_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_SAFE_IS_ASSUMED is defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_SAFE_IS_REVIEW is defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_USED
        ASSERT(!"BSLS_ASSERT_SAFE_IS_USED is defined!");
    #endif
    #ifdef BSLS_ASSERT_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_ASSERT_VALIDATE_DISABLED_MACROS is defined!");
    #endif
    #ifdef BSLS_REVIEW
        ASSERT(!"BSLS_REVIEW is defined!");
    #endif
    #ifdef BSLS_REVIEW_DISABLED_IMP
        ASSERT(!"BSLS_REVIEW_DISABLED_IMP is defined!");
    #endif
    #ifdef BSLS_REVIEW_INVOKE
        ASSERT(!"BSLS_REVIEW_INVOKE is defined!");
    #endif
    #ifdef BSLS_REVIEW_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_REVIEW_IS_USED
        ASSERT(!"BSLS_REVIEW_IS_USED is defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_NONE
        ASSERT(!"BSLS_REVIEW_LEVEL_NONE is defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW is defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW_OPT
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW_OPT is defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW_SAFE
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW_SAFE is defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT
        ASSERT(!"BSLS_REVIEW_OPT is defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_OPT_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT_IS_USED
        ASSERT(!"BSLS_REVIEW_OPT_IS_USED is defined!");
    #endif
    #ifdef BSLS_REVIEW_REVIEW_COUNT_IMP
        ASSERT(!"BSLS_REVIEW_REVIEW_COUNT_IMP is defined!");
    #endif
    #ifdef BSLS_REVIEW_REVIEW_IMP
        ASSERT(!"BSLS_REVIEW_REVIEW_IMP is defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE
        ASSERT(!"BSLS_REVIEW_SAFE is defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_SAFE_IS_ACTIVE is defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE_IS_USED
        ASSERT(!"BSLS_REVIEW_SAFE_IS_USED is defined!");
    #endif
    #ifdef BSLS_REVIEW_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_REVIEW_VALIDATE_DISABLED_MACROS is defined!");
    #endif
    #ifdef INCLUDED_BSLS_ASSERT
        ASSERT(!"INCLUDED_BSLS_ASSERT is defined!");
    #endif
    #ifdef INCLUDED_BSLS_ASSERTTEST
        ASSERT(!"INCLUDED_BSLS_ASSERTTEST is defined!");
    #endif
    #ifdef INCLUDED_BSLS_REVIEW
        ASSERT(!"INCLUDED_BSLS_REVIEW is defined!");
    #endif

    #define BDE_BUILD_TARGET_DBG 17
    #define BDE_BUILD_TARGET_OPT 17
    #define BDE_BUILD_TARGET_SAFE 17
    #define BDE_BUILD_TARGET_SAFE_2 17
    #define BSLS_ASSERT 17
    #define BSLS_ASSERTTEST_ASSERT_ACTIVE_FLAG 17
    #define BSLS_ASSERTTEST_ASSERT_FAIL 17
    #define BSLS_ASSERTTEST_ASSERT_FAIL_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL 17
    #define BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_INVOKE_PASS 17
    #define BSLS_ASSERTTEST_ASSERT_INVOKE_PASS_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_OPT_ACTIVE_FLAG 17
    #define BSLS_ASSERTTEST_ASSERT_OPT_FAIL 17
    #define BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_OPT_PASS 17
    #define BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_PASS 17
    #define BSLS_ASSERTTEST_ASSERT_PASS_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_SAFE_ACTIVE_FLAG 17
    #define BSLS_ASSERTTEST_ASSERT_SAFE_FAIL 17
    #define BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW 17
    #define BSLS_ASSERTTEST_ASSERT_SAFE_PASS 17
    #define BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW 17
    #define BSLS_ASSERTTEST_BRUTE_FORCE_IMP 17
    #define BSLS_ASSERTTEST_BRUTE_FORCE_IMP_RAW 17
    #define BSLS_ASSERTTEST_CAN_CHECK_LEVELS 17
    #define BSLS_ASSERTTEST_CHECK_LEVEL 17
    #define BSLS_ASSERTTEST_CHECK_LEVEL_ARG 17
    #define BSLS_ASSERTTEST_DISABLED_IMP 17
    #define BSLS_ASSERTTEST_IS_ACTIVE 17
    #define BSLS_ASSERTTEST_SAFE_2_BUILD_FLAG 17
    #define BSLS_ASSERTTEST_VALIDATE_DISABLED_MACROS 17
    #define BSLS_ASSERT_ASSERT 17
    #define BSLS_ASSERT_ASSERT_IMP 17
    #define BSLS_ASSERT_ASSUME_IMP 17
    #define BSLS_ASSERT_DISABLED_IMP 17
    #define BSLS_ASSERT_INVOKE 17
    #define BSLS_ASSERT_INVOKE_NORETURN 17
    #define BSLS_ASSERT_IS_ACTIVE 17
    #define BSLS_ASSERT_IS_ASSUMED 17
    #define BSLS_ASSERT_IS_REVIEW 17
    #define BSLS_ASSERT_IS_USED 17
    #define BSLS_ASSERT_LEVEL_ASSERT 17
    #define BSLS_ASSERT_LEVEL_ASSERT_OPT 17
    #define BSLS_ASSERT_LEVEL_ASSERT_SAFE 17
    #define BSLS_ASSERT_LEVEL_ASSUME_ASSERT 17
    #define BSLS_ASSERT_LEVEL_ASSUME_OPT 17
    #define BSLS_ASSERT_LEVEL_ASSUME_SAFE 17
    #define BSLS_ASSERT_LEVEL_NONE 17
    #define BSLS_ASSERT_OPT 17
    #define BSLS_ASSERT_OPT_IS_ACTIVE 17
    #define BSLS_ASSERT_OPT_IS_ASSUMED 17
    #define BSLS_ASSERT_OPT_IS_REVIEW 17
    #define BSLS_ASSERT_OPT_IS_USED 17
    #define BSLS_ASSERT_SAFE 17
    #define BSLS_ASSERT_SAFE_IS_ACTIVE 17
    #define BSLS_ASSERT_SAFE_IS_ASSUMED 17
    #define BSLS_ASSERT_SAFE_IS_REVIEW 17
    #define BSLS_ASSERT_SAFE_IS_USED 17
    #define BSLS_ASSERT_VALIDATE_DISABLED_MACROS 17
    #define BSLS_REVIEW 17
    #define BSLS_REVIEW_DISABLED_IMP 17
    #define BSLS_REVIEW_INVOKE 17
    #define BSLS_REVIEW_IS_ACTIVE 17
    #define BSLS_REVIEW_IS_USED 17
    #define BSLS_REVIEW_LEVEL_NONE 17
    #define BSLS_REVIEW_LEVEL_REVIEW 17
    #define BSLS_REVIEW_LEVEL_REVIEW_OPT 17
    #define BSLS_REVIEW_LEVEL_REVIEW_SAFE 17
    #define BSLS_REVIEW_OPT 17
    #define BSLS_REVIEW_OPT_IS_ACTIVE 17
    #define BSLS_REVIEW_OPT_IS_USED 17
    #define BSLS_REVIEW_REVIEW_COUNT_IMP 17
    #define BSLS_REVIEW_REVIEW_IMP 17
    #define BSLS_REVIEW_SAFE 17
    #define BSLS_REVIEW_SAFE_IS_ACTIVE 17
    #define BSLS_REVIEW_SAFE_IS_USED 17
    #define BSLS_REVIEW_VALIDATE_DISABLED_MACROS 17
    #define INCLUDED_BSLS_ASSERT 17
    #define INCLUDED_BSLS_ASSERTTEST 17
    #define INCLUDED_BSLS_REVIEW 17

    #undef INCLUDED_BSLS_ASSERTTEST_MACRORESET
    #include <bsls_asserttest_macroreset.h>

    #ifdef BDE_BUILD_TARGET_DBG
        ASSERT(!"BDE_BUILD_TARGET_DBG is still defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_OPT
        ASSERT(!"BDE_BUILD_TARGET_OPT is still defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_SAFE
        ASSERT(!"BDE_BUILD_TARGET_SAFE is still defined!");
    #endif
    #ifdef BDE_BUILD_TARGET_SAFE_2
        ASSERT(!"BDE_BUILD_TARGET_SAFE_2 is still defined!");
    #endif
    #ifdef BSLS_ASSERT
        ASSERT(!"BSLS_ASSERT is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_ACTIVE_FLAG is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_FAIL is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_FAIL_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_FAIL_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_PASS is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_INVOKE_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_INVOKE_PASS_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_ACTIVE_FLAG is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_FAIL is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_FAIL_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_PASS is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_OPT_PASS_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_PASS is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_PASS_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_ACTIVE_FLAG
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_ACTIVE_FLAG is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_FAIL
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_FAIL is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_FAIL_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_PASS
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_PASS is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW
        ASSERT(!"BSLS_ASSERTTEST_ASSERT_SAFE_PASS_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_BRUTE_FORCE_IMP
        ASSERT(!"BSLS_ASSERTTEST_BRUTE_FORCE_IMP is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_BRUTE_FORCE_IMP_RAW
        ASSERT(!"BSLS_ASSERTTEST_BRUTE_FORCE_IMP_RAW is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CAN_CHECK_LEVELS
        ASSERT(!"BSLS_ASSERTTEST_CAN_CHECK_LEVELS is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CHECK_LEVEL
        ASSERT(!"BSLS_ASSERTTEST_CHECK_LEVEL is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_CHECK_LEVEL_ARG
        ASSERT(!"BSLS_ASSERTTEST_CHECK_LEVEL_ARG is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_DISABLED_IMP
        ASSERT(!"BSLS_ASSERTTEST_DISABLED_IMP is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_IS_ACTIVE
        ASSERT(!"BSLS_ASSERTTEST_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_SAFE_2_BUILD_FLAG
        ASSERT(!"BSLS_ASSERTTEST_SAFE_2_BUILD_FLAG is still defined!");
    #endif
    #ifdef BSLS_ASSERTTEST_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_ASSERTTEST_VALIDATE_DISABLED_MACROS is still defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSERT
        ASSERT(!"BSLS_ASSERT_ASSERT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSERT_IMP
        ASSERT(!"BSLS_ASSERT_ASSERT_IMP is still defined!");
    #endif
    #ifdef BSLS_ASSERT_ASSUME_IMP
        ASSERT(!"BSLS_ASSERT_ASSUME_IMP is still defined!");
    #endif
    #ifdef BSLS_ASSERT_DISABLED_IMP
        ASSERT(!"BSLS_ASSERT_DISABLED_IMP is still defined!");
    #endif
    #ifdef BSLS_ASSERT_INVOKE
        ASSERT(!"BSLS_ASSERT_INVOKE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_INVOKE_NORETURN
        ASSERT(!"BSLS_ASSERT_INVOKE_NORETURN is still defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_IS_ASSUMED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_IS_REVIEW is still defined!");
    #endif
    #ifdef BSLS_ASSERT_IS_USED
        ASSERT(!"BSLS_ASSERT_IS_USED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT_OPT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT_OPT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSERT_SAFE
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSERT_SAFE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_ASSERT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_ASSERT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_OPT
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_OPT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_ASSUME_SAFE
        ASSERT(!"BSLS_ASSERT_LEVEL_ASSUME_SAFE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_LEVEL_NONE
        ASSERT(!"BSLS_ASSERT_LEVEL_NONE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT
        ASSERT(!"BSLS_ASSERT_OPT is still defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_OPT_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_OPT_IS_ASSUMED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_OPT_IS_REVIEW is still defined!");
    #endif
    #ifdef BSLS_ASSERT_OPT_IS_USED
        ASSERT(!"BSLS_ASSERT_OPT_IS_USED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE
        ASSERT(!"BSLS_ASSERT_SAFE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_ACTIVE
        ASSERT(!"BSLS_ASSERT_SAFE_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_ASSUMED
        ASSERT(!"BSLS_ASSERT_SAFE_IS_ASSUMED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_REVIEW
        ASSERT(!"BSLS_ASSERT_SAFE_IS_REVIEW is still defined!");
    #endif
    #ifdef BSLS_ASSERT_SAFE_IS_USED
        ASSERT(!"BSLS_ASSERT_SAFE_IS_USED is still defined!");
    #endif
    #ifdef BSLS_ASSERT_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_ASSERT_VALIDATE_DISABLED_MACROS is still defined!");
    #endif
    #ifdef BSLS_REVIEW
        ASSERT(!"BSLS_REVIEW is still defined!");
    #endif
    #ifdef BSLS_REVIEW_DISABLED_IMP
        ASSERT(!"BSLS_REVIEW_DISABLED_IMP is still defined!");
    #endif
    #ifdef BSLS_REVIEW_INVOKE
        ASSERT(!"BSLS_REVIEW_INVOKE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_IS_USED
        ASSERT(!"BSLS_REVIEW_IS_USED is still defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_NONE
        ASSERT(!"BSLS_REVIEW_LEVEL_NONE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW is still defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW_OPT
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW_OPT is still defined!");
    #endif
    #ifdef BSLS_REVIEW_LEVEL_REVIEW_SAFE
        ASSERT(!"BSLS_REVIEW_LEVEL_REVIEW_SAFE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT
        ASSERT(!"BSLS_REVIEW_OPT is still defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_OPT_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_OPT_IS_USED
        ASSERT(!"BSLS_REVIEW_OPT_IS_USED is still defined!");
    #endif
    #ifdef BSLS_REVIEW_REVIEW_COUNT_IMP
        ASSERT(!"BSLS_REVIEW_REVIEW_COUNT_IMP is still defined!");
    #endif
    #ifdef BSLS_REVIEW_REVIEW_IMP
        ASSERT(!"BSLS_REVIEW_REVIEW_IMP is still defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE
        ASSERT(!"BSLS_REVIEW_SAFE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE_IS_ACTIVE
        ASSERT(!"BSLS_REVIEW_SAFE_IS_ACTIVE is still defined!");
    #endif
    #ifdef BSLS_REVIEW_SAFE_IS_USED
        ASSERT(!"BSLS_REVIEW_SAFE_IS_USED is still defined!");
    #endif
    #ifdef BSLS_REVIEW_VALIDATE_DISABLED_MACROS
        ASSERT(!"BSLS_REVIEW_VALIDATE_DISABLED_MACROS is still defined!");
    #endif
    #ifdef INCLUDED_BSLS_ASSERT
        ASSERT(!"INCLUDED_BSLS_ASSERT is still defined!");
    #endif
    #ifdef INCLUDED_BSLS_ASSERTTEST
        ASSERT(!"INCLUDED_BSLS_ASSERTTEST is still defined!");
    #endif
    #ifdef INCLUDED_BSLS_REVIEW
        ASSERT(!"INCLUDED_BSLS_REVIEW is still defined!");
    #endif
//----------------------------------------------------------------END GENERATED
}

// ----------------------------------------------------------------------------
// Copyright 2018 Bloomberg Finance L.P.
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
