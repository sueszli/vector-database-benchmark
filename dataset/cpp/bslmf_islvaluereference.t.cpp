// bslmf_islvaluereference.t.cpp                                      -*-C++-*-
#include <bslmf_islvaluereference.h>

#include <bslmf_issame.h>

#include <bsls_bsltestutil.h>
#include <bsls_compilerfeatures.h>
#include <bsls_libraryfeatures.h>

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY
#include <type_traits>  // 'std::is_base_of',
                        // 'std::is_lvalue_reference', and
                        // 'std::is_lvalue_reference_v' (C++17)
#endif

#include <stdio.h>   // 'printf'
#include <stdlib.h>  // 'atoi'

using namespace BloombergLP;

//=============================================================================
//                                TEST PLAN
//-----------------------------------------------------------------------------
//                                Overview
//                                --------
// The component under test defines a meta-function,
// 'bsl::is_lvalue_reference' and a template variable
// 'bsl::is_lvalue_reference_v', that determine whether a template parameter
// type is an lvalue reference type.  Thus, we need to ensure that the value
// returned by this meta-function is correct for each possible category of
// types.
//
// ----------------------------------------------------------------------------
// PUBLIC CLASS DATA
// [ 1] bsl::is_lvalue_reference::value
// [ 1] bsl::is_lvalue_reference_v
//
// ----------------------------------------------------------------------------
// [ 3] USAGE EXAMPLE
// [ 2] CONCERN: Conforms to implementation constraints.

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
//                      WARNING SUPPRESSION
//-----------------------------------------------------------------------------

// This test driver intentional creates types with unusual use of cv-qualifiers
// in order to confirm that there are no strange corners of the type system
// that are not addressed by this traits component.  Consequently, we disable
// certain warnings from common compilers.

#if defined(BSLS_PLATFORM_HAS_PRAGMA_GCC_DIAGNOSTIC)
# pragma GCC diagnostic ignored "-Wignored-qualifiers"
#elif defined(BSLS_PLATFORM_CMP_MSVC)
# pragma warning(disable : 4180)
#elif defined(BSLS_PLATFORM_CMP_SUN)
# pragma error_messages(off, functypequal)
#endif

//=============================================================================
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

namespace {

enum EnumTestType {
    // This user-defined 'enum' type is intended to be used for testing as the
    // template parameter 'TYPE' of 'bsl::is_lvalue_reference'.
};

struct StructTestType {
    // This user-defined 'struct' type is intended to be used for testing as
    // the template parameter 'TYPE' of 'bsl::is_lvalue_reference'.
};

union UnionTestType {
    // This user-defined 'union' type is intended to be used for testing as the
    // template parameter 'TYPE' of 'bsl::is_lvalue_reference'.
};

class BaseClassTestType {
    // This user-defined base class type is intended to be used for testing as
    // the template parameter 'TYPE' of 'bsl::is_lvalue_reference'.
};

class DerivedClassTestType : public BaseClassTestType {
    // This user-defined derived class type is intended to be used for testing
    // as the template parameter 'TYPE' of 'bsl::is_lvalue_reference'.
};

typedef int (StructTestType::*MethodPtrTestType) ();
    // This pointer to non-static member function type is intended to be used
    // for testing as the template parameter 'TYPE' of
    // 'bsl::is_lvalue_reference'.

typedef void (*FunctionPtrTestType) ();
    // This function pointer type is intended to be used for testing as the
    // template parameter 'TYPE' of 'bsl::is_lvalue_reference'.

typedef int StructTestType::*PMD;
    // This pointer to member object type is intended to be used for testing as
    // the template parameter 'TYPE' of 'bsl::is_lvalue_reference'.

struct Incomplete;
    // This incomplete 'struct' type is intended to be used for testing as the
    // template parameter 'TYPE' of 'bsl::is_lvalue_reference'.

}  // close unnamed namespace

#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
#define ASSERT_V_EQ_VALUE(TYPE)                                               \
    ASSERT(bsl::is_lvalue_reference<TYPE>::value ==                           \
                                              bsl::is_lvalue_reference_v<TYPE>)
    // Test whether 'bsl::is_lvalue_reference_v<TYPE>' value equals to
    // 'bsl::is_lvalue_reference<TYPE>::value'.
#else
#define ASSERT_V_EQ_VALUE(TYPE)
#endif

#define TYPE_ASSERT(META_FUNC, TYPE, result)                  \
    ASSERT(result == META_FUNC<TYPE>::value);                 \
    ASSERT_V_EQ_VALUE(TYPE)
    // Test that the result of 'META_FUNC' has the same value as the expected
    // 'result'.  Confirm that the result value of the 'META_FUNC' and the
    // value of the 'META_FUNC_v' variable are the same.

#define TYPE_ASSERT_CVQ_PREFIX(META_FUNC, TYPE, result)       \
    TYPE_ASSERT(META_FUNC,                TYPE, result);      \
    TYPE_ASSERT(META_FUNC, const          TYPE, result);      \
    TYPE_ASSERT(META_FUNC, volatile       TYPE, result);      \
    TYPE_ASSERT(META_FUNC, const volatile TYPE, result);
    // Test cv-qualified combinations on the specified 'TYPE'.

#define TYPE_ASSERT_CVQ_SUFFIX(META_FUNC, TYPE, result)       \
    TYPE_ASSERT(META_FUNC, TYPE,                result);      \
    TYPE_ASSERT(META_FUNC, TYPE const,          result);      \
    TYPE_ASSERT(META_FUNC, TYPE volatile,       result);      \
    TYPE_ASSERT(META_FUNC, TYPE const volatile, result);
    // Test cv-qualified combinations on the specified 'TYPE'.

#define TYPE_ASSERT_CVQ_REF(META_FUNC, TYPE, result)           \
    TYPE_ASSERT(META_FUNC, TYPE&,                result);      \
    TYPE_ASSERT(META_FUNC, TYPE const&,          result);      \
    TYPE_ASSERT(META_FUNC, TYPE volatile&,       result);      \
    TYPE_ASSERT(META_FUNC, TYPE const volatile&, result);
    // Test references to cv-qualified combinations on the specified 'TYPE'.

#define TYPE_ASSERT_CVQ(META_FUNC, TYPE, result)                     \
    TYPE_ASSERT_CVQ_PREFIX(META_FUNC, TYPE,                result);  \
    TYPE_ASSERT_CVQ_PREFIX(META_FUNC, TYPE const,          result);  \
    TYPE_ASSERT_CVQ_PREFIX(META_FUNC, TYPE       volatile, result);  \
    TYPE_ASSERT_CVQ_PREFIX(META_FUNC, TYPE const volatile, result);  \
    // Test all cv-qualified combinations on the specified 'TYPE'.

//=============================================================================
//                              MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int                 test = argc > 1 ? atoi(argv[1]) : 0;
    bool             verbose = argc > 2;
    bool         veryVerbose = argc > 3;
    bool     veryVeryVerbose = argc > 4;
    bool veryVeryVeryVerbose = argc > 5;

    (void)veryVerbose;          // suppress warning
    (void)veryVeryVerbose;      // suppress warning
    (void)veryVeryVeryVerbose;  // suppress warning

    setbuf(stdout, NULL);       // Use unbuffered output

    printf("TEST " __FILE__ " CASE %d\n", test);

    switch (test) { case 0:
      case 3: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
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

        if (verbose) printf("USAGE EXAMPLE\n"
                            "=============\n");

///Usage
///-----
// In this section we show intended use of this component.
//
///Example 1: Verify Lvalue Reference Types
/// - - - - - - - - - - - - - - - - - - - -
// Suppose that we want to assert whether a set of types are lvalue reference
// types.
//
// Now, we instantiate the 'bsl::is_lvalue_reference' template for both a
// non-reference type and a reference type, and assert the 'value' static data
// member of each instantiation:
//..
    ASSERT(false == bsl::is_lvalue_reference<int>::value);
    ASSERT(true  == bsl::is_lvalue_reference<int&>::value);
//..
// Note that if the current compiler supports the variable templates C++14
// feature then we can re-write the snippet of code above using the
// 'bsl::is_function_v' variable as follows:
//..
#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
    ASSERT(false == bsl::is_lvalue_reference_v<int>);
    ASSERT(true  == bsl::is_lvalue_reference_v<int&>);
#endif
//..
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING IMPLEMENTATION
        //
        // Concerns:
        //: 1 The 'bsl::is_lvalue_reference' meta function is *never*
        //:   implemented as an alias to the 'std::is_lvalue_reference'
        //:   Standard meta function.
        //:
        //: 2 The 'bsl::is_lvalue_reference' meta function is *always* based on
        //:   either 'bsl::true_type' or 'bsl::false_type'.
        //:
        //: 3 The 'bsl::is_lvalue_reference_v' variable template *is*
        //:   implemented using the 'std::is_lvalue_reference_v' Standard
        //:   variable template.
        //
        // Plan:
        //: 1 When 'BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY' is
        //:   defined, use 'bsl::is_same' to compare 'bsl::is_lvalue_reference'
        //:   to 'std::is_lvalue_reference' using a representative type.  (C-1)
        //:
        //: 2 When 'BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY' is
        //:   defined, use 'std::is_base_of' to confirm that the
        //:   'bsl::is_floating_point' meta function has 'bsl::true_type' or
        //:   'bsl::false_type', as appropriate, as a base class.  (C-2)
        //:
        //:   o For all versions, explicitly access the 'value' member to
        //:     confirm that the inheritance is neither 'private', 'protected',
        //:     nor ambiguous.
        //:
        //: 3 When 'BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES' and (for
        //:   'std::is_lvalue_reference_v')
        //:   'BSLS_LIBRARYFEATURES_HAS_CPP17_BASELINE_LIBRARY' are defined,
        //:   compare for equality the addresses of
        //:   'bsl::is_lvalue_reference_v' and 'std::is_lvalue_reference_v'
        //:   using a representative type.  (C-3)
        //
        // Testing:
        //   CONCERN: Conforms to implementation constraints.
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING IMPLEMENTATION"
                            "\n======================\n");

        if (veryVerbose) printf(
                             "\nTesting 'is_lvalue_reference' using 'int'.\n");

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY
                                                  // 'std::is_lvalue_reference'
        ASSERT((false == bsl::is_same<bsl::is_lvalue_reference<int>,
                                      std::is_lvalue_reference<int> >::value));

        ASSERT((std::is_base_of<bsl::true_type,
                                bsl::is_lvalue_reference<int&>>::value));

        ASSERT((std::is_base_of<bsl::false_type,
                                bsl::is_lvalue_reference<int >>::value));
#endif

        ASSERT(true  == bsl::is_lvalue_reference<int&>::value);
        ASSERT(false == bsl::is_lvalue_reference<int >::value);

#if defined BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES                  \
 && defined BSLS_LIBRARYFEATURES_HAS_CPP17_BASELINE_LIBRARY
        if (veryVerbose) printf(
                           "\nTesting 'is_lvalue_reference_v' using 'int'.\n");

        typedef int T;

        ASSERT((&bsl::is_lvalue_reference_v<T>
             == &std::is_lvalue_reference_v<T>));
#endif
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // 'bsl::is_lvalue_reference::value'
        //   Ensure that the static data member 'value' of
        //   'bsl::is_lvalue_reference' instantiations having various (template
        //   parameter) 'TYPE's has the correct value.
        //
        // Concerns:
        //: 1 'is_lvalue_reference::value' is 'false' when 'TYPE' is a
        //:   (possibly cv-qualified) primitive type.
        //:
        //: 2 'is_lvalue_reference::value' is 'false' when 'TYPE' is a
        //:   (possibly cv-qualified) user-defined type.
        //:
        //: 3 'is_lvalue_reference::value' is 'false' when 'TYPE' is a
        //:   (possibly cv-qualified) pointer type.
        //:
        //: 4 'is_lvalue_reference::value' is 'true' when 'TYPE' is an
        //:   (possibly cv-qualified) lvalue reference type.
        //:
        //: 5 'is_lvalue_reference::value' is 'false' when 'TYPE' is a
        //:   (possibly cv-qualified) function type.
        //:
        //: 6 That 'is_lvalue_reference<T>::value' has the same value as
        //:   'is_lvalue_reference_v<T>' for a variety of template parameter
        //:   types.
        //
        // Plan:
        //: 1 Verify that 'bsl::is_lvalue_reference::value' has the correct
        //:   value for each (template parameter) 'TYPE' in the concerns.
        //
        // Testing:
        //   bsl::is_lvalue_reference::value
        //   bsl::is_lvalue_reference_v
        // --------------------------------------------------------------------

        if (verbose)  printf("'bsl::is_lvalue_reference::value'\n"
                             "=================================\n");

        // C-1
        TYPE_ASSERT_CVQ_SUFFIX(bsl::is_lvalue_reference, void, false);
        TYPE_ASSERT_CVQ_SUFFIX(bsl::is_lvalue_reference, int,  false);

        // C-2
        TYPE_ASSERT_CVQ_SUFFIX(
                        bsl::is_lvalue_reference, EnumTestType,         false);
        TYPE_ASSERT_CVQ_SUFFIX(
                        bsl::is_lvalue_reference, StructTestType,       false);
        TYPE_ASSERT_CVQ_SUFFIX(
                        bsl::is_lvalue_reference, UnionTestType,        false);
        TYPE_ASSERT_CVQ_SUFFIX(
                        bsl::is_lvalue_reference, BaseClassTestType,    false);
        TYPE_ASSERT_CVQ_SUFFIX(
                        bsl::is_lvalue_reference, DerivedClassTestType, false);

        // C-3
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, int*,                       false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, StructTestType*,            false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, int StructTestType::*,      false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, int StructTestType::* *,    false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, UnionTestType*,             false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, PMD BaseClassTestType::*,   false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, PMD BaseClassTestType::* *, false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, BaseClassTestType*,         false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, DerivedClassTestType*,      false);
        TYPE_ASSERT_CVQ(
                  bsl::is_lvalue_reference, Incomplete*,                false);
        TYPE_ASSERT_CVQ_SUFFIX(
                         bsl::is_lvalue_reference, MethodPtrTestType,   false);
        TYPE_ASSERT_CVQ_SUFFIX(
                         bsl::is_lvalue_reference, FunctionPtrTestType, false);

        // C-4
        TYPE_ASSERT_CVQ_REF(bsl::is_lvalue_reference, int,  true);

        TYPE_ASSERT_CVQ_REF(
                         bsl::is_lvalue_reference, EnumTestType,         true);
        TYPE_ASSERT_CVQ_REF(
                         bsl::is_lvalue_reference, StructTestType,       true);
        TYPE_ASSERT_CVQ_REF(
                         bsl::is_lvalue_reference, UnionTestType,        true);
        TYPE_ASSERT_CVQ_REF(
                         bsl::is_lvalue_reference, BaseClassTestType,    true);
        TYPE_ASSERT_CVQ_REF(
                         bsl::is_lvalue_reference, DerivedClassTestType, true);

        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, int*,                     true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, StructTestType*,          true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, int StructTestType::*,    true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, int StructTestType::* *,  true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, UnionTestType*,           true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, PMD BaseClassTestType::*, true);
        TYPE_ASSERT_CVQ_REF(
                   bsl::is_lvalue_reference, PMD BaseClassTestType::* *, true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, BaseClassTestType*,       true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, DerivedClassTestType*,    true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, Incomplete*,              true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, MethodPtrTestType,        true);
        TYPE_ASSERT_CVQ_REF(
                     bsl::is_lvalue_reference, FunctionPtrTestType,      true);

        ASSERT( bsl::is_lvalue_reference<int  (&)(int )>::value );
        ASSERT( bsl::is_lvalue_reference<void (&)(void)>::value );
        ASSERT( bsl::is_lvalue_reference<int  (&)(void)>::value );
        ASSERT( bsl::is_lvalue_reference<void (&)(int )>::value );

        // C-5
        TYPE_ASSERT_CVQ_PREFIX(bsl::is_lvalue_reference, int  (int),  false);
        TYPE_ASSERT_CVQ_PREFIX(bsl::is_lvalue_reference, void (void), false);
        TYPE_ASSERT_CVQ_PREFIX(bsl::is_lvalue_reference, int  (void), false);
        TYPE_ASSERT_CVQ_PREFIX(bsl::is_lvalue_reference, void (int),  false);
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
// Copyright 2013 Bloomberg Finance L.P.
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
