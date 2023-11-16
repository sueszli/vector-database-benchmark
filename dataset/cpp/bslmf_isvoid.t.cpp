// bslmf_isvoid.t.cpp                                                 -*-C++-*-
#include <bslmf_isvoid.h>

#include <bslmf_integralconstant.h>
#include <bslmf_metaint.h>

#include <bsls_bsltestutil.h>

#include <stdio.h>   // 'printf'
#include <stdlib.h>  // 'atoi'

using namespace BloombergLP;

//=============================================================================
//                                TEST PLAN
//-----------------------------------------------------------------------------
//                                Overview
//                                --------
// The component under test defines two meta-functions, 'bsl::is_void' and
// 'bslmf::IsVoid' and a template variable 'bsl::is_void_t', that determine
// whether a template parameter type is the (possibly cv-qualified) 'void'
// type.  Also the component defines template variable 'bsl::is_void_v' that
// represents the result value of the 'bsl::is_void' meta-function.  Thus, we
// need to ensure that the values returned by these meta-functions are correct
// for each possible category of types.  Since the two meta-functions and the
// variable are functionally equivalent, we will use the same set of types for
// both.
//
//-----------------------------------------------------------------------------
// [ 4] bsl::is_void conversion to bool
// [ 4] bsl::is_void conversion to bsl::true_type and bsl::false_type
// [ 4] bslmf::IsVoid conversion to bool
// [ 4] bslmf::IsVoid conversion to bslmf::MetaInt
// [ 3] bsl::is_void::type
// [ 3] bslmf::IsVoid::type
// [ 2] bsl::is_void::value
// [ 2] bsl::is_void_v
// [ 2] bslmf::IsVoid::value
//-----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [ 5] USAGE EXAMPLE

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
//                  GLOBAL HELPER FUNCTIONS FOR TESTING
//-----------------------------------------------------------------------------
namespace {
struct Empty {
    // This 'struct' is an empty dummy type to allow formation of pointer-to-
    // member types for testing.  Note that as all such uses expected for
    // testing this component will be initialized only with null values, there
    // is no need for any actual members to point to.
};

struct Incomplete;
    // A forward declaration for the type that shall remain incomplete for this
    // whole translation unit.  This is provided solely for the purpose of
    // testing the meta-functions with incomplete types.

template <int N>
bool dispatchFalseType(...)                 { return false; }
bool dispatchFalseType(bsl::false_type)     { return true;  }

template <int N>
bool dispatchTrueType(...)                  { return false; }
bool dispatchTrueType(bsl::true_type)       { return true;  }

template <int N>
bool dispatchMeta0Type(...)                 { return false; }
bool dispatchMeta0Type(bslmf::MetaInt<0>)   { return true;  }

template <int N>
bool dispatchMeta1Type(...)                  { return false; }
bool dispatchMeta1Type(bslmf::MetaInt<1>)    { return true;  }

template <class PREDICATE>
bool isFalseType() { return false; }

template <>
bool isFalseType<bsl::false_type>() { return true; }

template <class PREDICATE>
bool isTrueType() { return false; }

template <>
bool isTrueType<bsl::true_type>() { return true; }

}  // close unnamed namespace

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
      case 5: {
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

        if (verbose) printf("\nUSAGE EXAMPLE"
                            "\n=============\n");

///Usage
///-----
// In this section we show intended use of this component.
//
///Example 1: Determine Whether a Type is the 'void' Type
/// - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Suppose that we want to assert whether a particular type is the 'void' type.
//
// First, we create two 'typedef's -- the 'void' type and another type:
//..
    typedef int  MyType;
    typedef void MyVoidType;
//..
// Now, we instantiate the 'bsl::is_void' template for each of the 'typedef's
// and assert the 'value' static data member of each instantiation:
//..
    ASSERT(false == bsl::is_void<MyType>::value);
    ASSERT(true  == bsl::is_void<MyVoidType>::value);
//..
// Note that if the current compiler supports the variable templates C++14
// feature, then we can re-write the snippet of code above as follows:
//..
#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
    ASSERT(false == bsl::is_void_v<MyType>);
    ASSERT(true  == bsl::is_void_v<MyVoidType>);
#endif
//..
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING type conversion
        //
        // Concerns:
        //: 1 Objects of type 'bslmf::IsVoid' are unambiguously convertible to
        //:   'bslmf::MetaInt<1>' if the (template parameter) type is the
        //:   cv-qualified 'void' type, and 'bslmf::MetaInt<0>' otherwise.
        //:
        //: 2 Objects of type 'bsl::is_void' are unambiguously convertible to
        //:   'bsl::true_type' if the (template parameter) type is the
        //:   cv-qualified 'void' type, and 'bsl::false_type' otherwise.
        //:
        //: 3 Objects of both 'bslmf::IsVoid' and 'bsl::is_void' are
        //:   contextually convertible to the boolean value 'true' if the
        //:   (template parameter) type is the cv-qualified 'void' type, and
        //:   'false' otherwise.
        //
        // Plan:
        //: 1 Define a set of one-parameter function overloads that return
        //:   'true' if and only if the argument is one of the following types:
        //:   'bsl::true_type', 'bsl::false_type', 'bslmf::MetaInt<0>', and
        //:   'bslmf::MetaInt<1>'.
        //:
        //: 2 For 'bslmf::IsVoid' instantiated on a variety of ('void' and
        //:   non-'void') types, verify the return values from the functions
        //:   defined in P-1.  (C-1)
        //:
        //: 3 For 'bsl::is_void' instantiated on a variety of ('void' and
        //:   non-'void') types, verify the return values from the functions
        //:   defined in P-1.  (C-2)
        //:
        //: 3 For both 'bsl::is_void' and 'bslmf::IsVoid' instantiated on a
        //:   variety of ('void' and non-'void') types, verify the object
        //:   directly as a boolean value.  (C-3)
        //
        // Testing:
        //   bslmf::IsVoid conversion to bool
        //   bslmf::IsVoid conversion to bslmf::MetaInt
        //   bsl::is_void conversion to bool
        //   bsl::is_void conversion to bsl::true_type and bsl::false_type
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING type conversion"
                            "\n=======================\n");

        ASSERT(dispatchMeta1Type(bslmf::IsVoid<void>()));
        ASSERT(dispatchMeta1Type(bslmf::IsVoid<const void>()));
        ASSERT(dispatchMeta1Type(bslmf::IsVoid<volatile void>()));
        ASSERT(dispatchMeta1Type(bslmf::IsVoid<const volatile void>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<void *>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<void *&>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<void()>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<void(*)()>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<void *Empty::*>()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<bslmf::IsVoid<void> >()));
        ASSERT(dispatchMeta0Type(bslmf::IsVoid<Incomplete>()));

        ASSERT(dispatchTrueType( bsl::is_void<void>()));
        ASSERT(dispatchTrueType( bsl::is_void<const void>()));
        ASSERT(dispatchTrueType( bsl::is_void<volatile void>()));
        ASSERT(dispatchTrueType( bsl::is_void<const volatile void>()));
        ASSERT(dispatchFalseType(bsl::is_void<void *>()));
        ASSERT(dispatchFalseType(bsl::is_void<void *&>()));
        ASSERT(dispatchFalseType(bsl::is_void<void()>()));
        ASSERT(dispatchFalseType(bsl::is_void<void(*)()>()));
        ASSERT(dispatchFalseType(bsl::is_void<void *Empty::*>()));
        ASSERT(dispatchFalseType(bsl::is_void<bsl::is_void<void> >()));
        ASSERT(dispatchFalseType(bsl::is_void<Incomplete>()));

        // Basic test dataset for 'bool' conversion
        // Note that this test is C++03 specific, and should fail to compile
        // with C++11 where such conversions will be 'explicit'.  This set of
        // tests should check use in an 'if' statement instead.
        ASSERT( bslmf::IsVoid<void>());
        ASSERT( bslmf::IsVoid<const void>());
        ASSERT( bslmf::IsVoid<volatile void>());
        ASSERT( bslmf::IsVoid<const volatile void>());
        ASSERT(!bslmf::IsVoid<void *>());
        ASSERT(!bslmf::IsVoid<void *&>());
        ASSERT(!bslmf::IsVoid<void()>());
        ASSERT(!bslmf::IsVoid<void(*)()>());
        ASSERT(!bslmf::IsVoid<void *Empty::*>());
        ASSERT(!bslmf::IsVoid<bslmf::IsVoid<void> >());
        ASSERT(!bslmf::IsVoid<Incomplete>());

        ASSERT( bsl::is_void<void>());
        ASSERT( bsl::is_void<const void>());
        ASSERT( bsl::is_void<volatile void>());
        ASSERT( bsl::is_void<const volatile void>());
        ASSERT(!bsl::is_void<void *>());
        ASSERT(!bsl::is_void<void *&>());
        ASSERT(!bsl::is_void<void()>());
        ASSERT(!bsl::is_void<void(*)()>());
        ASSERT(!bsl::is_void<void *Empty::*>());
        ASSERT(!bsl::is_void<bsl::is_void<void> >());
        ASSERT(!bsl::is_void<Incomplete>());
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING 'bslmf::IsVoid::type' and 'bsl::is_void::type'
        //
        // Concerns:
        //: 1 The meta-function 'bslmf::IsVoid' contains a nested type alias
        //:   named 'type'.
        //:
        //: 2 The alias 'type' is unambiguously either 'bsl::false_type' or
        //:   'bsl::true_type'.
        //:
        //: 3 The meta-function 'bsl::is_void' contains a nested type alias
        //:   named 'type'.
        //:
        //: 4 The alias 'type' is unambiguously either 'bsl::false_type' or
        //:   'bsl::true_type'.
        //
        // Plan:
        //: 1 Define two meta-functions that return 'true' either if the
        //:   (template parameter) type is 'bsl::true_type' or if the (template
        //:   parameter) type is 'bsl::false_type'.
        //:
        //: 2 For 'bslmf::IsVoid' instantiated on a variety of ('void' and
        //:   non-'void') types, verify the return value of the meta-function
        //:   defined in P-1 for the 'type' alias of each instantiation.
        //:   (C-1..2)
        //:
        //: 4 For 'bsl::is_void' instantiated on a variety of ('void' and
        //:   non-'void') types, verify the return value of the meta-function
        //:   defined in P-3 for the 'type' alias of each instantiation.
        //:   (C-3..4)
        //
        // Testing:
        //   bslmf::IsVoid::type
        //   bsl::is_void::type
        // --------------------------------------------------------------------

        if (verbose)
          printf("\nTESTING 'bslmf::IsVoid::type' and 'bsl::is_void::type'"
                 "\n======================================================\n");

        // 'bsl::IsVoid::type'
        ASSERT(isTrueType <bslmf::IsVoid<void>::type>());
        ASSERT(isTrueType <bslmf::IsVoid<const void>::type>());
        ASSERT(isTrueType <bslmf::IsVoid<volatile void>::type>());
        ASSERT(isTrueType <bslmf::IsVoid<const volatile void>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<void *>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<void *&>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<void()>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<void(*)()>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<void *Empty::*>::type>());
        ASSERT(isFalseType<bslmf::IsVoid<bslmf::IsVoid<void> >::type>());
        ASSERT(isFalseType<bslmf::IsVoid<Incomplete>::type>());

        // 'bsl::is_void::type'
        ASSERT(isTrueType <bsl::is_void<void>::type>());
        ASSERT(isTrueType <bsl::is_void<const void>::type>());
        ASSERT(isTrueType <bsl::is_void<volatile void>::type>());
        ASSERT(isTrueType <bsl::is_void<const volatile void>::type>());
        ASSERT(isFalseType<bsl::is_void<void *>::type>());
        ASSERT(isFalseType<bsl::is_void<void *&>::type>());
        ASSERT(isFalseType<bsl::is_void<void()>::type>());
        ASSERT(isFalseType<bsl::is_void<void(*)()>::type>());
        ASSERT(isFalseType<bsl::is_void<void *Empty::*>::type>());
        ASSERT(isFalseType<bsl::is_void<bsl::is_void<void> >::type>());
        ASSERT(isFalseType<bsl::is_void<Incomplete>::type>());

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // 'bslmf::IsVoid::value', 'bsl::is_void::value' and
        // 'bsl::is_void::value_v'
        //
        // Concerns:
        //: 1 The meta-functions return 'true' for any 'void' type, regardless
        //:   of its cv-qualification.
        //:
        //: 2 The meta-functions return 'false' for every other type.
        //:
        //: 3 The meta-functions return the correct result, even when the
        //:   potentially 'void' type it is diagnosing is a type alias, such as
        //:   a dependent type name in a template.
        //
        // Plan:
        //   Verify that both 'bsl::is_void::value' and 'bslmf::IsVoid::value'
        //   has the correct value for each concern.
        //
        // Testing:
        //   bslmf::IsVoid::value
        //   bsl::is_void::value
        // --------------------------------------------------------------------

        if (verbose)
            printf("\n'bslmf::IsVoid::value' and 'bsl::is_void::value'"
                   "\n================================================\n");

        // Basic test dataset
        ASSERT( bslmf::IsVoid<void>::value);
        ASSERT( bslmf::IsVoid<const void>::value);
        ASSERT( bslmf::IsVoid<volatile void>::value);
        ASSERT( bslmf::IsVoid<const volatile void>::value);
        ASSERT(!bslmf::IsVoid<void *>::value);
        ASSERT(!bslmf::IsVoid<void *&>::value);
        ASSERT(!bslmf::IsVoid<void()>::value);
        ASSERT(!bslmf::IsVoid<void(*)()>::value);
        ASSERT(!bslmf::IsVoid<void *Empty::*>::value);
        ASSERT(!bslmf::IsVoid<bslmf::IsVoid<void> >::value);
        ASSERT(!bslmf::IsVoid<Incomplete>::value);

        ASSERT( bsl::is_void<void>::value);
        ASSERT( bsl::is_void<const void>::value);
        ASSERT( bsl::is_void<volatile void>::value);
        ASSERT( bsl::is_void<const volatile void>::value);
        ASSERT(!bsl::is_void<void *>::value);
        ASSERT(!bsl::is_void<void *&>::value);
        ASSERT(!bsl::is_void<void()>::value);
        ASSERT(!bsl::is_void<void(*)()>::value);
        ASSERT(!bsl::is_void<void *Empty::*>::value);
        ASSERT(!bsl::is_void<bsl::is_void<void> >::value);
        ASSERT(!bsl::is_void<Incomplete>::value);

#ifdef BSLS_COMPILERFEATURES_SUPPORT_VARIABLE_TEMPLATES
        ASSERT( bsl::is_void_v<               void>);
        ASSERT( bsl::is_void_v<const          void>);
        ASSERT( bsl::is_void_v<      volatile void>);
        ASSERT( bsl::is_void_v<const volatile void>);
        ASSERT(!bsl::is_void_v<void *             >);
        ASSERT(!bsl::is_void_v<void *&            >);
        ASSERT(!bsl::is_void_v<void()             >);
        ASSERT(!bsl::is_void_v<void(*)()          >);
        ASSERT(!bsl::is_void_v<void *Empty::*     >);
        ASSERT(!bsl::is_void_v<void *(Empty::*)() >);
        ASSERT(!bsl::is_void_v<bsl::is_void<void> >);
        ASSERT(!bsl::is_void_v<Incomplete         >);
#endif

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //   Developers' Sandbox.
        //
        // Plan:
        //   Perform an ad-hoc test of the two meta-functions.
        //
        // Testing:
        //   This "test" *exercises* basic functionality, but *tests* nothing.
        // --------------------------------------------------------------------

        if (verbose) printf("\nBREATHING TEST"
                            "\n==============\n");

        ASSERT( bslmf::IsVoid<void>::value);
        ASSERT(!bslmf::IsVoid<int>::value);

        ASSERT( bsl::is_void<void>::value);
        ASSERT(!bsl::is_void<int>::value);

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
