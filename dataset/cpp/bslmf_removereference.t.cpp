// bslmf_removereference.t.cpp                                        -*-C++-*-
#include <bslmf_removereference.h>

#include <bslmf_issame.h>  // for testing only

#include <bsls_bsltestutil.h>
#include <bsls_libraryfeatures.h>
#include <bsls_platform.h>

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY
#include <type_traits>  // 'std::remove_reference', 'std::remove_reference_t'
#endif

#include <stdio.h>   // 'printf'
#include <stdlib.h>  // 'atoi'

using namespace BloombergLP;

//=============================================================================
//                                TEST PLAN
//-----------------------------------------------------------------------------
//                                Overview
//                                --------
// The component under test defines two meta-functions, 'bsl::remove_reference'
// and 'bslmf::RemoveReference', both of which remove the reference-ness of the
// (template parameter) 'TYPE'.   Also the component defines an alias to the
// result type of the 'bsl::remove_reference' meta-function.  Thus, we need to
// ensure that the values returned by these meta-functions are correct for each
// possible category of types.
//
// 'bsl::remove_reference' and 'bsl::remove_reference_t' should be aliased to
// 'std::remove_reference' and 'std::remove_reference_t', respectively when the
// standard meta functions are available from the native library.
//
// ----------------------------------------------------------------------------
// PUBLIC TYPES
// [ 1] bsl::remove_reference::type
// [ 1] bsl::remove_reference_t
// [ 2] bslmf::RemoveReference::Type
//
// ----------------------------------------------------------------------------
// [ 4] USAGE EXAMPLE
// [ 3] CONCERN: Aliased to standard types when available.

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
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

enum   Enum   {};
struct Struct {};
union  Union  {};
class  Class  {};

typedef int INT;

typedef void        F ();
typedef void (  &  RF)();
typedef void (*    PF)();
typedef void (*&  RPF)();

typedef void      Fi  (int);
typedef void ( & RFi) (int);
typedef void      FRi (int&);
typedef void ( & RFRi)(int&);

typedef char      A [5];
typedef char ( & RA)[5];

#if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
typedef void ( && RRF) ();
typedef void (*&&RRPF) ();
typedef void ( &&RRFi) (int);
typedef void ( &&RRFRi)(int&);

typedef char ( &&RRA)[5];
#endif

#define ASSERT_SAME(X, Y)                                                     \
    ASSERT((bsl::is_same<bsl::remove_reference<X>::type, Y>::value));

#define ASSERT_REMOVE_REF(X, Y)                                               \
    ASSERT_SAME(X, Y)                                                         \
    ASSERT_SAME(const X, const Y)                                             \
    ASSERT_SAME(volatile X, volatile Y)                                       \
    ASSERT_SAME(const volatile X, const volatile Y)

#define ASSERT_SAME2(X, Y)                                                    \
    ASSERT((bsl::is_same<bslmf::RemoveReference<X>::Type, Y>::value));

#define ASSERT_REMOVE_REF2(X, Y)                                              \
    ASSERT_SAME2(X, Y)                                                        \
    ASSERT_SAME2(const X, const Y)                                            \
    ASSERT_SAME2(volatile X, volatile Y)                                      \
    ASSERT_SAME2(const volatile X, const volatile Y)

#define ASSERT_SAME3(X, Y)                                                    \
    ASSERT((bsl::is_same<bsl::remove_reference  <X>::type,                    \
                         bsl::remove_reference_t<Y> >::value));

#define ASSERT_REMOVE_REF_T(X, Y)                                             \
    ASSERT_SAME3(X, Y)                                                        \
    ASSERT_SAME3(const X, const Y)                                            \
    ASSERT_SAME3(volatile X, volatile Y)                                      \
    ASSERT_SAME3(const volatile X, const volatile Y)

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

    printf("TEST " __FILE__ " CASE %d\n", test);

    switch (test) { case 0:
      case 4: {
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
///Example 1: Remove Types' Reference-ness
///- - - - - - - - - - - - - - - - - - - -
// Suppose that we want to remove reference-ness on a set of types.
//
// Now, for a set of types, we remove the reference-ness of each type using
// 'bsl::remove_reference' and verify the result:
//..
    ASSERT(true  ==
               (bsl::is_same<bsl::remove_reference<int&>::type, int >::value));
    ASSERT(false ==
               (bsl::is_same<bsl::remove_reference<int&>::type, int&>::value));
    ASSERT(true  ==
               (bsl::is_same<bsl::remove_reference<int >::type, int >::value));
  #if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
    ASSERT(true ==
              (bsl::is_same<bsl::remove_reference<int&&>::type, int >::value));
  #endif
//..
// Note that rvalue reference is a feature introduced in the C++11 standard and
// may not be supported by all compilers.
//
// Finally, if the current compiler supports alias templates C++11 feature, we
// remove reference-ness from a set of types using 'bsl::remove_reference_t'
// and verify that the resulting type has any reference-ness removed:
//..
#ifdef BSLS_COMPILERFEATURES_SUPPORT_ALIAS_TEMPLATES
    ASSERT(true  ==
                  (bsl::is_same<bsl::remove_reference_t<int& >, int >::value));
    ASSERT(false ==
                  (bsl::is_same<bsl::remove_reference_t<int& >, int&>::value));
    ASSERT(true  ==
                  (bsl::is_same<bsl::remove_reference_t<int  >, int >::value));
#if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
    ASSERT(true ==
                  (bsl::is_same<bsl::remove_reference_t<int&&>, int >::value));
#endif
#endif
//..

      } break;
      case 3: {
        // --------------------------------------------------------------------
        // ALIASED TO STANDARD META FUNCTIONS
        //
        // Concerns:
        //: 1 The meta functions 'bsl::remove_reference' and
        //:   'bsl::remove_reference_v' should be aliased to their standard
        //:   library analogs when the latter is available from the native
        //:   library.
        //
        // Plan:
        //: 1 When 'BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY' is
        //:   defined, use 'bsl::is_same' to compare 'bsl::remove_reference' to
        //:   'std::remove_reference' using a representative type.
        //:
        //: 2 When 'BSLS_LIBRARYFEATURES_HAS_CPP14_BASELINE_LIBRARY' is
        //:   defined, use 'bsl::is_same' to compare 'bsl::remove_reference_t'
        //:   to 'std::remove_reference_t' using a representative type.
        //
        // Testing:
        //   CONCERN: Aliased to standard types when available.
        // --------------------------------------------------------------------

        if (verbose) printf("\nALIASED TO STANDARD META FUNCTIONS"
                            "\n==================================\n");

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY
        typedef int T;

        if (veryVerbose) printf(
                          "\nTesting 'remove_reference' alias using 'int'.\n");

        ASSERT((bsl::is_same<bsl::remove_reference<T>,
                             std::remove_reference<T> >::value));
#endif
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP14_BASELINE_LIBRARY
        if (veryVerbose) printf(
                        "\nTesting 'remove_reference_t' alias using 'int'.\n");

        ASSERT((bsl::is_same<bsl::remove_reference_t<T>,
                             std::remove_reference_t<T> >::value));
#endif
#if !defined BSLS_LIBRARYFEATURES_HAS_CPP11_BASELINE_LIBRARY                  \
 && !defined BSLS_LIBRARYFEATURES_HAS_CPP14_BASELINE_LIBRARY
        if (veryVerbose) printf(
                            "\nSkipped: neither standard type is available\n");
#endif
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING 'bslmf::RemoveReference::Type'
        //   Ensure that the 'typedef' 'Type' of 'bslmf::RemoveReference'
        //   instantiations having various (template parameter) 'TYPE's has the
        //   correct value.
        //
        // Concerns:
        //: 1 'RemoveReference' correctly removes reference-ness from 'TYPE' if
        //:   'TYPE' is an (lvalue or rvalue) reference type.
        //:
        //: 2 'RemoveReference' does not transform 'TYPE' when 'TYPE' is not a
        //:   reference type.
        //
        // Plan:
        //: 1 Instantiate 'bslmf::RemoveReference' with various types and
        //:   verify that the 'Type' member is initialized properly.  (C-1)
        //
        // Testing:
        //   bslmf::RemoveReference::Type
        // --------------------------------------------------------------------

        if (verbose) printf("TESTING 'bslmf::RemoveReference::Type'\n"
                            "======================================\n");

        // C-1

        ASSERT_REMOVE_REF2(int&,    int);
        ASSERT_REMOVE_REF2(void*&,  void*);
        ASSERT_REMOVE_REF2(Enum&,   Enum);
        ASSERT_REMOVE_REF2(Struct&, Struct);
        ASSERT_REMOVE_REF2(Union&,  Union);
        ASSERT_REMOVE_REF2(Class&,  Class);

        ASSERT_REMOVE_REF2(int Class::*&, int Class::*);

        ASSERT_SAME2(   F,   F);
        ASSERT_SAME2(  PF,  PF);
        ASSERT_SAME2( RPF,  PF);

        ASSERT_SAME2(  Fi,  Fi);
        ASSERT_SAME2( FRi, FRi);

        ASSERT_SAME2(  RF,   F);
        ASSERT_SAME2( RFi,  Fi);
        ASSERT_SAME2(RFRi, FRi);

        ASSERT_SAME2(RA, A);

        // C-2

        ASSERT_REMOVE_REF2(int,    int);
        ASSERT_REMOVE_REF2(void*,  void*);
        ASSERT_REMOVE_REF2(Enum,   Enum);
        ASSERT_REMOVE_REF2(Struct, Struct);
        ASSERT_REMOVE_REF2(Union,  Union);
        ASSERT_REMOVE_REF2(Class,  Class);

        ASSERT_REMOVE_REF2(int Class::*,  int Class::*);

        ASSERT_SAME2( F,  F);
        ASSERT_SAME2(PF, PF);

        ASSERT_SAME2( Fi,  Fi);
        ASSERT_SAME2(FRi, FRi);

        ASSERT_SAME2(A,  A);

#if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
        ASSERT_REMOVE_REF2(int&&,    int );
        ASSERT_REMOVE_REF2(void*&&,  void*);
        ASSERT_REMOVE_REF2(Struct&&, Struct);
        ASSERT_REMOVE_REF2(Union&&,  Union);
        ASSERT_REMOVE_REF2(Class&&,  Class);
        ASSERT_REMOVE_REF2(int Class::*&&, int Class::*);

        ASSERT_SAME2(RRF,   F);
        ASSERT_SAME2(RRPF,  PF);
        ASSERT_SAME2(RRFRi, FRi);

        ASSERT_SAME2(RRA, A);
#endif

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // TESTING 'bsl::remove_reference::type'
        //   Ensure that the 'typedef' 'type' of 'bsl::remove_reference'
        //   instantiations having various (template parameter) 'TYPE's has the
        //   correct value.
        //
        // Concerns:
        //: 1 'remove_reference' correctly removes reference-ness from 'TYPE'
        //:   if 'TYPE' is an (lvalue or rvalue) reference type.
        //:
        //: 2 'remove_reference' does not transform 'TYPE' when 'TYPE' is not a
        //:   reference type.
        //:
        //: 3 'remove_reference_t' represents the return type of
        //:   'remove_reference' meta-function for a variety of template
        //:   parameter types.
        //
        // Plan:
        //: 1 Instantiate 'bsl::remove_reference' with various types and
        //:   verify that the 'type' member is initialized properly.  (C-1,2)
        //:
        //: 2 Verify that 'bsl::remove_reference_t' has the same type as the
        //:   return type of 'bsl::remove_reference' for a variety of template
        //:   parameter types. (C-3)
        //
        // Testing:
        //   bsl::remove_reference::type
        //   bsl::remove_reference_t
        // --------------------------------------------------------------------

        if (verbose) printf("TESTING 'bsl::remove_reference::type'\n"
                            "=====================================\n");

        // C-1

        ASSERT_REMOVE_REF(int&,    int);
        ASSERT_REMOVE_REF(void*&,  void*);
        ASSERT_REMOVE_REF(Enum&,   Enum);
        ASSERT_REMOVE_REF(Struct&, Struct);
        ASSERT_REMOVE_REF(Union&,  Union);
        ASSERT_REMOVE_REF(Class&,  Class);

        ASSERT_REMOVE_REF(int Class::*&, int Class::*);

        ASSERT_SAME(   F,  F);
        ASSERT_SAME(  PF, PF);

        ASSERT_SAME(RA, A);

        // C-2

        ASSERT_REMOVE_REF(int,    int);
        ASSERT_REMOVE_REF(void*,  void*);
        ASSERT_REMOVE_REF(Enum,   Enum);
        ASSERT_REMOVE_REF(Struct, Struct);
        ASSERT_REMOVE_REF(Union,  Union);
        ASSERT_REMOVE_REF(Class,  Class);

        ASSERT_REMOVE_REF(int Class::*,  int Class::*);

        ASSERT_SAME(  F,  F);
        ASSERT_SAME( PF, PF);
        ASSERT_SAME(RPF, PF);

        ASSERT_SAME( Fi,  Fi);
        ASSERT_SAME(FRi, FRi);

        ASSERT_SAME( RF,  F);
        ASSERT_SAME( RFi, Fi);
        ASSERT_SAME(RFRi, FRi);

        ASSERT_SAME(A,  A);

  #if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
        ASSERT_REMOVE_REF(int&&,    int );
        ASSERT_REMOVE_REF(void*&&,  void*);
        ASSERT_REMOVE_REF(Enum&&,   Enum);
        ASSERT_REMOVE_REF(Struct&&, Struct);
        ASSERT_REMOVE_REF(Union&&,  Union);
        ASSERT_REMOVE_REF(Class&&,  Class);
        ASSERT_REMOVE_REF(int Class::*&&, int Class::*);

        ASSERT_SAME(RRF,   F);
        ASSERT_SAME(RRPF,  PF);
        ASSERT_SAME(RRFRi, FRi);

        ASSERT_SAME(RRA, A);
  #endif

#ifdef BSLS_COMPILERFEATURES_SUPPORT_ALIAS_TEMPLATES
        if (verbose) printf("TESTING 'bsl::remove_reference_t'\n"
                            "================================\n");

        // C-3
        ASSERT_REMOVE_REF_T(int&,          int);
        ASSERT_REMOVE_REF_T(void*&,        void*);
        ASSERT_REMOVE_REF_T(Enum&,         Enum);
        ASSERT_REMOVE_REF_T(Struct&,       Struct);
        ASSERT_REMOVE_REF_T(Union&,        Union);
        ASSERT_REMOVE_REF_T(Class&,        Class);
        ASSERT_REMOVE_REF_T(int Class::*&, int Class::*);

        ASSERT_REMOVE_REF_T(int,           int);
        ASSERT_REMOVE_REF_T(void*,         void*);
        ASSERT_REMOVE_REF_T(Enum,          Enum);
        ASSERT_REMOVE_REF_T(Struct,        Struct);
        ASSERT_REMOVE_REF_T(Union,         Union);
        ASSERT_REMOVE_REF_T(Class,         Class);
        ASSERT_REMOVE_REF_T(int Class::*,  int Class::*);

  #if defined(BSLS_COMPILERFEATURES_SUPPORT_RVALUE_REFERENCES)
        ASSERT_REMOVE_REF_T(int&&,          int&&);
        ASSERT_REMOVE_REF_T(void*&&,        void*&&);
        ASSERT_REMOVE_REF_T(Enum&&,         Enum&&);
        ASSERT_REMOVE_REF_T(Struct&&,       Struct&&);
        ASSERT_REMOVE_REF_T(Union&&,        Union&&);
        ASSERT_REMOVE_REF_T(Class&&,        Class&&);
        ASSERT_REMOVE_REF_T(int Class::*&&, int Class::*&&);
  #endif

#endif
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
