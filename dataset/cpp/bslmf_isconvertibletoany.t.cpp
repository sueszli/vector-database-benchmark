// bslmf_isconvertibletoany.t.cpp                                     -*-C++-*-
#include <bslmf_isconvertibletoany.h>

#include <bsls_bsltestutil.h>

#include <stdio.h>   // 'printf'
#include <stdlib.h>  // 'atoi'

using namespace BloombergLP;

//=============================================================================
//                                TEST PLAN
//-----------------------------------------------------------------------------
//                                Overview
//                                --------
// The component under test defines a meta-function,
// 'bslmf::IsConvertibleToAny', that determines whether a template parameter
// type can be converted to any type.  Thus, we need to ensure that the value
// returned by this meta-function is correct for the type with a template
// conversion operator.
//
//-----------------------------------------------------------------------------
// PUBLIC CLASS DATA
// [ 1] bslmf::IsEnum::value
//
// ----------------------------------------------------------------------------
// [ 2] USAGE EXAMPLE

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

namespace {

struct ConvertibleToSome {
    operator bool() { return false; }
    operator int() { return 1; }
    operator bsl::true_type() { return bsl::true_type(); }
};

struct ConvertibleToAny {
    template <class T>
    operator T() { return T(); }
};

struct ConvertibleToAnyConst {
    template <class T>
    operator T() const { return T(); }
};

}  // close unnamed namespace

//=============================================================================
//                              USAGE EXAMPLES
//-----------------------------------------------------------------------------

///Usage
//------
// In this section we show the intended use of this component.
//
///Example 1: Determine If a Class Has a Template Conversion Operator
///- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Suppose that we want to assert whether a particular type has a template
// conversion operator.
//
// First, we define a type with the template conversion operator:
//..
    struct TypeWithTemplateConversion {
        template <class TYPE>
        operator TYPE() {
            return TYPE();
        }
    };
//..

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

    switch (test) { case 0:  // Zero is always the leading case.
      case 2: {
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

// Now, we instantiate the 'bslmf::IsConvertibleToAny' template for
// 'TypeWithTemplateConversion' and assert the 'value' of the instantiation:
//..
    ASSERT(bslmf::IsConvertibleToAny<TypeWithTemplateConversion>::value);
//..

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // 'bslmf::IsConvertibleToAny::value'
        //   Ensure that the static data member 'value' of
        //   'bslmf::IsConvertibleToAny' instantiation with the (template
        //   parameter) 'TYPE' having a template conversion operator has a
        //   correct value.
        //
        // Concerns:
        //: 1 'bslmf::IsConvertibleToAny::value' is 'false' when 'TYPE' is a
        //:   primitive type.
        //:
        //: 2 'bslmf::IsConvertibleToAny::value' is 'false' when 'TYPE' doesn't
        //:   have a conversion operator that allows it to be converted to any
        //:   type, but might have any other conversion operators.
        //:
        //: 3 'bslmf::IsConvertibleToAny::value' is 'true' when 'TYPE'
        //:   has a conversion operator that allows it to be converted to any
        //:   type.
        //
        // Plan:
        //   Verify that 'bslmf::IsConvertibleToAny::value' has the correct
        //   value for each (template parameter) 'TYPE' in the concerns.
        //   (C-1..3)
        //
        // Testing:
        //   bslmf::IsConvertibleToAny::value
        // --------------------------------------------------------------------

        if (verbose) printf("bslmf::IsConvertibleToAny::value\n"
                            "================================\n");

        ASSERT(!bslmf::IsConvertibleToAny<int>::value);
        ASSERT(!bslmf::IsConvertibleToAny<ConvertibleToSome>::value);
        ASSERT(bslmf::IsConvertibleToAny<ConvertibleToAny>::value);
        ASSERT(bslmf::IsConvertibleToAny<ConvertibleToAnyConst>::value);

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
