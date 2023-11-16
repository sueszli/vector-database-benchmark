// bdlat_formattingmode.t.cpp                                         -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------


#include <bdlat_formattingmode.h>

#include <bslim_testutil.h>

#include <bslmf_assert.h>

#include <bsl_cstring.h>     // strlen()
#include <bsl_cstdlib.h>     // atoi()
#include <bsl_iostream.h>

using namespace BloombergLP;
using namespace bsl;  // automatically added by script

//=============================================================================
//                                 TEST PLAN
//-----------------------------------------------------------------------------
//                                 Overview
//                                 --------
// Due to the extremely simple nature of this component, we only have to test
// that the flags are defined and that they do not overlap in bit patterns.
//-----------------------------------------------------------------------------
//  bdlat_FormattingMode::DEFAULT
//  bdlat_FormattingMode::DEC
//  bdlat_FormattingMode::HEX
//  bdlat_FormattingMode::BASE64
//  bdlat_FormattingMode::TEXT
//  bdlat_FormattingMode::LIST
//  bdlat_FormattingMode::UNTAGGED
//  bdlat_FormattingMode::ATTRIBUTE
//  bdlat_FormattingMode::SIMPLE_CONTENT
//  bdlat_FormattingMode::NILLABLE
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

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
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

typedef bdlat_FormattingMode FM;

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test = argc > 1 ? atoi(argv[1]) : 0;
    int verbose = argc > 2;
    int veryVerbose = argc > 3;
//  int veryVeryVerbose = argc > 4;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;;

    switch (test) { case 0:  // Zero is always the leading case.
      case 2: {
        // --------------------------------------------------------------------
        // TEST ENUMERATORS
        //
        // Concerns
        //: 1 When BDE_OMIT_INTERNAL_DEPRECATED is not defined, the BDEAT_...
        //:   enumeration literals should exist and evaluate to their e_...
        //:   equivalents.
        //:
        //: 2 When BDE_OMIT_INTERNAL_DEPRECATED is defined, the BDEAT_...
        //:   enumeration literals should not exist.
        //
        // Plan
        //: 1 When BDE_OMIT_INTERNAL_DEPRECATED is not defined, check that the
        //:   BDEAT_... enumeration literals evaluate to their e_...
        //:   equivalents.  (C-1)
        //:
        //: 2 We cannot check for (C-2), so hope for the best.
        //
        // Testing:
        //   ENUMERATORS
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTEST ENUMERATORS"
                          << "\n================" << endl;

#undef e
#define e(Class, Enumerator)                                                  \
    { L_, #Class "::..._" #Enumerator,                                        \
      Class::e_##Enumerator, Class::BDEAT_##Enumerator }

        static struct {
            int         d_line;         // line number
            const char *d_name;         // printable enumerator name
            int         d_bdeat_value;  // value of BDEAT_... version
            int         d_bdlat_value;  // value of e_... version
        } DATA [] = {
#ifndef BDE_OMIT_INTERNAL_DEPRECATED
          e(bdlat_FormattingMode, DEFAULT),
          e(bdlat_FormattingMode, DEC),
          e(bdlat_FormattingMode, HEX),
          e(bdlat_FormattingMode, BASE64),
          e(bdlat_FormattingMode, TEXT),
          e(bdlat_FormattingMode, TYPE_MASK),
          e(bdlat_FormattingMode, UNTAGGED),
          e(bdlat_FormattingMode, ATTRIBUTE),
          e(bdlat_FormattingMode, SIMPLE_CONTENT),
          e(bdlat_FormattingMode, NILLABLE),
          e(bdlat_FormattingMode, LIST),
          e(bdlat_FormattingMode, FLAGS_MASK),
#endif
          { L_, "None", 0, 0 },
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        for (int i = 0; i < NUM_DATA; ++i) {
            const int   LINE   = DATA[i].d_line;
            const char *NAME   = DATA[i].d_name;
            const int   EVALUE = DATA[i].d_bdeat_value;
            const int   LVALUE = DATA[i].d_bdlat_value;

            if (veryVerbose) { P_(LINE) P_(NAME) P_(EVALUE) P(LVALUE) }

            ASSERTV(LINE, NAME, EVALUE, LVALUE, EVALUE == LVALUE);
        }
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // Basic Attribute Test:
        //
        // Concerns:
        //   - e_TYPE_MASK and e_FLAGS_MASK are disjoint (i.e., have no
        //     bits in common).
        //   - Each of the mode enumerations are unique.
        //   - All of the bits of each schema type enumeration are within the
        //     e_TYPE_MASK bit-mask.
        //   - All of the bits of formatting flag enumeration are within the
        //     e_FLAGS_MASK bit-mask.
        //   - None of the formatting flag enumerations share any bits in
        //     common with any of the other mode enumerations.
        //
        // Plan:
        //   - Test that e_TYPE_MASK and e_FLAGS_MASK have no bits in
        //     common.
        //   - Loop over each schema type enumeration and verify that it is
        //     within the TYPE_MASK bit-mask and does not overlap the
        //     e_FLAGS_MASK.
        //   - In a nested loop, test that each schema type enumeration is
        //     not equal to another schema type enumeration.
        //   - Loop over each formatting flag enumeration and verify that it is
        //     within the e_FLAGS_MASK bit-mask and does not overlap the
        //     e_TYPE_MASK.
        //   - In a nested loop, test that each formatting flag enumeration has
        //     no bits in common with another formatting flag enumeration.
        //
        // Testing:
        //    bdlat_FormattingMode::e_DEFAULT
        //    bdlat_FormattingMode::e_DEC
        //    bdlat_FormattingMode::e_HEX
        //    bdlat_FormattingMode::e_BASE64
        //    bdlat_FormattingMode::e_TEXT
        //    bdlat_FormattingMode::e_TYPE_MASK
        //    bdlat_FormattingMode::e_UNTAGGED
        //    bdlat_FormattingMode::e_ATTRIBUTE
        //    bdlat_FormattingMode::e_SIMPLE_CONTENT
        //    bdlat_FormattingMode::e_NILLABLE
        //    bdlat_FormattingMode::e_LIST
        //    bdlat_FormattingMode::e_FLAGS_MASK
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "Basic Attribute Test" << endl
                          << "====================" << endl;

        // Test that e_TYPE_MASK and e_FLAGS_MASK have no bits in
        // common.
        ASSERT(0 == (FM::e_TYPE_MASK & FM::e_FLAGS_MASK));

        if (veryVerbose) cout << "\tTesting type masks\n"
                              << "\t------------------" << endl;

        static const int TYPE_MODES[] = {
            FM::e_DEFAULT,
            FM::e_DEC,
            FM::e_HEX,
            FM::e_BASE64,
            FM::e_TEXT
        };

        static const int NUM_TYPE_MODES =
            sizeof(TYPE_MODES) / sizeof(TYPE_MODES[0]);

        // Loop over each schema type enumerations.
        for (int i = 0; i < NUM_TYPE_MODES; ++i) {
            const int TYPE_MODE1 = TYPE_MODES[i];

            // Verify that TYPE_MODE1 is within the TYPE_MASK bit-mask and
            // does not overlap the FLAGS_MASK.
            LOOP_ASSERT(TYPE_MODE1,
                        TYPE_MODE1 == (TYPE_MODE1 & FM::e_TYPE_MASK));
            LOOP_ASSERT(TYPE_MODE1,
                        0          == (TYPE_MODE1 & FM::e_FLAGS_MASK));

            for (int j = 0; j < NUM_TYPE_MODES; ++j) {
                if (j == i) continue;

                const int TYPE_MODE2 = TYPE_MODES[j];

                // test that each schema type enumeration is not equal to
                // another schema type enumeration.
                LOOP2_ASSERT(TYPE_MODE1, TYPE_MODE2, TYPE_MODE1 != TYPE_MODE2)
            }
        }

        if (veryVerbose) cout << "\tTesting flags masks\n"
                              << "\t-------------------" << endl;

        static const int FLAG_MODES[] = {
            FM::e_UNTAGGED,
            FM::e_ATTRIBUTE,
            FM::e_SIMPLE_CONTENT,
            FM::e_NILLABLE,
            FM::e_LIST
        };

        static const int NUM_FLAG_MODES =
            sizeof(FLAG_MODES) / sizeof(FLAG_MODES[0]);

        // Loop over each schema type enumeration.
        for (int i = 0; i < NUM_FLAG_MODES; ++i) {
            const int FLAG_MODE1 = FLAG_MODES[i];

            // Verify that FLAG_MODE1 is within the 'e_FLAGS_MASK' bit-mask
            // and does not overlap the 'e_TYPE_MASK'.
            LOOP_ASSERT(FLAG_MODE1,
                        FLAG_MODE1 == (FLAG_MODE1 & FM::e_FLAGS_MASK));
            LOOP_ASSERT(FLAG_MODE1,
                        0          == (FLAG_MODE1 & FM::e_TYPE_MASK));

            for (int j = 0; j < NUM_FLAG_MODES; ++j) {
                if (j == i) continue;

                const int FLAG_MODE2 = FLAG_MODES[j];

                // Test that each formatting flag enumeration has no bits in
                // common with another formatting flag enumeration.
                LOOP2_ASSERT(FLAG_MODE1, FLAG_MODE2,
                             0 == (FLAG_MODE1 & FLAG_MODE2));
            }
        }

      } break;

      default: {
        cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
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
