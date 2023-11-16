// bdlde_charconvertstatus.t.cpp                                      -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------


#include <bdlde_charconvertstatus.h>

#include <bslma_default.h>
#include <bslma_testallocator.h>

#include <bsls_types.h>

#include <bsl_cstdlib.h>       // 'atoi'
#include <bsl_cstring.h>       // 'strcmp', 'memcmp', 'memcpy'
#include <bsl_ios.h>
#include <bsl_iostream.h>
#include <bsl_sstream.h>

using namespace BloombergLP;
using bsl::cout;
using bsl::cerr;
using bsl::flush;
using bsl::endl;
using bsl::ends;
using bsl::size_t;

// ============================================================================
//                                  TEST PLAN
// ----------------------------------------------------------------------------
//                                  Overview
//                                  --------
// The component under test implements a single enumeration having enumerator
// values that specify bits within an integer status to be returned by
// translation functions.
//
// We will therefore follow our standard 3-step approach to testing enumeration
// types, with certain test cases omitted:
//: o [ 4] -- BDEX streaming is not (yet) implemented for this type.
//
// Global Concerns:
//: o No methods or free operators allocate memory.
//
// Global Assumptions:
//: o All CLASS METHODS and the '<<' free operator are 'const' thread-safe.
// ----------------------------------------------------------------------------
// TYPES
// [ 1] enum Enum { ... };
//
// CLASS METHODS
// [ 3] ostream& print(ostream& s, Enum val, int level = 0, int sPL = 4);
// [ 1] const char *toAscii(bdlde::CharConvertStatus::Enum val);
//
// FREE OPERATORS
// [ 2] operator<<(ostream& s, bdlde::CharConvertStatus::Enum val);
// ----------------------------------------------------------------------------
// [ 5] USAGE EXAMPLE

// ============================================================================
//                      STANDARD BDE ASSERT TEST MACROS
// ----------------------------------------------------------------------------

static int testStatus = 0;

static void aSsErT(int c, const char *s, int i)
{
    if (c) {
        cout << "Error " << __FILE__ << "(" << i << "): " << s
             << "    (failed)" << endl;
        if (testStatus >= 0 && testStatus <= 100) ++testStatus;
    }
}
#define ASSERT(X) { aSsErT(!(X), #X, __LINE__); }

// ============================================================================
//                   STANDARD BDE LOOP-ASSERT TEST MACROS
// ----------------------------------------------------------------------------

#define LOOP_ASSERT(I,X) { \
    if (!(X)) { cout << #I << ": " << I << "\n"; aSsErT(1, #X, __LINE__);}}

#define LOOP2_ASSERT(I,J,X) { \
    if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " \
              << J << "\n"; aSsErT(1, #X, __LINE__); } }

#define LOOP3_ASSERT(I,J,K,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" \
              << #K << ": " << K << "\n"; aSsErT(1, #X, __LINE__); } }

#define LOOP4_ASSERT(I,J,K,L,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" << \
       #K << ": " << K << "\t" << #L << ": " << L << "\n"; \
       aSsErT(1, #X, __LINE__); } }

#define LOOP5_ASSERT(I,J,K,L,M,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" << \
       #K << ": " << K << "\t" << #L << ": " << L << "\t" << \
       #M << ": " << M << "\n"; \
       aSsErT(1, #X, __LINE__); } }

#define LOOP6_ASSERT(I,J,K,L,M,N,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" << \
       #K << ": " << K << "\t" << #L << ": " << L << "\t" << \
       #M << ": " << M << "\t" << #N << ": " << N << "\n"; \
       aSsErT(1, #X, __LINE__); } }

// ============================================================================
//                     SEMI-STANDARD TEST OUTPUT MACROS
// ----------------------------------------------------------------------------

#define P(X) cout << #X " = " << (X) << endl; // Print identifier and value.
#define Q(X) cout << "<| " #X " |>" << endl;  // Quote identifier literally.
#define P_(X) cout << #X " = " << (X) << ", " << flush; // P(X) without '\n'
#define T_ cout << "\t" << flush;             // Print tab w/o newline.
#define L_ __LINE__                           // current Line number

// ============================================================================
//                        GLOBAL TYPEDEFS FOR TESTING
// ----------------------------------------------------------------------------

typedef bdlde::CharConvertStatus::Enum Enum;
typedef bdlde::CharConvertStatus       Obj;

// ============================================================================
//                       GLOBAL CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

const int NUM_ENUMERATORS = 2;

#define UNKNOWN_FORMAT "(* UNKNOWN *)"

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int             test = argc > 1 ? bsl::atoi(argv[1]) : 0;
    bool         verbose = argc > 2;
    bool     veryVerbose = argc > 3;
    bool veryVeryVerbose = argc > 4;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    bslma::TestAllocator defaultAllocator(veryVeryVerbose);
    ASSERT(0 == bslma::Default::setDefaultAllocator(&defaultAllocator));

    bslma::TestAllocator globalAllocator(veryVeryVerbose);
    bslma::Default::setGlobalAllocator(&globalAllocator);

    switch (test) { case 0:  // Zero is always the leading case.
      case 5: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //
        // Concerns:
        //: 1 The usage example provided in the component header file must
        //:   compile, link, and run as shown.  (P-1)
        //
        // Plan:
        //: 1 Incorporate usage example from header into test driver, replace
        //:   leading comment characters with spaces, replace 'assert' with
        //:   'ASSERT', and insert 'if (veryVerbose)' before all output
        //:   operations.  (C-1)
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------

        if (verbose) cout << endl << "Testing Usage Examples" << endl
                                  << "======================" << endl;

///Usage
///-----
// In this section we show intended usage of this component.
//
///Example 1: Basic Syntax
///- - - - - - - - - - - -
// The following snippets of code provide a simple illustration of
// 'bdlde::CharConvertStatus' usage.
//
// First, we create a variable 'value' of type 'bdlde::CharConvertStatus::Enum'
// and initialize it with the value 3, which is not a valid value of the
// 'enum'.
//..
        bdlde::CharConvertStatus::Enum value =
                                 bdlde::CharConvertStatus::k_INVALID_INPUT_BIT;
//..
// Next, we store a pointer to its ASCII representation in a variable
// 'asciiValue' of type 'const char *':
//..
        const char *asciiValue = bdlde::CharConvertStatus::toAscii(value);
        ASSERT(0 == bsl::strcmp(asciiValue, "INVALID_INPUT_BIT"));
//..
// Finally, we print 'value' to 'bsl::cout'.
//..
        if (veryVerbose) {
            bsl::cout << value << bsl::endl;
        }
//..
// This statement produces the following output on 'stdout':
//..
// INVALID_INPUT_BIT
//..

      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING 'bdex' STREAMING
        //   'bdex' streaming is not yet implemented for this enumeration.
        //
        // Concerns:
        //   Not applicable.
        //
        // Plan:
        //   Not applicable.
        //
        // Testing:
        //   Not applicable.
        // --------------------------------------------------------------------

      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING 'print'
        //
        // Concerns:
        //: 1 The 'print' method writes the output to the specified stream.
        //:   (P-1)
        //: 2 The 'print' method writes the string representation of each
        //:   enumerator in the intended format.  (P-1)
        //: 3 The 'print' method writes a distinguished string when passed an
        //:   out-of-band value.  (P-2)
        //: 4 There is no output when the stream is invalid.  (P-3)
        //: 5 The 'print' method has the expected signature.  (P-4)
        //
        // Plan:
        //: 1 Verify that the 'print' method produces the expected results for
        //:   each enumerator.  (C-1, C-2)
        //: 2 Verify that the 'print' method writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //: 3 Verify that there is no output when the stream is invalid.  (C-4)
        //: 4 Take the address of the 'print' (class) method and use the
        //:   result to initialize a variable of the appropriate type.  (C-5)
        //
        // Testing:
        //   ostream& print(ostream& s, Enum val, int level = 0, int sPL = 4);
        // --------------------------------------------------------------------

        if (verbose) cout << endl << "Testing 'print'" << endl
                                  << "===============" << endl;

        static const struct {
            int         d_lineNum;  // source line number
            int         d_level;    // level
            int         d_spl;      // spaces per level
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
#define NL "\n"
#define ICM "INVALID_INPUT_BIT"

            //line level spl    enumerator value           expected result
            //---- ----- --- ----------------------       -----------------
            { L_,    0,   4, Obj::k_INVALID_INPUT_BIT, "INVALID_INPUT_BIT\n" },
            { L_,    0,   4, Obj::k_OUT_OF_SPACE_BIT,  "OUT_OF_SPACE_BIT\n" },

            { L_,    0,   4, NUM_ENUMERATORS + 1,         UNKNOWN_FORMAT NL },
            { L_,    0,   4, -1,                          UNKNOWN_FORMAT NL },
            { L_,    0,   4, -5,                          UNKNOWN_FORMAT NL },
            { L_,    0,   4, 99,                          UNKNOWN_FORMAT NL },

            { L_,    0,  -1, Obj::k_INVALID_INPUT_BIT,    ICM },
            { L_,    0,   0, Obj::k_INVALID_INPUT_BIT,    ICM NL },
            { L_,    0,   2, Obj::k_INVALID_INPUT_BIT,    ICM NL },
            { L_,    1,   1, Obj::k_INVALID_INPUT_BIT,    " " ICM NL },
            { L_,    1,   2, Obj::k_INVALID_INPUT_BIT,    "  " ICM NL },
            { L_,   -1,   2, Obj::k_INVALID_INPUT_BIT,    ICM NL },
            { L_,   -2,   1, Obj::k_INVALID_INPUT_BIT,    ICM NL },
            { L_,    2,   1, Obj::k_INVALID_INPUT_BIT,    "  " ICM NL },
            { L_,    1,   3, Obj::k_INVALID_INPUT_BIT,    "   " ICM NL },
#undef ICM
#undef NL
        };
        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        enum { SIZE = 128 };            // big enough to hold output string
        const char  XX   = (char)0xFF;  // value of an unset 'char'

              char  mCtrl[SIZE];  memset(mCtrl, XX, SIZE);
        const char *CTRL = mCtrl;

        if (verbose) cout << "\nTesting 'print'." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            bsl::ostringstream out(std::string(CTRL, SIZE));
            Obj::print(out, VALUE, LEVEL, SPL) << ends;

            if (veryVerbose) cout << "  ACTUAL FORMAT: " << out.str() << endl;

            const size_t SZ = strlen(EXP) + 1;
            LOOP2_ASSERT(LINE, ti, SZ  < SIZE);           // Buffer is large
                                                          // enough.
            LOOP2_ASSERT(LINE,
                         ti,
                         XX == out.str()[SIZE - 1]);      // Check for overrun.
            LOOP3_ASSERT(LINE,
                         ti,
                         out.str(),
                         0 == memcmp(out.str().c_str(), EXP, SZ));
            LOOP2_ASSERT(LINE, ti,  0 == memcmp(out.str().c_str() + SZ,
                                                CTRL + SZ, SIZE - SZ));

            if (0 == LEVEL && 4 == SPL) {
                if (veryVerbose)
                    cout << "\tRepeat for 'print' default arguments." << endl;

                bsl::ostringstream out(std::string(CTRL, SIZE));
                Obj::print(out, VALUE) << ends;

                if (veryVerbose) {
                    cout << "  ACTUAL FORMAT: " << out.str() << endl;
                }

                LOOP2_ASSERT(LINE, ti, XX == out.str()[SIZE - 1]); // Check for
                                                                   // overrun.
                LOOP2_ASSERT(LINE,
                             ti,
                             0 == memcmp(out.str().c_str(), EXP, SZ));
                LOOP2_ASSERT(LINE, ti,  0 == memcmp(out.str().c_str() + SZ,
                                                    CTRL + SZ, SIZE - SZ));
            }
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            // const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            bsl::ostringstream out(bsl::string(CTRL, SIZE));
            out.setstate(bsl::ios::badbit);
            Obj::print(out, VALUE, LEVEL, SPL);

            const bsl::string ctrlStr(CTRL, SIZE);

            LOOP2_ASSERT(LINE, ti, out.str() == ctrlStr);
        }

        if (verbose) cout << "\nVerify 'print' signature." << endl;

        {
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum, int, int);

            const FuncPtr FP = &Obj::print;
            if (veryVerbose) (*FP)(cout, Obj::k_INVALID_INPUT_BIT, 0, 0);
        }

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING OUTPUT ('<<') OPERATOR
        //
        // Concerns:
        //: 1 The '<<' operator writes the output to the specified stream.
        //:   (P-1)
        //: 2 The '<<' operator writes the string representation of each
        //:   enumerator in the intended format.  (P-1)
        //: 3 The '<<' operator writes a distinguished string when passed an
        //:   out-of-band value.  (P-2)
        //: 4 The output produced by 'stream << value' is the same as that
        //:   produced by 'Obj::print(stream, value, 0, -1)'.  (P-3)
        //: 5 There is no output when the stream is invalid.  (P-4)
        //: 6 The '<<' operator has the expected signature.  (P-5)
        //
        // Plan:
        //: 1 Verify that the '<<' operator produces the expected results for
        //:   each enumerator.  (C-1, C-2)
        //: 2 Verify that the '<<' operator writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //: 3 Verify that 'stream << value' writes the same output as
        //:   'Obj::print(stream, value, 0, -1)'.  (C-4)
        //: 4 Verify that there is no output when the stream is invalid.  (C-5)
        //: 5 Take the address of the '<<' (free) operator and use the result
        //:   to initialize a variable of the appropriate type.  (C-6)
        //
        // Testing:
        //   operator<<(ostream& s, bdlde::CharConvertStatus::Enum val);
        // --------------------------------------------------------------------

        if (verbose) cout << endl << "Testing '<<' operator" << endl
                                  << "=====================" << endl;

        static const struct {
            int         d_lineNum;  // source line number
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
            //line       enumerator value              expected result
            //----    ----------------------          -----------------
            { L_,     Obj::k_INVALID_INPUT_BIT,       "INVALID_INPUT_BIT" },
            { L_,     Obj::k_OUT_OF_SPACE_BIT,        "OUT_OF_SPACE_BIT"  },

            { L_,     NUM_ENUMERATORS + 1,            UNKNOWN_FORMAT      },
            { L_,     -1,                             UNKNOWN_FORMAT      },
            { L_,     -5,                             UNKNOWN_FORMAT      },
            { L_,     99,                             UNKNOWN_FORMAT      },
        };
        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        const int   SIZE = 128;         // big enough to hold output string
        const char  XX   = (char)0xFF;  // value of an unset 'char'

              char  mCtrl[SIZE];  memset(mCtrl, XX, SIZE);
        const char *CTRL = mCtrl;

        if (verbose) cout << "\nTesting '<<' operator." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            bsl::ostringstream out(bsl::string(CTRL, SIZE));
            out << VALUE << ends;

            if (veryVerbose) cout << "  ACTUAL FORMAT: " << out.str() << endl;

            const size_t SZ = strlen(EXP) + 1;
            LOOP2_ASSERT(LINE, ti, SZ  < SIZE);           // Buffer is large
                                                          // enough.
            LOOP2_ASSERT(LINE,
                         ti,
                         XX == out.str()[SIZE - 1]);      // Check for overrun.
            LOOP2_ASSERT(LINE, ti,  0 == memcmp(out.str().c_str(), EXP, SZ));
            LOOP2_ASSERT(LINE, ti,  0 == memcmp(out.str().c_str() + SZ,
                                                CTRL + SZ, SIZE - SZ));
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            // const char *EXP   = DATA[ti].d_exp;

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            bsl::ostringstream out(bsl::string(CTRL, SIZE));
            out.setstate(bsl::ios::badbit);
            out << VALUE;

            const bsl::string ctrlStr(CTRL, SIZE);

            LOOP2_ASSERT(LINE, ti, out.str() == ctrlStr);
        }

        if (verbose) cout << "\nVerify '<<' operator signature." << endl;

        {
            using namespace bdlde;
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum);

            const FuncPtr FP = &operator<<;

            if (veryVerbose) (*FP)(cout, Obj::k_INVALID_INPUT_BIT);
        }
      } break;
      case 1: {
        // -------------------------------------------------------------------
        // TESTING 'enum' AND 'toAscii'
        //
        // Concerns:
        //: 1 The enumerator values are sequential, starting from 0.  (P-1)
        //: 2 The 'toAscii' method returns the expected string representation
        //:   for each enumerator.  (P-2)
        //: 3 The 'toAscii' method returns a distinguished string when passed
        //:   an out-of-band value.  (P-3)
        //: 4 The string returned by 'toAscii' is non-modifiable.  (P-4)
        //: 5 The 'toAscii' method has the expected signature.  (P-4)
        //
        // Plan:
        //: 1 Verify that the enumerator values are sequential, starting from
        //:   0.  (C-1)
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
        //   const char *toAscii(bdlde::CharConvertStatus::Enum val);
        // -------------------------------------------------------------------

        if (verbose) cout << endl << "Testing 'enum' and 'toAscii'" << endl
                                  << "============================" << endl;

        static const struct {
            int         d_lineNum;  // source line number
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
            // line         enumerator value            expected result
            // ----    ---------------------------     -----------------
            {  L_,     Obj::k_INVALID_INPUT_BIT,       "INVALID_INPUT_BIT" },
            {  L_,     Obj::k_OUT_OF_SPACE_BIT,        "OUT_OF_SPACE_BIT"  },

            {  L_,     NUM_ENUMERATORS + 1,            UNKNOWN_FORMAT     },
            {  L_,     -1,                             UNKNOWN_FORMAT     },
            {  L_,     -5,                             UNKNOWN_FORMAT     },
            {  L_,     99,                             UNKNOWN_FORMAT     }
        };
        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        bsls::Types::Int64 numBlocksTotal = defaultAllocator.numBlocksTotal();

        if (verbose) cout << "\nVerify enumerator values are sequential."
                          << endl;

        for (int ti = 0; ti < NUM_ENUMERATORS; ++ti) {
            const Enum VALUE = static_cast<Enum>(DATA[ti].d_value);

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            LOOP_ASSERT(ti, ti + 1 == VALUE);
        }

        if (verbose) cout << "\nTesting 'toAscii'." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            const char *result = Obj::toAscii(VALUE);

            if (veryVerbose) { T_; P_(ti); P_(VALUE); P_(EXP); P(result); }

            LOOP2_ASSERT(LINE, ti, strlen(EXP) == strlen(result));
            LOOP2_ASSERT(LINE, ti,           0 == strcmp(EXP, result));
        }

        if (verbose) cout << "\nVerify 'toAscii' signature." << endl;

        {
            typedef const char *(*FuncPtr)(Enum);

            const FuncPtr FP = &Obj::toAscii;

            (*FP)(Obj::k_INVALID_INPUT_BIT);
        }

        ASSERT(numBlocksTotal == defaultAllocator.numBlocksTotal());
      } break;
      default: {
        cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    ASSERT(0 ==  globalAllocator.numBlocksTotal());

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
    }
    return testStatus;
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
