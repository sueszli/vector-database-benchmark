// bdlde_byteorder.t.cpp                                              -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------


#include <bdlde_byteorder.h>

#include <bslma_default.h>
#include <bslma_testallocator.h>

#include <bsl_cstdlib.h>       // 'atoi'
#include <bsl_cstring.h>       // 'strcmp', 'memcmp', 'memcpy'
#include <bsl_ios.h>
#include <bsl_iostream.h>
#include <bsl_sstream.h>
#include <bsl_string.h>

using namespace BloombergLP;
using namespace bsl;

// ============================================================================
//                                  TEST PLAN
// ----------------------------------------------------------------------------
//                                  Overview
//                                  --------
// The component under test implements a single enumeration having 4
// identifiers of two values, always 0 or 1.
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
// [ 2] ostream& print(ostream& s, Enum val, int level = 0, int sPL = 4);
// [ 1] const char *toAscii(bdlde::ByteOrder::Enum val);
//
// FREE OPERATORS
// [ 3] operator<<(ostream& s, bdlde::ByteOrder::Enum val);
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

typedef bdlde::ByteOrder::Enum Enum;
typedef bdlde::ByteOrder       Obj;

// ============================================================================
//                       GLOBAL CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

const int NUM_VALUES = 2;

#define UNKNOWN_FORMAT "(* UNKNOWN *)"

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int             test = argc > 1 ? atoi(argv[1]) : 0;
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
        //:   compile, link, and run as shown.
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

        if (verbose) cout << endl << "USAGE EXAMPLE" << endl
                                  << "=============" << endl;

///Usage
///-----
// In this section we show intended use of this component.
//
///Example 1: Basic Syntax
///- - - - - - - - - - - -
// The following snippets of code provide a simple illustration of
// 'bdlde::ByteOrder' usage.
//
// First, we create a variable 'value' of type 'bdlde::ByteOrder::Enum' and
// initialize it with the enumerator value
// 'bdlde::ByteOrder::e_LITTLE_ENDIAN':
//..
    bdlde::ByteOrder::Enum value = bdlde::ByteOrder::e_LITTLE_ENDIAN;
//..
// Next, we store a pointer to its ASCII representation in a variable
// 'asciiValue' of type 'const char *':
//..
    const char *asciiValue = bdlde::ByteOrder::toAscii(value);
    ASSERT(0 == bsl::strcmp(asciiValue, "LITTLE_ENDIAN"));
//..
// Then, we try one of the aliased identifiers, and we get a string
// corresponding to the value it is aliased to:
//..
    value      = bdlde::ByteOrder::e_NETWORK;
    asciiValue = bdlde::ByteOrder::toAscii(value);

    ASSERT(0 == bsl::strcmp(asciiValue, "BIG_ENDIAN"));
//..
// Finally, we print 'value' to 'bsl::cout':
//..
if (veryVerbose) {
    bsl::cout << value << bsl::endl;
}
//..
// This statement produces the following output on 'stdout':
//..
//  BIG_ENDIAN
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
        // TESTING OUTPUT ('<<') OPERATOR
        //
        // Concerns:
        //: 1 The '<<' operator writes the output to the specified stream.
        //:
        //: 2 The '<<' operator writes the string representation of each
        //:   enumerator in the intended format.
        //:
        //: 3 The '<<' operator writes a distinguished string when passed an
        //:   out-of-band value.
        //:
        //: 4 The output produced by 'stream << value' is the same as that
        //:   produced by 'Obj::print(stream, value, 0, -1)'.
        //:
        //: 5 There is no output when the stream is invalid.
        //:
        //: 6 The '<<' operator has the expected signature.
        //:
        //: 7 The '<<' operator returns a reference to the stream it was
        //:   passed.
        //
        // Plan:
        //: 1 Verify that the '<<' operator produces the expected results for
        //:   each enumerator.  (C-1, C-2)
        //:
        //: 2 Verify that the '<<' operator returns a reference to the stream
        //:   it was passed.
        //:
        //: 3 Verify that the '<<' operator writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //:
        //: 4 Verify that 'stream << value' writes the same output as
        //:   'Obj::print(stream, value, 0, -1)'.  (C-4)
        //:
        //: 5 Verify that there is no output when the stream is invalid.  (C-5)
        //:
        //: 6 Take the address of the '<<' (free) operator and use the result
        //:   to initialize a variable of the appropriate type.  (C-6)
        //
        // Testing:
        //   operator<<(ostream& s, bdlde::ByteOrder::Enum val);
        // --------------------------------------------------------------------

        if (verbose) cout << endl << "Testing '<<' operator" << endl
                                  << "=====================" << endl;

        static const struct {
            int         d_lineNum;  // source line number
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
            //line       enumerator value           expected result
            //----     -----------------------      ---------------
            {  L_,     Obj::e_BIG_ENDIAN,           "BIG_ENDIAN"      },
            {  L_,     Obj::e_LITTLE_ENDIAN,        "LITTLE_ENDIAN"   },
            {  L_,     Obj::e_NETWORK,              "BIG_ENDIAN"      },
#ifdef BSLS_PLATFORM_IS_LITTLE_ENDIAN
            {  L_,     Obj::e_HOST,                 "LITTLE_ENDIAN"   },
#else
            {  L_,     Obj::e_HOST,                 "BIG_ENDIAN"      },
#endif

            {  L_,     NUM_VALUES,                  UNKNOWN_FORMAT    },
            {  L_,     -1,                          UNKNOWN_FORMAT    },
            {  L_,     -5,                          UNKNOWN_FORMAT    },
            {  L_,     99,                          UNKNOWN_FORMAT    }
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        const int   SIZE = 128;         // big enough to hold output string
        const char  XX   = (char)0xFF;  // value of an unset 'char'
              char  buf[SIZE];          // output buffer

              char  mCtrl[SIZE];  memset(mCtrl, XX, SIZE);
        const char *CTRL = mCtrl;

        if (verbose) cout << "\nTesting '<<' operator." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            memcpy(buf, CTRL, SIZE);  // Preset 'buf' to unset 'char' values.

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            bsl::ostringstream oss;
            ASSERT(&oss == &(oss << VALUE));

            if (veryVerbose) cout << "  ACTUAL FORMAT: " << oss.str() << endl;

            LOOP3_ASSERT(LINE, oss.str(), EXP, oss.str() == EXP);
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
//          const char *EXP   = DATA[ti].d_exp;

            memcpy(buf, CTRL, SIZE);  // Preset 'buf' to unset 'char' values.

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            bsl::ostringstream oss;    oss.setstate(bsl::ios::badbit);

            oss << VALUE;

            LOOP2_ASSERT(LINE, ti, oss.str().empty());
        }

        if (verbose) cout << "\nVerify '<<' operator signature." << endl;

        {
            using namespace bdlde;
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum);

            const FuncPtr FP = &operator<<;
            (void) FP;
        }

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING 'print'
        //
        // Concerns:
        //: 1 The 'print' method writes the output to the specified stream.
        //:
        //: 2 The 'print' method writes the string representation of each
        //:   enumerator in the intended format.
        //:
        //: 3 The 'print' method writes a distinguished string when passed an
        //:   out-of-band value.
        //:
        //: 4 There is no output when the stream is invalid.
        //:
        //: 5 The 'print' method has the expected signature.
        //:
        //: 6 The 'print' method returns a reference to the stream it was
        //:   passed.
        //
        // Plan:
        //: 1 Verify that the 'print' method produces the expected results for
        //:   each enumerator.  (C-1, C-2)
        //:
        //: 2 Verify that the 'print' method writes a distinguished string when
        //:   passed an out-of-band value.  (C-3)
        //:
        //: 3 Verify that the 'print' method returns a reference to the stream
        //:   it was passed.  (C-6)
        //:
        //: 4 Verify that there is no output when the stream is invalid.  (C-4)
        //:
        //: 5 Take the address of the 'print' (class) method and use the
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
            //line  level  spl    enumerator value        expected result
            //----  -----  ---  ----------------------    -----------------
            { L_,    0,    4,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN" NL    },
            { L_,    0,    4,  Obj::e_LITTLE_ENDIAN,     "LITTLE_ENDIAN" NL },
            { L_,    0,    4,  Obj::e_NETWORK,           "BIG_ENDIAN" NL    },
#ifdef BSLS_PLATFORM_IS_LITTLE_ENDIAN
            { L_,    0,    4,  Obj::e_HOST,              "LITTLE_ENDIAN" NL },
#else
            { L_,    0,    4,  Obj::e_HOST,              "BIG_ENDIAN" NL    },
#endif
            { L_,    0,    4,  NUM_VALUES,               UNKNOWN_FORMAT NL  },
            { L_,    0,    4,  -1,                       UNKNOWN_FORMAT NL  },
            { L_,    0,    4,  -5,                       UNKNOWN_FORMAT NL  },
            { L_,    0,    4,  99,                       UNKNOWN_FORMAT NL  },

            { L_,    0,   -1,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN"       },
            { L_,    0,    0,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN" NL    },
            { L_,    0,    2,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN" NL    },
            { L_,    1,    1,  Obj::e_BIG_ENDIAN,        " BIG_ENDIAN" NL   },
            { L_,    1,    2,  Obj::e_BIG_ENDIAN,        "  BIG_ENDIAN" NL  },
            { L_,   -1,    2,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN" NL    },
            { L_,   -2,    1,  Obj::e_BIG_ENDIAN,        "BIG_ENDIAN" NL    },
            { L_,    2,    1,  Obj::e_BIG_ENDIAN,        "  BIG_ENDIAN" NL  },
            { L_,    1,    3,  Obj::e_BIG_ENDIAN,        "   BIG_ENDIAN" NL },
#undef NL
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        const int   SIZE = 128;         // big enough to hold output string
        const char  XX   = (char)0xFF;  // value of an unset 'char'
              char  buf[SIZE];          // output buffer

              char  mCtrl[SIZE];  memset(mCtrl, XX, SIZE);
        const char *CTRL = mCtrl;

        if (verbose) cout << "\nTesting 'print'." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
            const char *EXP   = DATA[ti].d_exp;

            memcpy(buf, CTRL, SIZE);  // Preset 'buf' to unset 'char' values.

            if (veryVerbose) { T_; P_(ti); P(VALUE); }
            if (veryVerbose) cout << "EXPECTED FORMAT: " << EXP << endl;

            bsl::ostringstream oss;
            ASSERT(&oss == &Obj::print(oss, VALUE, LEVEL, SPL));

            if (veryVerbose) cout << "  ACTUAL FORMAT: " << oss.str() << endl;

            LOOP3_ASSERT(LINE, oss.str(), EXP, oss.str() == EXP);

            if (0 == LEVEL && 4 == SPL) {
                if (veryVerbose)
                    cout << "\tRepeat for 'print' default arguments." << endl;

                bsl::ostringstream ossB;
                Obj::print(ossB, VALUE, LEVEL, SPL);

                if (veryVerbose) cout << "  ACTUAL FORMAT: " << ossB.str() <<
                                                                          endl;

                LOOP3_ASSERT(LINE, oss.str(), EXP, ossB.str() == EXP);
            }
        }

        if (verbose) cout << "\tNothing is written to a bad stream." << endl;

        for (int ti = 0; ti < NUM_DATA; ++ti) {
            const int   LINE  = DATA[ti].d_lineNum;
            const int   LEVEL = DATA[ti].d_level;
            const int   SPL   = DATA[ti].d_spl;
            const Enum  VALUE = static_cast<Enum>(DATA[ti].d_value);
//          const char *EXP   = DATA[ti].d_exp;

            memcpy(buf, CTRL, SIZE);  // Preset 'buf' to unset 'char' values.

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            bsl::ostringstream oss;    oss.setstate(bsl::ios::badbit);
            Obj::print(oss, VALUE, LEVEL, SPL);

            LOOP2_ASSERT(LINE, ti, oss.str().empty());
        }

        if (verbose) cout << "\nVerify 'print' signature." << endl;

        {
            typedef bsl::ostream& (*FuncPtr)(bsl::ostream&, Enum, int, int);

            const FuncPtr FP = &Obj::print;
            (void) FP;
        }
      } break;
      case 1: {
        // -------------------------------------------------------------------
        // TESTING 'enum' AND 'toAscii'
        //
        // Concerns:
        //: 1 The enumerator values are sequential, starting from 0.
        //:
        //: 2 The 'toAscii' method returns the expected string representation
        //:   for each enumerator.
        //:
        //: 3 The 'toAscii' method returns a distinguished string when passed
        //:   an out-of-band value.
        //:
        //: 4 The 'toAscii' method has the expected signature.
        //
        // Plan:
        //: 1 Verify that the enumerator values are sequential, starting from
        //:   0.  (C-1)
        //:
        //: 2 Verify that the 'toAscii' method returns the expected string
        //:   representation for each enumerator.  (C-2)
        //:
        //: 3 Verify that the 'toAscii' method returns a distinguished string
        //:   when passed an out-of-band value.  (C-3)
        //:
        //: 4 Take the address of the 'toAscii' (class) method and use the
        //:   result to initialize a variable of the appropriate type.  (C-5)
        //
        // Testing:
        //   enum Enum { ... };
        //   const char *toAscii(bdlde::ByteOrder::Enum val);
        // -------------------------------------------------------------------

        if (verbose) cout << endl << "Testing 'enum' and 'toAscii'" << endl
                                  << "============================" << endl;

        static const struct {
            int         d_lineNum;  // source line number
            int         d_value;    // enumerator value
            const char *d_exp;      // expected result
        } DATA[] = {
            // line         enumerator value        expected result
            // ----    -----------------------      -----------------
            {  L_,     Obj::e_LITTLE_ENDIAN,        "LITTLE_ENDIAN"   },
            {  L_,     Obj::e_BIG_ENDIAN,           "BIG_ENDIAN"      },
            {  L_,     Obj::e_NETWORK,              "BIG_ENDIAN"      },
#ifdef BSLS_PLATFORM_IS_LITTLE_ENDIAN
            {  L_,     Obj::e_HOST,                 "LITTLE_ENDIAN"   },
#else
            {  L_,     Obj::e_HOST,                 "BIG_ENDIAN"      },
#endif
            {  L_,     NUM_VALUES,                  UNKNOWN_FORMAT    },
            {  L_,     -1,                          UNKNOWN_FORMAT    },
            {  L_,     -5,                          UNKNOWN_FORMAT    },
            {  L_,     99,                          UNKNOWN_FORMAT    }
        };
        const int NUM_DATA = sizeof DATA / sizeof *DATA;

        if (verbose) cout << "\nVerify enumerator values are sequential."
                          << endl;

        for (int ti = 0; ti < NUM_VALUES; ++ti) {
            const Enum VALUE = static_cast<Enum>(DATA[ti].d_value);

            if (veryVerbose) { T_; P_(ti); P(VALUE); }

            LOOP_ASSERT(ti, ti == VALUE);
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
            (void) FP;
        }

      } break;
      default: {
        cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    ASSERT(0 == defaultAllocator.numBlocksTotal());
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
