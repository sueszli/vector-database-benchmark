// balxml_base64parser.t.cpp                                          -*-C++-*-
#include <balxml_base64parser.h>

#include <bslim_testutil.h>

#include <bdlb_printmethods.h>

#include <bsls_libraryfeatures.h>
#include <bsls_nameof.h>
#include <bsls_review.h>

#include <bsl_cstddef.h>
#include <bsl_cstdlib.h>
#include <bsl_cstring.h>
#include <bsl_iostream.h>
#include <bsl_istream.h>
#include <bsl_iterator.h>
#include <bsl_sstream.h>
#include <bsl_vector.h>

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
# include <memory_resource>
#endif

#include <vector>

using namespace BloombergLP;

using bsl::cout;
using bsl::cerr;
using bsl::endl;
using bsl::flush;

// ============================================================================
//                             TEST PLAN
// ----------------------------------------------------------------------------
//                              Overview
//                              --------
// ----------------------------------------------------------------------------

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
//                       TEMPLATIZED OUTPUT FUNCTIONS
// ============================================================================

template <class T>
void printValue(bsl::ostream& out, const T& value)
{
    bdlb::PrintMethods::print(out, value, 0, -1);
}

static const char *printableCharacters[256]=
{
    "\\x00" , //   0   0
    "\\x01" , //   1   1
    "\\x02" , //   2   2
    "\\x03" , //   3   3
    "\\x04" , //   4   4
    "\\x05" , //   5   5
    "\\x06" , //   6   6
    "\\x07" , //   7   7
    "\\x08" , //   8   8 - BACKSPACE
    "\\t" ,   //   9   9 - TAB
    "\\n" ,   //  10   a - LF
    "\\x0b" , //  11   b
    "\\x0c" , //  12   c
    "\\r"   , //  13   d - CR
    "\\x0e" , //  14   e
    "\\x0f" , //  15   f
    "\\x10" , //  16  10
    "\\x11" , //  17  11
    "\\x12" , //  18  12
    "\\x13" , //  19  13
    "\\x14" , //  20  14
    "\\x15" , //  21  15
    "\\x16" , //  22  16
    "\\x17" , //  23  17
    "\\x18" , //  24  18
    "\\x19" , //  25  19
    "\\x1a" , //  26  1a
    "\\x1b" , //  27  1b
    "\\x1c" , //  28  1c
    "\\x1d" , //  29  1d
    "\\x1e" , //  30  1e
    "\\x1f" , //  31  1f
    " "     , //  32  20 - SPACE
    "!"     , //  33  21 - !
    "\\\""  , //  34  22 - "
    "#"     , //  35  23 - #
    "$"     , //  36  24 - $
    "%"     , //  37  25 - %
    "&"     , //  38  26 - &
    "'"     , //  39  27 - '
    "("     , //  40  28 - (
    ")"     , //  41  29 - )
    "*"     , //  42  2a - *
    "+"     , //  43  2b - +
    ","     , //  44  2c - ,
    "-"     , //  45  2d - -
    "."     , //  46  2e - .
    "/"     , //  47  2f - /
    "0"     , //  48  30 - 0
    "1"     , //  49  31 - 1
    "2"     , //  50  32 - 2
    "3"     , //  51  33 - 3
    "4"     , //  52  34 - 4
    "5"     , //  53  35 - 5
    "6"     , //  54  36 - 6
    "7"     , //  55  37 - 7
    "8"     , //  56  38 - 8
    "9"     , //  57  39 - 9
    ":"     , //  58  3a - :
    ";"     , //  59  3b - ;
    "<"     , //  60  3c - <
    "="     , //  61  3d - =
    ">"     , //  62  3e - >
    "?"     , //  63  3f - ?
    "@"     , //  64  40 - @
    "A"     , //  65  41 - A
    "B"     , //  66  42 - B
    "C"     , //  67  43 - C
    "D"     , //  68  44 - D
    "E"     , //  69  45 - E
    "F"     , //  70  46 - F
    "G"     , //  71  47 - G
    "H"     , //  72  48 - H
    "I"     , //  73  49 - I
    "J"     , //  74  4a - J
    "K"     , //  75  4b - K
    "L"     , //  76  4c - L
    "M"     , //  77  4d - M
    "N"     , //  78  4e - N
    "O"     , //  79  4f - O
    "P"     , //  80  50 - P
    "Q"     , //  81  51 - Q
    "R"     , //  82  52 - R
    "S"     , //  83  53 - S
    "T"     , //  84  54 - T
    "U"     , //  85  55 - U
    "V"     , //  86  56 - V
    "W"     , //  87  57 - W
    "X"     , //  88  58 - X
    "Y"     , //  89  59 - Y
    "Z"     , //  90  5a - Z
    "["     , //  91  5b - [
    "\\\\"  , //  92  5c - '\'
    "]"     , //  93  5d - ]
    "^"     , //  94  5e - ^
    "_"     , //  95  5f - _
    "`"     , //  96  60 - `
    "a"     , //  97  61 - a
    "b"     , //  98  62 - b
    "c"     , //  99  63 - c
    "d"     , // 100  64 - d
    "e"     , // 101  65 - e
    "f"     , // 102  66 - f
    "g"     , // 103  67 - g
    "h"     , // 104  68 - h
    "i"     , // 105  69 - i
    "j"     , // 106  6a - j
    "k"     , // 107  6b - k
    "l"     , // 108  6c - l
    "m"     , // 109  6d - m
    "n"     , // 110  6e - n
    "o"     , // 111  6f - o
    "p"     , // 112  70 - p
    "q"     , // 113  71 - q
    "r"     , // 114  72 - r
    "s"     , // 115  73 - s
    "t"     , // 116  74 - t
    "u"     , // 117  75 - u
    "v"     , // 118  76 - v
    "w"     , // 119  77 - w
    "x"     , // 120  78 - x
    "y"     , // 121  79 - y
    "z"     , // 122  7a - z
    "{"     , // 123  7b - {
    "|"     , // 124  7c - |
    "}"     , // 125  7d - }
    "~"     , // 126  7e - ~
    "\\x7f" , // 127  7f - DEL
    "\\x80" , // 128  80
    "\\x81" , // 129  81
    "\\x82" , // 130  82
    "\\x83" , // 131  83
    "\\x84" , // 132  84
    "\\x85" , // 133  85
    "\\x86" , // 134  86
    "\\x87" , // 135  87
    "\\x88" , // 136  88
    "\\x89" , // 137  89
    "\\x8a" , // 138  8a
    "\\x8b" , // 139  8b
    "\\x8c" , // 140  8c
    "\\x8d" , // 141  8d
    "\\x8e" , // 142  8e
    "\\x8f" , // 143  8f
    "\\x90" , // 144  90
    "\\x91" , // 145  91
    "\\x92" , // 146  92
    "\\x93" , // 147  93
    "\\x94" , // 148  94
    "\\x95" , // 149  95
    "\\x96" , // 150  96
    "\\x97" , // 151  97
    "\\x98" , // 152  98
    "\\x99" , // 153  99
    "\\x9a" , // 154  9a
    "\\x9b" , // 155  9b
    "\\x9c" , // 156  9c
    "\\x9d" , // 157  9d
    "\\x9e" , // 158  9e
    "\\x9f" , // 159  9f
    "\\xa0" , // 160  a0
    "\\xa1" , // 161  a1
    "\\xa2" , // 162  a2
    "\\xa3" , // 163  a3
    "\\xa4" , // 164  a4
    "\\xa5" , // 165  a5
    "\\xa6" , // 166  a6
    "\\xa7" , // 167  a7
    "\\xa8" , // 168  a8
    "\\xa9" , // 169  a9
    "\\xaa" , // 170  aa
    "\\xab" , // 171  ab
    "\\xac" , // 172  ac
    "\\xad" , // 173  ad
    "\\xae" , // 174  ae
    "\\xaf" , // 175  af
    "\\xb0" , // 176  b0
    "\\xb1" , // 177  b1
    "\\xb2" , // 178  b2
    "\\xb3" , // 179  b3
    "\\xb4" , // 180  b4
    "\\xb5" , // 181  b5
    "\\xb6" , // 182  b6
    "\\xb7" , // 183  b7
    "\\xb8" , // 184  b8
    "\\xb9" , // 185  b9
    "\\xba" , // 186  ba
    "\\xbb" , // 187  bb
    "\\xbc" , // 188  bc
    "\\xbd" , // 189  bd
    "\\xbe" , // 190  be
    "\\xbf" , // 191  bf
    "\\xc0" , // 192  c0
    "\\xc1" , // 193  c1
    "\\xc2" , // 194  c2
    "\\xc3" , // 195  c3
    "\\xc4" , // 196  c4
    "\\xc5" , // 197  c5
    "\\xc6" , // 198  c6
    "\\xc7" , // 199  c7
    "\\xc8" , // 200  c8
    "\\xc9" , // 201  c9
    "\\xca" , // 202  ca
    "\\xcb" , // 203  cb
    "\\xcc" , // 204  cc
    "\\xcd" , // 205  cd
    "\\xce" , // 206  ce
    "\\xcf" , // 207  cf
    "\\xd0" , // 208  d0
    "\\xd1" , // 209  d1
    "\\xd2" , // 210  d2
    "\\xd3" , // 211  d3
    "\\xd4" , // 212  d4
    "\\xd5" , // 213  d5
    "\\xd6" , // 214  d6
    "\\xd7" , // 215  d7
    "\\xd8" , // 216  d8
    "\\xd9" , // 217  d9
    "\\xda" , // 218  da
    "\\xdb" , // 219  db
    "\\xdc" , // 220  dc
    "\\xdd" , // 221  dd
    "\\xde" , // 222  de
    "\\xdf" , // 223  df
    "\\xe0" , // 224  e0
    "\\xe1" , // 225  e1
    "\\xe2" , // 226  e2
    "\\xe3" , // 227  e3
    "\\xe4" , // 228  e4
    "\\xe5" , // 229  e5
    "\\xe6" , // 230  e6
    "\\xe7" , // 231  e7
    "\\xe8" , // 232  e8
    "\\xe9" , // 233  e9
    "\\xea" , // 234  ea
    "\\xeb" , // 235  eb
    "\\xec" , // 236  ec
    "\\xed" , // 237  ed
    "\\xee" , // 238  ee
    "\\xef" , // 239  ef
    "\\xf0" , // 240  f0
    "\\xf1" , // 241  f1
    "\\xf2" , // 242  f2
    "\\xf3" , // 243  f3
    "\\xf4" , // 244  f4
    "\\xf5" , // 245  f5
    "\\xf6" , // 246  f6
    "\\xf7" , // 247  f7
    "\\xf8" , // 248  f8
    "\\xf9" , // 249  f9
    "\\xfa" , // 250  fa
    "\\xfb" , // 251  fb
    "\\xfc" , // 252  fc
    "\\xfd" , // 253  fd
    "\\xfe" , // 254  fe
    "\\xff"   // 255  ff
};

void printValue(bsl::ostream& out, const char* value)
    // Specialize for char*.  Need to expand \r, \n, \t and surround with
    // DQUOTE characters.
{
    out << '"';

    while (*value) {
        out << printableCharacters[*value];
        ++value;
    }

    out << '"';
}

void printValue(bsl::ostream& out, const bsl::string& value)
    // Need to expand \r, \n, \t and surround with DQUOTE characters.
{
    printValue(out, value.c_str());
}

void printValue(bsl::ostream& out, const char& value)
    // Need to expand \r, \n, \t and surround with SQUOTE characters.
{
    out << '\'' << printableCharacters[value] << '\'';
}

// ============================================================================
//                       GLOBAL VARIABLES FOR TESTING
// ----------------------------------------------------------------------------

int test;
int verbose;
int veryVerbose;
int veryVeryVerbose;

// ============================================================================
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

// ============================================================================
//                               USAGE EXAMPLE
// ----------------------------------------------------------------------------

///Usage
///-----
// The following snippets of code illustrate the usage of this component.
// Suppose you had an input stream that contained Base64 data.  The following
// 'loadFromBase64Stream' function loads this data into an 'bsl::vector<char>'
// blob:
//..
//  #include <balxml_base64parser.h>
//
//  #include <istream>
//  #include <iterator>
//  #include <vector>
//
//  using namespace BloombergLP;
//
int loadFromBase64Stream(bsl::vector<char> *result, bsl::istream& stream)
{
    enum { FAILURE = -1 };

    balxml::Base64Parser<bsl::vector<char> > parser;

    if (0 != parser.beginParse(result)) {
        return FAILURE;                                               // RETURN
    }

    if (0 != parser.pushCharacters(bsl::istreambuf_iterator<char>(stream),
                                   bsl::istreambuf_iterator<char>())) {
        return FAILURE;                                               // RETURN
    }

    return parser.endParse();
}
//..
// The following function demonstrates the 'loadFromBase64Stream' function:
//..
//  #include <sstream>
//
void usageExample()
{
    const char INPUT[] = "YWJjZA==";  // "abcd" in Base64

    bsl::vector<char>  vec;
    bsl::istringstream iss(INPUT);

    int result = loadFromBase64Stream(&vec, iss);

    ASSERT(0   == result);
    ASSERT(4   == vec.size());
    ASSERT('a' == vec[0]);
    ASSERT('b' == vec[1]);
    ASSERT('c' == vec[2]);
    ASSERT('d' == vec[3]);
}
//..

namespace {
namespace u {

enum { k_MAX_NUM_PUSHES = 10 };

struct Data {
    int         d_lineNum;     // source line number
    const struct {
        const char *d_chars;       // characters to push
        bool        d_success;     // true if push successful
    }           d_input[k_MAX_NUM_PUSHES];
    bool        d_endSuccess;  // true if 'endParse' successful
    const char *d_resultData;  // expected result data
};

template <class VECTOR_TYPE>
void test2(const Data& data)
{
    const int LINE = data.d_lineNum;

    if (verbose) cout <<
                 "Starting test2<" << bsls::NameOf<VECTOR_TYPE>() << ">(): " <<
                                                                  LINE << endl;

    const char              INIT_DATA[] = "InIt VaLuE";
    const VECTOR_TYPE       INIT_VALUE(INIT_DATA,
                                       INIT_DATA + sizeof INIT_DATA);

    VECTOR_TYPE             mX = INIT_VALUE;
    const VECTOR_TYPE&      X  = mX;

    balxml::Base64Parser<VECTOR_TYPE> parser;
    int                               retCode;

    LOOP2_ASSERT(LINE, X.size(), 0 != X.size());

    retCode = parser.beginParse(&mX);
    LOOP2_ASSERT(LINE, retCode,  0 == retCode);
    LOOP2_ASSERT(LINE, X.size(), 0 == X.size());

    bool areAllPushesSuccessful = true;

    for (int j = 0; j < k_MAX_NUM_PUSHES; ++j) {
        const char *CHARS   = data.d_input[j].d_chars;
        const bool  SUCCESS = data.d_input[j].d_success;

        if (0 == CHARS) {
            break;
        }

        if (veryVeryVerbose) {
            T_ T_ P_(CHARS) P(SUCCESS)
        }

        const char *begin = CHARS;
        const char *end   = CHARS + bsl::strlen(CHARS);

        retCode = parser.pushCharacters(begin, end);

        if (!SUCCESS) {
            areAllPushesSuccessful = false;

            LOOP3_ASSERT(LINE, j, retCode, 0 != retCode);

            break;
        }

        LOOP3_ASSERT(LINE, j, retCode, 0 == retCode);
    }

    if (!areAllPushesSuccessful) {
        if (verbose) cout <<
                   "Quitting test2<" << bsls::NameOf<VECTOR_TYPE>() << ">()\n";

        return;                                                       // RETURN
    }

    const bool END_SUCCESS = data.d_endSuccess;

    retCode = parser.endParse();

    if (!END_SUCCESS) {
        if (veryVerbose) {
            T_ P(END_SUCCESS)
        }

        LOOP2_ASSERT(LINE, retCode, 0 != retCode);

        if (verbose) cout <<
                   "Quitting test2<" << bsls::NameOf<VECTOR_TYPE>() << ">()\n";

        return;                                                       // RETURN
    }

    LOOP2_ASSERT(LINE, retCode, 0 == retCode);

    const char *EXPECTED_RESULT_DATA = data.d_resultData;

    if (veryVerbose) {
        T_ P_(END_SUCCESS) P(EXPECTED_RESULT_DATA)
    }

    LOOP3_ASSERT(LINE, bsl::strlen(EXPECTED_RESULT_DATA),   X.size(),
                       bsl::strlen(EXPECTED_RESULT_DATA) == X.size());

    for (bsl::size_t j = 0; j < X.size(); ++j) {
        LOOP4_ASSERT(LINE, j, EXPECTED_RESULT_DATA[j],   X[j],
                              EXPECTED_RESULT_DATA[j] == X[j]);
    }

    if (verbose) cout <<
                   "Ending   test2<" << bsls::NameOf<VECTOR_TYPE>() << ">()\n";
}

}  // close namespace u
}  // close unnamed namespace

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    test = argc > 1 ? bsl::atoi(argv[1]) : 0;
    verbose = argc > 2;
    veryVerbose = argc > 3;
    veryVeryVerbose = argc > 4;

    bsl::cout << "TEST " << __FILE__ << " CASE " << test << bsl::endl;;

    // CONCERN: 'BSLS_REVIEW' failures should lead to test failures.
    bsls::ReviewFailureHandlerGuard reviewGuard(&bsls::Review::failByAbort);

    switch (test) { case 0:  // Zero is always the leading case.
      case 3: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //
        // Concerns:
        //
        // Plan:
        // --------------------------------------------------------------------

        if (verbose) bsl::cout << "\nUSAGE EXAMPLE\n" << bsl::endl;

        usageExample();

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // THOROUGH TEST
        //   This will thoroughly test the 'balxml::Base64Parser' class.
        //
        // Concerns:
        //   This parser must reset the associated object when 'beginParse' is
        //   called.  It should decode incoming Base64 characters that may be
        //   passed through multiple calls of 'pushCharacters'.  The decoded
        //   data must be appended to the associated object.
        //
        // Plan:
        //   Use 'bsl::vector<char>' for the 'TYPE' parameter.  For each item
        //   in a set of test data, create an 'balxml::Base64Parser<TYPE>'
        //   object and associate it with an 'bsl::vector<char>' object, 'mX',
        //   that has been filled with some arbitrary data.  Verify that 'mX'
        //   has been reset.
        //
        //   Push the characters from the dataset into the parser object.
        //   Check that the return value is as expected.
        //
        //   Call 'endParse' and check that the return value is as expected.
        //   Finally, verify the contents of 'mX'.
        //
        //   The test data was generated using a Base64 oracle that can be
        //   accessed at the following URL:
        //
        //         - http://makcoder.sourceforge.net/demo/base64.php
        //
        // Testing:
        //   balxml::Base64Parser
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTHOROUGH TEST"
                          << "\n=============" << endl;

        static const u::Data DATA[] = {
            //line  input                                endSuccess  result
            //----  -----                                ----------  ------
            //          chars           success
            //          -----           -------

            { L_,   { { 0,                true  }   },   true,       ""      },
            { L_,   { { "",               true  },
                      { 0,                true  }   },   true,       ""      },
            { L_,   { { " ",              true  },
                      { 0,                true  }   },   true,       ""      },
            { L_,   { { "  ",             true  },
                      { 0,                true  }   },   true,       ""      },
            { L_,   { { " ",              true  },
                      { "  ",             true  },
                      { 0,                true  }   },   true,       ""      },

            { L_,   { { "YQ==",           true  },
                      { 0,                true  }   },   true,       "a"     },
            { L_,   { { "Y",              true  },
                      { "Q",              true  },
                      { "=",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "a"     },
            { L_,   { { "YQ",             true  },
                      { "=",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "a"     },
            { L_,   { { "YQ",             true  },
                      { "==",             true  },
                      { 0,                true  }   },   true,       "a"     },
            { L_,   { { "YQ=",            true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "a"     },
            { L_,   { { "YQ=",            true  },
                      { "=",              true  },
                      { "=",              false }   },   true,       ""      },
            { L_,   { { "YQ=",            true  },
                      { 0,                true  }   },   false,      ""      },

            { L_,   { { "YWI=",           true  },
                      { 0,                true  }   },   true,       "ab"    },
            { L_,   { { "Y",              true  },
                      { "W",              true  },
                      { "I",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "ab"    },
            { L_,   { { "YW",             true  },
                      { "I",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "ab"    },
            { L_,   { { "YW",             true  },
                      { "I=",             true  },
                      { 0,                true  }   },   true,       "ab"    },
            { L_,   { { "YWI",            true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "ab"    },
            { L_,   { { "YWI",            true  },
                      { "=",              true  },
                      { "=",              false }   },   true,       "ab"    },
            { L_,   { { "YWI",            true  },
                      { 0,                true  }   },   false,      ""      },

            { L_,   { { "YWJj",           true  },
                      { 0,                true  }   },   true,       "abc"   },
            { L_,   { { "Y",              true  },
                      { "W",              true  },
                      { "J",              true  },
                      { "j",              true  },
                      { 0,                true  }   },   true,       "abc"   },
            { L_,   { { "YW",             true  },
                      { "J",              true  },
                      { "j",              true  },
                      { 0,                true  }   },   true,       "abc"   },
            { L_,   { { "YW",             true  },
                      { "Jj",             true  },
                      { 0,                true  }   },   true,       "abc"   },
            { L_,   { { "YWJ",            true  },
                      { "j",              true  },
                      { 0,                true  }   },   true,       "abc"   },
            { L_,   { { "YWJ",            true  },
                      { "j",              true  },
                      { "=",              false }   },   true,       ""      },
            { L_,   { { "YWJ",            true  },
                      { "j",              true  },
                      { "j",              true  },
                      { 0,                true  }   },   false,      ""      },

            { L_,   { { "YWJjZA==",       true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "Y",              true  },
                      { "W",              true  },
                      { "J",              true  },
                      { "j",              true  },
                      { "Z",              true  },
                      { "A",              true  },
                      { "=",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "Z",              true  },
                      { "A",              true  },
                      { "=",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "ZA",             true  },
                      { "=",              true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "ZA",             true  },
                      { "==",             true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "ZA=",            true  },
                      { "=",              true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "ZA==",           true  },
                      { 0,                true  }   },   true,       "abcd"  },
            { L_,   { { "YWJj",           true  },
                      { "ZA=",            true  },
                      { "=",              true  },
                      { "=",              false }   },   true,       ""      },
            { L_,   { { "YWJj",           true  },
                      { "ZA=",            true  },
                      { 0,                true  }   },   false,      ""      },

            { L_,   { { "YWJjZA;();",     false }   },   true,       ""      },
        };
        enum { k_NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int ti = 0; ti < k_NUM_DATA; ++ti) {
            const u::Data& data = DATA[ti];

            u::test2<bsl::vector<char> >(data);
            u::test2<std::vector<char> >(data);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            u::test2<std::pmr::vector<char>>(data);
#endif
        }
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //
        // Concerns:
        //
        // Plan:
        // --------------------------------------------------------------------

        if (verbose) bsl::cout << "\nBREATHING TEST\n" << bsl::endl;

        {
            typedef bsl::vector<char>         Type;
            typedef balxml::Base64Parser<Type> Parser;

            Parser parser;
            Type   value;
            int    result;

            value.push_back('A');
            value.push_back('B');
            value.push_back('C');

            result = parser.beginParse(&value);
            LOOP_ASSERT(result, 0 == result);
            result = parser.endParse();
            LOOP_ASSERT(result, 0 == result);
            LOOP_ASSERT(value.size(), 0 == value.size());
        }

      } break;
      default: {
        bsl::cerr << "WARNING: CASE `" << test << "' NOT FOUND." << bsl::endl;
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        bsl::cerr << "Error, non-zero test status = "
                  << testStatus
                  << "."
                  << bsl::endl;
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
