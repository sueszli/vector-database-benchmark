// ball_rule.t.cpp                                                    -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------


#include <ball_rule.h>

#include <ball_severity.h>                      // for testing only
#include <ball_attributecontainerlist.h>        // for testing only
#include <ball_defaultattributecontainer.h>     // for testing only
#include <ball_attribute.h>

#include <bslim_testutil.h>
#include <bslma_default.h>
#include <bslma_testallocator.h>
#include <bslma_testallocatorexception.h>

#include <bsls_asserttest.h>

#include <bsl_climits.h>
#include <bsl_cstdlib.h>
#include <bsl_iostream.h>
#include <bsl_unordered_set.h>
#include <bsl_sstream.h>

using namespace BloombergLP;
using namespace ball;
using namespace bsl;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test implements a value-semantic 'Rule' class.  We
// choose the default constructor, 'setPattern', 'setLevels', 'addAttribute',
// and 'removeAttribute' as the primary manipulators.  We apply the standard
// 10 test procedure as well as a few test cases for additional methods such
// as 'hash' and 'evaluate'.
//-----------------------------------------------------------------------------
// [14] static int hash(const Rule&, int size);
// [ 2] Rule(bdema::Alct * = 0);
// [10] Rule(const char *pattern, int rl, int pl, int tl, int tal);
// [ 7] Rule(const Rule&, bdema::Alct * = 0);
// [ 2] ~Rule();
// [ 9] const Rule& operator=(const Rule& other);
// [ 2] int addAttribute(const ManagedAttribute& attribute);
// [ 2] int removeAttribute(const ManagedAttribute& attribute);
// [12] void removeAll();
// [ 2] int setLevels(int rl, int pl, int tl, int tal);
// [ 2] void setPattern(const char *value);
// [13] bool evaluate(const AttributeContainerList& context) const;
// [ 4] int numAttributes() const;
// [ 4] bool hasPredicate(const ManagedAttribute&) const;
// [11] ManagedAttributeSet::const_iterator begin() const;
// [11] ManagedAttributeSet::const_iterator end() const;
// [ 4] int recordLevel() const;
// [ 4] int passLevel() const;
// [ 4] int triggerLevel() const;
// [ 4] int triggerAllLevel() const;
// [ 4] const char *pattern() const;
// [15] bool isMatch(const char *inputString) const;
// [ 5] bsl::ostream& print(bsl::ostream& stream, int lvl, int spl) const;
// [ 6] bool operator==(const Rule& lhs, const Rule& rhs);
// [ 6] bool operator!=(const Rule& lhs, const Rule& rhs);
// [ 5] bsl::ostream& operator<<(bsl::ostream&, const PS&) const;
//-----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [ 3] PRIMITIVE TEST APPARATUS: 'gg'
// [ 8] UNUSED
// [16] USAGE EXAMPLE

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

#define Q            BSLIM_TESTUTIL_Q   // Quote identifier literally.
#define P            BSLIM_TESTUTIL_P   // Print identifier and value.
#define P_           BSLIM_TESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLIM_TESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLIM_TESTUTIL_L_  // current Line number

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
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

typedef Rule                      Obj;
typedef ThresholdAggregate        Levels;

#define VA_LEVELS 0, 0, 0, 0
#define VB_LEVELS 1, 0, 0, 0
#define VC_LEVELS 255, 255, 255, 255
#define VD_LEVELS 16, 32, 48, 64

#define VA_PATTERN ""
#define VB_PATTERN "eq"
#define VC_PATTERN "eq*"
#define VD_PATTERN "equity"

#define VA VA_PATTERN, VA_LEVELS
#define VB VB_PATTERN, VB_LEVELS
#define VC VC_PATTERN, VC_LEVELS
#define VD VD_PATTERN, VD_LEVELS

// The first three attributes are used as part of VB, VC, and VD.
// VA: none
// VB: P1
// VC: P2, P3
// VD: P1, P2, P3

const ManagedAttribute P1("A", "1",       bslma::Default::globalAllocator());
const ManagedAttribute P2("A", 1,         bslma::Default::globalAllocator());
const ManagedAttribute P3("A", 1LL,       bslma::Default::globalAllocator());
const ManagedAttribute P4("",  "",        bslma::Default::globalAllocator());
const ManagedAttribute P5("B", INT_MAX,   bslma::Default::globalAllocator());
const ManagedAttribute P6("B", LLONG_MAX, bslma::Default::globalAllocator());
const ManagedAttribute P7("C", LLONG_MIN, bslma::Default::globalAllocator());
const ManagedAttribute P8("a", 1,         bslma::Default::globalAllocator());
const ManagedAttribute P9("a", 1LL,       bslma::Default::globalAllocator());

const char *PATTERNS[] = {
    "",
    "eq",
    "eq*",
    "eq\\*",
    "eq\\\\",
    "e\\*q",
    "e\\\\q",
};

enum { NUM_PATTERNS = sizeof PATTERNS / sizeof *PATTERNS };

static const struct {
    int d_recordLevel;     // record level
    int d_passLevel;       // pass level
    int d_triggerLevel;    // trigger level
    int d_triggerAllLevel; // trigger all level
} LEVELS[] = {
    ///record    pass     trigger  triggerAll
    ///level     level     level     level
    ///------    ------    ------    -----
    {  0,        0,        0,        0,       },
    {  1,        0,        0,        0,       },
    {  0,        1,        0,        0,       },
    {  0,        0,        1,        0,       },
    {  0,        0,        0,        1,       },
    {  16,       32,       48,       64,      },
    {  64,       48,       32,       16,      },
    {  16,       32,       64,       48,      },
    {  16,       48,       32,       64,      },
    {  32,       16,       48,       64,      },
    {  255,      0,        0,        0,       },
    {  0,        255,      0,        0,       },
    {  0,        0,        255,      0,       },
    {  0,        0,        0,        255,     },
    {  255,      255,      255,      255,     },
};

enum { NUM_LEVELS = sizeof LEVELS / sizeof *LEVELS };

const ManagedAttribute PREDICATES[] =
{
    P1, P2, P3, P4, P5, P6, P7, P8, P9
};

enum { NUM_PREDICATES = sizeof PREDICATES / sizeof *PREDICATES };

//=============================================================================
//       GENERATOR FUNCTIONS 'g', 'gg', AND 'ggg' FOR TESTING
//-----------------------------------------------------------------------------
// The 'g' family of functions generate a 'Rule' object for testing.
// They interpret a given 'spec' (from left to right) to configure the
// attribute set according to a custom language.
//
// To simplify these generator functions, patterns, threshold levels, and
// attributes are all represented by two-character combinations.  The first
// character denotes the type: 'p' for pattern, 'L' for threshold levels, and
// 'P' for attribute.  The second character denotes the value, which is
// obtained from the respective array defined above (with 'A' corresponding to
// the first element, 'B' the second, etc.).
//
// LANGUAGE SPECIFICATION:
// -----------------------
//
// <SPEC>       ::= <ACTION> *
//
// <ACTION>     ::= <PATTERN>
//                | <LEVELS>
//                | <PREDICATE>
//
// <PATTERN>    ::= 'p' [ 'A' .. 'A' + NUM_PATTERNS - 1 ]
//
// <LEVELS>     ::= 'L' [ 'A' .. 'A' + NUM_LEVELS - 1 ]
//
// <PREDICATE>  ::= 'P' [ 'A' .. 'A' + NUM_PREDICATES - 1 ]
//
// Spec String      Description
// -----------      -----------------------------------------------------------
// ""               Has no effect; leaves the object unaltered.
// "pALAPA"         Produces: { "" 0 0 0 0 {} }
// "pBLBPB"         Produces: { "eq" 1 0 0 0 { A="1" } }
// "pBLBPBPC"       Produces: { "eq" 1 0 0 0 { A="1" A=1 } }
//-----------------------------------------------------------------------------

static Obj& gg(Obj *obj, const char *spec)
{
    while (*spec) {
        const char c = *spec;
        ++spec;
        switch (c) {
          case 'p': {
            obj->setPattern(PATTERNS[*spec - 'A']);
          } break;
          case 'L': {
            obj->setLevels(LEVELS[*spec - 'A'].d_recordLevel,
                           LEVELS[*spec - 'A'].d_passLevel,
                           LEVELS[*spec - 'A'].d_triggerLevel,
                           LEVELS[*spec - 'A'].d_triggerAllLevel);
          } break;
          case 'P': {
            obj->addAttribute(PREDICATES[*spec - 'A']);
          } break;
          default: {
            ASSERT(c);
          }
        }
        ++spec;
    };

    return *obj;
}

bool compareText(const bsl::string_view& lhs,
                 const bsl::string_view& rhs,
                 bsl::ostream&           errorStream = bsl::cout)
    // Return 'true' if the specified 'lhs' has the same value as the specified
    // 'rhs' and 'false' otherwise.  Optionally specify a 'errorStream', on
    // which, if 'lhs' and 'rhs' are not the same, a description of how the two
    // strings differ will be written.  If 'errorStream' is not supplied,
    // 'stdout' will be used to report an error description.
{
    for (unsigned int i = 0; i < lhs.length() && i < rhs.length(); ++i) {
        if (lhs[i] != rhs[i]) {
            errorStream << "lhs: \"" << lhs << "\"\n"
                        << "rhs: \"" << rhs << "\"\n"
                        << "Strings differ at index (" << i << ") "
                        << "lhs[i] = " << lhs[i] << "(" << (int)lhs[i] << ") "
                        << "rhs[i] = " << rhs[i] << "(" << (int)rhs[i] << ")"
                        << endl;
            return false;                                             // RETURN
        }
    }

    if (lhs.length() < rhs.length()) {
        bsl::size_t i = lhs.length();
        errorStream << "lhs: \"" << lhs << "\"\n"
                    << "rhs: \"" << rhs << "\"\n"
                    << "Strings differ at index (" << i << ") "
                    << "lhs[i] = END-OF-STRING "
                    << "rhs[i] = " << rhs[i] << "(" << (int)rhs[i] << ")"
                    << endl;
        return false;                                                 // RETURN

    }
    if (lhs.length() > rhs.length()) {
        bsl::size_t i = rhs.length();
        errorStream << "lhs: \"" << lhs << "\"\n"
                    << "rhs: \"" << rhs << "\"\n"
                    << "Strings differ at index (" << i << ") "
                    << "lhs[i] = " << lhs[i] << "(" << (int)lhs[i] << ") "
                    << "rhs[i] = END-OF-STRING"
                    << endl;
        return false;                                                 // RETURN
    }
    return true;

}

//=============================================================================
//                              MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test            = argc > 1 ? atoi(argv[1]) : 0;
    int verbose         = argc > 2;
    int veryVerbose     = argc > 3;
    int veryVeryVerbose = argc > 4;

    bslma::TestAllocator testAllocator(veryVeryVerbose);

    cout << "TEST " << __FILE__ << " CASE " << test << endl;;

    switch (test) { case 0:  // Zero is always the leading case.
      case 16: {
        // --------------------------------------------------------------------
        // TESTING USAGE EXAMPLE
        //
        // Concerns:
        //   The usage example provided in the component header file must
        //   compile, link, and run on all platforms as shown.
        //
        // Plan:
        //   Incorporate usage example from header into driver, remove leading
        //   comment characters, and replace 'assert' with 'ASSERT'.  Suppress
        //   all 'cout' statements in non-verbose mode, and add streaming to
        //   a buffer to test programmatically the printing examples.
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting usage example"
                          << "\n====================="
                          << endl;

        Rule rule("WEEKEND*",          // pattern
                  Severity::e_OFF,     // record level
                  Severity::e_INFO,    // pass-through level
                  Severity::e_OFF,     // trigger level
                  Severity::e_OFF);    // triggerAll level

        ManagedAttribute p1("myLib.uuid", 4044457);
        ManagedAttribute p2("myLib.name", "John Smith");
        rule.addAttribute(p1);

        ASSERT(true  == rule.hasAttribute(p1));
        ASSERT(false == rule.hasAttribute(p2));

        rule.addAttribute(p2);
        ASSERT(true  == rule.hasAttribute(p2));
        rule.removeAttribute(p1);
        ASSERT(false == rule.hasAttribute(p1));
        ASSERT(true  == rule.hasAttribute(p2));

        ASSERT(0 == strcmp(rule.pattern(), "WEEKEND*"));

        rule.setPattern("WEEKDAY*");
        ASSERT(0 == strcmp(rule.pattern(), "WEEKDAY*"));

        ASSERT(Severity::e_OFF  == rule.recordLevel());
        ASSERT(Severity::e_INFO == rule.passLevel());
        ASSERT(Severity::e_OFF  == rule.triggerLevel());
        ASSERT(Severity::e_OFF  == rule.triggerAllLevel());

        rule.setLevels(Severity::e_INFO,
                       Severity::e_OFF,
                       Severity::e_INFO,
                       Severity::e_INFO);

        ASSERT(Severity::e_INFO == rule.recordLevel());
        ASSERT(Severity::e_OFF  == rule.passLevel());
        ASSERT(Severity::e_INFO == rule.triggerLevel());
        ASSERT(Severity::e_INFO == rule.triggerAllLevel());

      } break;
      case 15: {
        // --------------------------------------------------------------------
        // TESTING 'isMatch'
        //
        // Concerns:
        //   Our concern is that 'isMatch' should match input strings
        //   correctly.
        //
        // Plan:
        //   Specify a set of pairs each of which consists of a 'Rule'
        //   objects and an input string.  For each pair, verify that
        //   'isMatch' returns the expected value.
        // Testing:
        //   bool isMatch(const char *inputString) const;
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting 'isMatch'"
                          << "\n================="
                          << endl;

        static const struct {
            int         d_line;         // line number
            const char *d_spec;         // spec for rule
            const char *d_input;        // input string
            bool        d_result;       // expected result
        } DATA[] = {
            // line   spec    pattern            expected value
            // ----   ----    -------            --------------
            {  L_,    "pA",   "",                1              },
            {  L_,    "pA",   "A",               0              },
            {  L_,    "pB",   "eq",              1              },
            {  L_,    "pB",   "equity",          0              },
            {  L_,    "pC",   "eq",              1              },
            {  L_,    "pC",   "equity",          1              },
            {  L_,    "pD",   "eq",              0              },
            {  L_,    "pD",   "equity",          0              },
            {  L_,    "pD",   "eq*",             1              },
            {  L_,    "pE",   "eq",              0              },
            {  L_,    "pE",   "equity",          0              },
            {  L_,    "pE",   "eq*",             0              },
            {  L_,    "pE",   "eq\\",            1              },
            {  L_,    "pF",   "eq",              0              },
            {  L_,    "pF",   "equity",          0              },
            {  L_,    "pF",   "eq*",             0              },
            {  L_,    "pF",   "eq\\",            0              },
            {  L_,    "pF",   "e*q",             1              },
            {  L_,    "pG",   "eq" ,             0              },
            {  L_,    "pG",   "equity",          0              },
            {  L_,    "pG",   "eq*",             0              },
            {  L_,    "pG",   "eq\\",            0              },
            {  L_,    "pG",   "e\\q",            1              },
        };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; i++) {
            int LINE = DATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            if (veryVerbose) { P_(X); P(DATA[i].d_input); }

            ASSERTV(LINE, DATA[i].d_result == X.isMatch(DATA[i].d_input));
        }

      } break;
      case 14: {
        // --------------------------------------------------------------------
        // TESTING HASH FUNCTION (VALUE):
        //   Verify the hash return value is constant across all platforms for
        //   a given input.
        //
        // Plan:
        //   Specifying a set of test vectors and verify the return value.
        //
        // Testing:
        //   static int hash(const Rule&, int size);
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting hash function"
                          << "\n====================="
                          << endl;

        static const struct {
            int         d_line;  // source line number
            const char *d_spec;  // string
            int         d_size;  // size of hash table
            int         d_hash;  // expected output
        } DATA[] = {
            //line    string                   size          hash
            //----    ------                   -----         ---
            {  L_,    "pALB",                  256,          168         },
            {  L_,    "pBLA",                  256,          133         },
            {  L_,    "pALAPA",                256,          118         },
            {  L_,    "pALAPB",                256,          12          },
            {  L_,    "pALBPA",                256,          53          },
            {  L_,    "pALBPB",                256,          203         },
            {  L_,    "pBLAPA",                256,          18          },
            {  L_,    "pBLAPB",                256,          168         },
            {  L_,    "pBLBPA",                256,          209         },
            {  L_,    "pBLBPB",                256,          103         },
            {  L_,    "pALAPAPC",              256,          152         },
            {  L_,    "pALAPBPC",              256,          46          },
            {  L_,    "pALBPAPC",              256,          87          },
            {  L_,    "pALBPBPC",              256,          237         },
            {  L_,    "pBLAPAPC",              256,          52          },
            {  L_,    "pBLAPBPC",              256,          202         },
            {  L_,    "pBLBPAPC",              256,          243         },
            {  L_,    "pBLBPBPC",              256,          137         },
            {  L_,    "pDLDPAPBPC",            256,          187         },
            {  L_,    "pDLDPAPBPCPD",          256,          213         },
            {  L_,    "pDLDPAPBPCPDPE",        256,          168         },
            {  L_,    "pDLDPAPBPCPDPEPF",      256,          200         },
            {  L_,    "pDLDPAPBPCPDPEPFPG",    256,          103         },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",  256,          85          },
            {  L_,    "pALB",                  65536,        53928       },
            {  L_,    "pBLA",                  65536,        32645       },
            {  L_,    "pALAPA",                65536,        12662       },
            {  L_,    "pALAPB",                65536,        38412       },
            {  L_,    "pALBPA",                65536,        18485       },
            {  L_,    "pALBPB",                65536,        44235       },
            {  L_,    "pBLAPA",                65536,        62738       },
            {  L_,    "pBLAPB",                65536,        22952       },
            {  L_,    "pBLBPA",                65536,        3025        },
            {  L_,    "pBLBPB",                65536,        28775       },
            {  L_,    "pALAPAPC",              65536,        53400       },
            {  L_,    "pALAPBPC",              65536,        13614       },
            {  L_,    "pALBPAPC",              65536,        59223       },
            {  L_,    "pALBPBPC",              65536,        19437       },
            {  L_,    "pBLAPAPC",              65536,        37940       },
            {  L_,    "pBLAPBPC",              65536,        63690       },
            {  L_,    "pBLBPAPC",              65536,        43763       },
            {  L_,    "pBLBPBPC",              65536,        3977        },
            {  L_,    "pDLDPAPBPC",            65536,        46267       },
            {  L_,    "pDLDPAPBPCPD",          65536,        22229       },
            {  L_,    "pDLDPAPBPCPDPE",        65536,        29096       },
            {  L_,    "pDLDPAPBPCPDPEPF",      65536,        27848       },
            {  L_,    "pDLDPAPBPCPDPEPFPG",    65536,        54631       },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",  65536,        11349       },
            {  L_,    "pALB",                  7,            3           },
            {  L_,    "pBLA",                  7,            2           },
            {  L_,    "pALAPA",                7,            6           },
            {  L_,    "pALAPB",                7,            4           },
            {  L_,    "pALBPA",                7,            5           },
            {  L_,    "pALBPB",                7,            3           },
            {  L_,    "pBLAPA",                7,            4           },
            {  L_,    "pBLAPB",                7,            2           },
            {  L_,    "pBLBPA",                7,            3           },
            {  L_,    "pBLBPB",                7,            1           },
            {  L_,    "pALAPAPC",              7,            0           },
            {  L_,    "pALAPBPC",              7,            5           },
            {  L_,    "pALBPAPC",              7,            6           },
            {  L_,    "pALBPBPC",              7,            4           },
            {  L_,    "pBLAPAPC",              7,            5           },
            {  L_,    "pBLAPBPC",              7,            3           },
            {  L_,    "pBLBPAPC",              7,            4           },
            {  L_,    "pBLBPBPC",              7,            2           },
            {  L_,    "pDLDPAPBPC",            7,            3           },
            {  L_,    "pDLDPAPBPCPD",          7,            1           },
            {  L_,    "pDLDPAPBPCPDPE",        7,            1           },
            {  L_,    "pDLDPAPBPCPDPEPF",      7,            0           },
            {  L_,    "pDLDPAPBPCPDPEPFPG",    7,            3           },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",  7,            0           },
            {  L_,    "pALB",                  1610612741,   982241950   },
            {  L_,    "pBLA",                  1610612741,   1160281979  },
            {  L_,    "pALAPA",                1610612741,   1313157479  },
            {  L_,    "pALAPB",                1610612741,   442209794   },
            {  L_,    "pALBPA",                1610612741,   1600866337  },
            {  L_,    "pALBPB",                1610612741,   729918652   },
            {  L_,    "pBLAPA",                1610612741,   168293625   },
            {  L_,    "pBLAPB",                1610612741,   907958681   },
            {  L_,    "pBLBPA",                1610612741,   456002483   },
            {  L_,    "pBLBPB",                1610612741,   1195667539  },
            {  L_,    "pALAPAPC",              1610612741,   1534185609  },
            {  L_,    "pALAPBPC",              1610612741,   663237924   },
            {  L_,    "pALBPAPC",              1610612741,   211281726   },
            {  L_,    "pALBPBPC",              1610612741,   950946782   },
            {  L_,    "pBLAPAPC",              1610612741,   389321755   },
            {  L_,    "pBLAPBPC",              1610612741,   1128986811  },
            {  L_,    "pBLBPAPC",              1610612741,   677030613   },
            {  L_,    "pBLBPBPC",              1610612741,   1416695669  },
            {  L_,    "pDLDPAPBPC",            1610612741,   1296872620  },
            {  L_,    "pDLDPAPBPCPD",          1610612741,   132142780   },
            {  L_,    "pDLDPAPBPCPDPE",        1610612741,   1024487818  },
            {  L_,    "pDLDPAPBPCPDPEPF",      1610612741,   1000238255  },
            {  L_,    "pDLDPAPBPCPDPEPFPG",    1610612741,   1088935241  },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",  1610612741,   1516973101  },
        };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; ++i) {
            int LINE = DATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;

            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            int hash = Obj::hash(X, DATA[i].d_size);
            if (veryVerbose) {
                cout << DATA[i].d_spec
                     << " , " << DATA[i].d_size
                     << " ---> " << hash << endl;
            }
            ASSERTV(LINE, DATA[i].d_hash, hash, DATA[i].d_hash == hash);
        }

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard guard;

            const Obj X;

            ASSERT_PASS(Obj::hash(X,  1));
            ASSERT_FAIL(Obj::hash(X,  0));
            ASSERT_FAIL(Obj::hash(X, -1));
        }

      } break;
      case 13: {
        // --------------------------------------------------------------------
        // TESTING 'evaluate':
        //   The 'evaluate' method must return correct results as to whether
        //   every attribute in the rule has an exact counterpart in the
        //   specified attribute set.
        //
        // Plan:
        //   Specify a set of pairs of a 'Rule' object and a
        //   'AttributeContainerList' object.  For each pair, verify that
        //   the 'evaluate' method returns the expected value.
        //
        // Testing:
        //   bool evaluate(const DefaultAttributeContainer& context)const;
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting 'evaluate'"
                          << "\n==================" << endl;

        const Attribute A1("A", "1");
        const Attribute A2("A", 1);
        const Attribute A3("A", 1LL);
        const Attribute A4("",  "");
        const Attribute A5("B", INT_MAX);
        const Attribute A6("B", LLONG_MAX);
        const Attribute A7("C", LLONG_MIN);
        const Attribute A8("a", 1);
        const Attribute A9("a", 1LL);

        DefaultAttributeContainer AS1;
        DefaultAttributeContainer AS2; AS2.addAttribute(A1);
        DefaultAttributeContainer AS3; AS3.addAttribute(A1);
                                       AS3.addAttribute(A2);
        DefaultAttributeContainer AS4; AS4.addAttribute(A1);
                                       AS4.addAttribute(A2);
                                       AS4.addAttribute(A3);
        DefaultAttributeContainer AS5; AS5.addAttribute(A4);
                                       AS5.addAttribute(A5);
                                       AS5.addAttribute(A6);
                                       AS5.addAttribute(A7);
        DefaultAttributeContainer AS6; AS6.addAttribute(A4);
                                       AS6.addAttribute(A5);
                                       AS6.addAttribute(A6);
                                       AS6.addAttribute(A7);
                                       AS6.addAttribute(A8);
                                       AS6.addAttribute(A9);

        static const struct {
            int                      d_line;         // source line number
            const char              *d_spec;         // spec for the rule
            const DefaultAttributeContainer
                                    *d_attributeSet; // attribute set
            bool                     d_result;       // expected result
        } DATA[] = {
            // line  rule               attributeSet   expected
            // ----  ------------       ------------   --------
            {  L_,   "",                &AS1,          1         },
            {  L_,   "",                &AS5,          1         },
            {  L_,   "PA",              &AS1,          0         },
            {  L_,   "PA",              &AS2,          1         },
            {  L_,   "PB",              &AS2,          0         },
            {  L_,   "PAPB",            &AS2,          0         },
            {  L_,   "PAPB",            &AS3,          1         },
            {  L_,   "PA",              &AS3,          1         },
            {  L_,   "PB",              &AS3,          1         },
            {  L_,   "PAPBPC",          &AS4,          1         },
            {  L_,   "PAPBPCPD",        &AS4,          0         },
            {  L_,   "PDPEPFPG",        &AS5,          1         },
            {  L_,   "PDPEPFPGPH",      &AS5,          0         },
            {  L_,   "PDPEPFPGPHPI",    &AS6,          1         },
            {  L_,   "PCPDPEPFPGPHPI",  &AS6,          0         },

        };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; ++i) {
            int LINE = DATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;

            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            const DefaultAttributeContainer *Y = DATA[i].d_attributeSet;
            AttributeContainerList list;
            const AttributeContainerList& LIST = list;;
            list.pushFront(Y);

            if (veryVerbose) { P_(X) P(Y); }

            ASSERTV(LINE, X.evaluate(LIST) == DATA[i].d_result);
        }
      } break;
      case 12: {
        // --------------------------------------------------------------------
        // TESTING 'removeAll'
        //   The 'removeAll' should effectively remove all attributes
        //   maintained by a 'Rule' object.
        //
        // Plan:
        //   Specify a set S of test vectors.  For each element in S,
        //   construct the corresponding 'Rule' object x using
        //   the 'gg' function.  Copy x into another object y.  After calling
        //   'removeAll' on x, verify that the length of x is zero, none of
        //   attributes in y can be found in x.  Then reconstruct x using the
        //   'gg' function again, and verify that x == y.
        //
        // Testing:
        //   void removeAll();
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTesting 'removeAll"
                          << "\n==================" << endl;

        static const char* SPECS[] = {
            "",
            "PA",
            "PB",
            "PC",
            "PAPB",
            "PAPC",
            "PBPC",
            "PAPBPC",
            "PAPBPCPD",
            "PAPBPCPDPE",
            "PAPBPCPDPEPF",
            "PAPBPCPDPEPFPG",
            "PAPBPCPDPEPFPGPH",
            "PAPBPCPDPEPFPGPHPI",
        };

        enum { NUM_SPECS = sizeof SPECS / sizeof *SPECS };

        for (int i = 0; i < NUM_SPECS; ++i) {
            for (int j = 0; j < NUM_SPECS; ++j) {

                if (veryVerbose) { P_(i) P_(j) P_(SPECS[i]) P(SPECS[j]); }

                Obj mX; const Obj& X = mX;
                ASSERTV(i, j, &mX == &gg(&mX, SPECS[i]));

                Obj mY; const Obj& Y = mY;
                ASSERTV(i, j, &mY == &gg(&mY, SPECS[j]));

                mX.removeAll();

                ASSERTV(i, j, 0 == X.numAttributes());
                for (ManagedAttributeSet::const_iterator iter = Y.begin();
                     iter != Y.end();
                     ++iter) {
                    ASSERTV(i, j, false == X.hasPredicate(*iter));
                }

                ASSERTV(i, j, &mX == &gg(&mX, SPECS[j]));
                ASSERTV(i, j, X == Y);
            }
        }

      } break;
      case 11: {
        // --------------------------------------------------------------------
        // TESTING 'begin' and 'end' methods
        //   This will test the 'begin' and 'end' methods.
        //
        // Concerns:
        //   The 'begin' and 'end' methods should return a range where each
        //   attribute in the attribute set appears exactly once.
        //
        // Plan:
        //   Construct an array consisting of 'ManagedAttribute' objects having
        //   distinct values.  For each n in [0 .. N] where N is the maximum
        //   number of attributes tested, create an empty 'Rule'
        //   object and add the first n attributes to the set.  Verify
        //   that every added attribute appears in the set exactly once.
        //
        // Testing:
        //   ManagedAttributeSet::const_iterator begin() const;
        //   ManagedAttributeSet::const_iterator end() const;
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting 'begin' and 'end' methods"
                          << "\n=================================" << endl;

        const ManagedAttribute PREDS[] = { P1, P2, P3, P4,
                                           P5, P6, P7, P8 };
        enum { NUM_PREDS = sizeof PREDS / sizeof *PREDS };

        int isPresentFlags[NUM_PREDS];

        for (int i = 0; i < NUM_PREDS; ++i) {
            Obj mX; const Obj& X = mX;

            int j, length;
            for (j = 0; j < i; ++j) {
                ASSERTV(i, j, 1 == mX.addAttribute(PREDS[j]));
                ASSERTV(i, j, X.hasAttribute(PREDS[j]));
                isPresentFlags[j] = 0;
            }

            ASSERTV(i, j == X.numAttributes());

            length = 0;
            for (ManagedAttributeSet::const_iterator iter = X.begin();
                 iter != X.end();
                 ++iter, ++length) {
                for (j = 0; j < i; ++j) {
                    if (*iter == PREDS[j]) {
                        ++isPresentFlags[j];
                    }
                }
            }

            ASSERTV(i, length == X.numAttributes());

            for (j = 0; j < i; ++j) {
                ASSERTV(i, j, 1 == isPresentFlags[j]);
            }
        }

      } break;
      case 10: {
        // --------------------------------------------------------------------
        // TESTING PATTERN/LEVELS CONSTRUCTOR
        //   The pattern/levels constructor must assign the specified value to
        //   the object.
        //
        // Plan:
        //   Specify a set S of patterns and a set T of threshold levels.  For
        //   each pair (x, y) in the cross product S X T, construct a
        //   corresponding 'Rule' object.  Verify that the object has the
        //   expected value.
        //
        // Testing:
        //   Rule(const char *pattern, int rl, int pl, int tl, int tal);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting Pattern/Levels Constructor'"
                          << "\n===================================" << endl;

        for (int i = 0; i < NUM_PATTERNS; ++i) {
            for (int j = 0; j < NUM_LEVELS; ++j) {
                const int RECORD_LEVEL      = LEVELS[j].d_recordLevel;
                const int PASS_LEVEL        = LEVELS[j].d_passLevel;
                const int TRIGGER_LEVEL     = LEVELS[j].d_triggerLevel;
                const int TRIGGER_ALL_LEVEL = LEVELS[j].d_triggerAllLevel;

                if (veryVerbose) {
                    P_(i);
                    P_(j);
                    P_(PATTERNS[i]);
                    P_(RECORD_LEVEL);
                    P_(PASS_LEVEL);
                    P_(TRIGGER_LEVEL);
                    P(TRIGGER_ALL_LEVEL);
                }

                const Obj X(PATTERNS[i],
                        LEVELS[j].d_recordLevel,
                        LEVELS[j].d_passLevel,
                        LEVELS[j].d_triggerLevel,
                        LEVELS[j].d_triggerAllLevel);

                ASSERTV(i, j, 0 == strcmp(PATTERNS[i], X.pattern()));

                ASSERTV(i, j, RECORD_LEVEL      == X.recordLevel());
                ASSERTV(i, j, PASS_LEVEL        == X.passLevel());
                ASSERTV(i, j, TRIGGER_LEVEL     == X.triggerLevel());
                ASSERTV(i, j, TRIGGER_ALL_LEVEL == X.triggerAllLevel());
            }
        }
      } break;
      case 9: {
        // --------------------------------------------------------------------
        // TESTING ASSIGNMENT OPERATOR
        //   Any value must be assignable to an object having any initial value
        //   without affecting the rhs operand value.  Also, any object must be
        //   assignable to itself.
        //
        // Plan:
        //   Specify a set S of (unique) objects with substantial and varied
        //   differences in value.  Construct and initialize all combinations
        //   (u, v) in the cross product S x S, copy construct a control w from
        //   v, assign v to u, and assert that w == u and w == v.  Then test
        //   aliasing by copy constructing a control w from each u in S,
        //   assigning u to itself, and verifying that w == u.
        //
        // Testing:
        //   const Rule& operator=(const Rule& other);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting Assignment Operator"
                          << "\n==========================" << endl;

        static const struct {
            int         d_line;   // source line number
            const char *d_spec;      // input 'spec' string for 'gg'
        } DATA[] = {
            // line    spec
            // ----    ----
            {  L_,    "pALB",                      },
            {  L_,    "pBLA",                      },

            {  L_,    "pALAPA",                    },
            {  L_,    "pALAPB",                    },
            {  L_,    "pALBPA",                    },
            {  L_,    "pALBPB",                    },
            {  L_,    "pBLAPA",                    },
            {  L_,    "pBLAPB",                    },
            {  L_,    "pBLBPA",                    },
            {  L_,    "pBLBPB",                    },

            {  L_,    "pALAPAPC",                  },
            {  L_,    "pALAPBPC",                  },
            {  L_,    "pALBPAPC",                  },
            {  L_,    "pALBPBPC",                  },
            {  L_,    "pBLAPAPC",                  },
            {  L_,    "pBLAPBPC",                  },
            {  L_,    "pBLBPAPC",                  },
            {  L_,    "pBLBPBPC",                  },

            {  L_,    "pDLDPAPBPC",                },
            {  L_,    "pDLDPAPBPCPD",              },
            {  L_,    "pDLDPAPBPCPDPE",            },
            {  L_,    "pDLDPAPBPCPDPEPF",          },
            {  L_,    "pDLDPAPBPCPDPEPFPG",        },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",      },
         };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; ++i) {
            const int LINE1 = DATA[i].d_line;

            for (int j = 0; j < NUM_DATA; ++j) {
                const int LINE2 = DATA[j].d_line;

                if (veryVerbose) {
                    P_(LINE1);
                    P_(DATA[i].d_spec);
                    P_(LINE2);
                    P(DATA[j].d_spec);
                }

                Obj mX(VA); const Obj& X = mX;
                ASSERTV(LINE1, &mX == &gg(&mX, DATA[i].d_spec));

                Obj mY(VA); const Obj& Y = mY;
                ASSERTV(LINE2, &mY == &gg(&mY, DATA[j].d_spec));

                Obj mW(Y); const Obj& W = mW;

                mX = Y;

                ASSERTV(LINE1, LINE2, Y == W);
                ASSERTV(LINE1, LINE2, W == Y);
                ASSERTV(LINE1, LINE2, X == W);
                ASSERTV(LINE1, LINE2, W == X);
            }
        }

        if (verbose) cout << "\nTesting assignment u = u (Aliasing)."
                          << endl;

        for (int i = 0; i < NUM_DATA; ++i) {
            const int LINE = DATA[i].d_line;

            if (veryVerbose) { P_(LINE); P_(DATA[i].d_spec); }

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            Obj mW(X); const Obj& W = mW;

            mX = X;

            ASSERTV(LINE, X == W);
            ASSERTV(LINE, W == X);
        }
      } break;
      case 8: {
        // --------------------------------------------------------------------
        // TESTING SECONDARY TEST APPARATUS:
        //   Void for 'Rule'.
        // --------------------------------------------------------------------

      } break;
      case 7: {
        // --------------------------------------------------------------------
        // TESTING COPY CONSTRUCTOR
        //   Any value must be able to be copy constructed without affecting
        //   its argument.
        //
        // Plan:
        //   Specify a set S whose elements have substantial and varied
        //   differences in value.  For each element in S, construct and
        //   initialize identically valued objects w and x using 'gg'.  Then
        //   copy construct an object y from x, and use the equality operator
        //   to assert that both x and y have the same value as w.
        //
        // Testing:
        //   Rule(const Rule&, bdema::Alct * = 0);
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting Copy Constructor"
                          << "\n========================" << endl;

        static const struct {
            int         d_line;      // source line number
            const char *d_spec;      // input 'spec' string for 'gg'
        } DATA[] = {
            // line    spec
            // ----    ----
            {  L_,    "pALB",                      },
            {  L_,    "pBLA",                      },

            {  L_,    "pALAPA",                    },
            {  L_,    "pALAPB",                    },
            {  L_,    "pALBPA",                    },
            {  L_,    "pALBPB",                    },
            {  L_,    "pBLAPA",                    },
            {  L_,    "pBLAPB",                    },
            {  L_,    "pBLBPA",                    },
            {  L_,    "pBLBPB",                    },

            {  L_,    "pALAPAPC",                  },
            {  L_,    "pALAPBPC",                  },
            {  L_,    "pALBPAPC",                  },
            {  L_,    "pALBPBPC",                  },
            {  L_,    "pBLAPAPC",                  },
            {  L_,    "pBLAPBPC",                  },
            {  L_,    "pBLBPAPC",                  },
            {  L_,    "pBLBPBPC",                  },

            {  L_,    "pDLDPAPBPC",                },
            {  L_,    "pDLDPAPBPCPD",              },
            {  L_,    "pDLDPAPBPCPDPE",            },
            {  L_,    "pDLDPAPBPCPDPEPF",          },
            {  L_,    "pDLDPAPBPCPDPEPFPG",        },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",      },
        };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; ++i) {
            const int LINE = DATA[i].d_line;

            if (veryVerbose) { P_(LINE) P(DATA[i].d_spec); }

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            Obj mW(VA); const Obj& W = mW;
            ASSERTV(LINE, &mW == &gg(&mW, DATA[i].d_spec));

            // construct y without an allocator
            {
                Obj mY(X); const Obj& Y = mY;

                ASSERTV(LINE, Y == W);
                ASSERTV(LINE, W == Y);
                ASSERTV(LINE, X == W);
                ASSERTV(LINE, W == X);
            }

            // construct y with an allocator but no exception
            {
                bslma::TestAllocator testAllocatorY(veryVeryVerbose);
                Obj mY(X, &testAllocatorY); const Obj& Y = mY;

                ASSERTV(LINE, Y == W);
                ASSERTV(LINE, W == Y);
                ASSERTV(LINE, X == W);
                ASSERTV(LINE, W == X);
            }

            // construct y with an allocator and exceptions
            {
                bslma::TestAllocator testAllocatorY(veryVeryVerbose);

                BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(testAllocator) {
                    Obj mY(X, &testAllocatorY); const Obj& Y = mY;

                    ASSERTV(LINE, Y == W);
                    ASSERTV(LINE, W == Y);
                    ASSERTV(LINE, X == W);
                    ASSERTV(LINE, W == X);
                } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END
            }
        }
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING EQUALITY OPERATORS
        //   Any subtle variation in value must be detected by the equality
        //   operators.
        //
        // Plan:
        //   First, specify a set S of unique object values that may have
        //   various minor or subtle differences.  Verify the correctness of
        //   'operator==' and 'operator!=' using all elements (u, v) of the
        //   cross product S X S.
        //
        //   Next, specify another set T where each element is a pair of
        //   different specifications having the same value (the same
        //   attributes were added in different orders).  For each element (u,
        //   v) in T, verify that 'operator==' and 'operator!=' return the
        //   correct value.

        // Testing:
        //   bool operator==(const Rule& lhs, const Rule& rhs)
        //   bool operator!=(const Rule& lhs, const Rule& rhs)
        // --------------------------------------------------------------------

        static const struct {
            int         d_line;      // source line number
            const char *d_spec;      // input 'spec' string for 'gg'
        } DATA[] = {
            // line   spec
            // ----   ----
            {  L_,    "pALB",                      },
            {  L_,    "pBLA",                      },

            {  L_,    "pALAPA",                    },
            {  L_,    "pALAPB",                    },
            {  L_,    "pALBPA",                    },
            {  L_,    "pALBPB",                    },
            {  L_,    "pBLAPA",                    },
            {  L_,    "pBLAPB",                    },
            {  L_,    "pBLBPA",                    },
            {  L_,    "pBLBPB",                    },

            {  L_,    "pALAPAPC",                  },
            {  L_,    "pALAPBPC",                  },
            {  L_,    "pALBPAPC",                  },
            {  L_,    "pALBPBPC",                  },
            {  L_,    "pBLAPAPC",                  },
            {  L_,    "pBLAPBPC",                  },
            {  L_,    "pBLBPAPC",                  },
            {  L_,    "pBLBPBPC",                  },

            {  L_,    "pDLDPAPBPC",                },
            {  L_,    "pDLDPAPBPCPD",              },
            {  L_,    "pDLDPAPBPCPDPE",            },
            {  L_,    "pDLDPAPBPCPDPEPF",          },
            {  L_,    "pDLDPAPBPCPDPEPFPG",        },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH",      },
        };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        if (verbose) cout << "\nTesting Equality Operators"
                          << "\n==========================" << endl;

        if (verbose) cout <<
            "\nCompare each pair of values (u, v) in S X S." << endl;

        for (int i = 0; i < NUM_DATA; ++i) {
            const int LINE1 = DATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE1, &mX == &gg(&mX, DATA[i].d_spec));

            for (int j = 0; j < NUM_DATA; ++j) {
                const int LINE2 = DATA[j].d_line;

                if (veryVerbose) {
                    P_(LINE1);
                    P_(DATA[i].d_spec);
                    P_(LINE2);
                    P(DATA[j].d_spec);
                }

                Obj mY(VA); const Obj& Y = mY;
                ASSERTV(LINE1, &mY == &gg(&mY, DATA[j].d_spec));

                ASSERT((i == j) == (X == Y));
                ASSERT((i != j) == (X != Y));
            }
        }

        if (verbose) cout <<
            "\nCompare each pair of values (u, v) in T." << endl;

        static const struct {
            int         d_line;       // source line number
            const char *d_spec1;      // input 'spec' string for 'gg'
            const char *d_spec2;      // input 'spec' string for 'gg'
        } TDATA[] = {
            // line    spec1                 spec2
            // ----    -----                 -----
            {  L_,    "pBLB",                "pALApBLB"                  },

            {  L_,    "pALAPAPC",            "pALAPCPA"                  },
            {  L_,    "pALAPBPC",            "pALAPCPB"                  },
            {  L_,    "pALBPAPC",            "pALBPCPA"                  },
            {  L_,    "pALBPBPC",            "pALBPCPB"                  },
            {  L_,    "pBLAPAPC",            "pBLAPCPA"                  },
            {  L_,    "pBLAPBPC",            "pBLAPCPB"                  },
            {  L_,    "pBLBPAPC",            "pBLBPCPA"                  },
            {  L_,    "pBLBPBPC",            "pBLBPCPB"                  },

            {  L_,    "pDLDPAPBPC",          "pDLDPCPBPA"                },
            {  L_,    "pDLDPAPBPCPD",        "pDLDPDPCPBPA"              },
            {  L_,    "pDLDPAPBPCPDPE",      "pDLDPEPDPCPBPA"            },
            {  L_,    "pDLDPAPBPCPDPEPF",    "pDLDPFPEPDPCPBPA"          },
            {  L_,    "pDLDPAPBPCPDPEPFPG",  "pDLDPGPFPEPDPCPBPA"        },
            {  L_,    "pDLDPAPBPCPDPEPFPGPH","pDLDPHPGPFPEPDPCPBPA"      },
        };

        enum { NUM_TDATA = sizeof TDATA / sizeof *TDATA };

        if (verbose) cout <<
            "\nCompare each pair of values (u, v) in T." << endl;

        for (int i = 0; i < NUM_TDATA; ++i) {
            const int LINE = TDATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, TDATA[i].d_spec1));

            Obj mY(VA); const Obj& Y = mY;
            ASSERTV(LINE, &mY == &gg(&mY, TDATA[i].d_spec2));

            if (veryVerbose) {
                P_(LINE);
                P_(TDATA[i].d_spec1);
                P(TDATA[i].d_spec2);
            }

            ASSERT(1 == (X == Y));
            ASSERT(0 == (X != Y));
        }
      } break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING 'operator<<' AND 'print'
        //   The output operator and 'print' method should print out the value
        //   of objects in the expected format.
        //
        // Plan:
        //   For each of a small representative set of object values, use
        //   'ostrstream' to write that object's value to a character buffer
        //   and then compare the contents of that buffer with the expected
        //   output format.
        //
        // Testing:
        //   bsl::ostream& operator<<(bsl::ostream&, const PS&) const;
        //   bsl::ostream& print(bsl::ostream& stream, int lvl, int spl) const;
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "Testing 'operator<<' and 'print'" << endl
                          << "================================" << endl;

        if (verbose) cout << "\nTesting 'operator<<' (ostream)." << endl;

        static const struct {
            int         d_line;    // line number
            const char *d_spec;    // spec
            const char *d_output;  // expected output format
        } DATA[] = {
  // line   spec          expected output
  // ----   ----          ---------------
  {  L_,    "pALA",        "[ pattern = \"\" "
                           "thresholds = [   0   0   0   0  ] "
                           " attributeSet = [ ] ]" },
  {  L_,    "pDLDPC",      "[ pattern = \"eq\\*\" "
                           "thresholds = [   0   0   1   0  ]  "
                           "attributeSet = [  [ \"A\" = 1 ] ] ]" },
       };

        enum { NUM_DATA = sizeof DATA / sizeof *DATA };

        for (int i = 0; i < NUM_DATA; ++i) {
            const int LINE = DATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, DATA[i].d_spec));

            ostringstream os;
            os << X;
            ASSERTV(LINE, compareText(os.str(), DATA[i].d_output));

            if (veryVerbose) {
                P_(LINE);
                P_(DATA[i].d_spec);
                P_(DATA[i].d_output);
                P(os.str());
            }
        }

        static const struct {
            int         d_line;            // line number
            const char *d_spec;            // spec for gg
            int         d_level;           // tab level
            int         d_spacesPerLevel;  // spaces per level
           const char  *d_output;          // expected output format
        } PDATA[] = {
            // line spec     level space expected
            // ---- ----     ----- ----- -------------------------------
            {  L_,  "pELEPC", 1,   2,    "  [\n"
                                         "    pattern = \"eq\\\\\"\n"
                                         "    thresholds = [\n"
                                         "      0\n"
                                         "      0\n"
                                         "      0\n"
                                         "      1\n"
                                         "    ]\n"
                                         "    attributeSet = [\n"
                                         "              [ \"A\" = 1 ]\n"
                                         "    ]\n"
                                         "  ]\n"  },
        };

        enum { NUM_PDATA = sizeof PDATA / sizeof *PDATA };

        if (verbose) cout << "\nTesting 'print'." << endl;

        for (int i = 0; i < NUM_PDATA; ++i) {
            int LINE = PDATA[i].d_line;

            Obj mX(VA); const Obj& X = mX;
            ASSERTV(LINE, &mX == &gg(&mX, PDATA[i].d_spec));

            ostringstream os;
            X.print(os, PDATA[i].d_level, PDATA[i].d_spacesPerLevel);

            if (veryVerbose) {
                P_(LINE);
                P_(PDATA[i].d_spec);
                P(PDATA[i].d_output);
                P_(os.str()); cout << endl;
            }

            ASSERTV(LINE, compareText(os.str(), PDATA[i].d_output));
        }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING BASIC ACCESSORS
        //   The pattern of the 'Rule' object must be verifiable by the
        //   'pattern' accessor.  The threshold levels must be verifiable by
        //   the threshold level accessors.  Every attribute added must be
        //   verifiable by the 'hasPredicate' accessor.
        //
        // Plan:
        //   Mechanically generate a series of specifications that are the
        //   cross product of the predefined pattern, level, and attribute
        //   arrays.  For each specification, create a 'Rule' object from
        //   the specification using the 'gg' function, and verify that each
        //   accessor returns the expected result.
        //
        // Testing:
        //   const char *pattern() const;
        //   int recordLevel() const;
        //   int passLevel() const;
        //   int triggerLevel() const;
        //   int triggerAllLevel() const;
        //   bool hasPredicate(const ManagedAttribute&) const;
        //   int numAttributes() const;
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "Testing Basic Accessors" << endl
                          << "=======================" << endl;

        for (int i = 0; i < NUM_PATTERNS; ++i) {
            for (int j = 0; j < NUM_LEVELS; ++j) {
                for (int k = 0; k < NUM_PREDICATES; ++k) {
                    const int RECORD_LEVEL      = LEVELS[j].d_recordLevel;
                    const int PASS_LEVEL        = LEVELS[j].d_passLevel;
                    const int TRIGGER_LEVEL     = LEVELS[j].d_triggerLevel;
                    const int TRIGGER_ALL_LEVEL = LEVELS[j].d_triggerAllLevel;

                    bsl::string spec;
                    spec += 'p';
                    spec += (char)(i + 'A');  // set the pattern
                    spec += 'L';
                    spec += (char)(j + 'A');  // the threshold levels
                    spec += 'P';
                    spec += (char)(k + 'A');  // add a attribute

                    Obj mX(VA); const Obj& X = mX;
                    ASSERTV(i, j, k, &mX == &gg(&mX, spec.c_str()));

                    if (veryVerbose) { P_(i) P_(j) P_(k) P_(spec) P(X); }

                    ASSERTV(i, j, k, 0 == strcmp(X.pattern(), PATTERNS[i]));

                    ASSERTV(i, j, k, RECORD_LEVEL      == X.recordLevel());
                    ASSERTV(i, j, k, PASS_LEVEL        == X.passLevel());
                    ASSERTV(i, j, k, TRIGGER_LEVEL     == X.triggerLevel());
                    ASSERTV(i, j, k, TRIGGER_ALL_LEVEL == X.triggerAllLevel());

                    ASSERTV(i, j, k, X.numAttributes() == 1);
                    ASSERTV(i, j, k, X.hasPredicate(PREDICATES[k]));
                }
            }
        }
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING GENERATOR FUNCTIONS 'gg'
        //   The 'gg' function should create objects having the expected value
        //   and return a reference to its first argument.
        //
        // Plan:
        //   Mechanically generate a series of specifications that are the
        //   cross product of the predefined pattern, level, and attribute
        //   arrays.  For each specification, create a 'Rule' object from
        //   the specification using the 'gg' function, and verify that the
        //   object has the expected value.
        //
        // Testing:
        //   Obj& gg(Obj *address, const char *spec);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
            << "Testing 'gg' generator functions" << endl
            << "================================" << endl;

        for (int i = 0; i < NUM_PATTERNS; ++i) {
            for (int j = 0; j < NUM_LEVELS; ++j) {
                for (int k = 0; k < NUM_PREDICATES; ++k) {
                    const int RECORD_LEVEL      = LEVELS[j].d_recordLevel;
                    const int PASS_LEVEL        = LEVELS[j].d_passLevel;
                    const int TRIGGER_LEVEL     = LEVELS[j].d_triggerLevel;
                    const int TRIGGER_ALL_LEVEL = LEVELS[j].d_triggerAllLevel;

                    bsl::string spec;
                    spec += 'p';
                    spec += (char)(i + 'A');  // set the pattern
                    spec += 'L';
                    spec += (char)(j + 'A');  // the threshold levels
                    spec += 'P';
                    spec += (char)(k + 'A');  // add a attribute

                    Obj mX(VA); const Obj& X = mX;
                    ASSERTV(i, j, k, &mX == &gg(&mX, spec.c_str()));

                    if (veryVerbose) { P_(i); P_(j); P_(k); P_(spec); P(X); }

                    ASSERTV(i, j, k, 0 == strcmp(X.pattern(), PATTERNS[i]));

                    ASSERTV(i, j, k, RECORD_LEVEL      == X.recordLevel());
                    ASSERTV(i, j, k, PASS_LEVEL        == X.passLevel());
                    ASSERTV(i, j, k, TRIGGER_LEVEL     == X.triggerLevel());
                    ASSERTV(i, j, k, TRIGGER_ALL_LEVEL == X.triggerAllLevel());

                    ASSERTV(i, j, k, X.numAttributes() == 1);
                    ASSERTV(i, j, k, X.hasPredicate(PREDICATES[k]));
                }
            }
        }
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING PRIMARY MANIPULATORS (BOOTSTRAP)
        //   Primary manipulators should correctly set the object to the
        //   specified value.
        //
        // Plan:
        //   For a sequence of independent test values, use the constructor to
        //   create an object and use the primary manipulators to set its
        //   value.  For each value, verify, using the basic accessors, that
        //   the value has been assigned to the object.  The destructor is
        //   tested as objects in various states go out of scope.
        //
        // Testing:
        //   Rule();
        //   int addAttribute(const ManagedAttribute& attribute);
        //   int removeAttribute(const ManagedAttribute& attribute);
        //   void setPattern(const char *value);
        //   int setLevels(int rl, int pl, int tl, int tal);
        //   ~Rule();
        // --------------------------------------------------------------------
        if (verbose) cout << "\nTesting Primary Manipulator"
                          << "\n===========================" << endl;

        if (verbose) cout << "\nTesting constructor (thoroughly)." << endl;

        if (verbose) cout << "\tWithout passing in an allocator." << endl;
        {
            const Obj X((bslma::Allocator *)0);
            ASSERT(0 == strcmp("", X.pattern()));
            ASSERT(Levels(0, 0, 0, 0) == Levels(X.recordLevel(),
                                                X.passLevel(),
                                                X.triggerLevel(),
                                                X.triggerAllLevel()));
            ASSERT(0 == X.numAttributes());
            if (veryVerbose) { cout << "\t\t"; P(X); }
        }

        if (verbose) cout << "\tPassing in an allocator." << endl;

        if (verbose) cout << "\t\tWith no exceptions." << endl;
        {
            const Obj X(&testAllocator);
            ASSERT(0 == strcmp("", X.pattern()));
            ASSERT(Levels(0, 0, 0, 0) == Levels(X.recordLevel(),
                                                X.passLevel(),
                                                X.triggerLevel(),
                                                X.triggerAllLevel()));
            ASSERT(0 == X.numAttributes());
            if (veryVerbose) { cout << "\t\t"; P(X); }
        }

        if (verbose) cout << "\t\tWith exceptions." << endl;
        {
          BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(testAllocator) {
            if (veryVerbose) cout <<
                "\tTesting Exceptions In Default Ctor" << endl;
            const Obj X(&testAllocator);
            ASSERT(0 == strcmp("", X.pattern()));
            ASSERT(Levels(0, 0, 0, 0) == Levels(X.recordLevel(),
                                                X.passLevel(),
                                                X.triggerLevel(),
                                                X.triggerAllLevel()));
            ASSERT(0 == X.numAttributes());
            if (veryVerbose) { cout << "\t\t"; P(X); }
          } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END
        }

        if (verbose) cout << "\nTesting primary manipulator." << endl;

        if (verbose) cout << "\tWithout passing in an allocator." << endl;
        {
            Obj mX(VA);  const Obj& X = mX;

            Obj mY(VB);  const Obj& Y = mY;
            ASSERT(1 == mY.addAttribute(P1));

            Obj mZ(VC);  const Obj& Z = mZ;
            ASSERT(1 == mZ.addAttribute(P2));
            ASSERT(1 == mZ.addAttribute(P3));

            if (veryVeryVerbose) { P(mX); P(mY); P(mZ); }

            ASSERT(0 == (X == Y));
            ASSERT(0 == (X == Z));
            ASSERT(0 == (Y == Z));

            ASSERT(0 == X.numAttributes());
            ASSERT(1 == Y.numAttributes());
            ASSERT(2 == Z.numAttributes());
            ASSERT(Y.hasPredicate(P1));
            ASSERT(Z.hasPredicate(P2));
            ASSERT(Z.hasPredicate(P3));

            if (veryVerbose) cout << "\tSetting mX with mY's initializer."
                            << endl;
            mX.setPattern(VB_PATTERN);
            mX.setLevels(VB_LEVELS);
            ASSERT(1 == mX.addAttribute(P1));
            ASSERT(1 == X.numAttributes());
            ASSERT(X.hasPredicate(P1));
            ASSERT(X == Y);

            if (veryVerbose) cout << "\tSetting mX with mZ's initializer."
                            << endl;
            mX.setPattern(VC_PATTERN);
            mX.setLevels(VC_LEVELS);
            ASSERT(1 == mX.removeAttribute(P1));
            ASSERT(false == X.hasPredicate(P1));
            ASSERT(1 == mX.addAttribute(P2));
            ASSERT(1 == mX.addAttribute(P3));
            ASSERT(2 == X.numAttributes());
            ASSERT(X.hasPredicate(P2));
            ASSERT(X.hasPredicate(P3));
            ASSERT(X == Z);
        }

        if (verbose) cout << "\tWith an allocator." << endl;
        if (verbose) cout << "\t\tWithout exceptions." << endl;
        {
            bslma::TestAllocator testAllocatorX(veryVeryVerbose);
            bslma::TestAllocator testAllocatorY(veryVeryVerbose);
            bslma::TestAllocator testAllocatorZ(veryVeryVerbose);

            Obj mX(VA, &testAllocatorX);  const Obj& X = mX;

            Obj mY(VB, &testAllocatorY);  const Obj& Y = mY;
            ASSERT(1 == mY.addAttribute(P1));

            Obj mZ(VC, &testAllocatorZ);  const Obj& Z = mZ;
            ASSERT(1 == mZ.addAttribute(P2));
            ASSERT(1 == mZ.addAttribute(P3));

            if (veryVeryVerbose) { P(mX); P(mY); P(mZ); }

            ASSERT(0 == (X == Y));
            ASSERT(0 == (X == Z));
            ASSERT(0 == (Y == Z));

            ASSERT(0 == X.numAttributes());
            ASSERT(1 == Y.numAttributes());
            ASSERT(2 == Z.numAttributes());
            ASSERT(Y.hasPredicate(P1));
            ASSERT(Z.hasPredicate(P2));
            ASSERT(Z.hasPredicate(P3));

            if (veryVerbose) cout << "\tSetting mX with mY's initializer."
                            << endl;
            mX.setPattern(VB_PATTERN);
            mX.setLevels(VB_LEVELS);
            ASSERT(1 == mX.addAttribute(P1));
            ASSERT(1 == X.numAttributes());
            ASSERT(X.hasPredicate(P1));
            ASSERT(X == Y);

            if (veryVerbose) cout << "\tSetting mX with mZ's initializer."
                            << endl;
            mX.setPattern(VC_PATTERN);
            mX.setLevels(VC_LEVELS);
            ASSERT(1 == mX.removeAttribute(P1));
            ASSERT(false == X.hasPredicate(P1));
            ASSERT(1 == mX.addAttribute(P2));
            ASSERT(1 == mX.addAttribute(P3));
            ASSERT(2 == X.numAttributes());
            ASSERT(X.hasPredicate(P2));
            ASSERT(X.hasPredicate(P3));
            ASSERT(X == Z);
        }

        if (verbose) cout << "\t\tWith exceptions." << endl;
        {
            bslma::TestAllocator testAllocatorX(veryVeryVerbose);
            bslma::TestAllocator testAllocatorY(veryVeryVerbose);
            bslma::TestAllocator testAllocatorZ(veryVeryVerbose);

          BSLMA_TESTALLOCATOR_EXCEPTION_TEST_BEGIN(testAllocator) {
            if (veryVerbose) cout <<
                "\tTesting Exceptions In Primary Manipulator" << endl;

            Obj mX(VA, &testAllocatorX);  const Obj& X = mX;

            Obj mY(VB, &testAllocatorY);  const Obj& Y = mY;
            ASSERT(1 == mY.addAttribute(P1));

            Obj mZ(VC, &testAllocatorZ);  const Obj& Z = mZ;
            ASSERT(1 == mZ.addAttribute(P2));
            ASSERT(1 == mZ.addAttribute(P3));

            if (veryVeryVerbose) { P(mX); P(mY); P(mZ); }

            ASSERT(0 == (X == Y));
            ASSERT(0 == (X == Z));
            ASSERT(0 == (Y == Z));

            ASSERT(0 == X.numAttributes());
            ASSERT(1 == Y.numAttributes());
            ASSERT(2 == Z.numAttributes());
            ASSERT(Y.hasPredicate(P1));
            ASSERT(Z.hasPredicate(P2));
            ASSERT(Z.hasPredicate(P3));

            if (veryVerbose) cout << "\tSetting mX with mY's initializer."
                            << endl;
            mX.setPattern(VB_PATTERN);
            mX.setLevels(VB_LEVELS);
            ASSERT(1 == mX.addAttribute(P1));
            ASSERT(1 == X.numAttributes());
            ASSERT(X.hasPredicate(P1));
            ASSERT(X == Y);

            if (veryVerbose) cout << "\tSetting mX with mZ's initializer."
                            << endl;
            mX.setPattern(VC_PATTERN);
            mX.setLevels(VC_LEVELS);
            ASSERT(1 == mX.removeAttribute(P1));
            ASSERT(false == X.hasPredicate(P1));
            ASSERT(1 == mX.addAttribute(P2));
            ASSERT(1 == mX.addAttribute(P3));
            ASSERT(2 == X.numAttributes());
            ASSERT(X.hasPredicate(P2));
            ASSERT(X.hasPredicate(P3));
            ASSERT(X == Z);

          } BSLMA_TESTALLOCATOR_EXCEPTION_TEST_END
        }

      } break;
      case 1: {
       // --------------------------------------------------------------------
        // BREATHING TEST
        //   Exercise a broad cross-section of value-semantic functionality
        //   before beginning testing in earnest.  Probe that functionality
        //   systematically and incrementally to discover basic errors in
        //   isolation.
        //
        // Plan:
        //   Create four test objects by using the initializing and copy
        //   constructors.  Exercise the basic value-semantic methods and the
        //   equality operators using these test objects.  Invoke the primary
        //   manipulator [3, 6], copy constructor [2, 8], and assignment
        //   operator without [9, 10] and with [11] aliasing.  Use the direct
        //   accessors to verify the expected results.  Display object values
        //   frequently in verbose mode.  Note that 'VA', 'VB', 'VC', and 'VD'
        //   denote unique, but otherwise arbitrary, object values.
        //
        // 1.  Create an object x1 using VA.        { x1:VA }
        // 2.  Create an object x2 (copy from x1).  { x1:VA x2:VA }
        // 3.  Set x1 to VB.                        { x1:VB x2:VA }
        // 4.  Set x2 to VB.                        { x1:VB x2:VB }
        // 5.  Set x2 to VC.                        { x1:VB x2:VC }
        // 6.  Set x1 to VA.                        { x1:VA x2:VC }
        // 7.  Create an object x3 (with value VD). { x1:VA x2:VC x3:VD }
        // 8.  Create an object x4 (copy from x1).  { x1:VA x2:VC x3:VD x4:VA }
        // 9.  Assign x2 = x1.                      { x1:VA x2:VA x3:VD x4:VA }
        // 10. Assign x2 = x3.                      { x1:VA x2:VD x3:VD x4:VA }
        // 11. Assign x1 = x1 (aliasing).           { x1:VA x2:VD x3:VD x4:VA }
        //
        // Testing:
        //   This Test Case exercises basic value-semantic functionality.
        // --------------------------------------------------------------------

        if (verbose) cout << "\nBREATHING TEST"
                          << "\n==============" << endl;

        if (verbose) cout << "\n 1. Create an object x1 using VA." << endl;

        Obj mX1(VA);  const Obj& X1 = mX1;

        ASSERT(0 == strcmp(X1.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X1.recordLevel(),
                                           X1.passLevel(),
                                           X1.triggerLevel(),
                                           X1.triggerAllLevel()));
        ASSERT(0 == X1.numAttributes());

        if (verbose) cout << "\n 2. Create an object x2 (copy from x1)."
                          << endl;

        Obj mX2(X1);  const Obj& X2 = mX2;

        ASSERT(0 == strcmp(X2.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X2.recordLevel(),
                                           X2.passLevel(),
                                           X2.triggerLevel(),
                                           X2.triggerAllLevel()));
        ASSERT(0 == X2.numAttributes());

        ASSERT(1 == (X2 == X2));        ASSERT(0 == (X2 != X2));
        ASSERT(1 == (X1 == X2));        ASSERT(0 == (X1 != X2));

        if (verbose) cout << "\n 3. Set x1 to VB." << endl;

        mX1.setPattern(VB_PATTERN);
        mX1.setLevels(VB_LEVELS);
        ASSERT(1 == mX1.addAttribute(P1));

        ASSERT(0 == strcmp(X1.pattern(), VB_PATTERN));
        ASSERT(Levels(VB_LEVELS) == Levels(X1.recordLevel(),
                                           X1.passLevel(),
                                           X1.triggerLevel(),
                                           X1.triggerAllLevel()));
        ASSERT(1 == X1.numAttributes());
        ASSERT(X1.hasPredicate(P1));

        ASSERT(1 == (X1 == X1));        ASSERT(0 == (X1 != X1));
        ASSERT(0 == (X1 == X2));        ASSERT(1 == (X1 != X2));

        if (verbose) cout << "\n 4. Set x2 to VB." << endl;

        mX2.setPattern(VB_PATTERN);
        mX2.setLevels(VB_LEVELS);
        ASSERT(1 == mX2.addAttribute(P1));

        ASSERT(0 == strcmp(X2.pattern(), VB_PATTERN));
        ASSERT(Levels(VB_LEVELS) == Levels(X2.recordLevel(),
                                           X2.passLevel(),
                                           X2.triggerLevel(),
                                           X2.triggerAllLevel()));
        ASSERT(1 == X2.numAttributes());
        ASSERT(X2.hasPredicate(P1));

        ASSERT(1 == (X2 == X2));        ASSERT(0 == (X2 != X2));
        ASSERT(1 == (X1 == X2));        ASSERT(0 == (X1 != X2));

        if (verbose) cout << "\n 5. Set x2 to VC." << endl;

        mX2.setPattern(VC_PATTERN);
        mX2.setLevels(VC_LEVELS);
        ASSERT(1 == mX2.removeAttribute(P1));
        ASSERT(1 == mX2.addAttribute(P2));
        ASSERT(1 == mX2.addAttribute(P3));

        ASSERT(0 == strcmp(X2.pattern(), VC_PATTERN));
        ASSERT(Levels(VC_LEVELS) == Levels(X2.recordLevel(),
                                           X2.passLevel(),
                                           X2.triggerLevel(),
                                           X2.triggerAllLevel()));
        ASSERT(2 == X2.numAttributes());
        ASSERT(X2.hasPredicate(P2));
        ASSERT(X2.hasPredicate(P3));

        ASSERT(1 == (X2 == X2));        ASSERT(0 == (X2 != X2));
        ASSERT(0 == (X1 == X2));        ASSERT(1 == (X1 != X2));

        if (verbose) cout << "\n 6. Set x1 to VA." << endl;

        mX1.setPattern(VA_PATTERN);
        mX1.setLevels(VA_LEVELS);
        ASSERT(1 == mX1.removeAttribute(P1));

        ASSERT(0 == strcmp(X1.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X1.recordLevel(),
                                           X1.passLevel(),
                                           X1.triggerLevel(),
                                           X1.triggerAllLevel()));
        ASSERT(0 == X1.numAttributes());

        ASSERT(1 == (X1 == X1));        ASSERT(0 == (X1 != X1));
        ASSERT(0 == (X1 == X2));        ASSERT(1 == (X1 != X2));

        if (verbose) cout << "\n 7. Create an object x3 (with value VD)."
                           << endl;

        Obj mX3(VD);  const Obj& X3 = mX3;
        ASSERT(1 == mX3.addAttribute(P1));
        ASSERT(1 == mX3.addAttribute(P2));
        ASSERT(1 == mX3.addAttribute(P3));

        ASSERT(0 == strcmp(X3.pattern(), VD_PATTERN));
        ASSERT(Levels(VD_LEVELS) == Levels(X3.recordLevel(),
                                           X3.passLevel(),
                                           X3.triggerLevel(),
                                           X3.triggerAllLevel()));
        ASSERT(3 == X3.numAttributes());
        ASSERT(X3.hasPredicate(P1));
        ASSERT(X3.hasPredicate(P2));
        ASSERT(X3.hasPredicate(P3));

        ASSERT(1 == (X3 == X3));        ASSERT(0 == (X3 != X3));
        ASSERT(0 == (X3 == X1));        ASSERT(1 == (X3 != X1));
        ASSERT(0 == (X3 == X2));        ASSERT(1 == (X3 != X2));

        if (verbose) cout << "\n 8. Create an object x4 (copy from x1)."
                           << endl;

        Obj mX4(X1);  const Obj& X4 = mX4;

        ASSERT(0 == strcmp(X4.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X4.recordLevel(),
                                           X4.passLevel(),
                                           X4.triggerLevel(),
                                           X4.triggerAllLevel()));
        ASSERT(0 == X4.numAttributes());

        ASSERT(1 == (X4 == X1));        ASSERT(0 == (X4 != X1));
        ASSERT(0 == (X4 == X2));        ASSERT(1 == (X4 != X2));
        ASSERT(0 == (X4 == X3));        ASSERT(1 == (X4 != X3));
        ASSERT(1 == (X4 == X4));        ASSERT(0 == (X4 != X4));

        if (verbose) cout << "\n 9. Assign x2 = x1." << endl;

        mX2 = X1;

        ASSERT(0 == strcmp(X2.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X2.recordLevel(),
                                           X2.passLevel(),
                                           X2.triggerLevel(),
                                           X2.triggerAllLevel()));
        ASSERT(0 == X2.numAttributes());

        ASSERT(1 == (X2 == X1));        ASSERT(0 == (X2 != X1));
        ASSERT(1 == (X2 == X2));        ASSERT(0 == (X2 != X2));
        ASSERT(0 == (X2 == X3));        ASSERT(1 == (X2 != X3));
        ASSERT(1 == (X2 == X4));        ASSERT(0 == (X2 != X4));

        if (verbose) cout << "\n 10. Assign x2 = x3." << endl;

        mX2 = X3;

        ASSERT(0 == strcmp(X2.pattern(), VD_PATTERN));
        ASSERT(Levels(VD_LEVELS) == Levels(X2.recordLevel(),
                                           X2.passLevel(),
                                           X2.triggerLevel(),
                                           X2.triggerAllLevel()));
        ASSERT(3 == X2.numAttributes());
        ASSERT(X2.hasPredicate(P1));
        ASSERT(X2.hasPredicate(P2));
        ASSERT(X2.hasPredicate(P3));

        ASSERT(0 == (X2 == X1));        ASSERT(1 == (X2 != X1));
        ASSERT(1 == (X2 == X2));        ASSERT(0 == (X2 != X2));
        ASSERT(1 == (X2 == X3));        ASSERT(0 == (X2 != X3));
        ASSERT(0 == (X2 == X4));        ASSERT(1 == (X2 != X4));

        if (verbose) cout << "\n 11. Assign x1 = x1 (aliasing)." << endl;

        mX1 = X1;

        ASSERT(0 == strcmp(X1.pattern(), VA_PATTERN));
        ASSERT(Levels(VA_LEVELS) == Levels(X1.recordLevel(),
                                           X1.passLevel(),
                                           X1.triggerLevel(),
                                           X1.triggerAllLevel()));
        ASSERT(0 == X1.numAttributes());

        ASSERT(1 == (X1 == X1));        ASSERT(0 == (X1 != X1));
        ASSERT(0 == (X1 == X2));        ASSERT(1 == (X1 != X2));
        ASSERT(0 == (X1 == X3));        ASSERT(1 == (X1 != X3));
        ASSERT(1 == (X1 == X4));        ASSERT(0 == (X1 != X4));
      } break;

      default: {
          cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
          testStatus = -1;
      }
    };

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
