// balxml_formatter.t.cpp                                             -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <balxml_formatter.h>

#include <bdlde_md5.h>

#include <bdlf_bind.h>

#include <bdlsb_memoutstreambuf.h>

#include <bdlt_datetime.h>
#include <bdlt_date.h>
#include <bdlt_time.h>

#include <bsla_maybeunused.h>

#include <bslim_testutil.h>

#include <bsls_assert.h>
#include <bsls_platform.h>
#include <bsls_types.h>

#include <bsl_iostream.h>
#include <bsl_functional.h>
#include <bsl_memory.h>
#include <bsl_sstream.h>

#include <bsl_climits.h>
#include <bsl_cstddef.h>
#include <bsl_cstdio.h>
#include <bsl_cstdlib.h>
#include <bsl_cstring.h>

using namespace BloombergLP;
using namespace bsl;

#ifdef BDE_VERIFY
#pragma bde_verify push
#pragma bde_verify -TP13
#endif

// ============================================================================
//                             TEST PLAN
// ----------------------------------------------------------------------------
//                              Overview
//                              --------
// Each of balxml::Formatter's the manipulators is tested in its own test case
// where the internal states may or may not be accessible through the public
// interface.  In the occasion where the internal states are not accessible,
// e.g., the missing '>' indicates BAEXML_IN_TAG state, we use other
// manipulators as helpers to indicate the such internal states have been
// reached or avoided (such as the use of flush()).
// ----------------------------------------------------------------------------

#ifdef BDE_VERIFY
#pragma bde_verify pop
#endif

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
//                   MACROS FOR TESTING WORKAROUNDS
// ----------------------------------------------------------------------------

#if defined(BSLS_PLATFORM_CMP_MSVC) && BSLS_PLATFORM_CMP_VERSION < 1900
    // 'snprintf' on older Windows libraries outputs an additional '0' in the
    // exponent for scientific notation.
# define BALXML_FORMATTER_EXTRA_ZERO_PADDING_FOR_EXPONENTS 1
#endif

namespace {
// ============================================================================
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------
typedef balxml::Formatter Obj;

// names
const char A[] = "a";
const char B[] = "ab";
const char C[] = "abcde";
const char D[] = "a234567890b234567890c234567890d234567890"
                 "e234567890f234567890g234567890h234567890";
// values
const char I[] = "";
const char J[] = "A";
const char K[] = "AB";
const char L[] = "AB DE";
const char M[] = "A234567890 234567890 234567890D234567890"
                 " 234567890F234567890 234567890 234567890";

// characters that can be escaped
const unsigned char R1[] = "&'<>\"";
const unsigned char R2[] = "&amp;&apos;&lt;&gt;&quot;";
const unsigned char S1[] = "Aa&Bb'Cc<Dd>Ee\"";
const unsigned char S2[] = "Aa&amp;Bb&apos;Cc&lt;Dd&gt;Ee&quot;";

// control characters that can be truncated
const unsigned char T1[] = { 'h', 'e', 'l', 'l', 'o', 0x1F, ' ', 0x01,
                             'w', 'o', 'r', 'l', 'd', 0 };
const unsigned char T2[] = { 'h', 'e', 'l', 'l', 'o', 0 };

const unsigned char U1[] = { 'h', 'e', 'l', 'l', 'o', 0x01, ' ', 0x1F,
                             'w', 'o', 'r', 'l', 'd', 0 };
const unsigned char U2[] = { 'h', 'e', 'l', 'l', 'o', 0 };

const unsigned char V1[] = { 'h', 'e', 0x0A, 'l', 'o', 0x09,
                             'w', 'o', 'r', 'l', 'd', 0 };
const unsigned char V2[] = { 'h', 'e', 0x0A, 'l', 'o', 0x09,
                             'w', 'o', 'r', 'l', 'd', 0 };

const unsigned char X[] = { 0xC2, 0xA9, 0 }; // Unicode U+00A9
const unsigned char Y[] = { 0xE2, 0x89, 0xA0, 0 }; // Unicode U+2260
const unsigned char Z[] = { 0xC2, 0xA9, 0xE2, 0x89, 0xA0, 0 }; // U+00A9 U+2260

enum Test {
    TEST_A = 23,
    TEST_B = 500
};

// ============================================================================
//                    CLASSES FOR TESTING USAGE EXAMPLES
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
//                            Perturbation class
// ----------------------------------------------------------------------------

struct Pert {
    // Provides permutations for orthogonal perturbation
    static bool s_doFlush[];
    static int s_initialIndent[];
    static int s_spacesPerLevel[];
    static int s_wrapColumn[];

    bool        d_doFlush;  // whether to perform flush() after operation
    int         d_initialIndent;
    int         d_spacesPerLevel;
    int         d_wrapColumn;
    Pert()
        : d_doFlush(false), d_initialIndent(0),
          d_spacesPerLevel(0), d_wrapColumn(0),
          d_count(-1) { }
    // MANIPULATORS
    bool next();
        // Return true if next perturbation is obtained
    // ACCESSORS
    int count() const { return d_count; }
  private:
    int d_count;
};

bool Pert::s_doFlush[] = { false, true };
int Pert::s_initialIndent[] = { 0, 1, 10, 100 };
int Pert::s_spacesPerLevel[] = { 0, 1, 2, 3, 4, 5, 11, 100 };
int Pert::s_wrapColumn[] = { 0, 1, 10, 80, 1000, INT_MAX };

bool Pert::next()
{
    static const int doFlushSize =
        sizeof(s_doFlush) / sizeof(*s_doFlush);
    static const int initialIndentSize =
        sizeof(s_initialIndent) / sizeof(*s_initialIndent);
    static const int spacesPerLevelSize =
        sizeof(s_spacesPerLevel) / sizeof(*s_spacesPerLevel);
    static const int wrapColumnSize =
        sizeof(s_wrapColumn) / sizeof(*s_wrapColumn);

    ++d_count;

    if (d_count >= (doFlushSize * initialIndentSize *
                    spacesPerLevelSize * wrapColumnSize)) {
        return false;                                                 // RETURN
    }

    int count = d_count;

    int i3 = count % wrapColumnSize;
    count /= wrapColumnSize;
    int i2 = count % spacesPerLevelSize;
    count /= spacesPerLevelSize;
    int i1 = count % initialIndentSize;
    count /= initialIndentSize;
    int i0 = count % doFlushSize;

    d_doFlush = s_doFlush[i0];
    d_initialIndent = s_initialIndent[i1];
    d_spacesPerLevel = s_spacesPerLevel[i2];
    d_wrapColumn = s_wrapColumn[i3];

    return true;
}

struct FieldType {
    enum Type {
        e_CHAR              =  0,
        e_SHORT             =  1,
        e_INT               =  2,
        e_INT64             =  3,
        e_FLOAT             =  4,
        e_DOUBLE            =  5,
        e_STRING            =  6,
        e_DATETIME          =  7,
        e_DATETIMETZ        =  8,
        e_DATE              =  9,
        e_TIME              =  10
    };
};

class ScalarData {
    // This class facilitates reading data of different types from the same
    // field in a table and provides a uniform interface to test these
    // different types.
  private:
    typedef FieldType Ft;

    Ft::Type           d_type;

    // N.B. A union would be natural here, but we cannot use a union for
    // non-builtin types, i.e., bdlt::*

    char               d_char;
    short              d_short;
    int                d_int;
    bsls::Types::Int64 d_int64;
    float              d_float;
    double             d_double;
    bsl::string        d_string;
    bdlt::Datetime     d_datetime;
    bdlt::DatetimeTz   d_datetimeTz;
    bdlt::Date         d_date;
    bdlt::Time         d_time;

  public:
    ScalarData(char c) : d_type(Ft::e_CHAR), d_char(c) {}
    ScalarData(short s) : d_type(Ft::e_SHORT), d_short(s) {}
    ScalarData(int i) : d_type(Ft::e_INT), d_int(i) {}
    ScalarData(bsls::Types::Int64 i)
        : d_type(Ft::e_INT64), d_int64(i) {}
    ScalarData(float f) : d_type(Ft::e_FLOAT), d_float(f) {}
    ScalarData(double d) : d_type(Ft::e_DOUBLE), d_double(d) {}
    ScalarData(const bsl::string& s) : d_type(Ft::e_STRING), d_string(s) {}
    ScalarData(const bdlt::Datetime& d)
        : d_type(Ft::e_DATETIME),
          d_datetime(d)
        {}
    ScalarData(const bdlt::DatetimeTz& d)
        : d_type(Ft::e_DATETIMETZ),
          d_datetimeTz(d)
        {}
    ScalarData(const bdlt::Date& d) : d_type(Ft::e_DATE), d_date(d) {}
    ScalarData(const bdlt::Time& t) : d_type(Ft::e_TIME), d_time(t) {}
    void addAttribute(const bsl::string&  attrName,
                      balxml::Formatter  *formatter) const;
        // Call addAttribute method of the specified 'formatter' with
        // 'attrName' as attribute name and the d_typeValue corresponding to
        // the d_type
    void addListData(Obj *formatter) const;
        // Call addListData method of the specified 'formatter' with the
        // d_typeValue corresponding to the d_type.
    void addData(Obj *formatter) const;
        // Call addData method of the specified 'formatter' with the
        // d_typeValue corresponding to the d_type.
    friend bsl::ostream& operator<<(bsl::ostream& os, const ScalarData& data);
        // Output only ScalarData of type short, int, bsls::Types::Int64,
        // float, double.  Other types result in undefined behavior
};

void ScalarData::addAttribute(const bsl::string&  attrName,
                              Obj                *formatter) const
{
    switch (d_type) {
      case Ft::e_CHAR:
        formatter->addAttribute(attrName, d_char); break;
      case Ft::e_SHORT:
        formatter->addAttribute(attrName, d_short); break;
      case Ft::e_INT:
        formatter->addAttribute(attrName, d_int); break;
      case Ft::e_INT64:
        formatter->addAttribute(attrName, d_int64); break;
      case Ft::e_FLOAT:
        formatter->addAttribute(attrName, d_float); break;
      case Ft::e_DOUBLE:
        formatter->addAttribute(attrName, d_double); break;
      case Ft::e_STRING:
        formatter->addAttribute(attrName, d_string); break;
      case Ft::e_DATETIME:
        formatter->addAttribute(attrName, d_datetime); break;
      case Ft::e_DATETIMETZ:
        formatter->addAttribute(attrName, d_datetimeTz); break;
      case Ft::e_DATE:
        formatter->addAttribute(attrName, d_date); break;
      case Ft::e_TIME:
        formatter->addAttribute(attrName, d_time); break;
      default:
        BSLS_ASSERT_OPT(0);
    }
}

void ScalarData::addData(Obj *formatter) const
{
    switch (d_type) {
      case Ft::e_CHAR:
        formatter->addData(d_char); break;
      case Ft::e_SHORT:
        formatter->addData(d_short); break;
      case Ft::e_INT:
        formatter->addData(d_int); break;
      case Ft::e_INT64:
        formatter->addData(d_int64); break;
      case Ft::e_FLOAT:
        formatter->addData(d_float); break;
      case Ft::e_DOUBLE:
        formatter->addData(d_double); break;
      case Ft::e_STRING:
        formatter->addData(d_string); break;
      case Ft::e_DATETIME:
        formatter->addData(d_datetime); break;
      case Ft::e_DATETIMETZ:
        formatter->addData(d_datetimeTz); break;
      case Ft::e_DATE:
        formatter->addData(d_date); break;
      case Ft::e_TIME:
        formatter->addData(d_time); break;
      default:
        BSLS_ASSERT_OPT(0);
    }
}

void ScalarData::addListData(Obj *formatter) const
{
    switch (d_type) {
      case Ft::e_CHAR:
        formatter->addListData(d_char); break;
      case Ft::e_SHORT:
        formatter->addListData(d_short); break;
      case Ft::e_INT:
        formatter->addListData(d_int); break;
      case Ft::e_INT64:
        formatter->addListData(d_int64); break;
      case Ft::e_FLOAT:
        formatter->addListData(d_float); break;
      case Ft::e_DOUBLE:
        formatter->addListData(d_double); break;
      case Ft::e_STRING:
        formatter->addListData(d_string); break;
      case Ft::e_DATETIME:
        formatter->addListData(d_datetime); break;
      case Ft::e_DATETIMETZ:
        formatter->addListData(d_datetimeTz); break;
      case Ft::e_DATE:
        formatter->addListData(d_date); break;
      case Ft::e_TIME:
        formatter->addListData(d_time); break;
      default:
        BSLS_ASSERT_OPT(0);
    }
}

bsl::ostream& operator<<(bsl::ostream& os, const ScalarData& data)
{
using namespace bsl;  // automatically added by script

    switch(data.d_type) {
      case ScalarData::Ft::e_CHAR:
        os << data.d_char; break;
      case ScalarData::Ft::e_SHORT:
        os << data.d_short; break;
      case ScalarData::Ft::e_INT:
        os << data.d_int; break;
      case ScalarData::Ft::e_INT64:
        os << data.d_int64; break;
      case ScalarData::Ft::e_FLOAT:
        os << data.d_float; break;
      case ScalarData::Ft::e_DOUBLE:
        os << data.d_double; break;
      case ScalarData::Ft::e_STRING:
        os << data.d_string; break;
      case ScalarData::Ft::e_DATETIME:
        os << data.d_datetime; break;
      case ScalarData::Ft::e_DATETIMETZ:
        os << data.d_datetimeTz; break;
      case ScalarData::Ft::e_DATE:
        os << data.d_date; break;
      case ScalarData::Ft::e_TIME:
        os << data.d_time; break;
      default:
        BSLS_ASSERT_OPT(!"ScalarData type not accepted");
    }
    return os;
}

                         // ==========================
                         // struct FormatterModelState
                         // ==========================

struct FormatterModelState {
    // An approximation of the state of the formatter used to determine whether
    // or not it is legal to call a particular member function.

    enum Enum {
        e_START,
        e_DEFAULT,
        e_IN_TAG,
        e_END
    };
};

                        // ===========================
                        // struct FormatterModelAction
                        // ===========================

struct FormatterModelAction {
    // A list of all XML-printing member functions on the formatter.

    // TYPES
    enum Enum {
        e_ADD_ATTRIBUTE,
        e_ADD_BLANK_LINE,
        e_ADD_COMMENT,
        e_ADD_VALID_COMMENT,
        e_ADD_DATA,
        e_ADD_LIST_DATA,
        e_ADD_ELEMENT_AND_DATA,
        e_ADD_HEADER,
        e_ADD_NEWLINE,
        e_CLOSE_ELEMENT,
        e_FLUSH,
        e_OPEN_ELEMENT_PRESERVE_WS,
        e_OPEN_ELEMENT_WORDWRAP,
        e_OPEN_ELEMENT_WORDWRAP_INDENT,
        e_OPEN_ELEMENT_NEWLINE_INDENT
    };

    enum {
        k_NUM_ENUMERATORS = 15
    };

    // CLASS METHODS
    static Enum fromInt(int index);

    static const char *toChars(Enum value);

};

// CLASS METHODS
FormatterModelAction::Enum FormatterModelAction::fromInt(int index)
{
    BSLS_ASSERT(0 <= index && index < k_NUM_ENUMERATORS);
    return static_cast<Enum>(index);
}

BSLA_MAYBE_UNUSED
const char *FormatterModelAction::toChars(Enum value)
{
#ifndef BSLA_MAYBE_UNUSED_IS_ACTIVE
    (void)&FormatterModelAction::toChars;
#endif

    switch (value) {
      case e_ADD_ATTRIBUTE:
        return "e_ADD_ATTRIBUTE";
      case e_ADD_BLANK_LINE:
        return "e_ADD_BLANK_LINE";
      case e_ADD_COMMENT:
        return "e_ADD_COMMENT";
      case e_ADD_VALID_COMMENT:
        return "e_ADD_VALID_COMMENT";
      case e_ADD_DATA:
        return "e_ADD_DATA";
      case e_ADD_LIST_DATA:
        return "e_ADD_LIST_DATA";
      case e_ADD_ELEMENT_AND_DATA:
        return "e_ADD_ELEMENT_AND_DATA";
      case e_ADD_HEADER:
        return "e_ADD_HEADER";
      case e_ADD_NEWLINE:
        return "e_ADD_NEWLINE";
      case e_CLOSE_ELEMENT:
        return "e_CLOSE_ELEMENT";
      case e_FLUSH:
        return "e_FLUSH";
      case e_OPEN_ELEMENT_PRESERVE_WS:
        return "e_OPEN_ELEMENT_PRESERVE_WS";
      case e_OPEN_ELEMENT_WORDWRAP:
        return "e_OPEN_ELEMENT_WORDWRAP";
      case e_OPEN_ELEMENT_WORDWRAP_INDENT:
        return "e_OPEN_ELEMENT_WORDWRAP_INDENT";
      case e_OPEN_ELEMENT_NEWLINE_INDENT:
        return "e_OPEN_ELEMENT_NEWLINE_INDENT";
    }

    return 0;
}

                         // =========================
                         // struct FormatterModelUtil
                         // =========================

struct FormatterModelUtil {

    // TYPES
    typedef FormatterModelAction Action;
    typedef FormatterModelState  State;

    // CLASS METHODS
    static bool isActionAllowed(State::Enum  state,
                                int          nestingDepth,
                                Action::Enum action);
        // Return 'true' if the behavior of invoking the member function
        // identified by the specified 'action' on a 'balxml::Formatter' in the
        // specified 'state' with the specified 'indentLevel' is defined, and
        // return 'false' otherwise.

    static State::Enum getNextState(State::Enum  state,
                                    int          nestingDepth,
                                    Action::Enum action);
        // Return the next state after performing the specified 'action' at the
        // specified 'state' with the specified 'nestingDepth'.

    static int getNextNestingDepth(State::Enum  state,
                                   int          nestingDepth,
                                   Action::Enum action);
        // Return the next nesting depth after performing the specified
        // 'action' at the specified 'state' with the specified 'nestingDepth'.

    static void performAction(Obj *formatter, Action::Enum action);
        // Invoke the member function identified by the specified 'action' on
        // the specified 'formatter', providing placeholders for any arguments.

    static void appendBehavior(
                             bdlde::Md5                    *digest,
                             int                            numActions,
                             int                            initialIndentLevel,
                             int                            spacesPerLevel,
                             int                            wrapColumn,
                             const balxml::EncoderOptions&  encoderOptions);
        // Append to the specified 'digest' the output of all valid sequences
        // of XML-printing member functions of the specified 'numActions'
        // length on a 'balxml::Formatter' initialized with the specified
        // 'initialIndentLevel', 'spacesPerLevel', 'wrapColumn', and
        // 'encoderOptions'.
};

// CLASS METHODS
bool FormatterModelUtil::isActionAllowed(State::Enum  state,
                                         int          nestingDepth,
                                         Action::Enum action)
{
    switch (state) {
      case State::e_START: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return false;
          case Action::e_ADD_BLANK_LINE:
            return true;
          case Action::e_ADD_COMMENT:
            return true;
          case Action::e_ADD_VALID_COMMENT:
            return true;
          case Action::e_ADD_DATA:
            return false;
          case Action::e_ADD_LIST_DATA:
            return false;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return true;
          case Action::e_ADD_HEADER:
            return true;
          case Action::e_ADD_NEWLINE:
            return true;
          case Action::e_CLOSE_ELEMENT:
            return false;
          case Action::e_FLUSH:
            return true;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return true;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return true;
        }
      } break;
      case State::e_DEFAULT: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return false;
          case Action::e_ADD_BLANK_LINE:
            return true;
          case Action::e_ADD_COMMENT:
            return true;
          case Action::e_ADD_VALID_COMMENT:
            return true;
          case Action::e_ADD_DATA:
            return 0 != nestingDepth;
          case Action::e_ADD_LIST_DATA:
            return 0 != nestingDepth;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return true;
          case Action::e_ADD_HEADER:
            return false;
          case Action::e_ADD_NEWLINE:
            return true;
          case Action::e_CLOSE_ELEMENT:
            return 0 != nestingDepth;
          case Action::e_FLUSH:
            return true;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return true;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return true;
        }
      } break;
      case State::e_IN_TAG: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return true;
          case Action::e_ADD_BLANK_LINE:
            return true;
          case Action::e_ADD_COMMENT:
            return true;
          case Action::e_ADD_VALID_COMMENT:
            return true;
          case Action::e_ADD_DATA:
            return true;
          case Action::e_ADD_LIST_DATA:
            return true;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return true;
          case Action::e_ADD_HEADER:
            return false;
          case Action::e_ADD_NEWLINE:
            return true;
          case Action::e_CLOSE_ELEMENT:
            return true;
          case Action::e_FLUSH:
            return true;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return true;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return true;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return true;
        }
      } break;
      case State::e_END:
        return false;
    }

    return false;
}

FormatterModelState::Enum FormatterModelUtil::getNextState(
                                                     State::Enum  state,
                                                     int          nestingDepth,
                                                     Action::Enum action)
{
    switch (state) {
      case State::e_START: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return State::e_END;
          case Action::e_ADD_BLANK_LINE:
            return State::e_START;
          case Action::e_ADD_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_VALID_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_DATA:
            return State::e_END;
          case Action::e_ADD_LIST_DATA:
            return State::e_END;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return State::e_DEFAULT;
          case Action::e_ADD_HEADER:
            return State::e_DEFAULT;
          case Action::e_ADD_NEWLINE:
            return State::e_START;
          case Action::e_CLOSE_ELEMENT:
            return State::e_END;
          case Action::e_FLUSH:
            return State::e_START;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return State::e_IN_TAG;
        }
      } break;
      case State::e_DEFAULT: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return State::e_END;
          case Action::e_ADD_BLANK_LINE:
            return State::e_DEFAULT;
          case Action::e_ADD_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_VALID_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_DATA:
            return 0 == nestingDepth ? State::e_END : State::e_DEFAULT;
          case Action::e_ADD_LIST_DATA:
            return 0 == nestingDepth ? State::e_END : State::e_DEFAULT;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return State::e_DEFAULT;
          case Action::e_ADD_HEADER:
            return State::e_END;
          case Action::e_ADD_NEWLINE:
            return State::e_DEFAULT;
          case Action::e_CLOSE_ELEMENT:
            return 0 == nestingDepth ? State::e_END : State::e_DEFAULT;
          case Action::e_FLUSH:
            return State::e_DEFAULT;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return State::e_IN_TAG;
        }
      } break;
      case State::e_IN_TAG: {
        switch (action) {
          case Action::e_ADD_ATTRIBUTE:
            return State::e_IN_TAG;
          case Action::e_ADD_BLANK_LINE:
            return State::e_DEFAULT;
          case Action::e_ADD_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_VALID_COMMENT:
            return State::e_DEFAULT;
          case Action::e_ADD_DATA:
            return State::e_DEFAULT;
          case Action::e_ADD_LIST_DATA:
            return State::e_DEFAULT;
          case Action::e_ADD_ELEMENT_AND_DATA:
            return State::e_DEFAULT;
          case Action::e_ADD_HEADER:
            return State::e_END;
          case Action::e_ADD_NEWLINE:
            return State::e_DEFAULT;
          case Action::e_CLOSE_ELEMENT:
            return State::e_DEFAULT;
          case Action::e_FLUSH:
            return State::e_DEFAULT;
          case Action::e_OPEN_ELEMENT_PRESERVE_WS:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT:
            return State::e_IN_TAG;
          case Action::e_OPEN_ELEMENT_NEWLINE_INDENT:
            return State::e_IN_TAG;
        }
      } break;
      case State::e_END:
        return State::e_END;
    }

    return State::e_END;
}

int FormatterModelUtil::getNextNestingDepth(State::Enum  state,
                                            int          nestingDepth,
                                            Action::Enum action)
{
    (void)state;

    if (Action::e_OPEN_ELEMENT_PRESERVE_WS == action ||
        Action::e_OPEN_ELEMENT_WORDWRAP == action ||
        Action::e_OPEN_ELEMENT_WORDWRAP_INDENT == action ||
        Action::e_OPEN_ELEMENT_NEWLINE_INDENT == action) {
        return nestingDepth + 1;                                      // RETURN
    }

    if (Action::e_CLOSE_ELEMENT == action) {
        return nestingDepth - 1;                                      // RETURN
    }

    return nestingDepth;
}

void FormatterModelUtil::performAction(Obj *formatter, Action::Enum action)
{
    switch (action) {
      case Action::e_ADD_ATTRIBUTE: {
          formatter->addAttribute("attrName", "attrValue");
      } break;
      case Action::e_ADD_BLANK_LINE: {
          formatter->addBlankLine();
      } break;
      case Action::e_ADD_COMMENT: {
          formatter->addComment("lorem ipsum dolor sit amet");
      } break;
      case Action::e_ADD_VALID_COMMENT: {
          formatter->addValidComment("lorem ipsum dolor sit amet");
      } break;
      case Action::e_ADD_DATA: {
          formatter->addData("data");
      } break;
      case Action::e_ADD_LIST_DATA: {
          formatter->addListData("listData");
      } break;
      case Action::e_ADD_ELEMENT_AND_DATA: {
          formatter->addElementAndData("element", "data");
      } break;
      case Action::e_ADD_HEADER: {
          formatter->addHeader();
      } break;
      case Action::e_ADD_NEWLINE: {
          formatter->addNewline();
      } break;
      case Action::e_CLOSE_ELEMENT: {
          formatter->closeElement("element");
      } break;
      case Action::e_FLUSH: {
          formatter->flush();
      } break;
      case Action::e_OPEN_ELEMENT_PRESERVE_WS: {
          formatter->openElement("element", Obj::e_PRESERVE_WHITESPACE);
      } break;
      case Action::e_OPEN_ELEMENT_WORDWRAP: {
          formatter->openElement("element", Obj::e_WORDWRAP);
      } break;
      case Action::e_OPEN_ELEMENT_WORDWRAP_INDENT: {
          formatter->openElement("element", Obj::e_WORDWRAP_INDENT);
      } break;
      case Action::e_OPEN_ELEMENT_NEWLINE_INDENT: {
          formatter->openElement("element", Obj::e_NEWLINE_INDENT);
      } break;
    }
}

void FormatterModelUtil::appendBehavior(
                             bdlde::Md5                    *digest,
                             int                            numActions,
                             int                            initialIndentLevel,
                             int                            spacesPerLevel,
                             int                            wrapColumn,
                             const balxml::EncoderOptions&  encoderOptions)
{
    BSLS_ASSERT(0 < numActions);

    // int numIterations = exponentiate(Action::k_NUM_ENUMERATORS,
    //                                  numActions);
    int numIterations = Action::k_NUM_ENUMERATORS;
    for (int i = 0; i != numActions - 1; ++i) {
        numIterations *= Action::k_NUM_ENUMERATORS;
    }

    for (int iterIdx = 0; iterIdx != numIterations; ++iterIdx) {
        bdlsb::MemOutStreamBuf streambuf;

        Obj formatter(&streambuf,
                      encoderOptions,
                      initialIndentLevel,
                      spacesPerLevel,
                      wrapColumn);

        State::Enum state        = State::e_START;
        int         nestingDepth = 0;
        int         seed         = iterIdx;

        for (int actionIdx = 0; actionIdx != numActions; ++actionIdx) {
            const Action::Enum action =
                Action::fromInt(seed % Action::k_NUM_ENUMERATORS);
            seed /= Action::k_NUM_ENUMERATORS;

            if (!isActionAllowed(state, nestingDepth, action)) {
                break;  // BREAK
            }

            performAction(&formatter, action);
            state        = getNextState(state, nestingDepth, action);
            nestingDepth = getNextNestingDepth(state, nestingDepth, action);
        }

        digest->update(streambuf.data(), static_cast<int>(streambuf.length()));
    }
}


                           // =====================
                           // class Method_Protocol
                           // =====================

class Method_Protocol {
    // This abstract base class provides a protocol for an object that can be
    // invoked on a 'balxml::Formatter *' (i.e. is a "method" of
    // 'balxml::Formatter') and copied.
    //
    // Note that this class is an implementation detail of 'Method' below.

  public:
    // CREATORS
    virtual ~Method_Protocol()
    {
    }

    // ACCESSORS
    virtual Method_Protocol *clone() const = 0;

    virtual void operator()(balxml::Formatter *formatter) const = 0;
};

                              // ================
                              // class Method_Imp
                              // ================

template <class METHOD>
class Method_Impl : public Method_Protocol {
    // This mechanism class template provides an implementation of the
    // 'Method_Protocol' whose virtual function call operator invokes
    // an object of the specified 'METHOD' type with the supplied formatter.
    //
    // Note that this class is an implementation detail of 'Method' below.

  public:
    // TYPES
    typedef Method_Protocol Protocol;

  private:
    // DATA
    METHOD d_method;

  public:
    // CREATORS
    explicit Method_Impl(const METHOD& method)
    : d_method(method)
    {
    }

    // ACCESSORS
    Protocol *clone() const BSLS_KEYWORD_OVERRIDE
    {
        return new Method_Impl(*this);
    }

    void operator()(balxml::Formatter *formatter) const BSLS_KEYWORD_OVERRIDE
    {
        d_method(formatter);
    }
};

                                // ============
                                // class Method
                                // ============

class Method {
    // This mechanism class provides a type-erased interface to an object of
    // any type that is invocable with a 'balxml::Formatter *' (e.g. that is a
    // "method" of 'balxml::Formatter'.)

    // PRIVATE TYPES
    typedef Method_Protocol Protocol;

    // DATA
    bsl::shared_ptr<Protocol> d_protocol;

  public:
    // CREATORS
    Method()
    : d_protocol()
    {
    }

    Method(const Method& original)
    : d_protocol(original.d_protocol->clone())
    {
    }

    template <class METHOD>
    /* IMPLICIT */ Method(const METHOD& method)
    : d_protocol(bsl::make_shared<Method_Impl<METHOD> >(method))
    {
    }

    // MANIPULATORS
    Method& operator=(const Method& original)
    {
        d_protocol = bsl::shared_ptr<Protocol>(original.d_protocol->clone());
        return *this;
    }

    // ACCESSORS
    void operator()(balxml::Formatter *formatter) const
    {
        if (d_protocol) {
            (*d_protocol)(formatter);
        }
    }
};

                       // =============================
                       // class AddElementAndDataMethod
                       // =============================

class AddElementAndDataMethod {
    // This class provides a function-call operator that is a factory for
    // 'Method' objects that invokes 'balxml::Formatter::addElementAndData'
    // with bound 'name', 'value', and 'formattingMode' arguments.

  public:
    // CREATORS
    AddElementAndDataMethod()
    {
    }

    // ACCESSORS
    template <class TYPE>
    Method operator()(const bsl::string_view& name,
                      const TYPE&             value,
                      int                     formattingMode = 0) const
    {
        void (balxml::Formatter:: *member)(
            const bsl::string_view&, const TYPE&, int) =
            &balxml::Formatter::addElementAndData<TYPE>;

        return bdlf::BindUtil::bind(
            member, bdlf::PlaceHolders::_1, name, value, formattingMode);
    }
};

                          // =======================
                          // class OpenElementMethod
                          // =======================

class OpenElementMethod {
    // This class provides a function-call operator that is a factory for
    // 'Method' objects that invokes 'balxml::Formatter::openElement' with
    // bound 'name', and 'ws' arguments.

  public:
    // CREATORS
    OpenElementMethod()
    {
    }

    // ACCESSORS
    Method operator()(const bsl::string_view&           name,
                      balxml::Formatter::WhitespaceType ws =
                          balxml::Formatter::e_PRESERVE_WHITESPACE) const
    {
        void (balxml::Formatter:: *member)(const bsl::string_view&,
                                          balxml::Formatter::WhitespaceType) =
            &balxml::Formatter::openElement;

        return bdlf::BindUtil::bind(member, bdlf::PlaceHolders::_1, name, ws);
    }
};

                          // ========================
                          // class CloseElementMethod
                          // ========================

class CloseElementMethod {
    // This class provides a function-call operator that is a factory for
    // 'Method' objects that invokes 'balxml::Formatter::closeElement' with
    // a bound 'name' argument.

  public:
    // CREATORS
    CloseElementMethod()
    {
    }

    // ACCESSORS
    Method operator()(const bsl::string_view& name) const
    {
        void (balxml::Formatter:: *member)(const bsl::string_view&) =
            &balxml::Formatter::closeElement;

        return bdlf::BindUtil::bind(member, bdlf::PlaceHolders::_1, name);
    }
};

}  // close unnamed namespace

// ============================================================================
//                               MAIN PROGRAM
// ----------------------------------------------------------------------------
#ifdef BSLS_PLATFORM_CMP_MSVC
// Disable warnings that the Microsoft compiler issues for overflow of floating
// point constants, that we are deliberately using to confirm how we handle
// infinities.  Unfortunately this warning must be disabled outside the
// function, rather than locally around the issue, due to a known compiler bug,
// see http://support.microsoft.com/kb/120968 for details.
#pragma warning( push )
#pragma warning( disable : 4756 ) // overflow in constant arithmetic
#endif

int main(int argc, char *argv[])
{
    int test = argc > 1 ? bsl::atoi(argv[1]) : 0;
    int verbose = argc > 2;
    int veryVerbose = argc > 3;
    int veryVeryVerbose = argc > 4;

    bsl::cout << "TEST " << __FILE__ << " CASE " << test << bsl::endl;;

    switch (test) { case 0:  // Zero is always the leading case.
      case 25: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        // --------------------------------------------------------------------

          if (verbose) {
              bsl::cout << "\nTESTING USAGE EXAMPLE\n" << bsl::endl;
          }

// Here is a basic example showing ten steps of how to create an XML document
// using this component's major manipulators:
//..
    {
        // 1.  Create a formatter:
        bsl::ostringstream outStream;
        balxml::Formatter formatter(outStream);

        // 2.  Add a header:
        formatter.addHeader("UTF-8");

        // 3.  Open the root element,
        //     Add attributes if there are any:
        formatter.openElement("Fruits");

        // 4. Open an element,
        //    Add attributes if there are any:
        formatter.openElement("Oranges");
        formatter.addAttribute("farm", "Frances' Orchard"); // ' is escaped
        formatter.addAttribute("size", 3.5);

        // 5.  If there are nested elements, recursively do
        // 6.  Else, there are no more nested elements, add data:
        formatter.openElement("pickDate");               // step 4
        formatter.addData(bdlt::Date(2004, 8, 31));      // step 6
        formatter.closeElement("pickDate");              // step 7
        formatter.addElementAndData("Quantity", 12);     // step 8
        // element "Quantity" has no attributes, can use shortcut
        // 'addElementAndData' to complete steps 4, 6 and 7 in one shot.

        // 7. Close the element:
        formatter.closeElement("Oranges");

        // 8. If there are more elements, repeat steps 4 - 8
        formatter.openElement("Apples");                 // step 4
        formatter.addAttribute("farm", "Fuji & Sons");   // '&' is escaped
        formatter.addAttribute("size", 3);
        formatter.closeElement("Apples");                // step 7

        // 9. Close the root element:
        formatter.closeElement("Fruits");
//..
// Indentation is correctly taken care of and the user only needs to concern
// her/himself with the correct ordering of XML elements s/he's trying
// to write.  The output of the above example is:
//..
        const char EXPECTED1[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
            "<Fruits>\n"
            "    <Oranges farm=\"Frances&apos; Orchard\" size=\"3.5\">\n"
            "        <pickDate>2004-08-31</pickDate>\n"
            "        <Quantity>12</Quantity>\n"
            "    </Oranges>\n"
            "    <Apples farm=\"Fuji &amp; Sons\" size=\"3\"/>\n"
            "</Fruits>\n";

        if (verbose) P(outStream.str());

        ASSERT(EXPECTED1 == outStream.str());
    }
//..
// Following is a more complete usage example that use most of the
// manipulators provided by balxml::Formatter:
//..
    {
        bsl::ostringstream outStream;
        balxml::Formatter formatter(outStream, 0, 4, 40);

        formatter.addHeader("UTF-8");

        formatter.openElement("Fruits");
        formatter.openElement("Oranges");
        formatter.addAttribute("farm", "Frances' Orchard");
        // notice that the apostrophe in the string will be escaped
        formatter.addAttribute("size", 3.5);

        formatter.addElementAndData("Quantity", 12);

        formatter.openElement("pickDate");
        formatter.addData(bdlt::Date(2004, 8, 31));
        formatter.closeElement("pickDate");

        formatter.openElement("Feature");
        formatter.addAttribute("shape", "round");
        formatter.closeElement("Feature");

        formatter.addComment("No wrapping for long comments");

        formatter.closeElement("Oranges");

        formatter.addBlankLine();

        formatter.openElement("Apples");
        formatter.addAttribute("farm", "Fuji & Sons");
        formatter.addAttribute("size", 3);

        formatter.openElement("pickDates",
                              balxml::Formatter::e_NEWLINE_INDENT);
        formatter.addListData(bdlt::Date(2005, 1, 17));
        formatter.addListData(bdlt::Date(2005, 2, 21));
        formatter.addListData(bdlt::Date(2005, 3, 25));
        formatter.addListData(bdlt::Date(2005, 5, 30));
        formatter.addListData(bdlt::Date(2005, 7, 4));
        formatter.addListData(bdlt::Date(2005, 9, 5));
        formatter.addListData(bdlt::Date(2005, 11, 24));
        formatter.addListData(bdlt::Date(2005, 12, 25));

        formatter.closeElement("pickDates");

        formatter.openElement("Feature");
        formatter.addAttribute("color", "red");
        formatter.addAttribute("taste", "juicy");
        formatter.closeElement("Feature");

        formatter.closeElement("Apples");

        formatter.closeElement("Fruits");

        formatter.reset();
        // reset the formatter for a new document in the same stream

        formatter.addHeader();
        formatter.openElement("Grains");

        bsl::ostream& os = formatter.rawOutputStream();
        os << "<free>anything that can mess up the XML doc</free>";
        // Now coming back to the formatter, but can't do the following:
        // formatter.addAttribute("country", "USA");
        formatter.addData("Corn, Wheat, Oat");
        formatter.closeElement("Grains");
//..
// Following are the two resulting documents, as separated by the call to
// reset(),
//..
        const char EXPECTED2[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
            "<Fruits>\n"
            "    <Oranges\n"
            "        farm=\"Frances&apos; Orchard\"\n"
            "        size=\"3.5\">\n"
            "        <Quantity>12</Quantity>\n"
            "        <pickDate>2004-08-31</pickDate>\n"
            "        <Feature shape=\"round\"/>\n"
            "        <!-- No wrapping for long comments -->\n"
            "    </Oranges>\n"
            "\n"
            "    <Apples farm=\"Fuji &amp; Sons\"\n"
            "        size=\"3\">\n"
            "        <pickDates>\n"
            "            2005-01-17 2005-02-21\n"
            "            2005-03-25 2005-05-30\n"
            "            2005-07-04 2005-09-05\n"
            "            2005-11-24 2005-12-25\n"
            "        </pickDates>\n"
            "        <Feature color=\"red\"\n"
            "            taste=\"juicy\"/>\n"
            "    </Apples>\n"
            "</Fruits>\n"
            "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
            "<Grains><free>anything that can mess up the XML doc</free>"
                                                 "Corn, Wheat, Oat</Grains>\n";

        if (verbose) P(outStream.str());

        ASSERT(EXPECTED2 == outStream.str());
    }
//..
      } break;
      case 24: {
          // ------------------------------------------------------------------
          // TESTING INDENTATION FOR 'closeElement'
          //   Ensure that 'closeElement' indents the closing tag when
          //   'wrapColumn' is set to 0.
          //
          // Concerns:
          //: 1 When 'spacesPerLevel' is non-zero and 'wrapColumn' is 0 (which
          //:   indicates an infinite wrap column), 'closeElement' properly
          //:   indents the printed closing tag if said tag is not
          //:   self-closing.
          //
          // Plan:
          //: 1 Set 'spacesPerLevel' to 1, 'wrapColumn' to 0, and the initial
          //:   indent level to 0, and print XML documents having nesting
          //:   depths of 0, 1, and 2, where no elements are self-closing, and
          //:   verify that all closing tags are indented by 1 space per
          //:   nesting level.
          //:
          //: 2 As a control, print the same documents using several other
          //:   combinations of 'spacesPerLevel, 'wrapColumn', and initial
          //:   indent level options, and verify the behavior of the formatter
          //:   is consistent.
          //
          // TESTING
          //   'closeElement(const bslstl::StringRef& name)'
          // ------------------------------------------------------------------

          if (verbose) {
              bsl::cout << "\nTESTING INDENTATION FOR 'closeElement'"
                        << "\n======================================"
                        << bsl::endl;
          }

          static const int k_MAX_METHODS = 10;

          static const bsl::string_view A = "A";
          static const bsl::string_view B = "B";
          static const bsl::string_view C = "C";
          static const bsl::string_view D = "D";

          const AddElementAndDataMethod a;
          const CloseElementMethod      c;
          const OpenElementMethod       o;

          const struct {
              int          d_line;
              int          d_indentLevel;
              int          d_spacesPerLevel;
              int          d_wrapColumn;
              Method       d_methods[k_MAX_METHODS];
              const char  *d_expectedString;
          } DATA[] = {
              //           INDENT LEVEL
              //          .------------
              //         /  SPACES PER LEVEL
              //  LINE  /  .----------------
              // .---- /  /  WRAP COLUMN
              // |    /  /  .-----------
              // |   /  /  /                  METHOD SEQUENCE
              // -- -- -- -- -------------------------------------------------
              //                     EXPECTED OUTPUT
              // -------------------------------------------------------------
              { L_, 0, 1, 0,{a(A,B)                     },
                "<A>B</A>\n"                                                 },
              { L_, 0, 1, 0,{o(A),a(B,C),c(A)           },
                "<A>\n <B>C</B>\n</A>\n"                                     },
              { L_, 0, 1, 0,{o(A),o(B),a(C,D),c(B),c(A) },
                "<A>\n <B>\n  <C>D</C>\n </B>\n</A>\n"                       },

              { L_, 0, 2, 0,{a(A,B)                     },
                "<A>B</A>\n"                                                 },
              { L_, 0, 2, 0,{o(A),a(B,C),c(A)           },
                "<A>\n  <B>C</B>\n</A>\n"                                    },
              { L_, 0, 2, 0,{o(A),o(B),a(C,D),c(B),c(A) },
                "<A>\n  <B>\n    <C>D</C>\n  </B>\n</A>\n"                   },

              { L_, 0, 1,80,{a(A,B)                     },
                "<A>B</A>\n"                                                 },
              { L_, 0, 1,80,{o(A),a(B,C),c(A)           },
                "<A>\n <B>C</B>\n</A>\n"                                     },
              { L_, 0, 1,80,{o(A),o(B),a(C,D),c(B),c(A) },
                "<A>\n <B>\n  <C>D</C>\n </B>\n</A>\n"                       },

              { L_, 0, 2, 0,{a(A,B)                     },
                "<A>B</A>\n"                                                 },
              { L_, 0, 2, 0,{o(A),a(B,C),c(A)           },
                "<A>\n  <B>C</B>\n</A>\n"                                    },
              { L_, 0, 2, 0,{o(A),o(B),a(C,D),c(B),c(A) },
                "<A>\n  <B>\n    <C>D</C>\n  </B>\n</A>\n"                   },

              { L_, 1, 1, 0,{a(A,B)                     },
                " <A>B</A>\n"                                                },
              { L_, 1, 1, 0,{o(A),a(B,C),c(A)           },
                " <A>\n  <B>C</B>\n </A>\n"                                  },
              { L_, 1, 1, 0,{o(A),o(B),a(C,D),c(B),c(A) },
                " <A>\n  <B>\n   <C>D</C>\n  </B>\n </A>\n"                  },

              { L_, 1, 2, 0,{a(A,B)                     },
                "  <A>B</A>\n"                                               },
              { L_, 1, 2, 0,{o(A),a(B,C),c(A)           },
                "  <A>\n    <B>C</B>\n  </A>\n"                              },
              { L_, 1, 2, 0,{o(A),o(B),a(C,D),c(B),c(A) },
                "  <A>\n    <B>\n      <C>D</C>\n    </B>\n  </A>\n"         },

              { L_, 1, 1,80,{a(A,B)                     },
                " <A>B</A>\n"                                                },
              { L_, 1, 1,80,{o(A),a(B,C),c(A)           },
                " <A>\n  <B>C</B>\n </A>\n"                                  },
              { L_, 1, 1,80,{o(A),o(B),a(C,D),c(B),c(A) },
                " <A>\n  <B>\n   <C>D</C>\n  </B>\n </A>\n"                  },

              { L_, 1, 2,80,{a(A,B)                     },
                "  <A>B</A>\n"                                               },
              { L_, 1, 2,80,{o(A),a(B,C),c(A)           },
                "  <A>\n    <B>C</B>\n  </A>\n"                              },
              { L_, 1, 2,80,{o(A),o(B),a(C,D),c(B),c(A) },
                "  <A>\n    <B>\n      <C>D</C>\n    </B>\n  </A>\n"         }
          };

          const int NUM_DATA = sizeof(DATA) / sizeof(DATA[0]);

          for (int i = 0; i != NUM_DATA; ++i) {
              const int           LINE             = DATA[i].d_line;
              const int           INDENT_LEVEL     = DATA[i].d_indentLevel;
              const int           SPACES_PER_LEVEL = DATA[i].d_spacesPerLevel;
              const int           WRAP_COLUMN      = DATA[i].d_wrapColumn;
              const Method *const METHODS          = DATA[i].d_methods;
              const char *const   EXPECTED_STRING  = DATA[i].d_expectedString;

              bdlsb::MemOutStreamBuf streamBuf;
              balxml::Formatter formatter(
                  &streamBuf, INDENT_LEVEL, SPACES_PER_LEVEL, WRAP_COLUMN);

              for (const Method *methodPtr  = METHODS;
                                 methodPtr != METHODS + k_MAX_METHODS;
                               ++methodPtr) {
                  const Method& method = *methodPtr;
                  method(&formatter);
              }

              const bsl::string_view EXPECTED = EXPECTED_STRING;
              const bsl::string_view actual(streamBuf.data(),
                                            streamBuf.length());

              ASSERTV(LINE, EXPECTED, actual, EXPECTED == actual);
          }
      } break;
      case 23: {
        // --------------------------------------------------------------------
        // TESTING BEHAVIOR CHECKSUMS
        //   Ensure that the behavior of 'balxml::Formatter' with certain
        //   permutations of options has not changed.
        //
        //   At around the time this test case was added, the
        //   'balxml_formatter' component underwent a heavy refactoring.  This
        //   test case intends to detect and reject any difference in behavior
        //   for certain initial configurations of the formatter that may have
        //   been introduced by the refactoring, or a subsequent change.  The
        //   test performs a depth-ordered enumeration of all possible valid
        //   sequences of method calls to 'balxml::Formatter' up to length 5,
        //   given a set of initial options.  Since the amount of output
        //   produced by this enumeration is enormous, this test case verifies
        //   that the output has a specific MD5 checksum, as opposed to testing
        //   all the output for equality.
        //
        // Concerns:
        //: 1 The checksum of all valid call sequences on 'balxml::Formatter'
        //:   up to length 5 are equal to the checksums calculated from an
        //:   earlier version of this component.
        //
        // Plan:
        //: 1 For several sets of settings, take the MD5 checksum of the output
        //:   of all valid call sequences of 'balxml::Formatter' up to length
        //:   5 (using sample arguments where necessary) and verify that the
        //:   checksums are equal to "known-good" values.
        //
        // TESTING
        //   ALL BEHAVIOR
        // --------------------------------------------------------------------


        if (verbose) {
            bsl::cout << "\nBEHAVIOR CHECKSUMS"
                      << "\n==================" << bsl::endl;
        }

        const struct {
            int         d_line;
            int         d_initialIndentLevel;
            int         d_spacesPerLevel;
            int         d_wrapColumn;
            const char *d_behaviorChecksum;
        } DATA[] = {
            //          INITIAL INDENT LEVEL
            //         .--------------------
            //  LINE  /   SPACES PER LEVEL
            // .---- /   .----------------
            // |    /   /   WRAP COLUMN
            // |   /   /   .-----------
            // |  /   /   /          MD5 BEHAVIOR CHECKSUM
            //-- --- --- --- -----------------------------------
            { L_,  0,  0,  0, "0332f6b2d3fe5970c2ccf771ccb5312a" },
            { L_,  0,  4,  0, "1dcb077b799541a607f8405b263af6b1" },
            { L_,  1,  4,  0, "53ee584707e538e5c01f90e5174dfdf3" },
            { L_,  0,  0, -1, "7c9762dff11cbe5a40b1c6f9c1ff82ee" },
            { L_,  0,  0, 80, "d9071b04adc47a7909214a8280d8c9ba" },
            { L_,  0,  4, 80, "ec05d90294a33f3ee4794e973ab7a21d" },
            { L_,  1,  4, 80, "480e5abd5dbbd1698a54cd8ae385b8fe" }
        };

        const int NUM_DATA = sizeof(DATA) / sizeof(DATA[0]);

        for (int i = 0; i != NUM_DATA; ++i) {
            const int LINE                 = DATA[i].d_line;
            const int INITIAL_INDENT_LEVEL = DATA[i].d_initialIndentLevel;
            const int SPACES_PER_LEVEL     = DATA[i].d_spacesPerLevel;
            const int WRAP_COLUMN          = DATA[i].d_wrapColumn;
            const bsl::string_view BEHAVIOR_CHECKSUM =
                DATA[i].d_behaviorChecksum;

            static const int k_BEHAVIOR_CHECKSUM_DEPTH = 5;

            bdlde::Md5 digest;
            const balxml::EncoderOptions DEFAULT_ENCODER_OPTIONS;
            FormatterModelUtil::appendBehavior(&digest,
                                               k_BEHAVIOR_CHECKSUM_DEPTH,
                                               INITIAL_INDENT_LEVEL,
                                               SPACES_PER_LEVEL,
                                               WRAP_COLUMN,
                                               DEFAULT_ENCODER_OPTIONS);

            bdlsb::MemOutStreamBuf digestStreambuf;
            bsl::ostream           digestStream(&digestStreambuf);
            digestStream << digest;
            const bsl::string_view digestView(digestStreambuf.data(),
                                              digestStreambuf.length());

            ASSERTV(LINE,
                    BEHAVIOR_CHECKSUM,
                    digestView,
                    BEHAVIOR_CHECKSUM == digestView);
        }
      } break;
      case 22: {
        // --------------------------------------------------------------------
        // TESTING that add* functions invalidate the stream on failure
        //
        // Concerns:
        //: 1 That the add* functions, 'addData', 'addListData', and
        //:   'addAttribute' invalidate the stream on error.
        //
        // Plan:
        //: 1 Create a 'ostringstream' object, ss.
        //:
        //: 2 Create a 'balxml::Formatter' object and associate 'ss' with it.
        //:
        //: 3 Invoke each of the three functions under test passing either an
        //:   invalid value or an incorrect formatting mode.
        //:
        //: 4 Verify that 'ss' is invalid after the call.
        //:
        //: 5 Repeat steps 1-4 for all the functions under test and for all
        //:   the error conditions.
        //
        // Testing:
        //   void addData(const TYPE& value, int formattingMode);
        //   void addListData(const TYPE& value, int formattingMode);
        //   void addAttribute(name, const TYPE& value, int formattingMode);
        // --------------------------------------------------------------------

        if (verbose) {
            bsl::cout << "\nTESTING that add* functions invalidate the "
                      << "stream on failure" << bsl::endl;
        }

        // non-UTF strings
        {
            const unsigned char *valuePtr = T1;
            bsl::string          value((const char *)valuePtr);

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addData(value);
                ASSERT(!ss.good());
            }

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addListData(value);
                ASSERT(!ss.good());
            }

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.addAttribute("test", value);
                ASSERT(!ss.good());
            }
        }

#if !defined(BSLS_ASSERT_SAFE_IS_ACTIVE) &&                                   \
    !defined(BSLS_ASSERT_SAFE_IS_ASSUMED)
        // non-generated enum
        {
            Test value = TEST_A;

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addData(value);
                ASSERT(!ss.good());
            }

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addListData(value);
                ASSERT(!ss.good());
            }

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.addAttribute("test", value);
                ASSERT(!ss.good());
            }
        }

        // invalid formatting mode
        {
            int value = 1;
            int mode  = 0x7;

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addData(value, mode);
                ASSERT(!ss.good());
            }

            {
                bsl::ostringstream ss;
                Obj mX(ss);

                mX.openElement("test");
                ASSERT( ss.good());

                mX.addListData(value, mode);
                ASSERT(!ss.good());
            }

            {
                int value = 1;

                bsl::ostringstream ss;
                Obj mX(ss);

                mX.addAttribute("test", value, mode);
                ASSERT(!ss.good());
            }
        }
#endif
      } break;
      case 21: {
        // --------------------------------------------------------------------
        // reset
        //
        // Plans:
        //
        // Since addHeader cannot be called after any manipulators unless
        // reset is called first, we use some calling sequence to verify that
        // reset correctly resets the internal states that may not be
        // otherwise measured through the public interface.  Use the following
        // sequences of manipulator calls:
        //
        //: addheader reset addHeader again
        //: openElement reset flush does not output '>'
        //: openElement closeElement reset addHeader
        //: addComment reset addHeader
        //: addBlankLine reset addHeader
        //: reset addHeader
        //: openElement flush reset addHeader
        // --------------------------------------------------------------------
        if (verbose) {
            bsl::cout << "\nTESTING reset\n" << bsl::endl;
        }

        const int INITINDENT = 5, SPACESPERLEVEL = 4, WRAPCOLUMN = 200;

        {
            if (veryVerbose) {
                bsl::cout << "addheader reset addHeader again" << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.addHeader();
            int outputColumn1 = formatter.outputColumn();
            int indentLevel1 = formatter.indentLevel();

            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            formatter.addHeader();
            int outputColumn3 = formatter.outputColumn();
            int indentLevel3 = formatter.indentLevel();
            ASSERT(outputColumn3 == outputColumn1);
            ASSERT(indentLevel3 == indentLevel1);
        }

        {
            if (veryVerbose) {
                bsl::cout << "openElement reset flush doe not output '>'"
                          << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.openElement("root");

            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            ss.str(bsl::string());
            formatter.flush();
            int outputColumn3 = formatter.outputColumn();
            ASSERT(outputColumn3 == 0); // there must be no '>'
            ASSERT(ss.str().empty());
        }

        {
            if (veryVerbose) {
                bsl::cout << "openElement closeElement reset addHeader"
                          << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.openElement("root");
            formatter.closeElement("root");

            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            formatter.addHeader(); // This is now a legal call
        }

        {
            if (veryVerbose) {
                bsl::cout << "addComment reset addHeader" << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.addComment("comment goes here", false);
            formatter.addComment("comment goes here", true);

            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            formatter.addHeader(); // This is now a legal call
        }

        {
            if (veryVerbose) {
                bsl::cout << "addBlankLine reset addHeader" << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.addBlankLine();
            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            formatter.addHeader(); // This is now a legal call
        }

        {
            if (veryVerbose) {
                bsl::cout << "openElement flush reset addHeader" << bsl::endl;
            }
            bsl::ostringstream ss;
            Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

            formatter.openElement("root");
            formatter.flush();

            formatter.reset();
            int outputColumn2 = formatter.outputColumn();
            int indentLevel2 = formatter.indentLevel();
            ASSERT(outputColumn2 == 0);
            ASSERT(indentLevel2 == INITINDENT);

            formatter.addHeader(); // This is now a legal call
        }
      } break;

      case 20: {
        // --------------------------------------------------------------------
        // addNewline, addBlankLine
        //
        // other than following openElement addAttribute, these don't add '>'.
        // add one or two '\n'.
        //
        // minimum testing for now
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING addNewline, addBlankLine\n" << bsl::endl;
          }
          const int INITINDENT = 5, SPACESPERLEVEL = 4, WRAPCOLUMN = 200;

          {
              if (veryVerbose) {
                  bsl::cout << "addNewline and addBlankLine after openElement"
                            << bsl::endl;
              }
              bsl::ostringstream ss;
              Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

              formatter.openElement("element1");
              ss.str(bsl::string());
              formatter.addNewline();
              ASSERT(ss.str() == ">\n");
              ASSERT(formatter.outputColumn() == 0);
              ASSERT(formatter.indentLevel() == INITINDENT + 1);

              formatter.openElement("element2");
              ss.str(bsl::string());
              formatter.addBlankLine();
              ASSERT(ss.str() == ">\n\n");
              ASSERT(formatter.outputColumn() == 0);
              ASSERT(formatter.indentLevel() == INITINDENT + 2);
          }
          {
              if (veryVerbose) {
                  bsl::cout << "addBlankLine after a newline" << bsl::endl;
              }
              bsl::ostringstream ss;
              Obj formatter(ss, INITINDENT, SPACESPERLEVEL, WRAPCOLUMN);

              formatter.openElement("root");
              formatter.addComment("comment with a newline afterwards", true);
              ss.str(bsl::string());
              formatter.addBlankLine();
              ASSERT(ss.str() == "\n"); // only one \n is added
              ASSERT(formatter.outputColumn() == 0);
              ASSERT(formatter.indentLevel() == INITINDENT + 1);
          }

      } break;
      case 19: {
        // --------------------------------------------------------------------
        // rawOutputStream
        //
        // leave empty for now
        // --------------------------------------------------------------------
      } break;
      case 18: {
        // --------------------------------------------------------------------
        // TESTING 'addValidComment'
        //
        // Concerns:
        //: 1 That 'addValidComment' appends a '>' for an opened element if
        //:   it's not closed with a '>'.  It correctly precedes the comment
        //:   with newline and indent when it's called with 'forceNewline'
        //:   true, and adds only a space when 'forceNewline' false.  It
        //:   adds another newline after comment if 'forceNewline' is true.
        //:
        //: 2 That 'addValidComment' omits padding white spaces in open '<!--'
        //:   and '-->' close comment tags if 'omitPaddingWhitespace' is true,
        //:   and uses '<!-- ' and ' -->' tags otherwise.
        //:
        //: 3 That 'addValidComment' returns non-zero value if the 'comment'
        //:   argument contains '--' (double-hyphen) sub-string.
        //:
        //: 4 That 'addValidComment' returns non-zero value if the 'comment'
        //:   argument ending in '-'.
        //
        // Plans:
        //:
        //: 1 Open an element with various initial indentations, with or
        //:   without flush() afterwards.  Then add comment with 'forceNewline'
        //:   true or false and 'omitPaddingWhitespace' true or false.  Check
        //:   for resulting string and 'd_column' value.  (C-1..2)
        //:
        //: 2 Create a test table with a list of valid or invalid comments.
        //:   Call 'addValidComment' to add the comment to 'Formatter' object.
        //:   Test the resultant output for a valid comment, or test the error
        //:   code for an invalid comment.  (C-3..4)
        //
        // Testing:
        //   int addValidComment(const bslstl::StringRef&, bool, bool);
        // --------------------------------------------------------------------

          if (verbose) {
              bsl::cout << "\nTESTING addValidComment\n" << bsl::endl;
          }
          {
              static struct {
                  const int   d_line;
                  const int   d_initIndent;
                  const bool  d_doFlush;
                  const bool  d_forceNewline;
                  const bool  d_omitEnclosingSpace;
                  const char *d_expectedOutput;
              } DATA[] = {
              //----+-----+-------+-------+-------+---------------------------
              // Ln | Ind | Do    | Force | Omit  | Expected
              //    |     | Flush | New   | Encl  | Output
              //    |     |       | Line  | Space |
              //----+-----+-------+-------+-------+---------------------------
              { L_,      0,  false,  false,   true, "> <!--comment-->"       },
              { L_,      0,  false,   true,   true, ">\n"
                                                    "    <!--comment-->\n"   },
              { L_,      0,   true,  false,   true, " <!--comment-->"        },
              { L_,      0,   true,   true,   true, "\n"
                                                    "    <!--comment-->\n"   },
              { L_,      0,  false,  false,  false, "> <!-- comment -->"     },
              { L_,      0,  false,   true,  false, ">\n"
                                                    "    <!-- comment -->\n" },
              { L_,      0,   true,  false,  false, " <!-- comment -->"      },
              { L_,      0,   true,   true,  false, "\n"
                                                    "    <!-- comment -->\n" },
              { L_,      2,  false,  false,   true, "> <!--comment-->"       },
              { L_,      2,  false,   true,   true, ">\n"
                                                    "            "
                                                    "<!--comment-->\n"       },
              { L_,      2,   true,  false,   true, " <!--comment-->"        },
              { L_,      2,   true,   true,   true, "\n"
                                                    "            "
                                                    "<!--comment-->\n"       },
              { L_,      2,  false,  false,  false, "> <!-- comment -->"     },
              { L_,      2,  false,   true,  false, ">\n"
                                                    "            "
                                                    "<!-- comment -->\n"     },
              { L_,      2,   true,  false,  false, " <!-- comment -->"      },
              { L_,      2,   true,   true,  false, "\n"
                                                    "            "
                                                    "<!-- comment -->\n"     },
               };
              enum { DATA_SIZE = sizeof DATA / sizeof *DATA };

              for (int i = 0; i < DATA_SIZE; ++i) {
                  const int   LINE =            DATA[i].d_line;
                  const int   INIT_INDENT =     DATA[i].d_initIndent;
                  const bool  DOFLUSH =         DATA[i].d_doFlush;
                  const bool  FORCENEWLINE =    DATA[i].d_forceNewline;
                  const bool  ENCLOSINGSPACE =  DATA[i].d_omitEnclosingSpace;
                  const char *EXP_OUTPUT =      DATA[i].d_expectedOutput;
                  const int   SPACES_PERLEVEL = 4;
                  const char *COMMENT =         "comment";
                  const char *ELEMENT =         "root";
                  const char *OPEN_TAG =        "<";
                  const char *CLOSE_TAG =       ">";

                  const int  EXP_COLUMN = static_cast<int>(
                                          FORCENEWLINE
                                          ? 0
                                          : INIT_INDENT * SPACES_PERLEVEL
                                            + bsl::strlen(OPEN_TAG)
                                            + bsl::strlen(ELEMENT)
                                            + (DOFLUSH
                                               ? bsl::strlen(CLOSE_TAG)
                                               : 0)
                                            + bsl::strlen(EXP_OUTPUT));

                  if (veryVeryVerbose) {
                      T_ P_(LINE) P_(INIT_INDENT) P_(DOFLUSH) P_(FORCENEWLINE)
                      P_(ENCLOSINGSPACE) P(EXP_COLUMN);
                      T_ P(EXP_OUTPUT);
                  }

                  bsl::ostringstream ss;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, INT_MAX);
                                                             // big wrap column

                  formatter.openElement(ELEMENT);

                  if (DOFLUSH) {
                      formatter.flush();
                  }

                  ss.str(bsl::string());
                  LOOP_ASSERT(LINE,
                              0 == formatter.addValidComment(COMMENT,
                                                             FORCENEWLINE,
                                                             ENCLOSINGSPACE));

                  LOOP3_ASSERT(LINE, ss.str(), EXP_OUTPUT,
                               ss.str() == EXP_OUTPUT);

                  LOOP3_ASSERT(LINE, formatter.outputColumn(), EXP_COLUMN,
                               formatter.outputColumn() == EXP_COLUMN);
              }
          }
          if (verbose) {
              bsl::cout << "\nTESTING addValidComment invalid comments\n"
                        << bsl::endl;
          }
          {
              static struct {
                  const int   d_line;
                  const int   d_result;
                  const bool  d_omitEnclosingSpace;
                  const char *d_comment;
                  const char *d_expectedOutput;
              } DATA[] = {
              //----+--------+-----------+------------+-----------------------
              // Ln | Result | Omit      | Comment    | Expected
              //    |        | Enclosing |            | Output
              //    |        | Space     |            |
              //----+--------+-----------+------------+-----------------------

              //------------------- Test single hyphen -----------------------
              { L_,        0,        true,  "comment",  " <!--comment-->"    },
              { L_,        0,       false,  "comment",  " <!-- comment -->"  },

              { L_,        0,        true,  "-comment", " <!---comment-->"   },
              { L_,        0,       false,  "-comment", " <!-- -comment -->" },

              { L_,        0,        true,  "com-ment", " <!--com-ment-->"   },
              { L_,        0,       false,  "com-ment", " <!-- com-ment -->" },

              { L_,        1,        true,  "comment-", ""                   },
              { L_,        0,       false,  "comment-", " <!-- comment- -->" },

              //------------------- Test double hyphen -----------------------
              { L_,        1,       false, "--comment", ""                   },
              { L_,        1,       false, "com--ment", ""                   },
              { L_,        1,       false, "comment--", ""                   },
              { L_,        1,        true, "--comment", ""                   },
              { L_,        1,        true, "com--ment", ""                   },
              { L_,        1,        true, "comment--", ""                   },
              };
              enum { DATA_SIZE = sizeof DATA / sizeof *DATA };

              for (int i = 0; i < DATA_SIZE; ++i) {
                  const int   LINE =            DATA[i].d_line;
                  const int   EXP_RESULT =      DATA[i].d_result;
                  const bool  ENCLOSINGSPACE =  DATA[i].d_omitEnclosingSpace;
                  const char *COMMENT =         DATA[i].d_comment;
                  const char *EXP_OUTPUT =      DATA[i].d_expectedOutput;
                  const int   INIT_INDENT =     0;
                  const int   SPACES_PERLEVEL = 4;
                  const char *ELEMENT =         "root";

                  if (veryVeryVerbose) {
                      T_ P_(LINE) P_(COMMENT) P(EXP_OUTPUT);
                  }

                  bsl::ostringstream ss;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, INT_MAX);
                                                             // big wrap column

                  formatter.openElement(ELEMENT);
                  formatter.flush();

                  ss.str(bsl::string());
                  LOOP_ASSERT(LINE,
                              EXP_RESULT == formatter.addValidComment(
                                                              COMMENT,
                                                              false,
                                                              ENCLOSINGSPACE));

                  LOOP3_ASSERT(LINE, ss.str(), EXP_OUTPUT,
                               ss.str() == EXP_OUTPUT);
              }
          }
      } break;
      case 17: {
        // --------------------------------------------------------------------
        // addComment (minimum testing for now)
        //
        // Concerns:
        //
        // addComment appends a '>' for an opened element if it's not closed
        // with a '>'.  It correctly precedes the comment with newline and
        // indent when it's called with forceNewline true, and adds only a
        // space when forceNewline false.  It adds another newline after
        // comment if forceNewline is true.
        //
        // Plans:
        //
        // open an element with various initial indentations, with or without
        // flush() afterwards.  Then add comment with forceNewline true or
        // false.  Check for resulting string, d_column, d_indentLevel.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING addComment\n" << bsl::endl;
          }
          static struct {
              int d_line;
              int d_initIndent;
              bool d_doFlush;
              bool d_forceNewline;
          } DATA[] = {
              { L_, 0, false, false },
              { L_, 0, false, true },
              { L_, 0, true, false },
              { L_, 0, true, true },
              { L_, 10, false, false },
              { L_, 10, false, true },
              { L_, 10, true, false },
              { L_, 10, true, true },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };

          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const int INIT_INDENT = DATA[i].d_initIndent;
              const bool DOFLUSH = DATA[i].d_doFlush;
              const bool FORCENEWLINE = DATA[i].d_forceNewline;
              const int SPACES_PERLEVEL = 4;
              const char *COMMENT = "comment";

              bsl::ostringstream ss;
              Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, INT_MAX);
                                                             // big wrap column

              formatter.openElement("root");

              int expectedColumn = static_cast<int>(
                                              INIT_INDENT * SPACES_PERLEVEL + 1
                                                        + bsl::strlen("root"));

              if (DOFLUSH) {
                  formatter.flush();
              }

              bsl::string expected;

              ss.str(bsl::string());
              formatter.addComment(COMMENT, FORCENEWLINE);

              if (!DOFLUSH) {
                  expected += '>';
              }
              ++expectedColumn;

              if (FORCENEWLINE) {
                  expected += '\n';
                  expected.append((INIT_INDENT + 1) * SPACES_PERLEVEL, ' ');
                  expectedColumn = (INIT_INDENT + 1) * SPACES_PERLEVEL;
              }
              else {
                  expected += ' ';
                  ++expectedColumn;
              }

              expected += "<!-- ";
              expected += COMMENT;
              expected += " -->";

              expectedColumn += 5 + static_cast<int>(bsl::strlen(COMMENT)) + 4;
              if (FORCENEWLINE) {
                  expected += '\n';
                  expectedColumn = 0;
              }
              LOOP_ASSERT(LINE, ss.str() == expected);
              LOOP_ASSERT(LINE, formatter.outputColumn() == expectedColumn);
          }
      } break;
      case 16: {
        // --------------------------------------------------------------------
        // flush
        //
        // leave empty for now
        // --------------------------------------------------------------------
      } break;
      case 15: {
        // --------------------------------------------------------------------
        // addData, addListData, addAttribute deals well with overloaded data
        // types.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout <<
                  "\nTESTING overloaded addData addListData addAttribute\n"
                        << bsl::endl;
          }

          balxml::EncoderOptions opt1;
          opt1.setDatetimeFractionalSecondPrecision(2);
          balxml::EncoderOptions opt2;
          opt2.setUseZAbbreviationForUtc(true);

          static struct {
              int                           d_line;
              const balxml::EncoderOptions *d_encoderOptions_p;
              const char                   *d_name;
              ScalarData                    d_originalValue;
              const char                   *d_displayedValue; // if 0, use
                                                              // inStream to
                                                              // generate
                                                              // displayed
                                                              // value
          } DATA[] = {
              { L_, 0, "Char", (char)'\0', "0" },
              { L_, 0, "Char", (char)'A', "65" },
              { L_, 0, "Char", (char)128, "-128" },
              { L_, 0, "Char", (char)-128, "-128" },
              { L_, 0, "Short", (short)0, "0" },
              { L_, 0, "Short", (short)SHRT_MAX, 0 },
              { L_, 0, "Short", (short)SHRT_MIN, 0 },
              { L_, 0, "Int", (int)0, "0" },
              { L_, 0, "Int", (int)INT_MAX, 0 },
              { L_, 0, "Int", (int)INT_MIN, 0 },
              { L_, 0, "Int64", (bsls::Types::Int64)0, 0 },
              { L_, 0, "Int64", (bsls::Types::Int64)LONG_MAX, 0 },
              { L_, 0, "Int64", (bsls::Types::Int64)LONG_MIN, 0 },

#ifdef BALXML_FORMATTER_EXTRA_ZERO_PADDING_FOR_EXPONENTS
              { L_, 0, "Float", (float)0.000000000314159, "3.14159e-010" },
              { L_, 0, "Float", (float)3.14159e100, "+INF" },
              { L_, 0, "Double", (double)0.0000000000000000314, "3.14e-017"  },
#else
              { L_, 0, "Float", (float)0.000000000314159, 0 },
              { L_, 0, "Float", (float)3.14159e100, "+INF" },
              { L_, 0, "Double", (double)0.0000000000000000314, 0 },
#endif

              { L_, 0, "Double", (double)3.14e200, 0 },
              { L_, 0, "Datetime", bdlt::Datetime(1, 1, 1, 0, 0, 0, 0),
                    "0001-01-01T00:00:00.000000" },

              { L_, 0, "Datetime", bdlt::Datetime(2005, 1, 22, 23, 59, 59, 999,
                                                 999),
                    "2005-01-22T23:59:59.999999" },
              { L_, &opt1, "Datetime", bdlt::Datetime(2005, 1, 22, 23, 59, 59,
                                                     999, 999),
                    "2005-01-22T23:59:59.99" },
              { L_, 0, "DatetimeTz", bdlt::DatetimeTz(
                      bdlt::Datetime(2016, 7, 4, 8, 21, 30, 123, 456), 0),
                    "2016-07-04T08:21:30.123456+00:00" },
              { L_, &opt2, "DatetimeTz", bdlt::DatetimeTz(
                      bdlt::Datetime(2016, 7, 4, 8, 21, 30, 123, 456), 0),
                    "2016-07-04T08:21:30.123456Z" },
              { L_, 0, "Date", bdlt::Date(), "0001-01-01" },
              { L_, 0, "Date", bdlt::Date(2005, 1, 22), "2005-01-22" },
              { L_, 0, "Time", bdlt::Time(0, 0, 0, 1), "00:00:00.001000" },
              { L_, 0, "Time", bdlt::Time(), "24:00:00.000000" },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          bsl::ostringstream ss;

          for (int i = 0; i < DATA_SIZE; ++i) {
              const int                     LINE     = DATA[i].d_line;
              const char                   *NAME     = DATA[i].d_name;
              const ScalarData              ORIGINAL = DATA[i].d_originalValue;
              const balxml::EncoderOptions *OPTIONS  =
                                                    DATA[i].d_encoderOptions_p;

              balxml::EncoderOptions options;
              if (OPTIONS != 0) {
                  options = *OPTIONS;
              }

              Obj formatter(ss, options, 0, 4, INT_MAX); // big wrap column
              formatter.openElement("root");

              if (veryVerbose) {
                  T_ P_(LINE) P_(NAME) P(ORIGINAL)
              }

              bsl::string DISPLAYED;
              if (!DATA[i].d_displayedValue) {
                  bsl::ostringstream displayed;
                  displayed << ORIGINAL;
                  DISPLAYED = displayed.str();
              }
              else {
                  DISPLAYED = DATA[i].d_displayedValue;
              }

              formatter.openElement(NAME);
              ss.str(bsl::string());

              ORIGINAL.addAttribute(NAME, &formatter);

              LOOP2_ASSERT(LINE, ss.str(), ss.str() ==
                          bsl::string(" ") + NAME + "=\"" + DISPLAYED +
                          "\"");
              formatter.flush(); // add a '>' to close the opening tag
              ss.str(bsl::string());

              ORIGINAL.addData(&formatter);

              LOOP2_ASSERT(LINE, ss.str(), ss.str() == DISPLAYED);
              formatter.closeElement(NAME);
              formatter.openElement(NAME);
              formatter.flush(); // add a '>' to close the opening tag
              ss.str(bsl::string());

              ORIGINAL.addListData(&formatter);

              LOOP2_ASSERT(LINE, ss.str(), ss.str() == DISPLAYED);
          }
      } break;
      case 14: {
        // --------------------------------------------------------------------
        // addData, addListData, addAttribute string value escaping, UTF8
        // string printing, control character truncation
        //
        // Concerns:
        //
        // The escapable characters (' " < > &) in a string value passed to
        // these manipulators are escaped properly, the UTF8 code are printed
        // properly, and string with control characters are truncated properly.
        //
        // Plans:
        //
        // Pass some strings with escapable characters and UTF8 code and check
        // the resulting string and d_column, d_indentLevel.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout <<
                  "\nTESTING addData addListData addAttribute string\n"
                        << bsl::endl;
          }
          static struct {
              int                  d_line;
              const char          *d_name;
              const unsigned char *d_originalValue;
              const unsigned char *d_displayedValue;
              bool                 d_isValid;
          } DATA[] = {
              { L_, "EscapableStringR",        R1, R2, true   },
              { L_, "EscapableStringS",        S1, S2, true   },
              { L_, "TruncatingControlCharsT", T1, T2, false  },
              { L_, "TruncatingControlCharsU", U1, U2, false  },
              { L_, "TruncatingControlCharsV", V1, V2, false  },
              { L_, "UTF8StringX",              X,  X, false  },
              { L_, "UTF8StringY",              Y,  Y, false  },
              { L_, "UTF8StringZ",              Z,  Z, false  },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };

          bsl::ostringstream ss;
          Obj formatter(ss, 0, 4, INT_MAX); // big wrap column
          formatter.openElement("root");

          for (int i = 0; i < DATA_SIZE; ++i) {
              const int   LINE      = DATA[i].d_line;
              const char *NAME      = DATA[i].d_name;
              const char *ORIGINAL  = (const char *) DATA[i].d_originalValue;
              const bool  VALID     = DATA[i].d_isValid;

              formatter.openElement(NAME);
              ss.str(bsl::string());

              formatter.addAttribute(NAME, ORIGINAL);

              LOOP_ASSERT(LINE, VALID == ss.good());
          }
      } break;
      case 13: {
        // --------------------------------------------------------------------
        // addElementAndData
        //
        // Concerns:
        //: 1 addElementAndData(name, value) is as the sequence of
        //:   openElement(name), addData(value), closeElement(name)
        //
        // Plans:
        //: 1 For a combination of tagName and dataValue, call both
        //:   addElementAndData, and its equivalent sequence and compare the
        //:   result.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING nested closeElement\n" << bsl::endl;
          }
          static const char *names[] = { A, B, C, D };
          static const char *values[] = { I, J, K, L, M };
          for (bsl::size_t name = 0;
               name < sizeof(names) / sizeof(*names);
               ++name) {
              const char *NAME = names[name];
              for (bsl::size_t value = 0;
                   value < sizeof(values) / sizeof(*values);
                   ++value) {
                  const char *VALUE = values[value];
                  Pert pert;
                  while (pert.next()) {
                      bsl::ostringstream ss;
                      const int INIT_INDENT = pert.d_initialIndent;
                      const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                      const int WRAP_COLUMN = pert.d_wrapColumn;
                      bsl::ostringstream ss1, ss2;
                      Obj formatter1(ss1,
                                     INIT_INDENT,
                                     SPACES_PERLEVEL,
                                     WRAP_COLUMN);
                      Obj formatter2(ss2,
                                     INIT_INDENT,
                                     SPACES_PERLEVEL,
                                     WRAP_COLUMN);
                      formatter1.openElement(NAME);
                      formatter1.addData(VALUE);
                      formatter1.closeElement(NAME);

                      formatter2.addElementAndData(NAME, VALUE);
                      LOOP5_ASSERT(name, value, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN,
                                   ss1.str() == ss2.str());
                      LOOP5_ASSERT(name, value, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN,
                                   formatter1.outputColumn() ==
                                   formatter2.outputColumn());
                      LOOP5_ASSERT(name, value, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN,
                                   formatter1.indentLevel() ==
                                   formatter2.indentLevel());
                  }
              }
          }
      } break;
      case 12: {
        // --------------------------------------------------------------------
        // closeElement for nested elements
        //
        // Concerns:
        //: 1 when called multiple times, each time for a nested element,
        //:   closeElement properly decrements the indent level.
        //: 2 add </tag> for each non-innermost element with proper
        //:   indentation, irrespective of whitespace constraint.
        //
        // Plans:
        //
        //     Open a series of elements(with the same whitespace handling
        // constraint), and write one piece of data for the innermost element.
        // Close all elements, after each closure, check for resulting string,
        // d_column and d_indentLevel.
        //
        // Test vector -
        //   tagName,                    // should all work
        //   level of nesting,           // should all work
        //   whitespace option,          // no effect for non-innermost
        //                               // element.  Same effect for the
        //                               // innermost element as in case 10
        //
        // Perturbation -
        //   initial indent level,       // should all work
        //   spaces per level,           // should all work
        //   wrap column,                // should all work
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING nested closeElement\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_name;
              int                 d_levelNesting;
              Obj::WhitespaceType d_ws;
          } DATA[] = {
              { L_, A, 2, Obj::e_PRESERVE_WHITESPACE },
              { L_, A, 5, Obj::e_PRESERVE_WHITESPACE },
              { L_, A, 10, Obj::e_PRESERVE_WHITESPACE },
              { L_, D, 2, Obj::e_PRESERVE_WHITESPACE },
              { L_, D, 10, Obj::e_PRESERVE_WHITESPACE },
              { L_, B, 2, Obj::e_NEWLINE_INDENT },
              { L_, B, 5, Obj::e_NEWLINE_INDENT },
              { L_, B, 10, Obj::e_NEWLINE_INDENT },
              { L_, C, 2, Obj:: e_NEWLINE_INDENT},
              { L_, C, 10, Obj::e_NEWLINE_INDENT },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char *NAME = DATA[i].d_name;
              const int LEVEL_NESTING = DATA[i].d_levelNesting;
              const Obj::WhitespaceType WS = DATA[i].d_ws;
              const char *FIXEDVALUE = "fixed value";
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);

                  for (int level = 0; level < LEVEL_NESTING; ++level) {
                      formatter.openElement(NAME, WS);
                  }
                  formatter.addData(FIXEDVALUE);
                  formatter.closeElement(NAME); // result of this has been
                                                // verified in case 10
                  for (int level = LEVEL_NESTING - 2; level >= 0; --level) {
                      ss.str(bsl::string()); // reset stream
                      formatter.closeElement(NAME);
                      bsl::string expected((INIT_INDENT + level) * // (a)
                                           SPACES_PERLEVEL, ' ');
                      if (WRAP_COLUMN < 0) {
                          expected = "";
                      }

                      expected += "</"; // (b)
                      expected += NAME;
                      expected += ">\n";
                      LOOP6_ASSERT(LINE, WRAP_COLUMN, level,
                                   INIT_INDENT, SPACES_PERLEVEL, expected,
                                   expected == ss.str());
                      LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                                   WRAP_COLUMN, level,
                                   0 == formatter.outputColumn());
                      LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                                   WRAP_COLUMN, level,
                                   INIT_INDENT + level ==
                                   formatter.indentLevel());
                  }
              }
          };
      } break;
      case 11: {
        // --------------------------------------------------------------------
        // closeElement for the root element
        //
        // Concerns:
        //: 1 closeElement properly decrements the indent level and add </tag>
        //:   to the element of name 'tag'.
        //:
        //: 2 After it closes the element, it gives a newline.
        //:
        //: 3 In the case of element opened with BAEXML_NEWLINE_INDENT
        //:   whitespace handling constraint, the closing tag does not share
        //:   the same line as the opening tag, nor does it share the same line
        //:   as the data if there is any.
        //:
        //: 4 In the case of an opened element that has not been completed with
        //:   '>', (d_state is BAEXML_IN_TAG), it closes the element with '/>',
        //:   no matter what whitespace constraint it was opened with.
        //
        // Plans:
        //     Open a single element (the root element) with tag name of
        // different lengths, with different whitespace handling constraints,
        // add one piece of data of different lengths (using addData), then
        // close this element.  Check the resulting string, d_indentLevel,
        // d_column.  The limitation is the resulting d_state should be
        // BAEXML_AT_END, which is not perceptible through the public
        // interface.
        //
        // Test vector -
        //   tagName,               // should all work
        //   dataValue,             // should all work
        //   whitespace handling    // may affect which line to close element
        //
        // Perturbation -
        //   initial indent level,  // should all work
        //   spaces per level,      // should all work
        //   wrap column,           // should all work
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING single closeElement\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_name;
              const char         *d_value;
              Obj::WhitespaceType d_ws;
          } DATA[] = {
              { L_, A, I, Obj::e_PRESERVE_WHITESPACE },
              { L_, A, J, Obj::e_PRESERVE_WHITESPACE },
              { L_, A, M, Obj::e_PRESERVE_WHITESPACE },
              { L_, C, J, Obj::e_PRESERVE_WHITESPACE },
              { L_, D, I, Obj::e_PRESERVE_WHITESPACE },
              { L_, D, M, Obj::e_PRESERVE_WHITESPACE },
              { L_, A, I, Obj::e_NEWLINE_INDENT },
              { L_, A, J, Obj::e_NEWLINE_INDENT },
              { L_, A, M, Obj::e_NEWLINE_INDENT },
              { L_, C, J, Obj::e_NEWLINE_INDENT },
              { L_, D, I, Obj::e_NEWLINE_INDENT },
              { L_, D, M, Obj::e_NEWLINE_INDENT },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char *NAME = DATA[i].d_name;
              const char *VALUE = DATA[i].d_value;
              const Obj::WhitespaceType WS = DATA[i].d_ws;
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  formatter.openElement(NAME, WS);

                  formatter.addData(VALUE); // if BAEXML_NEWLINE_INDENT, this
                                            // VALUE is added onto a different
                                            // line other than the opening
                                            // tag's line
                  ss.str(bsl::string()); // reset stream
                  formatter.closeElement(NAME);

                  bsl::string expected;
                  if (WRAP_COLUMN >= 0 && Obj::e_NEWLINE_INDENT == WS) { // (c)
                      if (bsl::strlen(VALUE) > 0) {
                          // do not add unnecessary newline if VALUE is ""
                          expected += '\n';
                      }
                      expected.append(INIT_INDENT * SPACES_PERLEVEL, ' ');
                  }
                  expected += "</";  // (a) (b)
                  expected += NAME;
                  expected += ">\n";

                  LOOP5_ASSERT(LINE, WS, WRAP_COLUMN, expected, ss.str(),
                               expected == ss.str());
                  LOOP5_ASSERT(LINE, WS, INIT_INDENT,
                               SPACES_PERLEVEL, WRAP_COLUMN,
                               0 == formatter.outputColumn());
                  LOOP5_ASSERT(LINE, WS, INIT_INDENT,
                         SPACES_PERLEVEL, WRAP_COLUMN,
                         INIT_INDENT == formatter.indentLevel());
              }
          }
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int                  LINE = DATA[i].d_line;
              const char                *NAME = DATA[i].d_name;
              const Obj::WhitespaceType  WS   = DATA[i].d_ws;
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  formatter.openElement(NAME, WS);

                  ss.str(bsl::string()); // reset
                  formatter.closeElement(NAME);
                  LOOP4_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, "/>\n" == ss.str()); // (d)
                  LOOP4_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, 0 == formatter.outputColumn());
                  LOOP4_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN,
                               INIT_INDENT == formatter.indentLevel());
              }
          }
      } break;
      case 10: {
        // --------------------------------------------------------------------
        // addListData<bsl::string> - no escaping is tested here
        //
        // Concerns:
        //: 1 addListData puts the formatter in the correct state of
        //:   BAEXML_BETWEEN_TAGS.
        //:
        //: 2 It performs limited line-wrapping for input values if the element
        //:   is opened with BAEXML_WORDWRAP;
        //:
        //: 3 It performs indentation after limited line-wrapping for
        //:   BAEXML_WORDWRAP_INDENT;
        //:
        //: 4 In the case of BAEXML_NEWLINE_INDENT, addListData starts from the
        //:   new line, and performs indentation after line-wrapping.
        //:
        //: 5 In the case of more than one call to addListData within a pair of
        //:   tags, a single space is inserted between adjacent data unless
        //:   when line is wrapped a newline and optionally indentation spaces
        //:   are inserted as in 2, 3, 4.
        //:
        //: 6 Empty data value should not change d_column.
        //
        // Plans:
        //     Add one root element with various whitespace handling.  Call
        // addListData multiple times.  Every time, check for resulting string,
        // d_indentLevel, d_column.
        //
        // Test vector -
        //     values,              // should all work
        //     number of values,    // should all work
        //
        // Perturbation -
        //   whitespace handling    // should all work(may affect first data's
        //                                             start line)
        //   initial indent level,  // should all work
        //   spaces per level,      // should all work
        //   wrap column,           // should all work
        //   call flush(),          // no effect
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING addListData\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_value[100];
              int                 d_numValues;
          } DATA[] = {
              { L_, { I }, 1 },
              { L_, { J }, 1 },
              { L_, { K }, 1 },
              { L_, { L }, 1 },
              { L_, { M }, 1 },
              { L_, { I, I }, 2 },
              { L_, { I, J }, 2 },
              { L_, { I, M }, 2 },
              { L_, { J, I }, 2 },
              { L_, { J, J }, 2 },
              { L_, { J, M }, 2 },
              { L_, { M, I }, 2 },
              { L_, { M, L }, 2 },
              { L_, { M, M }, 2 },
              { L_, { I, I, I, I, I }, 5 },
              { L_, { J, J, J, J, J }, 5 },
              { L_, { L, L, L, L, L }, 5 },
              { L_, { M, M, M, M, M }, 5 },
              { L_, { M, L, K, J, I, M, L, K, J, I,
                      M, L, K, J, I, M, L, K, J, I }, 20 },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char **VALUES = DATA[i].d_value;
              const int NUMVALUES = DATA[i].d_numValues;
              for (int ws = 0; ws <= Obj::e_NEWLINE_INDENT; ++ws) {
                  const Obj::WhitespaceType WS =
                      (Obj::WhitespaceType) ws;
                  Pert pert;
                  while (pert.next()) {
                      bsl::ostringstream ss;
                      const int INIT_INDENT = pert.d_initialIndent;
                      const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                      const int WRAP_COLUMN = pert.d_wrapColumn;
                      const bool DOFLUSH = pert.d_doFlush;
                      const bsl::string rootElemName = "root";
                      Obj formatter(ss,
                                    INIT_INDENT,
                                    SPACES_PERLEVEL,
                                    WRAP_COLUMN);
                      formatter.openElement(rootElemName, WS);
                      int expectedColumn = static_cast<int>(
                                              INIT_INDENT * SPACES_PERLEVEL + 1
                                                      + rootElemName.length());
                      bool isFirstAtLine = true;

                      for (int value = 0; value < NUMVALUES; ++value) {
                          const char *VALUE = VALUES[value];
                          ss.str(bsl::string());
                          bsl::string expected;
                          formatter.addListData(VALUE);
                          if (DOFLUSH) {
                              formatter.flush();
                          }

                          // First data should complete the opening tag with
                          // '>'.  If BAEXML_NEWLINE_INDENT, go to a new line
                          // and indent before data is added.
                          if (0 == value) {
                              expected += '>'; // (a)
                              ++expectedColumn;
                              if (WRAP_COLUMN >= 0
                               && Obj::e_NEWLINE_INDENT == WS) {
                                  expected += '\n'; // (d)
                                  expectedColumn = 0;
                              }
                          }

                          bool isWrapped = expectedColumn == 0;
                          if (bsl::strlen(VALUE) > 0) {
                              if (   WRAP_COLUMN > 0
                                  && expectedColumn + static_cast<int>(
                                             bsl::strlen(VALUE)) >= WRAP_COLUMN
                                  && Obj::e_PRESERVE_WHITESPACE != WS) {
                                  isWrapped = true;
                              }

                              if (isWrapped) { // (b), (c)
                                  switch (WS) {
                                    case Obj::e_WORDWRAP: {
                                        if (expectedColumn > 0) {
                                            expected += '\n'; // (b)
                                        }
                                        expectedColumn = 0;
                                    } break;
                                    case Obj::e_WORDWRAP_INDENT:
                                    case Obj::e_NEWLINE_INDENT: {
                                        if (expectedColumn > 0) {
                                            expected += '\n'; // (b)
                                        }
                                        expected.append((INIT_INDENT + 1) *
                                                        SPACES_PERLEVEL, ' ');
                                                        // (c)
                                        expectedColumn = (INIT_INDENT + 1) *
                                            SPACES_PERLEVEL;
                                    } break;
                                    case Obj::e_PRESERVE_WHITESPACE: {
                                        // Do nothing
                                    } break;
                                  }
                                  isFirstAtLine = true;
                              }
                              if (!isFirstAtLine) {
                                  expected += ' '; // (e)
                                  ++expectedColumn;
                              }

                              expected += VALUE;
                              expectedColumn +=
                                          static_cast<int>(bsl::strlen(VALUE));
                              isFirstAtLine = false;
                          } // else: (f)
                          else if (WRAP_COLUMN < 0) {
                              if (0 == value) {
                                  isFirstAtLine = false;
                              }
                              else {
                                  expected += ' ';
                                  ++expectedColumn;
                              }
                          }

                          LOOP6_ASSERT(LINE, WS, WRAP_COLUMN, value,
                                       expected, ss.str(),
                                       expected == ss.str());
                          if (WRAP_COLUMN >= 0) {
                              LOOP6_ASSERT(LINE, WS, INIT_INDENT,
                                           SPACES_PERLEVEL, WRAP_COLUMN, value,
                                           expectedColumn ==
                                           formatter.outputColumn());
                          }
                          LOOP6_ASSERT(LINE, WS, INIT_INDENT,
                                       SPACES_PERLEVEL, WRAP_COLUMN, value,
                                       INIT_INDENT + 1 ==
                                       formatter.indentLevel());
                      }
                  }
              }
          }
      } break;
      case 9: {
        // --------------------------------------------------------------------
        // addData<bsl::string> - no escaping is tested here
        //
        // Concerns:
        //: 1 addData puts the formatter in the correct state of
        //:   BAEXML_BETWEEN_TAGS.
        //:
        //: 2 addData performs no indentation or line-wrapping for any input
        //:   value.
        //:
        //: 3 In the case of openElement with BAEXML_NEWLINE_INDENT, addData
        //:  starts from the new line, and performs only initial indentation,
        //:  but no other indentation or line-wrapping.
        //:
        //: 4 In the case of more than one call to addData within a pair of
        //:   tags, no spacing is inserted between adjacent data whatsoever.
        //
        // Plans:
        //     Add one root element with various whitespace handling.  Call
        // addData multiple times.  Every time, check for resulting string,
        // d_indentLevel, d_column.
        //
        // Test vector -
        //     values,              // should all work
        //     number of values,    // should all work
        // Perturbation -
        //   whitespace handling    // should all work(may affect first data's
        //                                             start line)
        //   initial indent level,  // should all work
        //   spaces per level,      // should all work
        //   wrap column,           // should all work
        //   call flush(),          // no effect
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING addData\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_value[100];
              int                 d_numValues;
          } DATA[] = {
              { L_, { I }, 1 },
              { L_, { J }, 1 },
              { L_, { K }, 1 },
              { L_, { L }, 1 },
              { L_, { M }, 1 },
              { L_, { I, I }, 2 },
              { L_, { I, J }, 2 },
              { L_, { I, M }, 2 },
              { L_, { J, I }, 2 },
              { L_, { J, J }, 2 },
              { L_, { J, M }, 2 },
              { L_, { M, I }, 2 },
              { L_, { M, L }, 2 },
              { L_, { M, M }, 2 },
              { L_, { I, I, I, I, I }, 5 },
              { L_, { J, J, J, J, J }, 5 },
              { L_, { L, L, L, L, L }, 5 },
              { L_, { M, M, M, M, M }, 5 },
              { L_, { M, L, K, J, I, M, L, K, J, I,
                      M, L, K, J, I, M, L, K, J, I }, 20 },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char **VALUES = DATA[i].d_value;
              const int NUMVALUES = DATA[i].d_numValues;
              for (int ws = 0; ws <= Obj::e_NEWLINE_INDENT; ++ws) {
                  const Obj::WhitespaceType WS = (Obj::WhitespaceType) ws;
                  Pert pert;
                  while (pert.next()) {
                      bsl::ostringstream ss;
                      const int INIT_INDENT = pert.d_initialIndent;
                      const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                      const int WRAP_COLUMN = pert.d_wrapColumn;
                      const bool DOFLUSH = pert.d_doFlush;
                      const bsl::string rootElemName = "root";
                      Obj formatter(ss,
                                    INIT_INDENT,
                                    SPACES_PERLEVEL,
                                    WRAP_COLUMN);
                      formatter.openElement(rootElemName, WS);
                      int expectedColumn = INIT_INDENT * SPACES_PERLEVEL + 1
                                     + static_cast<int>(rootElemName.length());

                      bool isFirstData = true; // the first non-empty data
                      for (int value = 0; value < NUMVALUES; ++value) {
                          const char *VALUE = VALUES[value];
                          ss.str(bsl::string());
                          bsl::string expected;
                          formatter.addData(VALUE);
                          if (DOFLUSH) {
                              formatter.flush();
                          }

                          if (0 == value) {
                              expected += '>'; // (a)
                              ++expectedColumn;
                          }

                          if (WRAP_COLUMN >= 0
                           && Obj::e_NEWLINE_INDENT == WS
                           && 0 == value) {
                              expected += '\n'; // (c)
                              expectedColumn = 0;
                          }

                          if (WRAP_COLUMN >= 0
                           && Obj::e_NEWLINE_INDENT == WS
                           && isFirstData && bsl::strlen(VALUE) > 0) {
                              // first non-empty data write the indentation
                              // for BAEXML_NEWLINE_INDENT
                              expected.append((INIT_INDENT + 1) *   // (c)
                                              SPACES_PERLEVEL, ' ');
                              expectedColumn = (INIT_INDENT + 1) *
                                  SPACES_PERLEVEL;
                              isFirstData = false;
                          }
                          expected += VALUE; // (b) (c) (d)
                          expectedColumn +=
                                          static_cast<int>(bsl::strlen(VALUE));

                          LOOP6_ASSERT(LINE, WS, WRAP_COLUMN, value,
                                       expected, ss.str(),
                                       expected == ss.str());
                          if (WRAP_COLUMN >= 0) {
                              LOOP6_ASSERT(LINE, WS, INIT_INDENT,
                                           SPACES_PERLEVEL, WRAP_COLUMN, value,
                                           expectedColumn ==
                                           formatter.outputColumn());
                          }
                          LOOP6_ASSERT(LINE, WS, INIT_INDENT,
                                       SPACES_PERLEVEL, WRAP_COLUMN, value,
                                       INIT_INDENT + 1 ==
                                       formatter.indentLevel());
                      }
                  }
              }
          }
      } break;
      case 8: {
        // --------------------------------------------------------------------
        // addAttribute to add more than one attribute (no escaping is tested
        // here).
        //
        // Concerns:
        //   Multiple calls to addAttributes leave the formatter in the
        //   same state as a single call does (d_state := BAEXML_IN_TAG).
        //   Results in correct d_indentLevel and correct d_column).
        // Plans:
        //   Add one root element with default whitespace handling for
        //   element data, and add more than one attribute.  These attributes
        //   may cause line wrapping.  Check for resulting string,
        //   d_indentLevel, d_column, absence of '>' for the element's opening
        //   tag.
        // Test vector -
        //     pairs of:
        //      attribute name,          // should all work
        //      attribute value,         // should all work
        //     number of pairs,          // should all work
        // Perturbation -
        //   initial indent level,      // should all work(may affect wrapping)
        //   spaces per level,          // should all work(may affect wrapping)
        //   wrap column,               // should all work(may affect wrapping)
        //   call flush(),              // may add '>' to opened tag
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING multiple addAttribute\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_pair[100]; // name="value" pairs
              int                 d_numPairs;
          } DATA[] = {
              { L_, { A, I, A, J, A, K, A, L, A, M }, 5 },
              { L_, { B, I, B, J, B, K, B, L, B, M }, 5 },
              { L_, { C, I, C, J, C, K, C, L, C, M }, 5 },
              { L_, { D, I, D, J, D, K, D, L, D, M }, 5 },
              { L_, { A, I, B, I, C, I, D, I }, 4 },
              { L_, { A, J, B, J, C, J, D, J }, 4 },
              { L_, { A, K, B, K, C, K, D, K }, 4 },
              { L_, { A, L, B, L, C, L, D, L }, 4 },
              { L_, { A, M, B, M, C, M, D, M }, 4 },
              { L_, { D, I, D, J, D, K, D, L, D, M,
                      C, I, C, J, C, K, C, L, C, M,
                      B, I, B, J, B, K, B, L, B, M,
                      A, I, A, J, A, K, A, L, A, M }, 20 },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char **PAIRS = DATA[i].d_pair;
              const int NUMPAIRS = DATA[i].d_numPairs;
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  const bool DOFLUSH = pert.d_doFlush;
                  const bsl::string rootElemName = "root";
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  formatter.openElement(rootElemName);

                  bsl::string expected(INIT_INDENT * SPACES_PERLEVEL, ' ');
                  expected += "<" + rootElemName;

                  int expectedColumn = static_cast<int>(expected.length());

                  for (int attr = 0; attr < NUMPAIRS; ++attr) {
                      const char *NAME = PAIRS[attr * 2];
                      const char *VALUE = PAIRS[attr * 2 + 1];
                      formatter.addAttribute(NAME, VALUE);

                      bool isWrapped = false;
                      if (   WRAP_COLUMN > 0
                          && (static_cast<int>(  expectedColumn
                                               + 1
                                               + bsl::strlen(NAME)
                                               + 2
                                               + bsl::strlen(VALUE)
                                               + 3) >= WRAP_COLUMN)) {
                          // these numbers refer to the added characters one
                          // attribute might introduce: ' NAME="VALUE"/>'
                          //                            -1-  -2 -   -3  -
                          isWrapped = true;
                      }

                      if (isWrapped) {
                          expected += '\n';
                          expected.append((INIT_INDENT + 1) * SPACES_PERLEVEL,
                                          ' ');
                          expectedColumn = (INIT_INDENT + 1) * SPACES_PERLEVEL
                                    + static_cast<int>(bsl::strlen(NAME)) + 2
                                    + static_cast<int>(bsl::strlen(VALUE)) + 1;
                          // + 1 to account for only: ", but not: "/>
                      }
                      else {
                          expected += ' ';
                          expectedColumn += 1
                                    + static_cast<int>(bsl::strlen(NAME)) + 2
                                    + static_cast<int>(bsl::strlen(VALUE)) + 1;
                      }

                      expected += bsl::string(NAME) + "=\"" + VALUE + "\"";
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, attr,
                                   expected == ss.str());
                      if (WRAP_COLUMN < 0) {
                          LOOP5_ASSERT(LINE, INIT_INDENT,
                                       SPACES_PERLEVEL, WRAP_COLUMN, attr,
                                       expectedColumn >=
                                                     formatter.outputColumn());
                      }
                      else {
                          LOOP5_ASSERT(LINE, INIT_INDENT,
                                       SPACES_PERLEVEL, WRAP_COLUMN, attr,
                                       expectedColumn ==
                                                     formatter.outputColumn());
                      }
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, attr,
                                   INIT_INDENT + 1 == formatter.indentLevel());
                  }
                  if (DOFLUSH) {
                      formatter.flush();
                      expected += '>';
                      ++expectedColumn;
                  }
                  LOOP5_ASSERT(LINE, INIT_INDENT,
                               SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                               expected == ss.str());
                  if (WRAP_COLUMN < 0) {
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                                   expectedColumn >= formatter.outputColumn());
                  }
                  else {
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                                   expectedColumn == formatter.outputColumn());
                  }
                  LOOP5_ASSERT(LINE, INIT_INDENT,
                               SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                               INIT_INDENT + 1 == formatter.indentLevel());

              }
          }
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // addAttribute to add a single attribute (no escaping is tested here)
        //
        // Concerns:
        //     A single call to addAttribute leaves the formatter in the same
        // state (d_state) as openElement does.   Results in correct
        // d_indentLevel and correct d_column).
        //
        // Plans:
        //     Add one root element with default whitespace handling for
        // element data, and add one attribute.  This attribute can be on the
        // same line as the element tag, or can be on the next line because of
        // wrapping.  Check for resulting string(need white-box knowledge of
        // how name="value" is appended), d_indentLevel, d_column, absence of
        // '>' for the element's opening tag (i.e., d_state indirectly).
        //
        // Test vector -
        //     attribute name,          // should all work
        //     attribute value,         // should all work
        //
        // Perturbation -
        //   initial indent level,      // should all work(may affect wrapping)
        //   spaces per level,          // should all work(may affect wrapping)
        //   wrap column,               // should all work(may affect wrapping)
        //   call flush(),              // may add '>' to opened tag
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING single addAttribute\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_attrName;
              const char         *d_attrValue;
          } DATA[] = {
              { L_, A, I },
              { L_, A, J },
              { L_, A, K },
              { L_, A, L },
              { L_, A, M },
              { L_, B, I },
              { L_, B, J },
              { L_, B, K },
              { L_, B, L },
              { L_, B, M },
              { L_, D, I },
              { L_, D, J },
              { L_, D, K },
              { L_, D, M },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char *NAME = DATA[i].d_attrName;
              const char *VALUE = DATA[i].d_attrValue;
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  const bool DOFLUSH = pert.d_doFlush;
                  const bsl::string rootElemName = "root";
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  formatter.openElement(rootElemName);
                  formatter.addAttribute(NAME, VALUE);
                  if (DOFLUSH) {
                      formatter.flush();
                  }

                  bsl::string expected(INIT_INDENT * SPACES_PERLEVEL, ' ');
                  expected += "<" + rootElemName;

                  int expectedColumn = static_cast<int>(expected.length());

                  bool isWrapped = false;
                  if (   WRAP_COLUMN > 0
                      && (static_cast<int>(   expected.length()
                                           + 1
                                           + bsl::strlen(NAME)
                                           + 2
                                           + bsl::strlen(VALUE)
                                           + 3) >= WRAP_COLUMN)) {
                      // these numbers refer to the added characters one
                      // attribute might introduce: <tagName NAME="VALUE"/>
                      //                                   -1-  -2 -   -3  -
                      isWrapped = true;
                  }

                  if (isWrapped) {
                      expected += '\n';
                      expected.append((INIT_INDENT + 1) * SPACES_PERLEVEL,
                                      ' ');
                      expectedColumn = (INIT_INDENT + 1) * SPACES_PERLEVEL;
                  }
                  else {
                      expected += ' ';
                      ++expectedColumn;
                  }

                  expected += bsl::string(NAME) + "=\"" + VALUE + "\"";
                  expectedColumn += static_cast<int>(bsl::strlen(NAME )) + 2
                                  + static_cast<int>(bsl::strlen(VALUE)) + 1;
                  if (DOFLUSH) {
                      expected += '>';
                      ++expectedColumn;
                  }

                  LOOP5_ASSERT(LINE, INIT_INDENT,
                               SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                               expected == ss.str());
                  if (WRAP_COLUMN < 0) {
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                                   expectedColumn >= formatter.outputColumn());
                  }
                  else {
                      LOOP5_ASSERT(LINE, INIT_INDENT,
                                   SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                                   expectedColumn == formatter.outputColumn());
                  }
                  LOOP5_ASSERT(LINE, INIT_INDENT,
                               SPACES_PERLEVEL, WRAP_COLUMN, DOFLUSH,
                               INIT_INDENT + 1 == formatter.indentLevel());
              }
          }
      } break;

      case 6: {
        // --------------------------------------------------------------------
        // Nested openElement
        //
        // Plans:
        //     Check the resulting string and accessors for internal states.
        // Effects of having addHeader and different whitespace handlings are
        // checked in case 4.
        //     For every opened nested-element, calling flush() afterwards
        // should
        //
        // Test vector -
        //   tagName,                    // should all work
        //   level of nesting,           // should all work
        //
        // Perturbation -
        //   initial indent level,       // should all work
        //   spaces per level,           // should all work
        //   wrap column,                // should all work
        //   call flush() afterward each nested element,
        //                               // may add '>' to each opened nested
        //                               // element, but have no effect on the
        //                               // overall document, because
        //                               // openElement is able to complete
        //                               // '>' for previously opened element
        //                               // without using flush()
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING nested openElement\n" << bsl::endl;
          }
          static struct {
              int                 d_line;
              const char         *d_name;
              int                 d_levelNesting;
          } DATA[] = {
              { L_, A, 1 },
              { L_, A, 5 },
              { L_, A, 10 },
              { L_, B, 1 },
              { L_, B, 5 },
              { L_, B, 10 },
              { L_, B, 50 },
              { L_, C, 1 },
              { L_, C, 5 },
              { L_, C, 10 },
              { L_, C, 50 },
              { L_, D, 1 },
              { L_, D, 5 },
              { L_, D, 10 },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const char *NAME = DATA[i].d_name;
              const int LEVEL_NESTING = DATA[i].d_levelNesting;
              Pert pert;
              while (pert.next()) {
                  bsl::ostringstream ss;
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  const bool DOFLUSH = pert.d_doFlush;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  formatter.openElement(NAME);
                  bsl::string ssDoc;  // document from formatter
                  ssDoc += ss.str();

                  bsl::string expectedDoc;
                      // Expected XML doc
                  expectedDoc.append(INIT_INDENT * SPACES_PERLEVEL, ' ');
                  expectedDoc += '<';
                  expectedDoc += NAME;

                  for (int level = 1; level <= LEVEL_NESTING; ++level) {
                      if (DOFLUSH) {
                          ss.str(bsl::string()); // reset stream
                          formatter.flush();
                          ssDoc += ss.str();
                      }

                      ss.str(bsl::string()); // reset stream
                      formatter.openElement(NAME);
                      ssDoc += ss.str();

                      bsl::string expected; // what this single call to
                                            // openElement is expected to
                                            // produce
                      if (!DOFLUSH) {
                          expected += '>';
                      }
                      else {
                          expectedDoc += '>';
                      }

                      expected += '\n';
                      expected.append((INIT_INDENT + level) *
                                      SPACES_PERLEVEL, ' ');
                      expected += '<';
                      expected += NAME;

                      expectedDoc += expected;

                      int expectedColumn = DOFLUSH
                                     ? static_cast<int>(expected.length()) - 1
                                     : static_cast<int>(expected.length()) - 2;
                          // -1 to offset "\n", -2 to offset ">\n"
                      LOOP6_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                                   WRAP_COLUMN, DOFLUSH, level,
                                   expected == ss.str());
                      LOOP6_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                                   WRAP_COLUMN, DOFLUSH, level,
                                   expectedColumn ==
                                   formatter.outputColumn());
                      LOOP6_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                                   WRAP_COLUMN, DOFLUSH, level,
                                   INIT_INDENT + 1 + level ==
                                   formatter.indentLevel());
                  }
                  LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, DOFLUSH, ssDoc == expectedDoc);
                  if (ssDoc != expectedDoc) {
                      P(ssDoc);
                      P(expectedDoc);
                      break;
                  }
              }
          }
      } break;

      case 5: {
        // --------------------------------------------------------------------
        // Root-level openElement
        //
        // Plans:
        //     Call openElement with or without addHeader ahead of it, using
        // tags of different lengths.  Check the resulting string, as well as
        // the accessors for the internal states
        //
        // Test vector -
        //   whether has header,         // should not affect result
        //   tagName,                    // lengths should all work
        //   whitespace handling         // no effect.
        //
        // Perturbation -
        //   initial indent level,       // should all work
        //   spaces per level,           // should all work
        //   wrap column,                // should all work
        //   do flush                    // may add '>' to opened tag
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING root-level openElement\n" << bsl::endl;
          }

          static struct {
              int                 d_line;
              bool                d_hasHeader;
              const char         *d_name;
              Obj::WhitespaceType d_ws;
          } DATA[] = {
              //    hasHeader elemName  ws-handling
              { L_, true,     A,       Obj::e_NEWLINE_INDENT },
              { L_, true,     A,       Obj::e_WORDWRAP_INDENT },
              { L_, true,     A,       Obj::e_WORDWRAP },
              { L_, true,     A,       Obj::e_PRESERVE_WHITESPACE },
              { L_, false,    A,       Obj::e_NEWLINE_INDENT },
              { L_, false,    A,       Obj::e_WORDWRAP_INDENT },
              { L_, false,    A,       Obj::e_WORDWRAP },
              { L_, false,    A,       Obj::e_PRESERVE_WHITESPACE },
              { L_, true,     C,       Obj::e_NEWLINE_INDENT },
              { L_, true,     C,       Obj::e_WORDWRAP_INDENT },
              { L_, false,    D,       Obj::e_NEWLINE_INDENT },
              { L_, false,    D,       Obj::e_WORDWRAP_INDENT },
          };
          enum { DATA_SIZE = sizeof DATA / sizeof *DATA };
          for (int i = 0; i < DATA_SIZE; ++i) {
              const int LINE = DATA[i].d_line;
              const bool HAS_HEADER = DATA[i].d_hasHeader;
              const char *NAME = DATA[i].d_name;
              const Obj::WhitespaceType WS = DATA[i].d_ws;
              Pert pert;
              while (pert.next()) {
                  const int INIT_INDENT = pert.d_initialIndent;
                  const int SPACES_PERLEVEL = pert.d_spacesPerLevel;
                  const int WRAP_COLUMN = pert.d_wrapColumn;
                  const bool DOFLUSH = pert.d_doFlush;
                  bsl::ostringstream ss;
                  Obj formatter(ss, INIT_INDENT, SPACES_PERLEVEL, WRAP_COLUMN);
                  if (HAS_HEADER) {
                      formatter.addHeader();
                  }
                  ss.str(bsl::string());
                      // reset ostring to focus on openElement
                  formatter.openElement(NAME, WS);
                  if (DOFLUSH) {
                      formatter.flush();
                  }

                  int expectedColumn;
                  bsl::string expected(INIT_INDENT * SPACES_PERLEVEL, ' ');
                  expected.append("<");
                  expected.append(NAME);
                  expectedColumn = static_cast<int>(expected.length());

                  if (DOFLUSH) {
                      expected.append(">");
                      ++expectedColumn;
                  }

                  LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, DOFLUSH,
                               expected == ss.str());
                  LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, DOFLUSH,
                               expectedColumn ==
                               formatter.outputColumn());
                  LOOP5_ASSERT(LINE, INIT_INDENT, SPACES_PERLEVEL,
                               WRAP_COLUMN, DOFLUSH,
                               INIT_INDENT + 1 ==
                               formatter.indentLevel());
              }
          }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // addHeader
        //
        // plans:
        //     call addHeader with and without argument.  check that the state
        //     is almost unchanged (except the d_state) using the accessors
        //     and check the header string.  Limitation: Change of d_state is
        //     not perceptible through the public interface.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nTESTING addHeader\n" << bsl::endl;
          }
          int initialIndent = 5;

          bsl::ostringstream stream1;
          Obj formatter1(stream1, initialIndent);
          formatter1.addHeader(); // no arg

          ASSERT(stream1.str() ==
                 "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");
          ASSERT(formatter1.outputColumn() == 0);
          ASSERT(formatter1.indentLevel() == initialIndent);

          bsl::ostringstream stream2;
          Obj formatter2(stream2, initialIndent);
          formatter2.addHeader("iso8896-1"); // with arg

          ASSERT(stream2.str() ==
                 "<?xml version=\"1.0\" encoding=\"iso8896-1\" ?>\n");
          ASSERT(formatter2.outputColumn() == 0);
          ASSERT(formatter2.indentLevel() == initialIndent);
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // CONSTRUCTOR TEST
        //
        // Concerns:
        //     Call both constructors to check the initial states are good
        // using the accessors and rawOutputStream()
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nCONSTRUCTOR TEST\n" << bsl::endl;
          }
          bsl::ostringstream stream1, stream2;
          int initialIndent = 1, spacesPerLevel = 2, wrapColumn = 3;

          {
              Obj formatter1(stream1, initialIndent,
                             spacesPerLevel, wrapColumn);
              Obj formatter2(stream2.rdbuf(), initialIndent,
                             spacesPerLevel, wrapColumn);
              ASSERT(formatter1.outputColumn() == 0);
              ASSERT(formatter2.outputColumn() == 0);

              ASSERT(formatter1.indentLevel() == initialIndent);
              ASSERT(formatter2.indentLevel() == initialIndent);

              ASSERT(formatter1.spacesPerLevel() == spacesPerLevel);
              ASSERT(formatter2.spacesPerLevel() == spacesPerLevel);

              ASSERT(formatter1.wrapColumn() == wrapColumn);
              ASSERT(formatter2.wrapColumn() == wrapColumn);

              ASSERT(formatter1.rawOutputStream().rdbuf() == stream1.rdbuf());
              ASSERT(formatter2.rawOutputStream().rdbuf() == stream2.rdbuf());
          }
          {
              balxml::EncoderOptions opt;
              opt.setUseZAbbreviationForUtc(true);
              Obj formatter1(stream1, opt, initialIndent,
                             spacesPerLevel, wrapColumn);
              Obj formatter2(stream2.rdbuf(), opt, initialIndent,
                             spacesPerLevel, wrapColumn);
              ASSERT(formatter1.outputColumn() == 0);
              ASSERT(formatter2.outputColumn() == 0);

              ASSERT(formatter1.indentLevel() == initialIndent);
              ASSERT(formatter2.indentLevel() == initialIndent);

              ASSERT(formatter1.spacesPerLevel() == spacesPerLevel);
              ASSERT(formatter2.spacesPerLevel() == spacesPerLevel);

              ASSERT(formatter1.wrapColumn() == wrapColumn);
              ASSERT(formatter2.wrapColumn() == wrapColumn);

              ASSERT(formatter1.rawOutputStream().rdbuf() == stream1.rdbuf());
              ASSERT(formatter2.rawOutputStream().rdbuf() == stream2.rdbuf());
              ASSERT(opt == formatter1.encoderOptions());
              ASSERT(opt == formatter2.encoderOptions());
          }
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TEST helper class Pert for orthogonal perturbation
        //
        // Plans:
        //     Check that nested loops permuting from the outermost loop to
        // the innermost loop provide the same result as returned from
        // Pert::next()
        // --------------------------------------------------------------------
          Pert pert;

          for (bsl::size_t i0 = 0;
               i0 < sizeof(Pert::s_doFlush) /
                   sizeof(*Pert::s_doFlush);
               ++i0) {
              for (bsl::size_t i1 = 0;
                   i1 < sizeof(Pert::s_initialIndent) /
                       sizeof(*Pert::s_initialIndent);
                   ++i1) {
                  for (bsl::size_t i2 = 0;
                       i2 < sizeof(Pert::s_spacesPerLevel) /
                           sizeof(*Pert::s_spacesPerLevel);
                       ++i2) {
                      for (bsl::size_t i3 = 0;
                           i3 < sizeof(Pert::s_wrapColumn) /
                               sizeof(*Pert::s_wrapColumn);
                           ++i3) {
                          bool valid = pert.next();
                          LOOP4_ASSERT(i0, i1, i2, i3, valid);
                          LOOP4_ASSERT(i0, i1, i2, i3,
                                       pert.d_doFlush ==
                                       Pert::s_doFlush[i0]);
                          LOOP4_ASSERT(i0, i1, i2, i3,
                                       pert.d_initialIndent  ==
                                       Pert::s_initialIndent[i1]);
                          LOOP4_ASSERT(i0, i1, i2, i3,
                                       pert.d_spacesPerLevel ==
                                       Pert::s_spacesPerLevel[i2]);
                          LOOP4_ASSERT(i0, i1, i2, i3,
                                       pert.d_wrapColumn ==
                                       Pert::s_wrapColumn[i3]);
                      }
                  }
              }
          }
          ASSERT(!pert.next());

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //
        // Concerns:
        //     All manipulator works.
        // Plan:
        //     Write two document in the same stream, separated by a reset()
        // call, the first document shows the result for each of its
        // manipulator calls, while the second document shows the result as a
        // whole.
        // --------------------------------------------------------------------
          if (verbose) {
              bsl::cout << "\nBREATHING TEST\n" << bsl::endl;
          }
          bsl::ostringstream ss;

          balxml::Formatter formatter(ss);
          ASSERT(&ss == &formatter.rawOutputStream());

          formatter.addHeader("UTF-8");
          ASSERT(ss.str() ==
                 "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");

          ss.str(bsl::string());
          formatter.openElement("Fruits");
          ASSERT(ss.str() ==
                 "<Fruits");

          ss.str(bsl::string());
          formatter.openElement("Oranges");
          ASSERT(ss.str() ==
                 ">\n    <Oranges");

          ss.str(bsl::string());
          formatter.addAttribute("farm", "Frances' Orchard");
          ASSERT(ss.str() ==
                 " farm=\"Frances&apos; Orchard\"");

          ss.str(bsl::string());
          formatter.addAttribute("size", 3.5);
          ASSERT(ss.str() ==
                 " size=\"3.5\"");

          ss.str(bsl::string());
          formatter.addNewline();
          ASSERT(ss.str() ==
                 ">\n");

          ss.str(bsl::string());
          formatter.openElement("pickDate");
          ASSERT(ss.str() ==
                 "        <pickDate");

          ss.str(bsl::string());
          formatter.addData(bdlt::Date(2004, 8, 31));
          ASSERT(ss.str() ==
                 ">2004-08-31");

          ss.str(bsl::string());
          formatter.closeElement("pickDate");
          ASSERT(ss.str() ==
                 "</pickDate>\n");

          ss.str(bsl::string());
          formatter.openElement("Features",
                                balxml::Formatter::e_NEWLINE_INDENT);
          ASSERT(ss.str() ==
                 "        <Features");

          ss.str(bsl::string());
          formatter.addListData("Juicy");
          ASSERT(ss.str() ==
                 ">\n"
                 "            Juicy");

          ss.str(bsl::string());
          formatter.addListData("Round");
          ASSERT(ss.str() ==
                 " Round");

          ss.str(bsl::string());
          formatter.addListData("Also shown on Florida license plates");
          ASSERT(ss.str() ==
                 " Also shown on Florida license plates");

          ss.str(bsl::string());
          formatter.closeElement("Features");
          ASSERT(ss.str() ==
                 "\n"
                 "        </Features>\n");

          ss.str(bsl::string());
          formatter.addComment("There's no more data for Oranges");
          ASSERT(ss.str() ==
                 "        <!-- There's no more data for Oranges -->\n");

          ss.str(bsl::string());
          formatter.closeElement("Oranges");
          ASSERT(ss.str() ==
                 "    </Oranges>\n");

          ss.str(bsl::string());
          formatter.closeElement("Fruits");
          ASSERT(ss.str() ==
                 "</Fruits>\n");

          ss.str(bsl::string());

          ASSERT(0 == formatter.status());

          ss.clear(bsl::ios::failbit);
          ASSERT(! ss.good());
          ASSERT(-1 == formatter.status());

          formatter.reset();
          ASSERT(ss.good());
          ASSERT(0 == formatter.status());

          formatter.addHeader("iso8896");
          formatter.openElement("Grains");
          formatter.addBlankLine();
          bsl::ostream& rawStream = formatter.rawOutputStream();
          rawStream << "<someTag> anything </someTag>" << bsl::endl;
          formatter.closeElement("Grains");

          ASSERT(ss.str() ==
                 "<?xml version=\"1.0\" encoding=\"iso8896\" ?>\n"
                 "<Grains>\n"
                 "\n"
                 "<someTag> anything </someTag>\n"
                 "</Grains>\n"
              );
      } break;
      case -1: {
          balxml::Formatter formatter(bsl::cout, 0, 4, 40);
          formatter.openElement("holidays",
                                balxml::Formatter::e_NEWLINE_INDENT);
          formatter.addListData(bdlt::Date(2005, 1, 17));
          formatter.addListData(bdlt::Date(2005, 2, 21));
          formatter.addListData(bdlt::Date(2005, 3, 25));
          formatter.addListData(bdlt::Date(2005, 5, 30));
          formatter.addListData(bdlt::Date(2005, 7, 4));
          formatter.addListData(bdlt::Date(2005, 9, 5));
          formatter.addListData(bdlt::Date(2005, 11, 24));
          formatter.addListData(bdlt::Date(2005, 12, 25));
          formatter.closeElement("holidays");

      } break;
      case -2: {
          balxml::Formatter formatter(bsl::cout, 0, 4, 40);

          formatter.addHeader("UTF-8");

          formatter.openElement("Fruits");
          formatter.openElement("Oranges");
          formatter.addAttribute("farm", "Frances' Orchard");
          // notice that the apostrophe in the string will be escaped
          formatter.addAttribute("size", 3.5);

          formatter.addElementAndData("Quantity", 12);

          formatter.openElement("pickDate");
          formatter.addData(bdlt::Date(2004, 8, 31));
          formatter.closeElement("pickDate");

          formatter.openElement("Feature");
          formatter.addAttribute("shape", "round");
          formatter.closeElement("Feature");

          formatter.addComment("No wrapping for long comments");

          formatter.closeElement("Oranges");

          formatter.addBlankLine();

          formatter.openElement("Apples");
          formatter.addAttribute("farm", "Fuji & Sons");
          formatter.addAttribute("size", 3);

          formatter.openElement("pickDates",
                                balxml::Formatter::e_NEWLINE_INDENT);
          formatter.addListData(bdlt::Date(2005, 1, 17));
          formatter.addListData(bdlt::Date(2005, 2, 21));
          formatter.addListData(bdlt::Date(2005, 3, 25));
          formatter.addListData(bdlt::Date(2005, 5, 30));
          formatter.addListData(bdlt::Date(2005, 7, 4));
          formatter.addListData(bdlt::Date(2005, 9, 5));
          formatter.addListData(bdlt::Date(2005, 11, 24));
          formatter.addListData(bdlt::Date(2005, 12, 25));

          formatter.closeElement("pickDates");

          formatter.openElement("Feature");
          formatter.addAttribute("color", "red");
          formatter.addAttribute("taste", "juicy");
          formatter.closeElement("Feature");

          formatter.closeElement("Apples");

          formatter.closeElement("Fruits");

          formatter.reset();
          // reset the formatter for a new document in the same stream

          formatter.addHeader();
          formatter.openElement("Grains");

          bsl::ostream& os = formatter.rawOutputStream();
          os << "<free>anything that can mess up the XML doc</free>";
          // Now coming back to the formatter, but can't do the following:
          // formatter.addAttribute("country", "USA");
          formatter.addData("Corn, Wheat, Oat");
          formatter.closeElement("Grains");

      } break;
      case -3: {
          // Performance test
          //  Usage: balxml_formatter.t.dbg_exc_mt -3 <size> <reps> <verbose>
          //   Where <size> is the approximate message size, in bytes,
          //        <reps> is the number of test repetitions and
          //        <verbose> is non-empty to dump the messages to stdout
          //  Defaults: <size> = 10000, <reps> = 100

          bsl::size_t msgSize = argc > 2 ? bsl::atoi(argv[2]) : 10000;
          int reps            = argc > 3 ? bsl::atoi(argv[3]) : 100;
          verbose             = argc > 4;
          veryVerbose         = 0;
          veryVeryVerbose     = 0;

          bdlsb::MemOutStreamBuf output;

          for (int i = 0; i < reps; ++i) {
              output.reset();
              output.reserveCapacity(msgSize + 100);
              balxml::Formatter formatter(&output, 0, 0, -1);
              formatter.addHeader("UTF-8");
              formatter.openElement("root");
              while (output.length() < msgSize - 6) {
                  formatter.openElement("space");
                  formatter.openElement("solarSystem");
                  formatter.openElement("planet");
                  formatter.addAttribute("order", 1);
                  formatter.addData("Mercury");
                  formatter.closeElement("planet");
                  formatter.openElement("planet");
                  formatter.addAttribute("order", 3.0);
                  formatter.addData("Earth");
                  formatter.closeElement("planet");
                  formatter.closeElement("solarSystem");
                  formatter.closeElement("space");
              }
              formatter.closeElement("root");

              if (verbose && 0 == i) {
                  bslstl::StringRef outputStr((const char *) output.data(),
                                            output.length());
                  P(outputStr);
              }
          }
      } break;
      case -4: {
        // Wrap column test
        //  This is a legacy test case that verifies the behavior of
        //  various low and negative wrap column settings.
        bsl::cout << "Wrap Column 0" << bsl::endl;
        balxml::Formatter mX(bsl::cout.rdbuf(), 0, 4, 0);
        mX.openElement("Oranges");
        mX.addAttribute("farm", "Frances' Orchard"); // ' is escaped
        mX.addAttribute("size", 3.5);
        mX.closeElement("Oranges");

        bsl::cout << "Wrap Column 5" << bsl::endl;
        balxml::Formatter mY(bsl::cout.rdbuf(), 0, 4, 5);
        mY.openElement("Oranges");
        mY.addAttribute("farm", "Frances' Orchard"); // ' is escaped
        mY.addAttribute("size", 3.5);
        mY.closeElement("Oranges");

        bsl::cout << "Wrap Column -1" << bsl::endl;
        balxml::Formatter mZ(bsl::cout.rdbuf(), 0, 4, -1);
        mZ.openElement("Oranges");
        mZ.addAttribute("farm", "Frances' Orchard"); // ' is escaped
        mZ.addAttribute("size", 3.5);
        mZ.closeElement("Oranges");
        bsl::cout << bsl::endl;

        bsl::cout << "Wrap Column 0" << bsl::endl;
        balxml::Formatter mX1(bsl::cout.rdbuf(), 0, 4, 0);
        mX1.addElementAndData("farm", "Frances' Orchard"); // ' is escaped

        bsl::cout << "Wrap Column 5" << bsl::endl;
        balxml::Formatter mX2(bsl::cout.rdbuf(), 0, 4, 5);
        mX2.addElementAndData("farm", "Frances' Orchard"); // ' is escaped

        bsl::cout << "Wrap Column -1" << bsl::endl;
        balxml::Formatter mA(bsl::cout.rdbuf(), 0, 4, -1);
        mA.addElementAndData("farm", "Frances' Orchard"); // ' is escaped

      } break;
      default: {
        bsl::cerr << "WARNING: CASE `" << test << "' NOT FOUND." << bsl::endl;
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        bsl::cerr << "Error, non-zero test status = " << testStatus << "."
                  << bsl::endl;
    }
    return testStatus;
}

#ifdef BSLS_PLATFORM_CMP_MSVC
// Pop the stack of disabled warnings that were pushed before the 'main'
// function definition.
#pragma warning( pop )
#endif

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
