// bdlt_fixutil.t.cpp                                                 -*-C++-*-
#include <bdlt_fixutil.h>

#include <bdlt_date.h>
#include <bdlt_datetime.h>
#include <bdlt_datetimetz.h>
#include <bdlt_datetz.h>
#include <bdlt_time.h>
#include <bdlt_timetz.h>

#include <bslim_testutil.h>

#include <bsls_asserttest.h>
#include <bsls_review.h>

#include <bslx_testinstream.h>

#include <bsl_cctype.h>      // 'isdigit'
#include <bsl_cstdlib.h>
#include <bsl_cstring.h>
#include <bsl_iostream.h>
#include <bsl_sstream.h>
#include <bsl_string.h>

#undef SEC

using namespace BloombergLP;
using namespace bsl;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test consists of a suite of static member functions
// (pure functions) that perform conversions between the values of several
// 'bdlt' vocabulary types and corresponding string representations, where the
// latter are defined by the FIX standard.  The general plan is that each
// function is to be independently tested using the table-driven technique.  A
// set of test vectors is defined globally for use in testing all functions.
// This global data is sufficient for thoroughly testing the string generating
// functions, but additional test vectors are required to address concerns
// specific to the string parsing functions.  Hence, additional test data is
// defined locally to the test cases that verify parsing.
//
// Global Concerns:
//: o No memory is ever allocated from the global allocator.
//: o Precondition violations are detected in appropriate build modes.
//-----------------------------------------------------------------------------
// CLASS METHODS
// [ 1] int generate(char *, int, const Date&);
// [ 1] int generate(char *, int, const Date&, const Config&);
// [ 2] int generate(char *, int, const Time&);
// [ 2] int generate(char *, int, const Time&, const Config&);
// [ 3] int generate(char *, int, const Datetime&);
// [ 3] int generate(char *, int, const Datetime&, const Config&);
// [ 4] int generate(char *, int, const DateTz&);
// [ 4] int generate(char *, int, const DateTz&, const Config&);
// [ 5] int generate(char *, int, const TimeTz&);
// [ 5] int generate(char *, int, const TimeTz&, const Config&);
// [ 6] int generate(char *, int, const DatetimeTz&);
// [ 6] int generate(char *, int, const DatetimeTz&, const Config&);
// [ 1] int generate(string *, const Date&);
// [ 1] int generate(string *, const Date&, const Config&);
// [ 2] int generate(string *, const Time&);
// [ 2] int generate(string *, const Time&, const Config&);
// [ 3] int generate(string *, const Datetime&);
// [ 3] int generate(string *, const Datetime&, const Config&);
// [ 4] int generate(string *, const DateTz&);
// [ 4] int generate(string *, const DateTz&, const Config&);
// [ 5] int generate(string *, const TimeTz&);
// [ 5] int generate(string *, const TimeTz&, const Config&);
// [ 6] int generate(string *, const DatetimeTz&);
// [ 6] int generate(string *, const DatetimeTz&, const Config&);
// [ 1] ostream generate(ostream&, const Date&);
// [ 1] ostream generate(ostream&, const Date&, const Config&);
// [ 2] ostream generate(ostream&, const Time&);
// [ 2] ostream generate(ostream&, const Time&, const Config&);
// [ 3] ostream generate(ostream&, const Datetime&);
// [ 3] ostream generate(ostream&, const Datetime&, const Config&);
// [ 4] ostream generate(ostream&, const DateTz&);
// [ 4] ostream generate(ostream&, const DateTz&, const Config&);
// [ 5] ostream generate(ostream&, const TimeTz&);
// [ 5] ostream generate(ostream&, const TimeTz&, const Config&);
// [ 6] ostream generate(ostream&, const DatetimeTz&);
// [ 6] ostream generate(ostream&, const DatetimeTz&, const Config&);
// [ 1] int generateRaw(char *, const Date&);
// [ 1] int generateRaw(char *, const Date&, const Config&);
// [ 2] int generateRaw(char *, const Time&);
// [ 2] int generateRaw(char *, const Time&, const Config&);
// [ 3] int generateRaw(char *, const Datetime&);
// [ 3] int generateRaw(char *, const Datetime&, const Config&);
// [ 4] int generateRaw(char *, const DateTz&);
// [ 4] int generateRaw(char *, const DateTz&, const Config&);
// [ 5] int generateRaw(char *, const TimeTz&);
// [ 5] int generateRaw(char *, const TimeTz&, const Config&);
// [ 6] int generateRaw(char *, const DatetimeTz&);
// [ 6] int generateRaw(char *, const DatetimeTz&, const Config&);
// [ 7] int parse(Date *, const char *, int);
// [ 8] int parse(Time *, const char *, int);
// [ 9] int parse(Datetime *, const char *, int);
// [ 7] int parse(DateTz *, const char *, int);
// [ 8] int parse(TimeTz *, const char *, int);
// [ 9] int parse(DatetimeTz *, const char *, int);
// [ 7] int parse(Date *result, const StringRef& string);
// [ 8] int parse(Time *result, const StringRef& string);
// [ 9] int parse(Datetime *result, const StringRef& string);
// [ 7] int parse(DateTz *result, const StringRef& string);
// [ 8] int parse(TimeTz *result, const StringRef& string);
// [ 9] int parse(DatetimeTz *result, const StringRef& string);
//-----------------------------------------------------------------------------
// [10] USAGE EXAMPLE
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
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

//=============================================================================
//                  GLOBALS, TYPEDEFS, CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

typedef bdlt::FixUtil              Util;
typedef bdlt::FixUtilConfiguration Config;
typedef bsl::string_view           StrView;

const int k_DATE_MAX_PRECISION       = 3;
const int k_DATETZ_MAX_PRECISION     = 3;
const int k_DATETIME_MAX_PRECISION   = 6;
const int k_DATETIMETZ_MAX_PRECISION = 6;
const int k_TIME_MAX_PRECISION       = 6;

const int k_TIMETZ_MAX_PRECISION     = 0;  // ensures a fractional second is
                                           // never generated

// ============================================================================
//                             GLOBAL TEST DATA
// ----------------------------------------------------------------------------

// Define DEFAULT DATA generally usable across 'generate' and 'parse' test
// cases.

// *** 'Date' Data ***

struct DefaultDateDataRow {
    int         d_line;   // source line number
    int         d_year;   // year (of calendar date)
    int         d_month;  // month
    int         d_day;    // day
    const char *d_fix;    // FIX string
};

static
const DefaultDateDataRow DEFAULT_DATE_DATA[] =
{
    //LINE   YEAR   MONTH   DAY      FIX
    //----   ----   -----   ---    ----------
    { L_,       1,      1,    1,   "00010101" },
    { L_,       9,      9,    9,   "00090909" },
    { L_,      30,     10,   20,   "00301020" },
    { L_,     842,     12,   19,   "08421219" },
    { L_,    1847,      5,   19,   "18470519" },
    { L_,    2000,      2,   29,   "20000229" },
    { L_,    9999,     12,   31,   "99991231" },
};
const int NUM_DEFAULT_DATE_DATA =
        static_cast<int>(sizeof DEFAULT_DATE_DATA / sizeof *DEFAULT_DATE_DATA);

// *** 'Time' Data ***

struct DefaultTimeDataRow {
    int         d_line;  // source line number
    int         d_hour;  // hour (of day)
    int         d_min;   // minute
    int         d_sec;   // second
    int         d_msec;  // millisecond
    int         d_usec;  // microsecond
    const char *d_fix;   // FIX string
};

static
const DefaultTimeDataRow DEFAULT_TIME_DATA[] =
{
    //LINE   HOUR   MIN   SEC   MSEC   USEC           FIX
    //----   ----   ---   ---   ----   ----    -----------------
    { L_,       0,    0,    0,     0,     0,   "00:00:00.000000" },
    { L_,       1,    2,    3,     4,     5,   "01:02:03.004005" },
    { L_,      10,   20,   30,    40,    50,   "10:20:30.040050" },
    { L_,      19,   43,   27,   805,   107,   "19:43:27.805107" },
    { L_,      23,   59,   59,   999,   999,   "23:59:59.999999" },
    { L_,      24,    0,    0,     0,     0,   "00:00:00.000000" }
};
const int NUM_DEFAULT_TIME_DATA =
        static_cast<int>(sizeof DEFAULT_TIME_DATA / sizeof *DEFAULT_TIME_DATA);

// *** Zone Data ***

struct DefaultZoneDataRow {
    int         d_line;     // source line number
    int         d_offset;   // offset (in minutes) from UTC
    const char *d_fix;  // FIX string
};

static
const DefaultZoneDataRow DEFAULT_ZONE_DATA[] =
{
    //LINE   OFFSET     FIX
    //----   ------   --------
    { L_,     -1439,  "-23:59" },
    { L_,      -120,  "-02:00" },
    { L_,       -30,  "-00:30" },
    { L_,         0,  "+00:00" },
    { L_,        90,  "+01:30" },
    { L_,       240,  "+04:00" },
    { L_,      1439,  "+23:59" }
};
const int NUM_DEFAULT_ZONE_DATA =
        static_cast<int>(sizeof DEFAULT_ZONE_DATA / sizeof *DEFAULT_ZONE_DATA);

static
const DefaultZoneDataRow EXTENDED_ZONE_DATA[] =
{
    //LINE   OFFSET    FIX
    //----   ------   -----
    { L_,         0,  "Z"   },
    { L_,      -120,  "-02" },
    { L_,         0,  "-00" },
    { L_,         0,  "+00" },
    { L_,        60,  "+01" },
    { L_,       600,  "+10" }
};
const int NUM_EXTENDED_ZONE_DATA =
      static_cast<int>(sizeof EXTENDED_ZONE_DATA / sizeof *EXTENDED_ZONE_DATA);

// *** Configuration Data ***

struct DefaultCnfgDataRow {
    int  d_line;       // source line number
    int  d_precision;  // 'precision'                     "
    bool d_useZ;       // 'useZAbbreviationForUtc'        "
};

static
const DefaultCnfgDataRow DEFAULT_CNFG_DATA[] =
{
    //LINE   precision   use 'Z'
    //----   ---------   -------
    { L_,            3,    false },
    { L_,            3,     true },
    { L_,            6,    false },
    { L_,            6,     true },

    // additional configurations

    { L_,            0,    false },
    { L_,            1,    false },
    { L_,            2,    false },
    { L_,            4,    false },
    { L_,            5,    false }
};
const int NUM_DEFAULT_CNFG_DATA =
        static_cast<int>(sizeof DEFAULT_CNFG_DATA / sizeof *DEFAULT_CNFG_DATA);

// Define BAD (invalid) DATA generally usable across 'parse' test cases.

// *** Bad 'Date' Data ***

struct BadDateDataRow {
    int         d_line;     // source line number
    const char *d_invalid;  // test string
};

static
const BadDateDataRow BAD_DATE_DATA[] =
{
    //LINE  INPUT STRING
    //----  -------------------------
    { L_,   ""                      },  // length = 0

    { L_,   "0"                     },  // length = 1
    { L_,   "-"                     },
    { L_,   "+"                     },
    { L_,   "T"                     },
    { L_,   "Z"                     },
    { L_,   ":"                     },
    { L_,   " "                     },

    { L_,   "12"                    },  // length = 2
    { L_,   "3T"                    },
    { L_,   "4-"                    },
    { L_,   "x1"                    },
    { L_,   "T:"                    },
    { L_,   "+:"                    },

    { L_,   "999"                   },  // length = 3

    { L_,   "9999"                  },  // length = 4
    { L_,   "1-9-"                  },

    { L_,   "4-5-6"                 },  // length = 5
    { L_,   "+0130"                 },

    { L_,   "1-01-1"                },  // length = 6
    { L_,   "01-1-1"                },
    { L_,   "1-1-01"                },

    { L_,   "02-02-2"               },  // length = 7
    { L_,   "03-3-03"               },
    { L_,   "4-04-04"               },

    { L_,   "05-05-05"              },  // length = 8
    { L_,   "005-05-5"              },
    { L_,   "006-6-06"              },
    { L_,   "0006-6-6"              },

    { L_,   "0007-07-7"             },  // length = 9
    { L_,   "0008-8-08"             },
    { L_,   "009-09-09"             },

    { L_,   "0001 01-01"            },  // length = 10
    { L_,   "0001-01:01"            },
    { L_,   "0000-01-01"            },
    { L_,   "0001-00-01"            },
    { L_,   "0001-13-01"            },
    { L_,   "0001-01-00"            },
    { L_,   "0001-01-32"            },
    { L_,   "0001-04-31"            },
    { L_,   "1900-02-29"            },
    { L_,   "2000-02-30"            },

    { L_,   "0001-01-010"           },  // length = 11
    { L_,   "1970-12-310"           },
};
const int NUM_BAD_DATE_DATA =
                static_cast<int>(sizeof BAD_DATE_DATA / sizeof *BAD_DATE_DATA);

// *** Bad 'Time' Data ***

struct BadTimeDataRow {
    int         d_line;     // source line number
    const char *d_invalid;  // test string
};

static
const BadTimeDataRow BAD_TIME_DATA[] =
{
    //LINE  INPUT STRING
    //----  -------------------------
    { L_,   ""                       },  // length = 0

    { L_,   "0"                      },  // length = 1
    { L_,   "-"                      },
    { L_,   "+"                      },
    { L_,   "T"                      },
    { L_,   "Z"                      },
    { L_,   ":"                      },
    { L_,   "."                      },
    { L_,   ","                      },
    { L_,   " "                      },

    { L_,   "12"                     },  // length = 2
    { L_,   "3T"                     },
    { L_,   "4-"                     },
    { L_,   "x1"                     },
    { L_,   "T:"                     },
    { L_,   "+:"                     },
    { L_,   "24"                     },

    { L_,   "222"                    },  // length = 3
    { L_,   "000"                    },
    { L_,   "1:2"                    },

    { L_,   "1234"                   },  // length = 4
    { L_,   "1:19"                   },
    { L_,   "11:9"                   },

    { L_,   "12:60"                  },  // length = 5
    { L_,   "2:001"                  },
    { L_,   "23,01"                  },
    { L_,   "24:00"                  },
    { L_,   "24:01"                  },
    { L_,   "25:00"                  },
    { L_,   "99:00"                  },

    { L_,   "1:2:30"                 },  // length = 6
    { L_,   "1:20:3"                 },
    { L_,   "10:2:3"                 },
    { L_,   "1:2:3."                 },
    { L_,   "12:100"                 },
    { L_,   ":12:12"                 },

    { L_,   "12:00:1"                },  // length = 7
    { L_,   "12:0:01"                },
    { L_,   "2:10:01"                },
    { L_,   "24:00.1"                },

    { L_,   "12:2:001"               },  // length = 8
    { L_,   "3:02:001"               },
    { L_,   "3:2:0001"               },
    { L_,   "20:20,51"               },
    { L_,   "20:20:61"               },
    { L_,   "24:00:00"               },
    { L_,   "24:00:01"               },

    { L_,   "04:05:06."              },  // length = 9
    { L_,   "04:05:006"              },
    { L_,   "12:59:100"              },

    { L_,   "03:02:001."             },  // length = 10
    { L_,   "03:02:001,"             },
    { L_,   "03:2:001.1"             },
    { L_,   "24:00:00.1"             },

    { L_,   "24:00:00.01"            },  // length = 11
    { L_,   "03:02:001,9"            },

    { L_,   "23:00:00,000"           },  // length = 12
    { L_,   "24:00:00.000"           },
    { L_,   "24:00:00.001"           },
    { L_,   "24:00:00.999"           },
    { L_,   "25:00:00.000"           },
};
const int NUM_BAD_TIME_DATA =
                static_cast<int>(sizeof BAD_TIME_DATA / sizeof *BAD_TIME_DATA);

// *** Bad Zone Data ***

struct BadZoneDataRow {
    int         d_line;     // source line number
    const char *d_invalid;  // test string
};

static
const BadZoneDataRow BAD_ZONE_DATA[] =
{
    //LINE  INPUT STRING
    //----  -------------------------
    { L_,   "0"                      },  // length = 1
    { L_,   "+"                      },
    { L_,   "-"                      },
    { L_,   "T"                      },
    { L_,   "z"                      },

    { L_,   "+0"                     },  // length = 2
    { L_,   "-0"                     },
    { L_,   "Z0"                     },

    { L_,   "+24"                    },  // length = 3
    { L_,   "-24"                    },

    { L_,   "+10:"                   },  // length = 4
    { L_,   "-10:"                   },
    { L_,   "+120"                   },
    { L_,   "-030"                   },

    { L_,   "+01:1"                  },  // length = 5
    { L_,   "-01:1"                  },
    { L_,   "+1:12"                  },
    { L_,   "+12:1"                  },
    { L_,   "+0000"                  },
    { L_,   "-0000"                  },
    { L_,   "+2360"                  },
    { L_,   "-2360"                  },
    { L_,   "+2400"                  },
    { L_,   "-2400"                  },

    { L_,   "+12:1x"                 },  // length = 6
    { L_,   "+12:1 "                 },
    { L_,   "+1200x"                 },
    { L_,   "+23:60"                 },
    { L_,   "-23:60"                 },
    { L_,   "+24:00"                 },
    { L_,   "-24:00"                 },

    { L_,   "+123:23"                },  // length = 7
    { L_,   "+12:123"                },
    { L_,   "+011:23"                },
    { L_,   "+12:011"                },

    { L_,   "+123:123"               },  // length = 8
};
const int NUM_BAD_ZONE_DATA =
                static_cast<int>(sizeof BAD_ZONE_DATA / sizeof *BAD_ZONE_DATA);

//=============================================================================
//                  GLOBAL HELPER FUNCTIONS FOR TESTING
//-----------------------------------------------------------------------------

static
Config& gg(Config *object,
           int     fractionalSecondPrecision,
           bool    useZAbbreviationForUtcFlag)
    // Return, by reference, the specified '*object' with its value adjusted
    // according to the specified 'fractionalSecondPrecision' and
    // 'useZAbbreviationForUtcFlag'.
{
    if (fractionalSecondPrecision > 6) {
        fractionalSecondPrecision = 6;
    }

    object->setFractionalSecondPrecision(fractionalSecondPrecision);
    object->setUseZAbbreviationForUtc(useZAbbreviationForUtcFlag);

    return *object;
}

static
void updateExpectedPerConfig(bsl::string   *expected,
                             const Config&  configuration,
                             int            maxPrecision)
    // Update the specified 'expected' FIX string as if it were generated using
    // the specified 'configuration' with the precision limited by the
    // specified 'maxPrecision'.  The behavior is undefined unless the timezone
    // offset within 'expected' (if any) is of the form "(+|-)dd:dd".
{
    ASSERT(expected);

    const bsl::string::size_type index = expected->find('.');

    if (index != bsl::string::npos) {
        bsl::string::size_type length = 0;
        while (isdigit((*expected)[index + length + 1])) {
            ++length;
        }

        int precision = configuration.fractionalSecondPrecision();

        if (precision > maxPrecision) {
            precision = maxPrecision;
        }

        if (0 == precision) {
            expected->erase(index, length + 1);
        }
        else if (precision < static_cast<int>(length)) {
            expected->erase(index + precision + 1,
                            length - precision);
        }
    }

    // If there aren't enough characters in 'expected', don't bother with the
    // other configuration options.

    const int ZONELEN = static_cast<int>(sizeof "+dd:dd") - 1;

    if (expected->length() < ZONELEN
     || (!configuration.useZAbbreviationForUtc())) {
        return;                                                       // RETURN
    }

    // See if the tail of 'expected' has the pattern of a timezone offset.

    const bsl::string::size_type zdx = expected->length() - ZONELEN;

    if (('+' != (*expected)[zdx] && '-' != (*expected)[zdx])
      || !isdigit(static_cast<unsigned char>((*expected)[zdx + 1]))
      || !isdigit(static_cast<unsigned char>((*expected)[zdx + 2]))
      || ':' !=   (*expected)[zdx + 3]
      || !isdigit(static_cast<unsigned char>((*expected)[zdx + 4]))
      || !isdigit(static_cast<unsigned char>((*expected)[zdx + 5]))) {
        return;                                                       // RETURN
    }

    if (configuration.useZAbbreviationForUtc()) {
        const bsl::string zone = expected->substr(
                                                 expected->length() - ZONELEN);

        if (0 == zone.compare("+00:00")) {
            expected->erase(expected->length() - ZONELEN);
            expected->push_back('Z');

            return;                                                   // RETURN
        }
    }
}

static
bool containsOnlyDigits(const char *string)
    // Return 'true' if the specified 'string' contains nothing but digits, and
    // 'false' otherwise.
{
    while (*string) {
        if (!isdigit(*string)) {
            return false;                                             // RETURN
        }

        ++string;
    }

    return true;
}

//=============================================================================
//                              FUZZ TESTING
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// The following function, 'LLVMFuzzerTestOneInput', is the entry point for the
// clang fuzz testing facility.  See {http://bburl/BDEFuzzTesting} for details
// on how to build and run with fuzz testing enabled.
//-----------------------------------------------------------------------------

#ifdef BDE_ACTIVATE_FUZZ_TESTING
#define main test_driver_main
#endif

extern "C"
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
    // Use the specified 'data' array of 'size' bytes as input to methods of
    // this component and return zero.
{
    const char *FUZZ   = reinterpret_cast<const char *>(data);
    int         LENGTH = static_cast<int>(size);
    int         test   = 0;

    if (LENGTH > 0) {
        // Use first fuzz byte to select the test case.
        test = (*FUZZ++ & 0xFF) % 100;
        --LENGTH;
    }

    switch (test) { case 0:  // Zero is always the leading case.
      case 9: {
        // --------------------------------------------------------------------
        // PARSE: DATETIME & DATETIMETZ
        //
        // Plan:
        //   Parse a 'Datetime' object and a 'DatetimeTz' object from the fuzz
        //   data directly and from the fuzz represented as a string reference.
        //   The correctness of the parsing is not verified.
        //
        // Testing:
        //   int parse(Datetime *, const char *, int);
        //   int parse(DatetimeTz *, const char *, int);
        //   int parse(Datetime *result, const StringRef& string);
        //   int parse(DatetimeTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        bdlt::Datetime mX;
        Util::parse(&mX, FUZZ, LENGTH);
        Util::parse(&mX, StrView(FUZZ, LENGTH));

        bdlt::DatetimeTz mXZ;
        Util::parse(&mXZ, FUZZ, LENGTH);
        Util::parse(&mXZ, StrView(FUZZ, LENGTH));
      } break;
      case 8: {
        // --------------------------------------------------------------------
        // PARSE: TIME & TIMETZ
        //
        // Plan:
        //   Parse a 'Time' object and a 'TimeTz' object from the fuzz data
        //   directly and from the fuzz represented as a string reference.  The
        //   correctness of the parsing is not verified.
        //
        // Testing:
        //   int parse(Time *, const char *, int);
        //   int parse(TimeTz *, const char *, int);
        //   int parse(Time *result, const StringRef& string);
        //   int parse(TimeTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        bdlt::Time mX;
        Util::parse(&mX, FUZZ, LENGTH);
        Util::parse(&mX, StrView(FUZZ, LENGTH));

        bdlt::TimeTz mXZ;
        Util::parse(&mXZ, FUZZ, LENGTH);
        Util::parse(&mXZ, StrView(FUZZ, LENGTH));
      } break;
      case 7: {
        // --------------------------------------------------------------------
        // PARSE: DATE & DATETZ
        //
        // Plan:
        //   Parse a 'Date' object and a 'DateTz' object from the fuzz data
        //   directly and from the fuzz represented as a string reference.  The
        //   correctness of the parsing is not verified.
        //
        // Testing:
        //   int parse(Date *, const char *, int);
        //   int parse(DateTz *, const char *, int);
        //   int parse(Date *result, const StringRef& string);
        //   int parse(DateTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        bdlt::Date mX;
        Util::parse(&mX, FUZZ, LENGTH);
        Util::parse(&mX, StrView(FUZZ, LENGTH));

        bdlt::DateTz mXZ;
        Util::parse(&mXZ, FUZZ, LENGTH);
        Util::parse(&mXZ, StrView(FUZZ, LENGTH));
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // GENERATE 'DatetimeTz'
        //
        // Plan:
        //   Create a 'TestInStream' using the fuzz data as a source, create a
        //   configuration object using the first byte of the stream, attempt
        //   to stream in a 'DatetimeTz' object from the stream, and if
        //   successful, call a variety of 'generate' methods on the object.
        //   The correctness of the generation is not verified.
        //
        // Testing:
        //   int generate(char *, int, const DatetimeTz&);
        //   int generate(char *, int, const DatetimeTz&, const Config&);
        //   int generate(string *, const DatetimeTz&);
        //   int generate(string *, const DatetimeTz&, const Config&);
        //   ostream generate(ostream&, const DatetimeTz&);
        //   ostream generate(ostream&, const DatetimeTz&, const Config&);
        //   int generateRaw(char *, const DatetimeTz&);
        //   int generateRaw(char *, const DatetimeTz&, const Config&);
        // --------------------------------------------------------------------

        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::DatetimeTz        mXZ;
        const bdlt::DatetimeTz& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
            Util::generate(&ss, XZ, CF);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
            Util::generate(&sp, XZ, CF);
#endif
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      case 5: {
        // GENERATE 'TimeTz'
        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::TimeTz        mXZ;
        const bdlt::TimeTz& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
            Util::generate(&ss, XZ, CF);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
            Util::generate(&sp, XZ, CF);
#endif
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // GENERATE 'DateTz'
        //
        // Plan:
        //   Create a 'TestInStream' using the fuzz data as a source, create a
        //   configuration object using the first byte of the stream, attempt
        //   to stream in a 'DateTz' object from the stream, and if successful,
        //   call a variety of 'generate' methods on the object.  The
        //   correctness of the generation is not verified.
        //
        // Testing:
        //   int generate(char *, int, const DateTz&);
        //   int generate(char *, int, const DateTz&, const Config&);
        //   int generate(string *, const DateTz&);
        //   int generate(string *, const DateTz&, const Config&);
        //   ostream generate(ostream&, const DateTz&);
        //   ostream generate(ostream&, const DateTz&, const Config&);
        //   int generateRaw(char *, const DateTz&);
        //   int generateRaw(char *, const DateTz&, const Config&);
        // --------------------------------------------------------------------

        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::DateTz        mXZ;
        const bdlt::DateTz& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
            Util::generate(&ss, XZ, CF);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
            Util::generate(&sp, XZ, CF);
#endif
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // GENERATE 'Datetime'
        //
        // Plan:
        //   Create a 'TestInStream' using the fuzz data as a source, create a
        //   configuration object using the first byte of the stream, attempt
        //   to stream in a 'Datetime' object from the stream, and if
        //   successful, call a variety of 'generate' methods on the object.
        //   The correctness of the generation is not verified.
        //
        // Testing:
        //   int generate(char *, int, const Datetime&);
        //   int generate(char *, int, const Datetime&, const Config&);
        //   int generate(string *, const Datetime&);
        //   int generate(string *, const Datetime&, const Config&);
        //   ostream generate(ostream&, const Datetime&);
        //   ostream generate(ostream&, const Datetime&, const Config&);
        //   int generateRaw(char *, const Datetime&);
        //   int generateRaw(char *, const Datetime&, const Config&);
        // --------------------------------------------------------------------

        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::Datetime        mXZ;
        const bdlt::Datetime& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
            Util::generate(&ss, XZ, CF);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
            Util::generate(&sp, XZ, CF);
#endif
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // GENERATE 'Time'
        //
        // Plan:
        //   Create a 'TestInStream' using the fuzz data as a source, create a
        //   configuration object using the first byte of the stream, attempt
        //   to stream in a 'Time' object from the stream, and if successful
        //   call a variety of 'generate' methods on the object.  The
        //   correctness of the generation is not verified.
        //
        // Testing:
        //   int generate(char *, int, const Time&);
        //   int generate(char *, int, const Time&, const Config&);
        //   int generate(string *, const Time&);
        //   int generate(string *, const Time&, const Config&);
        //   ostream generate(ostream&, const Time&);
        //   ostream generate(ostream&, const Time&, const Config&);
        //   int generateRaw(char *, const Time&);
        //   int generateRaw(char *, const Time&, const Config&);
        // --------------------------------------------------------------------

        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::Time        mXZ;
        const bdlt::Time& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ, CF);
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
#endif
            Util::generate(&ss, XZ, CF);
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // GENERATE 'Date'
        //
        // Plan:
        //   Create a 'TestInStream' using the fuzz data as a source, create a
        //   configuration object using the first byte of the stream, attempt
        //   to stream in a 'Date' object from the stream, and if successful,
        //   call a variety of 'generate' methods on the object.  The
        //   correctness of the generation is not verified.
        //
        // Testing:
        //   int generate(char *, int, const Date&);
        //   int generate(char *, int, const Date&, const Config&);
        //   int generate(string *, const Date&);
        //   int generate(string *, const Date&, const Config&);
        //   ostream generate(ostream&, const Date&);
        //   ostream generate(ostream&, const Date&, const Config&);
        //   int generateRaw(char *, const Date&);
        //   int generateRaw(char *, const Date&, const Config&);
        // --------------------------------------------------------------------

        bslx::TestInStream in(FUZZ, LENGTH);
        in.setQuiet(true);

        Config        mC;       // default configuartion
        const Config& C = mC;
        Config        mCF;      // fuzzed configuration
        const Config& CF = mCF;

        unsigned char f = 0;
        in.getUint8(f);
        mCF.setUseZAbbreviationForUtc(0 != (f & 1));
        mCF.setFractionalSecondPrecision((f / 2) % 7);

        bdlt::Date        mXZ;
        const bdlt::Date& XZ = mXZ;

        if (mXZ.bdexStreamIn(in, mXZ.maxSupportedBdexVersion(20200917))) {
            char              buffer[Util::k_MAX_STRLEN + 1];
            bsl::string       s;
            std::string       ss;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            std::pmr::string  sp;
#endif
            bsl::stringstream out;

            Util::generateRaw(buffer, XZ);
            Util::generateRaw(buffer, XZ, C);
            Util::generateRaw(buffer, XZ, CF);
            Util::generate(&s,  XZ);
            Util::generate(&s,  XZ, C);
            Util::generate(&s,  XZ, CF);
            Util::generate(&ss, XZ);
            Util::generate(&ss, XZ, C);
            Util::generate(&ss, XZ, CF);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
            Util::generate(&sp, XZ);
            Util::generate(&sp, XZ, C);
            Util::generate(&sp, XZ, CF);
#endif
            Util::generate(out, XZ);
            Util::generate(out, XZ, C);
            Util::generate(out, XZ, CF);

            for (int i = 0; i <= Util::k_MAX_STRLEN + 1; ++i) {
                Util::generate(buffer, i, XZ);
                Util::generate(buffer, i, XZ, C);
                Util::generate(buffer, i, XZ, CF);
            }
        }
      } break;
      default: {
      } break;
    }

    if (testStatus > 0) {
        BSLS_ASSERT_INVOKE("FUZZ TEST FAILURES");
    }

    return 0;
}

//=============================================================================
//                              MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const int                 test = argc > 1 ? atoi(argv[1]) : 0;
    const bool             verbose = argc > 2;
    const bool         veryVerbose = argc > 3;
    const bool     veryVeryVerbose = argc > 4;
    const bool veryVeryVeryVerbose = argc > 5;

    (void)veryVeryVerbose;  // eliminate unused variable warning
    (void)veryVeryVeryVerbose;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    // CONCERN: 'BSLS_REVIEW' failures should lead to test failures.
    bsls::ReviewFailureHandlerGuard reviewGuard(&bsls::Review::failByAbort);

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
    if (verbose) cout << "std::pmr::string supported\n";
#endif

    switch (test) { case 0:  // Zero is always the leading case.
      case 10: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //   Extracted from component header file.
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

        if (verbose) cout << endl
                          << "USAGE EXAMPLE" << endl
                          << "=============" << endl;

///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: Basic 'bdlt::FixUtil' Usage
/// - - - - - - - - - - - - - - - - - - -
// This example demonstrates basic use of one 'generate' function and two
// 'parse' functions.
//
// First, we construct a few objects that are prerequisites for this and the
// following example:
//..
    const bdlt::Date date(2005, 1, 31);     // 2005/01/31
    const bdlt::Time time(8, 59, 59, 123);  // 08:59:59.123
    const int        tzOffset = 240;        // +04:00 (four hours west of UTC)
//..
// Then, we construct a 'bdlt::DatetimeTz' object for which a corresponding
// FIX-compliant string will be generated shortly:
//..
    const bdlt::DatetimeTz sourceDatetimeTz(bdlt::Datetime(date, time),
                                            tzOffset);
//..
// For comparison with the FIX string generated below, note that streaming the
// value of 'sourceDatetimeTz' to 'stdout':
//..
if (veryVerbose)
    bsl::cout << sourceDatetimeTz << bsl::endl;
//..
// produces:
//..
//  31JAN2005_08:59:59.123000+0400
//..
// Next, we use a 'generate' function to produce a FIX-compliant string for
// 'sourceDatetimeTz', writing the output to a 'bsl::ostringstream', and assert
// that both the return value and the string that is produced are as expected:
//..
    bsl::ostringstream  oss;
    const bsl::ostream& ret = bdlt::FixUtil::generate(oss, sourceDatetimeTz);
    ASSERT(&oss == &ret);

    const bsl::string fix = oss.str();
    ASSERT(fix == "20050131-08:59:59.123+04:00");
//..
// For comparison, see the output that was produced by the streaming operator
// above.
//
// Now, we parse the string that was just produced, loading the result of the
// parse into a second 'bdlt::DatetimeTz' object, and assert that the parse was
// successful and that the target object has the same value as that of the
// original (i.e., 'sourceDatetimeTz'):
//..
    bdlt::DatetimeTz targetDatetimeTz;

    int rc = bdlt::FixUtil::parse(&targetDatetimeTz,
                                  fix.c_str(),
                                  static_cast<int>(fix.length()));
    ASSERT(               0 == rc);
    ASSERT(sourceDatetimeTz == targetDatetimeTz);
//..
// Finally, we parse the 'fix' string a second time, this time loading the
// result into a 'bdlt::Datetime' object (instead of a 'bdlt::DatetimeTz'):
//..
    bdlt::Datetime targetDatetime;

    rc = bdlt::FixUtil::parse(&targetDatetime,
                              fix.c_str(),
                              static_cast<int>(fix.length()));
    ASSERT(                             0 == rc);
    ASSERT(sourceDatetimeTz.utcDatetime() == targetDatetime);
//..
// Note that this time the value of the target object has been converted to
// UTC.
//
///Example 2: Configuring FIX String Generation
///- - - - - - - - - - - - - - - - - - - - - -
// This example demonstrates use of a 'bdlt::FixUtilConfiguration' object to
// influence the format of the FIX strings that are generated by this component
// by passing that configuration object to 'generate'.  We also take this
// opportunity to illustrate the flavor of the 'generate' functions that
// outputs to a 'char *' buffer of a specified length.
//
// First, we construct the 'bdlt::FixUtilConfiguration' object that indicates
// how we would like to affect the generated output FIX string.  In this case,
// we want to have microsecond precision displayed:
//..
    bdlt::FixUtilConfiguration configuration;

    configuration.setFractionalSecondPrecision(6);
//..
// Then, we define the 'char *' buffer that will be used to stored the
// generated string.  A buffer of size 'bdlt::FixUtil::k_DATETIMETZ_STRLEN + 1'
// is large enough to hold any string generated by this component for a
// 'bdlt::DatetimeTz' object, including a null terminator:
//..
    const int BUFLEN = bdlt::FixUtil::k_DATETIMETZ_STRLEN + 1;
    char      buffer[BUFLEN];
//..
// Next, we use a 'generate' function that accepts our 'configuration' to
// produce a FIX-compliant string for 'sourceDatetimeTz', this time writing the
// output to a 'char *' buffer, and assert that both the return value and the
// string that is produced are as expected.  Note that in comparing the return
// value against 'BUFLEN - 1' we account for the fact that, although a null
// terminator was generated, it is not included in the character count returned
// by 'generate'.  Also note that we use 'bsl::strcmp' to compare the resulting
// string knowing that we supplied a buffer having sufficient capacity to
// accommodate a null terminator:
//..
    rc = bdlt::FixUtil::generate(buffer,
                                 BUFLEN,
                                 sourceDatetimeTz,
                                 configuration);
    ASSERT(BUFLEN - 1 == rc);
    ASSERT(         0 == bsl::strcmp(buffer,
                                     "20050131-08:59:59.123000+04:00"));
//..
// For comparison, see the output that was produced by the streaming operator
// above.
//
// Next, we parse the string that was just produced, loading the result of the
// parse into a second 'bdlt::DatetimeTz' object, and assert that the parse was
// successful and that the target object has the same value as that of the
// original (i.e., 'sourceDatetimeTz').  Note that 'BUFLEN - 1' is passed and
// *not* 'BUFLEN' because the former indicates the correct number of characters
// in 'buffer' that we wish to parse:
//..
    rc = bdlt::FixUtil::parse(&targetDatetimeTz, buffer, BUFLEN - 1);

    ASSERT(               0 == rc);
    ASSERT(sourceDatetimeTz == targetDatetimeTz);
//..
// Then, we parse the string in 'buffer' a second time, this time loading the
// result into a 'bdlt::Datetime' object (instead of a 'bdlt::DatetimeTz'):
//..
    rc = bdlt::FixUtil::parse(&targetDatetime, buffer, BUFLEN - 1);

    ASSERT(                             0 == rc);
    ASSERT(sourceDatetimeTz.utcDatetime() == targetDatetime);
//..
// Note that this time the value of the target object has been converted to
// UTC.
//
// Finally, we modify the 'configuration' to display the 'bdlt::DatetimeTz'
// without fractional seconds:
//..
    configuration.setFractionalSecondPrecision(0);
    rc = bdlt::FixUtil::generate(buffer,
                                 BUFLEN,
                                 sourceDatetimeTz,
                                 configuration);
    ASSERT(BUFLEN - 8 == rc);
    ASSERT(         0 == bsl::strcmp(buffer, "20050131-08:59:59+04:00"));
//..
      } break;
      case 9: {
        // --------------------------------------------------------------------
        // PARSE: DATETIME & DATETIMETZ
        //
        // Concerns:
        //: 1 All FIX string representations supported by this component
        //:   (as documented in the header file) for 'Datetime' and
        //:   'DatetimeTz' values are parsed successfully.
        //:
        //: 2 If parsing succeeds, the result 'Datetime' or 'DatetimeTz' object
        //:   has the expected value.
        //:
        //: 3 If the optional timezone offset is present in the input string
        //:   when parsing into a 'Datetime' object, the resulting value is
        //:   converted to the equivalent UTC datetime.
        //:
        //: 4 If the optional timezone offset is *not* present in the input
        //:   string when parsing into a 'DatetimeTz' object, it is assumed to
        //:   be UTC.
        //:
        //: 5 If parsing succeeds, 0 is returned.
        //:
        //: 6 All strings that are not FIX representations supported by
        //:   this component for 'Datetime' and 'DatetimeTz' values are
        //:   rejected (i.e., parsing fails).
        //:
        //: 7 If parsing fails, the result object is unaffected and a non-zero
        //:   value is returned.
        //:
        //: 8 The entire extent of the input string is parsed.
        //:
        //: 9 Leap seconds, fractional seconds containing more than three
        //:   digits, and extremal values (those that can overflow a
        //:   'Datetime') are handled correctly.
        //:
        //:10 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values ('D'), 'Time' values ('T'), timezone offsets ('Z'),
        //:   and configurations ('C').
        //:
        //: 2 Apply the (fully-tested) 'generateRaw' functions to each element
        //:   in the cross product, 'D x T x Z x C', of the test data from P-1.
        //:
        //: 3 Invoke the 'parse' functions on the strings generated in P-2 and
        //:   verify that parsing succeeds, i.e., that 0 is returned and the
        //:   result objects have the expected values.  (C-1..5)
        //:
        //: 4 Using the table-driven technique, specify a set of distinct
        //:   strings that are not FIX representations supported by this
        //:   component for 'Datetime' and 'DatetimeTz' values.
        //:
        //: 5 Invoke the 'parse' functions on the strings from P-4 and verify
        //:   that parsing fails, i.e., that a non-zero value is returned and
        //:   the result objects are unchanged.  (C-6..8)
        //:
        //: 6 Using the table-driven technique, specify a set of distinct FIX
        //:   strings that specifically cover cases involving leap
        //:   seconds, fractional seconds containing more than three digits,
        //:   and extremal values.
        //:
        //: 7 Invoke the 'parse' functions on the strings from P-6 and verify
        //:   the results are as expected.  (C-9)
        //:
        //: 8 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-10)
        //
        // Testing:
        //   int parse(Datetime *, const char *, int);
        //   int parse(DatetimeTz *, const char *, int);
        //   int parse(Datetime *result, const StringRef& string);
        //   int parse(DatetimeTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "PARSE: DATETIME & DATETIMETZ" << endl
                          << "============================" << endl;

        char buffer[Util::k_MAX_STRLEN];

        const bdlt::Date       DD(246, 8, 10);
        const bdlt::Time       TT(2, 4, 6, 8);

        const bdlt::Datetime   XX(DD, TT);  // 'XX' and 'ZZ' are controls,
        const bdlt::DatetimeTz ZZ(XX, -7);  // distinct from any test data

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_EXT_ZONE_DATA =  NUM_EXTENDED_ZONE_DATA;
        const DefaultZoneDataRow (&EXT_ZONE_DATA)[NUM_EXT_ZONE_DATA] =
                                                            EXTENDED_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        if (verbose) cout << "\nValid FIX strings." << endl;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int ILINE = DATE_DATA[ti].d_line;
            const int YEAR  = DATE_DATA[ti].d_year;
            const int MONTH = DATE_DATA[ti].d_month;
            const int DAY   = DATE_DATA[ti].d_day;

            const bdlt::Date DATE(YEAR, MONTH, DAY);

            for (int tj = 0; tj < NUM_TIME_DATA; ++tj) {
                const int JLINE = TIME_DATA[tj].d_line;
                const int HOUR  = TIME_DATA[tj].d_hour;
                const int MIN   = TIME_DATA[tj].d_min;
                const int SEC   = TIME_DATA[tj].d_sec;
                const int MSEC  = TIME_DATA[tj].d_msec;
                const int USEC  = TIME_DATA[tj].d_usec;

                for (int tk = 0; tk < NUM_ZONE_DATA; ++tk) {
                    const int KLINE  = ZONE_DATA[tk].d_line;
                    const int OFFSET = ZONE_DATA[tk].d_offset;

                    if (24 == HOUR) {
                        continue;  // skip invalid compositions
                    }

                    for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                        const int  CLINE     = CNFG_DATA[tc].d_line;
                        const int  PRECISION = CNFG_DATA[tc].d_precision;
                        const bool USEZ      = CNFG_DATA[tc].d_useZ;

                        int expMsec = MSEC;
                        int expUsec = USEC;
                        {
                            // adjust the expected milliseconds to account for
                            // PRECISION truncating the value generated

                            int precision = (PRECISION < 3 ? PRECISION : 3);

                            for (int i = 3; i > precision; --i) {
                                expMsec /= 10;
                            }

                            for (int i = 3; i > precision; --i) {
                                expMsec *= 10;
                            }

                            // adjust the expected microseconds to account for
                            // PRECISION truncating the value generated

                            precision = (PRECISION > 3 ? PRECISION - 3: 0);

                            for (int i = 3; i > precision; --i) {
                                expUsec /= 10;
                            }

                            for (int i = 3; i > precision; --i) {
                                expUsec *= 10;
                            }
                        }

                        const bdlt::Datetime   DATETIME(YEAR,
                                                        MONTH,
                                                        DAY,
                                                        HOUR,
                                                        MIN,
                                                        SEC,
                                                        expMsec,
                                                        expUsec);
                        const bdlt::DatetimeTz DATETIMETZ(DATETIME, OFFSET);

                        if (veryVerbose) {
                            if (0 == tc) {
                                T_ P_(ILINE) P_(JLINE) P_(KLINE)
                                                    P_(DATETIME) P(DATETIMETZ);
                            }
                            T_ P_(CLINE) P_(PRECISION) P(USEZ);
                        }

                        Config mC;  const Config& C = mC;
                        gg(&mC, PRECISION, USEZ);

                        // without timezone offset in parsed string
                        {
                            const int LENGTH = Util::generateRaw(buffer,
                                                                 DATETIME,
                                                                 C);

                            if (veryVerbose) {
                                const bsl::string STRING(buffer, LENGTH);
                                T_ T_ P(STRING)
                            }

                                  bdlt::Datetime    mX(XX);
                            const bdlt::Datetime&   X = mX;

                                  bdlt::DatetimeTz  mZ(ZZ);
                            const bdlt::DatetimeTz& Z = mZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIME == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIME == Z.localDatetime());
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                           0 == Z.offset());

                            mX = XX;
                            mZ = ZZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIME == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIME == Z.localDatetime());
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                           0 == Z.offset());
                        }

                        // with timezone offset in parsed string
                        {
                            if ((DATE == bdlt::Date() && OFFSET > 0)
                             || (DATE == bdlt::Date(9999, 12, 31)
                              && OFFSET < 0)) {
                                continue;  // skip invalid compositions
                            }

                            const int LENGTH = Util::generateRaw(buffer,
                                                                 DATETIMETZ,
                                                                 C);

                            if (veryVerbose) {
                                const bsl::string STRING(buffer, LENGTH);
                                T_ T_ P(STRING)
                            }

                                  bdlt::Datetime    mX(XX);
                            const bdlt::Datetime&   X = mX;

                                  bdlt::DatetimeTz  mZ(ZZ);
                            const bdlt::DatetimeTz& Z = mZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ.utcDatetime() == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ               == Z);

                            mX = XX;
                            mZ = ZZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ.utcDatetime() == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ               == Z);
                        }
                    }  // loop over 'CNFG_DATA'
                }  // loop over 'ZONE_DATA'

                for (int tk = 0; tk < NUM_EXT_ZONE_DATA; ++tk) {
                    const int KLINE  = EXT_ZONE_DATA[tk].d_line;
                    const int OFFSET = EXT_ZONE_DATA[tk].d_offset;

                    if (24 == HOUR) {
                        continue;  // skip invalid compositions
                    }

                    for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                        const int  CLINE     = CNFG_DATA[tc].d_line;
                        const int  PRECISION = CNFG_DATA[tc].d_precision;
                        const bool USEZ      = CNFG_DATA[tc].d_useZ;

                        int expMsec = MSEC;
                        int expUsec = USEC;
                        {
                            // adjust the expected milliseconds to account for
                            // PRECISION truncating the value generated

                            int precision = (PRECISION < 3 ? PRECISION : 3);

                            for (int i = 3; i > precision; --i) {
                                expMsec /= 10;
                            }

                            for (int i = 3; i > precision; --i) {
                                expMsec *= 10;
                            }

                            // adjust the expected microseconds to account for
                            // PRECISION truncating the value generated

                            precision = (PRECISION > 3 ? PRECISION - 3: 0);

                            for (int i = 3; i > precision; --i) {
                                expUsec /= 10;
                            }

                            for (int i = 3; i > precision; --i) {
                                expUsec *= 10;
                            }
                        }

                        const bdlt::Datetime   DATETIME(YEAR,
                                                        MONTH,
                                                        DAY,
                                                        HOUR,
                                                        MIN,
                                                        SEC,
                                                        expMsec,
                                                        expUsec);
                        const bdlt::DatetimeTz DATETIMETZ(DATETIME, OFFSET);

                        if (veryVerbose) {
                            if (0 == tc) {
                                T_ P_(ILINE) P_(JLINE) P_(KLINE)
                                                    P_(DATETIME) P(DATETIMETZ);
                            }
                            T_ P_(CLINE) P_(PRECISION) P(USEZ);
                        }

                        Config mC;  const Config& C = mC;
                        gg(&mC, PRECISION, USEZ);

                        // with timezone offset in parsed string
                        {
                            if ((DATE == bdlt::Date() && OFFSET > 0)
                             || (DATE == bdlt::Date(9999, 12, 31)
                              && OFFSET < 0)) {
                                continue;  // skip invalid compositions
                            }

                            const int LENGTH = Util::generateRaw(buffer,
                                                                 DATETIMETZ,
                                                                 C);

                            if (veryVerbose) {
                                const bsl::string STRING(buffer, LENGTH);
                                T_ T_ P(STRING)
                            }

                                  bdlt::Datetime    mX(XX);
                            const bdlt::Datetime&   X = mX;

                                  bdlt::DatetimeTz  mZ(ZZ);
                            const bdlt::DatetimeTz& Z = mZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ.utcDatetime() == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ, buffer, LENGTH));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ               == Z);

                            mX = XX;
                            mZ = ZZ;

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mX,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ.utcDatetime() == X);

                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    0 == Util::parse(&mZ,
                                                     StrView(buffer, LENGTH)));
                            ASSERTV(ILINE, JLINE, KLINE, CLINE,
                                    DATETIMETZ               == Z);
                        }
                    }  // loop over 'CNFG_DATA'
                }  // loop over 'ZONE_DATA'
            }  // loop over 'TIME_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nInvalid strings." << endl;
        {
            bdlt::Datetime   mX(XX);  const bdlt::Datetime&   X = mX;
            bdlt::DatetimeTz mZ(ZZ);  const bdlt::DatetimeTz& Z = mZ;

            const int              NUM_DATE_DATA =         NUM_BAD_DATE_DATA;
            const BadDateDataRow (&DATE_DATA)[NUM_DATE_DATA] = BAD_DATE_DATA;

            for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
                const int LINE = DATE_DATA[ti].d_line;

                bsl::string bad(DATE_DATA[ti].d_invalid);

                // Append a valid time.

                bad.append("T12:26:52.726");

                const char *STRING = bad.c_str();
                const int   LENGTH = static_cast<int>(bad.length());

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }

            const int              NUM_TIME_DATA =         NUM_BAD_TIME_DATA;
            const BadTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] = BAD_TIME_DATA;

            for (int tj = 0; tj < NUM_TIME_DATA; ++tj) {
                const int LINE = TIME_DATA[tj].d_line;

                // Initialize with a *valid* date string, then append an
                // invalid time.

                bsl::string bad("20100817");

                // Ensure that 'bad' is initially valid.

                static bool firstFlag = true;
                if (firstFlag) {
                    const char *STRING = bad.data();
                    const int   LENGTH = static_cast<int>(bad.length());

                    bdlt::Date mD(DD);  const bdlt::Date& D = mD;

                    ASSERT( 0 == Util::parse(&mD, STRING, LENGTH));
                    ASSERT(DD != D);

                    mD = DD;

                    ASSERT( 0 == Util::parse(&mD, StrView(STRING, LENGTH)));
                    ASSERT(DD != D);
                }

                bad.append("T");
                bad.append(TIME_DATA[tj].d_invalid);

                const char *STRING = bad.c_str();
                const int   LENGTH = static_cast<int>(bad.length());

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }

            const int              NUM_ZONE_DATA =         NUM_BAD_ZONE_DATA;
            const BadZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] = BAD_ZONE_DATA;

            for (int tk = 0; tk < NUM_ZONE_DATA; ++tk) {
                const int LINE = ZONE_DATA[tk].d_line;

                // Initialize with a *valid* datetime string, then append an
                // invalid timezone offset.

                bsl::string bad("20100817-12:26:52.726");

                // Ensure that 'bad' is initially valid.

                static bool firstFlag = true;
                if (firstFlag) {
                    const char *STRING = bad.data();
                    const int   LENGTH = static_cast<int>(bad.length());

                    bdlt::Datetime mD(XX);  const bdlt::Datetime& D = mD;

                    ASSERT( 0 == Util::parse(&mD, STRING, LENGTH));
                    ASSERT(XX != D);

                    mD = XX;

                    ASSERT( 0 == Util::parse(&mD, StrView(STRING, LENGTH)));
                    ASSERT(XX != D);
                }

                // If 'ZONE_DATA[tk].d_invalid' contains nothing but digits,
                // appending it to 'bad' simply extends the fractional second
                // (so 'bad' remains valid).

                if (containsOnlyDigits(ZONE_DATA[tk].d_invalid)) {
                    continue;
                }

                bad.append(ZONE_DATA[tk].d_invalid);

                const char *STRING = bad.c_str();
                const int   LENGTH = static_cast<int>(bad.length());

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }
        }

        if (verbose) {
            cout << "\nTesting optional, leap, and fractional seconds."
                 << endl;
        }
        {
            const struct {
                int         d_line;
                const char *d_input_p;
                int         d_year;
                int         d_month;
                int         d_day;
                int         d_hour;
                int         d_min;
                int         d_sec;
                int         d_msec;
                int         d_usec;
                int         d_offset;
            } DATA[] = {
                // optional seconds
                { L_, "00010101-00:00",
                                     0001, 01, 01, 00, 00, 00, 000, 000,   0 },
                { L_, "00010101-00:01",
                                     0001, 01, 01, 00, 01, 00, 000, 000,   0 },
                { L_, "00010101-01:00",
                                     0001, 01, 01, 01, 00, 00, 000, 000,   0 },
                { L_, "00010101-01:01",
                                     0001, 01, 01, 01, 01, 00, 000, 000,   0 },

                // leap seconds
                { L_, "00010101-00:00:60.000",
                                     0001, 01, 01, 00, 01, 00, 000, 000,   0 },
                { L_, "99981231-23:59:60.999",
                                     9999, 01, 01, 00, 00, 00, 999, 000,   0 },

                // fractional seconds
                { L_, "00010101-00:00:00.0000001",
                                     0001, 01, 01, 00, 00, 00, 000, 000,   0 },
                { L_, "00010101-00:00:00.0000009",
                                     0001, 01, 01, 00, 00, 00, 000,   1,   0 },
                { L_, "00010101-00:00:00.00000001",
                                     0001, 01, 01, 00, 00, 00, 000, 000,   0 },
                { L_, "00010101-00:00:00.00000049",
                                     0001, 01, 01, 00, 00, 00, 000, 000,   0 },
                { L_, "00010101-00:00:00.00000050",
                                     0001, 01, 01, 00, 00, 00, 000,   1,   0 },
                { L_, "00010101-00:00:00.00000099",
                                     0001, 01, 01, 00, 00, 00, 000,   1,   0 },
                { L_, "00010101-00:00:00.0001",
                                     0001, 01, 01, 00, 00, 00, 000, 100,   0 },
                { L_, "00010101-00:00:00.0009",
                                     0001, 01, 01, 00, 00, 00, 000, 900,   0 },
                { L_, "00010101-00:00:00.00001",
                                     0001, 01, 01, 00, 00, 00, 000,  10,   0 },
                { L_, "00010101-00:00:00.00049",
                                     0001, 01, 01, 00, 00, 00, 000, 490,   0 },
                { L_, "00010101-00:00:00.00050",
                                     0001, 01, 01, 00, 00, 00, 000, 500,   0 },
                { L_, "00010101-00:00:00.00099",
                                     0001, 01, 01, 00, 00, 00, 000, 990,   0 },
                { L_, "00010101-00:00:00.9994" ,
                                     0001, 01, 01, 00, 00, 00, 999, 400,   0 },
                { L_, "00010101-00:00:00.9995" ,
                                     0001, 01, 01, 00, 00, 00, 999, 500,   0 },
                { L_, "00010101-00:00:00.9999" ,
                                     0001, 01, 01, 00, 00, 00, 999, 900,   0 },
                { L_, "99981231-23:59:60.9999" ,
                                     9999, 01, 01, 00, 00, 00, 999, 900,   0 },
                { L_, "00010101-00:00:00.9999994" ,
                                     0001, 01, 01, 00, 00, 00, 999, 999,   0 },
                { L_, "00010101-00:00:00.9999995" ,
                                     0001, 01, 01, 00, 00, 01, 000, 000,   0 },
                { L_, "00010101-00:00:00.9999999" ,
                                     0001, 01, 01, 00, 00, 01, 000, 000,   0 },
                { L_, "99981231-23:59:60.9999999" ,
                                     9999, 01, 01, 00, 00, 01, 000, 000,   0 },

                // omit fractional seconds
                { L_, "00010101-00:00:60",
                                     0001, 01, 01, 00, 01, 00, 000, 000,   0 },
                { L_, "20141223-12:34:45",
                                     2014, 12, 23, 12, 34, 45, 000, 000,   0 },
                { L_, "20141223-12:34:45Z",
                                     2014, 12, 23, 12, 34, 45, 000, 000,   0 },
                { L_, "20141223-12:34:45+00:30",
                                     2014, 12, 23, 12, 34, 45, 000, 000,  30 },
                { L_, "20141223-12:34:45-01:30",
                                     2014, 12, 23, 12, 34, 45, 000, 000, -90 },
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int   LINE   = DATA[ti].d_line;
                const char *INPUT  = DATA[ti].d_input_p;
                const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));
                const int   YEAR   = DATA[ti].d_year;
                const int   MONTH  = DATA[ti].d_month;
                const int   DAY    = DATA[ti].d_day;
                const int   HOUR   = DATA[ti].d_hour;
                const int   MIN    = DATA[ti].d_min;
                const int   SEC    = DATA[ti].d_sec;
                const int   MSEC   = DATA[ti].d_msec;
                const int   USEC   = DATA[ti].d_usec;
                const int   OFFSET = DATA[ti].d_offset;

                if (veryVerbose) { T_ P_(LINE) P(INPUT) }

                bdlt::Datetime   mX(XX);  const bdlt::Datetime&   X = mX;
                bdlt::DatetimeTz mZ(ZZ);  const bdlt::DatetimeTz& Z = mZ;

                bdlt::DatetimeTz EXPECTED(bdlt::Datetime(YEAR,
                                                         MONTH,
                                                         DAY,
                                                         HOUR,
                                                         MIN,
                                                         SEC,
                                                         MSEC,
                                                         USEC),
                                          OFFSET);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mX, INPUT, LENGTH));
                ASSERTV(LINE, EXPECTED, X, EXPECTED.utcDatetime() == X);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mZ, INPUT, LENGTH));
                ASSERTV(LINE, EXPECTED, Z, EXPECTED == Z);

                mX = XX;
                mZ = ZZ;

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mX, StrView(INPUT, LENGTH)));
                ASSERTV(LINE, EXPECTED, X, EXPECTED.utcDatetime() == X);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mZ, StrView(INPUT, LENGTH)));
                ASSERTV(LINE, EXPECTED, Z, EXPECTED == Z);
            }
        }

        if (verbose)
            cout << "\nTesting timezone offsets that overflow a 'Datetime'."
                 << endl;
        {
            struct {
                int         d_line;
                const char *d_input_p;
                int         d_year;
                int         d_month;
                int         d_day;
                int         d_hour;
                int         d_min;
                int         d_sec;
                int         d_msec;
                int         d_offset;
            } DATA[] = {
                { L_, "00010101-00:00:00.000+00:00",
                                        0001, 01, 01, 00, 00, 00, 000,     0 },
                { L_, "00010101-00:00:00.000+00:01",
                                        0001, 01, 01, 00, 00, 00, 000,     1 },
                { L_, "00010101-23:58:59.000+23:59",
                                        0001, 01, 01, 23, 58, 59, 000,  1439 },

                { L_, "99991231-23:59:59.999+00:00",
                                        9999, 12, 31, 23, 59, 59, 999,     0 },
                { L_, "99991231-23:59:59.999-00:01",
                                        9999, 12, 31, 23, 59, 59, 999,    -1 },
                { L_, "99991231-00:01:00.000-23:59",
                                        9999, 12, 31, 00, 01, 00, 000, -1439 },
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int   LINE   = DATA[ti].d_line;
                const char *INPUT  = DATA[ti].d_input_p;
                const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));
                const int   YEAR   = DATA[ti].d_year;
                const int   MONTH  = DATA[ti].d_month;
                const int   DAY    = DATA[ti].d_day;
                const int   HOUR   = DATA[ti].d_hour;
                const int   MIN    = DATA[ti].d_min;
                const int   SEC    = DATA[ti].d_sec;
                const int   MSEC   = DATA[ti].d_msec;
                const int   OFFSET = DATA[ti].d_offset;

                if (veryVerbose) { T_ P_(LINE) P(INPUT) }

                bdlt::Datetime   mX(XX);  const bdlt::Datetime&   X = mX;
                bdlt::DatetimeTz mZ(ZZ);  const bdlt::DatetimeTz& Z = mZ;

                bdlt::DatetimeTz EXPECTED(bdlt::Datetime(YEAR, MONTH, DAY,
                                                         HOUR, MIN, SEC, MSEC),
                                          OFFSET);

                if (0 == OFFSET) {
                    ASSERTV(LINE, INPUT, 0 == Util::parse(&mX, INPUT, LENGTH));
                    ASSERTV(LINE, INPUT, EXPECTED.utcDatetime() == X);
                }
                else {
                    ASSERTV(LINE, INPUT, 0 != Util::parse(&mX, INPUT, LENGTH));
                    ASSERTV(LINE, INPUT, XX == X);
                }

                ASSERTV(LINE, INPUT, 0 == Util::parse(&mZ, INPUT, LENGTH));
                ASSERTV(LINE, INPUT, EXPECTED, Z, EXPECTED == Z);

                mX = XX;
                mZ = ZZ;

                if (0 == OFFSET) {
                    ASSERTV(LINE, INPUT,
                            0 == Util::parse(&mX, StrView(INPUT, LENGTH)));
                    ASSERTV(LINE, INPUT, EXPECTED.utcDatetime() == X);
                }
                else {
                    ASSERTV(LINE, INPUT,
                            0 != Util::parse(&mX, StrView(INPUT, LENGTH)));
                    ASSERTV(LINE, INPUT, XX == X);
                }

                ASSERTV(LINE, INPUT,
                        0 == Util::parse(&mZ, StrView(INPUT, LENGTH)));
                ASSERTV(LINE, INPUT, EXPECTED, Z, EXPECTED == Z);
            }
        }

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const char *INPUT  = "2013-10-23T01:23:45";
            const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));

            const StrView stringRef(INPUT, LENGTH);
            const StrView nullRef;

            bdlt::Datetime   result;
            bdlt::DatetimeTz resultTz;

            if (veryVerbose) cout << "\t'Invalid result'" << endl;
            {
                bdlt::Datetime   *bad   = 0;  (void)bad;
                bdlt::DatetimeTz *badTz = 0;  (void)badTz;

                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(      bad, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(    badTz, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(      bad, stringRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(    badTz, stringRef));
            }

            if (veryVerbose) cout << "\t'Invalid input'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(  &result,     0, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(&resultTz,     0, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(  &result, nullRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(&resultTz, nullRef));
            }

            if (veryVerbose) cout << "\t'Invalid length'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(  &result, INPUT,      0));
                ASSERT_FAIL(Util::parse(  &result, INPUT,     -1));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(&resultTz, INPUT,      0));
                ASSERT_FAIL(Util::parse(&resultTz, INPUT,     -1));
            }
        }
      } break;
      case 8: {
        // --------------------------------------------------------------------
        // PARSE: TIME & TIMETZ
        //
        // Concerns:
        //: 1 All FIX string representations supported by this component
        //:   (as documented in the header file) for 'Time' and 'TimeTz' values
        //:   are parsed successfully.
        //:
        //: 2 If parsing succeeds, the result 'Time' or 'TimeTz' object has the
        //:   expected value.
        //:
        //: 3 If the optional timezone offset is present in the input string
        //:   when parsing into a 'Time' object, the resulting value is
        //:   converted to the equivalent UTC time.
        //:
        //: 4 If the optional timezone offset is *not* present in the input
        //:   string when parsing into a 'TimeTz' object, it is assumed to be
        //:   UTC.
        //:
        //: 5 If parsing succeeds, 0 is returned.
        //:
        //: 6 All strings that are not FIX representations supported by
        //:   this component for 'Time' and 'TimeTz' values are rejected (i.e.,
        //:   parsing fails).
        //:
        //: 7 If parsing fails, the result object is unaffected and a non-zero
        //:   value is returned.
        //:
        //: 8 The entire extent of the input string is parsed.
        //:
        //: 9 Leap seconds and fractional seconds containing more than three
        //:   digits are handled correctly.
        //:
        //:10 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Time' values ('T'), timezone offsets ('Z'), and configurations
        //:   ('C').
        //:
        //: 2 Apply the (fully-tested) 'generateRaw' functions to each element
        //:   in the cross product, 'T x Z x C', of the test data from P-1.
        //:
        //: 3 Invoke the 'parse' functions on the strings generated in P-2 and
        //:   verify that parsing succeeds, i.e., that 0 is returned and the
        //:   result objects have the expected values.  (C-1..5)
        //:
        //: 4 Using the table-driven technique, specify a set of distinct
        //:   strings that are not FIX representations supported by this
        //:   component for 'Time' and 'TimeTz' values.
        //:
        //: 5 Invoke the 'parse' functions on the strings from P-4 and verify
        //:   that parsing fails, i.e., that a non-zero value is returned and
        //:   the result objects are unchanged.  (C-6..8)
        //:
        //: 6 Using the table-driven technique, specify a set of distinct
        //:   FIX strings that specifically cover cases involving leap
        //:   seconds and fractional seconds containing more than three digits.
        //:
        //: 7 Invoke the 'parse' functions on the strings from P-6 and verify
        //:   the results are as expected.  (C-9)
        //:
        //: 8 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-10)
        //
        // Testing:
        //   int parse(Time *, const char *, int);
        //   int parse(TimeTz *, const char *, int);
        //   int parse(Time *result, const StringRef& string);
        //   int parse(TimeTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "PARSE: TIME & TIMETZ" << endl
                          << "====================" << endl;

        char buffer[Util::k_MAX_STRLEN];

        const bdlt::Time   XX(2, 4, 6, 8);  // 'XX' and 'ZZ' are controls,
        const bdlt::TimeTz ZZ(XX, -7);      // distinct from any test data

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_EXT_ZONE_DATA =  NUM_EXTENDED_ZONE_DATA;
        const DefaultZoneDataRow (&EXT_ZONE_DATA)[NUM_EXT_ZONE_DATA] =
                                                            EXTENDED_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        if (verbose) cout << "\nValid FIX strings." << endl;

        for (int ti = 0; ti < NUM_TIME_DATA; ++ti) {
            const int ILINE = TIME_DATA[ti].d_line;
            const int HOUR  = TIME_DATA[ti].d_hour;
            const int MIN   = TIME_DATA[ti].d_min;
            const int SEC   = TIME_DATA[ti].d_sec;
            const int MSEC  = TIME_DATA[ti].d_msec;

            for (int tj = 0; tj < NUM_ZONE_DATA; ++tj) {
                const int JLINE  = ZONE_DATA[tj].d_line;
                const int OFFSET = ZONE_DATA[tj].d_offset;

                if (24 == HOUR) {
                    continue;  // skip invalid compositions
                }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    int expMsec = MSEC;
                    {
                        // adjust the expected milliseconds to account for
                        // PRECISION truncating the value generated

                        int precision = (PRECISION < 3 ? PRECISION : 3);

                        for (int i = 3; i > precision; --i) {
                            expMsec /= 10;
                        }

                        for (int i = 3; i > precision; --i) {
                            expMsec *= 10;
                        }
                    }

                    const bdlt::Time   TIME(HOUR, MIN, SEC, expMsec);
                    const bdlt::TimeTz TIMETZ(TIME, OFFSET);

                    if (veryVerbose) {
                        if (0 == tc) {
                            T_ P_(ILINE) P_(JLINE) P_(TIME) P(TIMETZ);
                        }
                        T_ P_(CLINE) P_(PRECISION) P(USEZ);
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // without timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer, TIME, C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Time   mX(XX);  const bdlt::Time&   X = mX;
                        bdlt::TimeTz mZ(ZZ);  const bdlt::TimeTz& Z = mZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, TIME, X, TIME == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));

                        ASSERTV(ILINE, JLINE, CLINE,
                                TIMETZ.localTime() == Z.localTime());
                        ASSERTV(ILINE, JLINE, CLINE, 0 == Z.offset());

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, TIME == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, TIME == Z.localTime());
                        ASSERTV(ILINE, JLINE, CLINE,    0 == Z.offset());
                    }

                    // with timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer,
                                                             TIMETZ,
                                                             C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Time   mX(XX);  const bdlt::Time&   X = mX;
                        bdlt::TimeTz mZ(ZZ);  const bdlt::TimeTz& Z = mZ;

                        // 'TimeTz' uses the FIX "TZTimeOnly" format during
                        // generation so there are no milliseconds.

                        const bdlt::TimeTz EXPTIMETZ(
                                                    bdlt::Time(HOUR, MIN, SEC),
                                                    OFFSET);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ.utcTime() == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ           == Z);

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ.utcTime() == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ           == Z);
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'

            for (int tj = 0; tj < NUM_EXT_ZONE_DATA; ++tj) {
                const int JLINE  = EXT_ZONE_DATA[tj].d_line;
                const int OFFSET = EXT_ZONE_DATA[tj].d_offset;

                if (24 == HOUR) {
                    continue;  // skip invalid compositions
                }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    int expMsec = MSEC;
                    {
                        // adjust the expected milliseconds to account for
                        // PRECISION truncating the value generated

                        int precision = (PRECISION < 3 ? PRECISION : 3);

                        for (int i = 3; i > precision; --i) {
                            expMsec /= 10;
                        }

                        for (int i = 3; i > precision; --i) {
                            expMsec *= 10;
                        }
                    }

                    const bdlt::Time   TIME(HOUR, MIN, SEC, expMsec);
                    const bdlt::TimeTz TIMETZ(TIME, OFFSET);

                    if (veryVerbose) {
                        if (0 == tc) {
                            T_ P_(ILINE) P_(JLINE) P_(TIME) P(TIMETZ);
                        }
                        T_ P_(CLINE) P_(PRECISION) P(USEZ);
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // with timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer,
                                                             TIMETZ,
                                                             C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Time   mX(XX);  const bdlt::Time&   X = mX;
                        bdlt::TimeTz mZ(ZZ);  const bdlt::TimeTz& Z = mZ;

                        // 'TimeTz' uses the FIX "TZTimeOnly" format during
                        // generation so there are no milliseconds.

                        const bdlt::TimeTz EXPTIMETZ(
                                                    bdlt::Time(HOUR, MIN, SEC),
                                                    OFFSET);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ.utcTime() == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ           == Z);

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ.utcTime() == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, EXPTIMETZ           == Z);
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'
        }  // loop over 'TIME_DATA'

        if (verbose) cout << "\nInvalid strings." << endl;
        {
            bdlt::Time   mX(XX);  const bdlt::Time&   X = mX;
            bdlt::TimeTz mZ(ZZ);  const bdlt::TimeTz& Z = mZ;

            const int              NUM_TIME_DATA =         NUM_BAD_TIME_DATA;
            const BadTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] = BAD_TIME_DATA;

            for (int ti = 0; ti < NUM_TIME_DATA; ++ti) {
                const int   LINE   = TIME_DATA[ti].d_line;
                const char *STRING = TIME_DATA[ti].d_invalid;

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                const int LENGTH = static_cast<int>(bsl::strlen(STRING));

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }

            const int              NUM_ZONE_DATA =         NUM_BAD_ZONE_DATA;
            const BadZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] = BAD_ZONE_DATA;

            for (int ti = 0; ti < NUM_ZONE_DATA; ++ti) {
                const int LINE = ZONE_DATA[ti].d_line;

                // Initialize with a *valid* time string, then append an
                // invalid timezone offset.

                bsl::string bad("12:26:52.726");

                // Ensure that 'bad' is initially valid.

                static bool firstFlag = true;
                if (firstFlag) {
                    const char *STRING = bad.data();
                    const int   LENGTH = static_cast<int>(bad.length());

                    bdlt::Time mT(XX);  const bdlt::Time& T = mT;

                    ASSERT( 0 == Util::parse(&mT, STRING, LENGTH));
                    ASSERT(XX != T);

                    mT = XX;

                    ASSERT( 0 == Util::parse(&mT, StrView(STRING, LENGTH)));
                    ASSERT(XX != T);
                }

                // If 'ZONE_DATA[ti].d_invalid' contains nothing but digits,
                // appending it to 'bad' simply extends the fractional second
                // (so 'bad' remains valid).

                if (containsOnlyDigits(ZONE_DATA[ti].d_invalid)) {
                    continue;
                }

                bad.append(ZONE_DATA[ti].d_invalid);

                const char *STRING = bad.c_str();
                const int   LENGTH = static_cast<int>(bad.length());

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }
        }

        if (verbose) {
            cout << "\nTesting optional, leap, and fractional seconds."
                 << endl;
        }
        {
            const struct {
                int         d_line;
                const char *d_input_p;
                int         d_hour;
                int         d_min;
                int         d_sec;
                int         d_msec;
                int         d_usec;
                int         d_offset;
            } DATA[] = {
                // line, input,             hour, min, sec, msec, usec, offset
                // optional seconds
                { L_,    "00:00",              0,   0,   0,    0,    0,   0 },
                { L_,    "00:01",              0,   1,   0,    0,    0,   0 },
                { L_,    "01:00",              1,   0,   0,    0,    0,   0 },
                { L_,    "01:01",              1,   1,   0,    0,    0,   0 },

                // Test all possible number of digits
                { L_,    "12:34:56",           12,  34,  56,    0,    0,   0 },
                { L_,    "12:34:56.7",         12,  34,  56,  700,    0,   0 },
                { L_,    "12:34:56.78",        12,  34,  56,  780,    0,   0 },
                { L_,    "12:34:56.789",       12,  34,  56,  789,    0,   0 },
                { L_,    "12:34:56.7898",      12,  34,  56,  789,  800,   0 },
                { L_,    "12:34:56.78987",     12,  34,  56,  789,  870,   0 },
                { L_,    "12:34:56.789876",    12,  34,  56,  789,  876,   0 },

                // leap seconds
                { L_,    "00:00:60.000",       0,   1,   0,    0,    0,   0 },
                { L_,    "22:59:60.999",      23,   0,   0,  999,    0,   0 },
                { L_,    "23:59:60.999",       0,   0,   0,  999,    0,   0 },

                // fractional seconds
                { L_,    "00:00:00.0000001",   0,   0,   0,    0,    0,   0 },
                { L_,    "00:00:00.0000009",   0,   0,   0,    0,    1,   0 },
                { L_,    "00:00:00.00000001",  0,   0,   0,    0,    0,   0 },
                { L_,    "00:00:00.00000049",  0,   0,   0,    0,    0,   0 },
                { L_,    "00:00:00.00000050",  0,   0,   0,    0,    1,   0 },
                { L_,    "00:00:00.00000099",  0,   0,   0,    0,    1,   0 },
                { L_,    "00:00:00.9999994",   0,   0,   0,  999,  999,   0 },
                { L_,    "00:00:00.9999995",   0,   0,   1,    0,    0,   0 },
                { L_,    "00:00:00.9999999",   0,   0,   1,    0,    0,   0 },
                { L_,    "00:00:59.9999999",   0,   1,   0,    0,    0,   0 },
                { L_,    "22:59:60.9999999",  23,   0,   1,    0,    0,   0 },
                { L_,    "23:59:60.9999999",   0,   0,   1,    0,    0,   0 },

                // omit fractional seconds
                { L_,    "00:00:60",           0,   1,   0,    0,    0,   0 },
                { L_,    "12:34:45",          12,  34,  45,    0,    0,   0 },
                { L_,    "12:34:45Z",         12,  34,  45,    0,    0,   0 },
                { L_,    "12:34:45+00:30",    12,  34,  45,    0,    0,  30 },
                { L_,    "00:00:00+00:30",     0,   0,   0,    0,    0,  30 },
                { L_,    "12:34:45-01:30",    12,  34,  45,    0,    0, -90 },
                { L_,    "23:59:59-01:30",    23,  59,  59,    0,    0, -90 },
            };
            const int NUM_DATA = static_cast<int>(sizeof DATA / sizeof *DATA);

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int   LINE   = DATA[ti].d_line;
                const char *INPUT  = DATA[ti].d_input_p;
                const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));
                const int   HOUR   = DATA[ti].d_hour;
                const int   MIN    = DATA[ti].d_min;
                const int   SEC    = DATA[ti].d_sec;
                const int   MSEC   = DATA[ti].d_msec;
                const int   USEC   = DATA[ti].d_usec;
                const int   OFFSET = DATA[ti].d_offset;

                if (veryVerbose) { T_ P_(LINE) P(INPUT) }

                bdlt::Time   mX(XX);  const bdlt::Time&   X = mX;
                bdlt::TimeTz mZ(ZZ);  const bdlt::TimeTz& Z = mZ;

                bdlt::TimeTz EXPECTED(bdlt::Time(HOUR, MIN, SEC, MSEC, USEC),
                                      OFFSET);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mX, INPUT, LENGTH));
                ASSERTV(LINE, EXPECTED, X, EXPECTED.utcTime() == X);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mZ, INPUT, LENGTH));
                ASSERTV(LINE, EXPECTED, Z, EXPECTED == Z);

                mX = XX;
                mZ = ZZ;

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mX, StrView(INPUT, LENGTH)));
                ASSERTV(LINE, EXPECTED, X, EXPECTED.utcTime() == X);

                ASSERTV(LINE, INPUT, LENGTH,
                        0 == Util::parse(&mZ, StrView(INPUT, LENGTH)));
                ASSERTV(LINE, EXPECTED, Z, EXPECTED == Z);
            }
        }

        // Read overflow - {DRQS 163929029}.  May need sanitizer to detect.
        if (verbose) cout << "\nSanitizer Test Cases." << endl;
        {
            bdlt::Time mX;
            char S[] = { '1', '1', ':', '1', '1', ':', '1' };
            ASSERTV(0 != Util::parse(&mX, S, int(sizeof(S))));
        }
        {
            bdlt::Time mX;
            char S[] = { '1', '2', ':', '5', '1', '-', '1', '2' };
            ASSERTV(0 == Util::parse(&mX, S, int(sizeof(S))));
        }

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const char *INPUT  = "01:23:45";
            const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));

            const StrView stringRef(INPUT, LENGTH);
            const StrView nullRef;

            bdlt::Time   result;
            bdlt::TimeTz resultTz;

            if (veryVerbose) cout << "\t'Invalid result'" << endl;
            {
                bdlt::Time   *bad   = 0;  (void)bad;
                bdlt::TimeTz *badTz = 0;  (void)badTz;

                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(      bad, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(    badTz, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(      bad, stringRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(    badTz, stringRef));
            }

            if (veryVerbose) cout << "\t'Invalid input'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(  &result,     0, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(&resultTz,     0, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(  &result, nullRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(&resultTz, nullRef));
            }

            if (veryVerbose) cout << "\t'Invalid length'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(  &result, INPUT,      0));
                ASSERT_FAIL(Util::parse(  &result, INPUT,     -1));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(&resultTz, INPUT,      0));
                ASSERT_FAIL(Util::parse(&resultTz, INPUT,     -1));
            }
        }

      } break;
      case 7: {
        // --------------------------------------------------------------------
        // PARSE: DATE & DATETZ
        //
        // Concerns:
        //: 1 All FIX string representations supported by this component
        //:   (as documented in the header file) for 'Date' and 'DateTz' values
        //:   are parsed successfully.
        //:
        //: 2 If parsing succeeds, the result 'Date' or 'DateTz' object has the
        //:   expected value.
        //:
        //: 3 If the optional timezone offset is present in the input string
        //:   when parsing into a 'Date' object, it is parsed for validity but
        //:   is otherwise ignored.
        //:
        //: 4 If the optional timezone offset is *not* present in the input
        //:   string when parsing into a 'DateTz' object, it is assumed to be
        //:   UTC.
        //:
        //: 5 If parsing succeeds, 0 is returned.
        //:
        //: 6 All strings that are not FIX representations supported by
        //:   this component for 'Date' and 'DateTz' values are rejected (i.e.,
        //:   parsing fails).
        //:
        //: 7 If parsing fails, the result object is unaffected and a non-zero
        //:   value is returned.
        //:
        //: 8 The entire extent of the input string is parsed.
        //:
        //: 9 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values ('D'), timezone offsets ('Z'), and configurations
        //:   ('C').
        //:
        //: 2 Apply the (fully-tested) 'generateRaw' functions to each element
        //:   in the cross product, 'D x Z x C', of the test data from P-1.
        //:
        //: 3 Invoke the 'parse' functions on the strings generated in P-2 and
        //:   verify that parsing succeeds, i.e., that 0 is returned and the
        //:   result objects have the expected values.  (C-1..5)
        //:
        //: 4 Using the table-driven technique, specify a set of distinct
        //:   strings that are not FIX representations supported by this
        //:   component for 'Date' and 'DateTz' values.
        //:
        //: 5 Invoke the 'parse' functions on the strings from P-4 and verify
        //:   that parsing fails, i.e., that a non-zero value is returned and
        //:   the result objects are unchanged.  (C-6..8)
        //:
        //: 6 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-9)
        //
        // Testing:
        //   int parse(Date *, const char *, int);
        //   int parse(DateTz *, const char *, int);
        //   int parse(Date *result, const StringRef& string);
        //   int parse(DateTz *result, const StringRef& string);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "PARSE: DATE & DATETZ" << endl
                          << "====================" << endl;

        char buffer[Util::k_MAX_STRLEN];

        const bdlt::Date   XX(246, 8, 10);  // 'XX' and 'ZZ' are controls,
        const bdlt::DateTz ZZ(XX, -7);      // distinct from any test data

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_EXT_ZONE_DATA =  NUM_EXTENDED_ZONE_DATA;
        const DefaultZoneDataRow (&EXT_ZONE_DATA)[NUM_EXT_ZONE_DATA] =
                                                            EXTENDED_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        if (verbose) cout << "\nValid FIX strings." << endl;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int ILINE = DATE_DATA[ti].d_line;
            const int YEAR  = DATE_DATA[ti].d_year;
            const int MONTH = DATE_DATA[ti].d_month;
            const int DAY   = DATE_DATA[ti].d_day;

            const bdlt::Date DATE(YEAR, MONTH, DAY);

            for (int tj = 0; tj < NUM_ZONE_DATA; ++tj) {
                const int JLINE  = ZONE_DATA[tj].d_line;
                const int OFFSET = ZONE_DATA[tj].d_offset;

                const bdlt::DateTz DATETZ(DATE, OFFSET);

                if (veryVerbose) { T_ P_(ILINE) P_(JLINE) P_(DATE) P(DATETZ) }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // without timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer, DATE, C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Date   mX(XX);  const bdlt::Date&   X = mX;
                        bdlt::DateTz mZ(ZZ);  const bdlt::DateTz& Z = mZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATE == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATE == Z.localDate());
                        ASSERTV(ILINE, JLINE, CLINE,    0 == Z.offset());

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATE == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATE == Z.localDate());
                        ASSERTV(ILINE, JLINE, CLINE,    0 == Z.offset());
                    }

                    // with timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer,
                                                             DATETZ,
                                                             C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Date   mX(XX);  const bdlt::Date&   X = mX;
                        bdlt::DateTz mZ(ZZ);  const bdlt::DateTz& Z = mZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATE   == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATETZ == Z);

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATE   == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATETZ == Z);
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'

            for (int tj = 0; tj < NUM_EXT_ZONE_DATA; ++tj) {
                const int JLINE  = EXT_ZONE_DATA[tj].d_line;
                const int OFFSET = EXT_ZONE_DATA[tj].d_offset;

                const bdlt::DateTz DATETZ(DATE, OFFSET);

                if (veryVerbose) { T_ P_(ILINE) P_(JLINE) P_(DATE) P(DATETZ) }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // with timezone offset in parsed string
                    {
                        const int LENGTH = Util::generateRaw(buffer,
                                                             DATETZ,
                                                             C);

                        if (veryVerbose) {
                            const bsl::string STRING(buffer, LENGTH);
                            T_ T_ P(STRING)
                        }

                        bdlt::Date   mX(XX);  const bdlt::Date&   X = mX;
                        bdlt::DateTz mZ(ZZ);  const bdlt::DateTz& Z = mZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mX, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATE   == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                                0 == Util::parse(&mZ, buffer, LENGTH));
                        ASSERTV(ILINE, JLINE, CLINE, DATETZ == Z);

                        mX = XX;
                        mZ = ZZ;

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mX, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATE   == X);

                        ASSERTV(ILINE, JLINE, CLINE,
                               0 == Util::parse(&mZ, StrView(buffer, LENGTH)));
                        ASSERTV(ILINE, JLINE, CLINE, DATETZ == Z);
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nInvalid strings." << endl;
        {
            bdlt::Date   mX(XX);  const bdlt::Date&   X = mX;
            bdlt::DateTz mZ(ZZ);  const bdlt::DateTz& Z = mZ;

            const int              NUM_DATE_DATA =         NUM_BAD_DATE_DATA;
            const BadDateDataRow (&DATE_DATA)[NUM_DATE_DATA] = BAD_DATE_DATA;

            for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
                const int   LINE   = DATE_DATA[ti].d_line;
                const char *STRING = DATE_DATA[ti].d_invalid;

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                const int LENGTH = static_cast<int>(bsl::strlen(STRING));

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }

            const int              NUM_ZONE_DATA =         NUM_BAD_ZONE_DATA;
            const BadZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] = BAD_ZONE_DATA;

            for (int ti = 0; ti < NUM_ZONE_DATA; ++ti) {
                const int LINE = ZONE_DATA[ti].d_line;

                // Initialize with a *valid* date string, then append an
                // invalid timezone offset.

                bsl::string bad("20100817");

                // Ensure that 'bad' is initially valid.

                static bool firstFlag = true;
                if (firstFlag) {
                    const char *STRING = bad.data();
                    const int   LENGTH = static_cast<int>(bad.length());

                    bdlt::Date mD(XX);  const bdlt::Date& D = mD;

                    ASSERT( 0 == Util::parse(&mD, STRING, LENGTH));
                    ASSERT(XX != D);

                    mD = XX;

                    ASSERT( 0 == Util::parse(&mD, StrView(STRING, LENGTH)));
                    ASSERT(XX != D);
                }

                bad.append(ZONE_DATA[ti].d_invalid);

                const char *STRING = bad.c_str();
                const int   LENGTH = static_cast<int>(bad.length());

                if (veryVerbose) { T_ P_(LINE) P(STRING) }

                ASSERTV(LINE, STRING,  0 != Util::parse(&mX, STRING, LENGTH));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,  0 != Util::parse(&mZ, STRING, LENGTH));
                ASSERTV(LINE, STRING, ZZ == Z);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mX, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, XX == X);

                ASSERTV(LINE, STRING,
                        0 != Util::parse(&mZ, StrView(STRING, LENGTH)));
                ASSERTV(LINE, STRING, ZZ == Z);
            }
        }

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const char *INPUT  = "2013-10-23";
            const int   LENGTH = static_cast<int>(bsl::strlen(INPUT));

            const StrView stringRef(INPUT, LENGTH);
            const StrView nullRef;

            bdlt::Date   result;
            bdlt::DateTz resultTz;

            if (veryVerbose) cout << "\t'Invalid result'" << endl;
            {
                bdlt::Date   *bad   = 0;  (void)bad;
                bdlt::DateTz *badTz = 0;  (void)badTz;

                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(      bad, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(    badTz, INPUT, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(      bad, stringRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(    badTz, stringRef));
            }

            if (veryVerbose) cout << "\t'Invalid input'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(  &result,     0, LENGTH));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_FAIL(Util::parse(&resultTz,     0, LENGTH));

                ASSERT_PASS(Util::parse(  &result, stringRef));
                ASSERT_FAIL(Util::parse(  &result, nullRef));

                ASSERT_PASS(Util::parse(&resultTz, stringRef));
                ASSERT_FAIL(Util::parse(&resultTz, nullRef));
            }

            if (veryVerbose) cout << "\t'Invalid length'" << endl;
            {
                ASSERT_PASS(Util::parse(  &result, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(  &result, INPUT,      0));
                ASSERT_FAIL(Util::parse(  &result, INPUT,     -1));

                ASSERT_PASS(Util::parse(&resultTz, INPUT, LENGTH));
                ASSERT_PASS(Util::parse(&resultTz, INPUT,      0));
                ASSERT_FAIL(Util::parse(&resultTz, INPUT,     -1));
            }
        }

      } break;
      case 6: {
        // --------------------------------------------------------------------
        // GENERATE 'DatetimeTz'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 In a second table, specify a set of distinct 'Time' values (one
        //:   per row) and their corresponding FIX string representations.
        //:
        //: 3 In a third table, specify a set of distinct timezone values (one
        //:   per row) and their corresponding FIX string representations.
        //:
        //: 4 For each element 'R' in the cross product of the tables from P-1,
        //:   P-2, and P-3:  (C-1..5)
        //:
        //:   1 Create a 'const' 'DatetimeTz' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const DatetimeTz&);
        //   int generate(char *, int, const DatetimeTz&, const Config&);
        //   int generate(string *, const DatetimeTz&);
        //   int generate(string *, const DatetimeTz&, const Config&);
        //   ostream generate(ostream&, const DatetimeTz&);
        //   ostream generate(ostream&, const DatetimeTz&, const Config&);
        //   int generateRaw(char *, const DatetimeTz&);
        //   int generateRaw(char *, const DatetimeTz&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'DatetimeTz'" << endl
                          << "=====================" << endl;

        typedef bdlt::DatetimeTz TYPE;

        const int OBJLEN = Util::k_DATETIMETZ_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int   ILINE = DATE_DATA[ti].d_line;
            const int   YEAR  = DATE_DATA[ti].d_year;
            const int   MONTH = DATE_DATA[ti].d_month;
            const int   DAY   = DATE_DATA[ti].d_day;
            const char *FIX   = DATE_DATA[ti].d_fix;

            const bdlt::Date  DATE(YEAR, MONTH, DAY);
            const bsl::string EXPECTED_DATE(FIX);

            for (int tj = 0; tj < NUM_TIME_DATA; ++tj) {
                const int   JLINE = TIME_DATA[tj].d_line;
                const int   HOUR  = TIME_DATA[tj].d_hour;
                const int   MIN   = TIME_DATA[tj].d_min;
                const int   SEC   = TIME_DATA[tj].d_sec;
                const int   MSEC  = TIME_DATA[tj].d_msec;
                const int   USEC  = TIME_DATA[tj].d_usec;
                const char *FIX   = TIME_DATA[tj].d_fix;

                const bdlt::Time  TIME(HOUR, MIN, SEC, MSEC);
                const bsl::string EXPECTED_TIME(FIX);

                for (int tk = 0; tk < NUM_ZONE_DATA; ++tk) {
                    const int   KLINE   = ZONE_DATA[tk].d_line;
                    const int   OFFSET  = ZONE_DATA[tk].d_offset;
                    const char *FIX = ZONE_DATA[tk].d_fix;

                    const bsl::string EXPECTED_ZONE(FIX);

                    if (TIME == bdlt::Time() && OFFSET != 0) {
                        continue;  // skip invalid compositions
                    }

                    const TYPE        X(bdlt::Datetime(YEAR,
                                                       MONTH,
                                                       DAY,
                                                       HOUR,
                                                       MIN,
                                                       SEC,
                                                       MSEC,
                                                       USEC),
                                        OFFSET);
                    const bsl::string BASE_EXPECTED(
                          EXPECTED_DATE + '-' + EXPECTED_TIME + EXPECTED_ZONE);

                    if (veryVerbose) {
                        T_ P_(ILINE) P_(JLINE) P_(KLINE) P_(X) P(BASE_EXPECTED)
                    }

                    for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                        const int  CLINE     = CNFG_DATA[tc].d_line;
                        const int  PRECISION = CNFG_DATA[tc].d_precision;
                        const bool USEZ      = CNFG_DATA[tc].d_useZ;

                        if (veryVerbose) {
                            T_ P_(CLINE) P_(PRECISION) P(USEZ)
                        }

                        Config mC;  const Config& C = mC;
                        gg(&mC, PRECISION, USEZ);

                        Config::setDefaultConfiguration(C);

                        bsl::string EXPECTED(BASE_EXPECTED);
                        updateExpectedPerConfig(&EXPECTED,
                                                C,
                                                k_DATETIMETZ_MAX_PRECISION);

                        const int OUTLEN = static_cast<int>(EXPECTED.length());

                        // 'generate' taking 'bufferLength'

                        for (int k = 0; k < BUFLEN; ++k) {
                            bsl::memset(buffer, '?', BUFLEN);

                            if (veryVeryVerbose) {
                                T_ T_ cout << "Length: "; P(k)
                            }

                            ASSERTV(ILINE, JLINE, KLINE, k, OUTLEN,
                                    OUTLEN == Util::generate(buffer, k, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(EXPECTED.c_str(),
                                                     buffer,
                                                     k < OUTLEN ? k : OUTLEN));

                            if (k <= OUTLEN) {
                                ASSERTV(ILINE, JLINE, KLINE, EXPECTED, buffer,
                                        0 == bsl::memcmp(chaste,
                                                         buffer + k,
                                                         BUFLEN - k));
                            }
                            else {
                                ASSERTV(ILINE, JLINE, KLINE, k, OUTLEN,
                                        '\0' == buffer[OUTLEN]);

                                ASSERTV(ILINE, JLINE, KLINE, EXPECTED, buffer,
                                        0 == bsl::memcmp(chaste,
                                                         buffer + k + 1,
                                                         BUFLEN - k - 1));
                            }
                        }

                        // 'generate' to a 'string'
                        {
                            bsl::string mS("qwerty");

                            ASSERTV(ILINE, JLINE, KLINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, mS,
                                    EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
                        {
                            std::string mS("qwerty");

                            ASSERTV(ILINE, JLINE, KLINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, mS,
                                    EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                        {
                            std::pmr::string mS("qwerty");

                            ASSERTV(ILINE, JLINE, KLINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, mS,
                                    EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
#endif

                        // 'generate' to an 'ostream'
                        {
                            bsl::ostringstream os;

                            ASSERTV(ILINE, JLINE, KLINE,
                                    &os == &Util::generate(os, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, os.str(),
                                    EXPECTED == os.str());

                            if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                        }

                        // 'generateRaw'
                        {
                            bsl::memset(buffer, '?', BUFLEN);

                            ASSERTV(ILINE, JLINE, KLINE, OUTLEN,
                                    OUTLEN == Util::generateRaw(buffer, X));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(EXPECTED.c_str(),
                                                     buffer,
                                                     OUTLEN));

                            ASSERTV(ILINE, JLINE, KLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + OUTLEN,
                                                     BUFLEN - OUTLEN));
                        }
                    }  // loop over 'CNFG_DATA'

                    for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                        const int  CLINE     = CNFG_DATA[tc].d_line;
                        const int  PRECISION = CNFG_DATA[tc].d_precision;
                        const bool USEZ      = CNFG_DATA[tc].d_useZ;

                        if (veryVerbose) {
                            T_ P_(CLINE) P_(PRECISION) P(USEZ)
                        }

                        Config mC;  const Config& C = mC;
                        gg(&mC, PRECISION, USEZ);

                        // Set the default configuration to the complement of
                        // 'C'.

                        Config mDFLT;  const Config& DFLT = mDFLT;
                        gg(&mDFLT,
                           9 - PRECISION,
                           !USEZ);
                        Config::setDefaultConfiguration(DFLT);

                        bsl::string EXPECTED(BASE_EXPECTED);
                        updateExpectedPerConfig(&EXPECTED,
                                                C,
                                                k_DATETIMETZ_MAX_PRECISION);

                        const int OUTLEN = static_cast<int>(EXPECTED.length());

                        // 'generate' taking 'bufferLength'

                        for (int k = 0; k < BUFLEN; ++k) {
                            bsl::memset(buffer, '?', BUFLEN);

                            if (veryVeryVerbose) {
                                T_ T_ cout << "Length: "; P(k)
                            }

                            ASSERTV(ILINE, k, OUTLEN,
                                    OUTLEN == Util::generate(buffer, k, X, C));

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(EXPECTED.c_str(),
                                                     buffer,
                                                     k < OUTLEN ? k : OUTLEN));

                            if (k <= OUTLEN) {
                                ASSERTV(ILINE, EXPECTED, buffer,
                                        0 == bsl::memcmp(chaste,
                                                         buffer + k,
                                                         BUFLEN - k));
                            }
                            else {
                                ASSERTV(ILINE, k, OUTLEN,
                                        '\0' == buffer[OUTLEN]);

                                ASSERTV(ILINE, EXPECTED, buffer,
                                        0 == bsl::memcmp(chaste,
                                                         buffer + k + 1,
                                                         BUFLEN - k - 1));
                            }
                        }

                        // 'generate' to a 'string'
                        {
                            bsl::string mS("qwerty");

                            ASSERTV(ILINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X, C));

                            ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
                        {
                            std::string mS("qwerty");

                            ASSERTV(ILINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X, C));

                            ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                        {
                            std::pmr::string mS("qwerty");

                            ASSERTV(ILINE, OUTLEN,
                                    OUTLEN == Util::generate(&mS, X, C));

                            ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                            if (veryVerbose) { P_(EXPECTED) P(mS); }
                        }
#endif

                        // 'generate' to an 'ostream'
                        {
                            bsl::ostringstream os;

                            ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                            ASSERTV(ILINE, EXPECTED, os.str(),
                                    EXPECTED == os.str());

                            if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                        }

                        // 'generateRaw'
                        {
                            bsl::memset(buffer, '?', BUFLEN);

                            ASSERTV(ILINE, OUTLEN,
                                    OUTLEN == Util::generateRaw(buffer, X, C));

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(EXPECTED.c_str(),
                                                     buffer,
                                                     OUTLEN));

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + OUTLEN,
                                                     BUFLEN - OUTLEN));
                        }
                    }  // loop over 'CNFG_DATA'
                }  // loop over 'ZONE_DATA'
            }  // loop over 'TIME_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate(  pb, X, C));
                ASSERT_FAIL(Util::generate(  ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate(  pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
            }
        }

      } break;
      case 5: {
        // --------------------------------------------------------------------
        // GENERATE 'TimeTz'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Time' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 In a second table, specify a set of distinct timezone values (one
        //:   per row) and their corresponding FIX string representations.
        //:
        //: 3 For each element 'R' in the cross product of the tables from P-1
        //:   and P-2:  (C-1..5)
        //:
        //:   1 Create a 'const' 'TimeTz' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const TimeTz&);
        //   int generate(char *, int, const TimeTz&, const Config&);
        //   int generate(string *, const TimeTz&);
        //   int generate(string *, const TimeTz&, const Config&);
        //   ostream generate(ostream&, const TimeTz&);
        //   ostream generate(ostream&, const TimeTz&, const Config&);
        //   int generateRaw(char *, const TimeTz&);
        //   int generateRaw(char *, const TimeTz&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'TimeTz'" << endl
                          << "=================" << endl;

        typedef bdlt::TimeTz TYPE;

        const int OBJLEN = Util::k_TIMETZ_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_TIME_DATA; ++ti) {
            const int   ILINE   = TIME_DATA[ti].d_line;
            const int   HOUR    = TIME_DATA[ti].d_hour;
            const int   MIN     = TIME_DATA[ti].d_min;
            const int   SEC     = TIME_DATA[ti].d_sec;
            const int   MSEC    = TIME_DATA[ti].d_msec;
            const int   USEC    = TIME_DATA[ti].d_usec;
            const char *FIX = TIME_DATA[ti].d_fix;

            const bdlt::Time  TIME(HOUR, MIN, SEC, MSEC, USEC);
            const bsl::string EXPECTED_TIME(FIX);

            for (int tj = 0; tj < NUM_ZONE_DATA; ++tj) {
                const int   JLINE   = ZONE_DATA[tj].d_line;
                const int   OFFSET  = ZONE_DATA[tj].d_offset;
                const char *FIX = ZONE_DATA[tj].d_fix;

                const bsl::string EXPECTED_ZONE(FIX);

                if (TIME == bdlt::Time() && OFFSET != 0) {
                    continue;  // skip invalid compositions
                }

                const TYPE        X(TIME, OFFSET);
                const bsl::string BASE_EXPECTED(EXPECTED_TIME + EXPECTED_ZONE);

                if (veryVerbose) {
                    T_ P_(ILINE) P_(JLINE) P_(X) P(BASE_EXPECTED)
                }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    Config::setDefaultConfiguration(C);

                    // 'k_TIMETZ_MAX_PRECISION' ensures no fractional seconds.

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_TIMETZ_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, JLINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, JLINE, k, OUTLEN,
                                    '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, JLINE, &os == &Util::generate(os, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // Set the default configuration to the complement of 'C'.

                    Config mDFLT;  const Config& DFLT = mDFLT;
                    gg(&mDFLT, 9 - PRECISION, !USEZ);
                    Config::setDefaultConfiguration(DFLT);

                    // 'k_TIMETZ_MAX_PRECISION' ensures no fractional seconds.

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_TIMETZ_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                        ASSERTV(ILINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'
        }  // loop over 'TIME_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate( pb, X, C));
                ASSERT_FAIL(Util::generate( ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
            }
        }

      } break;
      case 4: {
        // --------------------------------------------------------------------
        // GENERATE 'DateTz'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 In a second table, specify a set of distinct timezone values (one
        //:   per row) and their corresponding FIX string representations.
        //:
        //: 3 For each element 'R' in the cross product of the tables from P-1
        //:   and P-2:  (C-1..5)
        //:
        //:   1 Create a 'const' 'DateTz' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const DateTz&);
        //   int generate(char *, int, const DateTz&, const Config&);
        //   int generate(string *, const DateTz&);
        //   int generate(string *, const DateTz&, const Config&);
        //   ostream generate(ostream&, const DateTz&);
        //   ostream generate(ostream&, const DateTz&, const Config&);
        //   int generateRaw(char *, const DateTz&);
        //   int generateRaw(char *, const DateTz&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'DateTz'" << endl
                          << "=================" << endl;

        typedef bdlt::DateTz TYPE;

        const int OBJLEN = Util::k_DATETZ_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_ZONE_DATA =       NUM_DEFAULT_ZONE_DATA;
        const DefaultZoneDataRow (&ZONE_DATA)[NUM_ZONE_DATA] =
                                                             DEFAULT_ZONE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int   ILINE   = DATE_DATA[ti].d_line;
            const int   YEAR    = DATE_DATA[ti].d_year;
            const int   MONTH   = DATE_DATA[ti].d_month;
            const int   DAY     = DATE_DATA[ti].d_day;
            const char *FIX = DATE_DATA[ti].d_fix;

            const bdlt::Date  DATE(YEAR, MONTH, DAY);
            const bsl::string EXPECTED_DATE(FIX);

            for (int tj = 0; tj < NUM_ZONE_DATA; ++tj) {
                const int   JLINE   = ZONE_DATA[tj].d_line;
                const int   OFFSET  = ZONE_DATA[tj].d_offset;
                const char *FIX = ZONE_DATA[tj].d_fix;

                const bsl::string EXPECTED_ZONE(FIX);

                const TYPE        X(DATE, OFFSET);
                const bsl::string BASE_EXPECTED(EXPECTED_DATE + EXPECTED_ZONE);

                if (veryVerbose) {
                    T_ P_(ILINE) P_(JLINE) P_(X) P(BASE_EXPECTED)
                }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    Config::setDefaultConfiguration(C);

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_DATETZ_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, JLINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, JLINE, k, OUTLEN,
                                    '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, JLINE, &os == &Util::generate(os, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // Set the default configuration to the complement of 'C'.

                    Config mDFLT;  const Config& DFLT = mDFLT;
                    gg(&mDFLT, 9 - PRECISION, !USEZ);
                    Config::setDefaultConfiguration(DFLT);

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_DATETZ_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                        ASSERTV(ILINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'ZONE_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate( pb, X, C));
                ASSERT_FAIL(Util::generate( ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
            }
        }

      } break;
      case 3: {
        // --------------------------------------------------------------------
        // GENERATE 'Datetime'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 In a second table, specify a set of distinct 'Time' values (one
        //:   per row) and their corresponding FIX string representations.
        //:
        //: 3 For each element 'R' in the cross product of the tables from P-1
        //:   and P-2:  (C-1..5)
        //:
        //:   1 Create a 'const' 'Datetime' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const Datetime&);
        //   int generate(char *, int, const Datetime&, const Config&);
        //   int generate(string *, const Datetime&);
        //   int generate(string *, const Datetime&, const Config&);
        //   ostream generate(ostream&, const Datetime&);
        //   ostream generate(ostream&, const Datetime&, const Config&);
        //   int generateRaw(char *, const Datetime&);
        //   int generateRaw(char *, const Datetime&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'Datetime'" << endl
                          << "===================" << endl;

        typedef bdlt::Datetime TYPE;

        const int OBJLEN = Util::k_DATETIME_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int   ILINE   = DATE_DATA[ti].d_line;
            const int   YEAR    = DATE_DATA[ti].d_year;
            const int   MONTH   = DATE_DATA[ti].d_month;
            const int   DAY     = DATE_DATA[ti].d_day;
            const char *FIX = DATE_DATA[ti].d_fix;

            const bdlt::Date  DATE(YEAR, MONTH, DAY);
            const bsl::string EXPECTED_DATE(FIX);

            for (int tj = 0; tj < NUM_TIME_DATA; ++tj) {
                const int   JLINE   = TIME_DATA[tj].d_line;
                const int   HOUR    = TIME_DATA[tj].d_hour;
                const int   MIN     = TIME_DATA[tj].d_min;
                const int   SEC     = TIME_DATA[tj].d_sec;
                const int   MSEC    = TIME_DATA[tj].d_msec;
                const int   USEC    = TIME_DATA[tj].d_usec;
                const char *FIX = TIME_DATA[tj].d_fix;

                const bsl::string EXPECTED_TIME(FIX);

                const TYPE        X(YEAR,
                                    MONTH,
                                    DAY,
                                    HOUR,
                                    MIN,
                                    SEC,
                                    MSEC,
                                    USEC);
                const bsl::string BASE_EXPECTED(
                                          EXPECTED_DATE + '-' + EXPECTED_TIME);

                if (veryVerbose) {
                    T_ P_(ILINE) P_(JLINE) P_(X) P(BASE_EXPECTED)
                }

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    Config::setDefaultConfiguration(C);

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_DATETIME_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, JLINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, JLINE, k, OUTLEN,
                                    '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, JLINE, &os == &Util::generate(os, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, JLINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, JLINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'

                for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                    const int  CLINE     = CNFG_DATA[tc].d_line;
                    const int  PRECISION = CNFG_DATA[tc].d_precision;
                    const bool USEZ      = CNFG_DATA[tc].d_useZ;

                    if (veryVerbose) {
                        T_ P_(CLINE) P_(PRECISION) P(USEZ)
                    }

                    Config mC;  const Config& C = mC;
                    gg(&mC, PRECISION, USEZ);

                    // Set the default configuration to the complement of 'C'.

                    Config mDFLT;  const Config& DFLT = mDFLT;
                    gg(&mDFLT, 9 - PRECISION, !USEZ);
                    Config::setDefaultConfiguration(DFLT);

                    bsl::string EXPECTED(BASE_EXPECTED);
                    updateExpectedPerConfig(&EXPECTED,
                                            C,
                                            k_DATETIME_MAX_PRECISION);

                    const int OUTLEN = static_cast<int>(EXPECTED.length());

                    // 'generate' taking 'bufferLength'

                    for (int k = 0; k < BUFLEN; ++k) {
                        bsl::memset(buffer, '?', BUFLEN);

                        if (veryVeryVerbose) {
                            T_ T_ cout << "Length: "; P(k)
                        }

                        ASSERTV(ILINE, k, OUTLEN,
                                OUTLEN == Util::generate(buffer, k, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 k < OUTLEN ? k : OUTLEN));

                        if (k <= OUTLEN) {
                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k,
                                                     BUFLEN - k));
                        }
                        else {
                            ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                            ASSERTV(ILINE, EXPECTED, buffer,
                                    0 == bsl::memcmp(chaste,
                                                     buffer + k + 1,
                                                     BUFLEN - k - 1));
                        }
                    }

                    // 'generate' to a 'string'
                    {
                        bsl::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
                    {
                        std::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                    {
                        std::pmr::string mS("qwerty");

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generate(&mS, X, C));

                        ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                        if (veryVerbose) { P_(EXPECTED) P(mS); }
                    }
#endif

                    // 'generate' to an 'ostream'
                    {
                        bsl::ostringstream os;

                        ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                        ASSERTV(ILINE, EXPECTED, os.str(),
                                EXPECTED == os.str());

                        if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                    }

                    // 'generateRaw'
                    {
                        bsl::memset(buffer, '?', BUFLEN);

                        ASSERTV(ILINE, OUTLEN,
                                OUTLEN == Util::generateRaw(buffer, X, C));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(EXPECTED.c_str(),
                                                 buffer,
                                                 OUTLEN));

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + OUTLEN,
                                                 BUFLEN - OUTLEN));
                    }
                }  // loop over 'CNFG_DATA'
            }  // loop over 'TIME_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate( pb, X, C));
                ASSERT_FAIL(Util::generate( ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
            }
        }

      } break;
      case 2: {
        // --------------------------------------------------------------------
        // GENERATE 'Time'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Time' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 For each row 'R' in the table from P-1:  (C-1..5)
        //:
        //:   1 Create a 'const' 'Time' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const Time&);
        //   int generate(char *, int, const Time&, const Config&);
        //   int generate(string *, const Time&);
        //   int generate(string *, const Time&, const Config&);
        //   ostream generate(ostream&, const Time&);
        //   ostream generate(ostream&, const Time&, const Config&);
        //   int generateRaw(char *, const Time&);
        //   int generateRaw(char *, const Time&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'Time'" << endl
                          << "===============" << endl;

        typedef bdlt::Time TYPE;

        const int OBJLEN = Util::k_TIME_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_TIME_DATA =       NUM_DEFAULT_TIME_DATA;
        const DefaultTimeDataRow (&TIME_DATA)[NUM_TIME_DATA] =
                                                             DEFAULT_TIME_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_TIME_DATA; ++ti) {
            const int   ILINE   = TIME_DATA[ti].d_line;
            const int   HOUR    = TIME_DATA[ti].d_hour;
            const int   MIN     = TIME_DATA[ti].d_min;
            const int   SEC     = TIME_DATA[ti].d_sec;
            const int   MSEC    = TIME_DATA[ti].d_msec;
            const int   USEC    = TIME_DATA[ti].d_usec;
            const char *FIX     = TIME_DATA[ti].d_fix;

            const TYPE        X(HOUR, MIN, SEC, MSEC, USEC);
            const bsl::string BASE_EXPECTED(FIX);

            if (veryVerbose) { T_ P_(ILINE) P_(X) P(BASE_EXPECTED) }

            for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                const int  CLINE     = CNFG_DATA[tc].d_line;
                const int  PRECISION = CNFG_DATA[tc].d_precision;
                const bool USEZ      = CNFG_DATA[tc].d_useZ;

                if (veryVerbose) {
                    T_ P_(CLINE) P_(PRECISION) P(USEZ)
                }

                Config mC;  const Config& C = mC;
                gg(&mC, PRECISION, USEZ);

                Config::setDefaultConfiguration(C);

                bsl::string EXPECTED(BASE_EXPECTED);
                updateExpectedPerConfig(&EXPECTED,
                                        C,
                                        k_TIME_MAX_PRECISION);

                const int OUTLEN = static_cast<int>(EXPECTED.length());

                // 'generate' taking 'bufferLength'

                for (int k = 0; k < BUFLEN; ++k) {
                    bsl::memset(buffer, '?', BUFLEN);

                    if (veryVeryVerbose) {
                        T_ T_ cout << "Length: "; P(k)
                    }

                    ASSERTV(ILINE, k, OUTLEN,
                            OUTLEN == Util::generate(buffer, k, X));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             k < OUTLEN ? k : OUTLEN));

                    if (k <= OUTLEN) {
                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k,
                                                 BUFLEN - k));
                    }
                    else {
                        ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k + 1,
                                                 BUFLEN - k - 1));
                    }
                }

                // 'generate' to a 'string'
                {
                    bsl::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
                {
                    std::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                {
                    std::pmr::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#endif

                // 'generate' to an 'ostream'
                {
                    bsl::ostringstream os;

                    ASSERTV(ILINE, &os == &Util::generate(os, X));

                    ASSERTV(ILINE, EXPECTED, os.str(), EXPECTED == os.str());

                    if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                }

                // 'generateRaw'
                {
                    bsl::memset(buffer, '?', BUFLEN);

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generateRaw(buffer, X));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             OUTLEN));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(chaste,
                                             buffer + OUTLEN,
                                             BUFLEN - OUTLEN));
                }
            }  // loop over 'CNFG_DATA'

            for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                const int  CLINE     = CNFG_DATA[tc].d_line;
                const int  PRECISION = CNFG_DATA[tc].d_precision;
                const bool USEZ      = CNFG_DATA[tc].d_useZ;

                if (veryVerbose) {
                    T_ P_(CLINE) P_(PRECISION) P(USEZ)
                }

                Config mC;  const Config& C = mC;
                gg(&mC, PRECISION, USEZ);

                // Set the default configuration to the complement of 'C'.

                Config mDFLT;  const Config& DFLT = mDFLT;
                gg(&mDFLT, 9 - PRECISION, !USEZ);
                Config::setDefaultConfiguration(DFLT);

                bsl::string EXPECTED(BASE_EXPECTED);
                updateExpectedPerConfig(&EXPECTED,
                                        C,
                                        k_TIME_MAX_PRECISION);

                const int OUTLEN = static_cast<int>(EXPECTED.length());

                // 'generate' taking 'bufferLength'

                for (int k = 0; k < BUFLEN; ++k) {
                    bsl::memset(buffer, '?', BUFLEN);

                    if (veryVeryVerbose) {
                        T_ T_ cout << "Length: "; P(k)
                    }

                    ASSERTV(ILINE, k, OUTLEN,
                            OUTLEN == Util::generate(buffer, k, X, C));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             k < OUTLEN ? k : OUTLEN));

                    if (k <= OUTLEN) {
                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k,
                                                 BUFLEN - k));
                    }
                    else {
                        ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k + 1,
                                                 BUFLEN - k - 1));
                    }
                }

                // 'generate' to a 'string'
                {
                    bsl::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
                {
                    std::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                {
                    std::pmr::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#endif

                // 'generate' to an 'ostream'
                {
                    bsl::ostringstream os;

                    ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                    ASSERTV(ILINE, EXPECTED, os.str(), EXPECTED == os.str());

                    if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                }

                // 'generateRaw'
                {
                    bsl::memset(buffer, '?', BUFLEN);

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generateRaw(buffer, X, C));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             OUTLEN));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(chaste,
                                             buffer + OUTLEN,
                                             BUFLEN - OUTLEN));
                }
            }  // loop over 'CNFG_DATA'
        }  // loop over 'TIME_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate( pb, X, C));
                ASSERT_FAIL(Util::generate( ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
            }
        }

      } break;
      case 1: {
        // --------------------------------------------------------------------
        // GENERATE 'Date'
        //
        // Concerns:
        //: 1 The output generated by each method has the expected format and
        //:   contents.
        //:
        //: 2 When sufficient capacity is indicated, the method taking
        //:   'bufferLength' generates a null terminator.
        //:
        //: 3 Each method returns the expected value (the correct character
        //:   count or the supplied 'ostream', depending on the return type).
        //:
        //: 4 The value of the supplied object is unchanged.
        //:
        //: 5 The configuration that is in effect, whether user-supplied or the
        //:   process-wide default, has the desired affect on the output.
        //:
        //: 6 QoI: Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Using the table-driven technique, specify a set of distinct
        //:   'Date' values (one per row) and their corresponding FIX
        //:   string representations.
        //:
        //: 2 For each row 'R' in the table from P-1:  (C-1..5)
        //:
        //:   1 Create a 'const' 'Date' object, 'X', from 'R'.
        //:
        //:   2 Invoke the six methods under test on 'X' for all possible
        //:     configurations.  Also exercise the method taking 'bufferLength'
        //:     for all buffer lengths in the range '[0 .. L]', where 'L'
        //:     provides sufficient capacity for a null terminator and a few
        //:     extra characters.  For each call, verify that the generated
        //:     output matches the string from 'R' (taking the affect of the
        //:     configuration into account), a null terminator is appended when
        //:     expected, and the return value is correct.  (C-1..5)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid arguments, but not triggered for adjacent
        //:   valid ones (using the 'BSLS_ASSERTTEST_*' macros).  (C-6)
        //
        // Testing:
        //   int generate(char *, int, const Date&);
        //   int generate(char *, int, const Date&, const Config&);
        //   int generate(string *, const Date&);
        //   int generate(string *, const Date&, const Config&);
        //   ostream generate(ostream&, const Date&);
        //   ostream generate(ostream&, const Date&, const Config&);
        //   int generateRaw(char *, const Date&);
        //   int generateRaw(char *, const Date&, const Config&);
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "GENERATE 'Date'" << endl
                          << "===============" << endl;

        typedef bdlt::Date TYPE;

        const int OBJLEN = Util::k_DATE_STRLEN;
        const int BUFLEN = OBJLEN + 4;

        char buffer[BUFLEN];
        char chaste[BUFLEN];  bsl::memset(chaste, '?', BUFLEN);

        const int                  NUM_DATE_DATA =       NUM_DEFAULT_DATE_DATA;
        const DefaultDateDataRow (&DATE_DATA)[NUM_DATE_DATA] =
                                                             DEFAULT_DATE_DATA;

        const int                  NUM_CNFG_DATA =       NUM_DEFAULT_CNFG_DATA;
        const DefaultCnfgDataRow (&CNFG_DATA)[NUM_CNFG_DATA] =
                                                             DEFAULT_CNFG_DATA;

        for (int ti = 0; ti < NUM_DATE_DATA; ++ti) {
            const int   ILINE   = DATE_DATA[ti].d_line;
            const int   YEAR    = DATE_DATA[ti].d_year;
            const int   MONTH   = DATE_DATA[ti].d_month;
            const int   DAY     = DATE_DATA[ti].d_day;
            const char *FIX = DATE_DATA[ti].d_fix;

            const TYPE        X(YEAR, MONTH, DAY);
            const bsl::string BASE_EXPECTED(FIX);

            if (veryVerbose) { T_ P_(ILINE) P_(X) P(BASE_EXPECTED) }

            for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                const int  CLINE     = CNFG_DATA[tc].d_line;
                const int  PRECISION = CNFG_DATA[tc].d_precision;
                const bool USEZ      = CNFG_DATA[tc].d_useZ;

                if (veryVerbose) {
                    T_ P_(CLINE) P_(PRECISION) P(USEZ)
                }

                Config mC;  const Config& C = mC;
                gg(&mC, PRECISION, USEZ);

                Config::setDefaultConfiguration(C);

                bsl::string EXPECTED(BASE_EXPECTED);
                updateExpectedPerConfig(&EXPECTED,
                                        C,
                                        k_DATE_MAX_PRECISION);

                const int OUTLEN = static_cast<int>(EXPECTED.length());

                // 'generate' taking 'bufferLength'

                for (int k = 0; k < BUFLEN; ++k) {
                    bsl::memset(buffer, '?', BUFLEN);

                    if (veryVeryVerbose) {
                        T_ T_ cout << "Length: "; P(k)
                    }

                    ASSERTV(ILINE, k, OUTLEN,
                            OUTLEN == Util::generate(buffer, k, X));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             k < OUTLEN ? k : OUTLEN));

                    if (k <= OUTLEN) {
                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k,
                                                 BUFLEN - k));
                    }
                    else {
                        ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k + 1,
                                                 BUFLEN - k - 1));
                    }
                }

                // 'generate' to a 'string'
                {
                    bsl::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
                {
                    std::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                {
                    std::pmr::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN, OUTLEN == Util::generate(&mS, X));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#endif

                // 'generate' to an 'ostream'
                {
                    bsl::ostringstream os;

                    ASSERTV(ILINE, &os == &Util::generate(os, X));

                    ASSERTV(ILINE, EXPECTED, os.str(), EXPECTED == os.str());

                    if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                }

                // 'generateRaw'
                {
                    bsl::memset(buffer, '?', BUFLEN);

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generateRaw(buffer, X));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             OUTLEN));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(chaste,
                                             buffer + OUTLEN,
                                             BUFLEN - OUTLEN));
                }
            }  // loop over 'CNFG_DATA'

            for (int tc = 0; tc < NUM_CNFG_DATA; ++tc) {
                const int  CLINE     = CNFG_DATA[tc].d_line;
                const int  PRECISION = CNFG_DATA[tc].d_precision;
                const bool USEZ      = CNFG_DATA[tc].d_useZ;

                if (veryVerbose) {
                    T_ P_(CLINE) P_(PRECISION) P(USEZ)
                }

                Config mC;  const Config& C = mC;
                gg(&mC, PRECISION, USEZ);

                // Set the default configuration to the complement of 'C'.

                Config mDFLT;  const Config& DFLT = mDFLT;
                gg(&mDFLT, 9 - PRECISION, !USEZ);
                Config::setDefaultConfiguration(DFLT);

                bsl::string EXPECTED(BASE_EXPECTED);
                updateExpectedPerConfig(&EXPECTED,
                                        C,
                                        k_DATE_MAX_PRECISION);

                const int OUTLEN = static_cast<int>(EXPECTED.length());

                // 'generate' taking 'bufferLength'

                for (int k = 0; k < BUFLEN; ++k) {
                    bsl::memset(buffer, '?', BUFLEN);

                    if (veryVeryVerbose) {
                        T_ T_ cout << "Length: "; P(k)
                    }

                    ASSERTV(ILINE, k, OUTLEN,
                            OUTLEN == Util::generate(buffer, k, X, C));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             k < OUTLEN ? k : OUTLEN));

                    if (k <= OUTLEN) {
                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k,
                                                 BUFLEN - k));
                    }
                    else {
                        ASSERTV(ILINE, k, OUTLEN, '\0' == buffer[OUTLEN]);

                        ASSERTV(ILINE, EXPECTED, buffer,
                                0 == bsl::memcmp(chaste,
                                                 buffer + k + 1,
                                                 BUFLEN - k - 1));
                    }
                }

                // 'generate' to a 'string'
                {
                    bsl::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
                {
                    std::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                {
                    std::pmr::string mS("qwerty");

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generate(&mS, X, C));

                    ASSERTV(ILINE, EXPECTED, mS, EXPECTED == mS);

                    if (veryVerbose) { P_(EXPECTED) P(mS); }
                }
#endif

                // 'generate' to an 'ostream'
                {
                    bsl::ostringstream os;

                    ASSERTV(ILINE, &os == &Util::generate(os, X, C));

                    ASSERTV(ILINE, EXPECTED, os.str(), EXPECTED == os.str());

                    if (veryVerbose) { P_(EXPECTED) P(os.str()); }
                }

                // 'generateRaw'
                {
                    bsl::memset(buffer, '?', BUFLEN);

                    ASSERTV(ILINE, OUTLEN,
                            OUTLEN == Util::generateRaw(buffer, X, C));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(EXPECTED.c_str(),
                                             buffer,
                                             OUTLEN));

                    ASSERTV(ILINE, EXPECTED, buffer,
                            0 == bsl::memcmp(chaste,
                                             buffer + OUTLEN,
                                             BUFLEN - OUTLEN));
                }
            }  // loop over 'CNFG_DATA'
        }  // loop over 'DATE_DATA'

        if (verbose) cout << "\nNegative Testing." << endl;
        {
            bsls::AssertTestHandlerGuard hG;

            const Config C;

            if (verbose) cout << "\t'generate'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X));
                ASSERT_SAFE_FAIL(Util::generate(     0, OBJLEN, X));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X));
                ASSERT_SAFE_FAIL(Util::generate(buffer,     -1, X));

                ASSERT_SAFE_PASS(Util::generate(buffer, OBJLEN, X, C));
                ASSERT_FAIL(Util::generate(     0, OBJLEN, X, C));

                ASSERT_SAFE_PASS(Util::generate(buffer,      0, X, C));
                ASSERT_FAIL(Util::generate(buffer,     -1, X, C));

                bsl::string        mB("qwerty"), *pb = 0;
                std::string        mS("qwerty"), *ps = 0;
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                std::pmr::string   mP("qwerty"), *pp = 0;
#endif

                ASSERT_PASS(Util::generate(&mB, X));
                ASSERT_PASS(Util::generate(&mS, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X));
#endif
                ASSERT_FAIL(Util::generate( pb, X));
                ASSERT_FAIL(Util::generate( ps, X));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X));
#endif

                ASSERT_PASS(Util::generate(&mB, X, C));
                ASSERT_PASS(Util::generate(&mS, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_PASS(Util::generate(&mP, X, C));
#endif
                ASSERT_FAIL(Util::generate( pb, X, C));
                ASSERT_FAIL(Util::generate( ps, X, C));
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                ASSERT_FAIL(Util::generate( pp, X, C));
#endif
            }

            if (verbose) cout << "\t'generateRaw'" << endl;
            {
                const TYPE X;
                char       buffer[OBJLEN];

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X));
                ASSERT_SAFE_FAIL(Util::generateRaw(     0, X));

                ASSERT_SAFE_PASS(Util::generateRaw(buffer, X, C));
                ASSERT_FAIL(Util::generateRaw(     0, X, C));
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
// Copyright 2016 Bloomberg Finance L.P.
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
