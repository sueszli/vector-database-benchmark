// bdlt_datetimeinterval.cpp                                          -*-C++-*-
#include <bdlt_datetimeinterval.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdlt_datetimeinterval_cpp,"$Id$ $CSID$")

#include <bdlb_bitutil.h>

#include <bslim_printer.h>

#include <bslmf_assert.h>

#include <bsls_libraryfeatures.h>
#include <bsls_platform.h>

#include <bsl_ostream.h>

#include <bsl_cmath.h>     // 'modf, 'fabs', 'isinf', 'isnan'
#include <bsl_cstdio.h>    // 'sprintf'

namespace {
namespace u {

using namespace BloombergLP;

typedef bsls::Types::Int64  Int64;
typedef bsls::Types::Uint64 Uint64;

typedef bdlt::TimeUnitRatio TimeUnitRatio;

// Assert fundamental assumptions made in the implementation.

BSLMF_ASSERT(-3 / 2 == -1);
BSLMF_ASSERT(-5 % 4 == -1);

const Int64 k_MAX_INT64       = static_cast<bsls::Types::Int64>(
                                                 ~static_cast<Uint64>(0) >> 1);
const Int64 k_MIN_INT64       = ~k_MAX_INT64;

const Int64 k_MAX_INT         = static_cast<Int64>(
                                               ~static_cast<unsigned>(0) >> 1);
const Int64 k_MIN_INT         = ~k_MAX_INT;

const Int64 k_HOURS_CEILING   = (k_MAX_INT + 1) * TimeUnitRatio::k_H_PER_D - 1;
const Int64 k_HOURS_FLOOR     = (k_MIN_INT - 1) * TimeUnitRatio::k_H_PER_D + 1;
const Int64 k_MINUTES_CEILING = (k_MAX_INT + 1) * TimeUnitRatio::k_M_PER_D - 1;
const Int64 k_MINUTES_FLOOR   = (k_MIN_INT - 1) * TimeUnitRatio::k_M_PER_D + 1;
const Int64 k_SECONDS_CEILING = (k_MAX_INT + 1) * TimeUnitRatio::k_S_PER_D - 1;
const Int64 k_SECONDS_FLOOR   = (k_MIN_INT - 1) * TimeUnitRatio::k_S_PER_D + 1;
const Int64 k_MILLISECONDS_CEILING
                              = (k_MAX_INT + 1) * TimeUnitRatio::k_MS_PER_D -1;
const Int64 k_MILLISECONDS_FLOOR
                              = (k_MIN_INT - 1) * TimeUnitRatio::k_MS_PER_D +1;

const double maxInt64AsDouble = static_cast<double>(k_MAX_INT64);
const double minInt64AsDouble = static_cast<double>(k_MIN_INT64);

// HELPER FUNCTIONS
int printToBufferFormatted(char       *result,
                           int         numBytes,
                           const char *spec,
                           int         day,
                           int         hour,
                           int         minute,
                           int         second,
                           int         microsecond,
                           int         fractionalSecondPrecision)
{
#if defined(BSLS_LIBRARYFEATURES_HAS_C99_SNPRINTF)
    return 0 == fractionalSecondPrecision
           ? bsl::snprintf(result,
                           numBytes,
                           spec,
                           day,
                           hour,
                           minute,
                           second)
           : bsl::snprintf(result,
                           numBytes,
                           spec,
                           day,
                           hour,
                           minute,
                           second,
                           microsecond);

#elif defined(BSLS_PLATFORM_CMP_MSVC)
    // Windows uses a different variant of snprintf that does not necessarily
    // null-terminate and returns -1 (or 'numBytes') on overflow.

    int rc;

    if (0 == fractionalSecondPrecision) {
        rc = _snprintf(result,
                       numBytes,
                       spec,
                       day,
                       hour,
                       minute,
                       second);
    }
    else {
        rc = _snprintf(result,
                       numBytes,
                       spec,
                       day,
                       hour,
                       minute,
                       second,
                       microsecond);
    }

    if (0 > rc || rc == numBytes) {
        if (numBytes > 0) {
            result[numBytes - 1] = '\0';  // Make sure to null-terminate on
                                          // overflow.
        }

        // Need to determine the length that would have been printed without
        // overflow.

        if (0 == fractionalSecondPrecision) {
            //   '_' hh: mm: ss
            rc =  1 + 3 + 3 + 2;
        }
        else {
            //   '_' hh: mm: ss. mmm uuu
            rc =  1 + 3 + 3 + 3 + fractionalSecondPrecision;
        }

        if (-10 < day && 10 > day) {
            rc += 1;
        }
        else if (-100 < day && 100 > day) {
            rc += 2;
        }
        else if (-1000 < day && 1000 > day) {
            rc += 3;
        }
        else if (-10000 < day && 10000 > day) {
            rc += 4;
        }
        else if (-100000 < day && 100000 > day) {
            rc += 5;
        }
        else if (-1000000 < day && 1000000 > day) {
            rc += 6;
        }
        else if (-10000000 < day && 10000000 > day) {
            rc += 7;
        }
        else if (-100000000 < day && 100000000 > day) {
            rc += 8;
        }
        else if (-1000000000 < day && 1000000000 > day) {
            rc += 9;
        }
        else {
            rc += 10;
        }
    }

    return rc;
#else
# error THIS CONFIGURATION DOES NOT SUPPORT 'snprintf'
#endif
}

#ifdef BSLS_LIBRARYFEATURES_HAS_C99_FP_CLASSIFY
# define U_IS_INF(x)    bsl::isinf(x)
# define U_IS_NAN(x)    bsl::isnan(x)
#else
# define U_IS_INF(x)    false
# define U_IS_NAN(x)    false
#endif

}  // close namespace u
}  // close unnamed namespace

namespace BloombergLP {
namespace bdlt {

                          // ----------------------
                          // class DatetimeInterval
                          // ----------------------

// PRIVATE MANIPULATORS
void DatetimeInterval::assign(bsls::Types::Int64 days,
                              bsls::Types::Int64 microseconds)
{
    const Int64 micDays = microseconds / TimeUnitRatio::k_US_PER_D;

    BSLS_ASSERT(!(days > 0 && micDays > 0) || micDays <= u::k_MAX_INT - days);
    BSLS_ASSERT(!(days < 0 && micDays < 0) || u::k_MIN_INT - days <= micDays);

    days         += micDays;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    if (days > 0 && microseconds < 0) {
        --days;
        microseconds += TimeUnitRatio::k_US_PER_D;
    }
    else if (days < 0 && microseconds > 0) {
        ++days;
        microseconds -= TimeUnitRatio::k_US_PER_D;
    }

    BSLS_ASSERT(days <= u::k_MAX_INT);
    BSLS_ASSERT(days >= u::k_MIN_INT);

    d_days         = static_cast<int>(days);
    d_microseconds = microseconds;
}

int DatetimeInterval::assignIfValid(bsls::Types::Int64 days,
                                    bsls::Types::Int64 microseconds)
{
    const Int64 micDays = microseconds / TimeUnitRatio::k_US_PER_D;

    if   ((days > 0 && micDays > 0 && u::k_MAX_INT - days < micDays)
       || (days < 0 && micDays < 0 && micDays < u::k_MIN_INT - days)) {
        return -1;                                                    // RETURN
    }

    days         += micDays;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    if (days > 0 && microseconds < 0) {
        --days;
        microseconds += TimeUnitRatio::k_US_PER_D;
    }
    else if (days < 0 && microseconds > 0) {
        ++days;
        microseconds -= TimeUnitRatio::k_US_PER_D;
    }

    if   (days < u::k_MIN_INT || u::k_MAX_INT < days) {
        return -1;                                                    // RETURN
    }

    d_days         = static_cast<int>(days);
    d_microseconds = microseconds;

    return 0;
}

// CLASS METHODS
bool DatetimeInterval::isValid(int                days,
                               bsls::Types::Int64 hours,
                               bsls::Types::Int64 minutes,
                               bsls::Types::Int64 seconds,
                               bsls::Types::Int64 milliseconds,
                               bsls::Types::Int64 microseconds)
{
    if   (u::k_HOURS_CEILING        < hours
       || u::k_MINUTES_CEILING      < minutes
       || u::k_SECONDS_CEILING      < seconds
       || u::k_MILLISECONDS_CEILING < milliseconds
       || u::k_HOURS_FLOOR          > hours
       || u::k_MINUTES_FLOOR        > minutes
       || u::k_SECONDS_FLOOR        > seconds
       || u::k_MILLISECONDS_FLOOR   > milliseconds) {
        return false;                                                 // RETURN
    }

    bsls::Types::Int64 d = static_cast<bsls::Types::Int64>(days)
                         + hours        / TimeUnitRatio::k_H_PER_D
                         + minutes      / TimeUnitRatio::k_M_PER_D
                         + seconds      / TimeUnitRatio::k_S_PER_D
                         + milliseconds / TimeUnitRatio::k_MS_PER_D
                         + microseconds / TimeUnitRatio::k_US_PER_D;

    hours        %= TimeUnitRatio::k_H_PER_D;
    minutes      %= TimeUnitRatio::k_M_PER_D;
    seconds      %= TimeUnitRatio::k_S_PER_D;
    milliseconds %= TimeUnitRatio::k_MS_PER_D;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    bsls::Types::Int64 us = hours        * TimeUnitRatio::k_US_PER_H
                          + minutes      * TimeUnitRatio::k_US_PER_M
                          + seconds      * TimeUnitRatio::k_US_PER_S
                          + milliseconds * TimeUnitRatio::k_US_PER_MS
                          + microseconds;

    d += us / TimeUnitRatio::k_US_PER_D;

    if   (d < u::k_MIN_INT || u::k_MAX_INT < d) {
        return false;                                                 // RETURN
    }

    return true;
}

// MANIPULATORS
void DatetimeInterval::setInterval(int                days,
                                   bsls::Types::Int64 hours,
                                   bsls::Types::Int64 minutes,
                                   bsls::Types::Int64 seconds,
                                   bsls::Types::Int64 milliseconds,
                                   bsls::Types::Int64 microseconds)
{
    bsls::Types::Int64 d = static_cast<bsls::Types::Int64>(days)
                         + hours        / TimeUnitRatio::k_H_PER_D
                         + minutes      / TimeUnitRatio::k_M_PER_D
                         + seconds      / TimeUnitRatio::k_S_PER_D
                         + milliseconds / TimeUnitRatio::k_MS_PER_D
                         + microseconds / TimeUnitRatio::k_US_PER_D;

    hours        %= TimeUnitRatio::k_H_PER_D;
    minutes      %= TimeUnitRatio::k_M_PER_D;
    seconds      %= TimeUnitRatio::k_S_PER_D;
    milliseconds %= TimeUnitRatio::k_MS_PER_D;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    bsls::Types::Int64 us = hours        * TimeUnitRatio::k_US_PER_H
                          + minutes      * TimeUnitRatio::k_US_PER_M
                          + seconds      * TimeUnitRatio::k_US_PER_S
                          + milliseconds * TimeUnitRatio::k_US_PER_MS
                          + microseconds;

    assign(d, us);
}

int DatetimeInterval::setIntervalIfValid(int                days,
                                         bsls::Types::Int64 hours,
                                         bsls::Types::Int64 minutes,
                                         bsls::Types::Int64 seconds,
                                         bsls::Types::Int64 milliseconds,
                                         bsls::Types::Int64 microseconds)
{
    if   (u::k_HOURS_CEILING        < hours
       || u::k_MINUTES_CEILING      < minutes
       || u::k_SECONDS_CEILING      < seconds
       || u::k_MILLISECONDS_CEILING < milliseconds
       || u::k_HOURS_FLOOR          > hours
       || u::k_MINUTES_FLOOR        > minutes
       || u::k_SECONDS_FLOOR        > seconds
       || u::k_MILLISECONDS_FLOOR   > milliseconds) {
        return -1;                                                    // RETURN
    }

    bsls::Types::Int64 d = static_cast<bsls::Types::Int64>(days)
                         + hours        / TimeUnitRatio::k_H_PER_D
                         + minutes      / TimeUnitRatio::k_M_PER_D
                         + seconds      / TimeUnitRatio::k_S_PER_D
                         + milliseconds / TimeUnitRatio::k_MS_PER_D
                         + microseconds / TimeUnitRatio::k_US_PER_D;

    hours        %= TimeUnitRatio::k_H_PER_D;
    minutes      %= TimeUnitRatio::k_M_PER_D;
    seconds      %= TimeUnitRatio::k_S_PER_D;
    milliseconds %= TimeUnitRatio::k_MS_PER_D;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    bsls::Types::Int64 us = hours        * TimeUnitRatio::k_US_PER_H
                          + minutes      * TimeUnitRatio::k_US_PER_M
                          + seconds      * TimeUnitRatio::k_US_PER_S
                          + milliseconds * TimeUnitRatio::k_US_PER_MS
                          + microseconds;

    return assignIfValid(d, us);
}

void DatetimeInterval::setTotalSecondsFromDouble(double seconds)
{
    BSLS_ASSERT(!U_IS_INF(seconds));
    BSLS_ASSERT(!U_IS_NAN(seconds));

    double wholeDays;
    bsl::modf(seconds / TimeUnitRatio::k_S_PER_D, &wholeDays);
        // Ignoring fractional part to maintain as much accuracy from
        // 'seconds' as possible.

    BSLS_ASSERT(bsl::fabs(wholeDays) <= u::maxInt64AsDouble);
        // Failing for bsl::numeric_limits<bsls::Types::Int64>::min() is OK
        // here, because wholeDays has to fit into 32 bits.  Here we're just
        // checking that we are not about to run into UB when casting to
        // bsls::Types::Int64.

    volatile double microseconds =
                   (seconds - wholeDays * TimeUnitRatio::k_S_PER_D)
                   * TimeUnitRatio::k_US_PER_S + copysign(0.5, seconds);
        // On GCC x86 platforms we have to force copying a floating-point value
        // to memory using 'volatile' type qualifier to round-down the value
        // stored in x87 unit register.

    int rc = assignIfValid(static_cast<bsls::Types::Int64>(wholeDays),
                           static_cast<bsls::Types::Int64>(microseconds));

    BSLS_ASSERT(0 == rc);  (void)rc;
}

int DatetimeInterval::setTotalSecondsFromDoubleIfValid(double seconds)
{
    if (U_IS_INF(seconds) || U_IS_NAN(seconds)) {
        return -1;                                                    // RETURN
    }

    double wholeDays;
    bsl::modf(seconds / TimeUnitRatio::k_S_PER_D, &wholeDays);
        // Ignoring fractional part to maintain as much accuracy from
        // 'seconds' as possible.

    if (u::maxInt64AsDouble < wholeDays || wholeDays < u::minInt64AsDouble) {
        return -1;                                                    // RETURN
    }

    const double secondsRemainder =  seconds -
                                          wholeDays * TimeUnitRatio::k_S_PER_D;

    // On GCC x86 platforms we have to force copying a floating-point value to
    // memory using 'volatile' type qualifier to round-down the value stored in
    // x87 unit register.

    volatile double microseconds =
         secondsRemainder * TimeUnitRatio::k_US_PER_S + copysign(0.5, seconds);

    return assignIfValid(static_cast<bsls::Types::Int64>(wholeDays),
                         static_cast<bsls::Types::Int64>(microseconds));
}

DatetimeInterval& DatetimeInterval::addInterval(
                                               int                days,
                                               bsls::Types::Int64 hours,
                                               bsls::Types::Int64 minutes,
                                               bsls::Types::Int64 seconds,
                                               bsls::Types::Int64 milliseconds,
                                               bsls::Types::Int64 microseconds)
{
    bsls::Types::Int64 d = static_cast<bsls::Types::Int64>(days)
                         + hours        / TimeUnitRatio::k_H_PER_D
                         + minutes      / TimeUnitRatio::k_M_PER_D
                         + seconds      / TimeUnitRatio::k_S_PER_D
                         + milliseconds / TimeUnitRatio::k_MS_PER_D
                         + microseconds / TimeUnitRatio::k_US_PER_D;

    hours        %= TimeUnitRatio::k_H_PER_D;
    minutes      %= TimeUnitRatio::k_M_PER_D;
    seconds      %= TimeUnitRatio::k_S_PER_D;
    milliseconds %= TimeUnitRatio::k_MS_PER_D;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    bsls::Types::Int64 us = hours        * TimeUnitRatio::k_US_PER_H
                          + minutes      * TimeUnitRatio::k_US_PER_M
                          + seconds      * TimeUnitRatio::k_US_PER_S
                          + milliseconds * TimeUnitRatio::k_US_PER_MS
                          + microseconds;

    assign(static_cast<bsls::Types::Int64>(d_days) + d,
           d_microseconds + us);

    return *this;
}

int DatetimeInterval::addIntervalIfValid(int                days,
                                         bsls::Types::Int64 hours,
                                         bsls::Types::Int64 minutes,
                                         bsls::Types::Int64 seconds,
                                         bsls::Types::Int64 milliseconds,
                                         bsls::Types::Int64 microseconds)
{
    bsls::Types::Int64 d = static_cast<bsls::Types::Int64>(days)
                         + hours        / TimeUnitRatio::k_H_PER_D
                         + minutes      / TimeUnitRatio::k_M_PER_D
                         + seconds      / TimeUnitRatio::k_S_PER_D
                         + milliseconds / TimeUnitRatio::k_MS_PER_D
                         + microseconds / TimeUnitRatio::k_US_PER_D;

    hours        %= TimeUnitRatio::k_H_PER_D;
    minutes      %= TimeUnitRatio::k_M_PER_D;
    seconds      %= TimeUnitRatio::k_S_PER_D;
    milliseconds %= TimeUnitRatio::k_MS_PER_D;
    microseconds %= TimeUnitRatio::k_US_PER_D;

    bsls::Types::Int64 us = hours        * TimeUnitRatio::k_US_PER_H
                          + minutes      * TimeUnitRatio::k_US_PER_M
                          + seconds      * TimeUnitRatio::k_US_PER_S
                          + milliseconds * TimeUnitRatio::k_US_PER_MS
                          + microseconds;

    // Note that 'k_MAX_INT' and 'k_MIN_INT' are, unlike 'INT_MAX' or
    // 'INT_MIN', 64-bit entities.

    if   ((d_days > 0 && d > 0 && u::k_MAX_INT - d_days < d)
       || (d_days < 0 && d < 0 && d < u::k_MIN_INT - d_days)
       || (d_microseconds > 0 && us > 0 && u::k_MAX_INT64-d_microseconds < us)
       || (d_microseconds < 0 && us < 0 &&
                                       us < u::k_MIN_INT64 - d_microseconds)) {
        return -1;                                                    // RETURN
    }

    return assignIfValid(static_cast<bsls::Types::Int64>(d_days) + d,
                         d_microseconds + us);
}

// ACCESSORS

                                  // Aspects

int DatetimeInterval::printToBuffer(char *result,
                                    int   numBytes,
                                    int   fractionalSecondPrecision) const
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(0 <= numBytes);
    BSLS_ASSERT(0 <= fractionalSecondPrecision     );
    BSLS_ASSERT(     fractionalSecondPrecision <= 6);

    int d  = days();
    int h  = hours();
    int m  = minutes();
    int s  = seconds();
    int ms = milliseconds();
    int us = microseconds();

    // Values with a non-negative day component will have the sign
    // "pre-printed".

    int printedLength = 0;

    if (0 <= d_days) {
        if (1 < numBytes) {
            if (0 <= d_microseconds) {
                *result++ = '+';
            }
            else {
                *result++ = '-';
            }
            --numBytes;
        }
        printedLength = 1;
    }

    // Invert the time component values when the value is negative.

    if (0 > d_days || 0 > d_microseconds) {
        h  = -h;
        m  = -m;
        s  = -s;
        ms = -ms;
        us = -us;
    }

    int value;

    switch (fractionalSecondPrecision) {
      case 0: {
        char spec[] = "%d_%02d:%02d:%02d";

        // Add one for the sign.

        return u::printToBufferFormatted(result,
                                         numBytes,
                                         spec,
                                         d,
                                         h,
                                         m,
                                         s,
                                         0,
                                         0) + printedLength;          // RETURN
      } break;
      case 1: {
        value = ms / 100;
      } break;
      case 2: {
        value = ms / 10 ;
      } break;
      case 3: {
        value = ms;
      } break;
      case 4: {
        value = ms * 10   + us / 100;
      } break;
      case 5: {
        value = ms * 100  + us / 10;
      } break;
      default: {
        value = ms * 1000 + us;
      } break;
    }

    char spec[] = "%d_%02d:%02d:%02d.%0Xd";

    const int PRECISION_INDEX = sizeof spec - 3;

    spec[PRECISION_INDEX] = static_cast<char>('0' + fractionalSecondPrecision);

    // Add one for the sign.

    return u::printToBufferFormatted(
                                    result,
                                    numBytes,
                                    spec,
                                    d,
                                    h,
                                    m,
                                    s,
                                    value,
                                    fractionalSecondPrecision) + printedLength;
}

bsl::ostream& DatetimeInterval::print(bsl::ostream& stream,
                                      int           level,
                                      int           spacesPerLevel) const
{
    // Format the output to a buffer first instead of inserting into 'stream'
    // directly to improve performance and in case the caller has done
    // something like:
    //..
    //  os << bsl::setw(20) << myInterval;
    //..
    // The user-specified width will be effective when 'buffer' is written to
    // the 'stream' (below).

    const int k_BUFFER_SIZE = 32;
    char      buffer[k_BUFFER_SIZE];

    int rc = printToBuffer(buffer,
                           k_BUFFER_SIZE,
                           k_DEFAULT_FRACTIONAL_SECOND_PRECISION);

    (void)rc;

    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start(true);    // 'true' -> suppress '['
    stream << buffer;
    printer.end(true);      // 'true' -> suppress ']'

    return stream;
}

#ifndef BDE_OMIT_INTERNAL_DEPRECATED  // BDE2.22

// DEPRECATED METHODS
bsl::ostream& DatetimeInterval::streamOut(bsl::ostream& stream) const
{
    return stream << *this;
}

#endif // BDE_OMIT_INTERNAL_DEPRECATED -- BDE2.22

}  // close package namespace
}  // close enterprise namespace

// ----------------------------------------------------------------------------
// Copyright 2017 Bloomberg Finance L.P.
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
