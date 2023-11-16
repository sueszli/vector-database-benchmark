// bdlt_fixutil.cpp                                                   -*-C++-*-
#include <bdlt_fixutil.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdlt_fixutil_cpp,"$Id$ $CSID$")

#include <bdlt_date.h>
#include <bdlt_datetime.h>
#include <bdlt_datetimeinterval.h>
#include <bdlt_datetimetz.h>
#include <bdlt_datetz.h>
#include <bdlt_time.h>
#include <bdlt_timetz.h>

#include <bslmf_assert.h>
#include <bslmf_issame.h>

#include <bsl_algorithm.h>
#include <bsl_cctype.h>
#include <bsl_cstring.h>

namespace {
namespace u {

using namespace BloombergLP;
using namespace BloombergLP::bdlt;

                                  // ===========
                                  // struct Impl
                                  // ===========

class Impl {
    // This 'class' is private to this component and is not to be referred to
    // outside this component.

    // PRIVATE TYPES
    typedef BloombergLP::bdlt::FixUtil Util;

    template <class TYPE>
    struct IsString {
        static const bool value = bsl::is_same<TYPE, bsl::string>::value
                               || bsl::is_same<TYPE, std::string>::value
    #ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
                               || bsl::is_same<TYPE, std::pmr::string>::value
    #endif
        ;
    };

    // FRIENDS
    friend struct BloombergLP::bdlt::FixUtil;

    // In all cases below, the 'STRING' template parameter is to be either
    // 'bsl::string', 'std::string', or 'std::pmr::string'.

    // CLASS METHODS
    template <class STRING>
    static int generate(STRING                      *string,
                        const Date&                  object,
                        const FixUtilConfiguration&  configuration);
    template <class STRING>
    static int generate(STRING                      *string,
                        const Time&                  object,
                        const FixUtilConfiguration&  configuration);
    template <class STRING>
    static int generate(STRING                      *string,
                        const Datetime&              object,
                        const FixUtilConfiguration&  configuration);
    template <class STRING>
    static int generate(STRING                      *string,
                        const DateTz&                object,
                        const FixUtilConfiguration&  configuration);
    template <class STRING>
    static int generate(STRING                      *string,
                        const TimeTz&                object,
                        const FixUtilConfiguration&  configuration);
    template <class STRING>
    static int generate(STRING                      *string,
                        const DatetimeTz&            object,
                        const FixUtilConfiguration&  configuration);
        // Load the FIX representation of the specified 'object' into the
        // specified 'string'.  Specify a 'configuration' to affect the format
        // of the generated string.  'STRING' must be 'bsl::string',
        // 'std::string', or 'std::pmr::string'.  Return the number of
        // characters in the formatted string.  The previous contents of
        // 'string' (if any) are discarded.
};

                                // -----------
                                // struct Impl
                                // -----------

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const Date&                  object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_DATE_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);
    BSLS_ASSERT(Util::k_DATE_STRLEN >= len);

    string->resize(len);

    return len;
}

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const Time&                  object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_TIME_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);

    BSLS_ASSERT(Util::k_TIME_STRLEN >= len);

    string->resize(len);

    return len;
}

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const Datetime&              object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_DATETIME_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);

    BSLS_ASSERT(Util::k_DATETIME_STRLEN >= len);

    string->resize(len);

    return len;
}

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const DateTz&                object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_DATETZ_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);

    BSLS_ASSERT(Util::k_DATETZ_STRLEN >= len);

    string->resize(len);

    return len;
}

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const TimeTz&                object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_TIMETZ_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);

    BSLS_ASSERT(Util::k_TIMETZ_STRLEN >= len);

    string->resize(len);

    return len;
}

template <class STRING>
inline
int Impl::generate(STRING                      *string,
                   const DatetimeTz&            object,
                   const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    BSLMF_ASSERT(IsString<STRING>::value);

    string->resize(Util::k_DATETIMETZ_STRLEN);

    const int len = Util::generateRaw(&(*string)[0], object, configuration);

    BSLS_ASSERT(Util::k_DATETIMETZ_STRLEN >= len);

    string->resize(len);

    return len;
}

// STATIC HELPER FUNCTIONS

int asciiToInt(const char **nextPos,
               int         *result,
               const char  *begin,
               const char  *end)
    // Convert the (unsigned) ASCII decimal integer starting at the specified
    // 'begin' and ending immediately before the specified 'end' into its
    // corresponding 'int' value, load the value into the specified 'result',
    // and set the specified '*nextPos' to 'end'.  Return 0 on success, and a
    // non-zero value (with no effect) otherwise.  All characters in the range
    // '[begin .. end)' must be decimal digits.  The behavior is undefined
    // unless 'begin < end' and the parsed value does not exceed 'INT_MAX'.
{
    BSLS_ASSERT(nextPos);
    BSLS_ASSERT(result);
    BSLS_ASSERT(begin);
    BSLS_ASSERT(end);
    BSLS_ASSERT(begin < end);

    int tmp = 0;

    while (begin < end) {
        if (!isdigit(*begin)) {
            return -1;                                                // RETURN
        }

        tmp *= 10;
        tmp += *begin - '0';

        ++begin;
    }

    *result  = tmp;
    *nextPos = end;

    return 0;
}

int parseDate(const char **nextPos,
              Date        *date,
              const char  *begin,
              const char  *end)
    // Parse the date, represented in the "YYYYMMDD" FIX extended format, from
    // the string starting at the specified 'begin' and ending before the
    // specified 'end', load into the specified 'date' the parsed values, and
    // set the specified '*nextPos' to the location one past the last parsed
    // character.  Return 0 on success, and a non-zero value (with no effect on
    // '*nextPos') otherwise.  The behavior is undefined unless 'begin <= end'.
    // Note that successfully parsing a date before 'end' is reached is not an
    // error.
{
    BSLS_ASSERT(nextPos);
    BSLS_ASSERT(date);
    BSLS_ASSERT(begin);
    BSLS_ASSERT(end);
    BSLS_ASSERT(begin <= end);

    const char *p = begin;

    enum { k_MINIMUM_LENGTH = sizeof "YYYYMMDD" - 1 };

    if (end - p < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    int year;
    if (0 != u::asciiToInt(&p, &year, p, p + 4)) {
        return -1;                                                    // RETURN
    }

    int month;
    if (0 != u::asciiToInt(&p, &month, p, p + 2)) {
        return -1;                                                    // RETURN
    }

    int day;
    if (0 != u::asciiToInt(&p, &day, p, p + 2)) {
        return -1;                                                    // RETURN
    }

    if (0 != date->setYearMonthDayIfValid(year, month, day)) {
        return -1;                                                    // RETURN
    }

    *nextPos = p;

    return 0;
}

int parseFractionalSecond(const char **nextPos,
                          int         *microsecond,
                          const char  *begin,
                          const char  *end)
    // Parse the fractional second starting at the specified 'begin' and ending
    // before the specified 'end', load into the specified 'microsecond' the
    // parsed value (in microseconds) rounded to the closest microsecond, and
    // set the specified '*nextPos' to the location one past the last parsed
    // character (necessarily a decimal digit).  Return 0 on success, and a
    // non-zero value (with no effect) otherwise.  There must be at least one
    // digit, only the first 7 digits are significant, and all digits beyond
    // the first 7 are parsed but ignored.  The behavior is undefined unless
    // 'begin <= end'.  Note that successfully parsing a fractional second
    // before 'end' is reached is not an error.
{
    BSLS_ASSERT(nextPos);
    BSLS_ASSERT(microsecond);
    BSLS_ASSERT(begin);
    BSLS_ASSERT(end);
    BSLS_ASSERT(begin <= end);

    const char *p = begin;

    // There must be at least one digit.

    if (p == end || !isdigit(*p)) {
        return -1;                                                    // RETURN
    }

    // Only the first 7 digits are significant.

    const char *endSignificant = bsl::min(end, p + 7);

    int tmp    = 0;
    int factor = 10000000;  // Since the result is in microseconds, we have to
                            // adjust it according to how many digits are
                            // present.

    do {
        tmp    *= 10;
        tmp    += *p - '0';
        factor /= 10;
    } while (++p < endSignificant && isdigit(*p));

    tmp = tmp * factor;

    // round

    tmp = (tmp + 5) / 10;

    // Skip and ignore all digits beyond the first 7, if any.

    while (p < end && isdigit(*p)) {
        ++p;
    }

    *nextPos     = p;
    *microsecond = tmp;

    return 0;
}

int parseTimezoneOffset(const char **nextPos,
                        int         *minuteOffset,
                        const char  *begin,
                        const char  *end)
    // Parse the timezone offset, represented in the "Z|(+|-])hh{:mm}" FIX
    // extended format, from the string starting at the specified 'begin' and
    // ending before the specified 'end', load into the specified
    // 'minuteOffset' the indicated offset (in minutes) from UTC, and set the
    // specified '*nextPos' to the location one past the last parsed character.
    // Return 0 on success, and a non-zero value (with no effect on '*nextPos')
    // otherwise.  The behavior is undefined unless 'begin <= end'.  Note that
    // successfully parsing a timezone offset before 'end' is reached is not an
    // error.
{
    BSLS_ASSERT(nextPos);
    BSLS_ASSERT(minuteOffset);
    BSLS_ASSERT(begin);
    BSLS_ASSERT(end);
    BSLS_ASSERT(begin <= end);

    const char *p = begin;

    if (p >= end) {
        return -1;                                                    // RETURN
    }

    const char sign = *p++;  // store and skip '(+|-|Z)'

    if ('Z' == sign) {
        *minuteOffset = 0;
        *nextPos      = p;

        return 0;                                                     // RETURN
    }

    enum { k_MINIMUM_LENGTH = sizeof "hh" - 1 };

    if (('+' != sign && '-' != sign) || end - p < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    // We have parsed a '+' or '-' and established that there are sufficient
    // characters to represent "hh" (but not necessarily "hh:mm").

    // Parse hour.

    int hour;
    int minute = 0;

    if (0 != u::asciiToInt(&p, &hour, p, p + 2) || hour >= 24) {
        return -1;                                                    // RETURN
    }

    if (p < end && ':' == *p) {
        ++p;  // skip ':'

        if (end - p < 2) {
            return -1;                                                // RETURN
        }

        // Parse minute.

        if (0 != u::asciiToInt(&p, &minute, p, p + 2) || minute > 59) {
            return -1;                                                // RETURN
        }
    }

    *minuteOffset = hour * 60 + minute;

    if ('-' == sign) {
        *minuteOffset = -*minuteOffset;
    }

    *nextPos = p;

    return 0;
}

int parseTime(const char **nextPos,
              Time        *time,
              int         *tzOffset,
              bool        *isNextDay,
              const char  *begin,
              const char  *end)
    // Parse the time, represented in the "hh:mm[:ss[.s+]]" FIX extended
    // format, from the string starting at the specified 'begin' and ending
    // before the specified 'end', load into the specified 'time' the parsed
    // value with the fractional second rounded to the closest microsecond,
    // load into the specified 'tzOffset' the number of minutes the time is
    // offset from UTC, load into the specified 'isNextDay' whether the
    // specified 'time' would be 24:00:00.000000 or greater, and set the
    // specified '*nextPos' to the location one past the last parsed
    // character.  Return 0 on success, and a non-zero value (with no effect on
    // '*nextPos') otherwise.  The behavior is undefined unless
    // 'begin <= end'.  Note that successfully parsing a time before 'end' is
    // reached is not an error.
{
    BSLS_ASSERT(nextPos);
    BSLS_ASSERT(time);
    BSLS_ASSERT(tzOffset);
    BSLS_ASSERT(isNextDay);
    BSLS_ASSERT(begin);
    BSLS_ASSERT(end);
    BSLS_ASSERT(begin <= end);

    const char *p = begin;

    enum { k_MINIMUM_LENGTH = sizeof "hh:mm" - 1 };

    if (end - p < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    int hour;
    if (0 != u::asciiToInt(&p, &hour, p, p + 2) || ':' != *p) {
        return -1;                                                    // RETURN
    }

    // "24:00" is not a valid FIX time string according to the protocol.

    if (24 <= hour) {
        return -1;                                                    // RETURN
    }
    ++p;  // skip ':'

    int minute;
    if (0 != u::asciiToInt(&p, &minute, p, p + 2)) {
        return -1;                                                    // RETURN
    }

    int second = 0;
    int millisecond = 0;
    int microsecond = 0;
    bool hasLeapSecond = false;

    if (p < end && ':' == *p) {
        // We have seconds.

        ++p;  // skip ':'

        if (end - p < 2 || 0 != u::asciiToInt(&p, &second, p, p + 2)) {
            return -1;                                                // RETURN
        }

        if (p < end && '.' == *p) {
            // We have a fraction of a second.

            ++p;  // skip '.'

            if (0 != u::parseFractionalSecond(&p, &microsecond, p, end)) {
                return -1;                                            // RETURN
            }
            millisecond = microsecond / 1000;
            microsecond %= 1000;
        }

        if (60 == second) {
            hasLeapSecond = true;
            second        = 59;
        }
    }

    bool roundedUpMicroseconds = (1000 == millisecond);
    if (roundedUpMicroseconds) {
        millisecond = 0;
    }

    int localTzOffset = 0;
    if (p != end) {
        if (0 != u::parseTimezoneOffset(&p, &localTzOffset, p, end) ||
                                                                    p != end) {
            return -1;                                                // RETURN
        }
    }

    if (0 != time->setTimeIfValid(hour,
                                  minute,
                                  second,
                                  millisecond,
                                  microsecond)) {
        return -1;                                                    // RETURN
    }

    *tzOffset = localTzOffset;

    *isNextDay = false;
    if (roundedUpMicroseconds) {
        *isNextDay = (1 == time->addSeconds(1));
    }
    if (hasLeapSecond) {
        if (1 == time->addSeconds(1)) {
            *isNextDay = true;
        }
    }

    *nextPos = p;

    return 0;
}

int generateInt(char *buffer, int value, int paddedLen)
    // Write, to the specified 'buffer', the decimal string representation of
    // the specified 'value' padded with leading zeros to the specified
    // 'paddedLen', and return 'paddedLen'.  'buffer' is NOT null-terminated.
    // The behavior is undefined unless '0 <= value', '0 <= paddedLen', and
    // 'buffer' has sufficient capacity to hold 'paddedLen' characters.  Note
    // that if the decimal string representation of 'value' is more than
    // 'paddedLen' digits, only the low-order 'paddedLen' digits of 'value' are
    // output.
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= value);
    BSLS_ASSERT(0 <= paddedLen);

    char *p = buffer + paddedLen;

    while (p > buffer) {
        *--p = static_cast<char>('0' + value % 10);
        value /= 10;
    }

    return paddedLen;
}

inline
int generateInt(char *buffer, int value, int paddedLen, char separator)
    // Write, to the specified 'buffer', the decimal string representation of
    // the specified 'value' padded with leading zeros to the specified
    // 'paddedLen' followed by the specified 'separator' character, and return
    // 'paddedLen + 1'.  'buffer' is NOT null-terminated.  The behavior is
    // undefined unless '0 <= value', '0 <= paddedLen', and 'buffer' has
    // sufficient capacity to hold 'paddedLen' characters.  Note that if the
    // decimal string representation of 'value' is more than 'paddedLen'
    // digits, only the low-order 'paddedLen' digits of 'value' are output.
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= value);
    BSLS_ASSERT(0 <= paddedLen);

    buffer += u::generateInt(buffer, value, paddedLen);
    *buffer = separator;

    return paddedLen + 1;
}

int generateTimezoneOffset(char                        *buffer,
                           int                          tzOffset,
                           const FixUtilConfiguration&  configuration)
    // Write, to the specified 'buffer', the formatted timezone offset
    // indicated by the specified 'tzOffset' and 'configuration', and return
    // the number of bytes written.  The behavior is undefined unless 'buffer'
    // has sufficient capacity and '-(24 * 60) < tzOffset < 24 * 60'.
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(-(24 * 60) < tzOffset);
    BSLS_ASSERT(             tzOffset < 24 * 60);

    char *p = buffer;

    if (0 == tzOffset && configuration.useZAbbreviationForUtc()) {
        *p++ = 'Z';
    }
    else {
        char tzSign;

        if (0 > tzOffset) {
            tzOffset = -tzOffset;
            tzSign   = '-';
        }
        else {
            tzSign   = '+';
        }

        *p++ = tzSign;

        p += u::generateInt(p, tzOffset / 60, 2, ':');
        p += u::generateInt(p, tzOffset % 60, 2);
    }

    return static_cast<int>(p - buffer);
}

#if defined(BSLS_ASSERT_SAFE_IS_USED)
int generatedLengthForDateTzObject(int                         defaultLength,
                                   int                         tzOffset,
                                   const FixUtilConfiguration& configuration)
    // Return the number of bytes generated, when the specified 'configuration'
    // is used, for a 'bdlt::DateTz' object having the specified 'tzOffset'
    // whose FIX representation has the specified 'defaultLength' (in bytes).
    // The behavior is undefined unless '0 <= defaultLength' and
    // '-(24 * 60) < tzOffset < 24 * 60'.
{
    BSLS_ASSERT_SAFE(0 <= defaultLength);
    BSLS_ASSERT_SAFE(-(24 * 60) < tzOffset);
    BSLS_ASSERT_SAFE(             tzOffset < 24 * 60);

    // Consider only those 'configuration' options that can affect the length
    // of the output.

    if (0 == tzOffset && configuration.useZAbbreviationForUtc()) {
        return defaultLength - static_cast<int>(sizeof "00:00") + 1;  // RETURN
    }

    return defaultLength;
}

int generatedLengthForDatetimeObject(int                         defaultLength,
                                     const FixUtilConfiguration& configuration)
    // Return the number of bytes generated, when the specified 'configuration'
    // is used, for a 'bdlt::Datetime' object whose FIX representation has the
    // specified 'defaultLength' (in bytes).  The behavior is undefined unless
    // '0 <= defaultLength'.
{
    BSLS_ASSERT_SAFE(0 <= defaultLength);

    return defaultLength
         - (6 - configuration.fractionalSecondPrecision())
         - (0 == configuration.fractionalSecondPrecision() ? 1 : 0);
}

int generatedLengthForDatetimeTzObject(
                                     int                         defaultLength,
                                     int                         tzOffset,
                                     const FixUtilConfiguration& configuration)
    // Return the number of bytes generated, when the specified 'configuration'
    // is used, for a 'bdlt::DatetimeTz' object having the specified 'tzOffset'
    // whose FIX representation has the specified 'defaultLength' (in bytes).
    // The behavior is undefined unless '0 <= defaultLength' and
    // '-(24 * 60) < tzOffset < 24 * 60'.
{
    BSLS_ASSERT_SAFE(0 <= defaultLength);
    BSLS_ASSERT_SAFE(-(24 * 60) < tzOffset);
    BSLS_ASSERT_SAFE(             tzOffset < 24 * 60);

    // Consider only those 'configuration' options that can affect the length
    // of the output.

    defaultLength = defaultLength
                  - (6 - configuration.fractionalSecondPrecision())
                  - (0 == configuration.fractionalSecondPrecision() ? 1 : 0);

    if (0 == tzOffset && configuration.useZAbbreviationForUtc()) {
        return defaultLength - static_cast<int>(sizeof "00:00") + 1;  // RETURN
    }

    return defaultLength;
}

int generatedLengthForTimeObject(int                         defaultLength,
                                 const FixUtilConfiguration& configuration)
    // Return the number of bytes generated, when the specified 'configuration'
    // is used, for a 'bdlt::Time' object whose FIX representation has the
    // specified 'defaultLength' (in bytes).  The behavior is undefined unless
    // '0 <= defaultLength'.
{
    BSLS_ASSERT_SAFE(0 <= defaultLength);

    int precision = configuration.fractionalSecondPrecision();

    return defaultLength - (6 - precision) - (0 == precision ? 1 : 0);
}

int generatedLengthForTimeTzObject(int                         defaultLength,
                                   int                         tzOffset,
                                   const FixUtilConfiguration& configuration)
    // Return the number of bytes generated, when the specified 'configuration'
    // is used, for a 'bdlt::TimeTz' object having the specified 'tzOffset'
    // whose FIX representation has the specified 'defaultLength' (in bytes).
    // The behavior is undefined unless '0 <= defaultLength' and
    // '-(24 * 60) < tzOffset < 24 * 60'.
{
    BSLS_ASSERT_SAFE(0 <= defaultLength);
    BSLS_ASSERT_SAFE(-(24 * 60) < tzOffset);
    BSLS_ASSERT_SAFE(             tzOffset < 24 * 60);

    // Consider only those 'configuration' options that can affect the length
    // of the output.

    if (0 == tzOffset && configuration.useZAbbreviationForUtc()) {
        return defaultLength - static_cast<int>(sizeof "00:00") + 1;  // RETURN
    }

    return defaultLength;
}
#endif

void copyBuf(char *dst, int dstLen, const char *src, int srcLen)
    // Copy, to the specified 'dst' buffer having the specified 'dstLen', the
    // specified initial 'srcLen' characters in the specified 'src' string if
    // 'dstLen >= srcLen', and copy 'dstLen' characters otherwise.  Include a
    // null terminator if and only if 'dstLen > srcLen'.  The behavior is
    // undefined unless '0 <= dstLen' and '0 <= srcLen'.
{
    BSLS_ASSERT(dst);
    BSLS_ASSERT(0 <= dstLen);
    BSLS_ASSERT(src);
    BSLS_ASSERT(0 <= srcLen);

    if (dstLen > srcLen) {
        bsl::memcpy(dst, src, srcLen);
        dst[srcLen] = '\0';
    }
    else {
        bsl::memcpy(dst, src, dstLen);
    }
}

}  // close namespace u
}  // close unnamed namespace

namespace BloombergLP {
namespace bdlt {

                              // --------------
                              // struct FixUtil
                              // --------------

// CLASS METHODS
int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const Date&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_DATE_STRLEN + 1) {
        outLen = generateRaw(buffer, object, configuration);
        BSLS_ASSERT(outLen == k_DATE_STRLEN);

        buffer[outLen] = '\0';
    }
    else {
        char outBuf[k_DATE_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);
        BSLS_ASSERT(outLen == k_DATE_STRLEN);

        bsl::memcpy(buffer, outBuf, bufferLength);
    }

    return outLen;
}

int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const Time&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_TIME_STRLEN) {
        outLen = generateRaw(buffer, object, configuration);
    }
    else {
        char outBuf[k_TIME_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);

        bsl::memcpy(buffer, outBuf, bufferLength);
    }

    if (bufferLength > outLen) {
        buffer[outLen] = '\0';
    }

    BSLS_ASSERT_SAFE(outLen == u::generatedLengthForTimeObject(k_TIME_STRLEN,
                                                               configuration));

    return outLen;
}

int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const Datetime&              object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_DATETIME_STRLEN) {
        outLen = generateRaw(buffer, object, configuration);
    }
    else {
        char outBuf[k_DATETIME_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);

        bsl::memcpy(buffer, outBuf, bufferLength);
    }

    if (bufferLength > outLen) {
        buffer[outLen] = '\0';
    }

    BSLS_ASSERT_SAFE(outLen == u::generatedLengthForDatetimeObject(
                                                             k_DATETIME_STRLEN,
                                                             configuration));

    return outLen;
}

int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const DateTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_DATETZ_STRLEN + 1) {
        outLen = generateRaw(buffer, object, configuration);
        BSLS_ASSERT(outLen <= k_DATETZ_STRLEN);

        buffer[outLen] = '\0';
    }
    else {
        char outBuf[k_DATETZ_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);
        BSLS_ASSERT(outLen <= k_DATETZ_STRLEN);

        u::copyBuf(buffer, bufferLength, outBuf, outLen);
    }

    BSLS_ASSERT_SAFE(outLen ==
                             u::generatedLengthForDateTzObject(k_DATETZ_STRLEN,
                                                               object.offset(),
                                                               configuration));

    return outLen;
}

int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const TimeTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_TIMETZ_STRLEN) {
        outLen = generateRaw(buffer, object, configuration);

        BSLS_ASSERT(outLen <= k_TIMETZ_STRLEN);
    }
    else {
        char outBuf[k_TIMETZ_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);

        BSLS_ASSERT(outLen <= k_TIMETZ_STRLEN);

        u::copyBuf(buffer, bufferLength, outBuf, outLen);
    }

    if (bufferLength > outLen) {
        buffer[outLen] = '\0';
    }

    BSLS_ASSERT_SAFE(outLen ==
                             u::generatedLengthForTimeTzObject(k_TIMETZ_STRLEN,
                                                               object.offset(),
                                                               configuration));

    return outLen;
}

int FixUtil::generate(char                        *buffer,
                      int                          bufferLength,
                      const DatetimeTz&            object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);
    BSLS_ASSERT(0 <= bufferLength);

    int outLen;

    if (bufferLength >= k_DATETIMETZ_STRLEN) {
        outLen = generateRaw(buffer, object, configuration);

        BSLS_ASSERT(outLen <= k_DATETIMETZ_STRLEN);
    }
    else {
        char outBuf[k_DATETIMETZ_STRLEN];

        outLen = generateRaw(outBuf, object, configuration);

        BSLS_ASSERT(outLen <= k_DATETIMETZ_STRLEN);

        u::copyBuf(buffer, bufferLength, outBuf, outLen);
    }

    if (bufferLength > outLen) {
        buffer[outLen] = '\0';
    }

    BSLS_ASSERT_SAFE(outLen ==
                     u::generatedLengthForDatetimeTzObject(k_DATETIMETZ_STRLEN,
                                                           object.offset(),
                                                           configuration));

    return outLen;
}

int FixUtil::generateRaw(char                        *buffer,
                         const Date&                  object,
                         const FixUtilConfiguration&  )
{
    BSLS_ASSERT(buffer);

    char *p = buffer;

    p += u::generateInt(p, object.year() , 4);
    p += u::generateInt(p, object.month(), 2);
    p += u::generateInt(p, object.day()  , 2);

    return static_cast<int>(p - buffer);
}

int FixUtil::generateRaw(char                        *buffer,
                         const Time&                  object,
                         const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);

    char *p = buffer;

    p += u::generateInt(p, 24 > object.hour() ? object.hour() : 0, 2, ':');
    p += u::generateInt(p, object.minute(), 2, ':');

    int precision = configuration.fractionalSecondPrecision();

    if (precision) {
        p += u::generateInt(p, object.second(), 2, '.');

        int value = object.millisecond() * 1000 + object.microsecond();

        for (int i = 6; i > precision; --i) {
            value /= 10;
        }

        p += u::generateInt(p, value, precision);
    }
    else {
        p += u::generateInt(p, object.second(), 2);
    }

    return static_cast<int>(p - buffer);
}

int FixUtil::generateRaw(char                        *buffer,
                         const Datetime&              object,
                         const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);

    const int dateLen = generateRaw(buffer, object.date(), configuration);
    *(buffer + dateLen) = '-';

    char *p = buffer + dateLen + 1;

    p += u::generateInt(p, 24 > object.hour() ? object.hour() : 0, 2, ':');
    p += u::generateInt(p, object.minute(), 2, ':');

    int precision = configuration.fractionalSecondPrecision();

    if (precision) {
        p += u::generateInt(p, object.second(), 2, '.');

        int value = object.millisecond() * 1000 + object.microsecond();

        for (int i = 6; i > precision; --i) {
            value /= 10;
        }

        p += u::generateInt(p, value, precision);
    }
    else {
        p += u::generateInt(p, object.second(), 2);
    }

    return static_cast<int>(p - buffer);
}

int FixUtil::generateRaw(char                        *buffer,
                         const DateTz&                object,
                         const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);

    const int dateLen = generateRaw(buffer,
                                    object.localDate(),
                                    configuration);

    const int zoneLen = u::generateTimezoneOffset(buffer + dateLen,
                                                  object.offset(),
                                                  configuration);

    return dateLen + zoneLen;
}

int FixUtil::generateRaw(char                        *buffer,
                         const TimeTz&                object,
                         const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);

    const Time time = object.localTime();

    char *p = buffer;

    p += u::generateInt(p, 24 > time.hour() ? time.hour() : 0, 2, ':');
    p += u::generateInt(p, time.minute(), 2, ':');
    p += u::generateInt(p, time.second(), 2);

    const int timeLen = static_cast<int>(p - buffer);

    const int zoneLen = u::generateTimezoneOffset(buffer + timeLen,
                                                  object.offset(),
                                                  configuration);

    return timeLen + zoneLen;
}

int FixUtil::generateRaw(char                        *buffer,
                         const DatetimeTz&            object,
                         const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(buffer);

    const int datetimeLen = generateRaw(buffer,
                                        object.localDatetime(),
                                        configuration);

    const int zoneLen     = u::generateTimezoneOffset(buffer + datetimeLen,
                                                      object.offset(),
                                                      configuration);

    return datetimeLen + zoneLen;
}

int FixUtil::generate(bsl::string                 *string,
                      const Date&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(bsl::string                 *string,
                      const Time&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(bsl::string                 *string,
                      const Datetime&              object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(bsl::string                 *string,
                      const DateTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(bsl::string                 *string,
                      const TimeTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(bsl::string                 *string,
                      const DatetimeTz&            object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const Date&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const Time&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const Datetime&              object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const DateTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const TimeTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::string                 *string,
                      const DatetimeTz&            object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
int FixUtil::generate(std::pmr::string            *string,
                      const Date&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::pmr::string            *string,
                      const Time&                  object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::pmr::string            *string,
                      const Datetime&              object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::pmr::string            *string,
                      const DateTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::pmr::string            *string,
                      const TimeTz&                object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}

int FixUtil::generate(std::pmr::string            *string,
                      const DatetimeTz&            object,
                      const FixUtilConfiguration&  configuration)
{
    BSLS_ASSERT(string);

    return u::Impl::generate(string, object, configuration);
}
#endif

int FixUtil::parse(Date *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX date: "20050131+04:00"
    //
    // The timezone offset is optional.

    enum { k_MINIMUM_LENGTH = sizeof "YYYYMMDD" - 1 };

    if (length < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    const char *p   = string;
    const char *end = string + length;

    Date date;
    if (0 != u::parseDate(&p, &date, p, end)) {
        return -1;                                                    // RETURN
    }

    if (p != end) {
        int tzOffset;

        if (0 != u::parseTimezoneOffset(&p, &tzOffset, p, end) || p != end) {
            return -1;                                                // RETURN
        }
    }

    *result = date;

    return 0;
}

int FixUtil::parse(Time *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX time: "08:59:59.999-04:00"
    //
    // The fractional second and timezone offset are independently optional.

    enum { k_MINIMUM_LENGTH = sizeof "hh:mm" - 1 };

    if (length < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    const char *p   = string;
    const char *end = string + length;

    Time localTime;
    int tzOffset;
    bool isNextDay;
    if (0 != u::parseTime(&p, &localTime, &tzOffset, &isNextDay, p, end)) {
        return -1;                                                    // RETURN
    }

    if (tzOffset) {
        localTime.addMinutes(-tzOffset);  // convert to UTC
    }

    *result = localTime;

    return 0;
}

int FixUtil::parse(Datetime *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX datetime: "20050131-08:59:59.999-04:00"
    //
    // The fractional second and timezone offset are independently optional.

    // 1. Parse as a 'DatetimeTz'.

    DatetimeTz datetimeTz;

    const int rc = parse(&datetimeTz, string, length);

    if (0 != rc) {
        return rc;                                                    // RETURN
    }

    // 2. Account for edge cases.

    if (datetimeTz.offset() > 0) {
        Datetime minDatetime(0001, 01, 01, 00, 00, 00, 000, 000);

        minDatetime.addMinutes(datetimeTz.offset());

        if (minDatetime > datetimeTz.localDatetime()) {
            return -1;                                                // RETURN
        }
    }
    else if (datetimeTz.offset() < 0) {
        Datetime maxDatetime(9999, 12, 31, 23, 59, 59, 999, 999);

        maxDatetime.addMinutes(datetimeTz.offset());

        if (maxDatetime < datetimeTz.localDatetime()) {
            return -1;                                                // RETURN
        }
    }

    *result = datetimeTz.utcDatetime();

    return 0;
}

int FixUtil::parse(DateTz *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX date: "2005-01-31+04:00"
    //
    // The timezone offset is optional.

    enum { k_MINIMUM_LENGTH = sizeof "YYYYMMDD" - 1 };

    if (length < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    const char *p   = string;
    const char *end = string + length;

    Date date;
    if (0 != u::parseDate(&p, &date, p, end)) {
        return -1;                                                    // RETURN
    }

    int tzOffset = 0;  // minutes from UTC
    if (p != end) {
        if (0 != u::parseTimezoneOffset(&p, &tzOffset, p, end) || p != end) {
            return -1;                                                // RETURN
        }
    }

    result->setDateTz(date, tzOffset);

    return 0;
}

int FixUtil::parse(TimeTz *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX time: "08:59:59.999-04:00"
    //
    // The fractional second and timezone offset are independently optional.

    enum { k_MINIMUM_LENGTH = sizeof "hh:mm" - 1 };

    if (length < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    const char *p   = string;
    const char *end = string + length;

    Time localTime;
    int tzOffset;
    bool isNextDay;
    if (0 != u::parseTime(&p, &localTime, &tzOffset, &isNextDay, p, end)) {
        return -1;                                                    // RETURN
    }

    result->setTimeTz(localTime, tzOffset);

    return 0;
}

int FixUtil::parse(DatetimeTz *result, const char *string, int length)
{
    BSLS_ASSERT(result);
    BSLS_ASSERT(string);
    BSLS_ASSERT(0 <= length);

    // Sample FIX datetime: "20050131-08:59:59.999-04:00"
    //
    // The fractional second and timezone offset are independently optional.

    enum { k_MINIMUM_LENGTH = sizeof "YYYYMMDD-hh:mm" - 1 };

    if (length < k_MINIMUM_LENGTH) {
        return -1;                                                    // RETURN
    }

    const char *p   = string;
    const char *end = string + length;

    Date date;
    if (0 != u::parseDate(&p, &date, p, end) || p == end || '-' != *p) {
        return -1;                                                    // RETURN
    }
    ++p;  // skip '-'

    Time time;
    int tzOffset;
    bool isNextDay;
    if (0 != u::parseTime(&p, &time, &tzOffset, &isNextDay, p, end)) {
        return -1;                                                    // RETURN
    }

    if (isNextDay) {
        if (0 != date.addDaysIfValid(1)) {
            return -1;                                                // RETURN
        }
    }

    result->setDatetimeTz(Datetime(date, time), tzOffset);

    return 0;
}

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
