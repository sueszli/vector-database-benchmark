// bbldc_periodicmaactualactual.cpp                                   -*-C++-*-
#include <bbldc_periodicmaactualactual.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bbldc_periodicmaactualactual_cpp,"$Id$ $CSID$")

#include <bsls_assert.h>
#include <bsls_platform.h>

#include <bsl_algorithm.h>

namespace {
namespace u {

template <class ITER>
static bool isSortedAndUnique(const ITER& begin, const ITER& end)
    // Return 'true' if all values between the specified 'begin' and 'end'
    // iterators are unique and sorted from minimum to maximum value, and
    // 'false' otherwise.
{
    if (begin == end) {
        return true;                                                  // RETURN
    }

    ITER prev = begin;
    ITER at   = begin + 1;

    while (at != end) {
        if (*prev >= *at) {
            return false;                                             // RETURN
        }
        prev = at++;
    }

    return true;
}

}  // close namespace u
}  // close unnamed namespace

namespace BloombergLP {
namespace bbldc {

// STATIC HELPER FUNCTIONS

                      // -----------------------------
                      // struct PeriodIcmaActualActual
                      // -----------------------------

// CLASS METHODS
double PeriodIcmaActualActual::yearsDiff(const bdlt::Date&  beginDate,
                                         const bdlt::Date&  endDate,
                                         const bdlt::Date  *periodDateBegin,
                                         const bdlt::Date  *periodDateEnd,
                                         double             periodYearDiff)
{
    BSLS_ASSERT(2 <= periodDateEnd - periodDateBegin);
    BSLS_ASSERT(*periodDateBegin <= beginDate);
    BSLS_ASSERT(                    beginDate <= *(periodDateEnd - 1));
    BSLS_ASSERT(*periodDateBegin <= endDate);
    BSLS_ASSERT(                    endDate   <= *(periodDateEnd - 1));

    BSLS_ASSERT_SAFE(u::isSortedAndUnique(periodDateBegin, periodDateEnd));

    if (beginDate == endDate) {
        return 0.0;                                                   // RETURN
    }

#if defined(BSLS_PLATFORM_CMP_GNU) && (BSLS_PLATFORM_CMP_VERSION >= 50301)
    // Storing the result value in a 'volatile double' removes extra-precision
    // available in floating-point registers.

    volatile double result;
#else
    double result;
#endif

    // Compute the negation flag and produce sorted dates.

    bool negationFlag = beginDate > endDate;

    bdlt::Date minDate;
    bdlt::Date maxDate;
    if (false == negationFlag) {
        minDate = beginDate;
        maxDate = endDate;
    }
    else {
        minDate = endDate;
        maxDate = beginDate;
    }

    // Find the period dates bracketing 'minDate'.

    const bdlt::Date *beginIter2 =
                     bsl::upper_bound(periodDateBegin, periodDateEnd, minDate);
    const bdlt::Date *beginIter1 = beginIter2 - 1;

    // Find the period dates bracketing 'maxDate'.

    const bdlt::Date *endIter2 =
                     bsl::lower_bound(periodDateBegin, periodDateEnd, maxDate);
    const bdlt::Date *endIter1 = endIter2 - 1;

    // Compute the fractional number of periods * 'periodYearDiff'.

    result = (  static_cast<double>(*beginIter2 - minDate) /
                                 static_cast<double>(*beginIter2 - *beginIter1)
              + static_cast<double>(endIter1 - beginIter2)
              + static_cast<double>(maxDate - *endIter1) /
                                    static_cast<double>(*endIter2 - *endIter1))
             * periodYearDiff;

    // Negate the value if necessary.

    if (negationFlag) {
        result = -result;
    }

    return result;
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
