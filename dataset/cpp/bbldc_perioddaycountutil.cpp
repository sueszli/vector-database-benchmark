// bbldc_perioddaycountutil.cpp                                       -*-C++-*-
#include <bbldc_perioddaycountutil.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bbldc_perioddaycountutil_cpp,"$Id$ $CSID$")

#include <bbldc_periodicmaactualactual.h>

#include <bsls_assert.h>

namespace BloombergLP {
namespace bbldc {

// STATIC HELPER FUNCTIONS

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

                        // -------------------------
                        // struct PeriodDayCountUtil
                        // -------------------------

// PRIVATE CLASS METHODS
double PeriodDayCountUtil::yearsDiffImp(
                                     const bdlt::Date&         beginDate,
                                     const bdlt::Date&         endDate,
                                     const bdlt::Date         *periodDateBegin,
                                     const bdlt::Date         *periodDateEnd,
                                     double                    periodYearDiff,
                                     DayCountConvention::Enum  convention)
{
    BSLS_ASSERT(2 <= periodDateEnd - periodDateBegin);
    BSLS_ASSERT(*periodDateBegin <= beginDate);
    BSLS_ASSERT(                    beginDate <= *(periodDateEnd - 1));
    BSLS_ASSERT(*periodDateBegin <= endDate);
    BSLS_ASSERT(                    endDate   <= *(periodDateEnd - 1));

    BSLS_ASSERT_SAFE(isSortedAndUnique(periodDateBegin, periodDateEnd));

    double numYears;

    switch (convention) {
      case DayCountConvention::e_PERIOD_ICMA_ACTUAL_ACTUAL: {
        numYears = bbldc::PeriodIcmaActualActual::yearsDiff(beginDate,
                                                            endDate,
                                                            periodDateBegin,
                                                            periodDateEnd,
                                                            periodYearDiff);
      } break;
      default: {
        BSLS_ASSERT_OPT(0 && "Unrecognized convention");
        numYears = 0.0;
      } break;
    }

    return numYears;
}

// CLASS METHODS
int PeriodDayCountUtil::daysDiff(const bdlt::Date&        beginDate,
                                 const bdlt::Date&        endDate,
                                 DayCountConvention::Enum convention)
{
    int numDays;

    switch (convention) {
      case DayCountConvention::e_PERIOD_ICMA_ACTUAL_ACTUAL: {
        numDays = bbldc::PeriodIcmaActualActual::daysDiff(beginDate, endDate);
      } break;
      default: {
        BSLS_ASSERT_OPT(0 && "Unrecognized convention");
        numDays = 0;
      } break;
    }

    return numDays;
}

bool PeriodDayCountUtil::isSupported(DayCountConvention::Enum convention)
{
    bool rv = true;

    switch (convention) {
      case DayCountConvention::e_PERIOD_ICMA_ACTUAL_ACTUAL: {
      } break;
      default: {
        rv = false;
      } break;
    }

    return rv;
}

}  // close package namespace
}  // close enterprise namespace

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
