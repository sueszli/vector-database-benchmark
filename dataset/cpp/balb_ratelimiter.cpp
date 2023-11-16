// balb_ratelimiter.cpp                                               -*-C++-*-
#include <balb_ratelimiter.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(balb_ratelimiter.cpp, "$Id$ $CSID$")

namespace BloombergLP {
namespace balb {

                             //------------------
                             // class RateLimiter
                             //------------------

// CREATORS
RateLimiter::RateLimiter(bsls::Types::Uint64       sustainedRateLimit,
                         const bsls::TimeInterval& sustainedRateWindow,
                         bsls::Types::Uint64       peakRateLimit,
                         const bsls::TimeInterval& peakRateWindow,
                         const bsls::TimeInterval& currentTime)
: d_peakRateBucket(1, 1, currentTime)
, d_sustainedRateBucket(1, 1, currentTime)
{
    setRateLimits(sustainedRateLimit,
                  sustainedRateWindow,
                  peakRateLimit,
                  peakRateWindow);
}

RateLimiter::~RateLimiter()
{
    BSLS_ASSERT_SAFE(sustainedRateLimit() > 0);
    BSLS_ASSERT_SAFE(peakRateLimit() > 0);

    BSLS_ASSERT_SAFE(sustainedRateWindow() > bsls::TimeInterval(0));
    BSLS_ASSERT_SAFE(peakRateWindow() > bsls::TimeInterval(0));

    BSLS_ASSERT_SAFE(peakRateLimit() == 1 ||
                     peakRateWindow() <=
                         LeakyBucket::calculateDrainTime(
                             ULLONG_MAX, peakRateLimit(), true));

    BSLS_ASSERT_SAFE(sustainedRateLimit() == 1 ||
                     sustainedRateWindow() <=
                         LeakyBucket::calculateDrainTime(
                             ULLONG_MAX, sustainedRateLimit(), true));
}

// CLASS METHODS
namespace {
bool supportsExactly(bsls::Types::Uint64       limit,
                     const bsls::TimeInterval& window)
    // Return 'true' if the specified 'limit' and 'window' are legal values
    // with which to initialize a 'balb::LeakyBucket' object, and if so,
    // whether a 'balb::LeakyBucket' object so initialized would preserve the
    // value of 'window'.
{
    // Aside from checking that the capacity calculated from 'window' and
    // 'limit' can back out the same 'window' value, we also include checks on
    // the parameters that the functions called do as assertions, so that this
    // function will return 'false' for those values, e.g., 'window' is large
    // enough to cause integer overflow.
    return limit > 0 && window > bsls::TimeInterval() &&
           (limit == 1 || window <= LeakyBucket::calculateDrainTime(
                                        ULLONG_MAX, limit, true)) &&
           window == LeakyBucket::calculateTimeWindow(
                         limit, LeakyBucket::calculateCapacity(limit, window));
}

}  // close unnamed namespace

bool RateLimiter::supportsRateLimitsExactly(
                                 bsls::Types::Uint64       sustainedRateLimit,
                                 const bsls::TimeInterval& sustainedRateWindow,
                                 bsls::Types::Uint64       peakRateLimit,
                                 const bsls::TimeInterval& peakRateWindow)
{
    return supportsExactly(sustainedRateLimit, sustainedRateWindow) &&
           supportsExactly(peakRateLimit, peakRateWindow);
}

// MANIPULATORS
bsls::TimeInterval RateLimiter::calculateTimeToSubmit(
                                         const bsls::TimeInterval& currentTime)
{
    bsls::TimeInterval timeToSubmitPeak =
        d_peakRateBucket.calculateTimeToSubmit(currentTime);

    bsls::TimeInterval timeToSubmitSustained =
        d_sustainedRateBucket.calculateTimeToSubmit(currentTime);

    return bsl::max(timeToSubmitPeak, timeToSubmitSustained);
}

void RateLimiter::setRateLimits(bsls::Types::Uint64       sustainedRateLimit,
                                const bsls::TimeInterval& sustainedRateWindow,
                                bsls::Types::Uint64       peakRateLimit,
                                const bsls::TimeInterval& peakRateWindow)
{
    BSLS_ASSERT(sustainedRateLimit > 0);
    BSLS_ASSERT(peakRateLimit > 0);

    BSLS_ASSERT(sustainedRateWindow > bsls::TimeInterval(0));
    BSLS_ASSERT(peakRateWindow > bsls::TimeInterval(0));

    BSLS_ASSERT(peakRateLimit == 1 ||
                peakRateWindow <= LeakyBucket::calculateDrainTime(
                                      ULLONG_MAX, peakRateLimit, true));

    BSLS_ASSERT(sustainedRateLimit == 1 ||
                sustainedRateWindow <=
                    LeakyBucket::calculateDrainTime(
                        ULLONG_MAX, sustainedRateLimit, true));

    bsls::Types::Uint64 capacity = LeakyBucket::calculateCapacity(
        sustainedRateLimit, sustainedRateWindow);

    d_sustainedRateBucket.setRateAndCapacity(sustainedRateLimit, capacity);

    capacity = LeakyBucket::calculateCapacity(peakRateLimit, peakRateWindow);

    d_peakRateBucket.setRateAndCapacity(peakRateLimit, capacity);
}

}  // close package namespace
}  // close enterprise namespace

// ----------------------------------------------------------------------------
// Copyright 2021 Bloomberg Finance L.P.
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
