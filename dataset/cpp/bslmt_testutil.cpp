// bslmt_testutil.cpp                                                 -*-C++-*-
#include <bslmt_testutil.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bslmt_testutil_cpp,"$Id$ $CSID$")

#include <bslmt_lockguard.h>          // for testing only
#include <bslmt_mutex.h>              // for testing only

#include <bsls_bslonce.h>


namespace BloombergLP {
namespace bslmt {

                                // ---------------
                                // struct TestUtil
                                // ---------------

// CLASS METHODS
void *TestUtil::identityPtr(void *ptr)
{
    return ptr;
}

                             // --------------------
                             // class TestUtil_Guard
                             // --------------------

RecursiveMutex& TestUtil_Guard::singletonMutex()
    // Return a reference to the recursive mutex created by this singleton.
{
    static RecursiveMutex *mutex_p;
    static bsls::BslOnce   once = BSLS_BSLONCE_INITIALIZER;

    bsls::BslOnceGuard onceGuard;
    if (onceGuard.enter(&once)) {
        static RecursiveMutex mutex;

        mutex_p = &mutex;
    }

    return *mutex_p;
}

}  // close package namespace
}  // close enterprise namespace

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
