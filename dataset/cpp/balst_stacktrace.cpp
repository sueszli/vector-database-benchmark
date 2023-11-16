// balst_stacktrace.cpp                                               -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <balst_stacktrace.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(balst_stacktrace_cpp,"$Id$ $CSID$")

#include <bslim_printer.h>

#include <bsl_ostream.h>

namespace BloombergLP {
namespace balst {

                              // ----------------
                              // class StackTrace
                              // ----------------

// ACCESSORS
bsl::ostream& StackTrace::print(bsl::ostream& stream,
                                int           level,
                                int           spacesPerLevel) const
{
    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start();
    for (int i = 0; i < length(); ++i) {
        d_frames[i].print(stream, level + 1, spacesPerLevel);
    }
    printer.end();

    return stream;
}

}  // close package namespace

// FREE FUNCTIONS
void balst::swap(StackTrace& a, StackTrace& b)
{
    if (a.allocator() == b.allocator()) {
        a.swap(b);

        return;                                                       // RETURN
    }

    StackTrace futureA(b, a.allocator());
    StackTrace futureB(a, b.allocator());

    futureA.swap(a);
    futureB.swap(b);
}

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
