// bdlde_charconvertstatus.cpp                                        -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <bdlde_charconvertstatus.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdlde_charconvertstatus,"$Id$ $CSID$")

#include <bslim_printer.h>

#include <bsl_ostream.h>

namespace BloombergLP {
namespace bdlde {

                          // ------------------------
                          // struct CharConvertStatus
                          // ------------------------

// CLASS METHODS
bsl::ostream& CharConvertStatus::print(bsl::ostream&           stream,
                                       CharConvertStatus::Enum value,
                                       int                     level,
                                       int                     spacesPerLevel)
{
    if (stream.bad()) {
        return stream;                                                // RETURN
    }

    bslim::Printer printer(&stream, level, spacesPerLevel);
    printer.start(true);  // suppress bracket (here and below)
    stream << CharConvertStatus::toAscii(value);
    printer.end(true);
    return stream;
}

const char *CharConvertStatus::toAscii(CharConvertStatus::Enum value)
{
    switch (value) {
      case k_INVALID_INPUT_BIT: {
        return  "INVALID_INPUT_BIT";                                  // RETURN
      } break;
      case k_OUT_OF_SPACE_BIT: {
        return  "OUT_OF_SPACE_BIT";                                   // RETURN
      } break;
    }

    return "(* UNKNOWN *)";
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
