// bslhdrs_dummy.cpp                                                  -*-C++-*-

#include <bsls_ident.h>
BSLS_IDENT("$Id: $")

//@PURPOSE: Workaround for build tool.
//
//@DESCRIPTION: This is a workaround for the build tool since the tool requires
// at least one implementation file in a package.

// This symbol exists to avoid 'getarsym' errors when linking tests against the
// 'bsl+bslhdrs' package library on SunOS with gcc.
char bslhdrs_dummy_cpp_this_symbol_avoids_an_empty_package_library;

// ----------------------------------------------------------------------------
// Copyright 2013 Bloomberg Finance L.P.
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
