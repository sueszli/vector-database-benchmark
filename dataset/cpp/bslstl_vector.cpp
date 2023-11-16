// bslstl_vector.cpp                                                  -*-C++-*-
#include <bslstl_vector.h>

#include <bsls_ident.h>
BSLS_IDENT("$Id$ $CSID$")

// IMPLEMENTATION NOTES: The class 'bsl::vector' is split in two for a
// conflation of two reasons:
//
//  1. We want the members 'd_data...' and 'd_capacity' to appear *before* the
//     allocator, which is provided by 'bslalg::ContainerBase' (to potentially
//     take advantage of the empty-base-class-optimization).
//
//  2. The 'bsl::vectorBase' class containing these members need only be
//     parameterized by 'VALUE_TYPE' (and not 'ALLOCATOR'), and can provide the
//     iterator and element access methods, leading to shorter debug strings
//     for those methods.
//
// Moreover, in the spirit of template hoisting (providing functionality to all
// templates in a non-templated utility class), the 'swap' method is
// implemented below since its definition does not rely on the definition of
// 'VALUE_TYPE'.

#include <bsls_assert.h>

#include <string.h>  // for 'memcpy'

namespace bsl {

namespace {
                          // ------------------
                          // struct Vector_Base
                          // ------------------

struct Vector_Base {
    // This 'struct' must have the same layout as 'bsl::vectorBase' (defined in
    // the .h file).

    // PUBLIC DATA
    void        *d_dataBegin_p;
    void        *d_dataEnd_p;
    std::size_t  d_capacity;
};

}  // close unnamed namespace

                          // ------------------
                          // struct Vector_Util
                          // ------------------

// CLASS METHODS
std::size_t Vector_Util::computeNewCapacity(std::size_t newLength,
                                            std::size_t capacity,
                                            std::size_t maxSize)
{
    BSLS_ASSERT_SAFE(newLength > capacity);
    BSLS_ASSERT_SAFE(newLength <= maxSize);

    capacity += !capacity;
    while (capacity < newLength) {
        std::size_t oldCapacity = capacity;
        capacity *= 2;
        if (capacity < oldCapacity) {
            // We overflowed, e.g., on a 32-bit platform; 'newCapacity' is
            // larger than 2^31.  Terminate the loop.

            return maxSize;                                           // RETURN
        }
    }
    return capacity > maxSize ? maxSize : capacity;
}

void Vector_Util::swap(void *a, void *b)
{
    char c[sizeof(Vector_Base)];
    memcpy(c, a, sizeof(Vector_Base));
    memcpy(a, b, sizeof(Vector_Base));
    memcpy(b, c, sizeof(Vector_Base));
}

}  // close namespace bsl

#ifdef BSLS_COMPILERFEATURES_SUPPORT_EXTERN_TEMPLATE
template class bsl::vectorBase<bool>;
template class bsl::vectorBase<char>;
template class bsl::vectorBase<signed char>;
template class bsl::vectorBase<unsigned char>;
template class bsl::vectorBase<short>;
template class bsl::vectorBase<unsigned short>;
template class bsl::vectorBase<int>;
template class bsl::vectorBase<unsigned int>;
template class bsl::vectorBase<long>;
template class bsl::vectorBase<unsigned long>;
template class bsl::vectorBase<long long>;
template class bsl::vectorBase<unsigned long long>;
template class bsl::vectorBase<float>;
template class bsl::vectorBase<double>;
template class bsl::vectorBase<long double>;
template class bsl::vectorBase<void *>;
template class bsl::vectorBase<const char *>;

template class bsl::vector<bool>;
template class bsl::vector<char>;
template class bsl::vector<signed char>;
template class bsl::vector<unsigned char>;
template class bsl::vector<short>;
template class bsl::vector<unsigned short>;
template class bsl::vector<int>;
template class bsl::vector<unsigned int>;
template class bsl::vector<long>;
template class bsl::vector<unsigned long>;
template class bsl::vector<long long>;
template class bsl::vector<unsigned long long>;
template class bsl::vector<float>;
template class bsl::vector<double>;
template class bsl::vector<long double>;
template class bsl::vector<void *>;
template class bsl::vector<const char *>;
#endif

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
