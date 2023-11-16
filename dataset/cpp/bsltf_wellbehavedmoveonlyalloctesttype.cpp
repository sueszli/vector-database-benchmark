// bsltf_wellbehavedmoveonlyalloctesttype.cpp                         -*-C++-*-
#include <bsltf_wellbehavedmoveonlyalloctesttype.h>

#include <bsls_ident.h>
BSLS_IDENT("$Id$ $CSID$")

#include <bslma_allocator.h>
#include <bslma_default.h>

#include <bsls_assert.h>
#include <bsls_platform.h>

#if defined(BSLS_PLATFORM_CMP_MSVC)
#pragma warning(disable:4355) // ctor uses 'this' used in member-initializer
#endif

namespace BloombergLP {
namespace bsltf {

                    // --------------------------------------
                    // class WellBehavedMoveOnlyAllocTestType
                    // --------------------------------------

// CREATORS
WellBehavedMoveOnlyAllocTestType::WellBehavedMoveOnlyAllocTestType(
                                              bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_self_p(this)
, d_movedFrom(bsltf::MoveState::e_NOT_MOVED)
, d_movedInto(bsltf::MoveState::e_NOT_MOVED)
{
    d_data_p = reinterpret_cast<int *>(d_allocator_p->allocate(sizeof(int)));
    *d_data_p = 0;
}

WellBehavedMoveOnlyAllocTestType::WellBehavedMoveOnlyAllocTestType(
                                              int               data,
                                              bslma::Allocator *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_self_p(this)
, d_movedFrom(bsltf::MoveState::e_NOT_MOVED)
, d_movedInto(bsltf::MoveState::e_NOT_MOVED)
{
    d_data_p = reinterpret_cast<int *>(d_allocator_p->allocate(sizeof(int)));
    *d_data_p = data;
}

WellBehavedMoveOnlyAllocTestType::WellBehavedMoveOnlyAllocTestType(
                  bslmf::MovableRef<WellBehavedMoveOnlyAllocTestType> original)
                                                          BSLS_KEYWORD_NOEXCEPT
: d_allocator_p(bslmf::MovableRefUtil::access(original).d_allocator_p)
, d_self_p(this)
, d_movedInto(bsltf::MoveState::e_MOVED)
{
    WellBehavedMoveOnlyAllocTestType& lvalue = original;

    if (lvalue.d_data_p) {
        d_data_p = lvalue.d_data_p;
        lvalue.d_data_p = 0;
        lvalue.d_movedFrom = bsltf::MoveState::e_MOVED;
        d_movedFrom        = bsltf::MoveState::e_NOT_MOVED;
    }
    else {
        d_data_p = 0;

        // lvalue.d_movedFrom -- unchanged

        d_movedFrom = bsltf::MoveState::e_MOVED;
    }
}

WellBehavedMoveOnlyAllocTestType::WellBehavedMoveOnlyAllocTestType(
            bslmf::MovableRef<WellBehavedMoveOnlyAllocTestType> original,
            bslma::Allocator                                   *basicAllocator)
: d_allocator_p(bslma::Default::allocator(basicAllocator))
, d_self_p(this)
{
    WellBehavedMoveOnlyAllocTestType& lvalue = original;

    if (d_allocator_p == lvalue.d_allocator_p) {
        if (lvalue.d_data_p) {
            d_data_p = lvalue.d_data_p;
            lvalue.d_data_p = 0;
            lvalue.d_movedFrom = bsltf::MoveState::e_MOVED;
            d_movedFrom        = bsltf::MoveState::e_NOT_MOVED;
        }
        else {
            d_data_p = 0;

            // lvalue.d_movedFrom -- unchanged

            d_movedFrom = bsltf::MoveState::e_MOVED;
        }
        d_movedInto     = bsltf::MoveState::e_MOVED;
    }
    else {
        d_data_p =
                 reinterpret_cast<int *>(d_allocator_p->allocate(sizeof(int)));
        *d_data_p = lvalue.data();

        // lvalue.d_movedFrom -- unchanged

        d_movedFrom = bsltf::MoveState::e_NOT_MOVED;
        d_movedInto = bsltf::MoveState::e_NOT_MOVED;
    }
}

WellBehavedMoveOnlyAllocTestType::~WellBehavedMoveOnlyAllocTestType()
{
    d_allocator_p->deallocate(d_data_p);

    if ((!!d_data_p) != (bsltf::MoveState::e_NOT_MOVED == d_movedFrom)) {
        BSLS_ASSERT_INVOKE("!!d_data_p =="
                           "(bsltf::MoveState::e_NOT_MOVED == d_movedFrom");
    }

    // Ensure that this objects has not been bitwise moved.
    if (this != d_self_p) {
        BSLS_ASSERT_INVOKE("this != d_self_p");
    }
}

// MANIPULATORS
WellBehavedMoveOnlyAllocTestType&
WellBehavedMoveOnlyAllocTestType::operator=(
                       bslmf::MovableRef<WellBehavedMoveOnlyAllocTestType> rhs)
{
    WellBehavedMoveOnlyAllocTestType& lvalue = rhs;

    if (&lvalue != this) {
        if (d_allocator_p == lvalue.d_allocator_p) {
            if (lvalue.d_data_p) {
                if (d_data_p) {
                    d_allocator_p->deallocate(d_data_p);
                }
                d_data_p = lvalue.d_data_p;
                lvalue.d_data_p = 0;

                lvalue.d_movedFrom = bsltf::MoveState::e_MOVED;
                d_movedFrom        = bsltf::MoveState::e_NOT_MOVED;
            }
            else {
                if (d_data_p) {
                    d_allocator_p->deallocate(d_data_p);
                    d_data_p = 0;
                }

                // lvalue.d_movedFrom -- unchanged

                d_movedFrom = bsltf::MoveState::e_MOVED;
            }

            d_movedInto = bsltf::MoveState::e_MOVED;
        }
        else {
            int *newData = reinterpret_cast<int *>(
                                         d_allocator_p->allocate(sizeof(int)));
            if (d_data_p) {
                d_allocator_p->deallocate(d_data_p);
            }
            d_data_p = newData;
            *d_data_p = lvalue.data();

            // lvalue.d_movedFrom -- unchanged

            d_movedFrom = bsltf::MoveState::e_NOT_MOVED;
            d_movedInto = bsltf::MoveState::e_NOT_MOVED;
        }
    }
    return *this;
}

// MANIPULATORS
void WellBehavedMoveOnlyAllocTestType::setData(int value)
{
    if (!d_data_p) {
        int *newData = reinterpret_cast<int *>(
                                         d_allocator_p->allocate(sizeof(int)));
        d_data_p = newData;
    }
    *d_data_p = value;

    d_movedFrom = bsltf::MoveState::e_NOT_MOVED;
    d_movedInto = bsltf::MoveState::e_NOT_MOVED;
}

}  // close package namespace

// FREE FUNCTIONS
void bsltf::swap(bsltf::WellBehavedMoveOnlyAllocTestType& a,
                 bsltf::WellBehavedMoveOnlyAllocTestType& b)
{
    typedef bslmf::MovableRefUtil MRU;

    if (&a == &b) {
        return;                                                       // RETURN
    }

    bsltf::WellBehavedMoveOnlyAllocTestType intermediate(MRU::move(a));
    a = MRU::move(b);
    b = MRU::move(intermediate);
}

}  // close enterprise namespace

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
