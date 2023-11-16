// bdlma_multipool.cpp                                                -*-C++-*-
#include <bdlma_multipool.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdlma_multipool_cpp,"$Id$ $CSID$")

#include <bdlma_bufferedsequentialallocator.h>  // for testing only

#include <bdlb_bitutil.h>

#include <bslma_autodestructor.h>
#include <bslma_deallocatorproctor.h>
#include <bslma_default.h>

#include <bsls_assert.h>
#include <bsls_performancehint.h>
#include <bsls_platform.h>

#include <bsl_new.h>
#include <bsl_limits.h>

namespace BloombergLP {
namespace bdlma {

// TYPES
enum {
    k_DEFAULT_NUM_POOLS      = 10,  // default number of pools

    k_DEFAULT_MAX_CHUNK_SIZE = 32,  // default maximum number of blocks per
                                    // chunk

    k_MIN_BLOCK_SIZE         =  8   // minimum block size (in bytes)
};

                             // ---------------
                             // class Multipool
                             // ---------------

// PRIVATE MANIPULATORS
void Multipool::initialize(bsls::BlockGrowth::Strategy growthStrategy,
                           int                         maxBlocksPerChunk)
{
    BSLS_ASSERT(1 <= maxBlocksPerChunk);

    d_maxBlockSize = k_MIN_BLOCK_SIZE;

    d_pools_p = static_cast<Pool *>(
                      d_allocator_p->allocate(d_numPools * sizeof *d_pools_p));

    bslma::DeallocatorProctor<bslma::Allocator> autoPoolsDeallocator(
                                                                d_pools_p,
                                                                d_allocator_p);
    bslma::AutoDestructor<Pool> autoDtor(d_pools_p, 0);

    for (int i = 0; i < d_numPools; ++i, ++autoDtor) {
        new (d_pools_p + i) Pool(d_maxBlockSize
                                            + static_cast<int>(sizeof(Header)),
                                 growthStrategy,
                                 maxBlocksPerChunk,
                                 d_allocator_p);

        BSLS_ASSERT(d_maxBlockSize <=
                       bsl::numeric_limits<bsls::Types::size_type>::max() / 2);

        d_maxBlockSize *= 2;
    }

    d_maxBlockSize /= 2;

    autoDtor.release();
    autoPoolsDeallocator.release();
}

void Multipool::initialize(
                        const bsls::BlockGrowth::Strategy *growthStrategyArray,
                        int                                maxBlocksPerChunk)
{
    BSLS_ASSERT(growthStrategyArray);
    BSLS_ASSERT(1 <= maxBlocksPerChunk);

    d_maxBlockSize = k_MIN_BLOCK_SIZE;

    d_pools_p = static_cast<Pool *>(
                      d_allocator_p->allocate(d_numPools * sizeof *d_pools_p));

    bslma::DeallocatorProctor<bslma::Allocator> autoPoolsDeallocator(
                                                                d_pools_p,
                                                                d_allocator_p);
    bslma::AutoDestructor<Pool> autoDtor(d_pools_p, 0);

    for (int i = 0; i < d_numPools; ++i, ++autoDtor) {
        new (d_pools_p + i) Pool(d_maxBlockSize
                                            + static_cast<int>(sizeof(Header)),
                                 growthStrategyArray[i],
                                 maxBlocksPerChunk,
                                 d_allocator_p);

        BSLS_ASSERT(d_maxBlockSize <=
                       bsl::numeric_limits<bsls::Types::size_type>::max() / 2);

        d_maxBlockSize *= 2;
    }

    d_maxBlockSize /= 2;

    autoDtor.release();
    autoPoolsDeallocator.release();
}

void Multipool::initialize(bsls::BlockGrowth::Strategy  growthStrategy,
                           const int                   *maxBlocksPerChunkArray)
{
    BSLS_ASSERT(maxBlocksPerChunkArray);

    d_maxBlockSize = k_MIN_BLOCK_SIZE;

    d_pools_p = static_cast<Pool *>(
                      d_allocator_p->allocate(d_numPools * sizeof *d_pools_p));

    bslma::DeallocatorProctor<bslma::Allocator> autoPoolsDeallocator(
                                                                d_pools_p,
                                                                d_allocator_p);
    bslma::AutoDestructor<Pool> autoDtor(d_pools_p, 0);

    for (int i = 0; i < d_numPools; ++i, ++autoDtor) {
        new (d_pools_p + i) Pool(d_maxBlockSize
                                            + static_cast<int>(sizeof(Header)),
                                 growthStrategy,
                                 maxBlocksPerChunkArray[i],
                                 d_allocator_p);

        BSLS_ASSERT(d_maxBlockSize <=
                       bsl::numeric_limits<bsls::Types::size_type>::max() / 2);

        d_maxBlockSize *= 2;
    }

    d_maxBlockSize /= 2;

    autoDtor.release();
    autoPoolsDeallocator.release();
}

void Multipool::initialize(
                     const bsls::BlockGrowth::Strategy *growthStrategyArray,
                     const int                         *maxBlocksPerChunkArray)
{
    BSLS_ASSERT(growthStrategyArray);
    BSLS_ASSERT(maxBlocksPerChunkArray);

    d_maxBlockSize = k_MIN_BLOCK_SIZE;

    d_pools_p = static_cast<Pool *>(
                      d_allocator_p->allocate(d_numPools * sizeof *d_pools_p));

    bslma::DeallocatorProctor<bslma::Allocator> autoPoolsDeallocator(
                                                                d_pools_p,
                                                                d_allocator_p);
    bslma::AutoDestructor<Pool> autoDtor(d_pools_p, 0);

    for (int i = 0; i < d_numPools; ++i, ++autoDtor) {
        new (d_pools_p + i) Pool(d_maxBlockSize
                                            + static_cast<int>(sizeof(Header)),
                                 growthStrategyArray[i],
                                 maxBlocksPerChunkArray[i],
                                 d_allocator_p);

        BSLS_ASSERT(d_maxBlockSize <=
                       bsl::numeric_limits<bsls::Types::size_type>::max() / 2);

        d_maxBlockSize *= 2;
    }

    d_maxBlockSize /= 2;

    autoDtor.release();
    autoPoolsDeallocator.release();
}

// PRIVATE ACCESSORS
int Multipool::findPool(bsls::Types::size_type size) const
{
    BSLS_ASSERT(size <= d_maxBlockSize);

    return 31 - bdlb::BitUtil::numLeadingUnsetBits(static_cast<bsl::uint32_t>(
                                ((size + k_MIN_BLOCK_SIZE - 1) >> 3) * 2 - 1));
}

// CREATORS
Multipool::Multipool(bslma::Allocator *basicAllocator)
: d_numPools(k_DEFAULT_NUM_POOLS)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    initialize(bsls::BlockGrowth::BSLS_GEOMETRIC, k_DEFAULT_MAX_CHUNK_SIZE);
}

Multipool::Multipool(int               numPools,
                     bslma::Allocator *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);

    initialize(bsls::BlockGrowth::BSLS_GEOMETRIC, k_DEFAULT_MAX_CHUNK_SIZE);
}

Multipool::Multipool(bsls::BlockGrowth::Strategy  growthStrategy,
                     bslma::Allocator            *basicAllocator)
: d_numPools(k_DEFAULT_NUM_POOLS)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    initialize(growthStrategy, k_DEFAULT_MAX_CHUNK_SIZE);
}

Multipool::Multipool(int                          numPools,
                     bsls::BlockGrowth::Strategy  growthStrategy,
                     bslma::Allocator            *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);

    initialize(growthStrategy, k_DEFAULT_MAX_CHUNK_SIZE);
}

Multipool::Multipool(int                                numPools,
                     const bsls::BlockGrowth::Strategy *growthStrategyArray,
                     bslma::Allocator                  *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);
    BSLS_ASSERT(growthStrategyArray);

    initialize(growthStrategyArray, k_DEFAULT_MAX_CHUNK_SIZE);
}

Multipool::Multipool(int                          numPools,
                     bsls::BlockGrowth::Strategy  growthStrategy,
                     int                          maxBlocksPerChunk,
                     bslma::Allocator            *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);
    BSLS_ASSERT(1 <= maxBlocksPerChunk);

    initialize(growthStrategy, maxBlocksPerChunk);
}

Multipool::Multipool(int                                numPools,
                     const bsls::BlockGrowth::Strategy *growthStrategyArray,
                     int                                maxBlocksPerChunk,
                     bslma::Allocator                  *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);
    BSLS_ASSERT(growthStrategyArray);
    BSLS_ASSERT(1 <= maxBlocksPerChunk);

    initialize(growthStrategyArray, maxBlocksPerChunk);
}

Multipool::Multipool(int                          numPools,
                     bsls::BlockGrowth::Strategy  growthStrategy,
                     const int                   *maxBlocksPerChunkArray,
                     bslma::Allocator            *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);
    BSLS_ASSERT(maxBlocksPerChunkArray);

    initialize(growthStrategy, maxBlocksPerChunkArray);
}

Multipool::Multipool(int                                numPools,
                     const bsls::BlockGrowth::Strategy *growthStrategyArray,
                     const int                         *maxBlocksPerChunkArray,
                     bslma::Allocator                  *basicAllocator)
: d_numPools(numPools)
, d_blockList(basicAllocator)
, d_allocator_p(bslma::Default::allocator(basicAllocator))
{
    BSLS_ASSERT(1 <= numPools);
    BSLS_ASSERT(growthStrategyArray);
    BSLS_ASSERT(maxBlocksPerChunkArray);

    initialize(growthStrategyArray, maxBlocksPerChunkArray);
}

Multipool::~Multipool()
{
    BSLS_ASSERT(d_pools_p);
    BSLS_ASSERT(1 <= d_numPools);
    BSLS_ASSERT(1 <= d_maxBlockSize);
    BSLS_ASSERT(d_allocator_p);

    d_blockList.release();
    for (int i = 0; i < d_numPools; ++i) {
        d_pools_p[i].release();
        d_pools_p[i].~Pool();
    }
    d_allocator_p->deallocate(d_pools_p);
}

// MANIPULATORS
void *Multipool::allocate(bsls::Types::size_type size)
{
    if (BSLS_PERFORMANCEHINT_PREDICT_LIKELY(size)) {
        if (size <= d_maxBlockSize) {
            const int pool = findPool(size);

            Header *p = static_cast<Header *>(d_pools_p[pool].allocate());

            p->d_header.d_poolIdx = pool;

            return p + 1;                                             // RETURN
        }

        // The requested size is large and will not be pooled.

        Header *p = static_cast<Header *>(
                d_blockList.allocate(size + static_cast<int>(sizeof(Header))));

        p->d_header.d_poolIdx = -1;

        return p + 1;                                                 // RETURN
    }

    return 0;
}

void Multipool::deallocate(void *address)
{
    BSLS_ASSERT(address);

    Header *h = static_cast<Header *>(address) - 1;

    const int pool = h->d_header.d_poolIdx;

    if (-1 == pool) {
        d_blockList.deallocate(h);
    }
    else {
        d_pools_p[pool].deallocate(h);
    }
}

void Multipool::release()
{
    for (int i = 0; i < d_numPools; ++i) {
        d_pools_p[i].release();
    }
    d_blockList.release();
}

void Multipool::reserveCapacity(bsls::Types::size_type size, int numBlocks)
{
    BSLS_ASSERT(size <= d_maxBlockSize);
    BSLS_ASSERT(0    <= numBlocks);

    if (BSLS_PERFORMANCEHINT_PREDICT_LIKELY(size)) {
        const int pool = findPool(size);
        d_pools_p[pool].reserveCapacity(numBlocks);
    }
}

}  // close package namespace
}  // close enterprise namespace

// ----------------------------------------------------------------------------
// Copyright 2016 Bloomberg Finance L.P.
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
