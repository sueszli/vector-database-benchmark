// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "lidvectorcontext.h"
#include <vespa/searchlib/common/bitvector.h>
#include <vespa/searchlib/common/allocatedbitvector.h>
#include <vespa/vespalib/objects/nbostream.h>
#include <cassert>

#include <vespa/log/log.h>
LOG_SETUP(".proton.feedoperation.lidvectorcontext");

using search::BitVector;
using search::AllocatedBitVector;

namespace proton {

LidVectorContext::LidVectorContext(size_t docIdLimit)
    : _result(),
      _docIdLimit(docIdLimit)
{
}


LidVectorContext::LidVectorContext()
    : _result(),
      _docIdLimit(0)
{
}


LidVectorContext::LidVectorContext(size_t docIdLimit,
                                   const LidVector &lids)
    : _result(lids),
      _docIdLimit(docIdLimit)
{
}


void
LidVectorContext::addLid(const search::DocumentIdT lid)
{
    _result.push_back(lid);
}


void
LidVectorContext::serialize(vespalib::nbostream &os) const
{
    LOG(debug, "serialize: _result.size() = %ld, _docIdLimit = %ld",
        _result.size(), _docIdLimit);
    os << static_cast<uint64_t>(_docIdLimit);
    // Use of bitvector when > 1/32 of docs
    if (_result.size() > (_docIdLimit / 32)) {
        os << static_cast<int32_t>(BITVECTOR);
        auto bitVector = BitVector::create(_docIdLimit);
        for (auto docid : _result) {
            bitVector->setBit(docid);
        }
        os << *bitVector;
    } else {
        os << static_cast<int32_t>(ARRAY);
        os << _result;
    }
}


void
LidVectorContext::deserialize(vespalib::nbostream &is)
{
    int32_t format;
    uint64_t docIdLimit;
    is >> docIdLimit;
    _docIdLimit = docIdLimit;
    is >> format;
    LOG(debug, "deserialize: format = %d", format);
    // Use of bitvector when > 1/32 of docs
    if (format == BITVECTOR) {
        auto bitVector = std::make_unique<AllocatedBitVector>(_docIdLimit);
        is >> *bitVector;
        uint32_t sz(bitVector->size());
        assert(sz == _docIdLimit);
        LOG(spam, "deserialize: reading bitvector of size %u", sz);
        for (search::DocumentIdT lid(bitVector->getFirstTrueBit());
             lid < sz;
             lid = bitVector->getNextTrueBit(lid + 1)) {

            _result.push_back(lid);
        }
    } else if (format == ARRAY) {
        is >> _result;
    }
}


} // namespace proton
