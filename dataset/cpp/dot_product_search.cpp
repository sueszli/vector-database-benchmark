// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "dot_product_search.h"
#include "iterator_pack.h"
#include <vespa/vespalib/objects/visit.h>

using search::fef::TermFieldMatchData;
using search::fef::MatchData;
using vespalib::ObjectVisitor;

namespace search::queryeval {

template <typename HEAP, typename IteratorPack>
class DotProductSearchImpl : public DotProductSearch
{
private:
    using ref_t = uint32_t;
    using Weights = vespalib::ConstArrayRef<int32_t>;

    struct CmpDocId {
        const uint32_t *termPos;
        CmpDocId(const uint32_t *tp) noexcept : termPos(tp) {}
        bool operator()(const ref_t &a, const ref_t &b) const noexcept {
            return (termPos[a] < termPos[b]);
        }
    };

    TermFieldMatchData     &_tmd;
    Weights                 _weights;
    std::vector<uint32_t>   _termPos;
    CmpDocId                _cmpDocId;
    std::vector<ref_t>      _data_space;
    ref_t                  *_data_begin;
    ref_t                  *_data_stash;
    ref_t                  *_data_end;
    IteratorPack            _children;
    bool                    _field_is_filter;

    void seek_child(ref_t child, uint32_t docId) {
        _termPos[child] = _children.seek(child, docId);
    }

public:
    DotProductSearchImpl(TermFieldMatchData &tmd,
                         bool field_is_filter,
                         const std::vector<int32_t> &weights,
                         IteratorPack &&iteratorPack)
        : _tmd(tmd),
          _weights(weights),
          _termPos(weights.size()),
          _cmpDocId(&_termPos[0]),
          _data_space(),
          _data_begin(nullptr),
          _data_stash(nullptr),
          _data_end(nullptr),
          _children(std::move(iteratorPack)),
          _field_is_filter(field_is_filter)
    {
        HEAP::require_left_heap();
        assert(_weights.size() > 0);
        assert(_weights.size() == _children.size());
        _data_space.reserve(_weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            _data_space.push_back(i);
        }
        _data_begin = &_data_space[0];
        _data_end = _data_begin + _data_space.size();
        if (_field_is_filter || _tmd.isNotNeeded()) {
            _tmd.setRawScore(TermFieldMatchData::invalidId(), 0.0);
        }
    }

    void doSeek(uint32_t docId) override {
        while (_data_stash < _data_end) {
            seek_child(*_data_stash, docId);
            HEAP::push(_data_begin, ++_data_stash, _cmpDocId);
        }
        while (_termPos[HEAP::front(_data_begin, _data_stash)] < docId) {
            seek_child(HEAP::front(_data_begin, _data_stash), docId);
            HEAP::adjust(_data_begin, _data_stash, _cmpDocId);
        }
        setDocId(_termPos[HEAP::front(_data_begin, _data_stash)]);
    }

    void doUnpack(uint32_t docId) override {
        if (!_field_is_filter && !_tmd.isNotNeeded()) {
            feature_t score = 0.0;
            while ((_data_begin < _data_stash) &&
                   _termPos[HEAP::front(_data_begin, _data_stash)] == docId)
            {
                HEAP::pop(_data_begin, _data_stash--, _cmpDocId);
                const ref_t child = *_data_stash;
                double tmp = _weights[child];
                tmp *= _children.get_weight(child, docId);
                score += tmp;
            };
            _tmd.setRawScore(docId, score);
        } else {
            _tmd.resetOnlyDocId(docId);
        }
    }

    void initRange(uint32_t begin, uint32_t end) override {
        DotProductSearch::initRange(begin, end);
        _children.initRange(begin, end);
        for (size_t i = 0; i < _children.size(); ++i) {
            _termPos[i] = _children.get_docid(i);
        }
        _data_stash = _data_begin;
        while (_data_stash < _data_end) {
            HEAP::push(_data_begin, ++_data_stash, _cmpDocId);
        }
    }
    Trinary is_strict() const final { return Trinary::True; }

    void visitMembers(vespalib::ObjectVisitor &) const override {}
};

class SingleTermDotProductSearch : public DotProductSearch {
public:
    SingleTermDotProductSearch(TermFieldMatchData &tmd, SearchIterator::UP child,
                                const TermFieldMatchData &childTmd, feature_t weight, MatchData::UP md)
        : _child(std::move(child)),
          _childTmd(childTmd),
          _tmd(tmd),
          _weight(weight),
          _md(std::move(md))
    { }
private:
    void doSeek(uint32_t docid) override {
        _child->doSeek(docid);
        setDocId(_child->getDocId());
    }

    void doUnpack(uint32_t docid) override {
        _child->doUnpack(docid);
        _tmd.setRawScore(docid, _weight*_childTmd.getWeight());
    }

    void initRange(uint32_t beginId, uint32_t endId) override {
        SearchIterator::initRange(beginId, endId);
        _child->initRange(beginId, endId);
        setDocId(_child->getDocId());
    }
    SearchIterator::UP        _child;
    const TermFieldMatchData &_childTmd;
    TermFieldMatchData       &_tmd;
    feature_t                 _weight;
    MatchData::UP             _md;
};

//-----------------------------------------------------------------------------


SearchIterator::UP
DotProductSearch::create(const std::vector<SearchIterator*> &children,
                         TermFieldMatchData &tmd,
                         bool field_is_filter,
                         const std::vector<TermFieldMatchData*> &childMatch,
                         const std::vector<int32_t> &weights,
                         MatchData::UP md)
{
    using ArrayHeapImpl = DotProductSearchImpl<vespalib::LeftArrayHeap, SearchIteratorPack>;
    using HeapImpl = DotProductSearchImpl<vespalib::LeftHeap, SearchIteratorPack>;

    if (childMatch.size() == 1) {
        return std::make_unique<SingleTermDotProductSearch>(tmd, SearchIterator::UP(children[0]),
                                                             *childMatch[0], weights[0], std::move(md));
    }
    if (childMatch.size() < 128) {
        return std::make_unique<ArrayHeapImpl>(tmd, field_is_filter, weights, SearchIteratorPack(children, childMatch, std::move(md)));
    }
    return std::make_unique<HeapImpl>(tmd, field_is_filter, weights,  SearchIteratorPack(children, childMatch, std::move(md)));
}

//-----------------------------------------------------------------------------

SearchIterator::UP
DotProductSearch::create(TermFieldMatchData &tmd,
                         bool field_is_filter,
                         const std::vector<int32_t> &weights,
                         std::vector<DocumentWeightIterator> &&iterators)
{
    using ArrayHeapImpl = DotProductSearchImpl<vespalib::LeftArrayHeap, AttributeIteratorPack>;
    using HeapImpl = DotProductSearchImpl<vespalib::LeftHeap, AttributeIteratorPack>;

    if (iterators.size() < 128) {
        return std::make_unique<ArrayHeapImpl>(tmd, field_is_filter, weights, AttributeIteratorPack(std::move(iterators)));
    }
    return std::make_unique<HeapImpl>(tmd, field_is_filter, weights, AttributeIteratorPack(std::move(iterators)));
}

//-----------------------------------------------------------------------------

}
