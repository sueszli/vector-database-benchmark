// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "weighted_set_term_search.h"
#include <vespa/searchlib/common/bitvector.h>
#include <vespa/searchlib/attribute/document_weight_or_filter_search.h>
#include <vespa/vespalib/objects/visit.h>
#include <vespa/searchcommon/attribute/i_search_context.h>

#include "iterator_pack.h"
#include "blueprint.h"

using search::fef::TermFieldMatchData;
using vespalib::ObjectVisitor;

namespace search::queryeval {

template <typename HEAP, typename IteratorPack>
class WeightedSetTermSearchImpl : public WeightedSetTermSearch
{
private:
    using ref_t = uint32_t;

    struct CmpDocId {
        const uint32_t *termPos;
        explicit CmpDocId(const uint32_t *tp) : termPos(tp) {}
        bool operator()(const ref_t &a, const ref_t &b) const {
            return (termPos[a] < termPos[b]);
        }
    };

    struct CmpWeight {
        const int32_t *weight;
        explicit CmpWeight(const int32_t *w) : weight(w) {}
        bool operator()(const ref_t &a, const ref_t &b) const {
            return (weight[a] > weight[b]);
        }
    };

    fef::TermFieldMatchData                       &_tmd;
    const std::vector<int32_t>                    &_weights;
    std::vector<uint32_t>                          _termPos;
    CmpDocId                                       _cmpDocId;
    CmpWeight                                      _cmpWeight;
    std::vector<ref_t>                             _data_space;
    ref_t                                         *_data_begin;
    ref_t                                         *_data_stash;
    ref_t                                         *_data_end;
    IteratorPack                                   _children;
    bool                                           _need_match_data;

    void seek_child(ref_t child, uint32_t docId) {
        _termPos[child] = _children.seek(child, docId);
    }
    void get_matching_elements_child(ref_t child, uint32_t docId, const std::vector<Blueprint::UP> &child_blueprints, std::vector<uint32_t> &dst) {
        auto *sc = child_blueprints[child]->get_attribute_search_context();
        if (sc != nullptr) {
            int32_t weight(0);
            for (int32_t id = sc->find(docId, 0, weight); id >= 0; id = sc->find(docId, id + 1, weight)) {
                dst.push_back(id);
            }
        }
    }

public:
    WeightedSetTermSearchImpl(fef::TermFieldMatchData &tmd,
                              bool field_is_filter,
                              const std::vector<int32_t> &weights,
                              IteratorPack &&iteratorPack)
        : _tmd(tmd),
          _weights(weights),
          _termPos(weights.size()),
          _cmpDocId(&_termPos[0]),
          _cmpWeight(&_weights[0]),
          _data_space(),
          _data_begin(nullptr),
          _data_stash(nullptr),
          _data_end(nullptr),
          _children(std::move(iteratorPack)),
          _need_match_data(!field_is_filter && !_tmd.isNotNeeded())
    {
        HEAP::require_left_heap();
        assert(_children.size() > 0);
        assert(_children.size() == _weights.size());
        _data_space.reserve(_children.size());
        for (size_t i = 0; i < _children.size(); ++i) {
            _data_space.push_back(i);
        }
        _data_begin = &_data_space[0];
        _data_end = _data_begin + _data_space.size();
        if (_need_match_data) {
            _tmd.reservePositions(_children.size());
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

    void pop_matching_children(uint32_t docId) {
        while ((_data_begin < _data_stash) &&
               _termPos[HEAP::front(_data_begin, _data_stash)] == docId)
        {
            HEAP::pop(_data_begin, _data_stash--, _cmpDocId);
        }
    }

    void doUnpack(uint32_t docId) override {
        if (_need_match_data) {
            _tmd.reset(docId);
            pop_matching_children(docId);
            std::sort(_data_stash, _data_end, _cmpWeight);
            for (ref_t *ptr = _data_stash; ptr < _data_end; ++ptr) {
                fef::TermFieldMatchDataPosition pos;
                pos.setElementWeight(_weights[*ptr]);
                _tmd.appendPosition(pos);
            }
        } else {
            _tmd.resetOnlyDocId(docId);
        }
    }

    void initRange(uint32_t begin, uint32_t end) override {
        WeightedSetTermSearch::initRange(begin, end);
        _children.initRange(begin, end);
        for (size_t i = 0; i < _children.size(); ++i) {
            _termPos[i] = _children.get_docid(i);
        }
        _data_stash = _data_begin;
        while (_data_stash < _data_end) {
            HEAP::push(_data_begin, ++_data_stash, _cmpDocId);
        }
    }
    Trinary is_strict() const override { return Trinary::True; }

    void visitMembers(vespalib::ObjectVisitor &) const override { }

    BitVector::UP get_hits(uint32_t begin_id) override {
        return _children.get_hits(begin_id, getEndId());
    }

    void or_hits_into(BitVector &result, uint32_t begin_id) override {
        _children.or_hits_into(result, begin_id);
    }
    void and_hits_into(BitVector &result, uint32_t begin_id) override {
        result.andWith(*get_hits(begin_id));
    }
    void find_matching_elements(uint32_t docId, const std::vector<Blueprint::UP> &child_blueprints, std::vector<uint32_t> &dst) override {
        pop_matching_children(docId);
        for (ref_t *ptr = _data_stash; ptr < _data_end; ++ptr) {
            get_matching_elements_child(*ptr, docId, child_blueprints, dst);
        }
    }
};

//-----------------------------------------------------------------------------

SearchIterator::UP
WeightedSetTermSearch::create(const std::vector<SearchIterator *> &children,
                              TermFieldMatchData &tmd,
                              bool field_is_filter,
                              const std::vector<int32_t> &weights,
                              fef::MatchData::UP match_data)
{
    using ArrayHeapImpl = WeightedSetTermSearchImpl<vespalib::LeftArrayHeap, SearchIteratorPack>;
    using HeapImpl = WeightedSetTermSearchImpl<vespalib::LeftHeap, SearchIteratorPack>;

    if (tmd.isNotNeeded()) {
        return attribute::DocumentWeightOrFilterSearch::create(children, std::move(match_data));
    }

    if (children.size() < 128) {
        return SearchIterator::UP(new ArrayHeapImpl(tmd, field_is_filter, weights, SearchIteratorPack(children, std::move(match_data))));
    }
    return SearchIterator::UP(new HeapImpl(tmd, field_is_filter, weights, SearchIteratorPack(children, std::move(match_data))));
}

//-----------------------------------------------------------------------------

SearchIterator::UP
WeightedSetTermSearch::create(fef::TermFieldMatchData &tmd,
                              bool field_is_filter,
                              const std::vector<int32_t> &weights,
                              std::vector<DocumentWeightIterator> &&iterators)
{
    using ArrayHeapImpl = WeightedSetTermSearchImpl<vespalib::LeftArrayHeap, AttributeIteratorPack>;
    using HeapImpl = WeightedSetTermSearchImpl<vespalib::LeftHeap, AttributeIteratorPack>;

    if (iterators.size() < 128) {
        return SearchIterator::UP(new ArrayHeapImpl(tmd, field_is_filter, weights, AttributeIteratorPack(std::move(iterators))));
    }
    return SearchIterator::UP(new HeapImpl(tmd, field_is_filter, weights, AttributeIteratorPack(std::move(iterators))));
}

//-----------------------------------------------------------------------------

}
