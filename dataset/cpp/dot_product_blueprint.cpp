// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "dot_product_blueprint.h"
#include "dot_product_search.h"
#include "field_spec.hpp"
#include <vespa/vespalib/objects/visit.hpp>

namespace search::queryeval {

DotProductBlueprint::DotProductBlueprint(const FieldSpec &field)
    : ComplexLeafBlueprint(field),
      _layout(),
      _weights(),
      _terms()
{ }

DotProductBlueprint::~DotProductBlueprint() = default;

void
DotProductBlueprint::reserve(size_t num_children) {
    _weights.reserve(num_children);
    _terms.reserve(num_children);
    _layout.reserve(num_children);
}

void
DotProductBlueprint::addTerm(Blueprint::UP term, int32_t weight, HitEstimate & estimate)
{
    HitEstimate childEst = term->getState().estimate();
    if (! childEst.empty) {
        if (estimate.empty) {
            estimate = childEst;
        } else {
            estimate.estHits += childEst.estHits;
        }
    }
    _weights.push_back(weight);
    _terms.push_back(std::move(term));
}

SearchIterator::UP
DotProductBlueprint::createLeafSearch(const search::fef::TermFieldMatchDataArray &tfmda, bool) const
{
    assert(tfmda.size() == 1);
    assert(getState().numFields() == 1);
    fef::MatchData::UP md = _layout.createMatchData();
    std::vector<fef::TermFieldMatchData*> childMatch;
    std::vector<SearchIterator*> children(_terms.size());
    for (size_t i = 0; i < _terms.size(); ++i) {
        const State &childState = _terms[i]->getState();
        assert(childState.numFields() == 1);
        childMatch.push_back(childState.field(0).resolve(*md));
        // TODO: pass ownership with unique_ptr
        children[i] = _terms[i]->createSearch(*md, true).release();
    }
    bool field_is_filter = getState().fields()[0].isFilter();
    return DotProductSearch::create(children, *tfmda[0], field_is_filter, childMatch, _weights, std::move(md));
}

SearchIterator::UP
DotProductBlueprint::createFilterSearch(bool strict, FilterConstraint constraint) const
{
    return create_or_filter(_terms, strict, constraint);
}

void
DotProductBlueprint::fetchPostings(const ExecuteInfo &execInfo)
{
    ExecuteInfo childInfo = ExecuteInfo::create(true, execInfo);
    for (size_t i = 0; i < _terms.size(); ++i) {
        _terms[i]->fetchPostings(childInfo);
    }    
}

void
DotProductBlueprint::visitMembers(vespalib::ObjectVisitor &visitor) const
{
    LeafBlueprint::visitMembers(visitor);
    visit(visitor, "_weights", _weights);
    visit(visitor, "_terms", _terms);
}

}
