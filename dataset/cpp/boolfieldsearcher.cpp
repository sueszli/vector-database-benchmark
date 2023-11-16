// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "boolfieldsearcher.h"
#include <vespa/document/fieldvalue/boolfieldvalue.h>

using search::streaming::QueryTerm;
using search::streaming::QueryTermList;

namespace vsm {

namespace {
vespalib::stringref TRUE = "true";
vespalib::stringref FALSE = "false";
}

std::unique_ptr<FieldSearcher>
BoolFieldSearcher::duplicate() const
{
    return std::make_unique<BoolFieldSearcher>(*this);
}

BoolFieldSearcher::BoolFieldSearcher(FieldIdT fId) :
    FieldSearcher(fId),
    _terms()
{ }

BoolFieldSearcher::~BoolFieldSearcher() = default;

void BoolFieldSearcher::prepare(search::streaming::QueryTermList& qtl,
                                const SharedSearcherBuf& buf,
                                const vsm::FieldPathMapT& field_paths,
                                search::fef::IQueryEnvironment& query_env)
{
    _terms.clear();
    FieldSearcher::prepare(qtl, buf, field_paths, query_env);
    for (const QueryTerm * qt : qtl) {
        if (TRUE == qt->getTerm()) {
            _terms.push_back(true);
        } else if (FALSE == qt->getTerm()) {
            _terms.push_back(false);
        } else {
            int64_t low;
            int64_t high;
            bool valid = qt->getAsIntegerTerm(low, high);
            _terms.push_back(valid && (low > 0));
        }
    }
}

void BoolFieldSearcher::onValue(const document::FieldValue & fv)
{
    for(size_t j=0, jm(_terms.size()); j < jm; j++) {
        if (static_cast<const document::BoolFieldValue &>(fv).getValue() == _terms[j]) {
            addHit(*_qtl[j], 0);
        }
    }
    ++_words;
}

}
