// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "single_string_enum_hint_search_context.h"
#include <vespa/searchlib/query/query_term_ucs4.h>

namespace search::attribute {

SingleStringEnumHintSearchContext::SingleStringEnumHintSearchContext(std::unique_ptr<QueryTermSimple> qTerm, bool cased,
                                                                     vespalib::FuzzyMatchingAlgorithm fuzzy_matching_algorithm,
                                                                     const AttributeVector& toBeSearched,
                                                                     EnumIndices enum_indices,
                                                                     const EnumStoreT<const char*>& enum_store,
                                                                     uint64_t num_values)
    : SingleStringEnumSearchContext(std::move(qTerm), cased, fuzzy_matching_algorithm, toBeSearched, enum_indices, enum_store),
      EnumHintSearchContext(enum_store.get_dictionary(),
                            enum_indices.size(), num_values)
{
    setup_enum_hint_sc(enum_store, *this);
}

SingleStringEnumHintSearchContext::~SingleStringEnumHintSearchContext() = default;

}
