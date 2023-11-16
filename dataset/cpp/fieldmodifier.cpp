// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "fieldmodifier.h"
#include <vespa/vespalib/stllike/hash_map.hpp>

namespace vsm {

FieldModifierMap::FieldModifierMap() :
    _map()
{ }

FieldModifierMap::~FieldModifierMap() { }

FieldModifier *
FieldModifierMap::getModifier(FieldIdT fId) const
{
    auto itr = _map.find(fId);
    if (itr == _map.end()) {
        return nullptr;
    }
    return itr->second.get();
}

}
