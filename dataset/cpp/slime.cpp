// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "slime.h"
#include "symbol_table.h"
#include <vespa/vespalib/util/stash.h>

namespace vespalib {

Slime::Slime(Params params)
    : _names(params.detachSymbols()),
      _stash(std::make_unique<Stash>(params.getChunkSize())),
      _root(_stash.get())
{ }

Slime::Slime(Slime &&rhs) noexcept = default;
Slime & Slime::operator=(Slime &&rhs) noexcept = default;
Slime::~Slime() = default;

std::unique_ptr<slime::SymbolTable>
Slime::reclaimSymbols(Slime &&rhs) {
    rhs._stash.reset();
    rhs._root = RootValue(nullptr);
    return std::move(rhs._names);
}

bool operator == (const Slime & a, const Slime & b) noexcept
{
    return a.get() == b.get();
}

std::ostream & operator << (std::ostream & os, const Slime & slime)
{
    os << slime.toString();
    return os;
}

} // namespace vespalib
