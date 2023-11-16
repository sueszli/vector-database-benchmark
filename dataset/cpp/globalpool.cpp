// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespamalloc/malloc/globalpool.hpp>
#include <vespamalloc/malloc/memblock.h>

namespace vespamalloc {

template class AllocPoolT<MemBlock>;

}
