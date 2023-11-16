// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespamalloc/malloc/threadlist.hpp>
#include <vespamalloc/malloc/memblock.h>
#include <vespamalloc/malloc/stat.h>

namespace vespamalloc {

template class ThreadListT<MemBlock, NoStat>;

}
