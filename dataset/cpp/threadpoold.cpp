// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespamalloc/malloc/threadpool.hpp>
#include <vespamalloc/malloc/memblockboundscheck_d.h>
#include <vespamalloc/malloc/stat.h>

namespace vespamalloc {

template class ThreadPoolT<MemBlockBoundsCheck, Stat>;

}
