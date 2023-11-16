import logging
import types
import weakref
from dataclasses import dataclass
from . import config
log = logging.getLogger(__name__)
"\n[Note on cache size limit]\n\nBackground - TorchDynamo cache is a linked list. Each cache entry is a\n(check_fn, out_code, next pointer). These are stored on the f_code's co_extra\nscratch space. When a frame is invoked, we walk this linked list and run\ncheck_fn in each cache_entry to decide if the frame needs recompilation. If none\nof the check_fn's returns True, we recompile and add a new entry. To ensure we\ndon't end up recompiling infinitely, we put limits on the cache size.\n\nThere are two limits\n1) cache_size_limit\n2) accumulated_cache_size_limit\n\n\nEarlier we used to have only limit - maximum number of entries in 1 cache line\n(which is now represented by (2) above). So, why do we need two limits? Lets try\nto understand that.\n\nIn general, we want our cache limit value to be a small number (e.g. 8 or even\nlower). This ensures that for frames that cause too many recompilation fall to\neager quickly. However, there is another problem that prevents us from lowering\nthe value of cache_size_limit. This is due to ID_MATCH'd guards. Today, we put\nID_MATCH guards on nn module if there is a graph break. This means we will have\nmany recompilations for the same code object because the ID_MATCH guard fails\nfor different instances of the nn module. This is a common pattern in how models\nare authored. Therefore, this requires us to keep the cache_size_limit high.\n\nWe resolve this by introducing these two limits. The first limit (1) limits the\nnumber of cache entries that have an ID_MATCH'd guard for an nn module instance.\nAnd, (2)nd limit becomes a safeguard mechanism to have a maximum compilations\nfor a code object. One important question is - what is the limit for the code\nobject that does not have any ID_MATCH guard? For such code objects, we choose\n(1) as the cache size limit.\n\nLets take an example to understand how these limits help. Suppose, we have 16\ninstances of a nn module and we ID_MATCH on the self object. Further, suppose\nthe inputs to these functions have varying batch size, leading to one\nrecompilation. In total, there will be 32 recompilations, and therefore 32 cache\nentries on the forward code object. In the older case when we had only 1 limit,\nour cache size limit must be >= 32 to capture all these recompilations. Now,\nsuppose there is a separate function in the same program which is very dynamic\nand unsuitable for compilation. Such a function will need to undergo 32\ncompilations to burst the cache and fallback to eager. These 32 recompilations\nare too many and we want to fallback for these compilation-unfriendly functions\nsooner.\n\nIn the new scenario, we can have (1) cache_size_limit = 2, (2)\naccumulated_cache_size_limit = 32. This means that each ID_MATCH'd object can\nhave maximum of two cache entries, and the maximum number of cache entries\n(irrespective of ID_MATCH obj) is 32. This covers the case of forward code\nobject which has 32 recompilations. For the other function, the one unsuitable\nfor recompilation, our limit is 2. So, we will burst the cache in just 2\nrecompilations. In this manner, these 2 limits help us resolve the tension\nmentioned earlier.\n"

@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """
    num_cache_entries: int = 0
    num_cache_entries_with_same_id_matched_objs: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        if False:
            print('Hello World!')
        return self.num_cache_entries >= config.accumulated_cache_size_limit or self.num_cache_entries_with_same_id_matched_objs >= limit

def _get_weakref_from_f_locals(frame: types.FrameType, local_name: str):
    if False:
        print('Hello World!')
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass
    return weak_id

def _has_same_id_matched_objs(frame: types.FrameType, cache_entry) -> bool:
    if False:
        while True:
            i = 10
    "\n    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones\n    in frame.f_locals.\n    "
    if not cache_entry:
        return False
    for (local_name, weakref_from_cache_entry) in cache_entry.check_fn.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame != weakref_from_cache_entry:
                return False
    return True

def compute_cache_size(frame: types.FrameType, cache_entry) -> CacheSizeRelevantForFrame:
    if False:
        return 10
    num_cache_entries = 0
    num_cache_entries_with_same_id_matched_objs = 0
    while cache_entry:
        num_cache_entries += 1
        if _has_same_id_matched_objs(frame, cache_entry):
            num_cache_entries_with_same_id_matched_objs += 1
        cache_entry = cache_entry.next
    return CacheSizeRelevantForFrame(num_cache_entries, num_cache_entries_with_same_id_matched_objs)

def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    if False:
        print('Hello World!')
    "\n    If the frame (earlier parsed by compute_cache_size) has more than 1 cache\n    entry with same ID_MATCH'd objects, then its a recompilation.\n    "
    return cache_size.will_compilation_exceed(1)

def exceeds_cache_size_limit(cache_size: CacheSizeRelevantForFrame) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Checks if we are exceeding the cache size limit.\n    '
    return cache_size.will_compilation_exceed(config.cache_size_limit)