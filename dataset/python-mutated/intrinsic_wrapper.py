from .decorators import jit
import numba

@jit(device=True)
def all_sync(mask, predicate):
    if False:
        return 10
    '\n    If for all threads in the masked warp the predicate is true, then\n    a non-zero value is returned, otherwise 0 is returned.\n    '
    return numba.cuda.vote_sync_intrinsic(mask, 0, predicate)[1]

@jit(device=True)
def any_sync(mask, predicate):
    if False:
        while True:
            i = 10
    '\n    If for any thread in the masked warp the predicate is true, then\n    a non-zero value is returned, otherwise 0 is returned.\n    '
    return numba.cuda.vote_sync_intrinsic(mask, 1, predicate)[1]

@jit(device=True)
def eq_sync(mask, predicate):
    if False:
        while True:
            i = 10
    '\n    If for all threads in the masked warp the boolean predicate is the same,\n    then a non-zero value is returned, otherwise 0 is returned.\n    '
    return numba.cuda.vote_sync_intrinsic(mask, 2, predicate)[1]

@jit(device=True)
def ballot_sync(mask, predicate):
    if False:
        print('Hello World!')
    '\n    Returns a mask of all threads in the warp whose predicate is true,\n    and are within the given mask.\n    '
    return numba.cuda.vote_sync_intrinsic(mask, 3, predicate)[0]

@jit(device=True)
def shfl_sync(mask, value, src_lane):
    if False:
        i = 10
        return i + 15
    '\n    Shuffles value across the masked warp and returns the value\n    from src_lane. If this is outside the warp, then the\n    given value is returned.\n    '
    return numba.cuda.shfl_sync_intrinsic(mask, 0, value, src_lane, 31)[0]

@jit(device=True)
def shfl_up_sync(mask, value, delta):
    if False:
        print('Hello World!')
    '\n    Shuffles value across the masked warp and returns the value\n    from (laneid - delta). If this is outside the warp, then the\n    given value is returned.\n    '
    return numba.cuda.shfl_sync_intrinsic(mask, 1, value, delta, 0)[0]

@jit(device=True)
def shfl_down_sync(mask, value, delta):
    if False:
        print('Hello World!')
    '\n    Shuffles value across the masked warp and returns the value\n    from (laneid + delta). If this is outside the warp, then the\n    given value is returned.\n    '
    return numba.cuda.shfl_sync_intrinsic(mask, 2, value, delta, 31)[0]

@jit(device=True)
def shfl_xor_sync(mask, value, lane_mask):
    if False:
        return 10
    '\n    Shuffles value across the masked warp and returns the value\n    from (laneid ^ lane_mask).\n    '
    return numba.cuda.shfl_sync_intrinsic(mask, 3, value, lane_mask, 31)[0]