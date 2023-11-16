"""
Timsort implementation.  Mostly adapted from CPython's listobject.c.

For more information, see listsort.txt in CPython's source tree.
"""
import collections
from numba.core import types
TimsortImplementation = collections.namedtuple('TimsortImplementation', ('compile', 'count_run', 'binarysort', 'gallop_left', 'gallop_right', 'merge_init', 'merge_append', 'merge_pop', 'merge_compute_minrun', 'merge_lo', 'merge_hi', 'merge_at', 'merge_force_collapse', 'merge_collapse', 'run_timsort', 'run_timsort_with_values'))
MAX_MERGE_PENDING = 85
MIN_GALLOP = 7
MERGESTATE_TEMP_SIZE = 256
MergeState = collections.namedtuple('MergeState', ('min_gallop', 'keys', 'values', 'pending', 'n'))
MergeRun = collections.namedtuple('MergeRun', ('start', 'size'))

def make_timsort_impl(wrap, make_temp_area):
    if False:
        i = 10
        return i + 15
    make_temp_area = wrap(make_temp_area)
    intp = types.intp
    zero = intp(0)

    @wrap
    def has_values(keys, values):
        if False:
            print('Hello World!')
        return values is not keys

    @wrap
    def merge_init(keys):
        if False:
            print('Hello World!')
        '\n        Initialize a MergeState for a non-keyed sort.\n        '
        temp_size = min(len(keys) // 2 + 1, MERGESTATE_TEMP_SIZE)
        temp_keys = make_temp_area(keys, temp_size)
        temp_values = temp_keys
        pending = [MergeRun(zero, zero)] * MAX_MERGE_PENDING
        return MergeState(intp(MIN_GALLOP), temp_keys, temp_values, pending, zero)

    @wrap
    def merge_init_with_values(keys, values):
        if False:
            while True:
                i = 10
        '\n        Initialize a MergeState for a keyed sort.\n        '
        temp_size = min(len(keys) // 2 + 1, MERGESTATE_TEMP_SIZE)
        temp_keys = make_temp_area(keys, temp_size)
        temp_values = make_temp_area(values, temp_size)
        pending = [MergeRun(zero, zero)] * MAX_MERGE_PENDING
        return MergeState(intp(MIN_GALLOP), temp_keys, temp_values, pending, zero)

    @wrap
    def merge_append(ms, run):
        if False:
            return 10
        '\n        Append a run on the merge stack.\n        '
        n = ms.n
        assert n < MAX_MERGE_PENDING
        ms.pending[n] = run
        return MergeState(ms.min_gallop, ms.keys, ms.values, ms.pending, n + 1)

    @wrap
    def merge_pop(ms):
        if False:
            print('Hello World!')
        '\n        Pop the top run from the merge stack.\n        '
        return MergeState(ms.min_gallop, ms.keys, ms.values, ms.pending, ms.n - 1)

    @wrap
    def merge_getmem(ms, need):
        if False:
            while True:
                i = 10
        "\n        Ensure enough temp memory for 'need' items is available.\n        "
        alloced = len(ms.keys)
        if need <= alloced:
            return ms
        while alloced < need:
            alloced = alloced << 1
        temp_keys = make_temp_area(ms.keys, alloced)
        if has_values(ms.keys, ms.values):
            temp_values = make_temp_area(ms.values, alloced)
        else:
            temp_values = temp_keys
        return MergeState(ms.min_gallop, temp_keys, temp_values, ms.pending, ms.n)

    @wrap
    def merge_adjust_gallop(ms, new_gallop):
        if False:
            for i in range(10):
                print('nop')
        "\n        Modify the MergeState's min_gallop.\n        "
        return MergeState(intp(new_gallop), ms.keys, ms.values, ms.pending, ms.n)

    @wrap
    def LT(a, b):
        if False:
            print('Hello World!')
        '\n        Trivial comparison function between two keys.  This is factored out to\n        make it clear where comparisons occur.\n        '
        return a < b

    @wrap
    def binarysort(keys, values, lo, hi, start):
        if False:
            while True:
                i = 10
        "\n        binarysort is the best method for sorting small arrays: it does\n        few compares, but can do data movement quadratic in the number of\n        elements.\n        [lo, hi) is a contiguous slice of a list, and is sorted via\n        binary insertion.  This sort is stable.\n        On entry, must have lo <= start <= hi, and that [lo, start) is already\n        sorted (pass start == lo if you don't know!).\n        "
        assert lo <= start and start <= hi
        _has_values = has_values(keys, values)
        if lo == start:
            start += 1
        while start < hi:
            pivot = keys[start]
            l = lo
            r = start
            while l < r:
                p = l + (r - l >> 1)
                if LT(pivot, keys[p]):
                    r = p
                else:
                    l = p + 1
            for p in range(start, l, -1):
                keys[p] = keys[p - 1]
            keys[l] = pivot
            if _has_values:
                pivot_val = values[start]
                for p in range(start, l, -1):
                    values[p] = values[p - 1]
                values[l] = pivot_val
            start += 1

    @wrap
    def count_run(keys, lo, hi):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the length of the run beginning at lo, in the slice [lo, hi).\n        lo < hi is required on entry.  "A run" is the longest ascending sequence, with\n\n            lo[0] <= lo[1] <= lo[2] <= ...\n\n        or the longest descending sequence, with\n\n            lo[0] > lo[1] > lo[2] > ...\n\n        A tuple (length, descending) is returned, where boolean *descending*\n        is set to 0 in the former case, or to 1 in the latter.\n        For its intended use in a stable mergesort, the strictness of the defn of\n        "descending" is needed so that the caller can safely reverse a descending\n        sequence without violating stability (strict > ensures there are no equal\n        elements to get out of order).\n        '
        assert lo < hi
        if lo + 1 == hi:
            return (1, False)
        if LT(keys[lo + 1], keys[lo]):
            for k in range(lo + 2, hi):
                if not LT(keys[k], keys[k - 1]):
                    return (k - lo, True)
            return (hi - lo, True)
        else:
            for k in range(lo + 2, hi):
                if LT(keys[k], keys[k - 1]):
                    return (k - lo, False)
            return (hi - lo, False)

    @wrap
    def gallop_left(key, a, start, stop, hint):
        if False:
            while True:
                i = 10
        '\n        Locate the proper position of key in a sorted vector; if the vector contains\n        an element equal to key, return the position immediately to the left of\n        the leftmost equal element.  [gallop_right() does the same except returns\n        the position to the right of the rightmost equal element (if any).]\n\n        "a" is a sorted vector with stop elements, starting at a[start].\n        stop must be > start.\n\n        "hint" is an index at which to begin the search, start <= hint < stop.\n        The closer hint is to the final result, the faster this runs.\n\n        The return value is the int k in start..stop such that\n\n            a[k-1] < key <= a[k]\n\n        pretending that a[start-1] is minus infinity and a[stop] is plus infinity.\n        IOW, key belongs at index k; or, IOW, the first k elements of a should\n        precede key, and the last stop-start-k should follow key.\n\n        See listsort.txt for info on the method.\n        '
        assert stop > start
        assert hint >= start and hint < stop
        n = stop - start
        lastofs = 0
        ofs = 1
        if LT(a[hint], key):
            maxofs = stop - hint
            while ofs < maxofs:
                if LT(a[hint + ofs], key):
                    lastofs = ofs
                    ofs = (ofs << 1) + 1
                    if ofs <= 0:
                        ofs = maxofs
                else:
                    break
            if ofs > maxofs:
                ofs = maxofs
            lastofs += hint
            ofs += hint
        else:
            maxofs = hint - start + 1
            while ofs < maxofs:
                if LT(a[hint - ofs], key):
                    break
                else:
                    lastofs = ofs
                    ofs = (ofs << 1) + 1
                    if ofs <= 0:
                        ofs = maxofs
            if ofs > maxofs:
                ofs = maxofs
            (lastofs, ofs) = (hint - ofs, hint - lastofs)
        assert start - 1 <= lastofs and lastofs < ofs and (ofs <= stop)
        lastofs += 1
        while lastofs < ofs:
            m = lastofs + (ofs - lastofs >> 1)
            if LT(a[m], key):
                lastofs = m + 1
            else:
                ofs = m
        return ofs

    @wrap
    def gallop_right(key, a, start, stop, hint):
        if False:
            while True:
                i = 10
        '\n        Exactly like gallop_left(), except that if key already exists in a[start:stop],\n        finds the position immediately to the right of the rightmost equal value.\n\n        The return value is the int k in start..stop such that\n\n            a[k-1] <= key < a[k]\n\n        The code duplication is massive, but this is enough different given that\n        we\'re sticking to "<" comparisons that it\'s much harder to follow if\n        written as one routine with yet another "left or right?" flag.\n        '
        assert stop > start
        assert hint >= start and hint < stop
        n = stop - start
        lastofs = 0
        ofs = 1
        if LT(key, a[hint]):
            maxofs = hint - start + 1
            while ofs < maxofs:
                if LT(key, a[hint - ofs]):
                    lastofs = ofs
                    ofs = (ofs << 1) + 1
                    if ofs <= 0:
                        ofs = maxofs
                else:
                    break
            if ofs > maxofs:
                ofs = maxofs
            (lastofs, ofs) = (hint - ofs, hint - lastofs)
        else:
            maxofs = stop - hint
            while ofs < maxofs:
                if LT(key, a[hint + ofs]):
                    break
                else:
                    lastofs = ofs
                    ofs = (ofs << 1) + 1
                    if ofs <= 0:
                        ofs = maxofs
            if ofs > maxofs:
                ofs = maxofs
            lastofs += hint
            ofs += hint
        assert start - 1 <= lastofs and lastofs < ofs and (ofs <= stop)
        lastofs += 1
        while lastofs < ofs:
            m = lastofs + (ofs - lastofs >> 1)
            if LT(key, a[m]):
                ofs = m
            else:
                lastofs = m + 1
        return ofs

    @wrap
    def merge_compute_minrun(n):
        if False:
            while True:
                i = 10
        "\n        Compute a good value for the minimum run length; natural runs shorter\n        than this are boosted artificially via binary insertion.\n\n        If n < 64, return n (it's too small to bother with fancy stuff).\n        Else if n is an exact power of 2, return 32.\n        Else return an int k, 32 <= k <= 64, such that n/k is close to, but\n        strictly less than, an exact power of 2.\n\n        See listsort.txt for more info.\n        "
        r = 0
        assert n >= 0
        while n >= 64:
            r |= n & 1
            n >>= 1
        return n + r

    @wrap
    def sortslice_copy(dest_keys, dest_values, dest_start, src_keys, src_values, src_start, nitems):
        if False:
            while True:
                i = 10
        '\n        Upwards memcpy().\n        '
        assert src_start >= 0
        assert dest_start >= 0
        for i in range(nitems):
            dest_keys[dest_start + i] = src_keys[src_start + i]
        if has_values(src_keys, src_values):
            for i in range(nitems):
                dest_values[dest_start + i] = src_values[src_start + i]

    @wrap
    def sortslice_copy_down(dest_keys, dest_values, dest_start, src_keys, src_values, src_start, nitems):
        if False:
            for i in range(10):
                print('nop')
        '\n        Downwards memcpy().\n        '
        assert src_start >= 0
        assert dest_start >= 0
        for i in range(nitems):
            dest_keys[dest_start - i] = src_keys[src_start - i]
        if has_values(src_keys, src_values):
            for i in range(nitems):
                dest_values[dest_start - i] = src_values[src_start - i]
    DO_GALLOP = 1

    @wrap
    def merge_lo(ms, keys, values, ssa, na, ssb, nb):
        if False:
            i = 10
            return i + 15
        '\n        Merge the na elements starting at ssa with the nb elements starting at\n        ssb = ssa + na in a stable way, in-place.  na and nb must be > 0,\n        and should have na <= nb. See listsort.txt for more info.\n\n        An updated MergeState is returned (with possibly a different min_gallop\n        or larger temp arrays).\n\n        NOTE: compared to CPython\'s timsort, the requirement that\n            "Must also have that keys[ssa + na - 1] belongs at the end of the merge"\n\n        is removed. This makes the code a bit simpler and easier to reason about.\n        '
        assert na > 0 and nb > 0 and (na <= nb)
        assert ssb == ssa + na
        ms = merge_getmem(ms, na)
        sortslice_copy(ms.keys, ms.values, 0, keys, values, ssa, na)
        a_keys = ms.keys
        a_values = ms.values
        b_keys = keys
        b_values = values
        dest = ssa
        ssa = 0
        _has_values = has_values(a_keys, a_values)
        min_gallop = ms.min_gallop
        while nb > 0 and na > 0:
            acount = 0
            bcount = 0
            while True:
                if LT(b_keys[ssb], a_keys[ssa]):
                    keys[dest] = b_keys[ssb]
                    if _has_values:
                        values[dest] = b_values[ssb]
                    dest += 1
                    ssb += 1
                    nb -= 1
                    if nb == 0:
                        break
                    bcount += 1
                    acount = 0
                    if bcount >= min_gallop:
                        break
                else:
                    keys[dest] = a_keys[ssa]
                    if _has_values:
                        values[dest] = a_values[ssa]
                    dest += 1
                    ssa += 1
                    na -= 1
                    if na == 0:
                        break
                    acount += 1
                    bcount = 0
                    if acount >= min_gallop:
                        break
            if DO_GALLOP and na > 0 and (nb > 0):
                min_gallop += 1
                while acount >= MIN_GALLOP or bcount >= MIN_GALLOP:
                    min_gallop -= min_gallop > 1
                    k = gallop_right(b_keys[ssb], a_keys, ssa, ssa + na, ssa)
                    k -= ssa
                    acount = k
                    if k > 0:
                        sortslice_copy(keys, values, dest, a_keys, a_values, ssa, k)
                        dest += k
                        ssa += k
                        na -= k
                        if na == 0:
                            break
                    keys[dest] = b_keys[ssb]
                    if _has_values:
                        values[dest] = b_values[ssb]
                    dest += 1
                    ssb += 1
                    nb -= 1
                    if nb == 0:
                        break
                    k = gallop_left(a_keys[ssa], b_keys, ssb, ssb + nb, ssb)
                    k -= ssb
                    bcount = k
                    if k > 0:
                        sortslice_copy(keys, values, dest, b_keys, b_values, ssb, k)
                        dest += k
                        ssb += k
                        nb -= k
                        if nb == 0:
                            break
                    keys[dest] = a_keys[ssa]
                    if _has_values:
                        values[dest] = a_values[ssa]
                    dest += 1
                    ssa += 1
                    na -= 1
                    if na == 0:
                        break
                min_gallop += 1
        if nb == 0:
            sortslice_copy(keys, values, dest, a_keys, a_values, ssa, na)
        else:
            assert na == 0
            assert dest == ssb
        return merge_adjust_gallop(ms, min_gallop)

    @wrap
    def merge_hi(ms, keys, values, ssa, na, ssb, nb):
        if False:
            print('Hello World!')
        '\n        Merge the na elements starting at ssa with the nb elements starting at\n        ssb = ssa + na in a stable way, in-place.  na and nb must be > 0,\n        and should have na >= nb.  See listsort.txt for more info.\n\n        An updated MergeState is returned (with possibly a different min_gallop\n        or larger temp arrays).\n\n        NOTE: compared to CPython\'s timsort, the requirement that\n            "Must also have that keys[ssa + na - 1] belongs at the end of the merge"\n\n        is removed. This makes the code a bit simpler and easier to reason about.\n        '
        assert na > 0 and nb > 0 and (na >= nb)
        assert ssb == ssa + na
        ms = merge_getmem(ms, nb)
        sortslice_copy(ms.keys, ms.values, 0, keys, values, ssb, nb)
        a_keys = keys
        a_values = values
        b_keys = ms.keys
        b_values = ms.values
        dest = ssb + nb - 1
        ssb = nb - 1
        ssa = ssa + na - 1
        _has_values = has_values(b_keys, b_values)
        min_gallop = ms.min_gallop
        while nb > 0 and na > 0:
            acount = 0
            bcount = 0
            while True:
                if LT(b_keys[ssb], a_keys[ssa]):
                    keys[dest] = a_keys[ssa]
                    if _has_values:
                        values[dest] = a_values[ssa]
                    dest -= 1
                    ssa -= 1
                    na -= 1
                    if na == 0:
                        break
                    acount += 1
                    bcount = 0
                    if acount >= min_gallop:
                        break
                else:
                    keys[dest] = b_keys[ssb]
                    if _has_values:
                        values[dest] = b_values[ssb]
                    dest -= 1
                    ssb -= 1
                    nb -= 1
                    if nb == 0:
                        break
                    bcount += 1
                    acount = 0
                    if bcount >= min_gallop:
                        break
            if DO_GALLOP and na > 0 and (nb > 0):
                min_gallop += 1
                while acount >= MIN_GALLOP or bcount >= MIN_GALLOP:
                    min_gallop -= min_gallop > 1
                    k = gallop_right(b_keys[ssb], a_keys, ssa - na + 1, ssa + 1, ssa)
                    k = ssa + 1 - k
                    acount = k
                    if k > 0:
                        sortslice_copy_down(keys, values, dest, a_keys, a_values, ssa, k)
                        dest -= k
                        ssa -= k
                        na -= k
                        if na == 0:
                            break
                    keys[dest] = b_keys[ssb]
                    if _has_values:
                        values[dest] = b_values[ssb]
                    dest -= 1
                    ssb -= 1
                    nb -= 1
                    if nb == 0:
                        break
                    k = gallop_left(a_keys[ssa], b_keys, ssb - nb + 1, ssb + 1, ssb)
                    k = ssb + 1 - k
                    bcount = k
                    if k > 0:
                        sortslice_copy_down(keys, values, dest, b_keys, b_values, ssb, k)
                        dest -= k
                        ssb -= k
                        nb -= k
                        if nb == 0:
                            break
                    keys[dest] = a_keys[ssa]
                    if _has_values:
                        values[dest] = a_values[ssa]
                    dest -= 1
                    ssa -= 1
                    na -= 1
                    if na == 0:
                        break
                min_gallop += 1
        if na == 0:
            sortslice_copy(keys, values, dest - nb + 1, b_keys, b_values, ssb - nb + 1, nb)
        else:
            assert nb == 0
            assert dest == ssa
        return merge_adjust_gallop(ms, min_gallop)

    @wrap
    def merge_at(ms, keys, values, i):
        if False:
            return 10
        '\n        Merge the two runs at stack indices i and i+1.\n\n        An updated MergeState is returned.\n        '
        n = ms.n
        assert n >= 2
        assert i >= 0
        assert i == n - 2 or i == n - 3
        (ssa, na) = ms.pending[i]
        (ssb, nb) = ms.pending[i + 1]
        assert na > 0 and nb > 0
        assert ssa + na == ssb
        ms.pending[i] = MergeRun(ssa, na + nb)
        if i == n - 3:
            ms.pending[i + 1] = ms.pending[i + 2]
        ms = merge_pop(ms)
        k = gallop_right(keys[ssb], keys, ssa, ssa + na, ssa)
        na -= k - ssa
        ssa = k
        if na == 0:
            return ms
        k = gallop_left(keys[ssa + na - 1], keys, ssb, ssb + nb, ssb + nb - 1)
        nb = k - ssb
        if na <= nb:
            return merge_lo(ms, keys, values, ssa, na, ssb, nb)
        else:
            return merge_hi(ms, keys, values, ssa, na, ssb, nb)

    @wrap
    def merge_collapse(ms, keys, values):
        if False:
            while True:
                i = 10
        '\n        Examine the stack of runs waiting to be merged, merging adjacent runs\n        until the stack invariants are re-established:\n\n        1. len[-3] > len[-2] + len[-1]\n        2. len[-2] > len[-1]\n\n        An updated MergeState is returned.\n\n        See listsort.txt for more info.\n        '
        while ms.n > 1:
            pending = ms.pending
            n = ms.n - 2
            if n > 0 and pending[n - 1].size <= pending[n].size + pending[n + 1].size or (n > 1 and pending[n - 2].size <= pending[n - 1].size + pending[n].size):
                if pending[n - 1].size < pending[n + 1].size:
                    n -= 1
                ms = merge_at(ms, keys, values, n)
            elif pending[n].size < pending[n + 1].size:
                ms = merge_at(ms, keys, values, n)
            else:
                break
        return ms

    @wrap
    def merge_force_collapse(ms, keys, values):
        if False:
            print('Hello World!')
        '\n        Regardless of invariants, merge all runs on the stack until only one\n        remains.  This is used at the end of the mergesort.\n\n        An updated MergeState is returned.\n        '
        while ms.n > 1:
            pending = ms.pending
            n = ms.n - 2
            if n > 0:
                if pending[n - 1].size < pending[n + 1].size:
                    n -= 1
            ms = merge_at(ms, keys, values, n)
        return ms

    @wrap
    def reverse_slice(keys, values, start, stop):
        if False:
            i = 10
            return i + 15
        '\n        Reverse a slice, in-place.\n        '
        i = start
        j = stop - 1
        while i < j:
            (keys[i], keys[j]) = (keys[j], keys[i])
            i += 1
            j -= 1
        if has_values(keys, values):
            i = start
            j = stop - 1
            while i < j:
                (values[i], values[j]) = (values[j], values[i])
                i += 1
                j -= 1

    @wrap
    def run_timsort_with_mergestate(ms, keys, values):
        if False:
            print('Hello World!')
        '\n        Run timsort with the mergestate.\n        '
        nremaining = len(keys)
        if nremaining < 2:
            return
        minrun = merge_compute_minrun(nremaining)
        lo = zero
        while nremaining > 0:
            (n, desc) = count_run(keys, lo, lo + nremaining)
            if desc:
                reverse_slice(keys, values, lo, lo + n)
            if n < minrun:
                force = min(minrun, nremaining)
                binarysort(keys, values, lo, lo + force, lo + n)
                n = force
            ms = merge_append(ms, MergeRun(lo, n))
            ms = merge_collapse(ms, keys, values)
            lo += n
            nremaining -= n
        ms = merge_force_collapse(ms, keys, values)
        assert ms.n == 1
        assert ms.pending[0] == (0, len(keys))

    @wrap
    def run_timsort(keys):
        if False:
            print('Hello World!')
        '\n        Run timsort over the given keys.\n        '
        values = keys
        run_timsort_with_mergestate(merge_init(keys), keys, values)

    @wrap
    def run_timsort_with_values(keys, values):
        if False:
            print('Hello World!')
        '\n        Run timsort over the given keys and values.\n        '
        run_timsort_with_mergestate(merge_init_with_values(keys, values), keys, values)
    return TimsortImplementation(wrap, count_run, binarysort, gallop_left, gallop_right, merge_init, merge_append, merge_pop, merge_compute_minrun, merge_lo, merge_hi, merge_at, merge_force_collapse, merge_collapse, run_timsort, run_timsort_with_values)

def make_py_timsort(*args):
    if False:
        print('Hello World!')
    return make_timsort_impl(lambda f: f, *args)

def make_jit_timsort(*args):
    if False:
        i = 10
        return i + 15
    from numba import jit
    return make_timsort_impl(lambda f: jit(nopython=True)(f), *args)