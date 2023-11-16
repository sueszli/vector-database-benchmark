import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable

@register_jitable
def _siftdown(heap, startpos, pos):
    if False:
        print('Hello World!')
    newitem = heap[pos]
    while pos > startpos:
        parentpos = pos - 1 >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

@register_jitable
def _siftup(heap, pos):
    if False:
        while True:
            i = 10
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and (not heap[childpos] < heap[rightpos]):
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

@register_jitable
def _siftdown_max(heap, startpos, pos):
    if False:
        for i in range(10):
            print('nop')
    newitem = heap[pos]
    while pos > startpos:
        parentpos = pos - 1 >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

@register_jitable
def _siftup_max(heap, pos):
    if False:
        for i in range(10):
            print('nop')
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and (not heap[rightpos] < heap[childpos]):
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)

@register_jitable
def reversed_range(x):
    if False:
        for i in range(10):
            print('nop')
    return range(x - 1, -1, -1)

@register_jitable
def _heapify_max(x):
    if False:
        i = 10
        return i + 15
    n = len(x)
    for i in reversed_range(n // 2):
        _siftup_max(x, i)

@register_jitable
def _heapreplace_max(heap, item):
    if False:
        i = 10
        return i + 15
    returnitem = heap[0]
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem

def assert_heap_type(heap):
    if False:
        while True:
            i = 10
    if not isinstance(heap, (types.List, types.ListType)):
        raise TypingError('heap argument must be a list')
    dt = heap.dtype
    if isinstance(dt, types.Complex):
        msg = "'<' not supported between instances of 'complex' and 'complex'"
        raise TypingError(msg)

def assert_item_type_consistent_with_heap_type(heap, item):
    if False:
        print('Hello World!')
    if not heap.dtype == item:
        raise TypingError('heap type must be the same as item type')

@overload(hq.heapify)
def hq_heapify(x):
    if False:
        print('Hello World!')
    assert_heap_type(x)

    def hq_heapify_impl(x):
        if False:
            i = 10
            return i + 15
        n = len(x)
        for i in reversed_range(n // 2):
            _siftup(x, i)
    return hq_heapify_impl

@overload(hq.heappop)
def hq_heappop(heap):
    if False:
        for i in range(10):
            print('nop')
    assert_heap_type(heap)

    def hq_heappop_impl(heap):
        if False:
            print('Hello World!')
        lastelt = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            _siftup(heap, 0)
            return returnitem
        return lastelt
    return hq_heappop_impl

@overload(hq.heappush)
def heappush(heap, item):
    if False:
        i = 10
        return i + 15
    assert_heap_type(heap)
    assert_item_type_consistent_with_heap_type(heap, item)

    def hq_heappush_impl(heap, item):
        if False:
            print('Hello World!')
        heap.append(item)
        _siftdown(heap, 0, len(heap) - 1)
    return hq_heappush_impl

@overload(hq.heapreplace)
def heapreplace(heap, item):
    if False:
        i = 10
        return i + 15
    assert_heap_type(heap)
    assert_item_type_consistent_with_heap_type(heap, item)

    def hq_heapreplace(heap, item):
        if False:
            print('Hello World!')
        returnitem = heap[0]
        heap[0] = item
        _siftup(heap, 0)
        return returnitem
    return hq_heapreplace

@overload(hq.heappushpop)
def heappushpop(heap, item):
    if False:
        i = 10
        return i + 15
    assert_heap_type(heap)
    assert_item_type_consistent_with_heap_type(heap, item)

    def hq_heappushpop_impl(heap, item):
        if False:
            while True:
                i = 10
        if heap and heap[0] < item:
            (item, heap[0]) = (heap[0], item)
            _siftup(heap, 0)
        return item
    return hq_heappushpop_impl

def check_input_types(n, iterable):
    if False:
        print('Hello World!')
    if not isinstance(n, (types.Integer, types.Boolean)):
        raise TypingError("First argument 'n' must be an integer")
    if not isinstance(iterable, (types.Sequence, types.Array, types.ListType)):
        raise TypingError("Second argument 'iterable' must be iterable")

@overload(hq.nsmallest)
def nsmallest(n, iterable):
    if False:
        return 10
    check_input_types(n, iterable)

    def hq_nsmallest_impl(n, iterable):
        if False:
            return 10
        if n == 0:
            return [iterable[0] for _ in range(0)]
        elif n == 1:
            out = min(iterable)
            return [out]
        size = len(iterable)
        if n >= size:
            return sorted(iterable)[:n]
        it = iter(iterable)
        result = [(elem, i) for (i, elem) in zip(range(n), it)]
        _heapify_max(result)
        top = result[0][0]
        order = n
        for elem in it:
            if elem < top:
                _heapreplace_max(result, (elem, order))
                (top, _order) = result[0]
                order += 1
        result.sort()
        return [elem for (elem, order) in result]
    return hq_nsmallest_impl

@overload(hq.nlargest)
def nlargest(n, iterable):
    if False:
        return 10
    check_input_types(n, iterable)

    def hq_nlargest_impl(n, iterable):
        if False:
            for i in range(10):
                print('nop')
        if n == 0:
            return [iterable[0] for _ in range(0)]
        elif n == 1:
            out = max(iterable)
            return [out]
        size = len(iterable)
        if n >= size:
            return sorted(iterable)[::-1][:n]
        it = iter(iterable)
        result = [(elem, i) for (i, elem) in zip(range(0, -n, -1), it)]
        hq.heapify(result)
        top = result[0][0]
        order = -n
        for elem in it:
            if top < elem:
                hq.heapreplace(result, (elem, order))
                (top, _order) = result[0]
                order -= 1
        result.sort(reverse=True)
        return [elem for (elem, order) in result]
    return hq_nlargest_impl