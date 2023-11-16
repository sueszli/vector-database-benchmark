"""
Benchmark Sorted Dictionary Datatypes
"""
import warnings
from .benchmark import *

@register_test
def contains(func, size):
    if False:
        print('Hello World!')
    for val in lists[size][::100]:
        assert func(val)

@register_test
def getitem(func, size):
    if False:
        return 10
    for val in lists[size][::100]:
        assert func(val) == -val

@register_test
def setitem(func, size):
    if False:
        print('Hello World!')
    for val in lists[size][::100]:
        func(val, -val)

@register_test
def setitem_existing(func, size):
    if False:
        print('Hello World!')
    for val in lists[size][::100]:
        func(val, -val)

@register_test
def delitem(func, size):
    if False:
        while True:
            i = 10
    for val in lists[size][::100]:
        func(val)

@register_test
def iter(func, size):
    if False:
        print('Hello World!')
    assert all((idx == val for (idx, val) in enumerate(func())))

@register_test
def init(func, size):
    if False:
        return 10
    func(((val, -val) for val in lists[size]))

def do_nothing(obj, size):
    if False:
        return 10
    pass

def fill_values(obj, size):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(obj, 'update'):
        obj.update({val: -val for val in range(size)})
    else:
        for val in range(size):
            obj[val] = -val
from sortedcontainers import SortedDict
kinds['SortedDict'] = SortedDict
try:
    from blist import sorteddict
    kinds['B-Tree'] = sorteddict
except ImportError:
    warnings.warn('No module named blist', ImportWarning)
try:
    from bintrees import FastAVLTree
    kinds['AVL-Tree'] = FastAVLTree
except ImportError:
    warnings.warn('No module named bintrees', ImportWarning)
try:
    from banyan import SortedDict as BanyanSortedDict
    kinds['RB-Tree'] = BanyanSortedDict
except ImportError:
    warnings.warn('No module named banyan', ImportWarning)
try:
    from skiplistcollections import SkipListDict
    kinds['Skip-List'] = SkipListDict
except ImportError:
    warnings.warn('No module named skiplistcollections', ImportWarning)
try:
    from sortedmap import sortedmap as CppSortedMap
    kinds['std::map'] = CppSortedMap
except ImportError:
    warnings.warn('No module named sortedmap', ImportWarning)
try:
    from treap import treap
    kinds['Treap'] = treap
except ImportError:
    warnings.warn('No module named treap', ImportWarning)
for name in tests:
    impls[name] = OrderedDict()
for (name, kind) in kinds.items():
    impls['contains'][name] = {'setup': fill_values, 'ctor': kind, 'func': '__contains__', 'limit': 1000000}
remove('contains', 'Treap')
for (name, kind) in kinds.items():
    impls['getitem'][name] = {'setup': fill_values, 'ctor': kind, 'func': '__getitem__', 'limit': 1000000}
for (name, kind) in kinds.items():
    impls['setitem'][name] = {'setup': do_nothing, 'ctor': kind, 'func': '__setitem__', 'limit': 1000000}
for (name, kind) in kinds.items():
    impls['setitem_existing'][name] = {'setup': fill_values, 'ctor': kind, 'func': '__setitem__', 'limit': 1000000}
for (name, kind) in kinds.items():
    impls['delitem'][name] = {'setup': fill_values, 'ctor': kind, 'func': '__delitem__', 'limit': 1000000}
for (name, kind) in kinds.items():
    impls['iter'][name] = {'setup': fill_values, 'ctor': kind, 'func': '__iter__', 'limit': 1000000}
for (name, kind) in kinds.items():
    impls['init'][name] = {'setup': do_nothing, 'ctor': kind, 'func': 'update', 'limit': 1000000}
remove('init', 'Treap')
limit('init', 'B-Tree', 100000)
limit('init', 'Skip-List', 100000)
if __name__ == '__main__':
    main('SortedDict')