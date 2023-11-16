from __future__ import annotations
import cython
from typing import Dict, List, TypeVar, Optional, Generic, Tuple
try:
    import typing
    from typing import Set as _SET_
    from typing import ClassVar
except ImportError:
    pass
var = 1
var: cython.int = 2
fvar: cython.float = 1.2
some_number: cython.int
some_list: List[cython.int] = []
another_list: list[cython.int] = []
t: Tuple[cython.int, ...] = (1, 2, 3)
t2: tuple[cython.int, ...]
body: Optional[List[str]]
descr_only: 'descriptions are allowed but ignored'
some_number = 5
body = None

def f():
    if False:
        print('Hello World!')
    '\n    >>> f()\n    (2, 1.5, [], (1, 2, 3))\n    '
    var = 1
    var: cython.int = 2
    fvar: cython.float = 1.5
    some_number: cython.int
    some_list: List[cython.int] = []
    t: Tuple[cython.int, ...] = (1, 2, 3)
    body: Optional[List[str]]
    descr_only: 'descriptions are allowed but ignored'
    return (var, fvar, some_list, t)

class BasicStarship(object):
    """
    >>> bs = BasicStarship(5)
    >>> bs.damage
    5
    >>> bs.captain
    'Picard'
    >>> bs.stats
    {}
    >>> BasicStarship.stats
    {}
    """
    captain: str = 'Picard'
    damage: cython.int
    stats: ClassVar[Dict[str, cython.int]] = {}
    descr_only: 'descriptions are allowed but ignored'

    def __init__(self, damage):
        if False:
            return 10
        self.damage = damage

@cython.cclass
class BasicStarshipExt(object):
    """
    >>> bs = BasicStarshipExt(5)
    >>> bs.test()
    (5, 'Picard', {})
    """
    captain: str = 'Picard'
    damage: cython.int
    stats: ClassVar[Dict[str, cython.int]] = {}
    descr_only: 'descriptions are allowed but ignored'

    def __init__(self, damage):
        if False:
            while True:
                i = 10
        self.damage = damage

    def test(self):
        if False:
            while True:
                i = 10
        return (self.damage, self.captain, self.stats)
T = TypeVar('T')

class Cls(object):
    pass
c = Cls()
c.x: int = 0
c.y: int
d = {}
d['a']: int = 0
d['b']: int
(x): int
(y): int = 0

@cython.test_assert_path_exists('//WhileStatNode', '//WhileStatNode//DictIterationNextNode')
def iter_declared_dict(d):
    if False:
        while True:
            i = 10
    '\n    >>> d = {1.1: 2.5, 3.3: 4.5}\n    >>> iter_declared_dict(d)\n    7.0\n\n    # specialized "compiled" test in module-level __doc__\n    '
    typed_dict: Dict[cython.float, cython.float] = d
    s = 0.0
    for key in typed_dict:
        s += d[key]
    return s

@cython.test_assert_path_exists('//WhileStatNode', '//WhileStatNode//DictIterationNextNode')
def iter_declared_dict_arg(d: Dict[cython.float, cython.float]):
    if False:
        print('Hello World!')
    '\n    >>> d = {1.1: 2.5, 3.3: 4.5}\n    >>> iter_declared_dict_arg(d)\n    7.0\n\n    # module level "compiled" test in __doc__ below\n    '
    s = 0.0
    for key in d:
        s += d[key]
    return s

def literal_list_ptr():
    if False:
        while True:
            i = 10
    '\n    >>> literal_list_ptr()\n    4\n    '
    a: cython.p_int = [1, 2, 3, 4, 5]
    return a[3]

def test_subscripted_types():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_subscripted_types()\n    dict object\n    dict object\n    list object\n    list object\n    list object\n    set object\n    '
    a1: typing.Dict[cython.int, cython.float] = {}
    a2: dict[cython.int, cython.float] = {}
    b1: List[cython.int] = []
    b2: list[cython.int] = []
    b3: List = []
    c: _SET_[object] = set()
    print(cython.typeof(a1) + (' object' if not cython.compiled else ''))
    print(cython.typeof(a2) + (' object' if not cython.compiled else ''))
    print(cython.typeof(b1) + (' object' if not cython.compiled else ''))
    print(cython.typeof(b2) + (' object' if not cython.compiled else ''))
    print(cython.typeof(b3) + (' object' if not cython.compiled else ''))
    print(cython.typeof(c) + (' object' if not cython.compiled else ''))

def test_tuple(a: typing.Tuple[cython.int, cython.float], b: typing.Tuple[cython.int, ...], c: Tuple[cython.int, object]):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_tuple((1, 1.0), (1, 1.0), (1, 1.0))\n    int\n    int\n    float\n    Python object\n    (int, float)\n    tuple object\n    tuple object\n    tuple object\n    tuple object\n    '
    x: typing.Tuple[int, float] = (a[0], a[1])
    y: Tuple[cython.int, ...] = (1, 2.0)
    plain_tuple: Tuple = ()
    z = a[0]
    p = x[1]
    print(cython.typeof(z))
    print('int' if cython.compiled and cython.typeof(x[0]) == 'Python object' else cython.typeof(x[0]))
    if cython.compiled:
        print(cython.typeof(p))
    else:
        print('float' if cython.typeof(p) == 'float' else cython.typeof(p))
    print(cython.typeof(x[1]) if cython.compiled or cython.typeof(p) != 'float' else 'Python object')
    print(cython.typeof(a) if cython.compiled or cython.typeof(a) != 'tuple' else '(int, float)')
    print(cython.typeof(x) + (' object' if not cython.compiled else ''))
    print(cython.typeof(y) + (' object' if not cython.compiled else ''))
    print(cython.typeof(c) + (' object' if not cython.compiled else ''))
    print(cython.typeof(plain_tuple) + (' object' if not cython.compiled else ''))

def test_tuple_without_typing(a: tuple[cython.int, cython.float], b: tuple[cython.int, ...], c: tuple[cython.int, object]):
    if False:
        print('Hello World!')
    '\n    >>> test_tuple_without_typing((1, 1.0), (1, 1.0), (1, 1.0))\n    int\n    int\n    float\n    Python object\n    (int, float)\n    tuple object\n    tuple object\n    tuple object\n    tuple object\n    '
    x: tuple[int, float] = (a[0], a[1])
    y: tuple[cython.int, ...] = (1, 2.0)
    plain_tuple: tuple = ()
    z = a[0]
    p = x[1]
    print(cython.typeof(z))
    print('int' if cython.compiled and cython.typeof(x[0]) == 'Python object' else cython.typeof(x[0]))
    print(cython.typeof(p) if cython.compiled or cython.typeof(p) != 'float' else 'float')
    print(cython.typeof(x[1]) if cython.compiled or cython.typeof(p) != 'float' else 'Python object')
    print(cython.typeof(a) if cython.compiled or cython.typeof(a) != 'tuple' else '(int, float)')
    print(cython.typeof(x) + (' object' if not cython.compiled else ''))
    print(cython.typeof(y) + (' object' if not cython.compiled else ''))
    print(cython.typeof(c) + (' object' if not cython.compiled else ''))
    print(cython.typeof(plain_tuple) + (' object' if not cython.compiled else ''))

def test_use_typing_attributes_as_non_annotations():
    if False:
        print('Hello World!')
    '\n    >>> test_use_typing_attributes_as_non_annotations()\n    typing.Tuple typing.Tuple[int]\n    typing.Optional True\n    typing.Optional True\n    '
    x1 = typing.Tuple
    x2 = typing.Tuple[int]
    y1 = typing.Optional
    y2 = typing.Optional[typing.FrozenSet]
    z1 = Optional
    z2 = Optional[Dict]
    allowed_optional_frozenset_strings = ['typing.Union[typing.FrozenSet, NoneType]', 'typing.Optional[typing.FrozenSet]']
    allowed_optional_dict_strings = ['typing.Union[typing.Dict, NoneType]', 'typing.Optional[typing.Dict]']
    print(x1, x2)
    print(y1, str(y2) in allowed_optional_frozenset_strings or str(y2))
    print(z1, str(z2) in allowed_optional_dict_strings or str(z2))

def test_optional_ctuple(x: typing.Optional[tuple[float]]):
    if False:
        print('Hello World!')
    "\n    Should not be a C-tuple (because these can't be optional)\n    >>> test_optional_ctuple((1.0,))\n    tuple object\n    "
    print(cython.typeof(x) + (' object' if not cython.compiled else ''))
try:
    import numpy.typing as npt
    import numpy as np
except ImportError:
    pass

def list_float_to_numpy(z: List[float]) -> List[npt.NDArray[np.float64]]:
    if False:
        return 10
    assert cython.typeof(z) == 'list'
    return [z[0]]
if cython.compiled:
    __doc__ = '\n    # passing non-dicts to variables declared as dict now fails\n    >>> class D(object):\n    ...     def __getitem__(self, x): return 2\n    ...     def __iter__(self): return iter([1, 2, 3])\n    >>> iter_declared_dict(D())  # doctest:+IGNORE_EXCEPTION_DETAIL\n    Traceback (most recent call last):\n    ...\n    TypeError: Expected dict, got D\n    >>> iter_declared_dict_arg(D())  # doctest:+IGNORE_EXCEPTION_DETAIL\n    Traceback (most recent call last):\n    ...\n    TypeError: Expected dict, got D\n    '
_WARNINGS = '\n'