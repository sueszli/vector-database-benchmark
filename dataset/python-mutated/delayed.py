from __future__ import annotations
import operator
import types
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial
from tlz import concat, curry, merge, unique
from dask import config
from dask.base import DaskMethodsMixin, dont_optimize, is_dask_collection, named_schedulers, replace_name_in_key
from dask.base import tokenize as _tokenize
from dask.context import globalmethod
from dask.core import flatten, quote
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Graph, NestedKeys
from dask.utils import OperatorMethodMixin, apply, funcname, is_namedtuple_instance, methodcaller
__all__ = ['Delayed', 'delayed']
DEFAULT_GET = named_schedulers.get('threads', named_schedulers['sync'])

def unzip(ls, nout):
    if False:
        i = 10
        return i + 15
    'Unzip a list of lists into ``nout`` outputs.'
    out = list(zip(*ls))
    if not out:
        out = [()] * nout
    return out

def finalize(collection):
    if False:
        i = 10
        return i + 15
    assert is_dask_collection(collection)
    name = 'finalize-' + tokenize(collection)
    keys = collection.__dask_keys__()
    (finalize, args) = collection.__dask_postcompute__()
    layer = {name: (finalize, keys) + args}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
    return Delayed(name, graph)

def unpack_collections(expr):
    if False:
        for i in range(10):
            print('nop')
    "Normalize a python object and merge all sub-graphs.\n\n    - Replace ``Delayed`` with their keys\n    - Convert literals to things the schedulers can handle\n    - Extract dask graphs from all enclosed values\n\n    Parameters\n    ----------\n    expr : object\n        The object to be normalized. This function knows how to handle\n        dask collections, as well as most builtin python types.\n\n    Returns\n    -------\n    task : normalized task to be run\n    collections : a tuple of collections\n\n    Examples\n    --------\n    >>> import dask\n    >>> a = delayed(1, 'a')\n    >>> b = delayed(2, 'b')\n    >>> task, collections = unpack_collections([a, b, 3])\n    >>> task\n    ['a', 'b', 3]\n    >>> collections\n    (Delayed('a'), Delayed('b'))\n\n    >>> task, collections = unpack_collections({a: 1, b: 2})\n    >>> task\n    (<class 'dict'>, [['a', 1], ['b', 2]])\n    >>> collections\n    (Delayed('a'), Delayed('b'))\n    "
    if isinstance(expr, Delayed):
        return (expr._key, (expr,))
    if is_dask_collection(expr):
        finalized = finalize(expr)
        return (finalized._key, (finalized,))
    if type(expr) is type(iter(list())):
        expr = list(expr)
    elif type(expr) is type(iter(tuple())):
        expr = tuple(expr)
    elif type(expr) is type(iter(set())):
        expr = set(expr)
    typ = type(expr)
    if typ in (list, tuple, set):
        (args, collections) = unzip((unpack_collections(e) for e in expr), 2)
        args = list(args)
        collections = tuple(unique(concat(collections), key=id))
        if typ is not list:
            args = (typ, args)
        return (args, collections)
    if typ is dict:
        (args, collections) = unpack_collections([[k, v] for (k, v) in expr.items()])
        return ((dict, args), collections)
    if typ is slice:
        (args, collections) = unpack_collections([expr.start, expr.stop, expr.step])
        return ((slice, *args), collections)
    if is_dataclass(expr):
        (args, collections) = unpack_collections([[f.name, getattr(expr, f.name)] for f in fields(expr) if hasattr(expr, f.name)])
        if not collections:
            return (expr, ())
        try:
            _fields = {f.name: getattr(expr, f.name) for f in fields(expr) if hasattr(expr, f.name)}
            replace(expr, **_fields)
        except TypeError as e:
            raise TypeError(f'Failed to unpack {typ} instance. Note that using a custom __init__ is not supported.') from e
        except ValueError as e:
            raise ValueError(f'Failed to unpack {typ} instance. Note that using fields with `init=False` are not supported.') from e
        return ((apply, typ, (), (dict, args)), collections)
    if is_namedtuple_instance(expr):
        (args, collections) = unpack_collections([v for v in expr])
        return ((typ, *args), collections)
    return (expr, ())

def to_task_dask(expr):
    if False:
        print('Hello World!')
    "Normalize a python object and merge all sub-graphs.\n\n    - Replace ``Delayed`` with their keys\n    - Convert literals to things the schedulers can handle\n    - Extract dask graphs from all enclosed values\n\n    Parameters\n    ----------\n    expr : object\n        The object to be normalized. This function knows how to handle\n        ``Delayed``s, as well as most builtin python types.\n\n    Returns\n    -------\n    task : normalized task to be run\n    dask : a merged dask graph that forms the dag for this task\n\n    Examples\n    --------\n    >>> import dask\n    >>> a = delayed(1, 'a')\n    >>> b = delayed(2, 'b')\n    >>> task, dask = to_task_dask([a, b, 3])  # doctest: +SKIP\n    >>> task  # doctest: +SKIP\n    ['a', 'b', 3]\n    >>> dict(dask)  # doctest: +SKIP\n    {'a': 1, 'b': 2}\n\n    >>> task, dasks = to_task_dask({a: 1, b: 2})  # doctest: +SKIP\n    >>> task  # doctest: +SKIP\n    (dict, [['a', 1], ['b', 2]])\n    >>> dict(dask)  # doctest: +SKIP\n    {'a': 1, 'b': 2}\n    "
    warnings.warn('The dask.delayed.to_dask_dask function has been Deprecated in favor of unpack_collections', stacklevel=2)
    if isinstance(expr, Delayed):
        return (expr.key, expr.dask)
    if is_dask_collection(expr):
        name = 'finalize-' + tokenize(expr, pure=True)
        keys = expr.__dask_keys__()
        opt = getattr(expr, '__dask_optimize__', dont_optimize)
        (finalize, args) = expr.__dask_postcompute__()
        dsk = {name: (finalize, keys) + args}
        dsk.update(opt(expr.__dask_graph__(), keys))
        return (name, dsk)
    if type(expr) is type(iter(list())):
        expr = list(expr)
    elif type(expr) is type(iter(tuple())):
        expr = tuple(expr)
    elif type(expr) is type(iter(set())):
        expr = set(expr)
    typ = type(expr)
    if typ in (list, tuple, set):
        (args, dasks) = unzip((to_task_dask(e) for e in expr), 2)
        args = list(args)
        dsk = merge(dasks)
        return (args, dsk) if typ is list else ((typ, args), dsk)
    if typ is dict:
        (args, dsk) = to_task_dask([[k, v] for (k, v) in expr.items()])
        return ((dict, args), dsk)
    if is_dataclass(expr):
        (args, dsk) = to_task_dask([[f.name, getattr(expr, f.name)] for f in fields(expr) if hasattr(expr, f.name)])
        return ((apply, typ, (), (dict, args)), dsk)
    if is_namedtuple_instance(expr):
        (args, dsk) = to_task_dask([v for v in expr])
        return ((typ, *args), dsk)
    if typ is slice:
        (args, dsk) = to_task_dask([expr.start, expr.stop, expr.step])
        return ((slice,) + tuple(args), dsk)
    return (expr, {})

def tokenize(*args, pure=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Mapping function from task -> consistent name.\n\n    Parameters\n    ----------\n    args : object\n        Python objects that summarize the task.\n    pure : boolean, optional\n        If True, a consistent hash function is tried on the input. If this\n        fails, then a unique identifier is used. If False (default), then a\n        unique identifier is always used.\n    '
    if pure is None:
        pure = config.get('delayed_pure', False)
    if pure:
        return _tokenize(*args, **kwargs)
    else:
        return str(uuid.uuid4())

@curry
def delayed(obj, name=None, pure=None, nout=None, traverse=True):
    if False:
        while True:
            i = 10
    'Wraps a function or object to produce a ``Delayed``.\n\n    ``Delayed`` objects act as proxies for the object they wrap, but all\n    operations on them are done lazily by building up a dask graph internally.\n\n    Parameters\n    ----------\n    obj : object\n        The function or object to wrap\n    name : Dask key, optional\n        The key to use in the underlying graph for the wrapped object. Defaults\n        to hashing content. Note that this only affects the name of the object\n        wrapped by this call to delayed, and *not* the output of delayed\n        function calls - for that use ``dask_key_name=`` as described below.\n\n        .. note::\n\n           Because this ``name`` is used as the key in task graphs, you should\n           ensure that it uniquely identifies ``obj``. If you\'d like to provide\n           a descriptive name that is still unique, combine the descriptive name\n           with :func:`dask.base.tokenize` of the ``array_like``. See\n           :ref:`graphs` for more.\n\n    pure : bool, optional\n        Indicates whether calling the resulting ``Delayed`` object is a pure\n        operation. If True, arguments to the call are hashed to produce\n        deterministic keys. If not provided, the default is to check the global\n        ``delayed_pure`` setting, and fallback to ``False`` if unset.\n    nout : int, optional\n        The number of outputs returned from calling the resulting ``Delayed``\n        object. If provided, the ``Delayed`` output of the call can be iterated\n        into ``nout`` objects, allowing for unpacking of results. By default\n        iteration over ``Delayed`` objects will error. Note, that ``nout=1``\n        expects ``obj`` to return a tuple of length 1, and consequently for\n        ``nout=0``, ``obj`` should return an empty tuple.\n    traverse : bool, optional\n        By default dask traverses builtin python collections looking for dask\n        objects passed to ``delayed``. For large collections this can be\n        expensive. If ``obj`` doesn\'t contain any dask objects, set\n        ``traverse=False`` to avoid doing this traversal.\n\n    Examples\n    --------\n    Apply to functions to delay execution:\n\n    >>> from dask import delayed\n    >>> def inc(x):\n    ...     return x + 1\n\n    >>> inc(10)\n    11\n\n    >>> x = delayed(inc, pure=True)(10)\n    >>> type(x) == Delayed\n    True\n    >>> x.compute()\n    11\n\n    Can be used as a decorator:\n\n    >>> @delayed(pure=True)\n    ... def add(a, b):\n    ...     return a + b\n    >>> add(1, 2).compute()\n    3\n\n    ``delayed`` also accepts an optional keyword ``pure``. If False, then\n    subsequent calls will always produce a different ``Delayed``. This is\n    useful for non-pure functions (such as ``time`` or ``random``).\n\n    >>> from random import random\n    >>> out1 = delayed(random, pure=False)()\n    >>> out2 = delayed(random, pure=False)()\n    >>> out1.key == out2.key\n    False\n\n    If you know a function is pure (output only depends on the input, with no\n    global state), then you can set ``pure=True``. This will attempt to apply a\n    consistent name to the output, but will fallback on the same behavior of\n    ``pure=False`` if this fails.\n\n    >>> @delayed(pure=True)\n    ... def add(a, b):\n    ...     return a + b\n    >>> out1 = add(1, 2)\n    >>> out2 = add(1, 2)\n    >>> out1.key == out2.key\n    True\n\n    Instead of setting ``pure`` as a property of the callable, you can also set\n    it contextually using the ``delayed_pure`` setting. Note that this\n    influences the *call* and not the *creation* of the callable:\n\n    >>> @delayed\n    ... def mul(a, b):\n    ...     return a * b\n    >>> import dask\n    >>> with dask.config.set(delayed_pure=True):\n    ...     print(mul(1, 2).key == mul(1, 2).key)\n    True\n    >>> with dask.config.set(delayed_pure=False):\n    ...     print(mul(1, 2).key == mul(1, 2).key)\n    False\n\n    The key name of the result of calling a delayed object is determined by\n    hashing the arguments by default. To explicitly set the name, you can use\n    the ``dask_key_name`` keyword when calling the function:\n\n    >>> add(1, 2)   # doctest: +SKIP\n    Delayed(\'add-3dce7c56edd1ac2614add714086e950f\')\n    >>> add(1, 2, dask_key_name=\'three\')\n    Delayed(\'three\')\n\n    Note that objects with the same key name are assumed to have the same\n    result. If you set the names explicitly you should make sure your key names\n    are different for different results.\n\n    >>> add(1, 2, dask_key_name=\'three\')\n    Delayed(\'three\')\n    >>> add(2, 1, dask_key_name=\'three\')\n    Delayed(\'three\')\n    >>> add(2, 2, dask_key_name=\'four\')\n    Delayed(\'four\')\n\n    ``delayed`` can also be applied to objects to make operations on them lazy:\n\n    >>> a = delayed([1, 2, 3])\n    >>> isinstance(a, Delayed)\n    True\n    >>> a.compute()\n    [1, 2, 3]\n\n    The key name of a delayed object is hashed by default if ``pure=True`` or\n    is generated randomly if ``pure=False`` (default).  To explicitly set the\n    name, you can use the ``name`` keyword. To ensure that the key is unique\n    you should include the tokenized value as well, or otherwise ensure that\n    it\'s unique:\n\n    >>> from dask.base import tokenize\n    >>> data = [1, 2, 3]\n    >>> a = delayed(data, name=\'mylist-\' + tokenize(data))\n    >>> a  # doctest: +SKIP\n    Delayed(\'mylist-55af65871cb378a4fa6de1660c3e8fb7\')\n\n    Delayed results act as a proxy to the underlying object. Many operators\n    are supported:\n\n    >>> (a + [1, 2]).compute()\n    [1, 2, 3, 1, 2]\n    >>> a[1].compute()\n    2\n\n    Method and attribute access also works:\n\n    >>> a.count(2).compute()\n    1\n\n    Note that if a method doesn\'t exist, no error will be thrown until runtime:\n\n    >>> res = a.not_a_real_method() # doctest: +SKIP\n    >>> res.compute()  # doctest: +SKIP\n    AttributeError("\'list\' object has no attribute \'not_a_real_method\'")\n\n    "Magic" methods (e.g. operators and attribute access) are assumed to be\n    pure, meaning that subsequent calls must return the same results. This\n    behavior is not overrideable through the ``delayed`` call, but can be\n    modified using other ways as described below.\n\n    To invoke an impure attribute or operator, you\'d need to use it in a\n    delayed function with ``pure=False``:\n\n    >>> class Incrementer:\n    ...     def __init__(self):\n    ...         self._n = 0\n    ...     @property\n    ...     def n(self):\n    ...         self._n += 1\n    ...         return self._n\n    ...\n    >>> x = delayed(Incrementer())\n    >>> x.n.key == x.n.key\n    True\n    >>> get_n = delayed(lambda x: x.n, pure=False)\n    >>> get_n(x).key == get_n(x).key\n    False\n\n    In contrast, methods are assumed to be impure by default, meaning that\n    subsequent calls may return different results. To assume purity, set\n    ``pure=True``. This allows sharing of any intermediate values.\n\n    >>> a.count(2, pure=True).key == a.count(2, pure=True).key\n    True\n\n    As with function calls, method calls also respect the global\n    ``delayed_pure`` setting and support the ``dask_key_name`` keyword:\n\n    >>> a.count(2, dask_key_name="count_2")\n    Delayed(\'count_2\')\n    >>> import dask\n    >>> with dask.config.set(delayed_pure=True):\n    ...     print(a.count(2).key == a.count(2).key)\n    True\n    '
    if isinstance(obj, Delayed):
        return obj
    if is_dask_collection(obj) or traverse:
        (task, collections) = unpack_collections(obj)
    else:
        task = quote(obj)
        collections = set()
    if not (nout is None or (type(nout) is int and nout >= 0)):
        raise ValueError('nout must be None or a non-negative integer, got %s' % nout)
    if task is obj:
        if not name:
            try:
                prefix = obj.__name__
            except AttributeError:
                prefix = type(obj).__name__
            token = tokenize(obj, nout, pure=pure)
            name = f'{prefix}-{token}'
        return DelayedLeaf(obj, name, pure=pure, nout=nout)
    else:
        if not name:
            name = f'{type(obj).__name__}-{tokenize(task, pure=pure)}'
        layer = {name: task}
        graph = HighLevelGraph.from_collections(name, layer, dependencies=collections)
        return Delayed(name, graph, nout)

def _swap(method, self, other):
    if False:
        i = 10
        return i + 15
    return method(other, self)

def right(method):
    if False:
        print('Hello World!')
    "Wrapper to create 'right' version of operator given left version"
    return partial(_swap, method)

def optimize(dsk, keys, **kwargs):
    if False:
        i = 10
        return i + 15
    if not isinstance(keys, (list, set)):
        keys = [keys]
    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    dsk = dsk.cull(set(flatten(keys)))
    return dsk

class Delayed(DaskMethodsMixin, OperatorMethodMixin):
    """Represents a value to be computed by dask.

    Equivalent to the output from a single key in a dask graph.
    """
    __slots__ = ('_key', '_dask', '_length', '_layer')

    def __init__(self, key, dsk, length=None, layer=None):
        if False:
            while True:
                i = 10
        self._key = key
        self._dask = dsk
        self._length = length
        self._layer = layer or key
        if isinstance(dsk, HighLevelGraph) and self._layer not in dsk.layers:
            raise ValueError(f"Layer {self._layer} not in the HighLevelGraph's layers: {list(dsk.layers)}")

    @property
    def key(self):
        if False:
            return 10
        return self._key

    @property
    def dask(self):
        if False:
            print('Hello World!')
        return self._dask

    def __dask_graph__(self) -> Graph:
        if False:
            for i in range(10):
                print('nop')
        return self.dask

    def __dask_keys__(self) -> NestedKeys:
        if False:
            for i in range(10):
                print('nop')
        return [self.key]

    def __dask_layers__(self) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        return (self._layer,)

    def __dask_tokenize__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.key
    __dask_scheduler__ = staticmethod(DEFAULT_GET)
    __dask_optimize__ = globalmethod(optimize, key='delayed_optimize')

    def __dask_postcompute__(self):
        if False:
            return 10
        return (single_key, ())

    def __dask_postpersist__(self):
        if False:
            i = 10
            return i + 15
        return (self._rebuild, ())

    def _rebuild(self, dsk, *, rename=None):
        if False:
            while True:
                i = 10
        key = replace_name_in_key(self.key, rename) if rename else self.key
        if isinstance(dsk, HighLevelGraph) and len(dsk.layers) == 1:
            layer = next(iter(dsk.layers))
        else:
            layer = None
        return Delayed(key, dsk, self._length, layer=layer)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Delayed({repr(self.key)})'

    def __hash__(self):
        if False:
            return 10
        return hash(self.key)

    def __dir__(self):
        if False:
            while True:
                i = 10
        return dir(type(self))

    def __getattr__(self, attr):
        if False:
            return 10
        if attr.startswith('_'):
            raise AttributeError(f'Attribute {attr} not found')
        if attr == 'visualise':
            warnings.warn('dask.delayed objects have no `visualise` method. Perhaps you meant `visualize`?')
        return DelayedAttr(self, attr)

    def __setattr__(self, attr, val):
        if False:
            for i in range(10):
                print('nop')
        try:
            object.__setattr__(self, attr, val)
        except AttributeError:
            raise TypeError('Delayed objects are immutable')

    def __setitem__(self, index, val):
        if False:
            print('Hello World!')
        raise TypeError('Delayed objects are immutable')

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self._length is None:
            raise TypeError('Delayed objects of unspecified length are not iterable')
        for i in range(self._length):
            yield self[i]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._length is None:
            raise TypeError('Delayed objects of unspecified length have no len()')
        return self._length

    def __call__(self, *args, pure=None, dask_key_name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        func = delayed(apply, pure=pure)
        if dask_key_name is not None:
            return func(self, args, kwargs, dask_key_name=dask_key_name)
        return func(self, args, kwargs)

    def __bool__(self):
        if False:
            while True:
                i = 10
        raise TypeError('Truth of Delayed objects is not supported')
    __nonzero__ = __bool__

    def __get__(self, instance, cls):
        if False:
            i = 10
            return i + 15
        if instance is None:
            return self
        return types.MethodType(self, instance)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        if False:
            for i in range(10):
                print('nop')
        method = delayed(right(op) if inv else op, pure=True)
        return lambda *args, **kwargs: method(*args, **kwargs)
    _get_unary_operator = _get_binary_operator

def call_function(func, func_token, args, kwargs, pure=None, nout=None):
    if False:
        while True:
            i = 10
    dask_key_name = kwargs.pop('dask_key_name', None)
    pure = kwargs.pop('pure', pure)
    if dask_key_name is None:
        name = '{}-{}'.format(funcname(func), tokenize(func_token, *args, pure=pure, **kwargs))
    else:
        name = dask_key_name
    (args2, collections) = unzip(map(unpack_collections, args), 2)
    collections = list(concat(collections))
    if kwargs:
        (dask_kwargs, collections2) = unpack_collections(kwargs)
        collections.extend(collections2)
        task = (apply, func, list(args2), dask_kwargs)
    else:
        task = (func,) + args2
    graph = HighLevelGraph.from_collections(name, {name: task}, dependencies=collections)
    nout = nout if nout is not None else None
    return Delayed(name, graph, length=nout)

class DelayedLeaf(Delayed):
    __slots__ = ('_obj', '_pure', '_nout')

    def __init__(self, obj, key, pure=None, nout=None):
        if False:
            print('Hello World!')
        super().__init__(key, None)
        self._obj = obj
        self._pure = pure
        self._nout = nout

    @property
    def dask(self):
        if False:
            for i in range(10):
                print('nop')
        return HighLevelGraph.from_collections(self._key, {self._key: self._obj}, dependencies=())

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return call_function(self._obj, self._key, args, kwargs, pure=self._pure, nout=self._nout)

    @property
    def __name__(self):
        if False:
            return 10
        return self._obj.__name__

    @property
    def __doc__(self):
        if False:
            i = 10
            return i + 15
        return self._obj.__doc__

class DelayedAttr(Delayed):
    __slots__ = ('_obj', '_attr')

    def __init__(self, obj, attr):
        if False:
            print('Hello World!')
        key = 'getattr-%s' % tokenize(obj, attr, pure=True)
        super().__init__(key, None)
        self._obj = obj
        self._attr = attr

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        if attr == 'dtype' and self._attr == 'dtype':
            raise AttributeError('Attribute dtype not found')
        return super().__getattr__(attr)

    @property
    def dask(self):
        if False:
            return 10
        layer = {self._key: (getattr, self._obj._key, self._attr)}
        return HighLevelGraph.from_collections(self._key, layer, dependencies=[self._obj])

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        return call_function(methodcaller(self._attr), self._attr, (self._obj,) + args, kwargs)
for op in [operator.abs, operator.neg, operator.pos, operator.invert, operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv, operator.mod, operator.pow, operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift, operator.eq, operator.ge, operator.gt, operator.ne, operator.le, operator.lt, operator.getitem]:
    Delayed._bind_operator(op)
try:
    Delayed._bind_operator(operator.matmul)
except AttributeError:
    pass

def single_key(seq):
    if False:
        while True:
            i = 10
    'Pick out the only element of this list, a list of keys'
    return seq[0]