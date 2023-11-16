"""Functional-style utilities."""
import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
logger = get_logger(__name__)
__all__ = ('LRUCache', 'is_list', 'maybe_list', 'memoize', 'mlazy', 'noop', 'first', 'firstmethod', 'chunks', 'padlist', 'mattrgetter', 'uniq', 'regen', 'dictfilter', 'lazy', 'maybe_evaluate', 'head_from_fun', 'maybe', 'fun_accepts_kwargs')
FUNHEAD_TEMPLATE = '\ndef {fun_name}({fun_args}):\n    return {fun_value}\n'

class DummyContext:

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *exc_info):
        if False:
            print('Hello World!')
        pass

class mlazy(lazy):
    """Memoized lazy evaluation.

    The function is only evaluated once, every subsequent access
    will return the same value.
    """
    evaluated = False
    _value = None

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.evaluated:
            self._value = super().evaluate()
            self.evaluated = True
        return self._value

def noop(*args, **kwargs):
    if False:
        return 10
    'No operation.\n\n    Takes any arguments/keyword arguments and does nothing.\n    '

def pass1(arg, *args, **kwargs):
    if False:
        while True:
            i = 10
    'Return the first positional argument.'
    return arg

def evaluate_promises(it):
    if False:
        for i in range(10):
            print('nop')
    for value in it:
        if isinstance(value, promise):
            value = value()
        yield value

def first(predicate, it):
    if False:
        return 10
    "Return the first element in ``it`` that ``predicate`` accepts.\n\n    If ``predicate`` is None it will return the first item that's not\n    :const:`None`.\n    "
    return next((v for v in evaluate_promises(it) if (predicate(v) if predicate is not None else v is not None)), None)

def firstmethod(method, on_call=None):
    if False:
        print('Hello World!')
    'Multiple dispatch.\n\n    Return a function that with a list of instances,\n    finds the first instance that gives a value for the given method.\n\n    The list can also contain lazy instances\n    (:class:`~kombu.utils.functional.lazy`.)\n    '

    def _matcher(it, *args, **kwargs):
        if False:
            while True:
                i = 10
        for obj in it:
            try:
                meth = getattr(maybe_evaluate(obj), method)
                reply = on_call(meth, *args, **kwargs) if on_call else meth(*args, **kwargs)
            except AttributeError:
                pass
            else:
                if reply is not None:
                    return reply
    return _matcher

def chunks(it, n):
    if False:
        i = 10
        return i + 15
    'Split an iterator into chunks with `n` elements each.\n\n    Warning:\n        ``it`` must be an actual iterator, if you pass this a\n        concrete sequence will get you repeating elements.\n\n        So ``chunks(iter(range(1000)), 10)`` is fine, but\n        ``chunks(range(1000), 10)`` is not.\n\n    Example:\n        # n == 2\n        >>> x = chunks(iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2)\n        >>> list(x)\n        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]\n\n        # n == 3\n        >>> x = chunks(iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)\n        >>> list(x)\n        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]\n    '
    for item in it:
        yield ([item] + list(islice(it, n - 1)))

def padlist(container, size, default=None):
    if False:
        print('Hello World!')
    "Pad list with default elements.\n\n    Example:\n        >>> first, last, city = padlist(['George', 'Costanza', 'NYC'], 3)\n        ('George', 'Costanza', 'NYC')\n        >>> first, last, city = padlist(['George', 'Costanza'], 3)\n        ('George', 'Costanza', None)\n        >>> first, last, city, planet = padlist(\n        ...     ['George', 'Costanza', 'NYC'], 4, default='Earth',\n        ... )\n        ('George', 'Costanza', 'NYC', 'Earth')\n    "
    return list(container)[:size] + [default] * (size - len(container))

def mattrgetter(*attrs):
    if False:
        for i in range(10):
            print('nop')
    'Get attributes, ignoring attribute errors.\n\n    Like :func:`operator.itemgetter` but return :const:`None` on missing\n    attributes instead of raising :exc:`AttributeError`.\n    '
    return lambda obj: {attr: getattr(obj, attr, None) for attr in attrs}

def uniq(it):
    if False:
        i = 10
        return i + 15
    'Return all unique elements in ``it``, preserving order.'
    seen = set()
    return (seen.add(obj) or obj for obj in it if obj not in seen)

def lookahead(it):
    if False:
        while True:
            i = 10
    'Yield pairs of (current, next) items in `it`.\n\n    `next` is None if `current` is the last item.\n    Example:\n        >>> list(lookahead(x for x in range(6)))\n        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, None)]\n    '
    (a, b) = tee(it)
    next(b, None)
    return zip_longest(a, b)

def regen(it):
    if False:
        print('Hello World!')
    'Convert iterator to an object that can be consumed multiple times.\n\n    ``Regen`` takes any iterable, and if the object is an\n    generator it will cache the evaluated list on first access,\n    so that the generator can be "consumed" multiple times.\n    '
    if isinstance(it, (list, tuple)):
        return it
    return _regen(it)

class _regen(UserList, list):

    def __init__(self, it):
        if False:
            i = 10
            return i + 15
        self.__it = it
        self.__consumed = []
        self.__done = False

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (list, (self.data,))

    def map(self, func):
        if False:
            while True:
                i = 10
        self.__consumed = [func(el) for el in self.__consumed]
        self.__it = map(func, self.__it)

    def __length_hint__(self):
        if False:
            i = 10
            return i + 15
        return self.__it.__length_hint__()

    def __lookahead_consume(self, limit=None):
        if False:
            i = 10
            return i + 15
        if not self.__done and (limit is None or limit > 0):
            it = iter(self.__it)
            try:
                now = next(it)
            except StopIteration:
                return
            self.__consumed.append(now)
            while not self.__done:
                try:
                    next_ = next(it)
                    self.__consumed.append(next_)
                except StopIteration:
                    self.__done = True
                    break
                finally:
                    yield now
                now = next_
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        break

    def __iter__(self):
        if False:
            print('Hello World!')
        yield from self.__consumed
        yield from self.__lookahead_consume()

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if index < 0:
            return self.data[index]
        consume_count = index - len(self.__consumed) + 1
        for _ in self.__lookahead_consume(limit=consume_count):
            pass
        return self.__consumed[index]

    def __bool__(self):
        if False:
            return 10
        if len(self.__consumed):
            return True
        try:
            next(iter(self))
        except StopIteration:
            return False
        else:
            return True

    @property
    def data(self):
        if False:
            return 10
        if not self.__done:
            self.__consumed.extend(self.__it)
            self.__done = True
        return self.__consumed

    def __repr__(self):
        if False:
            return 10
        return '<{}: [{}{}]>'.format(self.__class__.__name__, ', '.join((repr(e) for e in self.__consumed)), '...' if not self.__done else '')

def _argsfromspec(spec, replace_defaults=True):
    if False:
        return 10
    if spec.defaults:
        split = len(spec.defaults)
        defaults = list(range(len(spec.defaults))) if replace_defaults else spec.defaults
        positional = spec.args[:-split]
        optional = list(zip(spec.args[-split:], defaults))
    else:
        (positional, optional) = (spec.args, [])
    varargs = spec.varargs
    varkw = spec.varkw
    if spec.kwonlydefaults:
        kwonlyargs = set(spec.kwonlyargs) - set(spec.kwonlydefaults.keys())
        if replace_defaults:
            kwonlyargs_optional = [(kw, i) for (i, kw) in enumerate(spec.kwonlydefaults.keys())]
        else:
            kwonlyargs_optional = list(spec.kwonlydefaults.items())
    else:
        (kwonlyargs, kwonlyargs_optional) = (spec.kwonlyargs, [])
    return ', '.join(filter(None, [', '.join(positional), ', '.join((f'{k}={v}' for (k, v) in optional)), f'*{varargs}' if varargs else None, '*' if (kwonlyargs or kwonlyargs_optional) and (not varargs) else None, ', '.join(kwonlyargs) if kwonlyargs else None, ', '.join((f'{k}="{v}"' for (k, v) in kwonlyargs_optional)), f'**{varkw}' if varkw else None]))

def head_from_fun(fun: Callable[..., Any], bound: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    'Generate signature function from actual function.'
    is_function = inspect.isfunction(fun)
    is_callable = callable(fun)
    is_cython = fun.__class__.__name__ == 'cython_function_or_method'
    is_method = inspect.ismethod(fun)
    if not is_function and is_callable and (not is_method) and (not is_cython):
        (name, fun) = (fun.__class__.__name__, fun.__call__)
    else:
        name = fun.__name__
    definition = FUNHEAD_TEMPLATE.format(fun_name=name, fun_args=_argsfromspec(inspect.getfullargspec(fun)), fun_value=1)
    logger.debug(definition)
    namespace = {'__name__': fun.__module__}
    exec(definition, namespace)
    result = namespace[name]
    result._source = definition
    if bound:
        return partial(result, object())
    return result

def arity_greater(fun, n):
    if False:
        return 10
    argspec = inspect.getfullargspec(fun)
    return argspec.varargs or len(argspec.args) > n

def fun_takes_argument(name, fun, position=None):
    if False:
        while True:
            i = 10
    spec = inspect.getfullargspec(fun)
    return spec.varkw or spec.varargs or (len(spec.args) >= position if position else name in spec.args)

def fun_accepts_kwargs(fun):
    if False:
        i = 10
        return i + 15
    'Return true if function accepts arbitrary keyword arguments.'
    return any((p for p in inspect.signature(fun).parameters.values() if p.kind == p.VAR_KEYWORD))

def maybe(typ, val):
    if False:
        while True:
            i = 10
    'Call typ on value if val is defined.'
    return typ(val) if val is not None else val

def seq_concat_item(seq, item):
    if False:
        for i in range(10):
            print('nop')
    'Return copy of sequence seq with item added.\n\n    Returns:\n        Sequence: if seq is a tuple, the result will be a tuple,\n           otherwise it depends on the implementation of ``__add__``.\n    '
    return seq + (item,) if isinstance(seq, tuple) else seq + [item]

def seq_concat_seq(a, b):
    if False:
        i = 10
        return i + 15
    'Concatenate two sequences: ``a + b``.\n\n    Returns:\n        Sequence: The return value will depend on the largest sequence\n            - if b is larger and is a tuple, the return value will be a tuple.\n            - if a is larger and is a list, the return value will be a list,\n    '
    prefer = type(max([a, b], key=len))
    if not isinstance(a, prefer):
        a = prefer(a)
    if not isinstance(b, prefer):
        b = prefer(b)
    return a + b

def is_numeric_value(value):
    if False:
        return 10
    return isinstance(value, (int, float)) and (not isinstance(value, bool))