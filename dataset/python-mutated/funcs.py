from operator import __not__
from functools import partial, reduce, wraps
from ._inspect import get_spec, Spec
from .primitives import EMPTY
from .funcmakers import make_func, make_pred
__all__ = ['identity', 'constantly', 'caller', 'partial', 'rpartial', 'func_partial', 'curry', 'rcurry', 'autocurry', 'iffy', 'compose', 'rcompose', 'complement', 'juxt', 'ljuxt']

def identity(x):
    if False:
        i = 10
        return i + 15
    'Returns its argument.'
    return x

def constantly(x):
    if False:
        return 10
    'Creates a function accepting any args, but always returning x.'
    return lambda *a, **kw: x

def caller(*a, **kw):
    if False:
        return 10
    'Creates a function calling its sole argument with given *a, **kw.'
    return lambda f: f(*a, **kw)

def func_partial(func, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'A functools.partial alternative, which returns a real function.\n       Can be used to construct methods.'
    return lambda *a, **kw: func(*args + a, **dict(kwargs, **kw))

def rpartial(func, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Partially applies last arguments.\n       New keyworded arguments extend and override kwargs.'
    return lambda *a, **kw: func(*a + args, **dict(kwargs, **kw))

def curry(func, n=EMPTY):
    if False:
        for i in range(10):
            print('nop')
    'Curries func into a chain of one argument functions.'
    if n is EMPTY:
        n = get_spec(func).max_n
    if n <= 1:
        return func
    elif n == 2:
        return lambda x: lambda y: func(x, y)
    else:
        return lambda x: curry(partial(func, x), n - 1)

def rcurry(func, n=EMPTY):
    if False:
        while True:
            i = 10
    'Curries func into a chain of one argument functions.\n       Arguments are passed from right to left.'
    if n is EMPTY:
        n = get_spec(func).max_n
    if n <= 1:
        return func
    elif n == 2:
        return lambda x: lambda y: func(y, x)
    else:
        return lambda x: rcurry(rpartial(func, x), n - 1)

def autocurry(func, n=EMPTY, _spec=None, _args=(), _kwargs={}):
    if False:
        while True:
            i = 10
    'Creates a version of func returning its partial applications\n       until sufficient arguments are passed.'
    spec = _spec or (get_spec(func) if n is EMPTY else Spec(n, set(), n, set(), False))

    @wraps(func)
    def autocurried(*a, **kw):
        if False:
            for i in range(10):
                print('nop')
        args = _args + a
        kwargs = _kwargs.copy()
        kwargs.update(kw)
        if not spec.varkw and len(args) + len(kwargs) >= spec.max_n:
            return func(*args, **kwargs)
        elif len(args) + len(set(kwargs) & spec.names) >= spec.max_n:
            return func(*args, **kwargs)
        elif len(args) + len(set(kwargs) & spec.req_names) >= spec.req_n:
            try:
                return func(*args, **kwargs)
            except TypeError:
                return autocurry(func, _spec=spec, _args=args, _kwargs=kwargs)
        else:
            return autocurry(func, _spec=spec, _args=args, _kwargs=kwargs)
    return autocurried

def iffy(pred, action=EMPTY, default=identity):
    if False:
        i = 10
        return i + 15
    'Creates a function, which conditionally applies action or default.'
    if action is EMPTY:
        return iffy(bool, pred, default)
    else:
        pred = make_pred(pred)
        action = make_func(action)
        return lambda v: action(v) if pred(v) else default(v) if callable(default) else default

def compose(*fs):
    if False:
        while True:
            i = 10
    'Composes passed functions.'
    if fs:
        pair = lambda f, g: lambda *a, **kw: f(g(*a, **kw))
        return reduce(pair, map(make_func, fs))
    else:
        return identity

def rcompose(*fs):
    if False:
        while True:
            i = 10
    'Composes functions, calling them from left to right.'
    return compose(*reversed(fs))

def complement(pred):
    if False:
        print('Hello World!')
    'Constructs a complementary predicate.'
    return compose(__not__, pred)

def ljuxt(*fs):
    if False:
        return 10
    'Constructs a juxtaposition of the given functions.\n       Result returns a list of results of fs.'
    extended_fs = list(map(make_func, fs))
    return lambda *a, **kw: [f(*a, **kw) for f in extended_fs]

def juxt(*fs):
    if False:
        return 10
    'Constructs a lazy juxtaposition of the given functions.\n       Result returns an iterator of results of fs.'
    extended_fs = list(map(make_func, fs))
    return lambda *a, **kw: (f(*a, **kw) for f in extended_fs)