import textwrap
import time
methods = {'fileno'}

class Proxy1:
    strategy = '__getattr__'
    works_for = 'any attr'

    def __init__(self, wrapped):
        if False:
            return 10
        self._wrapped = wrapped

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name in methods:
            return getattr(self._wrapped, name)
        raise AttributeError(name)

class Proxy2:
    strategy = 'generated methods (getattr + closure)'
    works_for = 'methods'

    def __init__(self, wrapped):
        if False:
            while True:
                i = 10
        self._wrapped = wrapped

def add_wrapper(cls, method):
    if False:
        print('Hello World!')

    def wrapper(self, *args, **kwargs):
        if False:
            return 10
        return getattr(self._wrapped, method)(*args, **kwargs)
    setattr(cls, method, wrapper)
for method in methods:
    add_wrapper(Proxy2, method)

class Proxy3:
    strategy = 'generated methods (exec)'
    works_for = 'methods'

    def __init__(self, wrapped):
        if False:
            print('Hello World!')
        self._wrapped = wrapped

def add_wrapper(cls, method):
    if False:
        for i in range(10):
            print('nop')
    code = textwrap.dedent(f'\n        def wrapper(self, *args, **kwargs):\n            return self._wrapped.{method}(*args, **kwargs)\n    ')
    ns = {}
    exec(code, ns)
    setattr(cls, method, ns['wrapper'])
for method in methods:
    add_wrapper(Proxy3, method)

class Proxy4:
    strategy = 'generated properties (getattr + closure)'
    works_for = 'any attr'

    def __init__(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        self._wrapped = wrapped

def add_wrapper(cls, attr):
    if False:
        print('Hello World!')

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._wrapped, attr)

    def setter(self, newval):
        if False:
            for i in range(10):
                print('nop')
        setattr(self._wrapped, attr, newval)

    def deleter(self):
        if False:
            print('Hello World!')
        delattr(self._wrapped, attr)
    setattr(cls, attr, property(getter, setter, deleter))
for method in methods:
    add_wrapper(Proxy4, method)

class Proxy5:
    strategy = 'generated properties (exec)'
    works_for = 'any attr'

    def __init__(self, wrapped):
        if False:
            while True:
                i = 10
        self._wrapped = wrapped

def add_wrapper(cls, attr):
    if False:
        for i in range(10):
            print('nop')
    code = textwrap.dedent(f'\n        def getter(self):\n            return self._wrapped.{attr}\n\n        def setter(self, newval):\n            self._wrapped.{attr} = newval\n\n        def deleter(self):\n            del self._wrapped.{attr}\n    ')
    ns = {}
    exec(code, ns)
    setattr(cls, attr, property(ns['getter'], ns['setter'], ns['deleter']))
for method in methods:
    add_wrapper(Proxy5, method)

class Proxy6:
    strategy = 'copy attrs from wrappee to wrapper'
    works_for = 'methods + constant attrs'

    def __init__(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        self._wrapper = wrapper
        for method in methods:
            setattr(self, method, getattr(self._wrapper, method))
classes = [Proxy1, Proxy2, Proxy3, Proxy4, Proxy5, Proxy6]

def check(cls):
    if False:
        return 10
    with open('/etc/passwd') as f:
        p = cls(f)
        assert p.fileno() == f.fileno()
for cls in classes:
    check(cls)
with open('/etc/passwd') as f:
    objs = [c(f) for c in classes]
    COUNT = 1000000
    try:
        import __pypy__
    except ImportError:
        pass
    else:
        COUNT *= 10
    while True:
        print('-------')
        for obj in objs:
            start = time.perf_counter()
            for _ in range(COUNT):
                obj.fileno()
            end = time.perf_counter()
            per_usec = COUNT / (end - start) / 1000000.0
            print(f'{per_usec:7.2f} / us: {obj.strategy} ({obj.works_for})')