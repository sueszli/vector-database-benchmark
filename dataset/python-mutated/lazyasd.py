"""Lazy and self destructive containers for speeding up module import."""
import builtins
import collections.abc as cabc
import importlib
import importlib.util
import os
import sys
import threading
import time
import types
import typing as tp
__version__ = '0.1.3'

class LazyObject:

    def __init__(self, load, ctx, name):
        if False:
            for i in range(10):
                print('nop')
        "Lazily loads an object via the load function the first time an\n        attribute is accessed. Once loaded it will replace itself in the\n        provided context (typically the globals of the call site) with the\n        given name.\n\n        For example, you can prevent the compilation of a regular expression\n        until it is actually used::\n\n            DOT = LazyObject((lambda: re.compile('.')), globals(), 'DOT')\n\n        Parameters\n        ----------\n        load : function with no arguments\n            A loader function that performs the actual object construction.\n        ctx : Mapping\n            Context to replace the LazyObject instance in\n            with the object returned by load().\n        name : str\n            Name in the context to give the loaded object. This *should*\n            be the name on the LHS of the assignment.\n        "
        self._lasdo = {'loaded': False, 'load': load, 'ctx': ctx, 'name': name}

    def _lazy_obj(self):
        if False:
            for i in range(10):
                print('nop')
        d = self._lasdo
        if d['loaded']:
            obj = d['obj']
        else:
            obj = d['load']()
            d['ctx'][d['name']] = d['obj'] = obj
            d['loaded'] = True
        return obj

    def __getattribute__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name == '_lasdo' or name == '_lazy_obj':
            return super().__getattribute__(name)
        obj = self._lazy_obj()
        return getattr(obj, name)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self._lazy_obj()
        return bool(obj)

    def __iter__(self):
        if False:
            print('Hello World!')
        obj = self._lazy_obj()
        yield from obj

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        obj = self._lazy_obj()
        return obj[item]

    def __setitem__(self, key, value):
        if False:
            return 10
        obj = self._lazy_obj()
        obj[key] = value

    def __delitem__(self, item):
        if False:
            i = 10
            return i + 15
        obj = self._lazy_obj()
        del obj[item]

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        obj = self._lazy_obj()
        return obj(*args, **kwargs)

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        obj = self._lazy_obj()
        return obj < other

    def __le__(self, other):
        if False:
            print('Hello World!')
        obj = self._lazy_obj()
        return obj <= other

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        obj = self._lazy_obj()
        return obj == other

    def __ne__(self, other):
        if False:
            return 10
        obj = self._lazy_obj()
        return obj != other

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        obj = self._lazy_obj()
        return obj > other

    def __ge__(self, other):
        if False:
            return 10
        obj = self._lazy_obj()
        return obj >= other

    def __hash__(self):
        if False:
            while True:
                i = 10
        obj = self._lazy_obj()
        return hash(obj)

    def __or__(self, other):
        if False:
            return 10
        obj = self._lazy_obj()
        return obj | other

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self._lazy_obj())

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self._lazy_obj())
RT = tp.TypeVar('RT')

def lazyobject(f: tp.Callable[..., RT]) -> RT:
    if False:
        return 10
    'Decorator for constructing lazy objects from a function.'
    return LazyObject(f, f.__globals__, f.__name__)

class LazyDict(cabc.MutableMapping):

    def __init__(self, loaders, ctx, name):
        if False:
            print('Hello World!')
        "Dictionary like object that lazily loads its values from an initial\n        dict of key-loader function pairs.  Each key is loaded when its value\n        is first accessed. Once fully loaded, this object will replace itself\n        in the provided context (typically the globals of the call site) with\n        the given name.\n\n        For example, you can prevent the compilation of a bunch of regular\n        expressions until they are actually used::\n\n            RES = LazyDict({\n                    'dot': lambda: re.compile('.'),\n                    'all': lambda: re.compile('.*'),\n                    'two': lambda: re.compile('..'),\n                    }, globals(), 'RES')\n\n        Parameters\n        ----------\n        loaders : Mapping of keys to functions with no arguments\n            A mapping of loader function that performs the actual value\n            construction upon access.\n        ctx : Mapping\n            Context to replace the LazyDict instance in\n            with the the fully loaded mapping.\n        name : str\n            Name in the context to give the loaded mapping. This *should*\n            be the name on the LHS of the assignment.\n        "
        self._loaders = loaders
        self._ctx = ctx
        self._name = name
        self._d = type(loaders)()

    def _destruct(self):
        if False:
            while True:
                i = 10
        if len(self._loaders) == 0:
            self._ctx[self._name] = self._d

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        d = self._d
        if key in d:
            val = d[key]
        else:
            loader = self._loaders.pop(key)
            d[key] = val = loader()
            self._destruct()
        return val

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self._d[key] = value
        if key in self._loaders:
            del self._loaders[key]
            self._destruct()

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        if key in self._d:
            del self._d[key]
        else:
            del self._loaders[key]
            self._destruct()

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield from (set(self._d.keys()) | set(self._loaders.keys()))

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._d) + len(self._loaders)

def lazydict(f):
    if False:
        print('Hello World!')
    'Decorator for constructing lazy dicts from a function.'
    return LazyDict(f, f.__globals__, f.__name__)

class LazyBool:

    def __init__(self, load, ctx, name):
        if False:
            while True:
                i = 10
        "Boolean like object that lazily computes it boolean value when it is\n        first asked. Once loaded, this result will replace itself\n        in the provided context (typically the globals of the call site) with\n        the given name.\n\n        For example, you can prevent the complex boolean until it is actually\n        used::\n\n            ALIVE = LazyDict(lambda: not DEAD, globals(), 'ALIVE')\n\n        Parameters\n        ----------\n        load : function with no arguments\n            A loader function that performs the actual boolean evaluation.\n        ctx : Mapping\n            Context to replace the LazyBool instance in\n            with the the fully loaded mapping.\n        name : str\n            Name in the context to give the loaded mapping. This *should*\n            be the name on the LHS of the assignment.\n        "
        self._load = load
        self._ctx = ctx
        self._name = name
        self._result = None

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        if self._result is None:
            res = self._ctx[self._name] = self._result = self._load()
        else:
            res = self._result
        return res

def lazybool(f):
    if False:
        while True:
            i = 10
    'Decorator for constructing lazy booleans from a function.'
    return LazyBool(f, f.__globals__, f.__name__)

class BackgroundModuleProxy(types.ModuleType):
    """Proxy object for modules loaded in the background that block attribute
    access until the module is loaded..
    """

    def __init__(self, modname):
        if False:
            for i in range(10):
                print('nop')
        self.__dct__ = {'loaded': False, 'modname': modname}

    def __getattribute__(self, name):
        if False:
            while True:
                i = 10
        passthrough = frozenset({'__dct__', '__class__', '__spec__'})
        if name in passthrough:
            return super().__getattribute__(name)
        dct = self.__dct__
        modname = dct['modname']
        if dct['loaded']:
            mod = sys.modules[modname]
        else:
            delay_types = (BackgroundModuleProxy, type(None))
            while isinstance(sys.modules.get(modname, None), delay_types):
                time.sleep(0.001)
            mod = sys.modules[modname]
            dct['loaded'] = True
        stall = 0
        while not hasattr(mod, name) and stall < 1000:
            stall += 1
            time.sleep(0.001)
        return getattr(mod, name)

class BackgroundModuleLoader(threading.Thread):
    """Thread to load modules in the background."""

    def __init__(self, name, package, replacements, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.daemon = True
        self.name = name
        self.package = package
        self.replacements = replacements
        self.start()

    def run(self):
        if False:
            while True:
                i = 10
        counter = 0
        last = -1
        while counter < 5:
            new = len(sys.modules)
            if new == last:
                counter += 1
            else:
                last = new
                counter = 0
            time.sleep(0.001)
        modname = importlib.util.resolve_name(self.name, self.package)
        if isinstance(sys.modules[modname], BackgroundModuleProxy):
            del sys.modules[modname]
        mod = importlib.import_module(self.name, package=self.package)
        for (targname, varname) in self.replacements.items():
            if targname in sys.modules:
                targmod = sys.modules[targname]
                setattr(targmod, varname, mod)

def load_module_in_background(name, package=None, debug='DEBUG', env=None, replacements=None):
    if False:
        while True:
            i = 10
    "Entry point for loading modules in background thread.\n\n    Parameters\n    ----------\n    name : str\n        Module name to load in background thread.\n    package : str or None, optional\n        Package name, has the same meaning as in importlib.import_module().\n    debug : str, optional\n        Debugging symbol name to look up in the environment.\n    env : Mapping or None, optional\n        Environment this will default to __xonsh__.env, if available, and\n        os.environ otherwise.\n    replacements : Mapping or None, optional\n        Dictionary mapping fully qualified module names (eg foo.bar.baz) that\n        import the lazily loaded module, with the variable name in that\n        module. For example, suppose that foo.bar imports module a as b,\n        this dict is then {'foo.bar': 'b'}.\n\n    Returns\n    -------\n    module : ModuleType\n        This is either the original module that is found in sys.modules or\n        a proxy module that will block until delay attribute access until the\n        module is fully loaded.\n    "
    modname = importlib.util.resolve_name(name, package)
    if modname in sys.modules:
        return sys.modules[modname]
    if env is None:
        xonsh_obj = getattr(builtins, '__xonsh__', None)
        env = os.environ if xonsh_obj is None else getattr(xonsh_obj, 'env', os.environ)
    if env.get(debug, None):
        mod = importlib.import_module(name, package=package)
        return mod
    proxy = sys.modules[modname] = BackgroundModuleProxy(modname)
    BackgroundModuleLoader(name, package, replacements or {})
    return proxy