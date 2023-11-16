"""
**NetRef**: a transparent *network reference*. This module contains quite a lot
of *magic*, so beware.
"""
import sys
import inspect
import types
from ..lib.compat import pickle, is_py3k, maxint
from . import consts
_local_netref_attrs = frozenset(['____conn__', '____oid__', '____refcount__', '__class__', '__cmp__', '__del__', '__delattr__', '__dir__', '__doc__', '__getattr__', '__getattribute__', '__hash__', '__init__', '__metaclass__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__slots__', '__str__', '__weakref__', '__dict__', '__members__', '__methods__'])
'the set of attributes that are local to the netref object'
_builtin_types = [type, object, bool, complex, dict, float, int, list, slice, str, tuple, set, frozenset, Exception, type(None), types.BuiltinFunctionType, types.GeneratorType, types.MethodType, types.CodeType, types.FrameType, types.TracebackType, types.ModuleType, types.FunctionType, type(int.__add__), type(1 .__add__), type(iter([])), type(iter(())), type(iter(set()))]
'a list of types considered built-in (shared between connections)'
try:
    BaseException
except NameError:
    pass
else:
    _builtin_types.append(BaseException)
if is_py3k:
    _builtin_types.extend([bytes, bytearray, type(iter(range(10))), memoryview])
    xrange = range
else:
    _builtin_types.extend([basestring, unicode, long, xrange, type(iter(xrange(10))), file, types.InstanceType, types.ClassType, types.DictProxyType])
_normalized_builtin_types = dict((((t.__name__, t.__module__), t) for t in _builtin_types))

def syncreq(proxy, handler, *args):
    if False:
        i = 10
        return i + 15
    'Performs a synchronous request on the given proxy object.\n    Not intended to be invoked directly.\n\n    :param proxy: the proxy on which to issue the request\n    :param handler: the request handler (one of the ``HANDLE_XXX`` members of\n                    ``.protocol.consts``)\n    :param args: arguments to the handler\n\n    :raises: any exception raised by the operation will be raised\n    :returns: the result of the operation\n    '
    conn = object.__getattribute__(proxy, '____conn__')()
    if not conn:
        raise ReferenceError('weakly-referenced object no longer exists')
    oid = object.__getattribute__(proxy, '____oid__')
    return conn.sync_request(handler, oid, *args)

def asyncreq(proxy, handler, *args):
    if False:
        print('Hello World!')
    'Performs an asynchronous request on the given proxy object.\n    Not intended to be invoked directly.\n\n    :param proxy: the proxy on which to issue the request\n    :param handler: the request handler (one of the ``HANDLE_XXX`` members of\n                    ``.protocol.consts``)\n    :param args: arguments to the handler\n\n    :returns: an :class:`AsyncResult <.core.async.AsyncResult>` representing\n              the operation\n    '
    conn = object.__getattribute__(proxy, '____conn__')()
    if not conn:
        raise ReferenceError('weakly-referenced object no longer exists')
    oid = object.__getattribute__(proxy, '____oid__')
    return conn.async_request(handler, oid, *args)

class NetrefMetaclass(type):
    """A *metaclass* used to customize the ``__repr__`` of ``netref`` classes.
    It is quite useless, but it makes debugging and interactive programming
    easier"""
    __slots__ = ()

    def __repr__(self):
        if False:
            return 10
        if self.__module__:
            return "<netref class '%s.%s'>" % (self.__module__, self.__name__)
        else:
            return "<netref class '%s'>" % (self.__name__,)

class BaseNetref(object):
    """The base netref class, from which all netref classes derive. Some netref
    classes are "pre-generated" and cached upon importing this module (those
    defined in the :data:`_builtin_types`), and they are shared between all
    connections.

    The rest of the netref classes are created by :meth:`.core.protocl.Connection._unbox`,
    and are private to the connection.

    Do not use this class directly; use :func:`class_factory` instead.

    :param conn: the :class:`.core.protocol.Connection` instance
    :param oid: the unique object ID of the remote object
    """
    __metaclass__ = NetrefMetaclass
    __slots__ = ('____conn__', '____oid__', '__weakref__', '____refcount__')

    def __init__(self, conn, oid):
        if False:
            return 10
        self.____conn__ = conn
        self.____oid__ = oid
        self.____refcount__ = 1

    def __del__(self):
        if False:
            print('Hello World!')
        try:
            asyncreq(self, consts.HANDLE_DEL, self.____refcount__)
        except Exception:
            pass

    def __getattribute__(self, name):
        if False:
            while True:
                i = 10
        if name in _local_netref_attrs:
            if name == '__class__':
                cls = object.__getattribute__(self, '__class__')
                if cls is None:
                    cls = self.__getattr__('__class__')
                return cls
            elif name == '__doc__':
                return self.__getattr__('__doc__')
            elif name == '__members__':
                return self.__dir__()
            else:
                return object.__getattribute__(self, name)
        elif name == '__call__':
            return object.__getattribute__(self, '__call__')
        else:
            return syncreq(self, consts.HANDLE_GETATTR, name)

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return syncreq(self, consts.HANDLE_GETATTR, name)

    def __delattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name in _local_netref_attrs:
            object.__delattr__(self, name)
        else:
            syncreq(self, consts.HANDLE_DELATTR, name)

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name in _local_netref_attrs:
            object.__setattr__(self, name, value)
        else:
            syncreq(self, consts.HANDLE_SETATTR, name, value)

    def __dir__(self):
        if False:
            i = 10
            return i + 15
        return list(syncreq(self, consts.HANDLE_DIR))

    def __hash__(self):
        if False:
            print('Hello World!')
        return syncreq(self, consts.HANDLE_HASH)

    def __cmp__(self, other):
        if False:
            i = 10
            return i + 15
        return syncreq(self, consts.HANDLE_CMP, other)

    def __repr__(self):
        if False:
            print('Hello World!')
        return syncreq(self, consts.HANDLE_REPR)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return syncreq(self, consts.HANDLE_STR)

    def __reduce_ex__(self, proto):
        if False:
            i = 10
            return i + 15
        return (pickle.loads, (syncreq(self, consts.HANDLE_PICKLE, proto),))
if not isinstance(BaseNetref, NetrefMetaclass):
    ns = dict(BaseNetref.__dict__)
    for slot in BaseNetref.__slots__:
        ns.pop(slot)
    BaseNetref = NetrefMetaclass(BaseNetref.__name__, BaseNetref.__bases__, ns)

def _make_method(name, doc):
    if False:
        print('Hello World!')
    'creates a method with the given name and docstring that invokes\n    :func:`syncreq` on its `self` argument'
    slicers = {'__getslice__': '__getitem__', '__delslice__': '__delitem__', '__setslice__': '__setitem__'}
    name = str(name)
    if name == '__call__':

        def __call__(_self, *args, **kwargs):
            if False:
                print('Hello World!')
            kwargs = tuple(kwargs.items())
            return syncreq(_self, consts.HANDLE_CALL, args, kwargs)
        __call__.__doc__ = doc
        return __call__
    elif name in slicers:

        def method(self, start, stop, *args):
            if False:
                print('Hello World!')
            if stop == maxint:
                stop = None
            return syncreq(self, consts.HANDLE_OLDSLICING, slicers[name], name, start, stop, args)
        method.__name__ = name
        method.__doc__ = doc
        return method
    else:

        def method(_self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs = tuple(kwargs.items())
            return syncreq(_self, consts.HANDLE_CALLATTR, name, args, kwargs)
        method.__name__ = name
        method.__doc__ = doc
        return method

def inspect_methods(obj):
    if False:
        while True:
            i = 10
    'introspects the given (local) object, returning a list of all of its\n    methods (going up the MRO).\n\n    :param obj: any local (not proxy) python object\n\n    :returns: a list of ``(method name, docstring)`` tuples of all the methods\n              of the given object\n    '
    methods = {}
    attrs = {}
    if isinstance(obj, type):
        mros = list(reversed(type(obj).__mro__)) + list(reversed(obj.__mro__))
    else:
        mros = reversed(type(obj).__mro__)
    for basecls in mros:
        attrs.update(basecls.__dict__)
    for (name, attr) in attrs.items():
        if name not in _local_netref_attrs and hasattr(attr, '__call__'):
            methods[name] = inspect.getdoc(attr)
    return methods.items()

def class_factory(clsname, modname, methods):
    if False:
        for i in range(10):
            print('nop')
    "Creates a netref class proxying the given class\n\n    :param clsname: the class's name\n    :param modname: the class's module name\n    :param methods: a list of ``(method name, docstring)`` tuples, of the methods\n                    that the class defines\n\n    :returns: a netref class\n    "
    clsname = str(clsname)
    modname = str(modname)
    ns = {'__slots__': ()}
    for (name, doc) in methods:
        name = str(name)
        if name not in _local_netref_attrs:
            ns[name] = _make_method(name, doc)
    ns['__module__'] = modname
    if modname in sys.modules and hasattr(sys.modules[modname], clsname):
        ns['__class__'] = getattr(sys.modules[modname], clsname)
    elif (clsname, modname) in _normalized_builtin_types:
        ns['__class__'] = _normalized_builtin_types[clsname, modname]
    else:
        ns['__class__'] = None
    return type(clsname, (BaseNetref,), ns)
builtin_classes_cache = {}
'The cache of built-in netref classes (each of the types listed in\n:data:`_builtin_types`). These are shared between all  connections'
for cls in _builtin_types:
    builtin_classes_cache[cls.__name__, cls.__module__] = class_factory(cls.__name__, cls.__module__, inspect_methods(cls))