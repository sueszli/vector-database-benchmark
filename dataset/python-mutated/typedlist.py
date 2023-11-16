"""
Python wrapper that connects CPython interpreter to the Numba typed-list.

This is the code that is used when creating typed lists outside of a `@jit`
context and when returning a typed-list from a `@jit` decorated function. It
basically a Python class that has a Numba allocated typed-list under the hood
and uses `@jit` functions to access it. Since it inherits from MutableSequence
it should really quack like the CPython `list`.

"""
from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import overload, box, unbox, NativeValue, type_callable, overload_classmethod
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
Int_or_Slice = pt.Union['pt.SupportsIndex', slice]
T_co = pt.TypeVar('T_co', covariant=True)

class _Sequence(pt.Protocol[T_co]):

    def __getitem__(self, i: int) -> T_co:
        if False:
            print('Hello World!')
        ...

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        ...
DEFAULT_ALLOCATED = listobject.DEFAULT_ALLOCATED

@njit
def _make_list(itemty, allocated=DEFAULT_ALLOCATED):
    if False:
        while True:
            i = 10
    return listobject._as_meminfo(listobject.new_list(itemty, allocated=allocated))

@njit
def _length(l):
    if False:
        print('Hello World!')
    return len(l)

@njit
def _allocated(l):
    if False:
        for i in range(10):
            print('nop')
    return l._allocated()

@njit
def _is_mutable(l):
    if False:
        for i in range(10):
            print('nop')
    return l._is_mutable()

@njit
def _make_mutable(l):
    if False:
        i = 10
        return i + 15
    return l._make_mutable()

@njit
def _make_immutable(l):
    if False:
        for i in range(10):
            print('nop')
    return l._make_immutable()

@njit
def _append(l, item):
    if False:
        for i in range(10):
            print('nop')
    l.append(item)

@njit
def _setitem(l, i, item):
    if False:
        for i in range(10):
            print('nop')
    l[i] = item

@njit
def _getitem(l, i):
    if False:
        print('Hello World!')
    return l[i]

@njit
def _contains(l, item):
    if False:
        return 10
    return item in l

@njit
def _count(l, item):
    if False:
        print('Hello World!')
    return l.count(item)

@njit
def _pop(l, i):
    if False:
        for i in range(10):
            print('nop')
    return l.pop(i)

@njit
def _delitem(l, i):
    if False:
        while True:
            i = 10
    del l[i]

@njit
def _extend(l, iterable):
    if False:
        return 10
    return l.extend(iterable)

@njit
def _insert(l, i, item):
    if False:
        i = 10
        return i + 15
    l.insert(i, item)

@njit
def _remove(l, item):
    if False:
        print('Hello World!')
    l.remove(item)

@njit
def _clear(l):
    if False:
        i = 10
        return i + 15
    l.clear()

@njit
def _reverse(l):
    if False:
        print('Hello World!')
    l.reverse()

@njit
def _copy(l):
    if False:
        print('Hello World!')
    return l.copy()

@njit
def _eq(t, o):
    if False:
        for i in range(10):
            print('nop')
    return t == o

@njit
def _ne(t, o):
    if False:
        print('Hello World!')
    return t != o

@njit
def _lt(t, o):
    if False:
        for i in range(10):
            print('nop')
    return t < o

@njit
def _le(t, o):
    if False:
        return 10
    return t <= o

@njit
def _gt(t, o):
    if False:
        i = 10
        return i + 15
    return t > o

@njit
def _ge(t, o):
    if False:
        print('Hello World!')
    return t >= o

@njit
def _index(l, item, start, end):
    if False:
        for i in range(10):
            print('nop')
    return l.index(item, start, end)

@njit
def _sort(l, key, reverse):
    if False:
        for i in range(10):
            print('nop')
    return l.sort(key, reverse)

def _from_meminfo_ptr(ptr, listtype):
    if False:
        for i in range(10):
            print('nop')
    return List(meminfo=ptr, lsttype=listtype)
T = pt.TypeVar('T')
T_or_ListT = pt.Union[T, 'List[T]']

class List(MutableSequence, pt.Generic[T]):
    """A typed-list usable in Numba compiled functions.

    Implements the MutableSequence interface.
    """
    _legal_kwargs = ['lsttype', 'meminfo', 'allocated']

    def __new__(cls, *args, lsttype=None, meminfo=None, allocated=DEFAULT_ALLOCATED, **kwargs):
        if False:
            while True:
                i = 10
        if config.DISABLE_JIT:
            return list(*args, **kwargs)
        else:
            return object.__new__(cls)

    @classmethod
    def empty_list(cls, item_type, allocated=DEFAULT_ALLOCATED):
        if False:
            i = 10
            return i + 15
        'Create a new empty List.\n\n        Parameters\n        ----------\n        item_type: Numba type\n            type of the list item.\n        allocated: int\n            number of items to pre-allocate\n        '
        if config.DISABLE_JIT:
            return list()
        else:
            return cls(lsttype=ListType(item_type), allocated=allocated)

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        For users, the constructor does not take any parameters.\n        The keyword arguments are for internal use only.\n\n        Parameters\n        ----------\n        args: iterable\n            The iterable to initialize the list from\n        lsttype : numba.core.types.ListType; keyword-only\n            Used internally for the list type.\n        meminfo : MemInfo; keyword-only\n            Used internally to pass the MemInfo object when boxing.\n        allocated: int; keyword-only\n            Used internally to pre-allocate space for items\n        '
        illegal_kwargs = any((kw not in self._legal_kwargs for kw in kwargs))
        if illegal_kwargs or (args and kwargs):
            raise TypeError('List() takes no keyword arguments')
        if kwargs:
            (self._list_type, self._opaque) = self._parse_arg(**kwargs)
        else:
            self._list_type = None
            if args:
                if not 0 <= len(args) <= 1:
                    raise TypeError('List() expected at most 1 argument, got {}'.format(len(args)))
                iterable = args[0]
                if hasattr(iterable, 'ndim') and iterable.ndim == 0:
                    self.append(iterable.item())
                else:
                    try:
                        iter(iterable)
                    except TypeError:
                        raise TypeError('List() argument must be iterable')
                    for i in args[0]:
                        self.append(i)

    def _parse_arg(self, lsttype, meminfo=None, allocated=DEFAULT_ALLOCATED):
        if False:
            while True:
                i = 10
        if not isinstance(lsttype, ListType):
            raise TypeError('*lsttype* must be a ListType')
        if meminfo is not None:
            opaque = meminfo
        else:
            opaque = _make_list(lsttype.item_type, allocated=allocated)
        return (lsttype, opaque)

    @property
    def _numba_type_(self):
        if False:
            i = 10
            return i + 15
        if self._list_type is None:
            raise TypeError('invalid operation on untyped list')
        return self._list_type

    @property
    def _typed(self):
        if False:
            while True:
                i = 10
        'Returns True if the list is typed.\n        '
        return self._list_type is not None

    @property
    def _dtype(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._typed:
            raise RuntimeError('invalid operation on untyped list')
        return self._list_type.dtype

    def _initialise_list(self, item):
        if False:
            return 10
        lsttype = types.ListType(typeof(item))
        (self._list_type, self._opaque) = self._parse_arg(lsttype)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        if not self._typed:
            return 0
        else:
            return _length(self)

    def _allocated(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._typed:
            return DEFAULT_ALLOCATED
        else:
            return _allocated(self)

    def _is_mutable(self):
        if False:
            while True:
                i = 10
        return _is_mutable(self)

    def _make_mutable(self):
        if False:
            print('Hello World!')
        return _make_mutable(self)

    def _make_immutable(self):
        if False:
            return 10
        return _make_immutable(self)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _eq(self, other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return _ne(self, other)

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _lt(self, other)

    def __le__(self, other):
        if False:
            while True:
                i = 10
        return _le(self, other)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        return _gt(self, other)

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _ge(self, other)

    def append(self, item: T) -> None:
        if False:
            return 10
        if not self._typed:
            self._initialise_list(item)
        _append(self, item)

    @pt.overload
    def __setitem__(self, i: int, o: T) -> None:
        if False:
            return 10
        ...

    @pt.overload
    def __setitem__(self, s: slice, o: 'List[T]') -> None:
        if False:
            print('Hello World!')
        ...

    def __setitem__(self, i: Int_or_Slice, item: T_or_ListT) -> None:
        if False:
            i = 10
            return i + 15
        if not self._typed:
            self._initialise_list(item)
        _setitem(self, i, item)

    @pt.overload
    def __getitem__(self, i: int) -> T:
        if False:
            print('Hello World!')
        ...

    @pt.overload
    def __getitem__(self, i: slice) -> 'List[T]':
        if False:
            return 10
        ...

    def __getitem__(self, i: Int_or_Slice) -> T_or_ListT:
        if False:
            print('Hello World!')
        if not self._typed:
            raise IndexError
        else:
            return _getitem(self, i)

    def __iter__(self) -> pt.Iterator[T]:
        if False:
            return 10
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: T) -> bool:
        if False:
            i = 10
            return i + 15
        return _contains(self, item)

    def __delitem__(self, i: Int_or_Slice) -> None:
        if False:
            return 10
        _delitem(self, i)

    def insert(self, i: int, item: T) -> None:
        if False:
            i = 10
            return i + 15
        if not self._typed:
            self._initialise_list(item)
        _insert(self, i, item)

    def count(self, item: T) -> int:
        if False:
            print('Hello World!')
        return _count(self, item)

    def pop(self, i: 'pt.SupportsIndex'=-1) -> T:
        if False:
            return 10
        return _pop(self, i)

    def extend(self, iterable: '_Sequence[T]') -> None:
        if False:
            return 10
        if len(iterable) == 0:
            return None
        if not self._typed:
            self._initialise_list(iterable[0])
        return _extend(self, iterable)

    def remove(self, item: T) -> None:
        if False:
            print('Hello World!')
        return _remove(self, item)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        return _clear(self)

    def reverse(self):
        if False:
            for i in range(10):
                print('nop')
        return _reverse(self)

    def copy(self):
        if False:
            print('Hello World!')
        return _copy(self)

    def index(self, item: T, start: pt.Optional[int]=None, stop: pt.Optional[int]=None) -> int:
        if False:
            print('Hello World!')
        return _index(self, item, start, stop)

    def sort(self, key=None, reverse=False):
        if False:
            for i in range(10):
                print('nop')
        'Sort the list inplace.\n\n        See also ``list.sort()``\n        '
        if callable(key) and (not isinstance(key, Dispatcher)):
            key = njit(key)
        return _sort(self, key, reverse)

    def __str__(self):
        if False:
            return 10
        buf = []
        for x in self:
            buf.append('{}'.format(x))
        try:
            get_ipython
            return '[{0}, ...]'.format(', '.join(buf[:1000]))
        except (NameError, IndexError):
            return '[{0}]'.format(', '.join(buf))

    def __repr__(self):
        if False:
            print('Hello World!')
        body = str(self)
        prefix = str(self._list_type) if self._typed else 'ListType[Undefined]'
        return '{prefix}({body})'.format(prefix=prefix, body=body)

@overload_classmethod(ListType, 'empty_list')
def typedlist_empty(cls, item_type, allocated=DEFAULT_ALLOCATED):
    if False:
        print('Hello World!')
    if cls.instance_type is not ListType:
        return

    def impl(cls, item_type, allocated=DEFAULT_ALLOCATED):
        if False:
            print('Hello World!')
        return listobject.new_list(item_type, allocated=allocated)
    return impl

@box(types.ListType)
def box_lsttype(typ, val, c):
    if False:
        while True:
            i = 10
    context = c.context
    builder = c.builder
    ctor = cgutils.create_struct_proxy(typ)
    lstruct = ctor(context, builder, value=val)
    boxed_meminfo = c.box(types.MemInfoPointer(types.voidptr), lstruct.meminfo)
    modname = c.context.insert_const_string(c.builder.module, 'numba.typed.typedlist')
    typedlist_mod = c.pyapi.import_module_noblock(modname)
    fmp_fn = c.pyapi.object_getattr_string(typedlist_mod, '_from_meminfo_ptr')
    lsttype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))
    result_var = builder.alloca(c.pyapi.pyobj)
    builder.store(cgutils.get_null_value(c.pyapi.pyobj), result_var)
    with builder.if_then(cgutils.is_not_null(builder, lsttype_obj)):
        res = c.pyapi.call_function_objargs(fmp_fn, (boxed_meminfo, lsttype_obj))
        c.pyapi.decref(fmp_fn)
        c.pyapi.decref(typedlist_mod)
        c.pyapi.decref(boxed_meminfo)
        builder.store(res, result_var)
    return builder.load(result_var)

@unbox(types.ListType)
def unbox_listtype(typ, val, c):
    if False:
        print('Hello World!')
    context = c.context
    builder = c.builder
    list_type = c.pyapi.unserialize(c.pyapi.serialize_object(List))
    valtype = c.pyapi.object_type(val)
    same_type = builder.icmp_unsigned('==', valtype, list_type)
    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, '_opaque')
            native = c.unbox(types.MemInfoPointer(types.voidptr), miptr)
            mi = native.value
            ctor = cgutils.create_struct_proxy(typ)
            lstruct = ctor(context, builder)
            data_pointer = context.nrt.meminfo_data(builder, mi)
            data_pointer = builder.bitcast(data_pointer, listobject.ll_list_type.as_pointer())
            lstruct.data = builder.load(data_pointer)
            lstruct.meminfo = mi
            lstobj = lstruct._getvalue()
            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block
        with orelse:
            c.pyapi.err_format('PyExc_TypeError', "can't unbox a %S as a %S", valtype, list_type)
            bb_else = c.builder.basic_block
    lstobj_res = c.builder.phi(lstobj.type)
    is_error_res = c.builder.phi(cgutils.bool_t)
    lstobj_res.add_incoming(lstobj, bb_unboxed)
    lstobj_res.add_incoming(lstobj.type(None), bb_else)
    is_error_res.add_incoming(cgutils.false_bit, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)
    c.pyapi.decref(list_type)
    c.pyapi.decref(valtype)
    return NativeValue(lstobj_res, is_error=is_error_res)

def _guess_dtype(iterable):
    if False:
        for i in range(10):
            print('nop')
    'Guess the correct dtype of the iterable type. '
    if not isinstance(iterable, types.IterableType):
        raise TypingError('List() argument must be iterable')
    elif isinstance(iterable, types.Array) and iterable.ndim > 1:
        return iterable.copy(ndim=iterable.ndim - 1, layout='A')
    elif hasattr(iterable, 'dtype'):
        return iterable.dtype
    elif hasattr(iterable, 'yield_type'):
        return iterable.yield_type
    elif isinstance(iterable, types.UnicodeType):
        return iterable
    elif isinstance(iterable, types.DictType):
        return iterable.key_type
    else:
        raise TypingError('List() argument does not have a suitable dtype')

@type_callable(ListType)
def typedlist_call(context):
    if False:
        return 10
    "Defines typing logic for ``List()`` and ``List(iterable)``.\n\n    If no argument is given, the returned typer types a new typed-list with an\n    undefined item type. If a single argument is given it must be iterable with\n    a guessable 'dtype'. In this case, the typer types a new typed-list with\n    the type set to the 'dtype' of the iterable arg.\n\n    Parameters\n    ----------\n    arg : single iterable (optional)\n        The single optional argument.\n\n    Returns\n    -------\n    typer : function\n        A typer suitable to type constructor calls.\n\n    Raises\n    ------\n    The returned typer raises a TypingError in case of unsuitable arguments.\n\n    "

    class Typer(object):

        def attach_sig(self):
            if False:
                i = 10
                return i + 15
            from inspect import signature as mypysig

            def mytyper(iterable):
                if False:
                    print('Hello World!')
                pass
            self.pysig = mypysig(mytyper)

        def __call__(self, *args, **kwargs):
            if False:
                return 10
            if kwargs:
                raise TypingError('List() takes no keyword arguments')
            elif args:
                if not 0 <= len(args) <= 1:
                    raise TypingError('List() expected at most 1 argument, got {}'.format(len(args)))
                rt = types.ListType(_guess_dtype(args[0]))
                self.attach_sig()
                return Signature(rt, args, None, pysig=self.pysig)
            else:
                item_type = types.undefined
                return types.ListType(item_type)
    return Typer()

@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls, *args):
    if False:
        return 10
    'Defines lowering for ``List()`` and ``List(iterable)``.\n\n    This defines the lowering logic to instantiate either an empty typed-list\n    or a typed-list initialised with values from a single iterable argument.\n\n    Parameters\n    ----------\n    cls : TypeRef\n        Expecting a TypeRef of a precise ListType.\n    args: tuple\n        A tuple that contains a single iterable (optional)\n\n    Returns\n    -------\n    impl : function\n        An implementation suitable for lowering the constructor call.\n\n    See also: `redirect_type_ctor` in numba/cpython/bulitins.py\n    '
    list_ty = cls.instance_type
    if not isinstance(list_ty, types.ListType):
        return
    if not list_ty.is_precise():
        msg = 'expecting a precise ListType but got {}'.format(list_ty)
        raise LoweringError(msg)
    item_type = types.TypeRef(list_ty.item_type)
    if args:
        if isinstance(args[0], types.Array) and args[0].ndim == 0:

            def impl(cls, *args):
                if False:
                    i = 10
                    return i + 15
                r = List.empty_list(item_type)
                r.append(args[0].item())
                return r
        else:

            def impl(cls, *args):
                if False:
                    for i in range(10):
                        print('nop')
                r = List.empty_list(item_type)
                for i in args[0]:
                    r.append(i)
                return r
    else:

        def impl(cls, *args):
            if False:
                i = 10
                return i + 15
            return List.empty_list(item_type)
    return impl