"""
This is a stripped-down of asizeof.py from the pympler module. It's vendored
because pympler is unmaintained, while having a critical vulnerability.

Differences from the original asizeof module:
- Removed code for running as __main__
- Removed `adict`
- `__all__` only includes `asizeof`

The *original* original copyright, license and disclaimer are at the end of this
file, exactly as they appeared in the pympler code. pympler itself is under the
Apache license, which appears in the project root.

The original module docstring that appears in pympler follows; note that some of
it no longer pertains here, but it's preserved to document implementation
details.
"""
"\n**Public Functions** [#unsafe]_\n\n   Function **asizesof** returns a tuple containing the (approximate)\n   size in bytes for each given Python object separately.\n\n   Function **asized** returns for each object an instance of class\n   **Asized** containing all the size information of the object and\n   a tuple with the referents [#refs]_.\n\n   Functions **basicsize** and **itemsize** return the *basic-*\n   respectively *itemsize* of the given object, both in bytes.  For\n   objects as ``array.array``, ``numpy.array``, ``numpy.ndarray``,\n   etc. where the item size varies depending on the instance-specific\n   data type, function **itemsize** returns that item size.\n\n   Function **flatsize** returns the *flat size* of a Python object\n   in bytes defined as the *basic size* plus the *item size* times\n   the *length* of the given object.\n\n   Function **leng** returns the *length* of an object, like standard\n   function ``len`` but extended for several types.  E.g. the **leng**\n   of a multi-precision int (formerly long) is the number of ``digits``\n   [#digit]_.  The length of most *mutable* sequence objects includes\n   an estimate of the over-allocation and therefore, the **leng** value\n   may differ from the standard ``len`` result.  For objects like\n   ``array.array``, ``numpy.array``, ``numpy.ndarray``, etc. function\n   **leng** returns the proper number of items.\n\n   Function **refs** returns (a generator for) the referents [#refs]_\n   of the given object.\n\n**Public Classes** [#unsafe]_\n\n   Class **Asizer** may be used to accumulate the results of several\n   **asizeof** or **asizesof** calls.  After creating an **Asizer**\n   instance, use methods **asizeof** and **asizesof** as needed to\n   size any number of additional objects.\n\n   Call methods **exclude_refs** and/or **exclude_types** to exclude\n   references to respectively instances or types of certain objects.\n\n   Use one of the **print\\_...** methods to report the statistics.\n\n   An instance of class **Asized** is returned for each object sized\n   by the **asized** function or method.\n\n**Duplicate Objects**\n\n   Any duplicate, given objects are sized only once and the size\n   is included in the accumulated total only once.  But functions\n   **asizesof** and **asized** will return a size value respectively\n   an **Asized** instance for each given object, including duplicates.\n\n**Definitions** [#arb]_\n\n   The *length* of an objects like ``dict``, ``list``, ``set``,\n   ``str``, ``tuple``, etc. is defined as the number of items held\n   in or allocated by the object.  Held items are *references* to\n   other objects, called the *referents*.\n\n   The *size* of an object is defined as the sum of the *flat size*\n   of the object plus the sizes of any referents [#refs]_.  Referents\n   are visited recursively up to the specified detail level.  However,\n   the size of objects referenced multiple times is included only once\n   in the total *size*.\n\n   The *flat size* of an object is defined as the *basic size* of the\n   object plus the *item size* times the number of allocated *items*,\n   *references* to referents.  The *flat size* does include the size\n   for the *references* to the referents, but not the size of the\n   referents themselves.\n\n   The *flat size* returned by function *flatsize* equals the result\n   of function *asizeof* with options *code=True*, *ignored=False*,\n   *limit=0* and option *align* set to the same value.\n\n   The accurate *flat size* for an object is obtained from function\n   ``sys.getsizeof()`` where available.  Otherwise, the *length* and\n   *size* of sequence objects as ``dicts``, ``lists``, ``sets``, etc.\n   is based on an estimate for the number of allocated items.  As a\n   result, the reported *length* and *size* may differ substantially\n   from the actual *length* and *size*.\n\n   The *basic* and *item size* are obtained from the ``__basicsize__``\n   respectively ``__itemsize__`` attributes of the (type of the)\n   object.  Where necessary (e.g. sequence objects), a zero\n   ``__itemsize__`` is replaced by the size of a corresponding C type.\n\n   The overhead for Python's garbage collector (GC) is included in\n   the *basic size* of (GC managed) objects as well as the space\n   needed for ``refcounts`` (used only in certain Python builds).\n\n   Optionally, size values can be aligned to any power-of-2 multiple.\n\n**Size of (byte)code**\n\n   The *(byte)code size* of objects like classes, functions, methods,\n   modules, etc. can be included by setting option *code=True*.\n\n   Iterators are handled like sequences: iterated object(s) are sized\n   like *referents* [#refs]_, but only up to the specified level or\n   recursion *limit* (and only if function ``gc.get_referents()``\n   returns the referent object of iterators).\n\n   Generators are sized as *(byte)code* only, but the objects are\n   never generated and never sized.\n\n**New-style Classes**\n\n   All ``class``, instance and ``type`` objects are handled uniformly\n   such that instance objects are distinguished from class objects.\n\n   Class and type objects are represented as ``<class .... def>``\n   respectively ``<type ... def>`` where the ``... def`` suffix marks\n   the *definition object*.  Instances of  classes are shown as\n   ``<class module.name>`` without the ``... def`` suffix.\n\n**Ignored Objects**\n\n   To avoid excessive sizes, several object types are ignored [#arb]_\n   by default, e.g. built-in functions, built-in types and classes\n   [#bi]_, function globals and module referents.  However, any\n   instances thereof and module objects will be sized when passed as\n   given objects.  Ignored object types are included unless option\n   *ignored* is set accordingly.\n\n   In addition, many ``__...__`` attributes of callable objects are\n   ignored [#arb]_, except crucial ones, e.g. class attributes ``__dict__``,\n   ``__doc__``, ``__name__`` and ``__slots__``.  For more details, see\n   the type-specific ``_..._refs()`` and ``_len_...()`` functions below.\n\n.. rubric:: Footnotes\n.. [#unsafe] The functions and classes in this module are not thread-safe.\n\n.. [#refs] The *referents* of an object are the objects referenced *by*\n     that object.  For example, the *referents* of a ``list`` are the\n     objects held in the ``list``, the *referents* of a ``dict`` are\n     the key and value objects in the ``dict``, etc.\n\n.. [#arb] These definitions and other assumptions are rather arbitrary\n     and may need corrections or adjustments.\n\n.. [#digit] The C ``sizeof(digit)`` in bytes can be obtained from the\n     ``int.__itemsize__``  attribute or since Python 3.1+  also from\n     attribute ``sys.int_info.sizeof_digit``.  Function **leng**\n     determines the number of ``digits`` of a multi-precision int.\n\n.. [#bi] All ``type``s and ``class``es in modules named in private set\n     ``_ignored_modules`` are ignored like other, standard built-ins.\n"
import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
__all__ = ['asizeof']
__version__ = '22.06.30'
_NN = ''
_Not_vari = _NN
_ignored_modules = {int.__module__, 'types', Exception.__module__, __name__}
_sizeof_Cbyte = calcsize('c')
_sizeof_Clong = calcsize('l')
_sizeof_Cvoidp = calcsize('P')
_z_P_L = 'P' if _sizeof_Clong < _sizeof_Cvoidp else 'L'

def _calcsize(fmt):
    if False:
        for i in range(10):
            print('nop')
    "Like struct.calcsize() but with 'z' for Py_ssize_t."
    return calcsize(fmt.replace('z', _z_P_L))
_sizeof_CPyCodeObject = _calcsize('Pz10P5i0P')
_sizeof_CPyFrameObject = _calcsize('Pzz13P63i0P')
_sizeof_CPyModuleObject = _calcsize('PzP0P')
_sizeof_CPyDictEntry = _calcsize('z2P')
_sizeof_Csetentry = _calcsize('lP')
u = '\x00'.encode('utf-8')
_sizeof_Cunicode = len(u)
del u
try:
    import _testcapi as t
    _sizeof_CPyGC_Head = t.SIZEOF_PYGC_HEAD
except (ImportError, AttributeError):
    t = calcsize('2d') - 1
    _sizeof_CPyGC_Head = _calcsize('2Pz') + t & ~t
t = hasattr(sys, 'gettotalrefcount')
_sizeof_Crefcounts = _calcsize('2z') if t else 0
del t
_Py_TPFLAGS_HEAPTYPE = 1 << 9
_Py_TPFLAGS_HAVE_GC = 1 << 14
_Type_type = type(type)
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents
if sys.platform == 'ios':
    _gc_getobjects = _getobjects

    def _getobjects():
        if False:
            return 10
        return tuple((o for o in _gc_getobjects() if not _isNULL(o)))
_getsizeof = sys.getsizeof

def _items(obj):
    if False:
        while True:
            i = 10
    'Return iter-/generator, preferably.'
    o = getattr(obj, 'iteritems', obj.items)
    return o() if callable(o) else o or ()

def _keys(obj):
    if False:
        for i in range(10):
            print('nop')
    'Return iter-/generator, preferably.'
    o = getattr(obj, 'iterkeys', obj.keys)
    return o() if callable(o) else o or ()

def _values(obj):
    if False:
        i = 10
        return i + 15
    'Return iter-/generator, preferably.'
    o = getattr(obj, 'itervalues', obj.values)
    return o() if callable(o) else o or ()
c = (lambda unused: lambda : unused)(None)
_cell_type = type(c.__closure__[0])
del c

def _basicsize(t, base=0, heap=False, obj=None):
    if False:
        while True:
            i = 10
    'Get non-zero basicsize of type,\n    including the header sizes.\n    '
    s = max(getattr(t, '__basicsize__', 0), base)
    if t != _Type_type:
        h = getattr(t, '__flags__', 0) & _Py_TPFLAGS_HAVE_GC
    elif heap:
        h = True
    else:
        h = getattr(obj, '__flags__', 0) & _Py_TPFLAGS_HEAPTYPE
    if h:
        s += _sizeof_CPyGC_Head
    return s + _sizeof_Crefcounts

def _classof(obj, dflt=None):
    if False:
        while True:
            i = 10
    "Return the object's class object."
    return getattr(obj, '__class__', dflt)

def _derive_typedef(typ):
    if False:
        print('Hello World!')
    'Return single, existing super type typedef or None.'
    v = [v for v in _values(_typedefs) if _issubclass(typ, v.type)]
    return v[0] if len(v) == 1 else None

def _dir2(obj, pref=_NN, excl=(), slots=None, itor=_NN):
    if False:
        i = 10
        return i + 15
    'Return an attribute name, object 2-tuple for certain\n    attributes or for the ``__slots__`` attributes of the\n    given object, but not both.  Any iterator referent\n    objects are returned with the given name if the\n    latter is non-empty.\n    '
    if slots:
        if hasattr(obj, slots):
            s = {}
            for c in type(obj).mro():
                n = _nameof(c)
                for a in getattr(c, slots, ()):
                    if a.startswith('__'):
                        a = '_' + n + a
                    if hasattr(obj, a):
                        s.setdefault(a, getattr(obj, a))
            for t in _items(s):
                yield t
    elif itor:
        for o in obj:
            yield (itor, o)
    else:
        for a in dir(obj):
            if a.startswith(pref) and hasattr(obj, a) and (a not in excl):
                yield (a, getattr(obj, a))

def _infer_dict(obj):
    if False:
        i = 10
        return i + 15
    'Return True for likely dict object via duck typing.'
    for attrs in (('items', 'keys', 'values'), ('iteritems', 'iterkeys', 'itervalues')):
        attrs += ('__len__', 'get', 'has_key')
        if all((callable(getattr(obj, a, None)) for a in attrs)):
            return True
    return False

def _isbuiltin2(typ):
    if False:
        i = 10
        return i + 15
    'Return True for built-in types as in Python 2.'
    return isbuiltin(typ) or typ is range

def _iscell(obj):
    if False:
        for i in range(10):
            print('nop')
    'Return True if obj is a cell as used in a closure.'
    return isinstance(obj, _cell_type)

def _isdictype(obj):
    if False:
        print('Hello World!')
    'Return True for known dict objects.'
    c = _classof(obj)
    n = _nameof(c)
    return n and n in _dict_types.get(_moduleof(c), ())

def _isframe(obj):
    if False:
        print('Hello World!')
    'Return True for a stack frame object.'
    try:
        return isframe(obj)
    except ReferenceError:
        return False

def _isignored(typ):
    if False:
        print('Hello World!')
    'Is this a type or class to be ignored?'
    return _moduleof(typ) in _ignored_modules

def _isnamedtuple(obj):
    if False:
        i = 10
        return i + 15
    'Named tuples are identified via duck typing:\n    <http://www.Gossamer-Threads.com/lists/python/dev/1142178>\n    '
    return isinstance(obj, tuple) and hasattr(obj, '_fields')

def _isNULL(obj):
    if False:
        i = 10
        return i + 15
    'Prevent asizeof(all=True, ...) crash.\n\n    Sizing gc.get_objects() crashes in Pythonista3 with\n    Python 3.5.1 on iOS due to 1-tuple (<Null>,) object,\n    see <http://forum.omz-software.com/user/mrjean1>.\n    '
    return isinstance(obj, tuple) and len(obj) == 1 and (repr(obj) == '(<NULL>,)')

def _issubclass(obj, Super):
    if False:
        i = 10
        return i + 15
    'Safe inspect.issubclass() returning None if Super is\n    *object* or if obj and Super are not a class or type.\n    '
    if Super is not object:
        try:
            return issubclass(obj, Super)
        except TypeError:
            pass
    return None

def _itemsize(t, item=0):
    if False:
        print('Hello World!')
    'Get non-zero itemsize of type.'
    return getattr(t, '__itemsize__', 0) or item

def _kwdstr(**kwds):
    if False:
        print('Hello World!')
    'Keyword arguments as a string.'
    return ', '.join(sorted(('%s=%r' % kv for kv in _items(kwds))))

def _lengstr(obj):
    if False:
        while True:
            i = 10
    'Object length as a string.'
    n = leng(obj)
    if n is None:
        r = _NN
    else:
        x = '!' if n > _len(obj) else _NN
        r = ' leng %d%s' % (n, x)
    return r

def _moduleof(obj, dflt=_NN):
    if False:
        for i in range(10):
            print('nop')
    "Return the object's module name."
    return getattr(obj, '__module__', dflt)

def _nameof(obj, dflt=_NN):
    if False:
        while True:
            i = 10
    'Return the name of an object.'
    return getattr(obj, '__name__', dflt)

def _objs_opts_x(where, objs, all=None, **opts):
    if False:
        print('Hello World!')
    "Return the given or 'all' objects plus\n    the remaining options and exclude flag\n    "
    if objs:
        (t, x) = (objs, False)
    elif all in (False, None):
        (t, x) = ((), True)
    elif all is True:
        (t, x) = (_getobjects(), True)
    else:
        raise _OptionError(where, all=all)
    return (t, opts, x)

def _OptionError(where, Error=ValueError, **options):
    if False:
        print('Hello World!')
    'Format an *Error* instance for invalid *option* or *options*.'
    t = (_plural(len(options)), _nameof(where), _kwdstr(**options))
    return Error('invalid option%s: %s(%s)' % t)

def _p100(part, total, prec=1):
    if False:
        i = 10
        return i + 15
    'Return percentage as string.'
    t = float(total)
    if t > 0:
        p = part * 100.0 / t
        r = '%.*f%%' % (prec, p)
    else:
        r = 'n/a'
    return r

def _plural(num):
    if False:
        for i in range(10):
            print('nop')
    "Return 's' if *num* is not one."
    return 's' if num != 1 else _NN

def _power_of_2(n):
    if False:
        i = 10
        return i + 15
    'Find the next power of 2.'
    p2 = 2 ** int(log(n, 2))
    while n > p2:
        p2 += p2
    return p2

def _prepr(obj, clip=0):
    if False:
        i = 10
        return i + 15
    'Prettify and clip long repr() string.'
    return _repr(obj, clip=clip).strip('<>').replace("'", _NN)

def _printf(fmt, *args, **print3options):
    if False:
        print('Hello World!')
    'Formatted print to sys.stdout or given stream.\n\n    *print3options* -- some keyword arguments, like Python 3+ print.\n    '
    if print3options:
        f = print3options.get('file', None) or sys.stdout
        if args:
            f.write(fmt % args)
        else:
            f.write(fmt)
        f.write(print3options.get('end', linesep))
        if print3options.get('flush', False):
            f.flush()
    elif args:
        print(fmt % args)
    else:
        print(fmt)

def _refs(obj, named, *attrs, **kwds):
    if False:
        while True:
            i = 10
    'Return specific attribute objects of an object.'
    if named:
        _N = _NamedRef
    else:

        def _N(unused, o):
            if False:
                i = 10
                return i + 15
            return o
    for a in attrs:
        if hasattr(obj, a):
            yield _N(a, getattr(obj, a))
    if kwds:
        for (a, o) in _dir2(obj, **kwds):
            yield _N(a, o)

def _repr(obj, clip=80):
    if False:
        i = 10
        return i + 15
    'Clip long repr() string.'
    try:
        r = repr(obj).replace(linesep, '\\n')
    except Exception:
        r = 'N/A'
    if len(r) > clip > 0:
        h = clip // 2 - 2
        if h > 0:
            r = r[:h] + '....' + r[-h:]
    return r

def _SI(size, K=1024, i='i'):
    if False:
        print('Hello World!')
    'Return size as SI string.'
    if 1 < K <= size:
        f = float(size)
        for si in iter('KMGPTE'):
            f /= K
            if f < K:
                return ' or %.1f %s%sB' % (f, si, i)
    return _NN

def _SI2(size, **kwds):
    if False:
        i = 10
        return i + 15
    'Return size as regular plus SI string.'
    return str(size) + _SI(size, **kwds)

def _cell_refs(obj, named):
    if False:
        i = 10
        return i + 15
    try:
        o = obj.cell_contents
        if named:
            o = _NamedRef('cell_contents', o)
        yield o
    except (AttributeError, ValueError):
        pass

def _class_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return specific referents of a class object.'
    return _refs(obj, named, '__class__', '__doc__', '__mro__', '__name__', '__slots__', '__weakref__', '__dict__')

def _co_refs(obj, named):
    if False:
        for i in range(10):
            print('nop')
    'Return specific referents of a code object.'
    return _refs(obj, named, pref='co_')

def _dict_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return key and value objects of a dict/proxy.'
    try:
        if named:
            for (k, v) in _items(obj):
                s = str(k)
                yield _NamedRef('[K] ' + s, k)
                s += ': ' + _repr(v)
                yield _NamedRef('[V] ' + s, v)
        else:
            for (k, v) in _items(obj):
                yield k
                yield v
    except (KeyError, ReferenceError, TypeError) as x:
        warnings.warn("Iterating '%s': %r" % (_classof(obj), x))

def _enum_refs(obj, named):
    if False:
        print('Hello World!')
    'Return specific referents of an enumerate object.'
    return _refs(obj, named, '__doc__')

def _exc_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return specific referents of an Exception object.'
    return _refs(obj, named, 'args', 'filename', 'lineno', 'msg', 'text')

def _file_refs(obj, named):
    if False:
        print('Hello World!')
    'Return specific referents of a file object.'
    return _refs(obj, named, 'mode', 'name')

def _frame_refs(obj, named):
    if False:
        print('Hello World!')
    'Return specific referents of a frame object.'
    return _refs(obj, named, pref='f_')

def _func_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return specific referents of a function or lambda object.'
    return _refs(obj, named, '__doc__', '__name__', '__code__', '__closure__', pref='func_', excl=('func_globals',))

def _gen_refs(obj, named):
    if False:
        while True:
            i = 10
    'Return the referent(s) of a generator (expression) object.'
    f = getattr(obj, 'gi_frame', None)
    return _refs(f, named, 'f_locals', 'f_code')

def _im_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return specific referents of a method object.'
    return _refs(obj, named, '__doc__', '__name__', '__code__', pref='im_')

def _inst_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return specific referents of a class instance.'
    return _refs(obj, named, '__dict__', '__class__', slots='__slots__')

def _iter_refs(obj, named):
    if False:
        print('Hello World!')
    'Return the referent(s) of an iterator object.'
    r = _getreferents(obj)
    return _refs(r, named, itor=_nameof(obj) or 'iteref')

def _module_refs(obj, named):
    if False:
        for i in range(10):
            print('nop')
    'Return specific referents of a module object.'
    n = _nameof(obj) == __name__
    return () if n else _dict_refs(obj.__dict__, named)

def _namedtuple_refs(obj, named):
    if False:
        print('Hello World!')
    'Return specific referents of obj-as-sequence and slots but exclude dict.'
    for r in _refs(obj, named, '__class__', slots='__slots__'):
        yield r
    for r in obj:
        yield r

def _prop_refs(obj, named):
    if False:
        return 10
    'Return specific referents of a property object.'
    return _refs(obj, named, '__doc__', pref='f')

def _seq_refs(obj, unused):
    if False:
        i = 10
        return i + 15
    'Return specific referents of a frozen/set, list, tuple and xrange object.'
    return obj

def _stat_refs(obj, named):
    if False:
        while True:
            i = 10
    'Return referents of a os.stat object.'
    return _refs(obj, named, pref='st_')

def _statvfs_refs(obj, named):
    if False:
        i = 10
        return i + 15
    'Return referents of a os.statvfs object.'
    return _refs(obj, named, pref='f_')

def _tb_refs(obj, named):
    if False:
        print('Hello World!')
    'Return specific referents of a traceback object.'
    return _refs(obj, named, pref='tb_')

def _type_refs(obj, named):
    if False:
        while True:
            i = 10
    'Return specific referents of a type object.'
    return _refs(obj, named, '__doc__', '__mro__', '__name__', '__slots__', '__weakref__', '__dict__')

def _weak_refs(obj, unused):
    if False:
        print('Hello World!')
    'Return weakly referent object.'
    try:
        return (obj(),)
    except Exception:
        return ()
_all_refs = {None, _cell_refs, _class_refs, _co_refs, _dict_refs, _enum_refs, _exc_refs, _file_refs, _frame_refs, _func_refs, _gen_refs, _im_refs, _inst_refs, _iter_refs, _module_refs, _namedtuple_refs, _prop_refs, _seq_refs, _stat_refs, _statvfs_refs, _tb_refs, _type_refs, _weak_refs}

def _len(obj):
    if False:
        return 10
    'Safe len().'
    try:
        return len(obj)
    except TypeError:
        return 0

def _len_bytearray(obj):
    if False:
        return 10
    'Bytearray size.'
    return obj.__alloc__()

def _len_code(obj):
    if False:
        return 10
    'Length of code object (stack and variables only).'
    return _len(obj.co_freevars) + obj.co_stacksize + _len(obj.co_cellvars) + obj.co_nlocals - 1

def _len_dict(obj):
    if False:
        i = 10
        return i + 15
    'Dict length in items (estimate).'
    n = len(obj)
    if n < 6:
        n = 0
    else:
        n = _power_of_2(n + 1)
    return n

def _len_frame(obj):
    if False:
        while True:
            i = 10
    'Length of a frame object.'
    c = getattr(obj, 'f_code', None)
    return _len_code(c) if c else 0

def _len_int(obj):
    if False:
        print('Hello World!')
    'Length of *int* (multi-precision, formerly long) in Cdigits.'
    n = _getsizeof(obj, 0) - int.__basicsize__
    return n // int.__itemsize__ if n > 0 else 0

def _len_iter(obj):
    if False:
        return 10
    'Length (hint) of an iterator.'
    n = getattr(obj, '__length_hint__', None)
    return n() if n and callable(n) else _len(obj)

def _len_list(obj):
    if False:
        for i in range(10):
            print('nop')
    'Length of list (estimate).'
    n = len(obj)
    if n > 8:
        n += 6 + (n >> 3)
    elif n:
        n += 4
    return n

def _len_module(obj):
    if False:
        return 10
    'Module length.'
    return _len(obj.__dict__)

def _len_set(obj):
    if False:
        return 10
    'Length of frozen/set (estimate).'
    n = len(obj)
    if n > 8:
        n = _power_of_2(n + n - 2)
    elif n:
        n = 8
    return n

def _len_slice(obj):
    if False:
        return 10
    'Slice length.'
    try:
        return (obj.stop - obj.start + 1) // obj.step
    except (AttributeError, TypeError):
        return 0

def _len_struct(obj):
    if False:
        i = 10
        return i + 15
    'Struct length in bytes.'
    try:
        return obj.size
    except AttributeError:
        return 0

def _len_unicode(obj):
    if False:
        print('Hello World!')
    'Unicode size.'
    return len(obj) + 1
_all_lens = {None, _len, _len_bytearray, _len_code, _len_dict, _len_frame, _len_int, _len_iter, _len_list, _len_module, _len_set, _len_slice, _len_struct, _len_unicode}

class _Claskey(object):
    """Wrapper for class objects."""
    __slots__ = ('_obj',)

    def __init__(self, obj):
        if False:
            i = 10
            return i + 15
        self._obj = obj

    def __str__(self):
        if False:
            i = 10
            return i + 15
        r = str(self._obj)
        return r[:-1] + ' def>' if r.endswith('>') else r + ' def'
    __repr__ = __str__
_claskeys = {}
_NoneNone = (None, None)

def _claskey(obj):
    if False:
        return 10
    'Wrap a class object.'
    i = id(obj)
    try:
        k = _claskeys[i]
    except KeyError:
        _claskeys[i] = k = _Claskey(obj)
    return k

def _key2tuple(obj):
    if False:
        print('Hello World!')
    'Return class and instance keys for a class.'
    t = type(obj) is _Type_type
    return (_claskey(obj), obj) if t else _NoneNone

def _objkey(obj):
    if False:
        return 10
    'Return the key for any object.'
    k = type(obj)
    if k is _Type_type:
        k = _claskey(obj)
    return k

class _NamedRef(object):
    """Store referred object along
    with the name of the referent.
    """
    __slots__ = ('name', 'ref')

    def __init__(self, name, ref):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.ref = ref
i = sys.intern
t = (_kind_static, _kind_dynamic, _kind_derived, _kind_ignored, _kind_inferred) = (i('static'), i('dynamic'), i('derived'), i('ignored'), i('inferred'))
_all_kinds = set(t)
del i, t

class _Typedef(object):
    """Type definition class."""
    base = 0
    both = None
    item = 0
    kind = None
    leng = None
    refs = None
    type = None
    vari = None
    xtyp = None

    def __init__(self, **kwds):
        if False:
            i = 10
            return i + 15
        self.reset(**kwds)

    def __lt__(self, unused):
        if False:
            for i in range(10):
                print('nop')
        return True

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.args())

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        t = [str(self.base), str(self.item)]
        for f in (self.leng, self.refs):
            t.append(_nameof(f) or 'n/a')
        if not self.both:
            t.append('(code only)')
        return ', '.join(t)

    def args(self):
        if False:
            print('Hello World!')
        'Return all attributes as arguments tuple.'
        return (self.base, self.item, self.leng, self.refs, self.both, self.kind, self.type, self.xtyp)

    def dup(self, other=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Duplicate attributes of dict or other typedef.'
        t = other or _dict_typedef
        d = t.kwds()
        d.update(kwds)
        self.reset(**d)

    def flat(self, obj, mask=0):
        if False:
            for i in range(10):
                print('nop')
        'Return the aligned flat size.'
        s = self.base
        if self.leng and self.item > 0:
            s += self.leng(obj) * self.item
        if not self.xtyp:
            s = _getsizeof(obj, s)
        if mask:
            s = s + mask & ~mask
        return s

    def format(self):
        if False:
            for i in range(10):
                print('nop')
        'Return format dict.'
        a = _nameof(self.leng)
        return dict(leng=' (%s)' % (a,) if a else _NN, item='var' if self.vari else self.item, code=_NN if self.both else ' (code only)', base=self.base, kind=self.kind)

    def kwds(self):
        if False:
            print('Hello World!')
        'Return all attributes as keywords dict.'
        return dict(base=self.base, both=self.both, item=self.item, kind=self.kind, leng=self.leng, refs=self.refs, type=self.type, vari=self.vari, xtyp=self.xtyp)

    def reset(self, base=0, item=0, leng=None, refs=None, both=True, kind=None, type=None, vari=_Not_vari, xtyp=False, **extra):
        if False:
            return 10
        'Reset all specified typedef attributes.'
        v = vari or _Not_vari
        if v != str(v):
            e = dict(vari=v)
        elif base < 0:
            e = dict(base=base)
        elif both not in (False, True):
            e = dict(both=both)
        elif item < 0:
            e = dict(item=item)
        elif kind not in _all_kinds:
            e = dict(kind=kind)
        elif leng not in _all_lens:
            e = dict(leng=leng)
        elif refs not in _all_refs:
            e = dict(refs=refs)
        elif xtyp not in (False, True):
            e = dict(xtyp=xtyp)
        elif extra:
            e = {}
        else:
            self.base = base
            self.both = both
            self.item = item
            self.kind = kind
            self.leng = leng
            self.refs = refs
            self.type = type
            self.vari = v
            self.xtyp = xtyp
            return
        e.update(extra)
        raise _OptionError(self.reset, **e)

    def save(self, t, base=0, heap=False):
        if False:
            return 10
        'Save this typedef plus its class typedef.'
        (c, k) = _key2tuple(t)
        if k and k not in _typedefs:
            _typedefs[k] = self
            if c and c not in _typedefs:
                b = _basicsize(type(t), base=base, heap=heap)
                k = _kind_ignored if _isignored(t) else self.kind
                _typedefs[c] = _Typedef(base=b, both=False, kind=k, type=t, refs=_type_refs)
        elif t not in _typedefs:
            if not _isbuiltin2(t):
                s = ' '.join((self.vari, _moduleof(t), _nameof(t)))
                s = '%r %s %s' % ((c, k), self.both, s.strip())
                raise KeyError('typedef %r bad: %s' % (self, s))
            _typedefs[t] = _Typedef(base=_basicsize(t, base=base), both=False, kind=_kind_ignored, type=t)

    def set(self, safe_len=False, **kwds):
        if False:
            while True:
                i = 10
        'Set one or more attributes.'
        if kwds:
            d = self.kwds()
            d.update(kwds)
            self.reset(**d)
        if safe_len and self.item:
            self.leng = _len
_typedefs = {}

def _typedef_both(t, base=0, item=0, leng=None, refs=None, kind=_kind_static, heap=False, vari=_Not_vari):
    if False:
        print('Hello World!')
    'Add new typedef for both data and code.'
    v = _Typedef(base=_basicsize(t, base=base), item=_itemsize(t, item), refs=refs, leng=leng, both=True, kind=kind, type=t, vari=vari)
    v.save(t, base=base, heap=heap)
    return v

def _typedef_code(t, base=0, refs=None, kind=_kind_static, heap=False):
    if False:
        for i in range(10):
            print('nop')
    'Add new typedef for code only.'
    v = _Typedef(base=_basicsize(t, base=base), refs=refs, both=False, kind=kind, type=t)
    v.save(t, base=base, heap=heap)
    return v
_typedef_both(complex)
_typedef_both(float)
_typedef_both(int, leng=_len_int)
_typedef_both(list, refs=_seq_refs, leng=_len_list, item=_sizeof_Cvoidp)
_typedef_both(tuple, refs=_seq_refs, leng=_len, item=_sizeof_Cvoidp)
_typedef_both(property, refs=_prop_refs)
_typedef_both(type(Ellipsis))
_typedef_both(type(None))
_dict_typedef = _typedef_both(dict, item=_sizeof_CPyDictEntry, leng=_len_dict, refs=_dict_refs)
_typedef_both(type(_Typedef.__dict__), item=_sizeof_CPyDictEntry, leng=_len_dict, refs=_dict_refs)
_dict_types = dict(UserDict=('IterableUserDict', 'UserDict'), weakref=('WeakKeyDictionary', 'WeakValueDictionary'))
try:
    _typedef_both(Types.ModuleType, base=_dict_typedef.base, item=_dict_typedef.item + _sizeof_CPyModuleObject, leng=_len_module, refs=_module_refs)
except AttributeError:
    pass
from array import array as _array

def _len_array(obj):
    if False:
        while True:
            i = 10
    'Array length (in bytes!).'
    return len(obj) * obj.itemsize

def _array_kwds(obj):
    if False:
        i = 10
        return i + 15
    b = max(56, _getsizeof(obj, 0) - _len_array(obj))
    return dict(base=b, leng=_len_array, item=_sizeof_Cbyte, vari='itemsize', xtyp=True)
_all_lens.add(_len_array)
try:
    _typedef_both(bool)
except NameError:
    pass
try:
    _typedef_both(bytearray, item=_sizeof_Cbyte, leng=_len_bytearray)
except NameError:
    pass
try:
    if type(bytes) is not type(str):
        _typedef_both(bytes, item=_sizeof_Cbyte, leng=_len)
except NameError:
    pass
try:
    _typedef_both(enumerate, refs=_enum_refs)
except NameError:
    pass
try:
    _typedef_both(Exception, refs=_exc_refs)
except Exception:
    pass
try:
    _typedef_both(frozenset, item=_sizeof_Csetentry, leng=_len_set, refs=_seq_refs)
except NameError:
    pass
try:
    _typedef_both(set, item=_sizeof_Csetentry, leng=_len_set, refs=_seq_refs)
except NameError:
    pass
try:
    _typedef_both(Types.GetSetDescriptorType)
except AttributeError:
    pass
try:
    _typedef_both(Types.MemberDescriptorType)
except AttributeError:
    pass
try:
    _typedef_both(type(NotImplemented))
except NameError:
    pass
try:
    import numpy as _numpy
    try:
        _numpy_memmap = _numpy.memmap
    except AttributeError:
        _numpy_memmap = None
    try:
        from mmap import PAGESIZE as _PAGESIZE
        if _PAGESIZE < 1024:
            raise ImportError
    except ImportError:
        _PAGESIZE = 4096

    def _isnumpy(obj):
        if False:
            print('Hello World!')
        'Return True for a NumPy arange, array, matrix, memmap, ndarray, etc. instance.'
        if hasattr(obj, 'dtype') and hasattr(obj, 'itemsize') and hasattr(obj, 'nbytes'):
            try:
                return _moduleof(_classof(obj)).startswith('numpy') or _moduleof(type(obj)).startswith('numpy')
            except (AttributeError, OSError, ValueError):
                pass
        return False

    def _len_numpy(obj):
        if False:
            print('Hello World!')
        'NumPy array, matrix, etc. length (in bytes!).'
        return obj.nbytes

    def _len_numpy_memmap(obj):
        if False:
            while True:
                i = 10
        'Approximate NumPy memmap in-memory size (in bytes!).'
        nb = int(obj.nbytes * _amapped)
        return (nb + _PAGESIZE - 1) // _PAGESIZE * _PAGESIZE

    def _numpy_kwds(obj):
        if False:
            for i in range(10):
                print('nop')
        t = type(obj)
        if t is _numpy_memmap:
            (b, _len_, nb) = (144, _len_numpy_memmap, 0)
        else:
            (b, _len_, nb) = (96, _len_numpy, obj.nbytes)
        return dict(base=_getsizeof(obj, b) - nb, item=_sizeof_Cbyte, leng=_len_, refs=_numpy_refs, vari='itemsize', xtyp=True)

    def _numpy_refs(obj, named):
        if False:
            return 10
        'Return the .base object for NumPy slices, views, etc.'
        return _refs(obj, named, 'base')
    _all_lens.add(_len_numpy)
    _all_lens.add(_len_numpy_memmap)
    _all_refs.add(_numpy_refs)
except ImportError:
    _numpy = _numpy_kwds = None

    def _isnumpy(unused):
        if False:
            for i in range(10):
                print('nop')
        'Not applicable, no NumPy.'
        return False
try:
    _typedef_both(range)
except NameError:
    pass
try:
    _typedef_both(reversed, refs=_enum_refs)
except NameError:
    pass
try:
    _typedef_both(slice, item=_sizeof_Cvoidp, leng=_len_slice)
except NameError:
    pass
try:
    from os import stat
    _typedef_both(type(stat(curdir)), refs=_stat_refs)
except ImportError:
    pass
try:
    from os import statvfs
    _typedef_both(type(statvfs(curdir)), refs=_statvfs_refs, item=_sizeof_Cvoidp, leng=_len)
except ImportError:
    pass
try:
    from struct import Struct
    _typedef_both(Struct, item=_sizeof_Cbyte, leng=_len_struct)
except ImportError:
    pass
try:
    _typedef_both(Types.TracebackType, refs=_tb_refs)
except AttributeError:
    pass
_typedef_both(str, leng=_len_unicode, item=_sizeof_Cunicode)
try:
    _typedef_both(Weakref.KeyedRef, refs=_weak_refs, heap=True)
except AttributeError:
    pass
try:
    _typedef_both(Weakref.ProxyType)
except AttributeError:
    pass
try:
    _typedef_both(Weakref.ReferenceType, refs=_weak_refs)
except AttributeError:
    pass
_typedef_code(object, kind=_kind_ignored)
_typedef_code(super, kind=_kind_ignored)
_typedef_code(_Type_type, kind=_kind_ignored)
try:
    _typedef_code(classmethod, refs=_im_refs)
except NameError:
    pass
try:
    _typedef_code(staticmethod, refs=_im_refs)
except NameError:
    pass
try:
    _typedef_code(Types.MethodType, refs=_im_refs)
except NameError:
    pass
try:
    _typedef_both(Types.GeneratorType, refs=_gen_refs)
except AttributeError:
    pass
try:
    _typedef_code(Weakref.CallableProxyType, refs=_weak_refs)
except AttributeError:
    pass
s = [_items({}), _keys({}), _values({})]
try:
    s.extend([reversed([]), reversed(())])
except NameError:
    pass
try:
    from re import finditer
    s.append(finditer(_NN, _NN))
    del finditer
except ImportError:
    pass
for t in _values(_typedefs):
    if t.type and t.leng:
        try:
            s.append(t.type())
        except TypeError:
            pass
for t in s:
    try:
        i = iter(t)
        _typedef_both(type(i), leng=_len_iter, refs=_iter_refs, item=0)
    except (KeyError, TypeError):
        pass
del i, s, t

def _typedef(obj, derive=False, frames=False, infer=False):
    if False:
        return 10
    'Create a new typedef for an object.'
    t = type(obj)
    v = _Typedef(base=_basicsize(t, obj=obj), kind=_kind_dynamic, type=t)
    if ismodule(obj):
        v.dup(item=_dict_typedef.item + _sizeof_CPyModuleObject, leng=_len_module, refs=_module_refs)
    elif _isframe(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyFrameObject, obj=obj), item=_itemsize(t), leng=_len_frame, refs=_frame_refs)
        if not frames:
            v.set(kind=_kind_ignored)
    elif iscode(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyCodeObject, obj=obj), item=_sizeof_Cvoidp, leng=_len_code, refs=_co_refs, both=False)
    elif callable(obj):
        if isclass(obj):
            v.set(refs=_class_refs, both=False)
            if _isignored(obj):
                v.set(kind=_kind_ignored)
        elif isbuiltin(obj):
            v.set(both=False, kind=_kind_ignored)
        elif isfunction(obj):
            v.set(refs=_func_refs, both=False)
        elif ismethod(obj):
            v.set(refs=_im_refs, both=False)
        elif isclass(t):
            v.set(item=_itemsize(t), safe_len=True, refs=_inst_refs)
        else:
            v.set(both=False)
    elif _issubclass(t, dict):
        v.dup(kind=_kind_derived)
    elif _isdictype(obj) or (infer and _infer_dict(obj)):
        v.dup(kind=_kind_inferred)
    elif _iscell(obj):
        v.set(item=_itemsize(t), refs=_cell_refs)
    elif _isnamedtuple(obj):
        v.set(refs=_namedtuple_refs)
    elif _numpy and _isnumpy(obj):
        v.set(**_numpy_kwds(obj))
    elif isinstance(obj, _array):
        v.set(**_array_kwds(obj))
    elif _isignored(obj):
        v.set(kind=_kind_ignored)
    else:
        if derive:
            p = _derive_typedef(t)
            if p:
                v.dup(other=p, kind=_kind_derived)
                return v
        if _issubclass(t, Exception):
            v.set(item=_itemsize(t), safe_len=True, refs=_exc_refs, kind=_kind_derived)
        elif isinstance(obj, Exception):
            v.set(item=_itemsize(t), safe_len=True, refs=_exc_refs)
        else:
            v.set(item=_itemsize(t), safe_len=True, refs=_inst_refs)
    return v

class _Prof(object):
    """Internal type profile class."""
    high = 0
    number = 0
    objref = None
    total = 0
    weak = False

    def __cmp__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.total < other.total:
            return -1
        elif self.total > other.total:
            return +1
        elif self.number < other.number:
            return -1
        elif self.number > other.number:
            return +1
        return 0

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__cmp__(other) < 0

    def format(self, clip=0, grand=None):
        if False:
            for i in range(10):
                print('nop')
        'Return format dict.'
        if self.number > 1:
            (a, p) = (int(self.total / self.number), 's')
        else:
            (a, p) = (self.total, _NN)
        o = self.objref
        if self.weak:
            o = o()
        t = _SI2(self.total)
        if grand:
            t += ' (%s)' % _p100(self.total, grand, prec=0)
        return dict(avg=_SI2(a), high=_SI2(self.high), lengstr=_lengstr(o), obj=_repr(o, clip=clip), plural=p, total=t)

    def update(self, obj, size):
        if False:
            for i in range(10):
                print('nop')
        'Update this profile.'
        self.number += 1
        self.total += size
        if self.high < size:
            self.high = size
            try:
                (self.objref, self.weak) = (Weakref.ref(obj), True)
            except TypeError:
                (self.objref, self.weak) = (obj, False)

class _Rank(object):
    """Internal largest object class."""
    deep = 0
    id = 0
    key = None
    objref = None
    pid = 0
    size = 0
    weak = False

    def __init__(self, key, obj, size, deep, pid):
        if False:
            while True:
                i = 10
        self.deep = deep
        self.id = id(obj)
        self.key = key
        try:
            (self.objref, self.weak) = (Weakref.ref(obj), True)
        except TypeError:
            (self.objref, self.weak) = (obj, False)
        self.pid = pid
        self.size = size

    def format(self, clip=0, id2x={}):
        if False:
            return 10
        'Return this *rank* as string.'

        def _ix(_id):
            if False:
                for i in range(10):
                    print('nop')
            return id2x.get(_id, '?')
        o = self.objref() if self.weak else self.objref
        d = ' (at %s)' % (self.deep,) if self.deep > 0 else _NN
        p = ', pix %s' % (_ix(self.pid),) if self.pid else _NN
        return '%s: %s%s, ix %s%s%s' % (_prepr(self.key, clip=clip), _repr(o, clip=clip), _lengstr(o), _ix(self.id), d, p)

class _Seen(dict):
    """Internal obj visits counter."""

    def again(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            s = self[key] + 1
        except KeyError:
            s = 1
        if s > 0:
            self[key] = s

class Asized(object):
    """Stores the results of an **asized** object in the following
    4 attributes:

     *size* -- total size of the object (including referents)

     *flat* -- flat size of the object (in bytes)

     *name* -- name or ``repr`` of the object

     *refs* -- tuple containing an **Asized** instance for each referent
    """
    __slots__ = ('flat', 'name', 'refs', 'size')

    def __init__(self, size, flat, refs=(), name=None):
        if False:
            for i in range(10):
                print('nop')
        self.size = size
        self.flat = flat
        self.name = name
        self.refs = tuple(refs)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'size %r, flat %r, refs[%d], name %r' % (self.size, self.flat, len(self.refs), self.name)

    def format(self, format='%(name)s size=%(size)d flat=%(flat)d', depth=-1, order_by='size', indent=_NN):
        if False:
            i = 10
            return i + 15
        "Format the size information of the object and of all\n        sized referents as a string.\n\n         *format* -- Specifies the format per instance (with 'name',\n                     'size' and 'flat' as interpolation parameters)\n\n         *depth* -- Recursion level up to which the referents are\n                    printed (use -1 for unlimited)\n\n         *order_by* -- Control sort order of referents, valid choices\n                       are 'name', 'size' and 'flat'\n\n         *indent* -- Optional indentation (default '')\n        "
        t = indent + format % dict(size=self.size, flat=self.flat, name=self.name)
        if depth and self.refs:
            rs = sorted(self.refs, key=lambda x: getattr(x, order_by), reverse=order_by in ('size', 'flat'))
            rs = [r.format(format=format, depth=depth - 1, order_by=order_by, indent=indent + '    ') for r in rs]
            t = '\n'.join([t] + rs)
        return t

    def get(self, name, dflt=None):
        if False:
            while True:
                i = 10
        'Return the named referent (or *dflt* if not found).'
        for ref in self.refs:
            if name == ref.name:
                return ref
        return dflt

class Asizer(object):
    """Sizer state and options to accumulate sizes."""
    _above_ = 1024
    _align_ = 8
    _clip_ = 80
    _code_ = False
    _cutoff_ = 0
    _derive_ = False
    _detail_ = 0
    _frames_ = False
    _infer_ = False
    _limit_ = 100
    _stats_ = 0
    _depth = 0
    _excl_d = None
    _ign_d = _kind_ignored
    _incl = _NN
    _mask = 7
    _missed = 0
    _profile = False
    _profs = None
    _ranked = 0
    _ranks = []
    _seen = None
    _stream = None
    _total = 0

    def __init__(self, **opts):
        if False:
            return 10
        'New **Asizer** accumulator.\n\n        See this module documentation for more details.\n        See method **reset** for all available options and defaults.\n        '
        self._excl_d = {}
        self.reset(**opts)

    def _c100(self, stats):
        if False:
            while True:
                i = 10
        'Cutoff as percentage (for backward compatibility)'
        s = int(stats)
        c = int((stats - s) * 100.0 + 0.5) or self.cutoff
        return (s, c)

    def _clear(self):
        if False:
            return 10
        'Clear state.'
        self._depth = 0
        self._incl = _NN
        self._missed = 0
        self._profile = False
        self._profs = {}
        self._ranked = 0
        self._ranks = []
        self._seen = _Seen()
        self._total = 0
        for k in _keys(self._excl_d):
            self._excl_d[k] = 0
        m = sys.modules[__name__]
        self.exclude_objs(self, self._excl_d, self._profs, self._ranks, self._seen, m, m.__dict__, m.__doc__, _typedefs)

    def _nameof(self, obj):
        if False:
            for i in range(10):
                print('nop')
        "Return the object's name."
        return _nameof(obj, _NN) or self._repr(obj)

    def _prepr(self, obj):
        if False:
            print('Hello World!')
        'Like **prepr()**.'
        return _prepr(obj, clip=self._clip_)

    def _printf(self, fmt, *args, **print3options):
        if False:
            print('Hello World!')
        'Print to sys.stdout or the configured stream if any is\n        specified and if the file keyword argument is not already\n        set in the **print3options** for this specific call.\n        '
        if self._stream and (not print3options.get('file', None)):
            if args:
                fmt = fmt % args
            _printf(fmt, file=self._stream, **print3options)
        else:
            _printf(fmt, *args, **print3options)

    def _prof(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Get _Prof object.'
        p = self._profs.get(key, None)
        if not p:
            self._profs[key] = p = _Prof()
            self.exclude_objs(p)
        return p

    def _rank(self, key, obj, size, deep, pid):
        if False:
            return 10
        'Rank 100 largest objects by size.'
        rs = self._ranks
        (i, j) = (0, len(rs))
        while i < j:
            m = (i + j) // 2
            if size < rs[m].size:
                i = m + 1
            else:
                j = m
        if i < 100:
            r = _Rank(key, obj, size, deep, pid)
            rs.insert(i, r)
            self.exclude_objs(r)
            while len(rs) > 100:
                rs.pop()
        self._ranked += 1

    def _repr(self, obj):
        if False:
            while True:
                i = 10
        'Like ``repr()``.'
        return _repr(obj, clip=self._clip_)

    def _sizer(self, obj, pid, deep, sized):
        if False:
            i = 10
            return i + 15
        'Size an object, recursively.'
        (s, f, i) = (0, 0, id(obj))
        if i not in self._seen:
            self._seen[i] = 1
        elif deep or self._seen[i]:
            self._seen.again(i)
            if sized:
                s = sized(s, f, name=self._nameof(obj))
                self.exclude_objs(s)
            return s
        else:
            self._seen.again(i)
        try:
            (k, rs) = (_objkey(obj), [])
            if k in self._excl_d:
                self._excl_d[k] += 1
            else:
                v = _typedefs.get(k, None)
                if not v:
                    _typedefs[k] = v = _typedef(obj, derive=self._derive_, frames=self._frames_, infer=self._infer_)
                if (v.both or self._code_) and v.kind is not self._ign_d:
                    s = f = v.flat(obj, self._mask)
                    if self._profile:
                        self._prof(k).update(obj, s)
                    if v.refs and deep < self._limit_ and (not (deep and ismodule(obj))):
                        (z, d) = (self._sizer, deep + 1)
                        if sized and deep < self._detail_:
                            self.exclude_objs(rs)
                            for o in v.refs(obj, True):
                                if isinstance(o, _NamedRef):
                                    r = z(o.ref, i, d, sized)
                                    r.name = o.name
                                else:
                                    r = z(o, i, d, sized)
                                    r.name = self._nameof(o)
                                rs.append(r)
                                s += r.size
                        else:
                            for o in v.refs(obj, False):
                                s += z(o, i, d, None)
                        if self._depth < d:
                            self._depth = d
                if self._stats_ and s > self._above_ > 0:
                    self._rank(k, obj, s, deep, pid)
        except RuntimeError:
            self._missed += 1
        if not deep:
            self._total += s
        if sized:
            s = sized(s, f, name=self._nameof(obj), refs=rs)
            self.exclude_objs(s)
        return s

    def _sizes(self, objs, sized=None):
        if False:
            i = 10
            return i + 15
        'Return the size or an **Asized** instance for each\n        given object plus the total size.  The total includes\n        the size of duplicates only once.\n        '
        self.exclude_refs(*objs)
        (s, t) = ({}, [])
        self.exclude_objs(s, t)
        for o in objs:
            i = id(o)
            if i in s:
                self._seen.again(i)
            else:
                s[i] = self._sizer(o, 0, 0, sized)
            t.append(s[i])
        return tuple(t)

    @property
    def above(self):
        if False:
            i = 10
            return i + 15
        'Get the large object size threshold (int).'
        return self._above_

    @property
    def align(self):
        if False:
            i = 10
            return i + 15
        'Get the size alignment (int).'
        return self._align_

    def asized(self, *objs, **opts):
        if False:
            while True:
                i = 10
        'Size each object and return an **Asized** instance with\n        size information and referents up to the given detail\n        level (and with modified options, see method **set**).\n\n        If only one object is given, the return value is the\n        **Asized** instance for that object.  The **Asized** size\n        of duplicate and ignored objects will be zero.\n        '
        if opts:
            self.set(**opts)
        t = self._sizes(objs, Asized)
        return t[0] if len(t) == 1 else t

    def asizeof(self, *objs, **opts):
        if False:
            i = 10
            return i + 15
        'Return the combined size of the given objects\n        (with modified options, see method **set**).\n        '
        if opts:
            self.set(**opts)
        self.exclude_refs(*objs)
        return sum((self._sizer(o, 0, 0, None) for o in objs))

    def asizesof(self, *objs, **opts):
        if False:
            for i in range(10):
                print('nop')
        'Return the individual sizes of the given objects\n        (with modified options, see method  **set**).\n\n        The size of duplicate and ignored objects will be zero.\n        '
        if opts:
            self.set(**opts)
        return self._sizes(objs, None)

    @property
    def clip(self):
        if False:
            return 10
        'Get the clipped string length (int).'
        return self._clip_

    @property
    def code(self):
        if False:
            i = 10
            return i + 15
        'Size (byte) code (bool).'
        return self._code_

    @property
    def cutoff(self):
        if False:
            print('Hello World!')
        'Stats cutoff (int).'
        return self._cutoff_

    @property
    def derive(self):
        if False:
            print('Hello World!')
        'Derive types (bool).'
        return self._derive_

    @property
    def detail(self):
        if False:
            print('Hello World!')
        'Get the detail level for **Asized** refs (int).'
        return self._detail_

    @property
    def duplicate(self):
        if False:
            print('Hello World!')
        'Get the number of duplicate objects seen so far (int).'
        return sum((1 for v in _values(self._seen) if v > 1))

    def exclude_objs(self, *objs):
        if False:
            i = 10
            return i + 15
        'Exclude the specified objects from sizing, profiling and ranking.'
        for o in objs:
            self._seen.setdefault(id(o), -1)

    def exclude_refs(self, *objs):
        if False:
            i = 10
            return i + 15
        'Exclude any references to the specified objects from sizing.\n\n        While any references to the given objects are excluded, the\n        objects will be sized if specified as positional arguments\n        in subsequent calls to methods **asizeof** and **asizesof**.\n        '
        for o in objs:
            self._seen.setdefault(id(o), 0)

    def exclude_types(self, *objs):
        if False:
            for i in range(10):
                print('nop')
        'Exclude the specified object instances and types from sizing.\n\n        All instances and types of the given objects are excluded,\n        even objects specified as positional arguments in subsequent\n        calls to methods **asizeof** and **asizesof**.\n        '
        for o in objs:
            for t in _key2tuple(o):
                if t and t not in self._excl_d:
                    self._excl_d[t] = 0

    @property
    def excluded(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the types being excluded (tuple).'
        return tuple(_keys(self._excl_d))

    @property
    def frames(self):
        if False:
            print('Hello World!')
        'Ignore stack frames (bool).'
        return self._frames_

    @property
    def ignored(self):
        if False:
            return 10
        'Ignore certain types (bool).'
        return True if self._ign_d else False

    @property
    def infer(self):
        if False:
            while True:
                i = 10
        'Infer types (bool).'
        return self._infer_

    @property
    def limit(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the recursion limit (int).'
        return self._limit_

    @property
    def missed(self):
        if False:
            i = 10
            return i + 15
        'Get the number of objects missed due to errors (int).'
        return self._missed

    def print_largest(self, w=0, cutoff=0, **print3options):
        if False:
            for i in range(10):
                print('nop')
        'Print the largest objects.\n\n        The available options and defaults are:\n\n         *w=0*           -- indentation for each line\n\n         *cutoff=100*    -- number of largest objects to print\n\n         *print3options* -- some keyword arguments, like Python 3+ print\n        '
        c = int(cutoff) if cutoff else self._cutoff_
        n = min(len(self._ranks), max(c, 0))
        s = self._above_
        if n > 0 and s > 0:
            self._printf('%s%*d largest object%s (of %d over %d bytes%s)', linesep, w, n, _plural(n), self._ranked, s, _SI(s), **print3options)
            id2x = dict(((r.id, i) for (i, r) in enumerate(self._ranks)))
            for r in self._ranks[:n]:
                (s, t) = (r.size, r.format(self._clip_, id2x))
                self._printf('%*d bytes%s: %s', w, s, _SI(s), t, **print3options)

    def print_profiles(self, w=0, cutoff=0, **print3options):
        if False:
            i = 10
            return i + 15
        'Print the profiles above *cutoff* percentage.\n\n        The available options and defaults are:\n\n             *w=0*           -- indentation for each line\n\n             *cutoff=0*      -- minimum percentage printed\n\n             *print3options* -- some keyword arguments, like Python 3+ print\n        '
        t = [(v, k) for (k, v) in _items(self._profs) if v.total > 0 or v.number > 1]
        if len(self._profs) - len(t) < 9:
            t = [(v, k) for (k, v) in _items(self._profs)]
        if t:
            s = _NN
            if self._total:
                s = ' (% of grand total)'
                c = int(cutoff) if cutoff else self._cutoff_
                C = int(c * 0.01 * self._total)
            else:
                C = c = 0
            self._printf('%s%*d profile%s:  total%s, average, and largest flat size%s:  largest object', linesep, w, len(t), _plural(len(t)), s, self._incl, **print3options)
            r = len(t)
            t = [(v, self._prepr(k)) for (v, k) in t]
            for (v, k) in sorted(t, reverse=True):
                s = 'object%(plural)s:  %(total)s, %(avg)s, %(high)s:  %(obj)s%(lengstr)s' % v.format(self._clip_, self._total)
                self._printf('%*d %s %s', w, v.number, k, s, **print3options)
                r -= 1
                if r > 1 and v.total < C:
                    self._printf('%+*d profiles below cutoff (%.0f%%)', w, r, c)
                    break
            z = len(self._profs) - len(t)
            if z > 0:
                self._printf('%+*d %r object%s', w, z, 'zero', _plural(z), **print3options)

    def print_stats(self, objs=(), opts={}, sized=(), sizes=(), stats=3, **print3options):
        if False:
            i = 10
            return i + 15
        'Prints the statistics.\n\n        The available options and defaults are:\n\n             *w=0*           -- indentation for each line\n\n             *objs=()*       -- optional, list of objects\n\n             *opts={}*       -- optional, dict of options used\n\n             *sized=()*      -- optional, tuple of **Asized** instances returned\n\n             *sizes=()*      -- optional, tuple of sizes returned\n\n             *stats=3*       -- print stats, see function **asizeof**\n\n             *print3options* -- some keyword arguments, like Python 3+ print\n        '
        s = min(opts.get('stats', stats) or 0, self.stats)
        if s > 0:
            w = len(str(self.missed + self.seen + self.total)) + 1
            t = c = _NN
            o = _kwdstr(**opts)
            if o and objs:
                c = ', '
            if sized and objs:
                n = len(objs)
                if n > 1:
                    self._printf('%sasized(...%s%s) ...', linesep, c, o, **print3options)
                    for i in range(n):
                        self._printf('%*d: %s', w - 1, i, sized[i], **print3options)
                else:
                    self._printf('%sasized(%s): %s', linesep, o, sized, **print3options)
            elif sizes and objs:
                self._printf('%sasizesof(...%s%s) ...', linesep, c, o, **print3options)
                for (z, o) in zip(sizes, objs):
                    self._printf('%*d bytes%s%s:  %s', w, z, _SI(z), self._incl, self._repr(o), **print3options)
            else:
                if objs:
                    t = self._repr(objs)
                self._printf('%sasizeof(%s%s%s) ...', linesep, t, c, o, **print3options)
            self.print_summary(w=w, objs=objs, **print3options)
            (s, c) = self._c100(s)
            self.print_largest(w=w, cutoff=c if s < 2 else 10, **print3options)
            if s > 1:
                self.print_profiles(w=w, cutoff=c, **print3options)
                if s > 2:
                    self.print_typedefs(w=w, **print3options)

    def print_summary(self, w=0, objs=(), **print3options):
        if False:
            while True:
                i = 10
        'Print the summary statistics.\n\n        The available options and defaults are:\n\n             *w=0*           -- indentation for each line\n\n             *objs=()*       -- optional, list of objects\n\n             *print3options* -- some keyword arguments, like Python 3+ print\n        '
        self._printf('%*d bytes%s%s', w, self._total, _SI(self._total), self._incl, **print3options)
        if self._mask:
            self._printf('%*d byte aligned', w, self._mask + 1, **print3options)
        self._printf('%*d byte sizeof(void*)', w, _sizeof_Cvoidp, **print3options)
        n = len(objs or ())
        self._printf('%*d object%s %s', w, n, _plural(n), 'given', **print3options)
        n = self.sized
        self._printf('%*d object%s %s', w, n, _plural(n), 'sized', **print3options)
        if self._excl_d:
            n = sum(_values(self._excl_d))
            self._printf('%*d object%s %s', w, n, _plural(n), 'excluded', **print3options)
        n = self.seen
        self._printf('%*d object%s %s', w, n, _plural(n), 'seen', **print3options)
        n = self.ranked
        if n > 0:
            self._printf('%*d object%s %s', w, n, _plural(n), 'ranked', **print3options)
        n = self.missed
        self._printf('%*d object%s %s', w, n, _plural(n), 'missed', **print3options)
        n = self.duplicate
        self._printf('%*d duplicate%s', w, n, _plural(n), **print3options)
        if self._depth > 0:
            self._printf('%*d deepest recursion', w, self._depth, **print3options)

    def print_typedefs(self, w=0, **print3options):
        if False:
            while True:
                i = 10
        'Print the types and dict tables.\n\n        The available options and defaults are:\n\n             *w=0*           -- indentation for each line\n\n             *print3options* -- some keyword arguments, like Python 3+ print\n        '
        for k in _all_kinds:
            t = [(self._prepr(a), v) for (a, v) in _items(_typedefs) if v.kind == k and (v.both or self._code_)]
            if t:
                self._printf('%s%*d %s type%s:  basicsize, itemsize, _len_(), _refs()', linesep, w, len(t), k, _plural(len(t)), **print3options)
                for (a, v) in sorted(t):
                    self._printf('%*s %s:  %s', w, _NN, a, v, **print3options)
        t = sum((len(v) for v in _values(_dict_types)))
        if t:
            self._printf('%s%*d dict/-like classes:', linesep, w, t, **print3options)
            for (m, v) in _items(_dict_types):
                self._printf('%*s %s:  %s', w, _NN, m, self._prepr(v), **print3options)

    @property
    def ranked(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the number objects ranked by size so far (int).'
        return self._ranked

    def reset(self, above=1024, align=8, clip=80, code=False, cutoff=10, derive=False, detail=0, frames=False, ignored=True, infer=False, limit=100, stats=0, stream=None, **extra):
        if False:
            i = 10
            return i + 15
        'Reset sizing options, state, etc. to defaults.\n\n        The available options and default values are:\n\n             *above=0*      -- threshold for largest objects stats\n\n             *align=8*      -- size alignment\n\n             *code=False*   -- incl. (byte)code size\n\n             *cutoff=10*    -- limit large objects or profiles stats\n\n             *derive=False* -- derive from super type\n\n             *detail=0*     -- **Asized** refs level\n\n             *frames=False* -- ignore frame objects\n\n             *ignored=True* -- ignore certain types\n\n             *infer=False*  -- try to infer types\n\n             *limit=100*    -- recursion limit\n\n             *stats=0*      -- print statistics, see function **asizeof**\n\n             *stream=None*  -- output stream for printing\n\n        See function **asizeof** for a description of the options.\n        '
        if extra:
            raise _OptionError(self.reset, Error=KeyError, **extra)
        self._above_ = above
        self._align_ = align
        self._clip_ = clip
        self._code_ = code
        self._cutoff_ = cutoff
        self._derive_ = derive
        self._detail_ = detail
        self._frames_ = frames
        self._infer_ = infer
        self._limit_ = limit
        self._stats_ = stats
        self._stream = stream
        if ignored:
            self._ign_d = _kind_ignored
        else:
            self._ign_d = None
        self._clear()
        self.set(align=align, code=code, cutoff=cutoff, stats=stats)

    @property
    def seen(self):
        if False:
            print('Hello World!')
        'Get the number objects seen so far (int).'
        return sum((v for v in _values(self._seen) if v > 0))

    def set(self, above=None, align=None, code=None, cutoff=None, frames=None, detail=None, limit=None, stats=None):
        if False:
            return 10
        'Set some sizing options.  See also **reset**.\n\n        The available options are:\n\n             *above*  -- threshold for largest objects stats\n\n             *align*  -- size alignment\n\n             *code*   -- incl. (byte)code size\n\n             *cutoff* -- limit large objects or profiles stats\n\n             *detail* -- **Asized** refs level\n\n             *frames* -- size or ignore frame objects\n\n             *limit*  -- recursion limit\n\n             *stats*  -- print statistics, see function **asizeof**\n\n        Any options not set remain unchanged from the previous setting.\n        '
        if above is not None:
            self._above_ = int(above)
        if align is not None:
            if align > 1:
                m = align - 1
                if m & align:
                    raise _OptionError(self.set, align=align)
            else:
                m = 0
            self._align_ = align
            self._mask = m
        if code is not None:
            self._code_ = code
            if code:
                self._incl = ' (incl. code)'
        if detail is not None:
            self._detail_ = detail
        if frames is not None:
            self._frames_ = frames
        if limit is not None:
            self._limit_ = limit
        if stats is not None:
            if stats < 0:
                raise _OptionError(self.set, stats=stats)
            (s, c) = self._c100(stats)
            self._cutoff_ = int(cutoff) if cutoff else c
            self._stats_ = s
            self._profile = s > 1

    @property
    def sized(self):
        if False:
            i = 10
            return i + 15
        'Get the number objects sized so far (int).'
        return sum((1 for v in _values(self._seen) if v > 0))

    @property
    def stats(self):
        if False:
            while True:
                i = 10
        'Get the stats and cutoff setting (float).'
        return self._stats_

    @property
    def total(self):
        if False:
            return 10
        'Get the total size (in bytes) accumulated so far.'
        return self._total

def amapped(percentage=None):
    if False:
        return 10
    'Set/get approximate mapped memory usage as a percentage\n    of the mapped file size.\n\n    Sets the new percentage if not None and returns the\n    previously set percentage.\n\n    Applies only to *numpy.memmap* objects.\n    '
    global _amapped
    p = _amapped * 100.0
    if percentage is not None:
        _amapped = max(0, min(1, percentage * 0.01))
    return p
_amapped = 0.01
_asizer = Asizer()

def asized(*objs, **opts):
    if False:
        for i in range(10):
            print('nop')
    'Return a tuple containing an **Asized** instance for each\n    object passed as positional argument.\n\n    The available options and defaults are:\n\n         *above=0*      -- threshold for largest objects stats\n\n         *align=8*      -- size alignment\n\n         *code=False*   -- incl. (byte)code size\n\n         *cutoff=10*    -- limit large objects or profiles stats\n\n         *derive=False* -- derive from super type\n\n         *detail=0*     -- Asized refs level\n\n         *frames=False* -- ignore stack frame objects\n\n         *ignored=True* -- ignore certain types\n\n         *infer=False*  -- try to infer types\n\n         *limit=100*    -- recursion limit\n\n         *stats=0*      -- print statistics\n\n    If only one object is given, the return value is the **Asized**\n    instance for that object.  Otherwise, the length of the returned\n    tuple matches the number of given objects.\n\n    The **Asized** size of duplicate and ignored objects will be zero.\n\n    Set *detail* to the desired referents level and *limit* to the\n    maximum recursion depth.\n\n    See function **asizeof** for descriptions of the other options.\n    '
    _asizer.reset(**opts)
    if objs:
        t = _asizer.asized(*objs)
        _asizer.print_stats(objs, opts=opts, sized=t)
        _asizer._clear()
    else:
        t = ()
    return t

def asizeof(*objs, **opts):
    if False:
        i = 10
        return i + 15
    'Return the combined size (in bytes) of all objects passed\n    as positional arguments.\n\n    The available options and defaults are:\n\n         *above=0*      -- threshold for largest objects stats\n\n         *align=8*      -- size alignment\n\n         *clip=80*      -- clip ``repr()`` strings\n\n         *code=False*   -- incl. (byte)code size\n\n         *cutoff=10*    -- limit large objects or profiles stats\n\n         *derive=False* -- derive from super type\n\n         *frames=False* -- ignore stack frame objects\n\n         *ignored=True* -- ignore certain types\n\n         *infer=False*  -- try to infer types\n\n         *limit=100*    -- recursion limit\n\n         *stats=0*      -- print statistics\n\n    Set *align* to a power of 2 to align sizes.  Any value less\n    than 2 avoids size alignment.\n\n    If *all* is True and if no positional arguments are supplied.\n    size all current gc objects, including module, global and stack\n    frame objects.\n\n    A positive *clip* value truncates all repr() strings to at\n    most *clip* characters.\n\n    The (byte)code size of callable objects like functions,\n    methods, classes, etc. is included only if *code* is True.\n\n    If *derive* is True, new types are handled like an existing\n    (super) type provided there is one and only of those.\n\n    By default certain base types like object, super, etc. are\n    ignored.  Set *ignored* to False to include those.\n\n    If *infer* is True, new types are inferred from attributes\n    (only implemented for dict types on callable attributes\n    as get, has_key, items, keys and values).\n\n    Set *limit* to a positive value to accumulate the sizes of\n    the referents of each object, recursively up to the limit.\n    Using *limit=0* returns the sum of the flat sizes of the\n    given objects.  High *limit* values may cause runtime errors\n    and miss objects for sizing.\n\n    A positive value for *stats* prints up to 9 statistics, (1)\n    a summary of the number of objects sized and seen and a list\n    of the largests objects with size over *above* bytes, (2) a\n    simple profile of the sized objects by type and (3+) up to 6\n    tables showing the static, dynamic, derived, ignored, inferred\n    and dict types used, found respectively installed.\n    The fractional part of the *stats* value (x 100) is the number\n    of largest objects shown for (*stats*1.+) or the cutoff\n    percentage for simple profiles for (*stats*=2.+).  For example,\n    *stats=1.10* shows the summary and the 10 largest objects,\n    also the default.\n\n    See this module documentation for the definition of flat size.\n    '
    (t, p, x) = _objs_opts_x(asizeof, objs, **opts)
    _asizer.reset(**p)
    if t:
        if x:
            _asizer.exclude_objs(t)
        s = _asizer.asizeof(*t)
        _asizer.print_stats(objs=t, opts=opts)
        _asizer._clear()
    else:
        s = 0
    return s

def asizesof(*objs, **opts):
    if False:
        for i in range(10):
            print('nop')
    'Return a tuple containing the size (in bytes) of all objects\n    passed as positional arguments.\n\n    The available options and defaults are:\n\n         *above=1024*   -- threshold for largest objects stats\n\n         *align=8*      -- size alignment\n\n         *clip=80*      -- clip ``repr()`` strings\n\n         *code=False*   -- incl. (byte)code size\n\n         *cutoff=10*    -- limit large objects or profiles stats\n\n         *derive=False* -- derive from super type\n\n         *frames=False* -- ignore stack frame objects\n\n         *ignored=True* -- ignore certain types\n\n         *infer=False*  -- try to infer types\n\n         *limit=100*    -- recursion limit\n\n         *stats=0*      -- print statistics\n\n    See function **asizeof** for a description of the options.\n\n    The length of the returned tuple equals the number of given\n    objects.\n\n    The size of duplicate and ignored objects will be zero.\n    '
    _asizer.reset(**opts)
    if objs:
        t = _asizer.asizesof(*objs)
        _asizer.print_stats(objs, opts=opts, sizes=t)
        _asizer._clear()
    else:
        t = ()
    return t

def _typedefof(obj, save=False, **opts):
    if False:
        print('Hello World!')
    'Get the typedef for an object.'
    k = _objkey(obj)
    v = _typedefs.get(k, None)
    if not v:
        v = _typedef(obj, **opts)
        if save:
            _typedefs[k] = v
    return v

def basicsize(obj, **opts):
    if False:
        return 10
    "Return the basic size of an object (in bytes).\n\n    The available options and defaults are:\n\n        *derive=False* -- derive type from super type\n\n        *infer=False*  -- try to infer types\n\n        *save=False*   -- save the object's type definition if new\n\n    See this module documentation for the definition of *basic size*.\n    "
    b = t = _typedefof(obj, **opts)
    if t:
        b = t.base
    return b

def flatsize(obj, align=0, **opts):
    if False:
        for i in range(10):
            print('nop')
    'Return the flat size of an object (in bytes), optionally aligned\n    to the given power-of-2.\n\n    See function **basicsize** for a description of other available options.\n\n    See this module documentation for the definition of *flat size*.\n    '
    f = t = _typedefof(obj, **opts)
    if t:
        if align > 1:
            m = align - 1
            if m & align:
                raise _OptionError(flatsize, align=align)
        else:
            m = 0
        f = t.flat(obj, mask=m)
    return f

def itemsize(obj, **opts):
    if False:
        while True:
            i = 10
    'Return the item size of an object (in bytes).\n\n    See function **basicsize** for a description of the available options.\n\n    See this module documentation for the definition of *item size*.\n    '
    i = t = _typedefof(obj, **opts)
    if t:
        (i, v) = (t.item, t.vari)
        if v and i == _sizeof_Cbyte:
            i = getattr(obj, v, i)
    return i

def leng(obj, **opts):
    if False:
        for i in range(10):
            print('nop')
    'Return the length of an object, in number of *items*.\n\n    See function **basicsize** for a description of the available options.\n    '
    n = t = _typedefof(obj, **opts)
    if t:
        n = t.leng
        if n and callable(n):
            (i, v, n) = (t.item, t.vari, n(obj))
            if v and i == _sizeof_Cbyte:
                i = getattr(obj, v, i)
                if i > _sizeof_Cbyte:
                    n = n // i
    return n

def named_refs(obj, **opts):
    if False:
        return 10
    'Return all named **referents** of an object (re-using\n    functionality from **asizeof**).\n\n    Does not return un-named *referents*, e.g. objects in a list.\n\n    See function **basicsize** for a description of the available options.\n    '
    rs = []
    v = _typedefof(obj, **opts)
    if v:
        v = v.refs
        if v and callable(v):
            for r in v(obj, True):
                try:
                    rs.append((r.name, r.ref))
                except AttributeError:
                    pass
    return rs

def refs(obj, **opts):
    if False:
        for i in range(10):
            print('nop')
    'Return (a generator for) specific *referents* of an object.\n\n    See function **basicsize** for a description of the available options.\n    '
    v = _typedefof(obj, **opts)
    if v:
        v = v.refs
        if v and callable(v):
            v = v(obj, False)
    return v