"""
From gdb 7 onwards, gdb's build can be configured --with-python, allowing gdb
to be extended with Python code e.g. for library-specific data visualizations,
such as for the C++ STL types.  Documentation on this API can be seen at:
http://sourceware.org/gdb/current/onlinedocs/gdb/Python-API.html


This python module deals with the case when the process being debugged (the
"inferior process" in gdb parlance) is itself python, or more specifically,
linked against libpython.  In this situation, almost every item of data is a
(PyObject*), and having the debugger merely print their addresses is not very
enlightening.

This module embeds knowledge about the implementation details of libpython so
that we can emit useful visualizations e.g. a string, a list, a dict, a frame
giving file/line information and the state of local variables

In particular, given a gdb.Value corresponding to a PyObject* in the inferior
process, we can generate a "proxy value" within the gdb process.  For example,
given a PyObject* in the inferior process that is in fact a PyListObject*
holding three PyObject* that turn out to be PyBytesObject* instances, we can
generate a proxy value within the gdb process that is a list of bytes
instances:
  [b"foo", b"bar", b"baz"]

Doing so can be expensive for complicated graphs of objects, and could take
some time, so we also have a "write_repr" method that writes a representation
of the data to a file-like object.  This allows us to stop the traversal by
having the file-like object raise an exception if it gets too much data.

With both "proxyval" and "write_repr" we keep track of the set of all addresses
visited so far in the traversal, to avoid infinite recursion due to cycles in
the graph of object references.

We try to defer gdb.lookup_type() invocations for python types until as late as
possible: for a dynamically linked python binary, when the process starts in
the debugger, the libpython.so hasn't been dynamically loaded yet, so none of
the type names are known to the debugger

The module also extends gdb with some python-specific commands.
"""
from __future__ import print_function
import gdb
import os
import locale
import sys
if sys.version_info[0] >= 3:
    unichr = chr
    xrange = range
    long = int

def _type_char_ptr():
    if False:
        while True:
            i = 10
    return gdb.lookup_type('char').pointer()

def _type_unsigned_char_ptr():
    if False:
        i = 10
        return i + 15
    return gdb.lookup_type('unsigned char').pointer()

def _type_unsigned_short_ptr():
    if False:
        return 10
    return gdb.lookup_type('unsigned short').pointer()

def _type_unsigned_int_ptr():
    if False:
        print('Hello World!')
    return gdb.lookup_type('unsigned int').pointer()

def _sizeof_void_p():
    if False:
        return 10
    return gdb.lookup_type('void').pointer().sizeof
_is_pep393 = None
Py_TPFLAGS_HEAPTYPE = 1 << 9
Py_TPFLAGS_LONG_SUBCLASS = 1 << 24
Py_TPFLAGS_LIST_SUBCLASS = 1 << 25
Py_TPFLAGS_TUPLE_SUBCLASS = 1 << 26
Py_TPFLAGS_BYTES_SUBCLASS = 1 << 27
Py_TPFLAGS_UNICODE_SUBCLASS = 1 << 28
Py_TPFLAGS_DICT_SUBCLASS = 1 << 29
Py_TPFLAGS_BASE_EXC_SUBCLASS = 1 << 30
Py_TPFLAGS_TYPE_SUBCLASS = 1 << 31
MAX_OUTPUT_LEN = 1024
hexdigits = '0123456789abcdef'
ENCODING = locale.getpreferredencoding()
FRAME_INFO_OPTIMIZED_OUT = '(frame information optimized out)'
UNABLE_READ_INFO_PYTHON_FRAME = 'Unable to read information on python frame'
EVALFRAME = '_PyEval_EvalFrameDefault'

class NullPyObjectPtr(RuntimeError):
    pass

def safety_limit(val):
    if False:
        i = 10
        return i + 15
    return min(val, 1000)

def safe_range(val):
    if False:
        print('Hello World!')
    return xrange(safety_limit(int(val)))
if sys.version_info[0] >= 3:

    def write_unicode(file, text):
        if False:
            return 10
        file.write(text)
else:

    def write_unicode(file, text):
        if False:
            i = 10
            return i + 15
        if isinstance(text, unicode):
            text = text.encode(ENCODING, 'backslashreplace')
        file.write(text)
try:
    os_fsencode = os.fsencode
except AttributeError:

    def os_fsencode(filename):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(filename, unicode):
            return filename
        encoding = sys.getfilesystemencoding()
        if encoding == 'mbcs':
            return filename.encode(encoding)
        encoded = []
        for char in filename:
            if 56448 <= ord(char) <= 56575:
                byte = chr(ord(char) - 56320)
            else:
                byte = char.encode(encoding)
            encoded.append(byte)
        return ''.join(encoded)

class StringTruncated(RuntimeError):
    pass

class TruncatedStringIO(object):
    """Similar to io.StringIO, but can truncate the output by raising a
    StringTruncated exception"""

    def __init__(self, maxlen=None):
        if False:
            print('Hello World!')
        self._val = ''
        self.maxlen = maxlen

    def write(self, data):
        if False:
            while True:
                i = 10
        if self.maxlen:
            if len(data) + len(self._val) > self.maxlen:
                self._val += data[0:self.maxlen - len(self._val)]
                raise StringTruncated()
        self._val += data

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        return self._val

class PyObjectPtr(object):
    """
    Class wrapping a gdb.Value that's either a (PyObject*) within the
    inferior process, or some subclass pointer e.g. (PyBytesObject*)

    There will be a subclass for every refined PyObject type that we care
    about.

    Note that at every stage the underlying pointer could be NULL, point
    to corrupt data, etc; this is the debugger, after all.
    """
    _typename = 'PyObject'

    def __init__(self, gdbval, cast_to=None):
        if False:
            print('Hello World!')
        if cast_to:
            self._gdbval = gdbval.cast(cast_to)
        else:
            self._gdbval = gdbval

    def field(self, name):
        if False:
            return 10
        '\n        Get the gdb.Value for the given field within the PyObject, coping with\n        some python 2 versus python 3 differences.\n\n        Various libpython types are defined using the "PyObject_HEAD" and\n        "PyObject_VAR_HEAD" macros.\n\n        In Python 2, this these are defined so that "ob_type" and (for a var\n        object) "ob_size" are fields of the type in question.\n\n        In Python 3, this is defined as an embedded PyVarObject type thus:\n           PyVarObject ob_base;\n        so that the "ob_size" field is located insize the "ob_base" field, and\n        the "ob_type" is most easily accessed by casting back to a (PyObject*).\n        '
        if self.is_null():
            raise NullPyObjectPtr(self)
        if name == 'ob_type':
            pyo_ptr = self._gdbval.cast(PyObjectPtr.get_gdb_type())
            return pyo_ptr.dereference()[name]
        if name == 'ob_size':
            pyo_ptr = self._gdbval.cast(PyVarObjectPtr.get_gdb_type())
            return pyo_ptr.dereference()[name]
        return self._gdbval.dereference()[name]

    def pyop_field(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a PyObjectPtr for the given PyObject* field within this PyObject,\n        coping with some python 2 versus python 3 differences.\n        '
        return PyObjectPtr.from_pyobject_ptr(self.field(name))

    def write_field_repr(self, name, out, visited):
        if False:
            return 10
        '\n        Extract the PyObject* field named "name", and write its representation\n        to file-like object "out"\n        '
        field_obj = self.pyop_field(name)
        field_obj.write_repr(out, visited)

    def get_truncated_repr(self, maxlen):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a repr-like string for the data, but truncate it at "maxlen" bytes\n        (ending the object graph traversal as soon as you do)\n        '
        out = TruncatedStringIO(maxlen)
        try:
            self.write_repr(out, set())
        except StringTruncated:
            return out.getvalue() + '...(truncated)'
        return out.getvalue()

    def type(self):
        if False:
            return 10
        return PyTypeObjectPtr(self.field('ob_type'))

    def is_null(self):
        if False:
            print('Hello World!')
        return 0 == long(self._gdbval)

    def is_optimized_out(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Is the value of the underlying PyObject* visible to the debugger?\n\n        This can vary with the precise version of the compiler used to build\n        Python, and the precise version of gdb.\n\n        See e.g. https://bugzilla.redhat.com/show_bug.cgi?id=556975 with\n        PyEval_EvalFrameEx\'s "f"\n        '
        return self._gdbval.is_optimized_out

    def safe_tp_name(self):
        if False:
            return 10
        try:
            ob_type = self.type()
            tp_name = ob_type.field('tp_name')
            return tp_name.string()
        except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
            return 'unknown'

    def proxyval(self, visited):
        if False:
            i = 10
            return i + 15
        '\n        Scrape a value from the inferior process, and try to represent it\n        within the gdb process, whilst (hopefully) avoiding crashes when\n        the remote data is corrupt.\n\n        Derived classes will override this.\n\n        For example, a PyIntObject* with ob_ival 42 in the inferior process\n        should result in an int(42) in this process.\n\n        visited: a set of all gdb.Value pyobject pointers already visited\n        whilst generating this value (to guard against infinite recursion when\n        visiting object graphs with loops).  Analogous to Py_ReprEnter and\n        Py_ReprLeave\n        '

        class FakeRepr(object):
            """
            Class representing a non-descript PyObject* value in the inferior
            process for when we don't have a custom scraper, intended to have
            a sane repr().
            """

            def __init__(self, tp_name, address):
                if False:
                    print('Hello World!')
                self.tp_name = tp_name
                self.address = address

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                if self.address == 0:
                    return '0x0'
                return '<%s at remote 0x%x>' % (self.tp_name, self.address)
        return FakeRepr(self.safe_tp_name(), long(self._gdbval))

    def write_repr(self, out, visited):
        if False:
            while True:
                i = 10
        '\n        Write a string representation of the value scraped from the inferior\n        process to "out", a file-like object.\n        '
        return out.write(repr(self.proxyval(visited)))

    @classmethod
    def subclass_from_type(cls, t):
        if False:
            i = 10
            return i + 15
        '\n        Given a PyTypeObjectPtr instance wrapping a gdb.Value that\'s a\n        (PyTypeObject*), determine the corresponding subclass of PyObjectPtr\n        to use\n\n        Ideally, we would look up the symbols for the global types, but that\n        isn\'t working yet:\n          (gdb) python print gdb.lookup_symbol(\'PyList_Type\')[0].value\n          Traceback (most recent call last):\n            File "<string>", line 1, in <module>\n          NotImplementedError: Symbol type not yet supported in Python scripts.\n          Error while executing Python code.\n\n        For now, we use tp_flags, after doing some string comparisons on the\n        tp_name for some special-cases that don\'t seem to be visible through\n        flags\n        '
        try:
            tp_name = t.field('tp_name').string()
            tp_flags = int(t.field('tp_flags'))
        except (RuntimeError, UnicodeDecodeError):
            return cls
        name_map = {'bool': PyBoolObjectPtr, 'classobj': PyClassObjectPtr, 'NoneType': PyNoneStructPtr, 'frame': PyFrameObjectPtr, 'set': PySetObjectPtr, 'frozenset': PySetObjectPtr, 'builtin_function_or_method': PyCFunctionObjectPtr, 'method-wrapper': wrapperobject}
        if tp_name in name_map:
            return name_map[tp_name]
        if tp_flags & Py_TPFLAGS_HEAPTYPE:
            return HeapTypeObjectPtr
        if tp_flags & Py_TPFLAGS_LONG_SUBCLASS:
            return PyLongObjectPtr
        if tp_flags & Py_TPFLAGS_LIST_SUBCLASS:
            return PyListObjectPtr
        if tp_flags & Py_TPFLAGS_TUPLE_SUBCLASS:
            return PyTupleObjectPtr
        if tp_flags & Py_TPFLAGS_BYTES_SUBCLASS:
            return PyBytesObjectPtr
        if tp_flags & Py_TPFLAGS_UNICODE_SUBCLASS:
            return PyUnicodeObjectPtr
        if tp_flags & Py_TPFLAGS_DICT_SUBCLASS:
            return PyDictObjectPtr
        if tp_flags & Py_TPFLAGS_BASE_EXC_SUBCLASS:
            return PyBaseExceptionObjectPtr
        return cls

    @classmethod
    def from_pyobject_ptr(cls, gdbval):
        if False:
            return 10
        '\n        Try to locate the appropriate derived class dynamically, and cast\n        the pointer accordingly.\n        '
        try:
            p = PyObjectPtr(gdbval)
            cls = cls.subclass_from_type(p.type())
            return cls(gdbval, cast_to=cls.get_gdb_type())
        except RuntimeError:
            pass
        return cls(gdbval)

    @classmethod
    def get_gdb_type(cls):
        if False:
            while True:
                i = 10
        return gdb.lookup_type(cls._typename).pointer()

    def as_address(self):
        if False:
            for i in range(10):
                print('nop')
        return long(self._gdbval)

class PyVarObjectPtr(PyObjectPtr):
    _typename = 'PyVarObject'

class ProxyAlreadyVisited(object):
    """
    Placeholder proxy to use when protecting against infinite recursion due to
    loops in the object graph.

    Analogous to the values emitted by the users of Py_ReprEnter and Py_ReprLeave
    """

    def __init__(self, rep):
        if False:
            for i in range(10):
                print('nop')
        self._rep = rep

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._rep

def _write_instance_repr(out, visited, name, pyop_attrdict, address):
    if False:
        return 10
    'Shared code for use by all classes:\n    write a representation to file-like object "out"'
    out.write('<')
    out.write(name)
    if isinstance(pyop_attrdict, PyDictObjectPtr):
        out.write('(')
        first = True
        for (pyop_arg, pyop_val) in pyop_attrdict.iteritems():
            if not first:
                out.write(', ')
            first = False
            out.write(pyop_arg.proxyval(visited))
            out.write('=')
            pyop_val.write_repr(out, visited)
        out.write(')')
    out.write(' at remote 0x%x>' % address)

class InstanceProxy(object):

    def __init__(self, cl_name, attrdict, address):
        if False:
            while True:
                i = 10
        self.cl_name = cl_name
        self.attrdict = attrdict
        self.address = address

    def __repr__(self):
        if False:
            print('Hello World!')
        if isinstance(self.attrdict, dict):
            kwargs = ', '.join(['%s=%r' % (arg, val) for (arg, val) in self.attrdict.iteritems()])
            return '<%s(%s) at remote 0x%x>' % (self.cl_name, kwargs, self.address)
        else:
            return '<%s at remote 0x%x>' % (self.cl_name, self.address)

def _PyObject_VAR_SIZE(typeobj, nitems):
    if False:
        print('Hello World!')
    if _PyObject_VAR_SIZE._type_size_t is None:
        _PyObject_VAR_SIZE._type_size_t = gdb.lookup_type('size_t')
    return (typeobj.field('tp_basicsize') + nitems * typeobj.field('tp_itemsize') + (_sizeof_void_p() - 1) & ~(_sizeof_void_p() - 1)).cast(_PyObject_VAR_SIZE._type_size_t)
_PyObject_VAR_SIZE._type_size_t = None

class HeapTypeObjectPtr(PyObjectPtr):
    _typename = 'PyObject'

    def get_attr_dict(self):
        if False:
            while True:
                i = 10
        "\n        Get the PyDictObject ptr representing the attribute dictionary\n        (or None if there's a problem)\n        "
        try:
            typeobj = self.type()
            dictoffset = int_from_int(typeobj.field('tp_dictoffset'))
            if dictoffset != 0:
                if dictoffset < 0:
                    type_PyVarObject_ptr = gdb.lookup_type('PyVarObject').pointer()
                    tsize = int_from_int(self._gdbval.cast(type_PyVarObject_ptr)['ob_size'])
                    if tsize < 0:
                        tsize = -tsize
                    size = _PyObject_VAR_SIZE(typeobj, tsize)
                    dictoffset += size
                    assert dictoffset > 0
                    assert dictoffset % _sizeof_void_p() == 0
                dictptr = self._gdbval.cast(_type_char_ptr()) + dictoffset
                PyObjectPtrPtr = PyObjectPtr.get_gdb_type().pointer()
                dictptr = dictptr.cast(PyObjectPtrPtr)
                return PyObjectPtr.from_pyobject_ptr(dictptr.dereference())
        except RuntimeError:
            pass
        return None

    def proxyval(self, visited):
        if False:
            i = 10
            return i + 15
        '\n        Support for classes.\n\n        Currently we just locate the dictionary using a transliteration to\n        python of _PyObject_GetDictPtr, ignoring descriptors\n        '
        if self.as_address() in visited:
            return ProxyAlreadyVisited('<...>')
        visited.add(self.as_address())
        pyop_attr_dict = self.get_attr_dict()
        if pyop_attr_dict:
            attr_dict = pyop_attr_dict.proxyval(visited)
        else:
            attr_dict = {}
        tp_name = self.safe_tp_name()
        return InstanceProxy(tp_name, attr_dict, long(self._gdbval))

    def write_repr(self, out, visited):
        if False:
            for i in range(10):
                print('nop')
        if self.as_address() in visited:
            out.write('<...>')
            return
        visited.add(self.as_address())
        pyop_attrdict = self.get_attr_dict()
        _write_instance_repr(out, visited, self.safe_tp_name(), pyop_attrdict, self.as_address())

class ProxyException(Exception):

    def __init__(self, tp_name, args):
        if False:
            print('Hello World!')
        self.tp_name = tp_name
        self.args = args

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s%r' % (self.tp_name, self.args)

class PyBaseExceptionObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyBaseExceptionObject* i.e. an exception
    within the process being debugged.
    """
    _typename = 'PyBaseExceptionObject'

    def proxyval(self, visited):
        if False:
            return 10
        if self.as_address() in visited:
            return ProxyAlreadyVisited('(...)')
        visited.add(self.as_address())
        arg_proxy = self.pyop_field('args').proxyval(visited)
        return ProxyException(self.safe_tp_name(), arg_proxy)

    def write_repr(self, out, visited):
        if False:
            while True:
                i = 10
        if self.as_address() in visited:
            out.write('(...)')
            return
        visited.add(self.as_address())
        out.write(self.safe_tp_name())
        self.write_field_repr('args', out, visited)

class PyClassObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyClassObject* i.e. a <classobj>
    instance within the process being debugged.
    """
    _typename = 'PyClassObject'

class BuiltInFunctionProxy(object):

    def __init__(self, ml_name):
        if False:
            i = 10
            return i + 15
        self.ml_name = ml_name

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<built-in function %s>' % self.ml_name

class BuiltInMethodProxy(object):

    def __init__(self, ml_name, pyop_m_self):
        if False:
            i = 10
            return i + 15
        self.ml_name = ml_name
        self.pyop_m_self = pyop_m_self

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<built-in method %s of %s object at remote 0x%x>' % (self.ml_name, self.pyop_m_self.safe_tp_name(), self.pyop_m_self.as_address())

class PyCFunctionObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyCFunctionObject*
    (see Include/methodobject.h and Objects/methodobject.c)
    """
    _typename = 'PyCFunctionObject'

    def proxyval(self, visited):
        if False:
            i = 10
            return i + 15
        m_ml = self.field('m_ml')
        try:
            ml_name = m_ml['ml_name'].string()
        except UnicodeDecodeError:
            ml_name = '<ml_name:UnicodeDecodeError>'
        pyop_m_self = self.pyop_field('m_self')
        if pyop_m_self.is_null():
            return BuiltInFunctionProxy(ml_name)
        else:
            return BuiltInMethodProxy(ml_name, pyop_m_self)

class PyCodeObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyCodeObject* i.e. a <code> instance
    within the process being debugged.
    """
    _typename = 'PyCodeObject'

    def addr2line(self, addrq):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the line number for a given bytecode offset\n\n        Analogous to PyCode_Addr2Line; translated from pseudocode in\n        Objects/lnotab_notes.txt\n        '
        co_lnotab = self.pyop_field('co_lnotab').proxyval(set())
        lineno = int_from_int(self.field('co_firstlineno'))
        addr = 0
        for (addr_incr, line_incr) in zip(co_lnotab[::2], co_lnotab[1::2]):
            addr += ord(addr_incr)
            if addr > addrq:
                return lineno
            lineno += ord(line_incr)
        return lineno

class PyDictObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyDictObject* i.e. a dict instance
    within the process being debugged.
    """
    _typename = 'PyDictObject'

    def iteritems(self):
        if False:
            return 10
        '\n        Yields a sequence of (PyObjectPtr key, PyObjectPtr value) pairs,\n        analogous to dict.iteritems()\n        '
        keys = self.field('ma_keys')
        values = self.field('ma_values')
        (entries, nentries) = self._get_entries(keys)
        for i in safe_range(nentries):
            ep = entries[i]
            if long(values):
                pyop_value = PyObjectPtr.from_pyobject_ptr(values[i])
            else:
                pyop_value = PyObjectPtr.from_pyobject_ptr(ep['me_value'])
            if not pyop_value.is_null():
                pyop_key = PyObjectPtr.from_pyobject_ptr(ep['me_key'])
                yield (pyop_key, pyop_value)

    def proxyval(self, visited):
        if False:
            for i in range(10):
                print('nop')
        if self.as_address() in visited:
            return ProxyAlreadyVisited('{...}')
        visited.add(self.as_address())
        result = {}
        for (pyop_key, pyop_value) in self.iteritems():
            proxy_key = pyop_key.proxyval(visited)
            proxy_value = pyop_value.proxyval(visited)
            result[proxy_key] = proxy_value
        return result

    def write_repr(self, out, visited):
        if False:
            print('Hello World!')
        if self.as_address() in visited:
            out.write('{...}')
            return
        visited.add(self.as_address())
        out.write('{')
        first = True
        for (pyop_key, pyop_value) in self.iteritems():
            if not first:
                out.write(', ')
            first = False
            pyop_key.write_repr(out, visited)
            out.write(': ')
            pyop_value.write_repr(out, visited)
        out.write('}')

    def _get_entries(self, keys):
        if False:
            while True:
                i = 10
        dk_nentries = int(keys['dk_nentries'])
        dk_size = int(keys['dk_size'])
        try:
            return (keys['dk_entries'], dk_size)
        except RuntimeError:
            pass
        if dk_size <= 255:
            offset = dk_size
        elif dk_size <= 65535:
            offset = 2 * dk_size
        elif dk_size <= 4294967295:
            offset = 4 * dk_size
        else:
            offset = 8 * dk_size
        ent_addr = keys['dk_indices'].address
        ent_addr = ent_addr.cast(_type_unsigned_char_ptr()) + offset
        ent_ptr_t = gdb.lookup_type('PyDictKeyEntry').pointer()
        ent_addr = ent_addr.cast(ent_ptr_t)
        return (ent_addr, dk_nentries)

class PyListObjectPtr(PyObjectPtr):
    _typename = 'PyListObject'

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        field_ob_item = self.field('ob_item')
        return field_ob_item[i]

    def proxyval(self, visited):
        if False:
            for i in range(10):
                print('nop')
        if self.as_address() in visited:
            return ProxyAlreadyVisited('[...]')
        visited.add(self.as_address())
        result = [PyObjectPtr.from_pyobject_ptr(self[i]).proxyval(visited) for i in safe_range(int_from_int(self.field('ob_size')))]
        return result

    def write_repr(self, out, visited):
        if False:
            print('Hello World!')
        if self.as_address() in visited:
            out.write('[...]')
            return
        visited.add(self.as_address())
        out.write('[')
        for i in safe_range(int_from_int(self.field('ob_size'))):
            if i > 0:
                out.write(', ')
            element = PyObjectPtr.from_pyobject_ptr(self[i])
            element.write_repr(out, visited)
        out.write(']')

class PyLongObjectPtr(PyObjectPtr):
    _typename = 'PyLongObject'

    def proxyval(self, visited):
        if False:
            while True:
                i = 10
        "\n        Python's Include/longobjrep.h has this declaration:\n           struct _longobject {\n               PyObject_VAR_HEAD\n               digit ob_digit[1];\n           };\n\n        with this description:\n            The absolute value of a number is equal to\n                 SUM(for i=0 through abs(ob_size)-1) ob_digit[i] * 2**(SHIFT*i)\n            Negative numbers are represented with ob_size < 0;\n            zero is represented by ob_size == 0.\n\n        where SHIFT can be either:\n            #define PyLong_SHIFT        30\n            #define PyLong_SHIFT        15\n        "
        ob_size = long(self.field('ob_size'))
        if ob_size == 0:
            return 0
        ob_digit = self.field('ob_digit')
        if gdb.lookup_type('digit').sizeof == 2:
            SHIFT = 15
        else:
            SHIFT = 30
        digits = [long(ob_digit[i]) * 2 ** (SHIFT * i) for i in safe_range(abs(ob_size))]
        result = sum(digits)
        if ob_size < 0:
            result = -result
        return result

    def write_repr(self, out, visited):
        if False:
            print('Hello World!')
        proxy = self.proxyval(visited)
        out.write('%s' % proxy)

class PyBoolObjectPtr(PyLongObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyBoolObject* i.e. one of the two
    <bool> instances (Py_True/Py_False) within the process being debugged.
    """

    def proxyval(self, visited):
        if False:
            print('Hello World!')
        if PyLongObjectPtr.proxyval(self, visited):
            return True
        else:
            return False

class PyNoneStructPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyObject* pointing to the
    singleton (we hope) _Py_NoneStruct with ob_type PyNone_Type
    """
    _typename = 'PyObject'

    def proxyval(self, visited):
        if False:
            while True:
                i = 10
        return None

class PyFrameObjectPtr(PyObjectPtr):
    _typename = 'PyFrameObject'

    def __init__(self, gdbval, cast_to=None):
        if False:
            i = 10
            return i + 15
        PyObjectPtr.__init__(self, gdbval, cast_to)
        if not self.is_optimized_out():
            self.co = PyCodeObjectPtr.from_pyobject_ptr(self.field('f_code'))
            self.co_name = self.co.pyop_field('co_name')
            self.co_filename = self.co.pyop_field('co_filename')
            self.f_lineno = int_from_int(self.field('f_lineno'))
            self.f_lasti = int_from_int(self.field('f_lasti'))
            self.co_nlocals = int_from_int(self.co.field('co_nlocals'))
            self.co_varnames = PyTupleObjectPtr.from_pyobject_ptr(self.co.field('co_varnames'))

    def iter_locals(self):
        if False:
            return 10
        '\n        Yield a sequence of (name,value) pairs of PyObjectPtr instances, for\n        the local variables of this frame\n        '
        if self.is_optimized_out():
            return
        f_localsplus = self.field('f_localsplus')
        for i in safe_range(self.co_nlocals):
            pyop_value = PyObjectPtr.from_pyobject_ptr(f_localsplus[i])
            if not pyop_value.is_null():
                pyop_name = PyObjectPtr.from_pyobject_ptr(self.co_varnames[i])
                yield (pyop_name, pyop_value)

    def iter_globals(self):
        if False:
            print('Hello World!')
        '\n        Yield a sequence of (name,value) pairs of PyObjectPtr instances, for\n        the global variables of this frame\n        '
        if self.is_optimized_out():
            return ()
        pyop_globals = self.pyop_field('f_globals')
        return pyop_globals.iteritems()

    def iter_builtins(self):
        if False:
            return 10
        '\n        Yield a sequence of (name,value) pairs of PyObjectPtr instances, for\n        the builtin variables\n        '
        if self.is_optimized_out():
            return ()
        pyop_builtins = self.pyop_field('f_builtins')
        return pyop_builtins.iteritems()

    def get_var_by_name(self, name):
        if False:
            print('Hello World!')
        "\n        Look for the named local variable, returning a (PyObjectPtr, scope) pair\n        where scope is a string 'local', 'global', 'builtin'\n\n        If not found, return (None, None)\n        "
        for (pyop_name, pyop_value) in self.iter_locals():
            if name == pyop_name.proxyval(set()):
                return (pyop_value, 'local')
        for (pyop_name, pyop_value) in self.iter_globals():
            if name == pyop_name.proxyval(set()):
                return (pyop_value, 'global')
        for (pyop_name, pyop_value) in self.iter_builtins():
            if name == pyop_name.proxyval(set()):
                return (pyop_value, 'builtin')
        return (None, None)

    def filename(self):
        if False:
            while True:
                i = 10
        'Get the path of the current Python source file, as a string'
        if self.is_optimized_out():
            return FRAME_INFO_OPTIMIZED_OUT
        return self.co_filename.proxyval(set())

    def current_line_num(self):
        if False:
            while True:
                i = 10
        'Get current line number as an integer (1-based)\n\n        Translated from PyFrame_GetLineNumber and PyCode_Addr2Line\n\n        See Objects/lnotab_notes.txt\n        '
        if self.is_optimized_out():
            return None
        f_trace = self.field('f_trace')
        if long(f_trace) != 0:
            return self.f_lineno
        try:
            return self.co.addr2line(self.f_lasti)
        except Exception:
            return None

    def current_line(self):
        if False:
            print('Hello World!')
        'Get the text of the current source line as a string, with a trailing\n        newline character'
        if self.is_optimized_out():
            return FRAME_INFO_OPTIMIZED_OUT
        lineno = self.current_line_num()
        if lineno is None:
            return '(failed to get frame line number)'
        filename = self.filename()
        try:
            with open(os_fsencode(filename), 'r') as fp:
                lines = fp.readlines()
        except IOError:
            return None
        try:
            return lines[lineno - 1]
        except IndexError:
            return None

    def write_repr(self, out, visited):
        if False:
            i = 10
            return i + 15
        if self.is_optimized_out():
            out.write(FRAME_INFO_OPTIMIZED_OUT)
            return
        lineno = self.current_line_num()
        lineno = str(lineno) if lineno is not None else '?'
        out.write('Frame 0x%x, for file %s, line %s, in %s (' % (self.as_address(), self.co_filename.proxyval(visited), lineno, self.co_name.proxyval(visited)))
        first = True
        for (pyop_name, pyop_value) in self.iter_locals():
            if not first:
                out.write(', ')
            first = False
            out.write(pyop_name.proxyval(visited))
            out.write('=')
            pyop_value.write_repr(out, visited)
        out.write(')')

    def print_traceback(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_optimized_out():
            sys.stdout.write('  %s\n' % FRAME_INFO_OPTIMIZED_OUT)
            return
        visited = set()
        lineno = self.current_line_num()
        lineno = str(lineno) if lineno is not None else '?'
        sys.stdout.write('  File "%s", line %s, in %s\n' % (self.co_filename.proxyval(visited), lineno, self.co_name.proxyval(visited)))

class PySetObjectPtr(PyObjectPtr):
    _typename = 'PySetObject'

    @classmethod
    def _dummy_key(self):
        if False:
            print('Hello World!')
        return gdb.lookup_global_symbol('_PySet_Dummy').value()

    def __iter__(self):
        if False:
            while True:
                i = 10
        dummy_ptr = self._dummy_key()
        table = self.field('table')
        for i in safe_range(self.field('mask') + 1):
            setentry = table[i]
            key = setentry['key']
            if key != 0 and key != dummy_ptr:
                yield PyObjectPtr.from_pyobject_ptr(key)

    def proxyval(self, visited):
        if False:
            for i in range(10):
                print('nop')
        if self.as_address() in visited:
            return ProxyAlreadyVisited('%s(...)' % self.safe_tp_name())
        visited.add(self.as_address())
        members = (key.proxyval(visited) for key in self)
        if self.safe_tp_name() == 'frozenset':
            return frozenset(members)
        else:
            return set(members)

    def write_repr(self, out, visited):
        if False:
            return 10
        tp_name = self.safe_tp_name()
        if self.as_address() in visited:
            out.write('(...)')
            return
        visited.add(self.as_address())
        if not self.field('used'):
            out.write(tp_name)
            out.write('()')
            return
        if tp_name != 'set':
            out.write(tp_name)
            out.write('(')
        out.write('{')
        first = True
        for key in self:
            if not first:
                out.write(', ')
            first = False
            key.write_repr(out, visited)
        out.write('}')
        if tp_name != 'set':
            out.write(')')

class PyBytesObjectPtr(PyObjectPtr):
    _typename = 'PyBytesObject'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        field_ob_size = self.field('ob_size')
        field_ob_sval = self.field('ob_sval')
        char_ptr = field_ob_sval.address.cast(_type_unsigned_char_ptr())
        return ''.join([chr(char_ptr[i]) for i in safe_range(field_ob_size)])

    def proxyval(self, visited):
        if False:
            while True:
                i = 10
        return str(self)

    def write_repr(self, out, visited):
        if False:
            for i in range(10):
                print('nop')
        proxy = self.proxyval(visited)
        quote = "'"
        if "'" in proxy and (not '"' in proxy):
            quote = '"'
        out.write('b')
        out.write(quote)
        for byte in proxy:
            if byte == quote or byte == '\\':
                out.write('\\')
                out.write(byte)
            elif byte == '\t':
                out.write('\\t')
            elif byte == '\n':
                out.write('\\n')
            elif byte == '\r':
                out.write('\\r')
            elif byte < ' ' or ord(byte) >= 127:
                out.write('\\x')
                out.write(hexdigits[(ord(byte) & 240) >> 4])
                out.write(hexdigits[ord(byte) & 15])
            else:
                out.write(byte)
        out.write(quote)

class PyTupleObjectPtr(PyObjectPtr):
    _typename = 'PyTupleObject'

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        field_ob_item = self.field('ob_item')
        return field_ob_item[i]

    def proxyval(self, visited):
        if False:
            for i in range(10):
                print('nop')
        if self.as_address() in visited:
            return ProxyAlreadyVisited('(...)')
        visited.add(self.as_address())
        result = tuple((PyObjectPtr.from_pyobject_ptr(self[i]).proxyval(visited) for i in safe_range(int_from_int(self.field('ob_size')))))
        return result

    def write_repr(self, out, visited):
        if False:
            return 10
        if self.as_address() in visited:
            out.write('(...)')
            return
        visited.add(self.as_address())
        out.write('(')
        for i in safe_range(int_from_int(self.field('ob_size'))):
            if i > 0:
                out.write(', ')
            element = PyObjectPtr.from_pyobject_ptr(self[i])
            element.write_repr(out, visited)
        if self.field('ob_size') == 1:
            out.write(',)')
        else:
            out.write(')')

class PyTypeObjectPtr(PyObjectPtr):
    _typename = 'PyTypeObject'

def _unichr_is_printable(char):
    if False:
        return 10
    if char == u' ':
        return True
    import unicodedata
    return unicodedata.category(char) not in ('C', 'Z')
if sys.maxunicode >= 65536:
    _unichr = unichr
else:

    def _unichr(x):
        if False:
            while True:
                i = 10
        if x < 65536:
            return unichr(x)
        x -= 65536
        ch1 = 55296 | x >> 10
        ch2 = 56320 | x & 1023
        return unichr(ch1) + unichr(ch2)

class PyUnicodeObjectPtr(PyObjectPtr):
    _typename = 'PyUnicodeObject'

    def char_width(self):
        if False:
            while True:
                i = 10
        _type_Py_UNICODE = gdb.lookup_type('Py_UNICODE')
        return _type_Py_UNICODE.sizeof

    def proxyval(self, visited):
        if False:
            while True:
                i = 10
        global _is_pep393
        if _is_pep393 is None:
            fields = gdb.lookup_type('PyUnicodeObject').fields()
            _is_pep393 = 'data' in [f.name for f in fields]
        if _is_pep393:
            may_have_surrogates = False
            compact = self.field('_base')
            ascii = compact['_base']
            state = ascii['state']
            is_compact_ascii = int(state['ascii']) and int(state['compact'])
            if not int(state['ready']):
                field_length = long(compact['wstr_length'])
                may_have_surrogates = True
                field_str = ascii['wstr']
            else:
                field_length = long(ascii['length'])
                if is_compact_ascii:
                    field_str = ascii.address + 1
                elif int(state['compact']):
                    field_str = compact.address + 1
                else:
                    field_str = self.field('data')['any']
                repr_kind = int(state['kind'])
                if repr_kind == 1:
                    field_str = field_str.cast(_type_unsigned_char_ptr())
                elif repr_kind == 2:
                    field_str = field_str.cast(_type_unsigned_short_ptr())
                elif repr_kind == 4:
                    field_str = field_str.cast(_type_unsigned_int_ptr())
        else:
            field_length = long(self.field('length'))
            field_str = self.field('str')
            may_have_surrogates = self.char_width() == 2
        if not may_have_surrogates:
            Py_UNICODEs = [int(field_str[i]) for i in safe_range(field_length)]
        else:
            Py_UNICODEs = []
            i = 0
            limit = safety_limit(field_length)
            while i < limit:
                ucs = int(field_str[i])
                i += 1
                if ucs < 55296 or ucs >= 56320 or i == field_length:
                    Py_UNICODEs.append(ucs)
                    continue
                ucs2 = int(field_str[i])
                if ucs2 < 56320 or ucs2 > 57343:
                    continue
                code = (ucs & 1023) << 10
                code |= ucs2 & 1023
                code += 65536
                Py_UNICODEs.append(code)
                i += 1
        result = u''.join([_unichr(ucs) if ucs <= 1114111 else '�' for ucs in Py_UNICODEs])
        return result

    def write_repr(self, out, visited):
        if False:
            print('Hello World!')
        proxy = self.proxyval(visited)
        if "'" in proxy and '"' not in proxy:
            quote = '"'
        else:
            quote = "'"
        out.write(quote)
        i = 0
        while i < len(proxy):
            ch = proxy[i]
            i += 1
            if ch == quote or ch == '\\':
                out.write('\\')
                out.write(ch)
            elif ch == '\t':
                out.write('\\t')
            elif ch == '\n':
                out.write('\\n')
            elif ch == '\r':
                out.write('\\r')
            elif ch < ' ' or ch == 127:
                out.write('\\x')
                out.write(hexdigits[ord(ch) >> 4 & 15])
                out.write(hexdigits[ord(ch) & 15])
            elif ord(ch) < 127:
                out.write(ch)
            else:
                ucs = ch
                ch2 = None
                if sys.maxunicode < 65536:
                    if i < len(proxy) and 55296 <= ord(ch) < 56320 and (56320 <= ord(proxy[i]) <= 57343):
                        ch2 = proxy[i]
                        ucs = ch + ch2
                        i += 1
                printable = _unichr_is_printable(ucs)
                if printable:
                    try:
                        ucs.encode(ENCODING)
                    except UnicodeEncodeError:
                        printable = False
                if not printable:
                    if ch2 is not None:
                        code = (ord(ch) & 1023) << 10
                        code |= ord(ch2) & 1023
                        code += 65536
                    else:
                        code = ord(ucs)
                    if code <= 255:
                        out.write('\\x')
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                    elif code >= 65536:
                        out.write('\\U')
                        out.write(hexdigits[code >> 28 & 15])
                        out.write(hexdigits[code >> 24 & 15])
                        out.write(hexdigits[code >> 20 & 15])
                        out.write(hexdigits[code >> 16 & 15])
                        out.write(hexdigits[code >> 12 & 15])
                        out.write(hexdigits[code >> 8 & 15])
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                    else:
                        out.write('\\u')
                        out.write(hexdigits[code >> 12 & 15])
                        out.write(hexdigits[code >> 8 & 15])
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                else:
                    out.write(ch)
                    if ch2 is not None:
                        out.write(ch2)
        out.write(quote)

class wrapperobject(PyObjectPtr):
    _typename = 'wrapperobject'

    def safe_name(self):
        if False:
            while True:
                i = 10
        try:
            name = self.field('descr')['d_base']['name'].string()
            return repr(name)
        except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
            return '<unknown name>'

    def safe_tp_name(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.field('self')['ob_type']['tp_name'].string()
        except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
            return '<unknown tp_name>'

    def safe_self_addresss(self):
        if False:
            return 10
        try:
            address = long(self.field('self'))
            return '%#x' % address
        except (NullPyObjectPtr, RuntimeError):
            return '<failed to get self address>'

    def proxyval(self, visited):
        if False:
            print('Hello World!')
        name = self.safe_name()
        tp_name = self.safe_tp_name()
        self_address = self.safe_self_addresss()
        return '<method-wrapper %s of %s object at %s>' % (name, tp_name, self_address)

    def write_repr(self, out, visited):
        if False:
            print('Hello World!')
        proxy = self.proxyval(visited)
        out.write(proxy)

def int_from_int(gdbval):
    if False:
        print('Hello World!')
    return int(gdbval)

def stringify(val):
    if False:
        print('Hello World!')
    if True:
        return repr(val)
    else:
        from pprint import pformat
        return pformat(val)

class PyObjectPtrPrinter:
    """Prints a (PyObject*)"""

    def __init__(self, gdbval):
        if False:
            print('Hello World!')
        self.gdbval = gdbval

    def to_string(self):
        if False:
            for i in range(10):
                print('nop')
        pyop = PyObjectPtr.from_pyobject_ptr(self.gdbval)
        if True:
            return pyop.get_truncated_repr(MAX_OUTPUT_LEN)
        else:
            proxyval = pyop.proxyval(set())
            return stringify(proxyval)

def pretty_printer_lookup(gdbval):
    if False:
        print('Hello World!')
    type = gdbval.type.unqualified()
    if type.code != gdb.TYPE_CODE_PTR:
        return None
    type = type.target().unqualified()
    t = str(type)
    if t in ('PyObject', 'PyFrameObject', 'PyUnicodeObject', 'wrapperobject'):
        return PyObjectPtrPrinter(gdbval)
'\nDuring development, I\'ve been manually invoking the code in this way:\n(gdb) python\n\nimport sys\nsys.path.append(\'/home/david/coding/python-gdb\')\nimport libpython\nend\n\nthen reloading it after each edit like this:\n(gdb) python reload(libpython)\n\nThe following code should ensure that the prettyprinter is registered\nif the code is autoloaded by gdb when visiting libpython.so, provided\nthat this python file is installed to the same path as the library (or its\n.debug file) plus a "-gdb.py" suffix, e.g:\n  /usr/lib/libpython2.6.so.1.0-gdb.py\n  /usr/lib/debug/usr/lib/libpython2.6.so.1.0.debug-gdb.py\n'

def register(obj):
    if False:
        return 10
    if obj is None:
        obj = gdb
    obj.pretty_printers.append(pretty_printer_lookup)
register(gdb.current_objfile())

class Frame(object):
    """
    Wrapper for gdb.Frame, adding various methods
    """

    def __init__(self, gdbframe):
        if False:
            return 10
        self._gdbframe = gdbframe

    def older(self):
        if False:
            print('Hello World!')
        older = self._gdbframe.older()
        if older:
            return Frame(older)
        else:
            return None

    def newer(self):
        if False:
            i = 10
            return i + 15
        newer = self._gdbframe.newer()
        if newer:
            return Frame(newer)
        else:
            return None

    def select(self):
        if False:
            while True:
                i = 10
        'If supported, select this frame and return True; return False if unsupported\n\n        Not all builds have a gdb.Frame.select method; seems to be present on Fedora 12\n        onwards, but absent on Ubuntu buildbot'
        if not hasattr(self._gdbframe, 'select'):
            print('Unable to select frame: this build of gdb does not expose a gdb.Frame.select method')
            return False
        self._gdbframe.select()
        return True

    def get_index(self):
        if False:
            while True:
                i = 10
        'Calculate index of frame, starting at 0 for the newest frame within\n        this thread'
        index = 0
        iter_frame = self
        while iter_frame.newer():
            index += 1
            iter_frame = iter_frame.newer()
        return index

    def is_python_frame(self):
        if False:
            for i in range(10):
                print('nop')
        'Is this a _PyEval_EvalFrameDefault frame, or some other important\n        frame? (see is_other_python_frame for what "important" means in this\n        context)'
        if self.is_evalframe():
            return True
        if self.is_other_python_frame():
            return True
        return False

    def is_evalframe(self):
        if False:
            for i in range(10):
                print('nop')
        'Is this a _PyEval_EvalFrameDefault frame?'
        if self._gdbframe.name() == EVALFRAME:
            '\n            I believe we also need to filter on the inline\n            struct frame_id.inline_depth, only regarding frames with\n            an inline depth of 0 as actually being this function\n\n            So we reject those with type gdb.INLINE_FRAME\n            '
            if self._gdbframe.type() == gdb.NORMAL_FRAME:
                return True
        return False

    def is_other_python_frame(self):
        if False:
            while True:
                i = 10
        'Is this frame worth displaying in python backtraces?\n        Examples:\n          - waiting on the GIL\n          - garbage-collecting\n          - within a CFunction\n         If it is, return a descriptive string\n         For other frames, return False\n         '
        if self.is_waiting_for_gil():
            return 'Waiting for the GIL'
        if self.is_gc_collect():
            return 'Garbage-collecting'
        frame = self._gdbframe
        caller = frame.name()
        if not caller:
            return False
        if caller.startswith('cfunction_vectorcall_') or caller == 'cfunction_call':
            arg_name = 'func'
            try:
                func = frame.read_var(arg_name)
                return str(func)
            except ValueError:
                return 'PyCFunction invocation (unable to read %s: missing debuginfos?)' % arg_name
            except RuntimeError:
                return 'PyCFunction invocation (unable to read %s)' % arg_name
        if caller == 'wrapper_call':
            arg_name = 'wp'
            try:
                func = frame.read_var(arg_name)
                return str(func)
            except ValueError:
                return '<wrapper_call invocation (unable to read %s: missing debuginfos?)>' % arg_name
            except RuntimeError:
                return '<wrapper_call invocation (unable to read %s)>' % arg_name
        return False

    def is_waiting_for_gil(self):
        if False:
            for i in range(10):
                print('nop')
        'Is this frame waiting on the GIL?'
        name = self._gdbframe.name()
        if name:
            return name == 'take_gil'

    def is_gc_collect(self):
        if False:
            print('Hello World!')
        'Is this frame "collect" within the garbage-collector?'
        return self._gdbframe.name() == 'collect'

    def get_pyop(self):
        if False:
            i = 10
            return i + 15
        try:
            f = self._gdbframe.read_var('f')
            frame = PyFrameObjectPtr.from_pyobject_ptr(f)
            if not frame.is_optimized_out():
                return frame
            orig_frame = frame
            caller = self._gdbframe.older()
            if caller:
                f = caller.read_var('f')
                frame = PyFrameObjectPtr.from_pyobject_ptr(f)
                if not frame.is_optimized_out():
                    return frame
            return orig_frame
        except ValueError:
            return None

    @classmethod
    def get_selected_frame(cls):
        if False:
            for i in range(10):
                print('nop')
        _gdbframe = gdb.selected_frame()
        if _gdbframe:
            return Frame(_gdbframe)
        return None

    @classmethod
    def get_selected_python_frame(cls):
        if False:
            for i in range(10):
                print('nop')
        'Try to obtain the Frame for the python-related code in the selected\n        frame, or None'
        try:
            frame = cls.get_selected_frame()
        except gdb.error:
            return None
        while frame:
            if frame.is_python_frame():
                return frame
            frame = frame.older()
        return None

    @classmethod
    def get_selected_bytecode_frame(cls):
        if False:
            while True:
                i = 10
        'Try to obtain the Frame for the python bytecode interpreter in the\n        selected GDB frame, or None'
        frame = cls.get_selected_frame()
        while frame:
            if frame.is_evalframe():
                return frame
            frame = frame.older()
        return None

    def print_summary(self):
        if False:
            i = 10
            return i + 15
        if self.is_evalframe():
            pyop = self.get_pyop()
            if pyop:
                line = pyop.get_truncated_repr(MAX_OUTPUT_LEN)
                write_unicode(sys.stdout, '#%i %s\n' % (self.get_index(), line))
                if not pyop.is_optimized_out():
                    line = pyop.current_line()
                    if line is not None:
                        sys.stdout.write('    %s\n' % line.strip())
            else:
                sys.stdout.write('#%i (unable to read python frame information)\n' % self.get_index())
        else:
            info = self.is_other_python_frame()
            if info:
                sys.stdout.write('#%i %s\n' % (self.get_index(), info))
            else:
                sys.stdout.write('#%i\n' % self.get_index())

    def print_traceback(self):
        if False:
            while True:
                i = 10
        if self.is_evalframe():
            pyop = self.get_pyop()
            if pyop:
                pyop.print_traceback()
                if not pyop.is_optimized_out():
                    line = pyop.current_line()
                    if line is not None:
                        sys.stdout.write('    %s\n' % line.strip())
            else:
                sys.stdout.write('  (unable to read python frame information)\n')
        else:
            info = self.is_other_python_frame()
            if info:
                sys.stdout.write('  %s\n' % info)
            else:
                sys.stdout.write('  (not a python frame)\n')

class PyList(gdb.Command):
    """List the current Python source code, if any

    Use
       py-list START
    to list at a different line number within the python source.

    Use
       py-list START, END
    to list a specific range of lines within the python source.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        gdb.Command.__init__(self, 'py-list', gdb.COMMAND_FILES, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            i = 10
            return i + 15
        import re
        start = None
        end = None
        m = re.match('\\s*(\\d+)\\s*', args)
        if m:
            start = int(m.group(0))
            end = start + 10
        m = re.match('\\s*(\\d+)\\s*,\\s*(\\d+)\\s*', args)
        if m:
            (start, end) = map(int, m.groups())
        frame = Frame.get_selected_bytecode_frame()
        if not frame:
            print('Unable to locate gdb frame for python bytecode interpreter')
            return
        pyop = frame.get_pyop()
        if not pyop or pyop.is_optimized_out():
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        filename = pyop.filename()
        lineno = pyop.current_line_num()
        if lineno is None:
            print('Unable to read python frame line number')
            return
        if start is None:
            start = lineno - 5
            end = lineno + 5
        if start < 1:
            start = 1
        try:
            f = open(os_fsencode(filename), 'r')
        except IOError as err:
            sys.stdout.write('Unable to open %s: %s\n' % (filename, err))
            return
        with f:
            all_lines = f.readlines()
            for (i, line) in enumerate(all_lines[start - 1:end]):
                linestr = str(i + start)
                if i + start == lineno:
                    linestr = '>' + linestr
                sys.stdout.write('%4s    %s' % (linestr, line))
PyList()

def move_in_stack(move_up):
    if False:
        return 10
    'Move up or down the stack (for the py-up/py-down command)'
    frame = Frame.get_selected_python_frame()
    if not frame:
        print('Unable to locate python frame')
        return
    while frame:
        if move_up:
            iter_frame = frame.older()
        else:
            iter_frame = frame.newer()
        if not iter_frame:
            break
        if iter_frame.is_python_frame():
            if iter_frame.select():
                iter_frame.print_summary()
            return
        frame = iter_frame
    if move_up:
        print('Unable to find an older python frame')
    else:
        print('Unable to find a newer python frame')

class PyUp(gdb.Command):
    """Select and print the python stack frame that called this one (if any)"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        gdb.Command.__init__(self, 'py-up', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            while True:
                i = 10
        move_in_stack(move_up=True)

class PyDown(gdb.Command):
    """Select and print the python stack frame called by this one (if any)"""

    def __init__(self):
        if False:
            return 10
        gdb.Command.__init__(self, 'py-down', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            return 10
        move_in_stack(move_up=False)
if hasattr(gdb.Frame, 'select'):
    PyUp()
    PyDown()

class PyBacktraceFull(gdb.Command):
    """Display the current python frame and all the frames within its call stack (if any)"""

    def __init__(self):
        if False:
            while True:
                i = 10
        gdb.Command.__init__(self, 'py-bt-full', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            print('Hello World!')
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        while frame:
            if frame.is_python_frame():
                frame.print_summary()
            frame = frame.older()
PyBacktraceFull()

class PyBacktrace(gdb.Command):
    """Display the current python frame and all the frames within its call stack (if any)"""

    def __init__(self):
        if False:
            print('Hello World!')
        gdb.Command.__init__(self, 'py-bt', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            i = 10
            return i + 15
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        sys.stdout.write('Traceback (most recent call first):\n')
        while frame:
            if frame.is_python_frame():
                frame.print_traceback()
            frame = frame.older()
PyBacktrace()

class PyPrint(gdb.Command):
    """Look up the given python variable name, and print it"""

    def __init__(self):
        if False:
            while True:
                i = 10
        gdb.Command.__init__(self, 'py-print', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            print('Hello World!')
        name = str(args)
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        pyop_frame = frame.get_pyop()
        if not pyop_frame:
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        (pyop_var, scope) = pyop_frame.get_var_by_name(name)
        if pyop_var:
            print('%s %r = %s' % (scope, name, pyop_var.get_truncated_repr(MAX_OUTPUT_LEN)))
        else:
            print('%r not found' % name)
PyPrint()

class PyLocals(gdb.Command):
    """Look up the given python variable name, and print it"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        gdb.Command.__init__(self, 'py-locals', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        if False:
            print('Hello World!')
        name = str(args)
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        pyop_frame = frame.get_pyop()
        if not pyop_frame:
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        for (pyop_name, pyop_value) in pyop_frame.iter_locals():
            print('%s = %s' % (pyop_name.proxyval(set()), pyop_value.get_truncated_repr(MAX_OUTPUT_LEN)))
PyLocals()
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback

def dont_suppress_errors(function):
    if False:
        while True:
            i = 10
    '*sigh*, readline'

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            return function(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
    return wrapper

class PyGlobals(gdb.Command):
    """List all the globals in the currently select Python frame"""

    def __init__(self):
        if False:
            return 10
        gdb.Command.__init__(self, 'py-globals', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        if False:
            return 10
        name = str(args)
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        pyop_frame = frame.get_pyop()
        if not pyop_frame:
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        for (pyop_name, pyop_value) in pyop_frame.iter_locals():
            print('%s = %s' % (pyop_name.proxyval(set()), pyop_value.get_truncated_repr(MAX_OUTPUT_LEN)))

    def get_namespace(self, pyop_frame):
        if False:
            i = 10
            return i + 15
        return pyop_frame.iter_globals()
PyGlobals()

def is_evalframeex(frame):
    if False:
        while True:
            i = 10
    'Is this a PyEval_EvalFrameEx frame?'
    if frame._gdbframe.name() == 'PyEval_EvalFrameEx':
        '\n        I believe we also need to filter on the inline\n        struct frame_id.inline_depth, only regarding frames with\n        an inline depth of 0 as actually being this function\n\n        So we reject those with type gdb.INLINE_FRAME\n        '
        if frame._gdbframe.type() == gdb.NORMAL_FRAME:
            return True
    return False

class PyNameEquals(gdb.Function):

    def _get_pycurframe_attr(self, attr):
        if False:
            while True:
                i = 10
        frame = Frame(gdb.selected_frame())
        if is_evalframeex(frame):
            pyframe = frame.get_pyop()
            if pyframe is None:
                warnings.warn("Use a Python debug build, Python breakpoints won't work otherwise.")
                return None
            return getattr(pyframe, attr).proxyval(set())
        return None

    @dont_suppress_errors
    def invoke(self, funcname):
        if False:
            for i in range(10):
                print('nop')
        attr = self._get_pycurframe_attr('co_name')
        return attr is not None and attr == funcname.string()
PyNameEquals('pyname_equals')

class PyModEquals(PyNameEquals):

    @dont_suppress_errors
    def invoke(self, modname):
        if False:
            return 10
        attr = self._get_pycurframe_attr('co_filename')
        if attr is not None:
            (filename, ext) = os.path.splitext(os.path.basename(attr))
            return filename == modname.string()
        return False
PyModEquals('pymod_equals')

class PyBreak(gdb.Command):
    """
    Set a Python breakpoint. Examples:

    Break on any function or method named 'func' in module 'modname'

        py-break modname.func

    Break on any function or method named 'func'

        py-break func
    """

    @dont_suppress_errors
    def invoke(self, funcname, from_tty):
        if False:
            for i in range(10):
                print('nop')
        if '.' in funcname:
            (modname, dot, funcname) = funcname.rpartition('.')
            cond = '$pyname_equals("%s") && $pymod_equals("%s")' % (funcname, modname)
        else:
            cond = '$pyname_equals("%s")' % funcname
        gdb.execute('break PyEval_EvalFrameEx if ' + cond)
PyBreak('py-break', gdb.COMMAND_RUNNING, gdb.COMPLETE_NONE)

class _LoggingState(object):
    """
    State that helps to provide a reentrant gdb.execute() function.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        f = tempfile.NamedTemporaryFile('r+')
        self.file = f
        self.filename = f.name
        self.fd = f.fileno()
        _execute('set logging file %s' % self.filename)
        self.file_position_stack = []

    def __enter__(self):
        if False:
            while True:
                i = 10
        if not self.file_position_stack:
            _execute('set logging redirect on')
            _execute('set logging on')
            _execute('set pagination off')
        self.file_position_stack.append(os.fstat(self.fd).st_size)
        return self

    def getoutput(self):
        if False:
            while True:
                i = 10
        gdb.flush()
        self.file.seek(self.file_position_stack[-1])
        result = self.file.read()
        return result

    def __exit__(self, exc_type, exc_val, tb):
        if False:
            print('Hello World!')
        startpos = self.file_position_stack.pop()
        self.file.seek(startpos)
        self.file.truncate()
        if not self.file_position_stack:
            _execute('set logging off')
            _execute('set logging redirect off')
            _execute('set pagination on')

def execute(command, from_tty=False, to_string=False):
    if False:
        while True:
            i = 10
    "\n    Replace gdb.execute() with this function and have it accept a 'to_string'\n    argument (new in 7.2). Have it properly capture stderr also. Ensure\n    reentrancy.\n    "
    if to_string:
        with _logging_state as state:
            _execute(command, from_tty)
            return state.getoutput()
    else:
        _execute(command, from_tty)
_execute = gdb.execute
gdb.execute = execute
_logging_state = _LoggingState()

def get_selected_inferior():
    if False:
        while True:
            i = 10
    '\n    Return the selected inferior in gdb.\n    '
    return gdb.inferiors()[0]
    selected_thread = gdb.selected_thread()
    for inferior in gdb.inferiors():
        for thread in inferior.threads():
            if thread == selected_thread:
                return inferior

def source_gdb_script(script_contents, to_string=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Source a gdb script with script_contents passed as a string. This is useful\n    to provide defines for py-step and py-next to make them repeatable (this is\n    not possible with gdb.execute()). See\n    http://sourceware.org/bugzilla/show_bug.cgi?id=12216\n    '
    (fd, filename) = tempfile.mkstemp()
    f = os.fdopen(fd, 'w')
    f.write(script_contents)
    f.close()
    gdb.execute('source %s' % filename, to_string=to_string)
    os.remove(filename)

def register_defines():
    if False:
        i = 10
        return i + 15
    source_gdb_script(textwrap.dedent('        define py-step\n        -py-step\n        end\n\n        define py-next\n        -py-next\n        end\n\n        document py-step\n        %s\n        end\n\n        document py-next\n        %s\n        end\n    ') % (PyStep.__doc__, PyNext.__doc__))

def stackdepth(frame):
    if False:
        print('Hello World!')
    'Tells the stackdepth of a gdb frame.'
    depth = 0
    while frame:
        frame = frame.older()
        depth += 1
    return depth

class ExecutionControlCommandBase(gdb.Command):
    """
    Superclass for language specific execution control. Language specific
    features should be implemented by lang_info using the LanguageInfo
    interface. 'name' is the name of the command.
    """

    def __init__(self, name, lang_info):
        if False:
            return 10
        super(ExecutionControlCommandBase, self).__init__(name, gdb.COMMAND_RUNNING, gdb.COMPLETE_NONE)
        self.lang_info = lang_info

    def install_breakpoints(self):
        if False:
            for i in range(10):
                print('nop')
        all_locations = itertools.chain(self.lang_info.static_break_functions(), self.lang_info.runtime_break_functions())
        for location in all_locations:
            result = gdb.execute('break %s' % location, to_string=True)
            yield re.search('Breakpoint (\\d+)', result).group(1)

    def delete_breakpoints(self, breakpoint_list):
        if False:
            return 10
        for bp in breakpoint_list:
            gdb.execute('delete %s' % bp)

    def filter_output(self, result):
        if False:
            i = 10
            return i + 15
        reflags = re.MULTILINE
        output_on_halt = [('^Program received signal .*', reflags | re.DOTALL), ('.*[Ww]arning.*', 0), ('^Program exited .*', reflags)]
        output_always = [('^(Old|New) value = .*', reflags), ('^\\d+: \\w+ = .*', reflags)]

        def filter_output(regexes):
            if False:
                return 10
            output = []
            for (regex, flags) in regexes:
                for match in re.finditer(regex, result, flags):
                    output.append(match.group(0))
            return '\n'.join(output)
        match_finish = re.search('^Value returned is \\$\\d+ = (.*)', result, re.MULTILINE)
        if match_finish:
            finish_output = 'Value returned: %s\n' % match_finish.group(1)
        else:
            finish_output = ''
        return (filter_output(output_on_halt), finish_output + filter_output(output_always))

    def stopped(self):
        if False:
            for i in range(10):
                print('nop')
        return get_selected_inferior().pid == 0

    def finish_executing(self, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        After doing some kind of code running in the inferior, print the line\n        of source code or the result of the last executed gdb command (passed\n        in as the `result` argument).\n        '
        (output_on_halt, output_always) = self.filter_output(result)
        if self.stopped():
            print(output_always)
            print(output_on_halt)
        else:
            frame = gdb.selected_frame()
            source_line = self.lang_info.get_source_line(frame)
            if self.lang_info.is_relevant_function(frame):
                raised_exception = self.lang_info.exc_info(frame)
                if raised_exception:
                    print(raised_exception)
            if source_line:
                if output_always.rstrip():
                    print(output_always.rstrip())
                print(source_line)
            else:
                print(result)

    def _finish(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute until the function returns (or until something else makes it\n        stop)\n        '
        if gdb.selected_frame().older() is not None:
            return gdb.execute('finish', to_string=True)
        else:
            return gdb.execute('cont', to_string=True)

    def _finish_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute until the function returns to a relevant caller.\n        '
        while True:
            result = self._finish()
            try:
                frame = gdb.selected_frame()
            except RuntimeError:
                break
            hitbp = re.search('Breakpoint (\\d+)', result)
            is_relevant = self.lang_info.is_relevant_function(frame)
            if hitbp or is_relevant or self.stopped():
                break
        return result

    def finish(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Implements the finish command.'
        result = self._finish_frame()
        self.finish_executing(result)

    def step(self, stepinto, stepover_command='next'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do a single step or step-over. Returns the result of the last gdb\n        command that made execution stop.\n\n        This implementation, for stepping, sets (conditional) breakpoints for\n        all functions that are deemed relevant. It then does a step over until\n        either something halts execution, or until the next line is reached.\n\n        If, however, stepover_command is given, it should be a string gdb\n        command that continues execution in some way. The idea is that the\n        caller has set a (conditional) breakpoint or watchpoint that can work\n        more efficiently than the step-over loop. For Python this means setting\n        a watchpoint for f->f_lasti, which means we can then subsequently\n        "finish" frames.\n        We want f->f_lasti instead of f->f_lineno, because the latter only\n        works properly with local trace functions, see\n        PyFrameObjectPtr.current_line_num and PyFrameObjectPtr.addr2line.\n        '
        if stepinto:
            breakpoint_list = list(self.install_breakpoints())
        beginframe = gdb.selected_frame()
        if self.lang_info.is_relevant_function(beginframe):
            beginline = self.lang_info.lineno(beginframe)
            if not stepinto:
                depth = stackdepth(beginframe)
        newframe = beginframe
        while True:
            if self.lang_info.is_relevant_function(newframe):
                result = gdb.execute(stepover_command, to_string=True)
            else:
                result = self._finish_frame()
            if self.stopped():
                break
            newframe = gdb.selected_frame()
            is_relevant_function = self.lang_info.is_relevant_function(newframe)
            try:
                framename = newframe.name()
            except RuntimeError:
                framename = None
            m = re.search('Breakpoint (\\d+)', result)
            if m:
                if is_relevant_function and m.group(1) in breakpoint_list:
                    break
            if newframe != beginframe:
                if not stepinto:
                    newdepth = stackdepth(newframe)
                    is_relevant_function = newdepth < depth and is_relevant_function
                if is_relevant_function:
                    break
            else:
                lineno = self.lang_info.lineno(newframe)
                if lineno and lineno != beginline:
                    break
        if stepinto:
            self.delete_breakpoints(breakpoint_list)
        self.finish_executing(result)

    def run(self, args, from_tty):
        if False:
            i = 10
            return i + 15
        self.finish_executing(gdb.execute('run ' + args, to_string=True))

    def cont(self, *args):
        if False:
            i = 10
            return i + 15
        self.finish_executing(gdb.execute('cont', to_string=True))

class LanguageInfo(object):
    """
    This class defines the interface that ExecutionControlCommandBase needs to
    provide language-specific execution control.

    Classes that implement this interface should implement:

        lineno(frame)
            Tells the current line number (only called for a relevant frame).
            If lineno is a false value it is not checked for a difference.

        is_relevant_function(frame)
            tells whether we care about frame 'frame'

        get_source_line(frame)
            get the line of source code for the current line (only called for a
            relevant frame). If the source code cannot be retrieved this
            function should return None

        exc_info(frame) -- optional
            tells whether an exception was raised, if so, it should return a
            string representation of the exception value, None otherwise.

        static_break_functions()
            returns an iterable of function names that are considered relevant
            and should halt step-into execution. This is needed to provide a
            performing step-into

        runtime_break_functions() -- optional
            list of functions that we should break into depending on the
            context
    """

    def exc_info(self, frame):
        if False:
            while True:
                i = 10
        "See this class' docstring."

    def runtime_break_functions(self):
        if False:
            print('Hello World!')
        '\n        Implement this if the list of step-into functions depends on the\n        context.\n        '
        return ()

class PythonInfo(LanguageInfo):

    def pyframe(self, frame):
        if False:
            for i in range(10):
                print('nop')
        pyframe = Frame(frame).get_pyop()
        if pyframe:
            return pyframe
        else:
            raise gdb.RuntimeError('Unable to find the Python frame, run your code with a debug build (configure with --with-pydebug or compile with -g).')

    def lineno(self, frame):
        if False:
            i = 10
            return i + 15
        return self.pyframe(frame).current_line_num()

    def is_relevant_function(self, frame):
        if False:
            print('Hello World!')
        return Frame(frame).is_evalframeex()

    def get_source_line(self, frame):
        if False:
            while True:
                i = 10
        try:
            pyframe = self.pyframe(frame)
            return '%4d    %s' % (pyframe.current_line_num(), pyframe.current_line().rstrip())
        except IOError:
            return None

    def exc_info(self, frame):
        if False:
            i = 10
            return i + 15
        try:
            tstate = frame.read_var('tstate').dereference()
            if gdb.parse_and_eval('tstate->frame == f'):
                if sys.version_info >= (3, 12, 0, 'alpha', 6):
                    inf_type = inf_value = tstate['current_exception']
                else:
                    inf_type = tstate['curexc_type']
                    inf_value = tstate['curexc_value']
                if inf_type:
                    return 'An exception was raised: %s' % (inf_value,)
        except (ValueError, RuntimeError):
            pass

    def static_break_functions(self):
        if False:
            for i in range(10):
                print('nop')
        yield 'PyEval_EvalFrameEx'

class PythonStepperMixin(object):
    """
    Make this a mixin so CyStep can also inherit from this and use a
    CythonCodeStepper at the same time.
    """

    def python_step(self, stepinto):
        if False:
            while True:
                i = 10
        '\n        Set a watchpoint on the Python bytecode instruction pointer and try\n        to finish the frame\n        '
        output = gdb.execute('watch f->f_lasti', to_string=True)
        watchpoint = int(re.search('[Ww]atchpoint (\\d+):', output).group(1))
        self.step(stepinto=stepinto, stepover_command='finish')
        gdb.execute('delete %s' % watchpoint)

class PyStep(ExecutionControlCommandBase, PythonStepperMixin):
    """Step through Python code."""
    stepinto = True

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        if False:
            while True:
                i = 10
        self.python_step(stepinto=self.stepinto)

class PyNext(PyStep):
    """Step-over Python code."""
    stepinto = False

class PyFinish(ExecutionControlCommandBase):
    """Execute until function returns to a caller."""
    invoke = dont_suppress_errors(ExecutionControlCommandBase.finish)

class PyRun(ExecutionControlCommandBase):
    """Run the program."""
    invoke = dont_suppress_errors(ExecutionControlCommandBase.run)

class PyCont(ExecutionControlCommandBase):
    invoke = dont_suppress_errors(ExecutionControlCommandBase.cont)

def _pointervalue(gdbval):
    if False:
        i = 10
        return i + 15
    '\n    Return the value of the pointer as a Python int.\n\n    gdbval.type must be a pointer type\n    '
    if gdbval.address is not None:
        return int(gdbval.address)
    else:
        return int(gdbval)

def pointervalue(gdbval):
    if False:
        for i in range(10):
            print('nop')
    pointer = _pointervalue(gdbval)
    try:
        if pointer < 0:
            raise gdb.GdbError('Negative pointer value, presumably a bug in gdb, aborting.')
    except RuntimeError:
        pass
    return pointer

def get_inferior_unicode_postfix():
    if False:
        return 10
    try:
        gdb.parse_and_eval('PyUnicode_FromEncodedObject')
    except RuntimeError:
        try:
            gdb.parse_and_eval('PyUnicodeUCS2_FromEncodedObject')
        except RuntimeError:
            return 'UCS4'
        else:
            return 'UCS2'
    else:
        return ''

class PythonCodeExecutor(object):
    Py_single_input = 256
    Py_file_input = 257
    Py_eval_input = 258

    def malloc(self, size):
        if False:
            for i in range(10):
                print('nop')
        chunk = gdb.parse_and_eval('(void *) malloc((size_t) %d)' % size)
        pointer = pointervalue(chunk)
        if pointer == 0:
            raise gdb.GdbError('No memory could be allocated in the inferior.')
        return pointer

    def alloc_string(self, string):
        if False:
            return 10
        pointer = self.malloc(len(string))
        get_selected_inferior().write_memory(pointer, string)
        return pointer

    def alloc_pystring(self, string):
        if False:
            i = 10
            return i + 15
        stringp = self.alloc_string(string)
        PyString_FromStringAndSize = 'PyString_FromStringAndSize'
        try:
            gdb.parse_and_eval(PyString_FromStringAndSize)
        except RuntimeError:
            PyString_FromStringAndSize = 'PyUnicode%s_FromStringAndSize' % (get_inferior_unicode_postfix(),)
        try:
            result = gdb.parse_and_eval('(PyObject *) %s((char *) %d, (size_t) %d)' % (PyString_FromStringAndSize, stringp, len(string)))
        finally:
            self.free(stringp)
        pointer = pointervalue(result)
        if pointer == 0:
            raise gdb.GdbError('Unable to allocate Python string in the inferior.')
        return pointer

    def free(self, pointer):
        if False:
            for i in range(10):
                print('nop')
        gdb.parse_and_eval('(void) free((void *) %d)' % pointer)

    def incref(self, pointer):
        if False:
            for i in range(10):
                print('nop')
        'Increment the reference count of a Python object in the inferior.'
        gdb.parse_and_eval('Py_IncRef((PyObject *) %d)' % pointer)

    def xdecref(self, pointer):
        if False:
            while True:
                i = 10
        'Decrement the reference count of a Python object in the inferior.'
        gdb.parse_and_eval('Py_DecRef((PyObject *) %d)' % pointer)

    def evalcode(self, code, input_type, global_dict=None, local_dict=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate python code `code` given as a string in the inferior and\n        return the result as a gdb.Value. Returns a new reference in the\n        inferior.\n\n        Of course, executing any code in the inferior may be dangerous and may\n        leave the debuggee in an unsafe state or terminate it altogether.\n        '
        if '\x00' in code:
            raise gdb.GdbError('String contains NUL byte.')
        code += '\x00'
        pointer = self.alloc_string(code)
        globalsp = pointervalue(global_dict)
        localsp = pointervalue(local_dict)
        if globalsp == 0 or localsp == 0:
            raise gdb.GdbError('Unable to obtain or create locals or globals.')
        code = '\n            PyRun_String(\n                (char *) %(code)d,\n                (int) %(start)d,\n                (PyObject *) %(globals)s,\n                (PyObject *) %(locals)d)\n        ' % dict(code=pointer, start=input_type, globals=globalsp, locals=localsp)
        with FetchAndRestoreError():
            try:
                pyobject_return_value = gdb.parse_and_eval(code)
            finally:
                self.free(pointer)
        return pyobject_return_value

class FetchAndRestoreError(PythonCodeExecutor):
    """
    Context manager that fetches the error indicator in the inferior and
    restores it on exit.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.sizeof_PyObjectPtr = gdb.lookup_type('PyObject').pointer().sizeof
        self.pointer = self.malloc(self.sizeof_PyObjectPtr * 3)
        type = self.pointer
        value = self.pointer + self.sizeof_PyObjectPtr
        traceback = self.pointer + self.sizeof_PyObjectPtr * 2
        self.errstate = (type, value, traceback)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        gdb.parse_and_eval('PyErr_Fetch(%d, %d, %d)' % self.errstate)

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        if gdb.parse_and_eval('(int) PyErr_Occurred()'):
            gdb.parse_and_eval('PyErr_Print()')
        pyerr_restore = 'PyErr_Restore((PyObject *) *%d,(PyObject *) *%d,(PyObject *) *%d)'
        try:
            gdb.parse_and_eval(pyerr_restore % self.errstate)
        finally:
            self.free(self.pointer)

class FixGdbCommand(gdb.Command):

    def __init__(self, command, actual_command):
        if False:
            i = 10
            return i + 15
        super(FixGdbCommand, self).__init__(command, gdb.COMMAND_DATA, gdb.COMPLETE_NONE)
        self.actual_command = actual_command

    def fix_gdb(self):
        if False:
            i = 10
            return i + 15
        "\n        It seems that invoking either 'cy exec' and 'py-exec' work perfectly\n        fine, but after this gdb's python API is entirely broken.\n        Maybe some uncleared exception value is still set?\n        sys.exc_clear() didn't help. A demonstration:\n\n        (gdb) cy exec 'hello'\n        'hello'\n        (gdb) python gdb.execute('cont')\n        RuntimeError: Cannot convert value to int.\n        Error while executing Python code.\n        (gdb) python gdb.execute('cont')\n        [15148 refs]\n\n        Program exited normally.\n        "
        warnings.filterwarnings('ignore', '.*', RuntimeWarning, re.escape(__name__))
        try:
            int(gdb.parse_and_eval('(void *) 0')) == 0
        except RuntimeError:
            pass

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        if False:
            while True:
                i = 10
        self.fix_gdb()
        try:
            gdb.execute('%s %s' % (self.actual_command, args))
        except RuntimeError as e:
            raise gdb.GdbError(str(e))
        self.fix_gdb()

def _evalcode_python(executor, code, input_type):
    if False:
        while True:
            i = 10
    '\n    Execute Python code in the most recent stack frame.\n    '
    global_dict = gdb.parse_and_eval('PyEval_GetGlobals()')
    local_dict = gdb.parse_and_eval('PyEval_GetLocals()')
    if pointervalue(global_dict) == 0 or pointervalue(local_dict) == 0:
        raise gdb.GdbError('Unable to find the locals or globals of the most recent Python function (relative to the selected frame).')
    return executor.evalcode(code, input_type, global_dict, local_dict)

class PyExec(gdb.Command):

    def readcode(self, expr):
        if False:
            return 10
        if expr:
            return (expr, PythonCodeExecutor.Py_single_input)
        else:
            lines = []
            while True:
                try:
                    if sys.version_info[0] == 2:
                        line = raw_input()
                    else:
                        line = input('>')
                except EOFError:
                    break
                else:
                    if line.rstrip() == 'end':
                        break
                    lines.append(line)
            return ('\n'.join(lines), PythonCodeExecutor.Py_file_input)

    @dont_suppress_errors
    def invoke(self, expr, from_tty):
        if False:
            while True:
                i = 10
        (expr, input_type) = self.readcode(expr)
        executor = PythonCodeExecutor()
        executor.xdecref(_evalcode_python(executor, input_type, global_dict, local_dict))
gdb.execute('set breakpoint pending on')
if hasattr(gdb, 'GdbError'):
    py_step = PyStep('-py-step', PythonInfo())
    py_next = PyNext('-py-next', PythonInfo())
    register_defines()
    py_finish = PyFinish('py-finish', PythonInfo())
    py_run = PyRun('py-run', PythonInfo())
    py_cont = PyCont('py-cont', PythonInfo())
    py_exec = FixGdbCommand('py-exec', '-py-exec')
    _py_exec = PyExec('-py-exec', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)
else:
    warnings.warn('Use gdb 7.2 or higher to use the py-exec command.')