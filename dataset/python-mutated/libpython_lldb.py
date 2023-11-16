import abc
import argparse
import collections
import io
import locale
import re
import shlex
import struct
import lldb
import six
ENCODING_RE = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-_.a-zA-Z0-9]+)')

class PyObject(object):

    def __init__(self, lldb_value):
        if False:
            for i in range(10):
                print('nop')
        self.lldb_value = lldb_value

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.value)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.value)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        assert isinstance(other, PyObject)
        return self.value == other.value

    def child(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.lldb_value.GetChildMemberWithName(name)

    @classmethod
    def from_value(cls, v):
        if False:
            i = 10
            return i + 15
        subclasses = {c.typename: c for c in cls.__subclasses__()}
        typename = cls.typename_of(v)
        return subclasses.get(typename, cls)(v)

    @staticmethod
    def typename_of(v):
        if False:
            while True:
                i = 10
        try:
            addr = v.GetChildMemberWithName('ob_type').GetChildMemberWithName('tp_name').unsigned
            if not addr:
                return
            process = v.GetProcess()
            return process.ReadCStringFromMemory(addr, 256, lldb.SBError())
        except Exception:
            pass

    @property
    def typename(self):
        if False:
            print('Hello World!')
        return self.typename_of(self.lldb_value)

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.lldb_value.addr)

    @property
    def target(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lldb_value.GetTarget()

    @property
    def process(self):
        if False:
            while True:
                i = 10
        return self.lldb_value.GetProcess()

    @property
    def deref(self):
        if False:
            for i in range(10):
                print('nop')
        if self.lldb_value.TypeIsPointerType():
            return self.lldb_value.deref
        else:
            return self.lldb_value

class PyLongObject(PyObject):
    typename = 'int'
    cpython_struct = 'PyLongObject'

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        The absolute value of a number is equal to:\n\n            SUM(for i=0 through abs(ob_size)-1) ob_digit[i] * 2**(SHIFT*i)\n\n        Negative numbers are represented with ob_size < 0;\n        zero is represented by ob_size == 0.\n\n        where SHIFT can be either:\n            #define PyLong_SHIFT        30\n        or:\n            #define PyLong_SHIFT        15\n\n        '
        long_type = self.target.FindFirstType(self.cpython_struct)
        digit_type = self.target.FindFirstType('digit')
        shift = 15 if digit_type.size == 2 else 30
        value = self.deref.Cast(long_type)
        size = value.GetChildMemberWithName('ob_base').GetChildMemberWithName('ob_size').signed
        if not size:
            return 0
        digits = value.GetChildMemberWithName('ob_digit')
        abs_value = sum((digits.GetChildAtIndex(i, 0, True).unsigned * 2 ** (shift * i) for i in range(0, abs(size))))
        return abs_value if size > 0 else -abs_value

class PyBoolObject(PyObject):
    typename = 'bool'

    @property
    def value(self):
        if False:
            return 10
        long_type = self.target.FindFirstType('PyLongObject')
        value = self.deref.Cast(long_type)
        digits = value.GetChildMemberWithName('ob_digit')
        return bool(digits.GetChildAtIndex(0).unsigned)

class PyFloatObject(PyObject):
    typename = 'float'
    cpython_struct = 'PyFloatObject'

    @property
    def value(self):
        if False:
            while True:
                i = 10
        float_type = self.target.FindFirstType(self.cpython_struct)
        value = self.deref.Cast(float_type)
        fval = value.GetChildMemberWithName('ob_fval')
        return float(fval.GetValue())

class PyBytesObject(PyObject):
    typename = 'bytes'
    cpython_struct = 'PyBytesObject'

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        bytes_type = self.target.FindFirstType(self.cpython_struct)
        value = self.deref.Cast(bytes_type)
        size = value.GetChildMemberWithName('ob_base').GetChildMemberWithName('ob_size').unsigned
        addr = value.GetChildMemberWithName('ob_sval').GetLoadAddress()
        return bytes(self.process.ReadMemory(addr, size, lldb.SBError())) if size else b''

class PyUnicodeObject(PyObject):
    typename = 'str'
    cpython_struct = 'PyUnicodeObject'
    U_WCHAR_KIND = 0
    U_1BYTE_KIND = 1
    U_2BYTE_KIND = 2
    U_4BYTE_KIND = 4

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        str_type = self.target.FindFirstType(self.cpython_struct)
        value = self.deref.Cast(str_type)
        state = value.GetChildMemberWithName('_base').GetChildMemberWithName('_base').GetChildMemberWithName('state')
        length = value.GetChildMemberWithName('_base').GetChildMemberWithName('_base').GetChildMemberWithName('length').unsigned
        if not length:
            return u''
        compact = bool(state.GetChildMemberWithName('compact').unsigned)
        is_ascii = bool(state.GetChildMemberWithName('ascii').unsigned)
        kind = state.GetChildMemberWithName('kind').unsigned
        ready = bool(state.GetChildMemberWithName('ready').unsigned)
        if is_ascii and compact and ready:
            ascii_type = self.target.FindFirstType('PyASCIIObject')
            value = value.Cast(ascii_type)
            addr = int(value.location, 16) + value.size
            rv = self.process.ReadMemory(addr, length, lldb.SBError())
            return rv.decode('ascii')
        elif compact and ready:
            compact_type = self.target.FindFirstType('PyCompactUnicodeObject')
            value = value.Cast(compact_type)
            addr = int(value.location, 16) + value.size
            rv = self.process.ReadMemory(addr, length * kind, lldb.SBError())
            if kind == self.U_1BYTE_KIND:
                return rv.decode('latin-1')
            elif kind == self.U_2BYTE_KIND:
                return rv.decode('utf-16')
            elif kind == self.U_4BYTE_KIND:
                return rv.decode('utf-32')
            else:
                raise ValueError('Unsupported PyUnicodeObject kind: {}'.format(kind))
        else:
            raise ValueError('Unsupported PyUnicodeObject kind: {}'.format(kind))

class PyNoneObject(PyObject):
    typename = 'NoneType'
    value = None

class _PySequence(object):

    @property
    def value(self):
        if False:
            return 10
        value = self.deref.Cast(self.lldb_type)
        size = value.GetChildMemberWithName('ob_base').GetChildMemberWithName('ob_size').signed
        items = value.GetChildMemberWithName('ob_item')
        return self.python_type((PyObject.from_value(items.GetChildAtIndex(i, 0, True)) for i in range(size)))

class PyListObject(_PySequence, PyObject):
    python_type = list
    typename = 'list'
    cpython_struct = 'PyListObject'

    @property
    def lldb_type(self):
        if False:
            print('Hello World!')
        return self.target.FindFirstType(self.cpython_struct)

class PyTupleObject(_PySequence, PyObject):
    python_type = tuple
    typename = 'tuple'
    cpython_struct = 'PyTupleObject'

    @property
    def lldb_type(self):
        if False:
            return 10
        return self.target.FindFirstType(self.cpython_struct)

class _PySetObject(object):
    cpython_struct = 'PySetObject'

    @property
    def value(self):
        if False:
            while True:
                i = 10
        set_type = self.target.FindFirstType(self.cpython_struct)
        value = self.deref.Cast(set_type)
        size = value.GetChildMemberWithName('mask').unsigned + 1
        table = value.GetChildMemberWithName('table')
        array = table.deref.Cast(table.type.GetPointeeType().GetArrayType(size))
        rv = set()
        for i in range(size):
            entry = array.GetChildAtIndex(i)
            key = entry.GetChildMemberWithName('key')
            hash_ = entry.GetChildMemberWithName('hash').signed
            if hash_ != -1 and (hash_ != 0 or key.unsigned != 0):
                rv.add(PyObject.from_value(key))
        return rv

class PySetObject(_PySetObject, PyObject):
    typename = 'set'

class PyFrozenSetObject(_PySetObject, PyObject):
    typename = 'frozenset'

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        return frozenset(super(PyFrozenSetObject, self).value)

class _PyDictObject(object):

    @property
    def value(self):
        if False:
            return 10
        byte_type = self.target.FindFirstType('char')
        dict_type = self.target.FindFirstType('PyDictObject')
        dictentry_type = self.target.FindFirstType('PyDictKeyEntry')
        object_type = self.target.FindFirstType('PyObject')
        value = self.deref.Cast(dict_type)
        ma_keys = value.GetChildMemberWithName('ma_keys')
        table_size = ma_keys.GetChildMemberWithName('dk_size').unsigned
        num_entries = ma_keys.GetChildMemberWithName('dk_nentries').unsigned
        if table_size < 255:
            index_size = 1
        elif table_size < 65535:
            index_size = 2
        elif table_size < 268435455:
            index_size = 4
        else:
            index_size = 8
        shift = table_size * index_size
        indices = ma_keys.GetChildMemberWithName('dk_indices')
        if indices.IsValid():
            entries = indices.Cast(byte_type.GetArrayType(shift)).GetChildAtIndex(shift, 0, True).AddressOf().Cast(dictentry_type.GetPointerType()).deref.Cast(dictentry_type.GetArrayType(num_entries))
        else:
            num_entries = table_size
            entries = ma_keys.GetChildMemberWithName('dk_entries').Cast(dictentry_type.GetArrayType(num_entries))
        ma_values = value.GetChildMemberWithName('ma_values')
        if ma_values.unsigned:
            is_split = True
            ma_values = ma_values.deref.Cast(object_type.GetPointerType().GetArrayType(num_entries))
        else:
            is_split = False
        rv = self.python_type()
        for i in range(num_entries):
            entry = entries.GetChildAtIndex(i)
            k = entry.GetChildMemberWithName('me_key')
            v = entry.GetChildMemberWithName('me_value')
            if k.unsigned != 0 and v.unsigned != 0:
                rv[PyObject.from_value(k)] = PyObject.from_value(v)
            elif k.unsigned != 0 and is_split:
                for j in range(i, table_size):
                    v = ma_values.GetChildAtIndex(j)
                    if v.unsigned != 0:
                        rv[PyObject.from_value(k)] = PyObject.from_value(v)
                        break
        return rv

class PyDictObject(_PyDictObject, PyObject):
    python_type = dict
    typename = 'dict'
    cpython_struct = 'PyDictObject'

class Counter(_PyDictObject, PyObject):
    python_type = collections.Counter
    typename = 'Counter'

class OrderedDict(_PyDictObject, PyObject):
    python_type = collections.OrderedDict
    typename = 'collections.OrderedDict'

class Defaultdict(PyObject):
    typename = 'collections.defaultdict'
    cpython_struct = 'defdictobject'

    @property
    def value(self):
        if False:
            while True:
                i = 10
        dict_type = self.target.FindFirstType('defdictobject')
        value = self.deref.Cast(dict_type)
        return PyDictObject(value.GetChildMemberWithName('dict').AddressOf()).value

class _CollectionsUserObject(object):

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        dict_offset = self.lldb_value.GetChildMemberWithName('ob_type').GetChildMemberWithName('tp_dictoffset').unsigned
        object_type = self.target.FindFirstType('PyObject')
        address = lldb.SBAddress(int(self.lldb_value.value, 16) + dict_offset, self.target)
        value = self.target.CreateValueFromAddress('value', address, object_type.GetPointerType())
        return next((v for (k, v) in PyDictObject(value).value.items() if k.value == 'data'))

class UserDict(_CollectionsUserObject, PyObject):
    typename = 'UserDict'

class UserList(_CollectionsUserObject, PyObject):
    typename = 'UserList'

class UserString(_CollectionsUserObject, PyObject):
    typename = 'UserString'

class PyCodeAddressRange(object):
    """A class for parsing the line number table implemented in PEP 626.

    The format of the line number table is not part of CPython's API and may
    change without warning. The PEP has a dedicated section
    (https://www.python.org/dev/peps/pep-0626/#out-of-process-debuggers-and-profilers),
    which says that tools, that can't use C-API, should just copy the implementation
    of functions required for parsing this table. We do just that below but also
    translate C to Python.
    """

    def __init__(self, co_linetable, co_firstlineno):
        if False:
            return 10
        'Implements PyLineTable_InitAddressRange from codeobject.c.'
        self.lo_next = 0
        self.co_linetable = co_linetable
        self.computed_line = co_firstlineno
        self.ar_start = -1
        self.ar_end = 0
        self.ar_line = -1

    def next_address_range(self):
        if False:
            return 10
        'Implements PyLineTable_NextAddressRange from codeobject.c.'
        if self._at_end():
            return False
        self._advance()
        while self.ar_start == self.ar_end:
            self._advance()
        return True

    def prev_address_range(self):
        if False:
            print('Hello World!')
        'Implements PyLineTable_PreviousAddressRange from codeobject.c.'
        if self.ar_start <= 0:
            return False
        self._retreat()
        while self.ar_start == self.ar_end:
            self._retreat()
        return True

    def _at_end(self):
        if False:
            print('Hello World!')
        return self.lo_next >= len(self.co_linetable)

    def _advance(self):
        if False:
            print('Hello World!')
        self.ar_start = self.ar_end
        delta = struct.unpack('B', self.co_linetable[self.lo_next:self.lo_next + 1])[0]
        self.ar_end += delta
        ldelta = struct.unpack('b', self.co_linetable[self.lo_next + 1:self.lo_next + 2])[0]
        self.lo_next += 2
        if ldelta == -128:
            self.ar_line = -1
        else:
            self.computed_line += ldelta
            self.ar_line = self.computed_line

    def _retreat(self):
        if False:
            for i in range(10):
                print('nop')
        ldelta = struct.unpack('b', self.co_linetable[self.lo_next - 1:self.lo_next])[0]
        if ldelta == -128:
            ldelta = 0
        self.computed_line -= ldelta
        self.lo_next -= 2
        self.ar_end = self.ar_start
        self.ar_start -= struct.unpack('B', self.co_linetable[self.lo_next - 2:self.lo_next - 1])[0]
        ldelta = struct.unpack('b', self.co_linetable[self.lo_next - 1:self.lo_next])[0]
        if ldelta == -128:
            self.ar_line = -1
        else:
            self.ar_line = self.computed_line

class PyCodeObject(PyObject):
    typename = 'code'

    def addr2line(self, f_lineno, f_lasti):
        if False:
            while True:
                i = 10
        addr_range_type = self.target.FindFirstType('PyCodeAddressRange')
        if addr_range_type.IsValid():
            if f_lineno:
                return f_lineno
            else:
                return self._from_co_linetable(f_lasti * 2)
        else:
            return self._from_co_lnotab(f_lasti) + f_lineno

    def _from_co_linetable(self, address):
        if False:
            print('Hello World!')
        'Translated code from Objects/codeobject.c:PyCode_Addr2Line.'
        co_linetable = PyObject.from_value(self.child('co_linetable')).value
        co_firstlineno = self.child('co_firstlineno').signed
        if address < 0:
            return co_firstlineno
        bounds = PyCodeAddressRange(co_linetable, co_firstlineno)
        while bounds.ar_end <= address:
            if not bounds.next_address_range():
                return -1
        while bounds.ar_start > address:
            if not bounds.prev_address_range():
                return -1
        return bounds.ar_line

    def _from_co_lnotab(self, address):
        if False:
            while True:
                i = 10
        'Translated pseudocode from Objects/lnotab_notes.txt.'
        co_lnotab = PyObject.from_value(self.child('co_lnotab')).value
        assert len(co_lnotab) % 2 == 0
        lineno = addr = 0
        for (addr_incr, line_incr) in zip(co_lnotab[::2], co_lnotab[1::2]):
            addr_incr = ord(addr_incr) if isinstance(addr_incr, (bytes, str)) else addr_incr
            line_incr = ord(line_incr) if isinstance(line_incr, (bytes, str)) else line_incr
            addr += addr_incr
            if addr > address:
                return lineno
            if line_incr >= 128:
                line_incr -= 256
            lineno += line_incr
        return lineno

class PyFrameObject(PyObject):
    typename = 'frame'

    def __init__(self, lldb_value):
        if False:
            i = 10
            return i + 15
        super(PyFrameObject, self).__init__(lldb_value)
        self.co = PyCodeObject(self.child('f_code'))

    @classmethod
    def _from_frame_no_walk(cls, frame):
        if False:
            while True:
                i = 10
        '\n        Extract PyFrameObject object from current frame w/o stack walking.\n        '
        f = frame.variables['f']
        if f and is_available(f[0]):
            return cls(f[0])
        else:
            return None

    @classmethod
    def _from_frame_heuristic(cls, frame):
        if False:
            return 10
        'Extract PyFrameObject object from current frame using heuristic.\n\n        When CPython is compiled with aggressive optimizations, the location\n        of PyFrameObject variable f can sometimes be lost. Usually, we still\n        can figure it out by analyzing the state of CPU registers. This is not\n        very reliable, because we basically try to cast the value stored in\n        each register to (PyFrameObject*) and see if it produces a valid\n        PyObject object.\n\n        This becomes especially ugly when there is more than one PyFrameObject*\n        in CPU registers at the same time. In this case we are looking for the\n        frame with a parent, that we have not seen yet.\n        '
        target = frame.GetThread().GetProcess().GetTarget()
        object_type = target.FindFirstType('PyObject')
        public_frame_type = target.FindFirstType('PyFrameObject')
        internal_frame_type = target.FindFirstType('_frame')
        frame_type = public_frame_type if public_frame_type.members else internal_frame_type
        found_frames = []
        for register in general_purpose_registers(frame):
            sbvalue = frame.register[register]
            if not sbvalue or not sbvalue.unsigned:
                continue
            pyobject = PyObject(sbvalue.Cast(object_type.GetPointerType()))
            if pyobject.typename != PyFrameObject.typename:
                continue
            found_frames.append(PyFrameObject(sbvalue.Cast(frame_type.GetPointerType())))
        found_frames_addresses = [frame.lldb_value.unsigned for frame in found_frames]
        eligible_frames = [frame for frame in found_frames if frame.child('f_back').unsigned not in found_frames_addresses]
        if eligible_frames:
            return eligible_frames[0]

    @classmethod
    def from_frame(cls, frame):
        if False:
            for i in range(10):
                print('nop')
        if frame is None:
            return None
        if frame.name not in ('_PyEval_EvalFrameDefault', 'PyEval_EvalFrameEx'):
            return None
        methods = (cls._from_frame_no_walk, lambda frame: frame.parent and cls._from_frame_no_walk(frame.parent), cls._from_frame_heuristic, lambda frame: frame.parent and cls._from_frame_heuristic(frame.parent))
        for method in methods:
            result = method(frame)
            if result is not None:
                return result

    @classmethod
    def get_pystack(cls, thread):
        if False:
            while True:
                i = 10
        pyframes = []
        frame = thread.GetSelectedFrame()
        while frame:
            pyframe = cls.from_frame(frame)
            if pyframe is not None:
                pyframes.append(pyframe)
            frame = frame.get_parent_frame()
        return pyframes

    @property
    def filename(self):
        if False:
            i = 10
            return i + 15
        return PyObject.from_value(self.co.child('co_filename')).value

    @property
    def line_number(self):
        if False:
            print('Hello World!')
        f_lineno = self.child('f_lineno').signed
        f_lasti = self.child('f_lasti').signed
        return self.co.addr2line(f_lineno, f_lasti)

    @property
    def line(self):
        if False:
            print('Hello World!')
        try:
            encoding = source_file_encoding(self.filename)
            return source_file_lines(self.filename, self.line_number, self.line_number + 1, encoding=encoding)[0]
        except (IOError, IndexError):
            return u'<source code is not available>'

    def to_pythonlike_string(self):
        if False:
            print('Hello World!')
        lineno = self.line_number
        co_name = PyObject.from_value(self.co.child('co_name')).value
        return u'File "{filename}", line {lineno}, in {co_name}'.format(filename=self.filename, co_name=co_name, lineno=lineno)

@six.add_metaclass(abc.ABCMeta)
class Command(object):
    """Base class for py-* command implementations.

    Takes care of commands registration and error handling.

    Subclasses' docstrings are used as help messages for the commands. The
    first line of a docstring act as a command description that appears ina
    the output of `help`.
    """

    def __init__(self, debugger, unused):
        if False:
            return 10
        pass

    def get_short_help(self):
        if False:
            return 10
        return self.__doc__.splitlines()[0]

    def get_long_help(self):
        if False:
            while True:
                i = 10
        return self.__doc__

    def __call__(self, debugger, command, exe_ctx, result):
        if False:
            return 10
        try:
            args = self.argument_parser.parse_args(shlex.split(command))
            self.execute(debugger, args, result)
        except Exception as e:
            msg = u'Failed to execute command `{}`: {}'.format(self.command, e)
            if six.PY2:
                msg = msg.encode('utf-8')
            result.SetError(msg)

    @property
    def argument_parser(self):
        if False:
            i = 10
            return i + 15
        'ArgumentParser instance used for this command.\n\n        The default parser does not have any arguments and only prints a help\n        message based on the command description.\n\n        Subclasses are expected to override this property in order to add\n        additional commands to the provided ArgumentParser instance.\n        '
        return argparse.ArgumentParser(prog=self.command, description=self.get_long_help(), formatter_class=argparse.RawDescriptionHelpFormatter)

    @abc.abstractproperty
    def command(self):
        if False:
            return 10
        'Command name.\n\n        This name will be used by LLDB in order to uniquely identify an\n        implementation that should be executed when a command is run\n        in the REPL.\n        '

    @abc.abstractmethod
    def execute(self, debugger, args, result):
        if False:
            return 10
        "Implementation of the command.\n\n        Subclasses override this method to implement the logic of a given\n        command, e.g. printing a stacktrace. The command output should be\n        communicated back via the provided result object, so that it's\n        properly routed to LLDB frontend. Any unhandled exception will be\n        automatically transformed into proper errors.\n\n        Args:\n            debugger: lldb.SBDebugger: the primary interface to LLDB scripting\n            args: argparse.Namespace: an object holding parsed command arguments\n            result: lldb.SBCommandReturnObject: a container which holds the\n                    result from command execution\n        "

class PyBt(Command):
    """Print a Python-level call trace of the selected thread."""
    command = 'py-bt'

    def execute(self, debugger, args, result):
        if False:
            while True:
                i = 10
        target = debugger.GetSelectedTarget()
        thread = target.GetProcess().GetSelectedThread()
        pystack = PyFrameObject.get_pystack(thread)
        lines = []
        for pyframe in reversed(pystack):
            lines.append(u'  ' + pyframe.to_pythonlike_string())
            lines.append(u'    ' + pyframe.line.strip())
        if lines:
            write_string(result, u'Traceback (most recent call last):')
            write_string(result, u'\n'.join(lines))
        else:
            write_string(result, u'No Python traceback found')

class PyList(Command):
    """List the source code of the Python module that is currently being executed.

    Use

        py-list

    to list the source code around (5 lines before and after) the line that is
    currently being executed.


    Use

        py-list start

    to list the source code starting at a different line number.


    Use

        py-list start end

    to list the source code within a specific range of lines.
    """
    command = 'py-list'

    @property
    def argument_parser(self):
        if False:
            for i in range(10):
                print('nop')
        parser = super(PyList, self).argument_parser
        parser.add_argument('linenum', nargs='*', type=int, default=[0, 0])
        return parser

    @staticmethod
    def linenum_range(current_line_num, specified_range):
        if False:
            for i in range(10):
                print('nop')
        if len(specified_range) == 2:
            (start, end) = specified_range
        elif len(specified_range) == 1:
            start = specified_range[0]
            end = start + 10
        else:
            start = None
            end = None
        start = start or max(current_line_num - 5, 1)
        end = end or current_line_num + 5
        return (start, end)

    def execute(self, debugger, args, result):
        if False:
            while True:
                i = 10
        linenum_range = args.linenum
        if len(linenum_range) > 2:
            write_string(result, u'Usage: py-list [start [end]]')
            return
        current_frame = select_closest_python_frame(debugger)
        if current_frame is None:
            write_string(result, u'<source code is not available>')
            return
        filename = current_frame.filename
        current_line_num = current_frame.line_number
        (start, end) = PyList.linenum_range(current_line_num, linenum_range)
        try:
            encoding = source_file_encoding(filename)
            lines = source_file_lines(filename, start, end + 1, encoding=encoding)
            for (i, line) in enumerate(lines, start):
                if i == current_line_num:
                    prefix = u'>{}'.format(i)
                else:
                    prefix = u'{}'.format(i)
                write_string(result, u'{:>5}    {}'.format(prefix, line.rstrip()))
        except IOError:
            write_string(result, u'<source code is not available>')

class PyUp(Command):
    """Select an older Python stack frame."""
    command = 'py-up'

    def execute(self, debugger, args, result):
        if False:
            return 10
        select_closest_python_frame(debugger, direction=Direction.UP)
        new_frame = move_python_frame(debugger, direction=Direction.UP)
        if new_frame is None:
            write_string(result, u'*** Oldest frame')
        else:
            print_frame_summary(result, new_frame)

class PyDown(Command):
    """Select a newer Python stack frame."""
    command = 'py-down'

    def execute(self, debugger, args, result):
        if False:
            while True:
                i = 10
        select_closest_python_frame(debugger, direction=Direction.DOWN)
        new_frame = move_python_frame(debugger, direction=Direction.DOWN)
        if new_frame is None:
            write_string(result, u'*** Newest frame')
        else:
            print_frame_summary(result, new_frame)

class PyLocals(Command):
    """Print the values of local variables in the selected Python frame."""
    command = 'py-locals'

    def execute(self, debugger, args, result):
        if False:
            print('Hello World!')
        current_frame = select_closest_python_frame(debugger, direction=Direction.UP)
        if current_frame is None:
            write_string(result, u'No locals found (symbols might be missing!)')
            return
        merged_locals = {}
        f_locals = current_frame.child('f_locals')
        if f_locals.unsigned != 0:
            for (k, v) in PyDictObject(f_locals).value.items():
                merged_locals[k.value] = v
        fast_locals = current_frame.child('f_localsplus')
        f_code = PyCodeObject(current_frame.child('f_code'))
        varnames = PyTupleObject(f_code.child('co_varnames'))
        for (i, name) in enumerate(varnames.value):
            value = fast_locals.GetChildAtIndex(i, 0, True)
            if value.unsigned != 0:
                merged_locals[name.value] = PyObject.from_value(value).value
            else:
                merged_locals.pop(name, None)
        for name in sorted(merged_locals.keys()):
            write_string(result, u'{} = {}'.format(name, repr(merged_locals[name])))

class Direction(object):
    DOWN = -1
    UP = 1

def print_frame_summary(result, frame):
    if False:
        for i in range(10):
            print('nop')
    'Print a short summary of a given Python frame: module and the line being executed.'
    write_string(result, u'  ' + frame.to_pythonlike_string())
    write_string(result, u'    ' + frame.line.strip())

def select_closest_python_frame(debugger, direction=Direction.UP):
    if False:
        for i in range(10):
            print('nop')
    'Select and return the closest Python frame (or do nothing if the current frame is a Python frame).'
    target = debugger.GetSelectedTarget()
    thread = target.GetProcess().GetSelectedThread()
    frame = thread.GetSelectedFrame()
    python_frame = PyFrameObject.from_frame(frame)
    if python_frame is None:
        return move_python_frame(debugger, direction)
    return python_frame

def move_python_frame(debugger, direction):
    if False:
        return 10
    'Select the next Python frame up or down the call stack.'
    target = debugger.GetSelectedTarget()
    thread = target.GetProcess().GetSelectedThread()
    current_frame = thread.GetSelectedFrame()
    if direction == Direction.UP:
        index_range = range(current_frame.idx + 1, thread.num_frames)
    else:
        index_range = reversed(range(0, current_frame.idx))
    for index in index_range:
        python_frame = PyFrameObject.from_frame(thread.GetFrameAtIndex(index))
        if python_frame is not None:
            thread.SetSelectedFrame(index)
            return python_frame

def write_string(result, string, end=u'\n', encoding=locale.getpreferredencoding()):
    if False:
        return 10
    'Helper function for writing to SBCommandReturnObject that expects bytes on py2 and str on py3.'
    if six.PY3:
        result.write(string + end)
    else:
        result.write((string + end).encode(encoding=encoding))

def is_available(lldb_value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to check if a variable is available and was not optimized out.\n    '
    return lldb_value.error.Success()

def source_file_encoding(filename):
    if False:
        i = 10
        return i + 15
    'Determine the text encoding of a Python source file.'
    with io.open(filename, 'rt', encoding='latin-1') as f:
        for _ in range(2):
            line = f.readline()
            match = re.match(ENCODING_RE, line)
            if match:
                return match.group(1)
    return 'utf-8'

def source_file_lines(filename, start, end, encoding='utf-8'):
    if False:
        print('Hello World!')
    'Return the contents of [start; end) lines of the source file.\n\n    1 based indexing is used for convenience.\n    '
    lines = []
    with io.open(filename, 'rt', encoding=encoding) as f:
        for (line_num, line) in enumerate(f, 1):
            if start <= line_num < end:
                lines.append(line)
            elif line_num > end:
                break
    return lines

def general_purpose_registers(frame):
    if False:
        i = 10
        return i + 15
    'Return a list of general purpose register names.'
    REGISTER_CLASS = 'General Purpose Registers'
    try:
        gpr = next((reg_class for reg_class in frame.registers if reg_class.name == REGISTER_CLASS))
        return [reg.name for reg in gpr.children]
    except StopIteration:
        return []

def register_commands(debugger):
    if False:
        for i in range(10):
            print('nop')
    for cls in Command.__subclasses__():
        debugger.HandleCommand('command script add -c cpython_lldb.{cls} {command}'.format(cls=cls.__name__, command=cls.command))

def pretty_printer(value, internal_dict):
    if False:
        print('Hello World!')
    'Provide a type summary for a PyObject instance.\n\n    Try to identify an actual object type and provide a representation for its\n    value (similar to `repr(something)` in Python).\n    '
    if value.TypeIsPointerType():
        type_name = value.type.GetPointeeType().name
    else:
        type_name = value.type.name
    v = pretty_printer._cpython_structs.get(type_name, PyObject.from_value)(value)
    return repr(v)

def register_summaries(debugger):
    if False:
        i = 10
        return i + 15
    debugger.HandleCommand('type summary add -F cpython_lldb.pretty_printer PyObject')
    cpython_structs = {cls.cpython_struct: cls for cls in PyObject.__subclasses__() if hasattr(cls, 'cpython_struct')}
    for type_ in cpython_structs:
        debugger.HandleCommand('type summary add -F cpython_lldb.pretty_printer {}'.format(type_))
    pretty_printer._cpython_structs = cpython_structs

def __lldb_init_module(debugger, internal_dict):
    if False:
        print('Hello World!')
    register_summaries(debugger)
    register_commands(debugger)