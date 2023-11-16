"""
PyICE is a memory mapped implementation of cached byte code.  Multiple Python
modules are compiled into a single memory mapped file.  The actual code objects
are stored in the memory mapped file.

PyICE will produce a file which is called an ice pack.  An ice pack consists of
multiple compiled Python byte codes.  An ice pack will further reduce memory
usage by folding constants across all of the modules.
"""
import io
import math
import mmap
import os
import re
import struct
import sys
import time
try:
    import _pyice
except ImportError:
    _pyice = None
from collections import deque, OrderedDict
from importlib.util import spec_from_file_location, decode_source
from os import path
from types import CodeType
"\nIcePack file format:\n\nICEPACK[byte version]\n\n4 bytes - TIMESTAMP\n\nSection offsets, an array of int32s that provide the offset to each section:\n    Modules: A sorted tree of the module names provided by this icepack.\n    Code Objects: The metadata for each code object.\n    Strings: A table of strings which are referred to in the icepack\n    Bytes: A table of bytes objects which are referred to in the icepack\n    Ints: A table of integer values which don't fit in 24-bit values\n    Big Ints: A table of integer values which don't fit in 32-bit values\n    Floats: A table of floating point numbers in the icepack\n    Complexs: A table of complex numbers\n    Tuples: A table of tuple objects, composed from this or other tables\n    FrozenSet: A table of frozen set objects\n\nSections\n========\n\nModules\n-------\n\nA sorted list of the modules/packages.  Top-level packages/modules are listed\nfirst, and then an offset to any child packages is provided.  If there are no\nchildren then the offset will be 0.\n\nEach section is organized as:\n[uint32 module count], ([uint32 name reference], [uint32 code id],\n                        [uint32 is_package], [uint32 str filename ref]\n                        [uint32 child offset])*\n\nThe module count is the number of name reference / child offsets.  The\nname reference is a reference into the string table.  The child offset is\nthe absolute offset from the start of the icepack to where anyh child modules\nexist.\n\nCode Objects\n------------\n\nCode objects are variable sized based upon the number of variables, etc...\nwhich they reference.  Therefore this is broken into two parts.  The first part\nis an array of uint32's which is the absolute offset to the code object.\n\n[uint32 absolute offset]*\n\nFollowing are the code objects themselves, laid out as:\n\n[uint32 co_bytes offset to bytes table]\n[uint32 co_argcount]\n[uint32 co_kwonlyargcount]\n[uint32 co_nlocals]\n[uint32 co_stacksize]\n[uint32 co_flags]\n[uint32 co_firstlineno]\n[uint32 co_name reference to str table]\n[uint32 co_filename reference to str table]\n[uint32 co_lnotab reference to bytes table]\n\n[uint32 length] [uint32 co_cellvars reference to str table]*\n[uint32 length] [uint32 co_freevars reference to str table]*\n[uint32 length] [uint32 co_names reference to str table]*\n[uint32 length] [uint32 co_varnames reference to str table]*\n[uint32 length] [uint32 co_consts object references]*\n\nStrings\n-------\nStrings are variable sized based upon the length of the string.\nTherefore this is broken into two parts.  The first parts is an array of int32's\nwhich is the absolute offset to the string contents.  Then the arrays of utf8\nencoded bytes represent the string.\n\n[uint32 count] number of strings\n[uint32 absolute offset]*\n\n[uint32 length] [byte utf8 encoded]*\n\nBytes\n-----\nBytes are variable sized based upon the length of the byte string.\nTherefore this is broken into two parts.  The first parts is an array of int32's\nwhich is the absolute offset to the byte object.  Then the arrays of bytes\nfollow.\n\n[uint32 count] number of byte objects\n[uint32 absolute offset]*\n\n[uint32 length] [byte]*\n\nInts\n----\nInts are fixed 32-bit values.  They are encoded as an array of the integers.\nThey are preceeded by a uint32 count.\n\nBig Ints\n--------\nTuples are variable sized based upon the number of bytes they contain.\nTherefore this is broken into two parts.  The first parts is an array of u\nuint32's which is the absolute offset to the big int representation.  Then\nthe big ints follow.\n\n[uint32 count] number of strings\n[uint32 absolute offset]*\n\nBig int's are stored as:\n\n[uint32 length] [byte array]*\n\nThe byte array is in the format returned by int.to_bytes(), serialized as\nlittle endian, and signed.\n\nFloats\n------\nFloats are fixed sized double values.  They are encoded as an array of the\nfloating point value which can simply be indexed.  They are preceeded by a\nuint32 count.\n\nComplexes\n---------\nComplex's are fixed sized pairs of double values.  They are encoded as an array\nof the real followed by the imaginary value stored as a double.  They are\npreceeded by a int32 count.\n\nTuples\n------\nTuples are variable sized based upon the number of items they contain.\nTherefore this is broken into two parts.  The first parts is an array of\nint32's which is the absolute offset to the tuple object.  Then the tuple\nobjects follow.\n\n[uint32 count] number of tuples\n[uint32 absolute offset]*\n\nTuples them selves are encoded as:\n\n[uint32 length] [uint32 object references]*\n\nFrozen Sets\n-----------\nFrozen sets are variable sized based upon the number of items they contain.\nTherefore this is broken into two parts.  The first parts is an array of\nint32's which is the absolute offset to the set object.  Then the set\nobjects follow.\n\n[uint32 count] number of sets\n[uint32 absolute offset]*\n\nSets them selves are encoded as:\n\n[uint32 length] [uint32 object references]*\n\nObject References\n=================\nOccasionally we need to embed an object reference to an untyped object.\nThis occurs in the co_const array on objects as well as for the elements\nof a tuple.\n\nReferences are encoded using a 32-bit value indicating the table to to\nget the value from and the index into that table.  The low byte is used\nto indicate the type of value and the value is encoded in the high 24\nbits.\n\n0: None\n1: Named constant\n    Upper 24-bits:\n        0 = False\n        1 = True\n        2 = Ellipsis\n2: small int, upper 24 bits is the int value\n3: int32, upper 24-bits is a table reference\n4: large int, upper 24-bits is a table reference\n5: bytes, upper 24-bits is a table reference\n6: str, upper 24-bits is a table reference\n7: float, upper 24-bits is a table reference\n8: complex, upper 24-bits is a table reference\n9: tuple, upper 24-bits is a table reference\n10: code, upper 24-bits is a table reference\n"
CODE_FORMAT = struct.Struct('I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + 'I' + '')
INT = struct.Struct('i')
UINT = struct.Struct('I')
FLOAT = struct.Struct('d')
COMPLEX = struct.Struct('dd')
TIMESTAMP_OFFSET = 8
SECTION_OFFSET = 12
SECTION_FORMAT = struct.Struct('IIIIIIIIII')
MODULE_ENTRY = struct.Struct('IIIII')
EXTENSION = '.icepack'

class ModuleInfo:

    def __init__(self, code=-1, filename=None, is_package=False):
        if False:
            i = 10
            return i + 15
        self.children = OrderedDict()
        self.code = code
        self.filename = filename
        self.child_offset = 0
        self.is_package = is_package
OBJECT_TYPE_NONE = 0
OBJECT_TYPE_NAMED_CONSTANT = 1
OBJECT_TYPE_INT32 = 3
OBJECT_TYPE_BIGINT = 4
OBJECT_TYPE_BYTES = 5
OBJECT_TYPE_STR = 6
OBJECT_TYPE_FLOAT = 7
OBJECT_TYPE_COMPLEX = 8
OBJECT_TYPE_TUPLE = 9
OBJECT_TYPE_CODE = 10
OBJECT_TYPE_FROZENSET = 11
OBJECT_FALSE = 1
OBJECT_TRUE = 257
OBJECT_ELLIPSIS = 513

def _float_equals(a, b):
    if False:
        return 10
    if math.isnan(a) and math.isnan(b):
        return True
    elif a == 0 and b == 0 and (math.copysign(1, a) != math.copysign(1, b)):
        return False
    else:
        return a == b

class PyObjectValue:
    """handles equality with slightly different semantics than normal
Python ==.  Disallows equality between conflicting types (e.g. 0 != 0.0 !=
False).  Allows NaN == NaN and +0.0 != -0.0 for both floats and components
of complex numbers"""

    def __init__(self, value):
        if False:
            while True:
                i = 10
        assert type(value) is not ObjectValue
        if type(value) == tuple:
            self.value = tuple((ObjectValue(v) for v in value))
        elif type(value) == frozenset:
            self.value = frozenset((ObjectValue(v) for v in value))
        else:
            self.value = value
        self.hash = hash(self.value) ^ hash(type(self.value))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ObjectValue(' + repr(self.value) + ')'

    def __hash__(self):
        if False:
            return 10
        return self.hash

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if type(other) is not ObjectValue:
            return False
        elif type(self.value) is not type(other.value):
            return False
        elif type(self.value) == float:
            return _float_equals(self.value, other.value)
        elif type(self.value) == complex:
            return _float_equals(self.value.real, other.value.real) and _float_equals(self.value.imag, other.value.imag)
        return self.value == other.value
if _pyice is None:
    ObjectValue = PyObjectValue

    class IcePackError(Exception):
        pass
else:
    ObjectValue = _pyice.CObjectValue
    IcePackError = _pyice.IcePackError

def _align_file(file, align=8):
    if False:
        print('Hello World!')
    len = file.tell()
    padding = (len + align - 1 & ~(align - 1)) - len
    file.write(b'\x00' * padding)

class _TypeTable:
    section_count = 1

    def add(self, value) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError(str(type(self)))

    def write(self, value, outfile):
        if False:
            while True:
                i = 10
        raise NotImplementedError(str(type(self)))

    def write_table(self, value, outfile):
        if False:
            while True:
                i = 10
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def add_const_to_table(table, value) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if value not in table:
            table[value] = len(table)
            return True
        return False

    @staticmethod
    def write_simple_table(table, outfile, get_bytes, align=4):
        if False:
            while True:
                i = 10
        values = [(y, x) for (x, y) in table.items()]
        values.sort()
        outfile.write(UINT.pack(len(values)))
        _align_file(outfile, align)
        for (_index, value) in values:
            outfile.write(get_bytes(value))

    @staticmethod
    def write_variable_len_table(table, outfile, get_bytes_and_len):
        if False:
            print('Hello World!')
        'writes the string table, in the format: (offset)* (utf8 value)*'
        values = [(y, x, get_bytes_and_len(x)) for (x, y) in table.items()]
        values.sort()
        outfile.write(UINT.pack(len(values)))
        offset = outfile.tell() + len(table) * 4
        for (_index, _value, (encoded, _length)) in values:
            outfile.write(UINT.pack(offset))
            offset += len(encoded) + 4
        for (_index, _value, (encoded, length)) in values:
            outfile.write(UINT.pack(length))
            outfile.write(encoded)

class _ConstantTable(_TypeTable):
    section_count = 0

    def __init__(self, id):
        if False:
            return 10
        self.id = id

    def add(self, value) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def write(self, value, outfile):
        if False:
            print('Hello World!')
        outfile.write(UINT.pack(self.id))

    def write_table(self, value, outfile):
        if False:
            return 10
        return ()

class _BoolTable(_TypeTable):
    section_count = 0

    def add(self, value) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def write(self, value, outfile):
        if False:
            i = 10
            return i + 15
        outfile.write(UINT.pack(OBJECT_TRUE if value else OBJECT_FALSE))

    def write_table(self, value, outfile):
        if False:
            for i in range(10):
                print('nop')
        return ()

class _SimpleTable(_TypeTable):
    object_type: int

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.table = {}

    def add(self, value) -> bool:
        if False:
            i = 10
            return i + 15
        return self.add_const_to_table(self.table, value)

    def write(self, value, outfile):
        if False:
            return 10
        str_id = self.table[value]
        outfile.write(UINT.pack(str_id << 8 | self.object_type))

    def __getitem__(self, value):
        if False:
            return 10
        return self.table[value]

class _IntTable(_TypeTable):
    section_count = 2

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.int_table = {}
        self.bigint_table = {}

    def add(self, value):
        if False:
            print('Hello World!')
        if -1 << 31 <= value <= (1 << 31) - 1:
            return self.add_const_to_table(self.int_table, value)
        else:
            return self.add_const_to_table(self.bigint_table, value)

    def write(self, value, outfile):
        if False:
            print('Hello World!')
        if -1 << 31 <= value <= (1 << 31) - 1:
            index = self.int_table[value]
            outfile.write(UINT.pack(OBJECT_TYPE_INT32 | index << 8))
        else:
            index = self.bigint_table[value]
            outfile.write(UINT.pack(OBJECT_TYPE_BIGINT | index << 8))

    @staticmethod
    def bigint_to_bytes_and_len(value):
        if False:
            return 10
        length = (value.bit_length() + 7) // 8
        try:
            res = value.to_bytes(length, 'little', signed=True)
        except OverflowError:
            res = value.to_bytes(length + 1, 'little', signed=True)
        return (res, len(res))

    def write_table(self, maker, outfile):
        if False:
            print('Hello World!')
        int_offset = outfile.tell()
        self.write_simple_table(self.int_table, outfile, INT.pack)
        bigint_offset = outfile.tell()
        self.write_variable_len_table(self.bigint_table, outfile, self.bigint_to_bytes_and_len)
        return (int_offset, bigint_offset)

class _BytesTable(_SimpleTable):
    object_type = OBJECT_TYPE_BYTES

    def write_table(self, maker, outfile):
        if False:
            while True:
                i = 10
        offset = outfile.tell()
        self.write_variable_len_table(self.table, outfile, lambda value: (value, len(value)))
        return (offset,)

class _StrTable(_SimpleTable):
    object_type = OBJECT_TYPE_STR

    def write_table(self, maker, outfile):
        if False:
            i = 10
            return i + 15
        offset = outfile.tell()
        self.write_variable_len_table(self.table, outfile, self.get_bytes_and_len)
        return (offset,)

    def get_bytes_and_len(self, value):
        if False:
            i = 10
            return i + 15
        res = value.encode('utf8', 'surrogatepass')
        return (res + b'\x00', len(res))

class _ObjectValueTable(_SimpleTable):

    def add(self, value):
        if False:
            i = 10
            return i + 15
        return super().add(ObjectValue(value))

    def write(self, value, outfile):
        if False:
            i = 10
            return i + 15
        super().write(ObjectValue(value), outfile)

    def __getitem__(self, value):
        if False:
            for i in range(10):
                print('nop')
        return super().__getitem__(ObjectValue(value))

class _FloatTable(_ObjectValueTable):
    object_type = OBJECT_TYPE_FLOAT

    def write_table(self, maker, outfile):
        if False:
            print('Hello World!')
        offset = outfile.tell()
        self.write_simple_table(self.table, outfile, lambda value: FLOAT.pack(value.value), 8)
        return (offset,)

class _ComplexTable(_ObjectValueTable):
    object_type = OBJECT_TYPE_COMPLEX

    def write_table(self, maker, outfile):
        if False:
            print('Hello World!')
        offset = outfile.tell()

        def writer(value):
            if False:
                print('Hello World!')
            return COMPLEX.pack(value.value.real, value.value.imag)
        self.write_simple_table(self.table, outfile, writer, 8)
        return (offset,)

class _SequenceTable(_ObjectValueTable):

    def __init__(self, object_type, maker):
        if False:
            print('Hello World!')
        super().__init__()
        self.maker = maker
        self.object_type = object_type

    def add(self, value) -> bool:
        if False:
            i = 10
            return i + 15
        if super().add(value):
            for x in value:
                self.maker.add_const(x)
            return True
        return False

    def write_table(self, maker, outfile):
        if False:
            print('Hello World!')
        table_offset = outfile.tell()
        values = [(y, x) for (x, y) in self.table.items()]
        values.sort()
        outfile.write(UINT.pack(len(values)))
        offset = outfile.tell() + len(self.table) * 4
        for (_index, value) in values:
            outfile.write(UINT.pack(offset))
            offset += 4 + len(value.value) * 4
        for (_index, value) in values:
            maker.write_array(value.value)
        return (table_offset,)

class _CodeTable(_TypeTable):

    def __init__(self, maker):
        if False:
            for i in range(10):
                print('nop')
        self.table = {}
        self.maker = maker

    @property
    def count(self):
        if False:
            i = 10
            return i + 15
        return len(self.table)

    def code_id(self, code):
        if False:
            print('Hello World!')
        "Python has very weird equality semantics around code objects, so that\nit doesn't compare the filename or the co_lnotab fields to determine if they're\nequal.  That results in oddities such as:\n\n>>> def f():\n...     x = 1\n...     x = 2\n...\n>>> g = f\n>>> def f():\n...     x = 1\n...\n...     x = 2\n...\n>>> f.__code__ == g.__code__\nTrue\n\nSo we always consider object identify to ensure we don't merge code objects\ntogether and end up with wrong filename or line number information.\n"
        return (id(code), code)

    def __getitem__(self, code):
        if False:
            return 10
        return self.table[self.code_id(code)]

    def write(self, value, outfile):
        if False:
            while True:
                i = 10
        code_id = self.code_id(value)
        outfile.write(UINT.pack(self.table[code_id] << 8 | OBJECT_TYPE_CODE))
        pass

    def add(self, value):
        if False:
            for i in range(10):
                print('nop')
        code_id = self.code_id(value)
        if code_id not in self.table:
            self.table[code_id] = len(self.table)
            self.maker.add_const(value.co_code)
            self.maker.enqueue_code(value)

    def write_table(self, maker, outfile):
        if False:
            while True:
                i = 10
        table_offset = outfile.tell()
        code_offsets = []
        outfile.write(INT.pack(len(self.table)))
        code_start = outfile.tell()
        outfile.truncate(code_start + 4 * len(self.table))
        outfile.seek(0, 2)
        codes = [(y, x) for ((_, x), y) in self.table.items()]
        codes.sort()
        for (_i, code) in codes:
            code_offsets.append(outfile.tell())
            header = CODE_FORMAT.pack(maker.bytes[code.co_code], code.co_argcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, code.co_firstlineno, maker.strs[code.co_name], maker.strs[code.co_filename], maker.bytes[code.co_lnotab], maker.get_tuple_id(code.co_cellvars), maker.get_tuple_id(code.co_freevars), maker.get_tuple_id(code.co_names), maker.get_tuple_id(code.co_varnames), maker.get_tuple_id(code.co_consts))
            outfile.write(header)
        outfile.seek(code_start)
        for offset in code_offsets:
            outfile.write(UINT.pack(offset))
        outfile.seek(0, io.SEEK_END)
        return (table_offset,)

class IceMaker:
    """Generates an ice pack from a set of modules and saves it to a file like
object."""

    def __init__(self, outfile):
        if False:
            print('Hello World!')
        'Creates a new IceMaker which will save the contents of the provided\nmodules to outfile which should be a seekable file-like object'
        self.outfile = outfile
        self.consts = set()
        self.queue = deque()
        self.modules = {}
        self.timestamp = 0
        self.codes = _CodeTable(self)
        self.strs = _StrTable()
        self.tuples = _SequenceTable(OBJECT_TYPE_TUPLE, self)
        self.bytes = _BytesTable()
        self.ints = _IntTable()
        self.floats = _FloatTable()
        self.complexes = _ComplexTable()
        self.frozensets = _SequenceTable(OBJECT_TYPE_FROZENSET, self)
        self.type_handlers = {type(None): _ConstantTable(0), type(Ellipsis): _ConstantTable(OBJECT_ELLIPSIS), bool: _BoolTable(), int: self.ints, bytes: self.bytes, str: self.strs, float: self.floats, complex: self.complexes, CodeType: self.codes, ObjectValue: None, tuple: self.tuples, frozenset: self.frozensets}
        self.empty_code = compile('', '', 'exec', dont_inherit=True)
        self.enqueue_code(self.empty_code)

    def add_module(self, code, name, filename, is_package=False, timestamp=0):
        if False:
            for i in range(10):
                print('nop')
        if timestamp > self.timestamp:
            self.timestamp = timestamp
        self.enqueue_code(code)
        self.modules[name] = ModuleInfo(self.codes[code], filename, is_package)
        self.add_const(filename)
        for name_part in name.split('.'):
            self.add_const(name_part)
        self.process()

    def enqueue_code(self, code):
        if False:
            while True:
                i = 10
        self.add_const(code)
        self.queue.append(code)

    def write_str(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.outfile.write(UINT.pack(self.strs[value]))

    def add_const(self, const):
        if False:
            i = 10
            return i + 15
        handler = self.type_handlers[type(const)]
        return handler.add(const)

    def process(self):
        if False:
            return 10
        while self.queue:
            code = self.queue.popleft()
            self.add_const(code.co_filename)
            self.add_const(code.co_name)
            self.add_const(code.co_lnotab)
            self.add_const(code.co_names)
            self.add_const(code.co_varnames)
            self.add_const(code.co_cellvars)
            self.add_const(code.co_freevars)
            self.add_const(code.co_consts)

    def write_object_value(self, value):
        if False:
            i = 10
            return i + 15
        if type(value.value) == tuple:
            tuple_id = self.tuples.table[value]
            self.outfile.write(UINT.pack(tuple_id << 8 | OBJECT_TYPE_TUPLE))
        elif type(value.value) == frozenset:
            set_id = self.frozensets.table[value]
            self.outfile.write(UINT.pack(set_id << 8 | OBJECT_TYPE_FROZENSET))
        else:
            self.write_reference(value.value)

    def write_reference(self, value):
        if False:
            print('Hello World!')
        if type(value) is ObjectValue:
            self.write_object_value(value)
        else:
            handler = self.type_handlers[type(value)]
            handler.write(value, self.outfile)

    def write_array(self, arr):
        if False:
            return 10
        self.outfile.write(UINT.pack(len(arr)))
        for value in arr:
            self.write_reference(value)

    def write_str_array(self, arr):
        if False:
            return 10
        self.outfile.write(UINT.pack(len(arr)))
        for value in arr:
            self.write_str(value)

    def write(self):
        if False:
            for i in range(10):
                print('nop')
        sections = [self.codes, self.strs, self.bytes, self.ints, self.floats, self.complexes, self.tuples, self.frozensets]
        sec_count = sum((section.section_count for section in sections)) + 1
        self.outfile.write(b'ICEPACK\x00')
        self.outfile.write(UINT.pack(self.timestamp))
        offset_start = self.outfile.tell()
        self.outfile.write(b'\x00\x00\x00\x00' * sec_count)
        offsets = []
        _align_file(self.outfile)
        offsets.append(self.outfile.tell())
        self.write_modules()
        for section in sections:
            _align_file(self.outfile)
            offsets.extend(section.write_table(self, self.outfile))
        self.outfile.seek(offset_start)
        for offset in offsets:
            self.outfile.write(UINT.pack(offset))
        self.outfile.seek(0, io.SEEK_END)

    def make_module_tree(self):
        if False:
            while True:
                i = 10
        tree = ModuleInfo()
        all_modules = list(self.modules.items())
        all_modules.sort()
        for (mod_name, mod_info) in all_modules:
            cur = tree
            names = mod_name.split('.')
            for name in names[:-1]:
                next = cur.children.get(name)
                if next is None:
                    ns_module = ModuleInfo(self.codes[self.empty_code], '', is_package=True)
                    next = cur.children[name] = ns_module
                cur = next
            cur.children[names[-1]] = mod_info
        self.calculate_module_relative_offsets(tree)
        return tree

    def calculate_module_relative_offsets(self, tree, offset=0):
        if False:
            i = 10
            return i + 15
        'Recurses through the tree and calculates the relative offsets for\nwhere their children will live'
        offset += 4 + len(tree.children) * MODULE_ENTRY.size
        for (_name, item) in tree.children.items():
            if item.children:
                item.child_offset = offset
                offset = self.calculate_module_relative_offsets(item, offset)
        return offset

    def write_modules(self):
        if False:
            while True:
                i = 10
        'We write out the module table as a sorted tree that we can binary\nsearch.  The format at each level is:\ncount, (name_offset, children_offset)*\nIf a level has no children 0 is written'
        self.write_module(self.make_module_tree(), self.outfile.tell())

    def write_module(self, tree, base_offset):
        if False:
            for i in range(10):
                print('nop')
        self.outfile.write(UINT.pack(len(tree.children)))
        for (name, item) in tree.children.items():
            self.write_str(name)
            if item.code == -1:
                self.outfile.write(UINT.pack(4294967295))
            else:
                self.outfile.write(UINT.pack(item.code))
            self.outfile.write(UINT.pack(1 if item.is_package else 0))
            self.write_str(item.filename)
            if item.child_offset == 0:
                self.outfile.write(b'\x00\x00\x00\x00')
            else:
                self.outfile.write(UINT.pack(base_offset + item.child_offset))
        for item in tree.children.values():
            if item.children:
                self.write_module(item, base_offset)

    def get_tuple_id(self, value):
        if False:
            i = 10
            return i + 15
        return self.tuples[value]

class PyIceBreaker:

    def __init__(self, icepack, base_dir=''):
        if False:
            i = 10
            return i + 15
        self.file = icepack
        self.base_dir = base_dir
        self.map = mmap.mmap(self.file.fileno(), length=0, access=mmap.ACCESS_READ)
        self.str_cache = {}
        self.bytes_cache = {}
        self.const_cache = {}
        self.code_cache = {}
        self.tuple_cache = {}
        self.int_cache = {}
        self.float_cache = {}
        self.complex_cache = {}
        self.bigint_cache = {}
        self.frozenset_cache = {}
        header = self.map[0:7]
        if header != b'ICEPACK':
            raise IcePackError('Invalid ice pack file: ' + repr(header))
        version = self.map[7]
        if version != 0:
            raise IcePackError('Unsupported IcePack version')
        ts_bytes = self.map[TIMESTAMP_OFFSET:SECTION_OFFSET]
        (self.timestamp,) = UINT.unpack(ts_bytes)
        sec_data = self.map[SECTION_OFFSET:SECTION_OFFSET + SECTION_FORMAT.size]
        sections = SECTION_FORMAT.unpack(sec_data)
        (self.modules, self.code, self.strings, self.bytes, self.ints, self.bigints, self.floats, self.complexes, self.tuples, self.frozensets) = sections
        (self.bytes_count,) = UINT.unpack(self.map[self.bytes:self.bytes + 4])
        (self.str_count,) = UINT.unpack(self.map[self.strings:self.strings + 4])
        (self.code_count,) = UINT.unpack(self.map[self.code:self.code + 4])
        (self.int_count,) = UINT.unpack(self.map[self.ints:self.ints + 4])
        (self.bigint_count,) = UINT.unpack(self.map[self.bigints:self.bigints + 4])
        (self.float_count,) = UINT.unpack(self.map[self.floats:self.floats + 4])
        (self.complex_count,) = UINT.unpack(self.map[self.complexes:self.complexes + 4])
        (self.tuple_count,) = UINT.unpack(self.map[self.tuples:self.tuples + 4])

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        self.file.close()

    def read_none(self, const):
        if False:
            for i in range(10):
                print('nop')
        if const != 0:
            raise ValueError('Invalid const value')
        return None

    def read_named_constant(self, const):
        if False:
            return 10
        if const <= 2:
            return (False, True, ...)[const]
        raise ValueError('Invalid const value')

    def read_bytes(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index > self.bytes_count:
            raise ValueError('Invalid bytes index')
        res = self.bytes_cache.get(index)
        if res is None:
            start = self.bytes + 4 + index * 4
            (location,) = UINT.unpack(self.map[start:start + 4])
            (len,) = UINT.unpack(self.map[location:location + 4])
            res = self.map[location + 4:location + 4 + len]
            self.bytes_cache[index] = res
        return res

    def read_str(self, index):
        if False:
            i = 10
            return i + 15
        if index > self.str_count:
            raise ValueError('Invalid str index ' + str(index))
        res = self.str_cache.get(index)
        if res is None:
            start = self.strings + 4 + index * 4
            (location,) = UINT.unpack(self.map[start:start + 4])
            (len,) = UINT.unpack(self.map[location:location + 4])
            utf8 = self.map[location + 4:location + 4 + len]
            self.str_cache[index] = res = utf8.decode('utf8', 'surrogatepass')
        return res

    def read_const(self, const):
        if False:
            while True:
                i = 10
        return _CONST_READERS[const & 255](self, const >> 8)

    def read_const_array(self, offset):
        if False:
            return 10
        (count,) = UINT.unpack(self.map[offset:offset + 4])
        o = offset + 4
        map = self.map
        return (self.read_const(UINT.unpack(map[o + i * 4:o + 4 + i * 4])[0]) for i in range(count))

    def read_tuple(self, index):
        if False:
            while True:
                i = 10
        if index > self.tuple_count:
            raise ValueError('Invalid tuple index')
        res = self.tuple_cache.get(index)
        if res is None:
            start = self.tuples + 4 + index * 4
            (location,) = UINT.unpack(self.map[start:start + 4])
            res = tuple(self.read_const_array(location))
            self.tuple_cache[index] = res
        return res

    def read_frozenset(self, index):
        if False:
            print('Hello World!')
        if index > self.tuple_count:
            raise ValueError('Invalid frozenset index')
        res = self.frozenset_cache.get(index)
        if res is None:
            start = self.frozensets + 4 + index * 4
            (location,) = UINT.unpack(self.map[start:start + 4])
            res = frozenset(self.read_const_array(location))
            self.frozenset_cache[index] = res
        return res

    def read_int(self, index):
        if False:
            print('Hello World!')
        if index > self.int_count:
            raise ValueError('Invalid int index')
        res = self.int_cache.get(index)
        if res is None:
            start = self.ints + 4 + index * INT.size
            (res,) = INT.unpack(self.map[start:start + 4])
            self.int_cache[index] = res
        return res

    def read_float(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index > self.float_count:
            raise ValueError('Invalid float index')
        res = self.float_cache.get(index)
        if res is None:
            start = self.floats + 4 + 4 + index * FLOAT.size
            (res,) = FLOAT.unpack(self.map[start:start + FLOAT.size])
            self.float_cache[index] = res
        return res

    def read_complex(self, index):
        if False:
            while True:
                i = 10
        if index > self.complex_count:
            raise ValueError('Invalid complex index')
        res = self.complex_cache.get(index)
        if res is None:
            start = self.complexes + 4 + 4 + index * COMPLEX.size
            (real, imag) = COMPLEX.unpack(self.map[start:start + COMPLEX.size])
            res = self.complex_cache[index] = complex(real, imag)
        return res

    def read_bigint(self, index):
        if False:
            while True:
                i = 10
        if index > self.bigint_count:
            raise ValueError('Invalid bigint index')
        res = self.bigint_cache.get(index)
        if res is None:
            start = self.bigints + 4 + index * 4
            (location,) = UINT.unpack(self.map[start:start + 4])
            (len,) = UINT.unpack(self.map[location:location + 4])
            bigint_bytes = self.map[location + 4:location + 4 + len]
            res = int.from_bytes(bigint_bytes, 'little', signed=True)
            self.bigint_cache[index] = res
        return res

    def read_code(self, index):
        if False:
            while True:
                i = 10
        if index > self.code_count:
            raise ValueError('Invalid code index')
        start = self.code + 4 + index * 4
        (location,) = UINT.unpack(self.map[start:start + 4])
        header = self.map[location:location + CODE_FORMAT.size]
        (bytes, argcount, kwonlyargcount, nlocals, stacksize, flags, firstlineno, name, filename, lnotab, cellvars, freevars, names, varnames, consts) = CODE_FORMAT.unpack(header)
        code = self.get_code_buffer(bytes)
        cellvars = self.read_tuple(cellvars)
        freevars = self.read_tuple(freevars)
        names = self.read_tuple(names)
        varnames = self.read_tuple(varnames)
        consts = self.read_tuple(consts)
        fixed_fn = path.join(self.base_dir, self.read_str(filename))
        return CodeType(argcount, kwonlyargcount, nlocals, stacksize, flags, code, consts, names, varnames, fixed_fn, self.read_str(name), firstlineno, self.read_bytes(lnotab), freevars, cellvars)

    def get_code_buffer(self, index):
        if False:
            i = 10
            return i + 15
        start = self.bytes + 4 + index * 4
        (location,) = UINT.unpack(self.map[start:start + 4])
        (len,) = UINT.unpack(self.map[location:location + 4])
        return memoryview(self.map)[location + 4:location + 4 + len]

    def find_module(self, name):
        if False:
            while True:
                i = 10
        'Finds a module in the module tree.  Returns a tuple of the code\nand a bool indicating if the module is a package'
        parts = name.split('.')
        cur = self.modules
        res = None
        for part in parts:
            if cur == 0:
                return None
            (count,) = UINT.unpack(self.map[cur:cur + 4])
            for i in range(count):
                start = cur + 4 + i * MODULE_ENTRY.size
                entry_bytes = self.map[start:start + MODULE_ENTRY.size]
                (iname, code, is_package, filename, children) = MODULE_ENTRY.unpack(entry_bytes)
                if self.read_str(iname) == part:
                    cur = children
                    res = code
                    break
            else:
                res = None
        if res is not None:
            return (self.read_code(res), is_package, self.read_str(filename))
        return None

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.file.close()
IceBreaker = PyIceBreaker
if _pyice is not None:

    class IceBreaker(_pyice.CIceBreaker):

        def __new__(cls, icepack, base_dir=''):
            if False:
                print('Hello World!')
            map = mmap.mmap(icepack.fileno(), length=0, access=mmap.ACCESS_READ)
            if isinstance(base_dir, str) and base_dir and (not base_dir.endswith(path.sep)):
                base_dir += path.sep
            self = super().__new__(cls, map, base_dir)
            self.map = map
            self.file = icepack
            return self

        def __enter__(self):
            if False:
                while True:
                    i = 10
            return self

        def __exit__(self, type, value, traceback):
            if False:
                for i in range(10):
                    print('nop')
            super().__exit__(type, value, traceback)
            self.file.close()
_CONST_READERS = {OBJECT_TYPE_NONE: PyIceBreaker.read_none, OBJECT_TYPE_NAMED_CONSTANT: PyIceBreaker.read_named_constant, OBJECT_TYPE_INT32: PyIceBreaker.read_int, OBJECT_TYPE_BIGINT: PyIceBreaker.read_bigint, OBJECT_TYPE_BYTES: PyIceBreaker.read_bytes, OBJECT_TYPE_STR: PyIceBreaker.read_str, OBJECT_TYPE_FLOAT: PyIceBreaker.read_float, OBJECT_TYPE_COMPLEX: PyIceBreaker.read_complex, OBJECT_TYPE_TUPLE: PyIceBreaker.read_tuple, OBJECT_TYPE_CODE: PyIceBreaker.read_code, OBJECT_TYPE_FROZENSET: PyIceBreaker.read_frozenset}

class Freezer:

    def __init__(self, output, modules, optimize, exclude, verbose):
        if False:
            for i in range(10):
                print('nop')
        self.modules = modules
        self.optimize = optimize
        self.exclude = exclude
        self.verbose = verbose
        self.outfile = open(output, 'wb')
        self.maker = IceMaker(self.outfile)

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.outfile.close()

    def build_file(self, basedir, fullpath):
        if False:
            print('Hello World!')
        (dir, file) = path.split(fullpath)
        if file == '__init__.py':
            relname = path.relpath(dir, basedir)
            module_name = relname.replace('/', '.').replace('\\', '.')
            is_package = True
        else:
            relname = path.relpath(path.splitext(fullpath)[0], basedir)
            module_name = relname.replace('/', '.').replace('\\', '.')
            is_package = False
        relfn = path.relpath(fullpath, basedir)
        for exclusion in self.exclude:
            if re.match(exclusion, relfn):
                if self.verbose:
                    print('Skipping', relfn)
                break
        else:
            if self.verbose:
                print('Including', module_name, 'from', relfn)
            with open(fullpath, 'rb') as inp:
                try:
                    bytes = inp.read()
                    compiled = compile(bytes, relfn, 'exec', dont_inherit=True, optimize=self.optimize)
                    timestamp = int(os.stat(fullpath).st_mtime)
                    self.maker.add_module(compiled, module_name, relfn, is_package, timestamp)
                except SyntaxError as se:
                    if self.verbose:
                        print('Ignoring module with error: ', relfn, se)

    def build_dir(self, dir):
        if False:
            for i in range(10):
                print('nop')
        for (dirpath, _dirnames, filenames) in os.walk(dir):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                fullpath = path.join(dirpath, filename)
                self.build_file(dir, fullpath)

    def freeze(self):
        if False:
            while True:
                i = 10
        start = time.time()
        for module in self.modules:
            if path.isdir(module):
                if self.verbose:
                    print('Including directory', module)
                self.build_dir(module)
            else:
                self.build_file(path.dirname(module), module)
            self.maker.write()
        end = time.time()
        print('IcePack built in', end - start, 'seconds')

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parser.parse_args()
    if not args.modules:
        print('No modules specified!')
        sys.exit(1)
    with Freezer(args.output, args.modules, args.optimize, args.exclude or (), args.verbose) as freezer:
        freezer.freeze()

class PyIceImporter:

    def __init__(self, import_path):
        if False:
            return 10
        self.path = import_path
        try:
            if EXTENSION + '/' in import_path:
                components = import_path.split(EXTENSION + '/')
                pack_name = components[0] + EXTENSION
                if path.isfile(pack_name):
                    self.disk_loc = components[1]
                    self.breaker = IceBreaker(open(pack_name, 'rb'), self.disk_loc)
                    return
        except IcePackError as e:
            print('failed to load ice pack (invalid)', e)
        except OSError as e:
            print('failed to load ice pack: ' + str(e), e)
        raise ImportError()

    def find_spec(self, fullname, target=None):
        if False:
            print('Hello World!')
        if '\x00' in fullname:
            return None
        mod_info = self.breaker.find_module(fullname)
        if mod_info is None:
            return None
        (mod, is_package, filename) = mod_info
        disk_loc = path.join(self.disk_loc, fullname.replace('.', '/'))
        if filename:
            file_path = path.join(self.disk_loc, filename)
            try:
                mtime = os.stat(file_path).st_mtime
                if int(mtime) > self.breaker.timestamp:
                    return None
            except OSError:
                pass
        else:
            file_path = None
        if is_package:
            search = [self.path, disk_loc]
        else:
            search = None
        loader = PyIceLoader(mod, self, file_path, is_package)
        spec = spec_from_file_location(fullname, file_path, loader=loader, submodule_search_locations=search)
        if not file_path:
            spec.has_location = False
        return spec

class PyIceLoader:

    def __init__(self, code, importer, filename, is_package):
        if False:
            for i in range(10):
                print('nop')
        self.code = code
        self.importer = importer
        self.path = filename
        self._is_package = is_package

    def create_module(self, spec):
        if False:
            return 10
        return None

    def exec_module(self, mod):
        if False:
            print('Hello World!')
        if self.path:
            mod.__file__ = self.path
        exec(self.code, mod.__dict__)

    def is_package(self, fullname):
        if False:
            i = 10
            return i + 15
        return self._is_package

    def get_code(self, x):
        if False:
            i = 10
            return i + 15
        return self.code

    def get_source(self, fullname):
        if False:
            print('Hello World!')
        if not self.path:
            return ''
        with open(self.path, 'rb') as f:
            return decode_source(f.read())

    def get_data(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Return the data from path as raw bytes.'
        with open(path, 'rb') as file:
            return file.read()

    def get_filename(self, fullname):
        if False:
            while True:
                i = 10
        return self.path

def install():
    if False:
        while True:
            i = 10
    sys.path_hooks.append(PyIceImporter)

def uninstall():
    if False:
        i = 10
        return i + 15
    sys.path_hooks.remove(PyIceImporter)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='\n    PyICE - Produces an icepack, a memory mappable pre-compiled modules.\n    ')
    parser.add_argument('--exclude', type=str, nargs='+', dest='exclude', help='Adds a module to be excluded in the icepack.')
    parser.add_argument('--optimize', default=-1, type=int, action='store', dest='optimize', help='The optimization level (-1, 0, 1 or 2).')
    parser.add_argument('--verbose', default=False, action='store_true', dest='verbose', help='Enable verbose output.')
    parser.add_argument('modules', nargs='*', help='Directories or files to be included in the IcePack.')
    parser.add_argument('--output', type=str, default='out' + EXTENSION, help='The destination filename')
    main()