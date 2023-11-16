"""
all Python Standard Library objects (currently: CH 1-15 @ 2.7)
and some other common objects (i.e. numpy.ndarray)
"""
__all__ = ['registered', 'failures', 'succeeds']
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import sys
PY3 = hex(sys.hexversion) >= '0x30000f0'
if PY3:
    import queue as Queue
    import dbm as anydbm
else:
    import Queue
    import anydbm
    import sets
    import mutex
try:
    from cStringIO import StringIO
except ImportError:
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
import re
import array
import collections
import codecs
import struct
import datetime
import calendar
import weakref
import pprint
import decimal
import functools
import itertools
import operator
import tempfile
import shelve
import zlib
import gzip
import zipfile
import tarfile
import xdrlib
import csv
import hashlib
import hmac
import os
import logging
import optparse
import threading
import socket
import contextlib
try:
    import bz2
    import sqlite3
    if PY3:
        import dbm.ndbm as dbm
    else:
        import dbm
    HAS_ALL = True
except ImportError:
    HAS_ALL = False
try:
    HAS_CURSES = True
except ImportError:
    HAS_CURSES = False
try:
    import ctypes
    HAS_CTYPES = True
    IS_PYPY = not hasattr(ctypes, 'pythonapi')
except ImportError:
    HAS_CTYPES = False
    IS_PYPY = False

class _class:

    def _method(self):
        if False:
            i = 10
            return i + 15
        pass

class _class2:

    def __call__(self):
        if False:
            return 10
        pass
_instance2 = _class2()

class _newclass(object):

    def _method(self):
        if False:
            print('Hello World!')
        pass

class _newclass2(object):
    __slots__ = ['descriptor']

def _function(x):
    if False:
        print('Hello World!')
    yield x

def _function2():
    if False:
        while True:
            i = 10
    try:
        raise
    except:
        from sys import exc_info
        (e, er, tb) = exc_info()
        return (er, tb)
if HAS_CTYPES:

    class _Struct(ctypes.Structure):
        pass
    _Struct._fields_ = [('_field', ctypes.c_int), ('next', ctypes.POINTER(_Struct))]
(_filedescrip, _tempfile) = tempfile.mkstemp('r')
_tmpf = tempfile.TemporaryFile('w')
try:
    from collections import OrderedDict as odict
except ImportError:
    try:
        from ordereddict import OrderedDict as odict
    except ImportError:
        odict = dict
registered = d = odict()
failures = x = odict()
succeeds = a = odict()
a['BooleanType'] = bool(1)
a['BuiltinFunctionType'] = len
a['BuiltinMethodType'] = a['BuiltinFunctionType']
a['BytesType'] = _bytes = codecs.latin_1_encode('\x00')[0]
a['ClassType'] = _class
a['ComplexType'] = complex(1)
a['DictType'] = _dict = {}
a['DictionaryType'] = a['DictType']
a['FloatType'] = float(1)
a['FunctionType'] = _function
a['InstanceType'] = _instance = _class()
a['IntType'] = _int = int(1)
a['ListType'] = _list = []
a['NoneType'] = None
a['ObjectType'] = object()
a['StringType'] = _str = str(1)
a['TupleType'] = _tuple = ()
a['TypeType'] = type
if PY3:
    a['LongType'] = _int
    a['UnicodeType'] = _str
else:
    a['LongType'] = long(1)
    a['UnicodeType'] = unicode(1)
a['CopyrightType'] = copyright
a['ClassObjectType'] = _newclass
a['ClassInstanceType'] = _newclass()
a['SetType'] = _set = set()
a['FrozenSetType'] = frozenset()
a['ExceptionType'] = _exception = _function2()[0]
a['SREPatternType'] = _srepattern = re.compile('')
a['ArrayType'] = array.array('f')
a['DequeType'] = collections.deque([0])
a['DefaultDictType'] = collections.defaultdict(_function, _dict)
a['TZInfoType'] = datetime.tzinfo()
a['DateTimeType'] = datetime.datetime.today()
a['CalendarType'] = calendar.Calendar()
if not PY3:
    a['SetsType'] = sets.Set()
    a['ImmutableSetType'] = sets.ImmutableSet()
    a['MutexType'] = mutex.mutex()
a['DecimalType'] = decimal.Decimal(1)
a['CountType'] = itertools.count(0)
a['TarInfoType'] = tarfile.TarInfo()
a['LoggerType'] = logging.getLogger()
a['FormatterType'] = logging.Formatter()
a['FilterType'] = logging.Filter()
a['LogRecordType'] = logging.makeLogRecord(_dict)
a['OptionParserType'] = _oparser = optparse.OptionParser()
a['OptionGroupType'] = optparse.OptionGroup(_oparser, 'foo')
a['OptionType'] = optparse.Option('--foo')
if HAS_CTYPES:
    a['CCharType'] = _cchar = ctypes.c_char()
    a['CWCharType'] = ctypes.c_wchar()
    a['CByteType'] = ctypes.c_byte()
    a['CUByteType'] = ctypes.c_ubyte()
    a['CShortType'] = ctypes.c_short()
    a['CUShortType'] = ctypes.c_ushort()
    a['CIntType'] = ctypes.c_int()
    a['CUIntType'] = ctypes.c_uint()
    a['CLongType'] = ctypes.c_long()
    a['CULongType'] = ctypes.c_ulong()
    a['CLongLongType'] = ctypes.c_longlong()
    a['CULongLongType'] = ctypes.c_ulonglong()
    a['CFloatType'] = ctypes.c_float()
    a['CDoubleType'] = ctypes.c_double()
    a['CSizeTType'] = ctypes.c_size_t()
    a['CLibraryLoaderType'] = ctypes.cdll
    a['StructureType'] = _Struct
    if not IS_PYPY:
        a['BigEndianStructureType'] = ctypes.BigEndianStructure()
try:
    import fractions
    import number
    import io
    from io import StringIO as TextIO
    a['ByteArrayType'] = bytearray([1])
    a['FractionType'] = fractions.Fraction()
    a['NumberType'] = numbers.Number()
    a['IOBaseType'] = io.IOBase()
    a['RawIOBaseType'] = io.RawIOBase()
    a['TextIOBaseType'] = io.TextIOBase()
    a['BufferedIOBaseType'] = io.BufferedIOBase()
    a['UnicodeIOType'] = TextIO()
    a['LoggingAdapterType'] = logging.LoggingAdapter(_logger, _dict)
    if HAS_CTYPES:
        a['CBoolType'] = ctypes.c_bool(1)
        a['CLongDoubleType'] = ctypes.c_longdouble()
except ImportError:
    pass
try:
    import argparse
    a['OrderedDictType'] = collections.OrderedDict(_dict)
    a['CounterType'] = collections.Counter(_dict)
    if HAS_CTYPES:
        a['CSSizeTType'] = ctypes.c_ssize_t()
    a['NullHandlerType'] = logging.NullHandler()
    a['ArgParseFileType'] = argparse.FileType()
except (AttributeError, ImportError):
    pass
a['CodeType'] = compile('', '', 'exec')
a['DictProxyType'] = type.__dict__
a['DictProxyType2'] = _newclass.__dict__
a['EllipsisType'] = Ellipsis
a['ClosedFileType'] = open(os.devnull, 'wb', buffering=0).close()
a['GetSetDescriptorType'] = array.array.typecode
a['LambdaType'] = _lambda = lambda x: lambda y: x
a['MemberDescriptorType'] = _newclass2.descriptor
if not IS_PYPY:
    a['MemberDescriptorType2'] = datetime.timedelta.days
a['MethodType'] = _method = _class()._method
a['ModuleType'] = datetime
a['NotImplementedType'] = NotImplemented
a['SliceType'] = slice(1)
a['UnboundMethodType'] = _class._method
a['TextWrapperType'] = open(os.devnull, 'r')
a['BufferedRandomType'] = open(os.devnull, 'r+b')
a['BufferedReaderType'] = open(os.devnull, 'rb')
a['BufferedWriterType'] = open(os.devnull, 'wb')
try:
    from _pyio import open as _open
    a['PyTextWrapperType'] = _open(os.devnull, 'r', buffering=-1)
    a['PyBufferedRandomType'] = _open(os.devnull, 'r+b', buffering=-1)
    a['PyBufferedReaderType'] = _open(os.devnull, 'rb', buffering=-1)
    a['PyBufferedWriterType'] = _open(os.devnull, 'wb', buffering=-1)
except ImportError:
    pass
if PY3:
    d['CellType'] = _lambda(0).__closure__[0]
    a['XRangeType'] = _xrange = range(1)
else:
    d['CellType'] = _lambda(0).func_closure[0]
    a['XRangeType'] = _xrange = xrange(1)
if not IS_PYPY:
    d['MethodDescriptorType'] = type.__dict__['mro']
    d['WrapperDescriptorType'] = type.__repr__
    a['WrapperDescriptorType2'] = type.__dict__['__module__']
    d['ClassMethodDescriptorType'] = type.__dict__['__prepare__' if PY3 else 'mro']
if PY3 or IS_PYPY:
    _methodwrap = 1 .__lt__
else:
    _methodwrap = 1 .__cmp__
d['MethodWrapperType'] = _methodwrap
a['StaticMethodType'] = staticmethod(_method)
a['ClassMethodType'] = classmethod(_method)
a['PropertyType'] = property()
d['SuperType'] = super(Exception, _exception)
if PY3:
    _in = _bytes
else:
    _in = _str
a['InputType'] = _cstrI = StringIO(_in)
a['OutputType'] = _cstrO = StringIO()
a['WeakKeyDictionaryType'] = weakref.WeakKeyDictionary()
a['WeakValueDictionaryType'] = weakref.WeakValueDictionary()
a['ReferenceType'] = weakref.ref(_instance)
a['DeadReferenceType'] = weakref.ref(_class())
a['ProxyType'] = weakref.proxy(_instance)
a['DeadProxyType'] = weakref.proxy(_class())
a['CallableProxyType'] = weakref.proxy(_instance2)
a['DeadCallableProxyType'] = weakref.proxy(_class2())
a['QueueType'] = Queue.Queue()
d['PartialType'] = functools.partial(int, base=2)
if PY3:
    a['IzipType'] = zip('0', '1')
else:
    a['IzipType'] = itertools.izip('0', '1')
a['ChainType'] = itertools.chain('0', '1')
d['ItemGetterType'] = operator.itemgetter(0)
d['AttrGetterType'] = operator.attrgetter('__repr__')
if PY3:
    _fileW = _cstrO
else:
    _fileW = _tmpf
if HAS_ALL:
    a['ConnectionType'] = _conn = sqlite3.connect(':memory:')
    a['CursorType'] = _conn.cursor()
a['ShelveType'] = shelve.Shelf({})
if HAS_ALL:
    if hex(sys.hexversion) < '0x2070ef0' or PY3:
        a['BZ2FileType'] = bz2.BZ2File(os.devnull)
    a['BZ2CompressorType'] = bz2.BZ2Compressor()
    a['BZ2DecompressorType'] = bz2.BZ2Decompressor()
a['TarFileType'] = tarfile.open(fileobj=_fileW, mode='w')
a['DialectType'] = csv.get_dialect('excel')
a['PackerType'] = xdrlib.Packer()
a['LockType'] = threading.Lock()
a['RLockType'] = threading.RLock()
a['NamedLoggerType'] = _logger = logging.getLogger(__name__)
if PY3:
    a['SocketType'] = _socket = socket.socket()
    a['SocketPairType'] = socket.socketpair()[0]
else:
    a['SocketType'] = _socket = socket.socket()
    a['SocketPairType'] = _socket._sock
if PY3:
    a['GeneratorContextManagerType'] = contextlib.contextmanager(max)([1])
else:
    a['GeneratorContextManagerType'] = contextlib.GeneratorContextManager(max)
try:
    __IPYTHON__ is True
except NameError:
    a['QuitterType'] = quit
    d['ExitType'] = a['QuitterType']
try:
    from numpy import ufunc as _numpy_ufunc
    from numpy import array as _numpy_array
    from numpy import int32 as _numpy_int32
    a['NumpyUfuncType'] = _numpy_ufunc
    a['NumpyArrayType'] = _numpy_array
    a['NumpyInt32Type'] = _numpy_int32
except ImportError:
    pass
try:
    a['ProductType'] = itertools.product('0', '1')
    a['FileHandlerType'] = logging.FileHandler(os.devnull)
    a['RotatingFileHandlerType'] = logging.handlers.RotatingFileHandler(os.devnull)
    a['SocketHandlerType'] = logging.handlers.SocketHandler('localhost', 514)
    a['MemoryHandlerType'] = logging.handlers.MemoryHandler(1)
except AttributeError:
    pass
try:
    a['WeakSetType'] = weakref.WeakSet()
except AttributeError:
    pass
a['FileType'] = open(os.devnull, 'rb', buffering=0)
a['ListIteratorType'] = iter(_list)
a['TupleIteratorType'] = iter(_tuple)
a['XRangeIteratorType'] = iter(_xrange)
a['PrettyPrinterType'] = pprint.PrettyPrinter()
a['CycleType'] = itertools.cycle('0')
a['TemporaryFileType'] = _tmpf
a['GzipFileType'] = gzip.GzipFile(fileobj=_fileW)
a['StreamHandlerType'] = logging.StreamHandler()
try:
    a['PermutationsType'] = itertools.permutations('0')
    a['CombinationsType'] = itertools.combinations('0', 1)
except AttributeError:
    pass
try:
    a['RepeatType'] = itertools.repeat(0)
    a['CompressType'] = itertools.compress('0', [1])
except AttributeError:
    pass
x['GeneratorType'] = _generator = _function(1)
x['FrameType'] = _generator.gi_frame
x['TracebackType'] = _function2()[1]
x['SetIteratorType'] = iter(_set)
if PY3:
    x['DictionaryItemIteratorType'] = iter(type.__dict__.items())
    x['DictionaryKeyIteratorType'] = iter(type.__dict__.keys())
    x['DictionaryValueIteratorType'] = iter(type.__dict__.values())
else:
    x['DictionaryItemIteratorType'] = type.__dict__.iteritems()
    x['DictionaryKeyIteratorType'] = type.__dict__.iterkeys()
    x['DictionaryValueIteratorType'] = type.__dict__.itervalues()
x['StructType'] = struct.Struct('c')
x['CallableIteratorType'] = _srepattern.finditer('')
x['SREMatchType'] = _srepattern.match('')
x['SREScannerType'] = _srepattern.scanner('')
x['StreamReader'] = codecs.StreamReader(_cstrI)
if HAS_ALL:
    x['DbmType'] = dbm.open(_tempfile, 'n')
x['ZlibCompressType'] = zlib.compressobj()
x['ZlibDecompressType'] = zlib.decompressobj()
x['CSVReaderType'] = csv.reader(_cstrI)
x['CSVWriterType'] = csv.writer(_cstrO)
x['CSVDictReaderType'] = csv.DictReader(_cstrI)
x['CSVDictWriterType'] = csv.DictWriter(_cstrO, {})
x['HashType'] = hashlib.md5()
if hex(sys.hexversion) < '0x30800a1':
    x['HMACType'] = hmac.new(_in)
else:
    x['HMACType'] = hmac.new(_in, digestmod='md5')
if HAS_CURSES:
    pass
if HAS_CTYPES:
    x['CCharPType'] = ctypes.c_char_p()
    x['CWCharPType'] = ctypes.c_wchar_p()
    x['CVoidPType'] = ctypes.c_void_p()
    if sys.platform[:3] == 'win':
        x['CDLLType'] = _cdll = ctypes.cdll.msvcrt
    else:
        x['CDLLType'] = _cdll = ctypes.CDLL(None)
    if not IS_PYPY:
        x['PyDLLType'] = _pydll = ctypes.pythonapi
    x['FuncPtrType'] = _cdll._FuncPtr()
    x['CCharArrayType'] = ctypes.create_string_buffer(1)
    x['CWCharArrayType'] = ctypes.create_unicode_buffer(1)
    x['CParamType'] = ctypes.byref(_cchar)
    x['LPCCharType'] = ctypes.pointer(_cchar)
    x['LPCCharObjType'] = _lpchar = ctypes.POINTER(ctypes.c_char)
    x['NullPtrType'] = _lpchar()
    x['NullPyObjectType'] = ctypes.py_object()
    x['PyObjectType'] = ctypes.py_object(lambda : None)
    x['FieldType'] = _field = _Struct._field
    x['CFUNCTYPEType'] = _cfunc = ctypes.CFUNCTYPE(ctypes.c_char)
    x['CFunctionType'] = _cfunc(str)
try:
    x['MethodCallerType'] = operator.methodcaller('mro')
except AttributeError:
    pass
try:
    x['MemoryType'] = memoryview(_in)
    x['MemoryType2'] = memoryview(bytearray(_in))
    if PY3:
        x['DictItemsType'] = _dict.items()
        x['DictKeysType'] = _dict.keys()
        x['DictValuesType'] = _dict.values()
    else:
        x['DictItemsType'] = _dict.viewitems()
        x['DictKeysType'] = _dict.viewkeys()
        x['DictValuesType'] = _dict.viewvalues()
    x['RawTextHelpFormatterType'] = argparse.RawTextHelpFormatter('PROG')
    x['RawDescriptionHelpFormatterType'] = argparse.RawDescriptionHelpFormatter('PROG')
    x['ArgDefaultsHelpFormatterType'] = argparse.ArgumentDefaultsHelpFormatter('PROG')
except NameError:
    pass
try:
    x['CmpKeyType'] = _cmpkey = functools.cmp_to_key(_methodwrap)
    x['CmpKeyObjType'] = _cmpkey('0')
except AttributeError:
    pass
if PY3:
    x['BufferType'] = x['MemoryType']
else:
    x['BufferType'] = buffer('')
a.update(d)
if sys.platform[:3] == 'win':
    os.close(_filedescrip)
os.remove(_tempfile)