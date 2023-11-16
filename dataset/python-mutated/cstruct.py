from __future__ import print_function
from builtins import zip
from functools import reduce
import struct
from future.utils import PY3
type_size = {}
size2type = {}
for t in ('B', 'H', 'I', 'Q'):
    s = struct.calcsize(t)
    type_size[t] = s * 8
    size2type[s * 8] = t
type_size['u08'] = size2type[8]
type_size['u16'] = size2type[16]
type_size['u32'] = size2type[32]
type_size['u64'] = size2type[64]

def fix_size(fields, wsize):
    if False:
        return 10
    out = []
    for (name, v) in fields:
        if v.endswith('s'):
            pass
        elif v == 'ptr':
            v = size2type[wsize]
        elif not v in type_size:
            raise ValueError('unknown Cstruct type', v)
        else:
            v = type_size[v]
        out.append((name, v))
    fields = out
    return fields

class Cstruct_Metaclass(type):

    def __new__(cls, name, bases, dct):
        if False:
            return 10
        o = super(Cstruct_Metaclass, cls).__new__(cls, name, bases, dct)
        o._packstring = o._packformat + ''.join((x[1] for x in o._fields))
        o._size = struct.calcsize(o._packstring)
        return o

class CStruct(object):
    _packformat = ''
    _fields = []

    @classmethod
    def _from_file(cls, f):
        if False:
            for i in range(10):
                print('nop')
        return cls(f.read(cls._size))

    def __init__(self, sex, wsize, *args, **kargs):
        if False:
            i = 10
            return i + 15
        if sex == 1:
            sex = '<'
        else:
            sex = '>'
        if self._packformat:
            sex = ''
        pstr = fix_size(self._fields, wsize)
        self._packstring = sex + self._packformat + ''.join((x[1] for x in pstr))
        self._size = struct.calcsize(self._packstring)
        self._names = [x[0] for x in self._fields]
        if kargs:
            self.__dict__.update(kargs)
        else:
            if args:
                s = args[0]
            else:
                s = b''
            s += b'\x00' * self._size
            s = s[:self._size]
            self._unpack(s)

    def _unpack(self, s):
        if False:
            while True:
                i = 10
        disas = struct.unpack(self._packstring, s)
        for (n, v) in zip(self._names, disas):
            setattr(self, n, v)

    def _pack(self):
        if False:
            return 10
        return struct.pack(self._packstring, *(getattr(self, x) for x in self._names))

    def _spack(self, superstruct, shift=0):
        if False:
            for i in range(10):
                print('nop')
        attr = []
        for name in self._names:
            s = getattr(self, name)
            if isinstance(s, CStruct):
                if s in superstruct:
                    s = reduce(lambda x, y: x + len(y), superstruct[:superstruct.index(s)], 0)
                    s += shift
                else:
                    raise Exception('%r is not a superstructure' % s)
            attr.append(s)
        return struct.pack(self._packstring, *attr)

    def _copy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(**self.__dict__)

    def __len__(self):
        if False:
            return 10
        return self._size

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if PY3:
            return repr(self)
        return self.__bytes__()

    def __bytes__(self):
        if False:
            return 10
        return self._pack()

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s=%s>' % (self.__class__.__name__, '/'.join((repr(getattr(self, x[0])) for x in self._fields)))

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return getattr(self, item)

    def _show(self):
        if False:
            print('Hello World!')
        print('##%s:' % self.__class__.__name__)
        fmt = '%%-%is = %%r' % max((len(x[0]) for x in self._fields))
        for (fn, ft) in self._fields:
            print(fmt % (fn, getattr(self, fn)))

class CStructStruct(object):

    def __init__(self, lst, shift=0):
        if False:
            for i in range(10):
                print('nop')
        self._lst = lst
        self._shift = shift

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._lst, attr)

    def __str__(self):
        if False:
            return 10
        if PY3:
            return repr(self)
        return self.__bytes__()

    def __bytes__(self):
        if False:
            i = 10
            return i + 15
        return b''.join((a if isinstance(a, bytes) else a._spack(self._lst, self._shift) for a in self._lst))