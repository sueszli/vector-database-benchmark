import ast
from typing import Any, Tuple

class Type:
    NULL = ord('0')
    NONE = ord('N')
    FALSE = ord('F')
    TRUE = ord('T')
    STOPITER = ord('S')
    ELLIPSIS = ord('.')
    INT = ord('i')
    INT64 = ord('I')
    FLOAT = ord('f')
    BINARY_FLOAT = ord('g')
    COMPLEX = ord('x')
    BINARY_COMPLEX = ord('y')
    LONG = ord('l')
    STRING = ord('s')
    INTERNED = ord('t')
    REF = ord('r')
    TUPLE = ord('(')
    LIST = ord('[')
    DICT = ord('{')
    CODE = ord('c')
    UNICODE = ord('u')
    UNKNOWN = ord('?')
    SET = ord('<')
    FROZENSET = ord('>')
    ASCII = ord('a')
    ASCII_INTERNED = ord('A')
    SMALL_TUPLE = ord(')')
    SHORT_ASCII = ord('z')
    SHORT_ASCII_INTERNED = ord('Z')
FLAG_REF = 128
NULL = object()
CO_FAST_LOCAL = 32
CO_FAST_CELL = 64
CO_FAST_FREE = 128

class Code:

    def __init__(self, **kwds: Any):
        if False:
            return 10
        self.__dict__.update(kwds)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Code(**{self.__dict__})'
    co_localsplusnames: Tuple[str]
    co_localspluskinds: Tuple[int]

    def get_localsplus_names(self, select_kind: int) -> Tuple[str, ...]:
        if False:
            i = 10
            return i + 15
        varnames: list[str] = []
        for (name, kind) in zip(self.co_localsplusnames, self.co_localspluskinds):
            if kind & select_kind:
                varnames.append(name)
        return tuple(varnames)

    @property
    def co_varnames(self) -> Tuple[str, ...]:
        if False:
            return 10
        return self.get_localsplus_names(CO_FAST_LOCAL)

    @property
    def co_cellvars(self) -> Tuple[str, ...]:
        if False:
            for i in range(10):
                print('nop')
        return self.get_localsplus_names(CO_FAST_CELL)

    @property
    def co_freevars(self) -> Tuple[str, ...]:
        if False:
            print('Hello World!')
        return self.get_localsplus_names(CO_FAST_FREE)

    @property
    def co_nlocals(self) -> int:
        if False:
            print('Hello World!')
        return len(self.co_varnames)

class Reader:

    def __init__(self, data: bytes):
        if False:
            for i in range(10):
                print('nop')
        self.data: bytes = data
        self.end: int = len(self.data)
        self.pos: int = 0
        self.refs: list[Any] = []
        self.level: int = 0

    def r_string(self, n: int) -> bytes:
        if False:
            i = 10
            return i + 15
        assert 0 <= n <= self.end - self.pos
        buf = self.data[self.pos:self.pos + n]
        self.pos += n
        return buf

    def r_byte(self) -> int:
        if False:
            return 10
        buf = self.r_string(1)
        return buf[0]

    def r_short(self) -> int:
        if False:
            i = 10
            return i + 15
        buf = self.r_string(2)
        x = buf[0]
        x |= buf[1] << 8
        x |= -(x & 1 << 15)
        return x

    def r_long(self) -> int:
        if False:
            print('Hello World!')
        buf = self.r_string(4)
        x = buf[0]
        x |= buf[1] << 8
        x |= buf[2] << 16
        x |= buf[3] << 24
        x |= -(x & 1 << 31)
        return x

    def r_long64(self) -> int:
        if False:
            print('Hello World!')
        buf = self.r_string(8)
        x = buf[0]
        x |= buf[1] << 8
        x |= buf[2] << 16
        x |= buf[3] << 24
        x |= buf[1] << 32
        x |= buf[1] << 40
        x |= buf[1] << 48
        x |= buf[1] << 56
        x |= -(x & 1 << 63)
        return x

    def r_PyLong(self) -> int:
        if False:
            while True:
                i = 10
        n = self.r_long()
        size = abs(n)
        x = 0
        for i in range(size):
            x |= self.r_short() << i * 15
        if n < 0:
            x = -x
        return x

    def r_float_bin(self) -> float:
        if False:
            while True:
                i = 10
        buf = self.r_string(8)
        import struct
        return struct.unpack('d', buf)[0]

    def r_float_str(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        n = self.r_byte()
        buf = self.r_string(n)
        return ast.literal_eval(buf.decode('ascii'))

    def r_ref_reserve(self, flag: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        if flag:
            idx = len(self.refs)
            self.refs.append(None)
            return idx
        else:
            return 0

    def r_ref_insert(self, obj: Any, idx: int, flag: int) -> Any:
        if False:
            print('Hello World!')
        if flag:
            self.refs[idx] = obj
        return obj

    def r_ref(self, obj: Any, flag: int) -> Any:
        if False:
            print('Hello World!')
        assert flag & FLAG_REF
        self.refs.append(obj)
        return obj

    def r_object(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        old_level = self.level
        try:
            return self._r_object()
        finally:
            self.level = old_level

    def _r_object(self) -> Any:
        if False:
            print('Hello World!')
        code = self.r_byte()
        flag = code & FLAG_REF
        type = code & ~FLAG_REF
        self.level += 1

        def R_REF(obj: Any) -> Any:
            if False:
                return 10
            if flag:
                obj = self.r_ref(obj, flag)
            return obj
        if type == Type.NULL:
            return NULL
        elif type == Type.NONE:
            return None
        elif type == Type.ELLIPSIS:
            return Ellipsis
        elif type == Type.FALSE:
            return False
        elif type == Type.TRUE:
            return True
        elif type == Type.INT:
            return R_REF(self.r_long())
        elif type == Type.INT64:
            return R_REF(self.r_long64())
        elif type == Type.LONG:
            return R_REF(self.r_PyLong())
        elif type == Type.FLOAT:
            return R_REF(self.r_float_str())
        elif type == Type.BINARY_FLOAT:
            return R_REF(self.r_float_bin())
        elif type == Type.COMPLEX:
            return R_REF(complex(self.r_float_str(), self.r_float_str()))
        elif type == Type.BINARY_COMPLEX:
            return R_REF(complex(self.r_float_bin(), self.r_float_bin()))
        elif type == Type.STRING:
            n = self.r_long()
            return R_REF(self.r_string(n))
        elif type == Type.ASCII_INTERNED or type == Type.ASCII:
            n = self.r_long()
            return R_REF(self.r_string(n).decode('ascii'))
        elif type == Type.SHORT_ASCII_INTERNED or type == Type.SHORT_ASCII:
            n = self.r_byte()
            return R_REF(self.r_string(n).decode('ascii'))
        elif type == Type.INTERNED or type == Type.UNICODE:
            n = self.r_long()
            return R_REF(self.r_string(n).decode('utf8', 'surrogatepass'))
        elif type == Type.SMALL_TUPLE:
            n = self.r_byte()
            idx = self.r_ref_reserve(flag)
            retval: Any = tuple((self.r_object() for _ in range(n)))
            self.r_ref_insert(retval, idx, flag)
            return retval
        elif type == Type.TUPLE:
            n = self.r_long()
            idx = self.r_ref_reserve(flag)
            retval = tuple((self.r_object() for _ in range(n)))
            self.r_ref_insert(retval, idx, flag)
            return retval
        elif type == Type.LIST:
            n = self.r_long()
            retval = R_REF([])
            for _ in range(n):
                retval.append(self.r_object())
            return retval
        elif type == Type.DICT:
            retval = R_REF({})
            while True:
                key = self.r_object()
                if key == NULL:
                    break
                val = self.r_object()
                retval[key] = val
            return retval
        elif type == Type.SET:
            n = self.r_long()
            retval = R_REF(set())
            for _ in range(n):
                v = self.r_object()
                retval.add(v)
            return retval
        elif type == Type.FROZENSET:
            n = self.r_long()
            s: set[Any] = set()
            idx = self.r_ref_reserve(flag)
            for _ in range(n):
                v = self.r_object()
                s.add(v)
            retval = frozenset(s)
            self.r_ref_insert(retval, idx, flag)
            return retval
        elif type == Type.CODE:
            retval = R_REF(Code())
            retval.co_argcount = self.r_long()
            retval.co_posonlyargcount = self.r_long()
            retval.co_kwonlyargcount = self.r_long()
            retval.co_stacksize = self.r_long()
            retval.co_flags = self.r_long()
            retval.co_code = self.r_object()
            retval.co_consts = self.r_object()
            retval.co_names = self.r_object()
            retval.co_localsplusnames = self.r_object()
            retval.co_localspluskinds = self.r_object()
            retval.co_filename = self.r_object()
            retval.co_name = self.r_object()
            retval.co_qualname = self.r_object()
            retval.co_firstlineno = self.r_long()
            retval.co_linetable = self.r_object()
            retval.co_exceptiontable = self.r_object()
            return retval
        elif type == Type.REF:
            n = self.r_long()
            retval = self.refs[n]
            assert retval is not None
            return retval
        else:
            breakpoint()
            raise AssertionError(f'Unknown type {type} {chr(type)!r}')

def loads(data: bytes) -> Any:
    if False:
        i = 10
        return i + 15
    assert isinstance(data, bytes)
    r = Reader(data)
    return r.r_object()

def main():
    if False:
        print('Hello World!')
    import marshal, pprint
    sample = {'foo': {(42, 'bar', 3.14)}}
    data = marshal.dumps(sample)
    retval = loads(data)
    assert retval == sample, retval
    sample = main.__code__
    data = marshal.dumps(sample)
    retval = loads(data)
    assert isinstance(retval, Code), retval
    pprint.pprint(retval.__dict__)
if __name__ == '__main__':
    main()