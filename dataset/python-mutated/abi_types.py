from vyper.exceptions import InvalidABIType
from vyper.utils import ceil32

class ABIType:

    def is_dynamic(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('ABIType.is_dynamic')

    def embedded_static_size(self):
        if False:
            i = 10
            return i + 15
        if self.is_dynamic():
            return 32
        return self.static_size()

    def embedded_dynamic_size_bound(self):
        if False:
            while True:
                i = 10
        if not self.is_dynamic():
            return 0
        return self.size_bound()

    def embedded_min_dynamic_size(self):
        if False:
            print('Hello World!')
        if not self.is_dynamic():
            return 0
        return self.min_size()

    def static_size(self):
        if False:
            return 10
        raise NotImplementedError('ABIType.static_size')

    def dynamic_size_bound(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_dynamic():
            return 0
        raise NotImplementedError('ABIType.dynamic_size_bound')

    def size_bound(self):
        if False:
            print('Hello World!')
        return self.static_size() + self.dynamic_size_bound()

    def min_size(self):
        if False:
            print('Hello World!')
        return self.static_size() + self.min_dynamic_size()

    def min_dynamic_size(self):
        if False:
            print('Hello World!')
        if not self.is_dynamic():
            return 0
        raise NotImplementedError('ABIType.min_dynamic_size')

    def selector_name(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('ABIType.selector_name')

    def is_complex_type(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('ABIType.is_complex_type')

    def __repr__(self):
        if False:
            return 10
        return str({type(self).__name__: vars(self)})

class ABI_GIntM(ABIType):

    def __init__(self, m_bits, signed):
        if False:
            i = 10
            return i + 15
        if not (0 < m_bits <= 256 and 0 == m_bits % 8):
            raise InvalidABIType('Invalid M provided for GIntM')
        self.m_bits = m_bits
        self.signed = signed

    def is_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def static_size(self):
        if False:
            return 10
        return 32

    def selector_name(self):
        if False:
            while True:
                i = 10
        return ('' if self.signed else 'u') + f'int{self.m_bits}'

    def is_complex_type(self):
        if False:
            for i in range(10):
                print('nop')
        return False

class ABI_Address(ABI_GIntM):

    def __init__(self):
        if False:
            print('Hello World!')
        return super().__init__(160, False)

    def selector_name(self):
        if False:
            while True:
                i = 10
        return 'address'

class ABI_Bool(ABI_GIntM):

    def __init__(self):
        if False:
            return 10
        return super().__init__(8, False)

    def selector_name(self):
        if False:
            print('Hello World!')
        return 'bool'

class ABI_FixedMxN(ABIType):

    def __init__(self, m_bits, n_places, signed):
        if False:
            i = 10
            return i + 15
        if not (0 < m_bits <= 256 and 0 == m_bits % 8):
            raise InvalidABIType('Invalid M for FixedMxN')
        if not (0 < n_places and n_places <= 80):
            raise InvalidABIType('Invalid N for FixedMxN')
        self.m_bits = m_bits
        self.n_places = n_places
        self.signed = signed

    def is_dynamic(self):
        if False:
            print('Hello World!')
        return False

    def static_size(self):
        if False:
            return 10
        return 32

    def selector_name(self):
        if False:
            return 10
        return ('' if self.signed else 'u') + f'fixed{self.m_bits}x{self.n_places}'

    def is_complex_type(self):
        if False:
            return 10
        return False

class ABI_BytesM(ABIType):

    def __init__(self, m_bytes):
        if False:
            return 10
        if not 0 < m_bytes <= 32:
            raise InvalidABIType('Invalid M for BytesM')
        self.m_bytes = m_bytes

    def is_dynamic(self):
        if False:
            while True:
                i = 10
        return False

    def static_size(self):
        if False:
            print('Hello World!')
        return 32

    def selector_name(self):
        if False:
            return 10
        return f'bytes{self.m_bytes}'

    def is_complex_type(self):
        if False:
            i = 10
            return i + 15
        return False

class ABI_Function(ABI_BytesM):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        return super().__init__(24)

    def selector_name(self):
        if False:
            return 10
        return 'function'

class ABI_StaticArray(ABIType):

    def __init__(self, subtyp, m_elems):
        if False:
            for i in range(10):
                print('nop')
        if not m_elems >= 0:
            raise InvalidABIType('Invalid M')
        self.subtyp = subtyp
        self.m_elems = m_elems

    def is_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        return self.subtyp.is_dynamic()

    def static_size(self):
        if False:
            print('Hello World!')
        return self.m_elems * self.subtyp.embedded_static_size()

    def dynamic_size_bound(self):
        if False:
            i = 10
            return i + 15
        return self.m_elems * self.subtyp.embedded_dynamic_size_bound()

    def min_dynamic_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.m_elems * self.subtyp.embedded_min_dynamic_size()

    def selector_name(self):
        if False:
            print('Hello World!')
        return f'{self.subtyp.selector_name()}[{self.m_elems}]'

    def is_complex_type(self):
        if False:
            while True:
                i = 10
        return True

class ABI_Bytes(ABIType):

    def __init__(self, bytes_bound):
        if False:
            return 10
        if not bytes_bound >= 0:
            raise InvalidABIType('Negative bytes_bound provided to ABI_Bytes')
        self.bytes_bound = bytes_bound

    def is_dynamic(self):
        if False:
            i = 10
            return i + 15
        return True

    def static_size(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def dynamic_size_bound(self):
        if False:
            for i in range(10):
                print('nop')
        return 32 + ceil32(self.bytes_bound)

    def min_dynamic_size(self):
        if False:
            i = 10
            return i + 15
        return 32

    def selector_name(self):
        if False:
            print('Hello World!')
        return 'bytes'

    def is_complex_type(self):
        if False:
            for i in range(10):
                print('nop')
        return False

class ABI_String(ABI_Bytes):

    def selector_name(self):
        if False:
            return 10
        return 'string'

class ABI_DynamicArray(ABIType):

    def __init__(self, subtyp, elems_bound):
        if False:
            print('Hello World!')
        if not elems_bound >= 0:
            raise InvalidABIType('Negative bound provided to DynamicArray')
        self.subtyp = subtyp
        self.elems_bound = elems_bound

    def is_dynamic(self):
        if False:
            print('Hello World!')
        return True

    def static_size(self):
        if False:
            print('Hello World!')
        return 0

    def dynamic_size_bound(self):
        if False:
            i = 10
            return i + 15
        subtyp_size = self.subtyp.embedded_static_size() + self.subtyp.embedded_dynamic_size_bound()
        return 32 + subtyp_size * self.elems_bound

    def min_dynamic_size(self):
        if False:
            print('Hello World!')
        return 32

    def selector_name(self):
        if False:
            print('Hello World!')
        return f'{self.subtyp.selector_name()}[]'

    def is_complex_type(self):
        if False:
            return 10
        return True

class ABI_Tuple(ABIType):

    def __init__(self, subtyps):
        if False:
            i = 10
            return i + 15
        self.subtyps = subtyps

    def is_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        return any([t.is_dynamic() for t in self.subtyps])

    def static_size(self):
        if False:
            return 10
        return sum([t.embedded_static_size() for t in self.subtyps])

    def dynamic_size_bound(self):
        if False:
            while True:
                i = 10
        return sum([t.embedded_dynamic_size_bound() for t in self.subtyps])

    def min_dynamic_size(self):
        if False:
            print('Hello World!')
        return sum([t.embedded_min_dynamic_size() for t in self.subtyps])

    def is_complex_type(self):
        if False:
            i = 10
            return i + 15
        return True

    def selector_name(self):
        if False:
            i = 10
            return i + 15
        return '(' + ','.join((t.selector_name() for t in self.subtyps)) + ')'