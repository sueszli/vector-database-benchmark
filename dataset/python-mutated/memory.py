from abc import ABCMeta, abstractmethod
from weakref import WeakValueDictionary
from dataclasses import dataclass, field
from ..core.state import EventSolver
from ..core.smtlib import Operators, ConstraintSet, arithmetic_simplify, SelectedSolver, TooManySolutions, BitVec, BitVecConstant, expression, issymbolic, Expression
from ..native.mappings import mmap, munmap
from ..utils.helpers import interval_intersection
from ..utils import config
import functools
import logging
from typing import Dict, Generator, Iterable, List, MutableMapping, Optional, Set, Union
logger = logging.getLogger(__name__)
consts = config.get_group('native')
consts.add('fast_crash', default=False, description='If True, throws a memory safety error if ANY concretization of a pointer is out of bounds. Otherwise, forks into valid and invalid memory access states.')

class MemoryException(Exception):
    """
    Memory exceptions
    """

    def __init__(self, message: str, address=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a memory exception.\n\n        :param message: exception message.\n        :param address: memory address where the exception occurred.\n        '
        self.address = address
        self.message = message
        if address is not None and (not issymbolic(address)):
            self.message += f' <{address:x}>'

    def __str__(self):
        if False:
            print('Hello World!')
        return self.message

class ConcretizeMemory(MemoryException):
    """
    Raised when a symbolic memory cell needs to be concretized.
    """

    def __init__(self, mem: 'Memory', address: Union[int, Expression], size: int, message: Optional[str]=None, policy: str='MINMAX'):
        if False:
            print('Hello World!')
        if message is None:
            self.message = f'Concretizing memory address {address} size {size}'
        else:
            self.message = message
        super().__init__(self.message, address)
        self.mem = mem
        self.address = address
        self.size = size
        self.policy = policy

class InvalidMemoryAccess(MemoryException):
    _message = 'Invalid memory access'

    def __init__(self, address, mode: str):
        if False:
            print('Hello World!')
        assert mode in 'rwx'
        message = f'{self._message} (mode:{mode})'
        super(InvalidMemoryAccess, self).__init__(message, address)
        self.mode = mode

class InvalidSymbolicMemoryAccess(InvalidMemoryAccess):
    _message = 'Invalid symbolic memory access'

    def __init__(self, address, mode: str, size, constraint):
        if False:
            while True:
                i = 10
        super(InvalidSymbolicMemoryAccess, self).__init__(address, mode)
        self.constraint = constraint
        self.size = size

def _normalize(c):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a byte-like value into a canonical byte (a value of type 'bytes' of len 1)\n\n    :param c:\n    :return:\n    "
    if isinstance(c, int):
        return bytes([c])
    elif isinstance(c, str):
        return bytes([ord(c)])
    else:
        return c

class Map(object, metaclass=ABCMeta):
    """
    A memory map.

    It represents a convex chunk of memory with a start and an end address.
    It may be implemented as an actual file mapping or as a StringIO/bytearray.

    >>>           ######################################
                  ^                                    ^
                start                                 end

    """

    def __init__(self, start: int, size: int, perms: str, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract memory map.\n\n        :param start: the first valid address.\n        :param size: the size of the map.\n        :param perms: the access permissions of the map (rwx).\n        '
        assert isinstance(start, int) and start >= 0, 'Invalid start address'
        assert isinstance(size, int) and size > 0, 'Invalid end address'
        super().__init__()
        self._start = start
        self._end = start + size
        self._set_perms(perms)
        self._name = name

    def _get_perms(self) -> str:
        if False:
            i = 10
            return i + 15
        'Gets the access permissions of the map.'
        return self._perms

    def _set_perms(self, perms: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Sets the access permissions of the map.\n\n        :param perms: the new permissions.\n        '
        assert isinstance(perms, str) and len(perms) <= 3 and (perms.strip() in ['', 'r', 'w', 'x', 'rw', 'r x', 'rx', 'rwx', 'wx'])
        self._perms = perms
    perms = property(_get_perms, _set_perms)

    def access_ok(self, access) -> bool:
        if False:
            return 10
        'Check if there is enough permissions for access'
        for c in access:
            if c not in self.perms:
                return False
        return True

    @property
    def start(self) -> int:
        if False:
            return 10
        return self._start

    @property
    def end(self) -> int:
        if False:
            while True:
                i = 10
        return self._end

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Returns the current size in bytes.'
        return self._end - self._start

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the string representation of the map mapping.\n\n        :rtype: str\n        '
        return f'<{self.__class__.__name__} 0x{self.start:016x}-0x{self.end:016x} {self.perms}>'

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Iterate all valid addresses\n        '
        return iter(range(self._start, self._end))

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        return self.start == other.start and self.end == other.end and (self.perms == other.perms) and (self.name == other.name)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if self.start != other.start:
            return self.start < other.start
        if self.end != other.end:
            return self.end < other.end
        if self.perms != other.perms:
            return self.perms < other.perms
        return self.name < other.name

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return object.__hash__(self)

    def _in_range(self, index) -> bool:
        if False:
            print('Hello World!')
        'Returns True if index is in range'
        if isinstance(index, slice):
            in_range = index.start < index.stop and index.start >= self.start and (index.stop <= self.end)
        else:
            in_range = index >= self.start and index <= self.end
        return in_range

    def _get_offset(self, index):
        if False:
            while True:
                i = 10
        '\n        Translates the index to the internal offsets.\n\n        self.start   -> 0\n        self.start+1 -> 1\n        ...\n        self.end     -> len(self)\n        '
        if not self._in_range(index):
            raise IndexError('Map index out of range')
        if isinstance(index, slice):
            index = slice(index.start - self.start, index.stop - self.start)
        else:
            index -= self.start
        return index

    @abstractmethod
    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        '\n        Reads a byte from an address or a sequence of bytes from a range of addresses\n\n        :param index: the address or slice where to obtain the bytes from.\n        :return: the character or sequence at the specified address.\n        :rtype: byte or array\n        '

    @abstractmethod
    def __setitem__(self, index, value):
        if False:
            while True:
                i = 10
        '\n        Writes a byte to an address or a sequence of bytes to a range of addresses\n\n        :param index: the address or slice where to put the data.\n        :param value: byte or sequence of bytes to put in this map.\n        '

    @abstractmethod
    def split(self, address):
        if False:
            for i in range(10):
                print('nop')
        '\n        Split the current map into two mappings\n\n        :param address: The address at which to split the Map.\n        '

class AnonMap(Map):
    """A concrete anonymous memory map"""

    def __init__(self, start: int, size: int, perms: str, data_init=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a concrete anonymous memory map.\n\n        :param start: the first valid address of the map.\n        :param size: the size of the map.\n        :param perms: the access permissions of the map.\n        :param data_init: the data to initialize the map.\n        '
        super().__init__(start, size, perms, name)
        self._data = bytearray(size)
        if data_init is not None:
            assert len(data_init) <= size, 'More initial data than reserved memory'
            if isinstance(data_init[0], int):
                self._data[0:len(data_init)] = data_init
            else:
                self._data[0:len(data_init)] = [ord(s) for s in data_init]

    def __reduce__(self):
        if False:
            return 10
        return (self.__class__, (self.start, len(self), self.perms, self._data, self.name))

    def split(self, address):
        if False:
            print('Hello World!')
        if address <= self.start:
            return (None, self)
        if address >= self.end:
            return (self, None)
        assert address > self.start and address < self.end
        head = AnonMap(self.start, address - self.start, self.perms, self[self.start:address])
        tail = AnonMap(address, self.end - address, self.perms, self[address:self.end])
        return (head, tail)

    def __setitem__(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        assert not isinstance(index, slice) or len(value) == index.stop - index.start
        index = self._get_offset(index)
        if issymbolic(value[0]) and isinstance(self._data, bytearray):
            self._data = [Operators.ORD(b) for b in self._data]
        if isinstance(index, slice):
            if not isinstance(value[0], int):
                value = [Operators.ORD(n) for n in value]
            self._data[index] = value
        else:
            self._data[index] = Operators.ORD(value)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        index = self._get_offset(index)
        if isinstance(index, slice):
            return [Operators.CHR(i) for i in self._data[index]]
        return Operators.CHR(self._data[index])

class ArrayMap(Map):

    def __init__(self, start: int, size: int, perms: str, index_bits, backing_array=None, name=None):
        if False:
            print('Hello World!')
        super().__init__(start, size, perms, name)
        if name is None:
            name = 'ArrayMap_{:x}'.format(start)
        if backing_array is not None:
            self._array = backing_array
        else:
            self._array = expression.ArrayProxy(array=expression.ArrayVariable(index_bits=index_bits, index_max=size, value_bits=8, name=name))

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.__class__, (self.start, len(self), self._perms, self._array.index_bits, self._array, self._array.name))

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self._array[key] = value

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self._array[key]

    def split(self, address: int):
        if False:
            for i in range(10):
                print('nop')
        if address <= self.start:
            return (None, self)
        if address >= self.end:
            return (self, None)
        assert self.start < address < self.end
        (index_bits, value_bits) = (self._array.index_bits, self._array.value_bits)
        (left_size, right_size) = (address - self.start, self.end - address)
        (left_name, right_name) = ['{}_{:d}'.format(self._array.name, i) for i in range(2)]
        head_arr = expression.ArrayProxy(array=expression.ArrayVariable(index_bits=index_bits, index_max=left_size, value_bits=value_bits, name=left_name))
        tail_arr = expression.ArrayProxy(array=expression.ArrayVariable(index_bits=index_bits, index_max=right_size, value_bits=value_bits, name=right_name))
        head = ArrayMap(self.start, left_size, self.perms, index_bits, head_arr, left_name)
        tail = ArrayMap(address, right_size, self.perms, index_bits, tail_arr, right_name)
        return (head, tail)

class FileMap(Map):
    """
    A file map.

    A  file is mapped in multiples of the page size.  For a file that is not a
    multiple of the page size, the remaining memory is zeroed when mapped, and
    writes to that region are not written out to the file. The effect of
    changing the size of the underlying file of a mapping on the pages that
    correspond to added or removed regions of the file is unspecified.
    """

    def __init__(self, addr: int, size: int, perms: str, filename: str, offset: int=0, overlay=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a map of memory  initialized with the content of filename.\n\n        :param addr: the first valid address of the file map.\n        :param size: the size of the file map.\n        :param perms: the access permissions of the file map.\n        :param filename: the file to map in memory.\n        :param offset: the offset into the file where to start the mapping.                 This offset must be a multiple of pagebitsize.\n        '
        super().__init__(addr, size, perms)
        assert isinstance(offset, int)
        assert offset >= 0
        self._filename = filename
        self._offset = offset
        with open(filename, 'r') as fileobject:
            fileobject.seek(0, 2)
            file_size = fileobject.tell()
            self._mapped_size = min(size, file_size - offset)
            self._data = mmap(fileobject.fileno(), offset, self._mapped_size)
        if overlay is not None:
            self._overlay = dict(overlay)
        else:
            self._overlay = dict()

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.__class__, (self.start, len(self), self.perms, self._filename, self._offset, self._overlay))

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_data'):
            munmap(self._data, self._mapped_size)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<{self.__class__.__name__} [{self._filename}+{self._offset:x}] 0x{self.start:016x}-0x{self.end:016x} {self.perms}>'

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        assert not isinstance(index, slice) or len(value) == index.stop - index.start
        index = self._get_offset(index)
        if isinstance(index, slice):
            for i in range(index.stop - index.start):
                self._overlay[index.start + i] = value[i]
        else:
            self._overlay[index] = value

    def __getitem__(self, index):
        if False:
            return 10

        def get_byte_at_offset(offset):
            if False:
                for i in range(10):
                    print('nop')
            if offset in self._overlay:
                return _normalize(self._overlay[offset])
            else:
                if offset >= self._mapped_size:
                    return b'\x00'
                return _normalize(self._data[offset])
        index = self._get_offset(index)
        if isinstance(index, slice):
            result = []
            for i in range(index.start, index.stop):
                result.append(get_byte_at_offset(i))
            return result
        else:
            return get_byte_at_offset(index)

    def split(self, address: int):
        if False:
            for i in range(10):
                print('nop')
        if address <= self.start:
            return (None, self)
        if address >= self.end:
            return (self, None)
        assert self.start < address <= self.end
        head = COWMap(self, size=address - self.start)
        tail = COWMap(self, offset=address - self.start)
        return (head, tail)

class COWMap(Map):
    """
    Copy-on-write based map.
    """

    def __init__(self, parent: Map, offset: int=0, perms: Optional[str]=None, size=None, **kwargs):
        if False:
            return 10
        '\n        A copy on write copy of parent. Writes to the parent after a copy on\n        write are unspecified.\n\n        :param parent: the parent map.\n        :param offset: an offset within the parent map from where to create the new map.\n        :param perms: Permissions on new mapping, or None if inheriting.\n        :param size: the size of the new map or max.\n        '
        assert isinstance(parent, Map)
        assert offset >= 0 and offset < len(parent)
        if size is None:
            size = len(parent) - offset
        assert parent.start + offset + size <= parent.end
        if perms is None:
            perms = parent.perms
        super().__init__(parent.start + offset, size, perms, **kwargs)
        self._parent = parent
        self._parent.__setitem__ = False
        self._cow: Dict = {}

    def __setitem__(self, index, value):
        if False:
            return 10
        assert self._in_range(index)
        if isinstance(index, slice):
            for i in range(index.stop - index.start):
                self._cow[index.start + i] = _normalize(value[i])
        else:
            self._cow[index] = _normalize(value)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        assert self._in_range(index)
        if isinstance(index, slice):
            result = []
            for i in range(index.start, index.stop):
                c = self._cow.get(i, self._parent[i])
                result.append(_normalize(c))
            return result
        else:
            return _normalize(self._cow.get(index, self._parent[index]))

    def split(self, address: int):
        if False:
            i = 10
            return i + 15
        if address <= self.start:
            return (None, self)
        if address >= self.end:
            return (self, None)
        assert address > self.start and address < self.end
        head = COWMap(self, size=address - self.start)
        tail = COWMap(self, offset=address - self.start)
        return (head, tail)

class StubCPU:

    def _publish(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return None

@dataclass
class ProcSelfMapInfo:
    start: int
    end: int
    rwx_perms: str
    shared_perms: str = '-'
    offset: int = 0
    device: str = '00:00'
    inode: int = 0
    pathname: str = ''
    perms: str = field(init=False)

    def __post_init__(self):
        if False:
            return 10
        self.perms = self.rwx_perms.replace(' ', '-') + self.shared_perms
        if self.pathname == 'stack':
            self.pathname = '[' + self.pathname + ']'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.address} {self.perms:>4s} {self.offset:08x} {self.device} {self.inode:9} {self.pathname}'

    @property
    def address(self):
        if False:
            print('Hello World!')
        return f'{self.start:016x}-{self.end:016x}'

class Memory(object, metaclass=ABCMeta):
    """
    The memory manager.
    This class handles all virtual memory mappings and symbolic chunks.
    """

    def __init__(self, maps: Optional[Iterable[Map]]=None, cpu=StubCPU()):
        if False:
            i = 10
            return i + 15
        '\n        Builds a memory manager.\n        '
        super().__init__()
        if maps is None:
            self._maps: Set[Map] = set()
        else:
            self._maps = set(maps)
        self.cpu = cpu
        self._page2map: MutableMapping[int, Map] = WeakValueDictionary()
        self._recording_stack: List = []
        self._solver = EventSolver()
        for m in self._maps:
            for i in range(self._page(m.start), self._page(m.end)):
                assert i not in self._page2map
                self._page2map[i] = m

    def __reduce__(self):
        if False:
            return 10
        return (self.__class__, (self._maps, self.cpu))

    @property
    @abstractmethod
    def memory_bit_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return 32

    @property
    @abstractmethod
    def page_bit_size(self) -> int:
        if False:
            while True:
                i = 10
        return 12

    @property
    def memory_size(self) -> int:
        if False:
            print('Hello World!')
        return 1 << self.memory_bit_size

    @property
    def page_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1 << self.page_bit_size

    @property
    def memory_mask(self) -> int:
        if False:
            print('Hello World!')
        return self.memory_size - 1

    @property
    def page_mask(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.page_size - 1

    @property
    def maps(self) -> Set[Map]:
        if False:
            print('Hello World!')
        return self._maps

    def _ceil(self, address) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the smallest page boundary value not less than the address.\n        :param address: the address to calculate its ceil.\n        :return: the ceil of C{address}.\n        '
        return address - 1 + self.page_size & ~self.page_mask & self.memory_mask

    def _floor(self, address) -> int:
        if False:
            return 10
        '\n        Returns largest page boundary value not greater than the address.\n\n        :param address: the address to calculate its floor.\n        :return: the floor of C{address}.\n        '
        return address & ~self.page_mask

    def _page(self, address) -> int:
        if False:
            while True:
                i = 10
        '\n        Calculates the page number of an address.\n\n        :param address: the address to calculate its page number.\n        :return: the page number of address.\n        '
        return address >> self.page_bit_size

    def _search(self, size, start=None, counter=0) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Recursively searches the address space for enough free space to allocate C{size} bytes.\n\n        :param size: the size in bytes to allocate.\n        :param start: an address from where to start the search.\n        :param counter: internal parameter to know if all the memory was already scanned.\n        :return: the address of an available space to map C{size} bytes.\n        :raises MemoryException: if there is no space available to allocate the desired memory.\n\n\n        todo: Document what happens when you try to allocate something that goes round the address 32/64 bit representation.\n        '
        assert size & self.page_mask == 0
        if start is None:
            end = {32: 4160749568, 64: 140737488355328}[self.memory_bit_size]
            start = end - size
        else:
            if start > self.memory_size - size:
                start = self.memory_size - size
            end = start + size
        consecutive_free = 0
        for p in range(self._page(end - 1), -1, -1):
            if p not in self._page2map:
                consecutive_free += 4096
            else:
                consecutive_free = 0
            if consecutive_free >= size:
                return p << self.page_bit_size
            counter += 1
            if counter >= self.memory_size // self.page_size:
                raise MemoryException('Not enough memory')
        return self._search(size, self.memory_size - size, counter)

    def mmapFile(self, addr, size, perms, filename, offset=0):
        if False:
            print('Hello World!')
        "\n        Creates a new file mapping in the memory address space.\n\n        :param addr: the starting address (took as hint). If C{addr} is C{0} the first big enough\n                     chunk of memory will be selected as starting address.\n        :param size: the contents of a file mapping are initialized using C{size} bytes starting\n                     at offset C{offset} in the file C{filename}.\n        :param perms: the access permissions to this memory.\n        :param filename: the pathname to the file to map.\n        :param offset: the contents of a file mapping are initialized using C{size} bytes starting\n                      at offset C{offset} in the file C{filename}.\n        :return: the starting address where the file was mapped.\n        :rtype: int\n        :raises error:\n                   - 'Address shall be concrete' if C{addr} is not an integer number.\n                   - 'Address too big' if C{addr} goes beyond the limit of the memory.\n                   - 'Map already used' if the piece of memory starting in C{addr} and with length C{size} isn't free.\n        "
        assert addr is None or isinstance(addr, int), 'Address shall be concrete'
        assert size > 0
        self.cpu._publish('will_map_memory', addr, size, perms, filename, offset)
        if addr is not None:
            assert addr < self.memory_size, 'Address too big'
            addr = self._floor(addr)
        size = self._ceil(size)
        addr = self._search(size, addr)
        for i in range(self._page(addr), self._page(addr + size)):
            assert i not in self._page2map, 'Map already used'
        m = FileMap(addr, size, perms, filename, offset)
        self._add(m)
        logger.debug(f'New file-memory map @{addr:#x} size:{size:#x}')
        self.cpu._publish('did_map_memory', addr, size, perms, filename, offset, addr)
        return addr

    def mmap(self, addr, size, perms, data_init=None, name=None):
        if False:
            while True:
                i = 10
        "\n        Creates a new mapping in the memory address space.\n\n        :param addr: the starting address (took as hint). If C{addr} is C{0} the first big enough\n                     chunk of memory will be selected as starting address.\n        :param size: the length of the mapping.\n        :param perms: the access permissions to this memory.\n        :param data_init: optional data to initialize this memory.\n        :param name: optional name to give to this mapping\n        :return: the starting address where the memory was mapped.\n        :raises error:\n                   - 'Address shall be concrete' if C{addr} is not an integer number.\n                   - 'Address too big' if C{addr} goes beyond the limit of the memory.\n                   - 'Map already used' if the piece of memory starting in C{addr} and with length C{size} isn't free.\n        :rtype: int\n\n        "
        assert addr is None or isinstance(addr, int), 'Address shall be concrete'
        self.cpu._publish('will_map_memory', addr, size, perms, None, None)
        if addr is not None:
            assert addr < self.memory_size, 'Address too big'
            addr = self._floor(addr)
        size = self._ceil(size)
        addr = self._search(size, addr)
        for i in range(self._page(addr), self._page(addr + size)):
            assert i not in self._page2map, 'Map already used'
        m = AnonMap(start=addr, size=size, perms=perms, data_init=data_init, name=name)
        self._add(m)
        logger.debug(f'New memory map @{addr:#x} size:{size:#x}')
        self.cpu._publish('did_map_memory', addr, size, perms, None, None, addr)
        return addr

    def _add(self, m: Map) -> None:
        if False:
            while True:
                i = 10
        assert isinstance(m, Map)
        assert m not in self._maps
        assert m.start & self.page_mask == 0
        assert m.end & self.page_mask == 0
        self._maps.add(m)
        for i in range(self._page(m.start), self._page(m.end)):
            self._page2map[i] = m

    def _del(self, m: Map) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(m, Map)
        assert m in self._maps
        for p in range(self._page(m.start), self._page(m.end)):
            del self._page2map[p]
        self._maps.remove(m)

    def map_containing(self, address: int) -> Map:
        if False:
            print('Hello World!')
        '\n        Returns the L{MMap} object containing the address.\n\n        :param address: the address to obtain its mapping.\n        :rtype: L{MMap}\n\n        @todo: symbolic address\n        '
        page_offset = self._page(address)
        if page_offset not in self._page2map:
            raise MemoryException('Page not mapped', address)
        return self._page2map[page_offset]

    def mappings(self):
        if False:
            while True:
                i = 10
        '\n        Returns a sorted list of all the mappings for this memory.\n\n        :return: a list of mappings.\n        :rtype: list\n        '
        result = []
        for m in self.maps:
            if isinstance(m, AnonMap):
                result.append((m.start, m.end, m.perms, 0, ''))
            elif isinstance(m, FileMap):
                result.append((m.start, m.end, m.perms, m._offset, m._filename))
            else:
                result.append((m.start, m.end, m.perms, 0, m.name))
        return sorted(result)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join([f"{start:016x}-{end:016x} {p:>4s} {offset:08x} {name or ''}" for (start, end, p, offset, name) in self.mappings()])

    def proc_self_mappings(self) -> List[ProcSelfMapInfo]:
        if False:
            print('Hello World!')
        '\n        Returns a sorted list of all the mappings for this memory for /proc/self/maps.\n        Device, inode, and private/shared permissions are unsupported.\n        Stack is the only memory section supported in the memory map (heap, vdso, etc.)\n        are unsupported.\n        Pathname is substituted by filename\n\n        :return: a list of mappings.\n        '
        result = []
        for m in self.maps:
            if isinstance(m, AnonMap):
                if m.name is not None:
                    result.append(ProcSelfMapInfo(m.start, m.end, rwx_perms=m.perms, pathname=m.name))
                else:
                    result.append(ProcSelfMapInfo(m.start, m.end, rwx_perms=m.perms))
            elif isinstance(m, FileMap):
                result.append(ProcSelfMapInfo(m.start, m.end, rwx_perms=m.perms, offset=m._offset, pathname=m._filename))
            else:
                result.append(ProcSelfMapInfo(m.start, m.end, rwx_perms=m.perms, pathname=m.name))
        return sorted(result, key=lambda m: m.start)

    @property
    def __proc_self__(self):
        if False:
            return 10
        self.proc_self_mappings()
        return '\n'.join([f'{m}' for m in self.proc_self_mappings()])

    def _maps_in_range(self, start: int, end: int) -> Generator[Map, None, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates the list of maps that overlaps with the range [start:end]\n        '
        addr = start
        while addr < end:
            if addr not in self:
                addr += self.page_size
            else:
                m = self._page2map[self._page(addr)]
                yield m
                addr = m.end

    def munmap(self, start, size):
        if False:
            return 10
        '\n        Deletes the mappings for the specified address range and causes further\n        references to addresses within the range to generate invalid memory\n        references.\n\n        :param start: the starting address to delete.\n        :param size: the length of the unmapping.\n        '
        start = self._floor(start)
        end = self._ceil(start + size)
        self.cpu._publish('will_unmap_memory', start, size)
        for m in self._maps_in_range(start, end):
            self._del(m)
            (head, tail) = m.split(start)
            (middle, tail) = tail.split(end)
            assert middle is not None
            if head:
                self._add(head)
            if tail:
                self._add(tail)
        self.cpu._publish('did_unmap_memory', start, size)
        logger.debug(f'Unmapped memory @{start:#x} size:{size:#x}')

    def mprotect(self, start, size, perms):
        if False:
            return 10
        assert size > 0
        start = self._floor(start)
        end = self._ceil(start + size)
        self.cpu._publish('will_protect_memory', start, size, perms)
        for m in self._maps_in_range(start, end):
            self._del(m)
            (head, tail) = m.split(start)
            (middle, tail) = tail.split(end)
            assert middle is not None
            middle.perms = perms
            self._add(middle)
            if head:
                self._add(head)
            if tail:
                self._add(tail)
        self.cpu._publish('did_protect_memory', start, size, perms)

    def __contains__(self, address):
        if False:
            return 10
        return self._page(address) in self._page2map

    def perms(self, index):
        if False:
            return 10
        if isinstance(index, slice):
            raise NotImplementedError('No perms for slices')
        else:
            return self.map_containing(index).perms

    def max_exec_size(self, addr, max_size):
        if False:
            return 10
        '\n        Finds maximum executable memory size\n        starting from addr and up to max_size.\n\n        :param addr: the starting address.\n        :param size: the maximum size.\n        :param access: the wished access.\n        '
        size = 0
        max_addr = addr + max_size
        while addr < max_addr:
            if addr not in self:
                return size
            m = self.map_containing(addr)
            if not m.access_ok('x'):
                return size
            size += m.end - addr
            addr = m.end
        return max_size

    def access_ok(self, index, access, force=False):
        if False:
            print('Hello World!')
        if isinstance(index, slice):
            assert index.stop - index.start >= 0
            addr = index.start
            while addr < index.stop:
                if addr not in self:
                    return False
                m = self.map_containing(addr)
                if not force and (not m.access_ok(access)):
                    return False
                until_next_page = min(m.end - addr, index.stop - addr)
                addr += until_next_page
            assert addr == index.stop
            return True
        else:
            if index not in self:
                return False
            m = self.map_containing(index)
            return force or m.access_ok(access)

    def read(self, addr, size, force: bool=False) -> List[bytes]:
        if False:
            return 10
        if not self.access_ok(slice(addr, addr + size), 'r', force):
            raise InvalidMemoryAccess(addr, 'r')
        assert size > 0
        result: List[bytes] = []
        stop = addr + size
        p = addr
        while p < stop:
            m = self.map_containing(p)
            _size = min(m.end - p, stop - p)
            result += m[p:p + _size]
            p += _size
        assert p == stop
        return result

    def push_record_writes(self):
        if False:
            return 10
        '\n        Begin recording all writes. Retrieve all writes with `pop_record_writes()`\n        '
        self._recording_stack.append([])

    def pop_record_writes(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Stop recording trace and return a `list[(address, value)]` of all the writes\n        that occurred, where `value` is of type list[str]. Can be called without\n        intermediate `pop_record_writes()`.\n\n        For example::\n\n            mem.push_record_writes()\n                mem.write(1, 'a')\n                mem.push_record_writes()\n                    mem.write(2, 'b')\n                mem.pop_record_writes()  # Will return [(2, 'b')]\n            mem.pop_record_writes()  # Will return [(1, 'a'), (2, 'b')]\n\n        Multiple writes to the same address will all be included in the trace in the\n        same order they occurred.\n\n        :return: list[tuple]\n        "
        lst = self._recording_stack.pop()
        if self._recording_stack:
            self._recording_stack[-1].extend(lst)
        return lst

    def write(self, addr, buf, force=False):
        if False:
            while True:
                i = 10
        size = len(buf)
        if not self.access_ok(slice(addr, addr + size), 'w', force):
            raise InvalidMemoryAccess(addr, 'w')
        assert size > 0
        stop = addr + size
        start = addr
        if self._recording_stack:
            self._recording_stack[-1].append((addr, buf))
        while addr < stop:
            m = self.map_containing(addr)
            size = min(m.end - addr, stop - addr)
            m[addr:addr + size] = buf[addr - start:addr - start + size]
            addr += size
        assert addr == stop

    def _get_size(self, size):
        if False:
            return 10
        return size

    def __setitem__(self, index, value):
        if False:
            i = 10
            return i + 15
        if isinstance(index, slice):
            size = self._get_size(index.stop - index.start)
            assert len(value) == size
            self.write(index.start, value)
        else:
            self.write(index, (value,))

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if isinstance(index, slice):
            result = self.read(index.start, index.stop - index.start)
        else:
            result = self.read(index, 1)[0]
        return result

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Iterate all valid addresses\n        '
        for page_addr in sorted(self._page2map.keys()):
            start = page_addr * self.page_size
            end = start + self.page_size
            for addr in range(start, end):
                yield addr

class SMemory(Memory):
    """
    The symbolic memory manager.
    This class handles all virtual memory mappings and symbolic chunks.

    :todo: improve comments
    """

    def __init__(self, constraints: ConstraintSet, symbols=None, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Builds a memory.\n\n        :param constraints:  a set of initial constraints\n        :param symbols: Symbolic chunks in format: {chunk_start_addr: (condition, value), ...}\n\n        `symbols` or `self._symbols` is a mapping of symbolic chunks/memory cells starting addresses\n        to their condition and value.\n\n        The condition of a symbolic chunk can be concrete (True/False) or symbolic. The value should\n        always be symbolic (e.g. a BitVecVariable).\n        '
        super().__init__(*args, **kwargs)
        assert isinstance(constraints, ConstraintSet)
        self._constraints = constraints
        self._symbols: Dict
        if symbols is None:
            self._symbols = {}
        else:
            self._symbols = dict(symbols)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (self.__class__, (self.constraints, self._symbols, self._maps), {'cpu': self.cpu})

    @property
    def constraints(self):
        if False:
            print('Hello World!')
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if False:
            while True:
                i = 10
        self._constraints = constraints

    def _get_size(self, size):
        if False:
            i = 10
            return i + 15
        if isinstance(size, BitVec):
            size = arithmetic_simplify(size)
        else:
            size = BitVecConstant(size=self.memory_bit_size, value=size)
        assert isinstance(size, BitVecConstant)
        return size.value

    def munmap(self, start, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes the mappings for the specified address range and causes further\n        references to addresses within the range to generate invalid memory\n        references.\n\n        :param start: the starting address to delete.\n        :param size: the length of the unmapping.\n        '
        for addr in range(start, start + size):
            if len(self._symbols) == 0:
                break
            if addr in self._symbols:
                del self._symbols[addr]
        super().munmap(start, size)

    def read(self, address, size, force=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read a stream of potentially symbolic bytes from a potentially symbolic\n        address\n\n        :param address: Where to read from\n        :param size: How many bytes\n        :param force: Whether to ignore permissions\n        :rtype: list\n        '
        size = self._get_size(size)
        assert not issymbolic(size)
        if issymbolic(address):
            logger.debug(f'Reading {size} bytes from symbolic address {address}')
            try:
                solutions = self._try_get_solutions(address, size, 'r', force=force)
                assert len(solutions) > 0
            except TooManySolutions as e:
                (m, M) = self._solver.minmax(self.constraints, address)
                logger.debug(f'Got TooManySolutions on a symbolic read. Range [{m:#x}, {M:#x}]. Not crashing!')
                crashing_condition = True
                for (start, end, perms, offset, name) in self.mappings():
                    if start <= M + size and end >= m:
                        if 'r' in perms:
                            crashing_condition = Operators.AND(Operators.OR((address + size).ult(start), address.uge(end)), crashing_condition)
                can_crash = self._solver.can_be_true(self.constraints, crashing_condition)
                if can_crash:
                    raise InvalidSymbolicMemoryAccess(address, 'r', size, crashing_condition)
                logger.info('INCOMPLETE Result! Using the sampled solutions we have as result')
                condition = False
                for base in e.solutions:
                    condition = Operators.OR(address == base, condition)
                from ..core.state import ForkState
                raise ForkState('Forking state on incomplete result', condition)
            condition = False
            for base in solutions:
                condition = Operators.OR(address == base, condition)
            result = []
            for offset in range(size):
                for base in solutions:
                    addr_value = base + offset
                    byte = Operators.ORD(self.map_containing(addr_value)[addr_value])
                    if addr_value in self._symbols:
                        for (condition, value) in self._symbols[addr_value]:
                            byte = Operators.ITEBV(8, condition, Operators.ORD(value), byte)
                    if len(result) > offset:
                        result[offset] = Operators.ITEBV(8, address == base, byte, result[offset])
                    else:
                        result.append(byte)
                    assert len(result) == offset + 1
            return list(map(Operators.CHR, result))
        else:
            result = list(map(Operators.ORD, super().read(address, size, force)))
            for offset in range(size):
                if address + offset in self._symbols:
                    for (condition, value) in self._symbols[address + offset]:
                        if condition is True:
                            result[offset] = Operators.ORD(value)
                        else:
                            result[offset] = Operators.ITEBV(8, condition, Operators.ORD(value), result[offset])
            return list(map(Operators.CHR, result))

    def write(self, address, value, force: bool=False) -> None:
        if False:
            return 10
        '\n        Write a value at address.\n\n        :param address: The address at which to write\n        :type address: int or long or Expression\n        :param value: Bytes to write\n        :type value: str or list\n        :param force: Whether to ignore permissions\n\n        '
        size = len(value)
        if issymbolic(address):
            solutions = self._try_get_solutions(address, size, 'w', force=force)
            for offset in range(size):
                for base in solutions:
                    condition = base == address
                    self._symbols.setdefault(base + offset, []).append((condition, value[offset]))
        else:
            concrete_start = None
            for offset in range(size):
                ea = address + offset
                if issymbolic(value[offset]):
                    if concrete_start is not None:
                        super().write(address + concrete_start, value[concrete_start:offset], force)
                        concrete_start = None
                    if not self.access_ok(ea, 'w', force):
                        raise InvalidMemoryAccess(ea, 'w')
                    self._symbols[ea] = [(True, value[offset])]
                else:
                    if ea in self._symbols:
                        del self._symbols[ea]
                    if concrete_start is None:
                        concrete_start = offset
            if concrete_start is not None:
                super().write(address + concrete_start, value[concrete_start:], force)

    def _try_get_solutions(self, address, size, access, max_solutions=4096, force=False):
        if False:
            print('Hello World!')
        "\n        Try to solve for a symbolic address, checking permissions when reading/writing size bytes.\n\n        :param Expression address: The address to solve for\n        :param int size: How many bytes to check permissions for\n        :param str access: 'r' or 'w'\n        :param int max_solutions: Will raise if more solutions are found\n        :param force: Whether to ignore permission failure\n        :rtype: list\n        "
        assert issymbolic(address)
        solutions = self._solver.get_all_values(self.constraints, address, maxcnt=max_solutions, silent=True)
        crashing_condition = False
        for base in solutions:
            if not self.access_ok(slice(base, base + size), access, force):
                crashing_condition = Operators.OR(address == base, crashing_condition)
        crash_or_not = self._solver.get_all_values(self.constraints, crashing_condition, maxcnt=3)
        if not consts.fast_crash and len(crash_or_not) == 2:
            from ..core.state import Concretize

            def setstate(state, _value):
                if False:
                    i = 10
                    return i + 15
                'Roll back PC to redo last instruction'
                state.cpu.PC = state.cpu._last_pc
            raise Concretize('Forking on memory safety', expression=crashing_condition, setstate=setstate)
        elif any(crash_or_not):
            raise InvalidSymbolicMemoryAccess(address, access, size, crashing_condition)
        return solutions

class LazySMemory(SMemory):
    """
    A fully symbolic memory.

    Currently does not support cross-page reads/writes.
    """

    def __init__(self, constraints, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(LazySMemory, self).__init__(constraints, *args, **kwargs)
        self.backing_array = constraints.new_array(index_bits=self.memory_bit_size)
        self.backed_by_symbolic_store = set()

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (self.__class__, (self.constraints, self._symbols, self._maps), {'backing_array': self.backing_array, 'backed_by_symbolic_store': self.backed_by_symbolic_store, 'cpu': self.cpu})

    def __setstate__(self, state):
        if False:
            return 10
        self.backing_array = state['backing_array']
        self.backed_by_symbolic_store = state['backed_by_symbolic_store']

    def mmapFile(self, addr, size, perms, filename, offset=0):
        if False:
            i = 10
            return i + 15
        "\n        Creates a new file mapping in the memory address space.\n\n        :param addr: the starting address (took as hint). If C{addr} is C{0} the first big enough\n                     chunk of memory will be selected as starting address.\n        :param size: the contents of a file mapping are initialized using C{size} bytes starting\n                     at offset C{offset} in the file C{filename}.\n        :param perms: the access permissions to this memory.\n        :param filename: the pathname to the file to map.\n        :param offset: the contents of a file mapping are initialized using C{size} bytes starting\n                      at offset C{offset} in the file C{filename}.\n        :return: the starting address where the file was mapped.\n        :rtype: int\n        :raises error:\n                   - 'Address shall be concrete' if C{addr} is not an integer number.\n                   - 'Address too big' if C{addr} goes beyond the limit of the memory.\n                   - 'Map already used' if the piece of memory starting in C{addr} and with length C{size} isn't free.\n        "
        assert addr is None or isinstance(addr, int), 'Address shall be concrete'
        assert addr < self.memory_size, 'Address too big'
        assert size > 0
        self.cpu._publish('will_map_memory', addr, size, perms, filename, offset)
        map = AnonMap(addr, size, perms)
        self._add(map)
        if addr is not None:
            addr = self._floor(addr)
        size = self._ceil(size)
        with open(filename, 'rb') as f:
            fdata = f.read()
        fdata = fdata[offset:]
        fdata = fdata.ljust(size, b'\x00')
        for i in range(size):
            Memory.write(self, addr + i, chr(fdata[i]), force=True)
        logger.debug('New file-memory map @{addr:#x} size:{size:#x}')
        self.cpu._publish('did_map_memory', addr, size, perms, filename, offset, addr)
        return addr

    def _deref_can_succeed(self, mapping, address, size):
        if False:
            print('Hello World!')
        if not issymbolic(address):
            return address >= mapping.start and address + size < mapping.end
        else:
            constraint = Operators.AND(address >= mapping.start, address + size < mapping.end)
            deref_can_succeed = self._solver.can_be_true(self.constraints, constraint)
            return deref_can_succeed

    def _import_concrete_memory(self, from_addr, to_addr):
        if False:
            for i in range(10):
                print('nop')
        "\n        for each address in this range need to read from concrete and write to symbolic\n        it's possible that there will be invalid/unmapped addresses in this range. need to skip to next map if so\n        also need to mark all of these addresses as now in the symbolic store\n\n        :param int from_addr:\n        :param int to_addr:\n        :return:\n        "
        logger.debug(f'Importing concrete memory: {from_addr:#x} - {to_addr:#x} ({to_addr - from_addr} bytes)')
        for m in self.maps:
            span = interval_intersection(m.start, m.end, from_addr, to_addr)
            if span is None:
                continue
            (start, stop) = span
            for addr in range(start, stop):
                if addr in self.backed_by_symbolic_store:
                    continue
                self.backing_array[addr] = Memory.read(self, addr, 1)[0]
                self.backed_by_symbolic_store.add(addr)

    def _map_deref_expr(self, map, address):
        if False:
            return 10
        return Operators.AND(Operators.UGE(address, map.start), Operators.ULT(address, map.end))

    def _reachable_range(self, sym_address, size):
        if False:
            i = 10
            return i + 15
        (addr_min, addr_max) = self._solver.minmax(self.constraints, sym_address)
        return (addr_min, addr_max + size - 1)

    def valid_ptr(self, address):
        if False:
            i = 10
            return i + 15
        assert issymbolic(address)
        expressions = [self._map_deref_expr(m, address) for m in self._maps]
        valid = functools.reduce(Operators.OR, expressions)
        return valid

    def invalid_ptr(self, address):
        if False:
            for i in range(10):
                print('nop')
        return Operators.NOT(self.valid_ptr(address))

    def read(self, address, size, force=False):
        if False:
            while True:
                i = 10
        (access_min, access_max) = self._reachable_range(address, size)
        if issymbolic(address):
            self._import_concrete_memory(access_min, access_max)
        retvals = []
        addrs_to_access = [address + i for i in range(size)]
        for addr in addrs_to_access:
            if issymbolic(addr):
                from_array = True
            elif addr in self.backed_by_symbolic_store:
                m = self.map_containing(addr)
                from_array = not m or 'w' in m.perms
            else:
                from_array = False
            if from_array:
                val = self.backing_array[addr]
            else:
                val = Memory.read(self, addr, 1)[0]
            retvals.append(val)
        return retvals

    def write(self, address, value, force=False):
        if False:
            while True:
                i = 10
        size = len(value)
        addrs_to_access = [address + i for i in range(size)]
        if issymbolic(address):
            (access_min, access_max) = self._reachable_range(address, size)
            self._import_concrete_memory(access_min, access_max)
            for (addr, byte) in zip(addrs_to_access, value):
                self.backing_array[addr] = Operators.ORD(byte)
        else:
            self.backed_by_symbolic_store -= set(addrs_to_access)
            Memory.write(self, address, value)

    def scan_mem(self, data_to_find):
        if False:
            for i in range(10):
                print('nop')
        '\n        Scan for concrete bytes in all mapped memory. Successively yield addresses of all matches.\n\n        :param bytes data_to_find: String to locate\n        :return:\n        '
        if isinstance(data_to_find, bytes):
            data_to_find = [bytes([c]) for c in data_to_find]
        for mapping in sorted(self.maps):
            for ptr in mapping:
                if ptr + len(data_to_find) >= mapping.end:
                    break
                candidate = mapping[ptr:ptr + len(data_to_find)]
                if issymbolic(candidate[0]):
                    break
                if candidate == data_to_find:
                    yield ptr

class Memory32(Memory):
    memory_bit_size: int = 32
    page_bit_size: int = 12

class Memory64(Memory):
    memory_bit_size: int = 64
    page_bit_size: int = 12

class SMemory32(SMemory):
    memory_bit_size: int = 32
    page_bit_size: int = 12

class SMemory32L(SMemory):
    memory_bit_size: int = 32
    page_bit_size: int = 13

class SMemory64(SMemory):
    memory_bit_size: int = 64
    page_bit_size: int = 12

class LazySMemory32(LazySMemory):
    memory_bit_size: int = 32
    page_bit_size: int = 12

class LazySMemory64(LazySMemory):
    memory_bit_size: int = 64
    page_bit_size: int = 12