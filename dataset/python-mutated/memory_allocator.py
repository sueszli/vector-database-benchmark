from typing import List
from vyper.exceptions import CompilerPanic, MemoryAllocationException
from vyper.utils import MemoryPositions

class FreeMemory:
    __slots__ = ('position', 'size')

    def __init__(self, position: int, size: int) -> None:
        if False:
            i = 10
            return i + 15
        self.position = position
        self.size = size

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'(FreeMemory: pos={self.position}, size={self.size})'

    def partially_allocate(self, size: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Reduce the size of the free memory by allocating from the initial offset.\n\n        Arguments\n        ---------\n        size : int\n            Number of bytes to allocate\n\n        Returns\n        -------\n        int\n            Position of the newly allocated memory\n        '
        if size >= self.size:
            raise CompilerPanic('Attempted to allocate more memory than available')
        position = self.position
        self.position += size
        self.size -= size
        return position

class MemoryAllocator:
    """
    Low-level memory alloctor. Used to allocate and de-allocate memory slots.

    This object should not be accessed directly. Memory allocation happens via
    declaring variables within `Context`.
    """
    next_mem: int
    _ALLOCATION_LIMIT: int = 2 ** 64

    def __init__(self, start_position: int=MemoryPositions.RESERVED_MEMORY):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializer.\n\n        Arguments\n        ---------\n        start_position : int, optional\n            The initial offset to use as the free memory pointer. Offsets\n            prior to this value are considered permanently allocated.\n        '
        self.next_mem = start_position
        self.size_of_mem = start_position
        self.deallocated_mem: List[FreeMemory] = []

    def get_next_memory_position(self) -> int:
        if False:
            print('Hello World!')
        return self.next_mem

    def allocate_memory(self, size: int) -> int:
        if False:
            print('Hello World!')
        '\n        Allocate `size` bytes in memory.\n\n        *** No guarantees are made that allocated memory is clean! ***\n\n        If no memory was previously de-allocated, memory is expanded\n        and the free memory pointer is increased.\n\n        If sufficient space is available within de-allocated memory, the lowest\n        available offset is returned and that memory is now marked as allocated.\n\n        Arguments\n        ---------\n        size : int\n            The number of bytes to allocate. Must be divisible by 32.\n\n        Returns\n        -------\n        int\n            Start offset of the newly allocated memory.\n        '
        if size % 32 != 0:
            raise CompilerPanic(f'tried to allocate {size} bytes, only multiples of 32 supported.')
        for (i, free_memory) in enumerate(self.deallocated_mem):
            if free_memory.size == size:
                del self.deallocated_mem[i]
                return free_memory.position
            if free_memory.size > size:
                return free_memory.partially_allocate(size)
        return self._expand_memory(size)

    def _expand_memory(self, size: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Allocate `size` bytes in memory, starting from the free memory pointer.\n        '
        if size % 32 != 0:
            raise CompilerPanic('Memory misaligment, only multiples of 32 supported.')
        before_value = self.next_mem
        self.next_mem += size
        self.size_of_mem = max(self.size_of_mem, self.next_mem)
        if self.size_of_mem >= self._ALLOCATION_LIMIT:
            raise MemoryAllocationException(f'Tried to allocate {self.size_of_mem} bytes! (limit is {self._ALLOCATION_LIMIT} (2**64) bytes)')
        return before_value

    def deallocate_memory(self, pos: int, size: int) -> None:
        if False:
            while True:
                i = 10
        '\n        De-allocate memory.\n\n        Arguments\n        ---------\n        pos : int\n            The initial memory position to de-allocate.\n        size : int\n            The number of bytes to de-allocate. Must be divisible by 32.\n        '
        if size % 32 != 0:
            raise CompilerPanic('Memory misaligment, only multiples of 32 supported.')
        self.deallocated_mem.append(FreeMemory(position=pos, size=size))
        self.deallocated_mem.sort(key=lambda k: k.position)
        if not self.deallocated_mem:
            return
        i = 1
        active = self.deallocated_mem[0]
        while len(self.deallocated_mem) > i:
            next_slot = self.deallocated_mem[i]
            if next_slot.position == active.position + active.size:
                active.size += next_slot.size
                self.deallocated_mem.remove(next_slot)
            else:
                active = next_slot
                i += 1
        last = self.deallocated_mem[-1]
        if last.position + last.size == self.next_mem:
            self.next_mem = last.position
            del self.deallocated_mem[-1]