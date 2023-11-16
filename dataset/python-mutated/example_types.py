"""This script is just a short example of common usages for miasm.core.types.
For a more complete view of what is possible, tests/core/types.py covers
most of the module possibilities, and the module doc gives useful information
as well.
"""
from __future__ import print_function
from miasm.core.utils import iterbytes
from miasm.analysis.machine import Machine
from miasm.core.types import MemStruct, Self, Void, Str, Array, Ptr, Num, Array, set_allocator
from miasm.os_dep.common import heap
from miasm.core.locationdb import LocationDB
loc_db = LocationDB()
my_heap = heap()
set_allocator(my_heap.vm_alloc)

class ListNode(MemStruct):
    fields = [('next', Ptr('<I', Self())), ('data', Ptr('<I', Void()))]

    def get_next(self):
        if False:
            while True:
                i = 10
        if self.next.val == 0:
            return None
        return self.next.deref

    def get_data(self, data_type=None):
        if False:
            print('Hello World!')
        if data_type is not None:
            return self.data.deref.cast(data_type)
        else:
            return self.data.deref

class LinkedList(MemStruct):
    fields = [('head', Ptr('<I', ListNode)), ('tail', Ptr('<I', ListNode)), ('size', Num('<I'))]

    def get_head(self):
        if False:
            return 10
        'Returns the head ListNode instance'
        if self.head == 0:
            return None
        return self.head.deref

    def get_tail(self):
        if False:
            return 10
        'Returns the tail ListNode instance'
        if self.tail == 0:
            return None
        return self.tail.deref

    def push(self, data):
        if False:
            return 10
        'Push a data (MemType instance) to the linked list.'
        node = ListNode(self._vm)
        node.data = data.get_addr()
        if self.head != 0:
            head = self.get_head()
            node.next = head.get_addr()
        self.head = node.get_addr()
        if self.tail == 0:
            self.tail = node.get_addr()
        self.size += 1

    def pop(self, data_type=None):
        if False:
            while True:
                i = 10
        'Pop one data from the LinkedList.'
        if self.head == 0:
            return None
        node = self.get_head()
        self.head = node.next
        if self.head == 0:
            self.tail = 0
        self.size -= 1
        return node.get_data(data_type)

    def empty(self):
        if False:
            i = 10
            return i + 15
        'True if the list is empty.'
        return self.head == 0

    def __iter__(self):
        if False:
            print('Hello World!')
        if not self.empty():
            cur = self.get_head()
            while cur is not None:
                yield cur.data.deref
                cur = cur.get_next()

class DataArray(MemStruct):
    fields = [('val1', Num('B')), ('val2', Num('B')), ('arrayptr', Ptr('<I', Array(Num('B'), 16))), ('array', Array(Num('B'), 16))]

class DataStr(MemStruct):
    fields = [('valshort', Num('<H')), ('data', Ptr('<I', Str('utf16')))]
print('This script demonstrates a LinkedList implementation using the types ')
print('module in the first part, and how to play with some casts in the second.')
print()
jitter = Machine('x86_32').jitter(loc_db, 'python')
vm = jitter.vm
link = LinkedList(vm)
link.memset()
link.push(DataArray(vm))
link.push(DataArray(vm))
link.push(DataArray(vm))
assert link.size == 3
raw_size = vm.get_mem(link.get_addr('size'), link.get_type().get_field_type('size').size)
assert raw_size == b'\x03\x00\x00\x00'
print('The linked list just built:')
print(repr(link), '\n')
print('Its uninitialized data elements:')
for data in link:
    real_data = data.cast(DataArray)
    print(repr(real_data))
print()
data = link.pop(DataArray)
assert link.size == 2
data.arrayptr = data.get_addr('array')
assert data.arrayptr.deref == data.array
datastr = data.cast(DataStr)
print('First element casted to DataStr:')
print(repr(datastr))
print()
data.val1 = 52
data.val2 = 18
assert datastr.valshort == 4660
datastr.valshort = 4386
assert data.val1 == 34 and data.val2 == 17
memstr = datastr.data.deref
memstr.val = 'Miams'
print('Cast data.array to MemStr and set the string value:')
print(repr(memstr))
print()
raw_miams = 'Miams'.encode('utf-16le') + b'\x00' * 2
raw_miams_array = [ord(c) for c in iterbytes(raw_miams)]
assert list(data.array)[:len(raw_miams_array)] == raw_miams_array
assert data.array.cast(Str('utf16')) == memstr
assert data.array.cast(Str()) != memstr
assert data.array.cast(Str('utf16')).val == memstr.val
print('See that the original array has been modified:')
print(repr(data))
print()
argv_t = Array(Ptr('<I', Str()), 4)
print('3 arguments argv type:', argv_t)
argv = argv_t.lval(vm)
MemStrAnsi = Str().lval
argv[0].val = MemStrAnsi.from_str(vm, './my-program').get_addr()
argv[1].val = MemStrAnsi.from_str(vm, 'arg1').get_addr()
argv[2].val = MemStrAnsi.from_str(vm, '27').get_addr()
argv[3].val = 0
argv[2].deref.val = '42'
print('An argv instance:', repr(argv))
print('argv values:', repr([val.deref.val for val in argv[:-1]]))
print()
print('See test/core/types.py and the miasm.core.types module doc for ')
print('more information.')