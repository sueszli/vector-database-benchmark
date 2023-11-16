import micropython

class GetSlice:

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        return idx
sl = GetSlice()[:]
micropython.heap_lock()
try:
    bytearray(4)
except MemoryError:
    print('MemoryError: bytearray create')
micropython.heap_unlock()
micropython.heap_lock()
try:
    bytearray(b'0123')
except MemoryError:
    print('MemoryError: bytearray create from bytes')
micropython.heap_unlock()
r = range(4)
micropython.heap_lock()
try:
    bytearray(r)
except MemoryError:
    print('MemoryError: bytearray create from iter')
micropython.heap_unlock()
b = bytearray(4)
micropython.heap_lock()
try:
    b + b'01'
except MemoryError:
    print('MemoryError: bytearray.__add__')
micropython.heap_unlock()
b = bytearray(4)
micropython.heap_lock()
try:
    b += b'01234567'
except MemoryError:
    print('MemoryError: bytearray.__iadd__')
micropython.heap_unlock()
print(b)
b = bytearray(4)
micropython.heap_lock()
try:
    for i in range(100):
        b.append(1)
except MemoryError:
    print('MemoryError: bytearray.append')
micropython.heap_unlock()
b = bytearray(4)
micropython.heap_lock()
try:
    b.extend(b'01234567')
except MemoryError:
    print('MemoryError: bytearray.extend')
micropython.heap_unlock()
b = bytearray(4)
micropython.heap_lock()
try:
    b[sl]
except MemoryError:
    print('MemoryError: bytearray subscr get')
micropython.heap_unlock()
b = bytearray(4)
micropython.heap_lock()
try:
    b[sl] = b'01234567'
except MemoryError:
    print('MemoryError: bytearray subscr grow')
micropython.heap_unlock()
print(b)