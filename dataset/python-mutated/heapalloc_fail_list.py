import micropython

class GetSlice:

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        return idx
sl = GetSlice()[:]
l = [1, 2, 3]
micropython.heap_lock()
try:
    print(l[0:1])
except MemoryError:
    print('MemoryError: list index')
micropython.heap_unlock()
micropython.heap_lock()
try:
    l[sl]
except MemoryError:
    print('MemoryError: list get slice')
micropython.heap_unlock()
l = [1, 2]
l2 = [3, 4, 5, 6, 7, 8, 9, 10]
micropython.heap_lock()
try:
    l[sl] = l2
except MemoryError:
    print('MemoryError: list extend slice')
micropython.heap_unlock()
print(l)