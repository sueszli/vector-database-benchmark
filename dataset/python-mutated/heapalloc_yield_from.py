import micropython

def sub_gen(a):
    if False:
        while True:
            i = 10
    for i in range(a):
        yield i

def gen(g):
    if False:
        print('Hello World!')
    yield from g
g = gen(sub_gen(4))
micropython.heap_lock()
print(next(g))
print(next(g))
micropython.heap_unlock()

class G:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 0

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        v = self.value
        self.value += 1
        return v
g = gen(G())
micropython.heap_lock()
print(next(g))
print(next(g))
micropython.heap_unlock()