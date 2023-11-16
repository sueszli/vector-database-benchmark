class CtxMgr:

    def __init__(self, id):
        if False:
            print('Hello World!')
        self.id = id

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        print('__enter__', self.id)
        return self

    def __exit__(self, a, b, c):
        if False:
            print('Hello World!')
        print('__exit__', self.id, repr(a), repr(b))

def foo():
    if False:
        return 10
    with CtxMgr(1):
        return 4
print(foo())

def f():
    if False:
        return 10
    with CtxMgr(1):
        for i in [1, 2]:
            return i
print(f())

def f():
    if False:
        for i in range(10):
            print('nop')
    with CtxMgr(1):
        for i in [1, 2]:
            for j in [3, 4]:
                return (i, j)
print(f())

def f():
    if False:
        print('Hello World!')
    with CtxMgr(1):
        for i in [1, 2]:
            for j in [3, 4]:
                with CtxMgr(2):
                    for k in [5, 6]:
                        for l in [7, 8]:
                            return (i, j, k, l)
print(f())

def f():
    if False:
        print('Hello World!')
    with CtxMgr(1):
        for i in range(1, 3):
            for j in range(3, 5):
                with CtxMgr(2):
                    for k in range(5, 7):
                        for l in range(7, 9):
                            return (i, j, k, l)
print(f())