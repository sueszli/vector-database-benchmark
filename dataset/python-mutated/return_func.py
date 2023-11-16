def lazy_sum(*args):
    if False:
        while True:
            i = 10

    def sum():
        if False:
            i = 10
            return i + 15
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum
f = lazy_sum(1, 2, 4, 5, 7, 8, 9)
print(f)
print(f())

def count():
    if False:
        for i in range(10):
            print('nop')
    fs = []
    for i in range(1, 4):

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return i * i
        fs.append(f)
    return fs
(f1, f2, f3) = count()
print(f1())
print(f2())
print(f3())

def count():
    if False:
        for i in range(10):
            print('nop')
    fs = []

    def f(n):
        if False:
            print('Hello World!')

        def j():
            if False:
                while True:
                    i = 10
            return n * n
        return j
    for i in range(1, 4):
        fs.append(f(i))
    return fs
(f1, f2, f3) = count()
print(f1())
print(f2())
print(f3())