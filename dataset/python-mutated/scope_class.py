def test1():
    if False:
        return 10

    def method():
        if False:
            for i in range(10):
                print('nop')
        pass

    class A:

        def method():
            if False:
                return 10
            pass
    print(hasattr(A, 'method'))
    print(hasattr(A(), 'method'))
test1()

def test2():
    if False:
        while True:
            i = 10

    def method():
        if False:
            i = 10
            return i + 15
        return 'outer'

    class A:
        nonlocal method

        def method():
            if False:
                return 10
            return 'inner'
    print(hasattr(A, 'method'))
    print(hasattr(A(), 'method'))
    return method()
print(test2())

def test3(x):
    if False:
        return 10

    class A:
        local = x
    x += 1
    return (x, A.local)
print(test3(42))

def test4(global_):
    if False:
        print('Hello World!')

    class A:
        local = global_
        global_ = 'global2'
    global_ += 1
    return (global_, A.local, A.global_)
global_ = 'global'
print(test4(42), global_)

def test5(x):
    if False:
        print('Hello World!')

    def closure():
        if False:
            while True:
                i = 10
        return x

    class A:

        def method():
            if False:
                print('Hello World!')
            return (x, closure())
    closure = lambda : x + 1
    return A
print(test5(42).method())