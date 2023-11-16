def f1(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    print(kwargs)
f1()
f1(a=1)

def f2(a, **kwargs):
    if False:
        while True:
            i = 10
    print(a, kwargs)
f2(1)
f2(1, b=2)

def f3(a, *vargs, **kwargs):
    if False:
        i = 10
        return i + 15
    print(a, vargs, kwargs)
f3(1)
f3(1, 2)
f3(1, b=2)
f3(1, 2, b=3)

def f4(*vargs, **kwargs):
    if False:
        return 10
    print(vargs, kwargs)
f4(*(1, 2))
f4(kw_arg=3)
f4(*(1, 2), kw_arg=3)

def f5(*vargs, **kwargs):
    if False:
        print('Hello World!')
    print(vargs, kwargs)

def print_ret(x):
    if False:
        i = 10
        return i + 15
    print(x)
    return x
f5(*print_ret(['a', 'b']), kw_arg=print_ret(None))