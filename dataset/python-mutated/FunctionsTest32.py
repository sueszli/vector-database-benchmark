def displayDict(d):
    if False:
        while True:
            i = 10
    result = '{'
    first = True
    for (key, value) in sorted(d.items()):
        if not first:
            result += ','
        result += '%s: %s' % (repr(key), repr(value))
        first = False
    result += '}'
    return result

def kwonlysimple(*, a):
    if False:
        for i in range(10):
            print('nop')
    return a
print('Keyword only function case: ', kwonlysimple(a=3))

def kwonlysimpledefaulted(*, a=5):
    if False:
        while True:
            i = 10
    return a
print('Keyword only function, using default value: ', kwonlysimpledefaulted())

def default1():
    if False:
        print('Hello World!')
    print('Called', default1)
    return 1

def default2():
    if False:
        print('Hello World!')
    print('Called', default2)
    return 2

def default3():
    if False:
        i = 10
        return i + 15
    print('Called', default3)
    return 3

def default4():
    if False:
        return 10
    print('Called', default4)
    return 4

def annotation1():
    if False:
        print('Hello World!')
    print('Called', annotation1)
    return 'a1'

def annotation2():
    if False:
        return 10
    print('Called', annotation2)
    return 'a2'

def annotation3():
    if False:
        i = 10
        return i + 15
    print('Called', annotation3)
    return 'a3'

def annotation4():
    if False:
        i = 10
        return i + 15
    print('Called', annotation4)
    return 'a4'

def annotation5():
    if False:
        return 10
    print('Called', annotation5)
    return 'a5'

def annotation6():
    if False:
        while True:
            i = 10
    print('Called', annotation6)
    return 'a6'

def annotation7():
    if False:
        while True:
            i = 10
    print('Called', annotation7)
    return 'a7'

def annotation8():
    if False:
        i = 10
        return i + 15
    print('Called', annotation8)
    return 'a8'

def annotation9():
    if False:
        i = 10
        return i + 15
    print('Called', annotation9)
    return 'a9'
print('Defining function with annotations, and defaults as functions for everything:')

def kwonlyfunc(x: annotation1(), y: annotation2()=default1(), z: annotation3()=default2(), *, a: annotation4(), b: annotation5()=default3(), c: annotation6()=default4(), d: annotation7(), **kw: annotation8()) -> annotation9():
    if False:
        i = 10
        return i + 15
    print(x, y, z, a, b, c, d)
print('__kwdefaults__', displayDict(kwonlyfunc.__kwdefaults__))
print('Keyword only function called:')
kwonlyfunc(7, a=8, d=12)
print('OK.')
print('Annotations come out as', sorted(kwonlyfunc.__annotations__))
kwonlyfunc.__annotations__ = {}
print('After updating to None it is', kwonlyfunc.__annotations__)
kwonlyfunc.__annotations__ = {'k': 9}
print('After updating to None it is', kwonlyfunc.__annotations__)

def kwonlystarfunc(*, a, b, **d):
    if False:
        for i in range(10):
            print('nop')
    return (a, b, d)
print('kwonlystarfunc', kwonlystarfunc(a=8, b=12, k=9, j=7))

def deeplyNestedNonLocalWrite():
    if False:
        return 10
    x = 0
    y = 0

    def f():
        if False:
            while True:
                i = 10

        def g():
            if False:
                while True:
                    i = 10
            nonlocal x
            x = 3
            return x
        return g()
    return (f(), x)
print('Deeply nested non local writing function', deeplyNestedNonLocalWrite())

def deletingClosureVariable():
    if False:
        i = 10
        return i + 15
    try:
        x = 1

        def g():
            if False:
                print('Hello World!')
            nonlocal x
            del x
        g()
        g()
    except Exception as e:
        return repr(e)
print('Using deleted non-local variable:', deletingClosureVariable())