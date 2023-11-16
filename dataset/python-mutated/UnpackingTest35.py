def tupleUnpacking():
    if False:
        for i in range(10):
            print('nop')
    return (*a, b, *c)

def listUnpacking():
    if False:
        return 10
    return [*a, b, *c]

def setUnpacking():
    if False:
        return 10
    return {*a, b, *c}

def dictUnpacking():
    if False:
        return 10
    return {'a': 1, **d}
a = range(3)
b = 5
c = range(8, 10)
d = {'a': 2}
print('Tuple unpacked', tupleUnpacking())
print('List unpacked', listUnpacking())
print('Set unpacked', setUnpacking())
print('Dict unpacked', dictUnpacking())
non_iterable = 2.0

def tupleUnpackingError():
    if False:
        return 10
    try:
        return (*a, *non_iterable, *c)
    except Exception as e:
        return e

def listUnpackingError():
    if False:
        while True:
            i = 10
    try:
        return [*a, *non_iterable, *c]
    except Exception as e:
        return e

def setUnpackingError():
    if False:
        print('Hello World!')
    try:
        return {*a, *non_iterable, *c}
    except Exception as e:
        return e

def dictUnpackingError():
    if False:
        while True:
            i = 10
    try:
        return {'a': 1, **non_iterable}
    except Exception as e:
        return e
print('Tuple unpacked error:', tupleUnpackingError())
print('List unpacked error:', listUnpackingError())
print('Set unpacked error:', setUnpackingError())
print('Dict unpacked error:', dictUnpackingError())