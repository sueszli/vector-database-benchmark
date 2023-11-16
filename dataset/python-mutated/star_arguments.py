def simple(a):
    if False:
        print('Hello World!')
    return a

def nested(*args):
    if False:
        while True:
            i = 10
    return simple(*args)
nested(1)
nested()

def nested_no_call_to_function(*args):
    if False:
        i = 10
        return i + 15
    return simple(1, *args)

def simple2(a, b, c):
    if False:
        i = 10
        return i + 15
    return b

def nested(*args):
    if False:
        i = 10
        return i + 15
    return simple2(1, *args)

def nested_twice(*args1):
    if False:
        return 10
    return nested(*args1)
nested_twice(2, 3)
nested_twice(2)
nested_twice(2, 3, 4)

def star_args_with_named(*args):
    if False:
        return 10
    return simple2(*args, c='')
star_args_with_named(1, 2)

def kwargs_test(**kwargs):
    if False:
        return 10
    return simple2(1, **kwargs)
kwargs_test(c=3, b=2)
kwargs_test(c=3)
kwargs_test(b=2)
kwargs_test(b=2, c=3, d=4)
kwargs_test(b=2, c=3, a=4)

def kwargs_nested(**kwargs):
    if False:
        i = 10
        return i + 15
    return kwargs_test(b=2, **kwargs)
kwargs_nested(c=3)
kwargs_nested()
kwargs_nested(c=2, d=4)
kwargs_nested(c=2, a=4)

def simple_mixed(a, b, c):
    if False:
        while True:
            i = 10
    return b

def mixed(*args, **kwargs):
    if False:
        return 10
    return simple_mixed(1, *args, **kwargs)
mixed(1, 2)
mixed(1, c=2)
mixed(b=2, c=3)
mixed(c=4, b='')

def mixed2(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return simple_mixed(1, *args, **kwargs)
mixed2(c=2)
mixed2(3)
mixed2(3, 4, 5)
mixed2(3, b=5)
simple(1, **[])
simple(1, **1)

class A:
    pass
simple(1, **A())
simple(1, *1)