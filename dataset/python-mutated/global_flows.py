from builtins import _test_source, _test_sink

def update_arg1(arg1, arg2):
    if False:
        return 10
    ...

def update_arg2(arg1, arg2):
    if False:
        while True:
            i = 10
    ...
x = 1

def update_x_at_arg1():
    if False:
        print('Hello World!')
    update_arg1(x, _test_source())

def unaffected_x_at_arg1():
    if False:
        while True:
            i = 10
    update_arg1(x, 'not a taint source')

def update_x_at_arg2():
    if False:
        for i in range(10):
            print('nop')
    update_arg2(_test_source(), x)

def unaffected_x_at_arg2():
    if False:
        return 10
    update_arg2('not a taint source', x)

def indirectly_update_x_arg1(arg):
    if False:
        i = 10
        return i + 15
    update_arg1(x, arg)

def x_tainted_indirectly_arg1():
    if False:
        for i in range(10):
            print('nop')
    indirectly_update_x_arg1(_test_source())

def x_not_tainted():
    if False:
        print('Hello World!')
    indirectly_update_x_arg1(1)

def indirectly_update_x_arg2(arg):
    if False:
        return 10
    update_arg2(arg, x)

def x_tainted_indirectly_arg2():
    if False:
        return 10
    indirectly_update_x_arg2(_test_source())

class MyList:

    def append(self, item):
        if False:
            for i in range(10):
                print('nop')
        ...
l: MyList = ...

def append_directly():
    if False:
        print('Hello World!')
    l.append(_test_source())

def append_argument(arg):
    if False:
        while True:
            i = 10
    l.append(arg)

def append_indirectly():
    if False:
        return 10
    append_argument(_test_source())
tainted = ...

def global_source():
    if False:
        for i in range(10):
            print('nop')
    _test_sink(tainted)