def test1_d1(f, *args, **kwargs):
    if False:
        return 10
    pass

def test1_d2(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    pass

def test2_d1(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    pass

def test2_d2(f, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

def test3_d1(f, *args, **kwargs):
    if False:
        print('Hello World!')
    pass

def test4_d1(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    pass

def test5_d1(f, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

@test1_d1
def test1_alarm1():
    if False:
        i = 10
        return i + 15
    return None

@test1_d2
def test1_noalarm1():
    if False:
        return 10
    return None

@test2_d1
def test2_alarm1():
    if False:
        for i in range(10):
            print('nop')
    return None

@test2_d2
def test2_noalarm1():
    if False:
        print('Hello World!')
    return None
arg1 = None

@test3_d1(arg1, 2, arg3='Foo', arg4='Bar')
def test3_alarm1():
    if False:
        print('Hello World!')
    return None

@test3_d1(arg1, 2, 'bar', arg3='Foo')
def test3_alarm2():
    if False:
        for i in range(10):
            print('nop')
    return None

@test3_d1(arg1, 2, arg3='Foo')
def test3_alarm3():
    if False:
        print('Hello World!')
    return None

@test3_d1()
def test3_noalarm1():
    if False:
        while True:
            i = 10
    return None

@test3_d1(arg3='Foo')
def test3_noalarm2():
    if False:
        for i in range(10):
            print('nop')
    return None

@test4_d1(arg1, 2, arg3='Foo')
def test4_alarm1():
    if False:
        while True:
            i = 10
    return None

@test4_d1(arg1, 2, arg3='Foo', arg4='Bar')
def test4_noalarm1():
    if False:
        i = 10
        return i + 15
    return None

@test4_d1()
def test4_noalarm2():
    if False:
        return 10
    return None

@test4_d1(arg3='Foo')
def test4_noalarm3():
    if False:
        i = 10
        return i + 15
    return None

@test5_d1(arg1, 2, arg3='Foo')
def test5_alarm1():
    if False:
        i = 10
        return i + 15
    return None

@test5_d1(arg1, 2, 3, 4, arg4='Bar', arg3='Foo')
def test5_alarm2():
    if False:
        return 10
    return None

@test5_d1(2, arg1, arg3='Foo')
def test5_noalarm1():
    if False:
        return 10
    return None

@test5_d1(arg1, 3, 2, 4, arg3='Foo')
def test5_noalarm2():
    if False:
        while True:
            i = 10
    return None

class Application:

    def decorator(self, f, *args, **kwargs):
        if False:
            return 10
        pass
app = Application()

@app.decorator
def test_local_variable_method_decorator():
    if False:
        while True:
            i = 10
    return None

def decorator_ignored(f):
    if False:
        return 10
    return f

@decorator_ignored
def source_on_decorator_ignored(x):
    if False:
        i = 10
        return i + 15
    return x