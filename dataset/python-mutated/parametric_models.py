from builtins import _test_source

def f():
    if False:
        return 10
    ...

def g():
    if False:
        print('Hello World!')
    ...

def sink(x):
    if False:
        print('Hello World!')
    ...

def f_and_g_to_test():
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        a = f()
    else:
        a = g()
    sink(a)

def sink_subkind_a(x):
    if False:
        return 10
    ...

def sink_subkind_b(x):
    if False:
        return 10
    ...

def inferred_sink(x):
    if False:
        print('Hello World!')
    if 1 > 2:
        sink_subkind_a(x)
    else:
        sink_subkind_b(x)

def test_to_subkind_sink():
    if False:
        print('Hello World!')
    x = _test_source()
    inferred_sink(x)