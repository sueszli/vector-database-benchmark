from builtins import _test_sink, _test_source

def source_on_first():
    if False:
        for i in range(10):
            print('nop')
    return (1, 0)

def source_on_second():
    if False:
        i = 10
        return i + 15
    return (0, 1)

def source_on_0_1():
    if False:
        for i in range(10):
            print('nop')
    return ((0, 1), (0, 0))

def issue_only_with_source_first():
    if False:
        return 10
    (issue, no_issue) = source_on_first()
    _test_sink(issue)
    _test_sink(no_issue)

def issue_only_with_source_second():
    if False:
        for i in range(10):
            print('nop')
    (no_issue, issue) = source_on_second()
    _test_sink(no_issue)
    _test_sink(issue)

def issue_only_with_source_nested_first():
    if False:
        print('Hello World!')
    (first, second) = source_on_0_1()
    (a, issue) = first
    (c, d) = second
    _test_sink(issue)
    _test_sink(a)
    _test_sink(c)
    _test_sink(d)
    return source_on_0_1()

def source_on_key_a():
    if False:
        print('Hello World!')
    return {'a': 1}

def issue_only_with_source_key_a():
    if False:
        return 10
    d = source_on_key_a()
    _test_sink(d['a'])
    _test_sink(d['b'])

def source_on_member_a():
    if False:
        i = 10
        return i + 15
    ...

def issue_with_source_member():
    if False:
        while True:
            i = 10
    x = source_on_member_a()
    _test_sink(x.a)
    _test_sink(x.b)

def sink_on_first(arg):
    if False:
        print('Hello World!')
    return

def sink_on_second(arg):
    if False:
        print('Hello World!')
    return

def sink_on_0_1(arg):
    if False:
        while True:
            i = 10
    return

def issue_only_with_sink_first():
    if False:
        return 10
    sink_on_first(arg=(_test_source(), 0))
    sink_on_first(arg=(0, _test_source()))

def issue_only_with_sink_second():
    if False:
        return 10
    sink_on_second(arg=(_test_source(), 0))
    sink_on_second(arg=(0, _test_source()))

def issue_only_with_sink_nested_first():
    if False:
        for i in range(10):
            print('nop')
    sink_on_0_1(arg=((_test_source(), 0), (0, 0)))
    sink_on_0_1(arg=((0, _test_source()), (0, 0)))
    sink_on_0_1(arg=((0, 0), (_test_source(), 0)))
    sink_on_0_1(arg=((0, 0), (0, _test_source())))

def sink_on_key_a(arg):
    if False:
        print('Hello World!')
    return

def sink_on_member_a(arg):
    if False:
        while True:
            i = 10
    return

def issue_only_with_sink_key_a():
    if False:
        while True:
            i = 10
    sink_on_key_a({'a': _test_source(), 'b': 0})
    sink_on_key_a({'a': 0, 'b': _test_source()})

def issue_with_sink_member():
    if False:
        for i in range(10):
            print('nop')
    x = object()
    x.a = _test_source()
    sink_on_member_a(x)
    y = object()
    y.b = _test_source()
    sink_on_member_a(y)

def tito_from_first(arg):
    if False:
        print('Hello World!')
    return

def tito_from_second(arg):
    if False:
        return 10
    return

def issue_tito_from_first():
    if False:
        i = 10
        return i + 15
    _test_sink(tito_from_first(arg=(_test_source(), 0)))
    _test_sink(tito_from_first(arg=(0, _test_source())))

def issue_tito_from_second():
    if False:
        for i in range(10):
            print('nop')
    _test_sink(tito_from_second(arg=(_test_source(), 0)))
    _test_sink(tito_from_second(arg=(0, _test_source())))

def tito_from_first_to_second(arg):
    if False:
        for i in range(10):
            print('nop')
    return

def issue_tito_first_to_second():
    if False:
        i = 10
        return i + 15
    _test_sink(tito_from_first_to_second(arg=(_test_source(), 0))[0])
    _test_sink(tito_from_first_to_second(arg=(0, _test_source()))[0])
    _test_sink(tito_from_first_to_second(arg=(_test_source(), 0))[1])
    _test_sink(tito_from_first_to_second(arg=(0, _test_source()))[1])

def tito_from_b_to_a(arg):
    if False:
        i = 10
        return i + 15
    return

def issue_tito_b_to_a():
    if False:
        while True:
            i = 10
    _test_sink(tito_from_b_to_a({'a': _test_source(), 'b': 0})['a'])
    _test_sink(tito_from_b_to_a({'a': 0, 'b': _test_source()})['a'])
    _test_sink(tito_from_b_to_a({'a': _test_source(), 'b': 0})['b'])
    _test_sink(tito_from_b_to_a({'a': 0, 'b': _test_source()})['b'])

def tito_from_a_to_self_b(self, arg):
    if False:
        for i in range(10):
            print('nop')
    return

def issue_tito_from_a_to_self_b():
    if False:
        print('Hello World!')
    x = {}
    tito_from_a_to_self_b(x, {'a': _test_source(), 'b': 0})
    _test_sink(x['a'])
    _test_sink(x['b'])
    x = {}
    tito_from_a_to_self_b(x, {'a': 0, 'b': _test_source()})
    _test_sink(x['a'])
    _test_sink(x['b'])

def complex_tito(arg):
    if False:
        print('Hello World!')
    return

def issue_complex_tito():
    if False:
        while True:
            i = 10
    _test_sink(complex_tito({'a': {'x': {_test_source(): 0}}})['foo'])
    _test_sink(complex_tito({'a': {'x': {0: _test_source()}}})['foo'])
    _test_sink(complex_tito({'a': {'x': {_test_source(): 0}}})['bar'])
    _test_sink(complex_tito({'a': {'x': {0: _test_source()}}})['bar'])
    _test_sink(complex_tito({'b': {'x': {_test_source(): 0}}})['foo'])
    _test_sink(complex_tito({'b': {'x': {0: _test_source()}}})['foo'])
    _test_sink(complex_tito({'b': {'x': {_test_source(): 0}}})['bar'])
    _test_sink(complex_tito({'b': {'x': {0: _test_source()}}})['bar'])