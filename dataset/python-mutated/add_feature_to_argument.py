from builtins import _test_sink, _test_source

def add_feature_to_first(first, second):
    if False:
        print('Hello World!')
    pass

def add_feature_to_second(first, second):
    if False:
        for i in range(10):
            print('nop')
    pass

def adds_and_taints():
    if False:
        while True:
            i = 10
    x = _test_source()
    add_feature_to_first(x, 0)
    return x

def propagate_add_feature(parameter):
    if False:
        while True:
            i = 10
    return add_feature_to_first(parameter, 0)

def add_via_value_of(first, second):
    if False:
        i = 10
        return i + 15
    pass

def test_add_via_value_of_second():
    if False:
        return 10
    x = _test_source()
    add_via_value_of(x, 'second')
    return x

def dict_test_add_via_value_of_second():
    if False:
        while True:
            i = 10
    x = _test_source()
    add_via_value_of(x['foo'], 'second')
    return x

def test_add_feature_to_sink(parameter):
    if False:
        while True:
            i = 10
    add_feature_to_first(parameter, '')
    _test_sink(parameter)

def test_add_feature_in_comprehension():
    if False:
        print('Hello World!')
    sources = [_test_source()]
    v = [s for s in sources if add_feature_to_first(s, 0)]
    _test_sink(v[0])

def test_add_feature_to_sink_in_comprehension(parameter):
    if False:
        i = 10
        return i + 15
    x = [s for s in [1, 2, 3] if add_feature_to_first(parameter, 0)]
    _test_sink(parameter)

def propagate_multiple_add_feature(parameter):
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        add_feature_to_first(parameter.foo, 0)
    else:
        add_feature_to_second(0, parameter.bar)

def test_add_multiple_feature(parameter):
    if False:
        while True:
            i = 10
    propagate_multiple_add_feature(parameter)
    _test_sink(parameter)

def tito_with_feature(x):
    if False:
        print('Hello World!')
    ...

def add_feature_to_argument_accumulates_features(x):
    if False:
        print('Hello World!')
    x = tito_with_feature(x)
    add_feature_to_first(x, 0)

def source_add_feature_to_argument_accumulates_features():
    if False:
        for i in range(10):
            print('nop')
    x = _test_source()
    add_feature_to_argument_accumulates_features(x)
    return x