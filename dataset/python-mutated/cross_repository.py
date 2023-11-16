def cross_repository_source(source_parameter):
    if False:
        return 10
    _test_sink(source_parameter)

def returns_crtex_source():
    if False:
        i = 10
        return i + 15
    pass

def reaches_crtex_sink(x):
    if False:
        while True:
            i = 10
    pass

def test():
    if False:
        return 10
    s = returns_crtex_source()
    _test_sink(s)

def cross_repository_anchor_sink(sink_parameter):
    if False:
        while True:
            i = 10
    pass

def test_cross_repository_anchor():
    if False:
        while True:
            i = 10
    source = _test_source()
    cross_repository_anchor_sink(source)

def test_propagate_cross_repository_source_once():
    if False:
        print('Hello World!')
    return returns_crtex_source()

def test_propagate_cross_repository_source_twice():
    if False:
        print('Hello World!')
    return test_propagate_cross_repository_source_once()

def test_propagate_cross_repository_sink_once(y):
    if False:
        i = 10
        return i + 15
    reaches_crtex_sink(y)

def test_propagate_cross_repository_sink_twice(z):
    if False:
        return 10
    test_propagate_cross_repository_sink_once(z)