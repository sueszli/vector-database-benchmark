from builtins import _test_sink, _test_source

class C:

    def obscure(self, x=0, y=0):
        if False:
            print('Hello World!')
        ...

    def obscure_with_skip_overrides(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        ...

    def obscure_with_skip_inlining(self, x, y):
        if False:
            return 10
        ...

    def obscure_with_source(self, x, y):
        if False:
            while True:
                i = 10
        ...

    def obscure_with_skip_obscure(self, x, y):
        if False:
            print('Hello World!')
        ...

    def obscure_with_skip_obscure_and_tito(self, x, y):
        if False:
            i = 10
            return i + 15
        ...

    def obscure_with_multiple_models(self, x, y):
        if False:
            print('Hello World!')
        ...

    def obscure_with_tito(self, x):
        if False:
            i = 10
            return i + 15
        ...

def test_obscure(c: C):
    if False:
        i = 10
        return i + 15
    return c.obscure(0, _test_source())

def test_obscure_with_skip_overrides(c: C):
    if False:
        while True:
            i = 10
    return c.obscure_with_skip_overrides(0, _test_source())

def test_obscure_with_skip_inlining(c: C):
    if False:
        i = 10
        return i + 15
    return c.obscure_with_skip_inlining(0, _test_source())

def test_obscure_with_source(c: C):
    if False:
        for i in range(10):
            print('nop')
    return c.obscure_with_source(0, _test_source())

def test_obscure_with_skip_obscure(c: C):
    if False:
        while True:
            i = 10
    return c.obscure_with_skip_obscure(0, _test_source())

def test_obscure_with_skip_obscure_and_tito(c: C):
    if False:
        print('Hello World!')
    return c.obscure_with_skip_obscure_and_tito(0, _test_source())

def test_obscure_with_multiple_models(c: C):
    if False:
        print('Hello World!')
    return c.obscure_with_multiple_models(0, _test_source())

def test_obscure_with_tito(c: C):
    if False:
        print('Hello World!')
    _test_sink(c.obscure_with_tito(_test_source()))

def test_issue(c: C):
    if False:
        return 10
    x = _test_source()
    y = c.obscure(x)
    _test_sink(y)

def test_collapse_source(c: C):
    if False:
        return 10
    x = {'a': _test_source()}
    y = c.obscure(x)
    _test_sink(y['b'])

def test_sink_collapse(arg, c: C):
    if False:
        i = 10
        return i + 15
    x = c.obscure(arg)
    _test_sink(x['a'])

def should_collapse_depth_zero(arg, c: C):
    if False:
        while True:
            i = 10
    return c.obscure(arg)

def test_collapse_depth():
    if False:
        while True:
            i = 10
    x = {'a': _test_source()}
    y = should_collapse_depth_zero(x, C())
    _test_sink(y['b'])

def test_skip_obscure_via_model_query(arg):
    if False:
        while True:
            i = 10
    ...