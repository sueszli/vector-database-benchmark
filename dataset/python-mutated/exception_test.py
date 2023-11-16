from builtins import _test_sink, _test_source

def test_parameter_flow(ex: Exception):
    if False:
        while True:
            i = 10
    return str(ex)

def test_constructed_exception():
    if False:
        i = 10
        return i + 15
    ex = Exception('message')
    return str(ex)

def test_caught_exception():
    if False:
        while True:
            i = 10
    try:
        return ''
    except Exception as ex:
        return str(ex)

def none_throws(x):
    if False:
        return 10
    if x is None:
        raise Exception('none')
    return x

def test_sink_in_finally(x):
    if False:
        print('Hello World!')
    try:
        return none_throws(x)
    finally:
        _test_sink(x)

def test_before_try_to_finally():
    if False:
        i = 10
        return i + 15
    x = _test_source()
    try:
        return none_throws(x)
    finally:
        _test_sink(x)

def test_within_try_to_finally():
    if False:
        i = 10
        return i + 15
    x = None
    try:
        x = _test_source()
        return none_throws(x)
    finally:
        _test_sink(x)

def test_except_to_finally():
    if False:
        print('Hello World!')
    x = None
    try:
        return none_throws(x)
    except:
        x = _test_source()
    finally:
        _test_sink(x)

def test_return_finally():
    if False:
        while True:
            i = 10
    try:
        print('test')
    finally:
        return _test_source()

def test_return_twice_finally():
    if False:
        return 10
    try:
        return 'hello'
    finally:
        return _test_source()

def test_return_overrides_finally():
    if False:
        for i in range(10):
            print('nop')
    try:
        return _test_source()
    finally:
        return 'hello'