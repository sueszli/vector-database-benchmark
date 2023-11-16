from builtins import _test_sink, _test_source, _tito

def source_distance_zero():
    if False:
        while True:
            i = 10
    return _test_source()

def source_distance_one():
    if False:
        for i in range(10):
            print('nop')
    return source_distance_zero()

def source_distance_two():
    if False:
        for i in range(10):
            print('nop')
    return source_distance_one()

def sink_distance_zero(x):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x)

def sink_distance_one(x):
    if False:
        for i in range(10):
            print('nop')
    sink_distance_zero(x)

def sink_distance_two(x):
    if False:
        for i in range(10):
            print('nop')
    sink_distance_one(x)

def issue_source_zero_sink_zero():
    if False:
        for i in range(10):
            print('nop')
    sink_distance_zero(source_distance_zero())

def issue_source_one_sink_zero():
    if False:
        return 10
    sink_distance_zero(source_distance_one())

def issue_source_one_sink_one():
    if False:
        while True:
            i = 10
    sink_distance_one(source_distance_one())

def issue_source_two_sink_one():
    if False:
        print('Hello World!')
    sink_distance_one(source_distance_two())

def issue_source_one_sink_two():
    if False:
        return 10
    sink_distance_two(source_distance_one())

def multi_sink(x):
    if False:
        print('Hello World!')
    y = _tito(x, x.foo)
    sink_distance_one(y)