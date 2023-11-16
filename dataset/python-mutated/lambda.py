from builtins import _test_sink, _test_source

def test_map_lambda(i: int):
    if False:
        return 10
    elements = list(map(lambda x: x, [_test_source()]))
    _test_sink(elements[0])
    _test_sink(elements[i])
    elements = list(map(lambda x: x, [0, _test_source(), 0]))
    _test_sink(elements[i])
    _test_sink(elements[1])
    _test_sink(elements[0])
    elements = list(map(lambda x: {'a': x, 'b': 'safe'}, [_test_source()]))
    _test_sink(elements[i])
    _test_sink(elements[i]['a'])
    _test_sink(elements[i]['b'])
    elements = list(map(lambda x: x['a'], [{'a': _test_source(), 'b': 'safe'}]))
    _test_sink(elements[i])
    elements = list(map(lambda x: x['b'], [{'a': _test_source(), 'b': 'safe'}]))
    _test_sink(elements[i])
    elements = list(map(lambda x: _test_source(), ['safe']))
    _test_sink(elements[i])
    elements = list(map(lambda x: _test_sink(x), [_test_source()]))

def test_filter_lambda(i: int):
    if False:
        return 10
    elements = list(filter(lambda x: x != 0, [_test_source()]))
    _test_sink(elements[0])
    _test_sink(elements[i])
    elements = list(filter(lambda x: x != 0, [0, _test_source(), 1]))
    _test_sink(elements[i])
    _test_sink(elements[0])
    _test_sink(elements[1])
    elements = list(filter(lambda x: x['a'], [{'a': _test_source(), 'b': 'safe'}]))
    _test_sink(elements[i])
    _test_sink(elements[i]['a'])
    _test_sink(elements[i]['b'])
    elements = list(filter(lambda x: _test_sink(x), [_test_source()]))