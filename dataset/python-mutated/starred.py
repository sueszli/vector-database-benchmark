from builtins import _test_sink, _test_source

def sink(json):
    if False:
        while True:
            i = 10
    _test_sink(json)

def test():
    if False:
        i = 10
        return i + 15
    query = {'json': _test_source()}
    sink(query)