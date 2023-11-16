from builtins import _test_sink, _test_source

def test():
    if False:
        return 10
    _test_sink(['foo', _test_source()])