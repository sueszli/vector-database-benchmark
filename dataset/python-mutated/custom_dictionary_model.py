from builtins import _test_sink, _test_source

def _test_source_2():
    if False:
        i = 10
        return i + 15
    return "I'M TAINTED"

def untainted_dictionary():
    if False:
        while True:
            i = 10
    d = {}
    d['a'] = "I'm not tainted!"
    _test_sink(d)

def sink_dictionary_value():
    if False:
        i = 10
        return i + 15
    d = {}
    d['a'] = _test_source_2()

def sink_dictionary_key():
    if False:
        return 10
    d = {}
    d[_test_source_2()] = 'b'

def tainted_dictionary_value_sink():
    if False:
        i = 10
        return i + 15
    d = {}
    d['a'] = _test_source()
    _test_sink(d)

def tainted_dictionary_key_sink():
    if False:
        for i in range(10):
            print('nop')
    d = {}
    d[_test_source()] = 1
    _test_sink(d)