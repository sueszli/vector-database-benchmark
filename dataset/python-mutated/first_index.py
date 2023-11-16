from builtins import _test_sink, _test_source

def alternate_fields():
    if False:
        return 10
    d = {'a': _test_source(), 'b': _test_source()}
    if 1 > 2:
        x = d['a']
    else:
        x = d['b']
    _test_sink(x)
    return x

def local_fields():
    if False:
        return 10
    d = alternate_fields()
    if 1 > 2:
        x = d['c']
    else:
        x = d['d']
    return x

def local_fields_hop():
    if False:
        while True:
            i = 10
    d = local_fields()
    if 1 > 2:
        x = d['e']
    else:
        x = d['f']
    return x