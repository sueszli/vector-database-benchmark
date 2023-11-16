from builtins import _test_sink, _test_source

def test_tito_regular_parameters(x, y, z):
    if False:
        return 10
    pass

def test_tito_args_kwargs(x, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

def test_tito_positional_only_parameter(__x, /):
    if False:
        for i in range(10):
            print('nop')
    pass

def test_tito_keyword_only_parameter(x, *, y):
    if False:
        print('Hello World!')
    pass

def test_tito_mix_positional_and_named_parameters(__x, /, y, *, z):
    if False:
        i = 10
        return i + 15
    pass