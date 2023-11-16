from builtins import _test_sink, _test_source, to_callable_target

@to_callable_target
def callable_target(arg):
    if False:
        print('Hello World!')
    _test_sink(arg)

def test_callable_target():
    if False:
        for i in range(10):
            print('nop')
    x = _test_source()
    callable_target.async_schedule(x)