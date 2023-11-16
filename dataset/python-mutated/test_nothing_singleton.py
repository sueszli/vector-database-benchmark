from returns.maybe import _Nothing

def test_nothing_singleton():
    if False:
        for i in range(10):
            print('nop')
    'Ensures `_Nothing` is a singleton.'
    assert _Nothing() is _Nothing()