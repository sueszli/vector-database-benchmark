import nni

def test_positional_only():
    if False:
        return 10

    def foo(a, b, /, c):
        if False:
            while True:
                i = 10
        pass
    d = nni.trace(foo)(1, 2, c=3)
    assert d.trace_args == [1, 2]
    assert d.trace_kwargs == dict(c=3)