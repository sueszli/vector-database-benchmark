import pytest
from xarray.util.deprecation_helpers import _deprecate_positional_args

def test_deprecate_positional_args_warns_for_function():
    if False:
        return 10

    @_deprecate_positional_args('v0.1')
    def f1(a, b, *, c='c', d='d'):
        if False:
            return 10
        return (a, b, c, d)
    result = f1(1, 2)
    assert result == (1, 2, 'c', 'd')
    result = f1(1, 2, c=3, d=4)
    assert result == (1, 2, 3, 4)
    with pytest.warns(FutureWarning, match='.*v0.1'):
        result = f1(1, 2, 3)
    assert result == (1, 2, 3, 'd')
    with pytest.warns(FutureWarning, match="Passing 'c' as positional"):
        result = f1(1, 2, 3)
    assert result == (1, 2, 3, 'd')
    with pytest.warns(FutureWarning, match="Passing 'c, d' as positional"):
        result = f1(1, 2, 3, 4)
    assert result == (1, 2, 3, 4)

    @_deprecate_positional_args('v0.1')
    def f2(a='a', *, b='b', c='c', d='d'):
        if False:
            while True:
                i = 10
        return (a, b, c, d)
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = f2(1, 2)
    assert result == (1, 2, 'c', 'd')

    @_deprecate_positional_args('v0.1')
    def f3(a, *, b='b', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return (a, b, kwargs)
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = f3(1, 2)
    assert result == (1, 2, {})
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = f3(1, 2, f='f')
    assert result == (1, 2, {'f': 'f'})

    @_deprecate_positional_args('v0.1')
    def f4(a, /, *, b='b', **kwargs):
        if False:
            print('Hello World!')
        return (a, b, kwargs)
    result = f4(1)
    assert result == (1, 'b', {})
    result = f4(1, b=2, f='f')
    assert result == (1, 2, {'f': 'f'})
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = f4(1, 2, f='f')
    assert result == (1, 2, {'f': 'f'})
    with pytest.raises(TypeError, match='Keyword-only param without default'):

        @_deprecate_positional_args('v0.1')
        def f5(a, *, b, c=3, **kwargs):
            if False:
                while True:
                    i = 10
            pass

def test_deprecate_positional_args_warns_for_class():
    if False:
        return 10

    class A1:

        @_deprecate_positional_args('v0.1')
        def method(self, a, b, *, c='c', d='d'):
            if False:
                i = 10
                return i + 15
            return (a, b, c, d)
    result = A1().method(1, 2)
    assert result == (1, 2, 'c', 'd')
    result = A1().method(1, 2, c=3, d=4)
    assert result == (1, 2, 3, 4)
    with pytest.warns(FutureWarning, match='.*v0.1'):
        result = A1().method(1, 2, 3)
    assert result == (1, 2, 3, 'd')
    with pytest.warns(FutureWarning, match="Passing 'c' as positional"):
        result = A1().method(1, 2, 3)
    assert result == (1, 2, 3, 'd')
    with pytest.warns(FutureWarning, match="Passing 'c, d' as positional"):
        result = A1().method(1, 2, 3, 4)
    assert result == (1, 2, 3, 4)

    class A2:

        @_deprecate_positional_args('v0.1')
        def method(self, a=1, b=1, *, c='c', d='d'):
            if False:
                print('Hello World!')
            return (a, b, c, d)
    with pytest.warns(FutureWarning, match="Passing 'c' as positional"):
        result = A2().method(1, 2, 3)
    assert result == (1, 2, 3, 'd')
    with pytest.warns(FutureWarning, match="Passing 'c, d' as positional"):
        result = A2().method(1, 2, 3, 4)
    assert result == (1, 2, 3, 4)

    class A3:

        @_deprecate_positional_args('v0.1')
        def method(self, a, *, b='b', **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return (a, b, kwargs)
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = A3().method(1, 2)
    assert result == (1, 2, {})
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = A3().method(1, 2, f='f')
    assert result == (1, 2, {'f': 'f'})

    class A4:

        @_deprecate_positional_args('v0.1')
        def method(self, a, /, *, b='b', **kwargs):
            if False:
                print('Hello World!')
            return (a, b, kwargs)
    result = A4().method(1)
    assert result == (1, 'b', {})
    result = A4().method(1, b=2, f='f')
    assert result == (1, 2, {'f': 'f'})
    with pytest.warns(FutureWarning, match="Passing 'b' as positional"):
        result = A4().method(1, 2, f='f')
    assert result == (1, 2, {'f': 'f'})
    with pytest.raises(TypeError, match='Keyword-only param without default'):

        class A5:

            @_deprecate_positional_args('v0.1')
            def __init__(self, a, *, b, c=3, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pass