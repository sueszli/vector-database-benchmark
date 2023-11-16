import vaex
import pytest

class Foo(object):

    def __init__(self, df):
        if False:
            return 10
        self.df = df

class Spam(object):

    def __init__(self, df):
        if False:
            while True:
                i = 10
        self.df = df

class Egg(object):

    def __init__(self, spam):
        if False:
            for i in range(10):
                print('nop')
        self.spam = spam
        self.df = spam.df

def test_accessor_basic():
    if False:
        for i in range(10):
            print('nop')
    vaex._add_lazy_accessor('foo', lambda : Foo)
    df = vaex.example()
    assert isinstance(df.foo, Foo)
    assert df.foo is df.foo
    assert df.foo.df is df

def test_accessor_expression():
    if False:
        i = 10
        return i + 15
    vaex._add_lazy_accessor('foo', lambda : Foo, vaex.expression.Expression)
    df = vaex.example()
    assert isinstance(df.x.foo, Foo)
    assert df.x.foo is df.x.foo
    assert df.x.foo.df is df.x

def test_accessor_nested():
    if False:
        print('Hello World!')
    df = vaex.example()
    vaex._add_lazy_accessor('spam.egg', lambda : Egg)
    with pytest.raises(expected_exception=AttributeError):
        a = df.spam
    vaex._add_lazy_accessor('spam.egg.foo', lambda : Foo)
    with pytest.raises(expected_exception=AttributeError):
        a = df.spam
    vaex._add_lazy_accessor('spam', lambda : Spam)
    assert df.spam is df.spam
    assert df.spam.df is df
    assert isinstance(df.spam, Spam)
    assert df.spam.egg is df.spam.egg
    assert df.spam.egg.spam is df.spam
    assert isinstance(df.spam.egg, Egg)
    assert df.spam.egg.foo is df.spam.egg.foo
    assert df.spam.egg.foo.df is df.spam.egg
    assert isinstance(df.spam.egg.foo, Foo)