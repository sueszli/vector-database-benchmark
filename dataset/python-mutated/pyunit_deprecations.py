import os
import re
import sys
import warnings
from h2o.utils.metaclass import deprecated_params, deprecated_property, deprecated_fn, fullname
sys.path.insert(1, os.path.join('..', '..'))
from tests import pyunit_utils as pu

def test_deprecated_params_without_new_param():
    if False:
        print('Hello World!')

    class Foo:

        @deprecated_params(dict(baz=None, biz=None))
        def __init__(self, foo=1, bar=2):
            if False:
                i = 10
                return i + 15
            self.foo = foo
            self.bar = bar

        @deprecated_params(dict(operator=None))
        def foobar(self, op='+'):
            if False:
                return 10
            return eval('%s %s %s' % (self.foo, op, self.bar))
    prefix = fullname(Foo.__init__)[:-8]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = Foo()
        assert f.foobar() == 3
        assert len(w) == 0
        f = Foo(foo=3, baz=5)
        assert not hasattr(f, 'baz')
        assert len(w) == 1
        assert '``baz`` param of ``{}__init__`` is deprecated and will be ignored'.format(prefix) in str(w[0].message)
        del w[:]
        assert f.foobar(operator='*') == 5
        assert len(w) == 1
        assert '``operator`` param of ``{}foobar`` is deprecated and will be ignored'.format(prefix) in str(w[0].message)

def test_deprecated_params_with_replacement():
    if False:
        return 10

    class Foo:

        @deprecated_params(dict(Foo='foo', Bar='bar'))
        def __init__(self, foo=1, bar=2):
            if False:
                return 10
            self.foo = foo
            self.bar = bar

        @deprecated_params(dict(operator='op'))
        def foobar(self, op='+'):
            if False:
                for i in range(10):
                    print('nop')
            return eval('%s %s %s' % (self.foo, op, self.bar))
    prefix = fullname(Foo.__init__)[:-8]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = Foo()
        assert f.foobar() == 3
        assert len(w) == 0
        f = Foo(foo=3, Bar=5)
        assert f.bar == 5
        assert f.foobar() == 8
        assert len(w) == 1
        assert '``Bar`` param of ``{}__init__`` is deprecated, please use ``bar`` instead'.format(prefix) in str(w[0].message)
        del w[:]
        assert f.foobar(operator='*') == 15
        assert len(w) == 1
        assert '``operator`` param of ``{}foobar`` is deprecated, please use ``op`` instead'.format(prefix) in str(w[0].message)
        del w[:]
        f_conflict = Foo(foo=3, Foo=6)
        assert f_conflict.foo == 3
        assert f_conflict.foobar() == 5
        assert len(w) == 2
        assert '``Foo`` param of ``{}__init__`` is deprecated, please use ``foo`` instead'.format(prefix) in str(w[0].message)
        assert 'Using both deprecated param ``Foo`` and new param(s) ``foo`` in call to ``{}__init__``, the deprecated param will be ignored.'.format(prefix) in str(w[1].message)
        del w[:]
        f_conflict = Foo(Foo=6, foo=3)
        assert f_conflict.foo == 3
        assert f_conflict.foobar() == 5
        assert len(w) == 2
        assert '``Foo`` param of ``{}__init__`` is deprecated, please use ``foo`` instead'.format(prefix) in str(w[0].message)
        assert 'Using both deprecated param ``Foo`` and new param(s) ``foo`` in call to ``{}__init__``, the deprecated param will be ignored.'.format(prefix) in str(w[1].message)
        del w[:]

def test_deprecated_params_message_can_be_customized():
    if False:
        return 10

    class Foo:

        @deprecated_params(dict(Foo=('foo', 'Foo custom message'), Baz=(None, 'Baz custom message')))
        def __init__(self, foo=1, bar=2):
            if False:
                while True:
                    i = 10
            self.foo = foo
            self.bar = bar
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = Foo(Foo=5, Baz=17)
        assert f.foo == 5
        assert f.bar == 2
        assert len(w) == 2
        assert str(w[0].message) == 'Foo custom message'
        assert str(w[1].message) == 'Baz custom message'

def test_deprecated_params_advanced_syntax():
    if False:
        print('Hello World!')

    class Foo:

        @deprecated_params(dict(duration_millis=(lambda millis: dict(duration=millis, unit='ms'), 'duration_millis custom message')))
        def __init__(self, duration=1, unit='s'):
            if False:
                i = 10
                return i + 15
            self.duration = duration
            self.unit = unit
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = Foo(duration_millis=600)
        assert f.duration == 600
        assert f.unit == 'ms'
        assert len(w) == 1
        assert str(w[0].message) == 'duration_millis custom message'

def test_deprecated_property():
    if False:
        print('Hello World!')

    class Foo(object):

        def __init__(self, bar=1):
            if False:
                for i in range(10):
                    print('nop')
            self._bar = bar

        @property
        def bar(self):
            if False:
                return 10
            return self._bar

        @bar.setter
        def bar(self, v):
            if False:
                i = 10
                return i + 15
            self._bar = v
        Bar = deprecated_property('Bar', replaced_by=bar)
        Baz = deprecated_property('Baz')
        Biz = deprecated_property('Biz', message='Biz custom message')
    assert Foo.Bar.__doc__ == '[Deprecated] Use ``bar`` instead'
    assert Foo.Baz.__doc__ == '[Deprecated] The property was removed and will be ignored.'
    assert Foo.Biz.__doc__ == 'Biz custom message'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = Foo(bar=5)
        assert f.bar == 5
        assert f.Bar == 5
        assert f.Baz is None
        assert f.Biz is None
        f.Bar = 7
        assert f.bar == 7
        f.Baz = 'useless'
        assert f.Baz is None
        assert len(w) == 6
        assert str(w[0].message) == '``Bar`` is deprecated, please use ``bar`` instead.'
        assert str(w[1].message) == '``Baz`` is deprecated and will be ignored.'
        assert str(w[2].message) == 'Biz custom message'
        assert str(w[3].message) == str(w[0].message)
        assert str(w[4].message) == str(w[1].message)
        assert str(w[5].message) == str(w[1].message)

def test_deprecated_function():
    if False:
        return 10

    def foo(bar=1):
        if False:
            while True:
                i = 10
        return bar * bar

    @deprecated_fn()
    def fee(baz=3):
        if False:
            i = 10
            return i + 15
        return foo(baz + 2)

    @deprecated_fn(replaced_by=foo)
    def Foo():
        if False:
            i = 10
            return i + 15
        pass

    @deprecated_fn(replaced_by=foo, msg='custom FOO message')
    def FOO():
        if False:
            print('Hello World!')
        pass

    @deprecated_fn(msg='deprecated, replaced by ``foo``')
    def foo_old_defaults(bar=2):
        if False:
            i = 10
            return i + 15
        return foo(bar=bar)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert fee() == 25
        assert Foo() == 1
        assert Foo(3) == 9
        assert FOO() == 1
        assert FOO(9) == 81
        assert foo_old_defaults() == 4
        assert foo_old_defaults(3) == 9
        assert len(w) == 7
        assert re.match('``[\\w.<>]*fee`` is deprecated.', str(w[0].message))
        assert re.match('``[\\w.<>]*Foo`` is deprecated, please use ``[\\w.<>]*foo`` instead.', str(w[1].message))
        assert str(w[3].message) == 'custom FOO message'
        assert str(w[5].message) == 'deprecated, replaced by ``foo``'
pu.run_tests([test_deprecated_params_without_new_param, test_deprecated_params_with_replacement, test_deprecated_params_message_can_be_customized, test_deprecated_params_advanced_syntax, test_deprecated_property, test_deprecated_function])