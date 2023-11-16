"""Tests for the object inspection functionality.
"""
from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
inspector = None

def setup_module():
    if False:
        i = 10
        return i + 15
    global inspector
    inspector = oinspect.Inspector()

class SourceModuleMainTest:
    __module__ = '__main__'
THIS_LINE_NUMBER = 47

def test_find_source_lines():
    if False:
        return 10
    assert oinspect.find_source_lines(test_find_source_lines) == THIS_LINE_NUMBER + 3
    assert oinspect.find_source_lines(type) is None
    assert oinspect.find_source_lines(SourceModuleMainTest) is None
    assert oinspect.find_source_lines(SourceModuleMainTest()) is None

def test_getsource():
    if False:
        for i in range(10):
            print('nop')
    assert oinspect.getsource(type) is None
    assert oinspect.getsource(SourceModuleMainTest) is None
    assert oinspect.getsource(SourceModuleMainTest()) is None

def test_inspect_getfile_raises_exception():
    if False:
        i = 10
        return i + 15
    'Check oinspect.find_file/getsource/find_source_lines expectations'
    with pytest.raises(TypeError):
        inspect.getfile(type)
    with pytest.raises(OSError if sys.version_info >= (3, 10) else TypeError):
        inspect.getfile(SourceModuleMainTest)

def pyfile(fname):
    if False:
        while True:
            i = 10
    return os.path.normcase(re.sub('.py[co]$', '.py', fname))

def match_pyfiles(f1, f2):
    if False:
        while True:
            i = 10
    assert pyfile(f1) == pyfile(f2)

def test_find_file():
    if False:
        i = 10
        return i + 15
    match_pyfiles(oinspect.find_file(test_find_file), os.path.abspath(__file__))
    assert oinspect.find_file(type) is None
    assert oinspect.find_file(SourceModuleMainTest) is None
    assert oinspect.find_file(SourceModuleMainTest()) is None

def test_find_file_decorated1():
    if False:
        for i in range(10):
            print('nop')

    @decorator
    def noop1(f):
        if False:
            while True:
                i = 10

        def wrapper(*a, **kw):
            if False:
                return 10
            return f(*a, **kw)
        return wrapper

    @noop1
    def f(x):
        if False:
            while True:
                i = 10
        'My docstring'
    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == 'My docstring'

def test_find_file_decorated2():
    if False:
        i = 10
        return i + 15

    @decorator
    def noop2(f, *a, **kw):
        if False:
            i = 10
            return i + 15
        return f(*a, **kw)

    @noop2
    @noop2
    @noop2
    def f(x):
        if False:
            print('Hello World!')
        'My docstring 2'
    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == 'My docstring 2'

def test_find_file_magic():
    if False:
        i = 10
        return i + 15
    run = ip.find_line_magic('run')
    assert oinspect.find_file(run) is not None

class Call(object):
    """This is the class docstring."""

    def __init__(self, x, y=1):
        if False:
            for i in range(10):
                print('nop')
        'This is the constructor docstring.'

    def __call__(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        'This is the call docstring.'

    def method(self, x, z=2):
        if False:
            print('Hello World!')
        "Some method's docstring"

class HasSignature(object):
    """This is the class docstring."""
    __signature__ = Signature([Parameter('test', Parameter.POSITIONAL_OR_KEYWORD)])

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        'This is the init docstring'

class SimpleClass(object):

    def method(self, x, z=2):
        if False:
            i = 10
            return i + 15
        "Some method's docstring"

class Awkward(object):

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        raise Exception(name)

class NoBoolCall:
    """
    callable with `__bool__` raising should still be inspect-able.
    """

    def __call__(self):
        if False:
            return 10
        'does nothing'
        pass

    def __bool__(self):
        if False:
            return 10
        'just raise NotImplemented'
        raise NotImplementedError('Must be implemented')

class SerialLiar(object):
    """Attribute accesses always get another copy of the same class.

    unittest.mock.call does something similar, but it's not ideal for testing
    as the failure mode is to eat all your RAM. This gives up after 10k levels.
    """

    def __init__(self, max_fibbing_twig, lies_told=0):
        if False:
            return 10
        if lies_told > 10000:
            raise RuntimeError('Nose too long, honesty is the best policy')
        self.max_fibbing_twig = max_fibbing_twig
        self.lies_told = lies_told
        max_fibbing_twig[0] = max(max_fibbing_twig[0], lies_told)

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        return SerialLiar(self.max_fibbing_twig, self.lies_told + 1)

def test_info():
    if False:
        return 10
    'Check that Inspector.info fills out various fields as expected.'
    i = inspector.info(Call, oname='Call')
    assert i['type_name'] == 'type'
    expected_class = str(type(type))
    assert i['base_class'] == expected_class
    assert re.search("<class 'IPython.core.tests.test_oinspect.Call'( at 0x[0-9a-f]{1,9})?>", i['string_form'])
    fname = __file__
    if fname.endswith('.pyc'):
        fname = fname[:-1]
    assert i['file'].lower() == compress_user(fname).lower()
    assert i['definition'] == None
    assert i['docstring'] == Call.__doc__
    assert i['source'] == None
    assert i['isclass'] is True
    assert i['init_definition'] == 'Call(x, y=1)'
    assert i['init_docstring'] == Call.__init__.__doc__
    i = inspector.info(Call, detail_level=1)
    assert i['source'] is not None
    assert i['docstring'] == None
    c = Call(1)
    c.__doc__ = 'Modified instance docstring'
    i = inspector.info(c)
    assert i['type_name'] == 'Call'
    assert i['docstring'] == 'Modified instance docstring'
    assert i['class_docstring'] == Call.__doc__
    assert i['init_docstring'] == Call.__init__.__doc__
    assert i['call_docstring'] == Call.__call__.__doc__

def test_class_signature():
    if False:
        i = 10
        return i + 15
    info = inspector.info(HasSignature, 'HasSignature')
    assert info['init_definition'] == 'HasSignature(test)'
    assert info['init_docstring'] == HasSignature.__init__.__doc__

def test_info_awkward():
    if False:
        for i in range(10):
            print('nop')
    inspector.info(Awkward())

def test_bool_raise():
    if False:
        for i in range(10):
            print('nop')
    inspector.info(NoBoolCall())

def test_info_serialliar():
    if False:
        i = 10
        return i + 15
    fib_tracker = [0]
    inspector.info(SerialLiar(fib_tracker))
    assert fib_tracker[0] < 9000

def support_function_one(x, y=2, *a, **kw):
    if False:
        i = 10
        return i + 15
    'A simple function.'

def test_calldef_none():
    if False:
        return 10
    for obj in [support_function_one, SimpleClass().method, any, str.upper]:
        i = inspector.info(obj)
        assert i['call_def'] is None

def f_kwarg(pos, *, kwonly):
    if False:
        i = 10
        return i + 15
    pass

def test_definition_kwonlyargs():
    if False:
        print('Hello World!')
    i = inspector.info(f_kwarg, oname='f_kwarg')
    assert i['definition'] == 'f_kwarg(pos, *, kwonly)'

def test_getdoc():
    if False:
        for i in range(10):
            print('nop')

    class A(object):
        """standard docstring"""
        pass

    class B(object):
        """standard docstring"""

        def getdoc(self):
            if False:
                i = 10
                return i + 15
            return 'custom docstring'

    class C(object):
        """standard docstring"""

        def getdoc(self):
            if False:
                return 10
            return None
    a = A()
    b = B()
    c = C()
    assert oinspect.getdoc(a) == 'standard docstring'
    assert oinspect.getdoc(b) == 'custom docstring'
    assert oinspect.getdoc(c) == 'standard docstring'

def test_empty_property_has_no_source():
    if False:
        while True:
            i = 10
    i = inspector.info(property(), detail_level=1)
    assert i['source'] is None

def test_property_sources():
    if False:
        print('Hello World!')

    def simple_add(a, b):
        if False:
            for i in range(10):
                print('nop')
        'Adds two numbers'
        return a + b

    class A(object):

        @property
        def foo(self):
            if False:
                while True:
                    i = 10
            return 'bar'
        foo = foo.setter(lambda self, v: setattr(self, 'bar', v))
        dname = property(oinspect.getdoc)
        adder = property(simple_add)
    i = inspector.info(A.foo, detail_level=1)
    assert 'def foo(self):' in i['source']
    assert 'lambda self, v:' in i['source']
    i = inspector.info(A.dname, detail_level=1)
    assert 'def getdoc(obj)' in i['source']
    i = inspector.info(A.adder, detail_level=1)
    assert 'def simple_add(a, b)' in i['source']

def test_property_docstring_is_in_info_for_detail_level_0():
    if False:
        print('Hello World!')

    class A(object):

        @property
        def foobar(self):
            if False:
                print('Hello World!')
            'This is `foobar` property.'
            pass
    ip.user_ns['a_obj'] = A()
    assert 'This is `foobar` property.' == ip.object_inspect('a_obj.foobar', detail_level=0)['docstring']
    ip.user_ns['a_cls'] = A
    assert 'This is `foobar` property.' == ip.object_inspect('a_cls.foobar', detail_level=0)['docstring']

def test_pdef():
    if False:
        while True:
            i = 10

    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    inspector.pdef(foo, 'foo')

@contextmanager
def cleanup_user_ns(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    On exit delete all the keys that were not in user_ns before entering.\n\n    It does not restore old values !\n\n    Parameters\n    ----------\n\n    **kwargs\n        used to update ip.user_ns\n\n    '
    try:
        known = set(ip.user_ns.keys())
        ip.user_ns.update(kwargs)
        yield
    finally:
        added = set(ip.user_ns.keys()) - known
        for k in added:
            del ip.user_ns[k]

def test_pinfo_bool_raise():
    if False:
        while True:
            i = 10
    '\n    Test that bool method is not called on parent.\n    '

    class RaiseBool:
        attr = None

        def __bool__(self):
            if False:
                print('Hello World!')
            raise ValueError('pinfo should not access this method')
    raise_bool = RaiseBool()
    with cleanup_user_ns(raise_bool=raise_bool):
        ip._inspect('pinfo', 'raise_bool.attr', detail_level=0)

def test_pinfo_getindex():
    if False:
        for i in range(10):
            print('nop')

    def dummy():
        if False:
            while True:
                i = 10
        '\n        MARKER\n        '
    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints('MARKER'):
            ip._inspect('pinfo', 'container[0]', detail_level=0)
    assert 'container' not in ip.user_ns.keys()

def test_qmark_getindex():
    if False:
        while True:
            i = 10

    def dummy():
        if False:
            i = 10
            return i + 15
        '\n        MARKER 2\n        '
    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints('MARKER 2'):
            ip.run_cell('container[0]?')
    assert 'container' not in ip.user_ns.keys()

def test_qmark_getindex_negatif():
    if False:
        i = 10
        return i + 15

    def dummy():
        if False:
            while True:
                i = 10
        '\n        MARKER 3\n        '
    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints('MARKER 3'):
            ip.run_cell('container[-1]?')
    assert 'container' not in ip.user_ns.keys()

def test_pinfo_nonascii():
    if False:
        print('Hello World!')
    from . import nonascii2
    ip.user_ns['nonascii2'] = nonascii2
    ip._inspect('pinfo', 'nonascii2', detail_level=1)

def test_pinfo_type():
    if False:
        while True:
            i = 10
    '\n    type can fail in various edge case, for example `type.__subclass__()`\n    '
    ip._inspect('pinfo', 'type')

def test_pinfo_docstring_no_source():
    if False:
        for i in range(10):
            print('nop')
    'Docstring should be included with detail_level=1 if there is no source'
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=0)
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=1)

def test_pinfo_no_docstring_if_source():
    if False:
        print('Hello World!')
    'Docstring should not be included with detail_level=1 if source is found'

    def foo():
        if False:
            while True:
                i = 10
        'foo has a docstring'
    ip.user_ns['foo'] = foo
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=0)
    with AssertPrints('Source:'):
        ip._inspect('pinfo', 'foo', detail_level=1)
    with AssertNotPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=1)

def test_pinfo_docstring_if_detail_and_no_source():
    if False:
        print('Hello World!')
    ' Docstring should be displayed if source info not available '
    obj_def = 'class Foo(object):\n                  """ This is a docstring for Foo """\n                  def bar(self):\n                      """ This is a docstring for Foo.bar """\n                      pass\n              '
    ip.run_cell(obj_def)
    ip.run_cell('foo = Foo()')
    with AssertNotPrints('Source:'):
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=0)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=1)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=0)
    with AssertNotPrints('Docstring:'):
        with AssertPrints('Source:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=1)

def test_pinfo_docstring_dynamic():
    if False:
        while True:
            i = 10
    obj_def = 'class Bar:\n    __custom_documentations__ = {\n     "prop" : "cdoc for prop",\n     "non_exist" : "cdoc for non_exist",\n    }\n    @property\n    def prop(self):\n        \'\'\'\n        Docstring for prop\n        \'\'\'\n        return self._prop\n    \n    @prop.setter\n    def prop(self, v):\n        self._prop = v\n    '
    ip.run_cell(obj_def)
    ip.run_cell('b = Bar()')
    with AssertPrints('Docstring:   cdoc for prop'):
        ip.run_line_magic('pinfo', 'b.prop')
    with AssertPrints('Docstring:   cdoc for non_exist'):
        ip.run_line_magic('pinfo', 'b.non_exist')
    with AssertPrints('Docstring:   cdoc for prop'):
        ip.run_cell('b.prop?')
    with AssertPrints('Docstring:   cdoc for non_exist'):
        ip.run_cell('b.non_exist?')
    with AssertPrints('Docstring:   <no docstring>'):
        ip.run_cell('b.undefined?')

def test_pinfo_magic():
    if False:
        print('Hello World!')
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'lsmagic', detail_level=0)
    with AssertPrints('Source:'):
        ip._inspect('pinfo', 'lsmagic', detail_level=1)

def test_init_colors():
    if False:
        while True:
            i = 10
    info = inspector.info(HasSignature)
    init_def = info['init_definition']
    assert '[0m' not in init_def

def test_builtin_init():
    if False:
        return 10
    info = inspector.info(list)
    init_def = info['init_definition']
    assert init_def is not None

def test_render_signature_short():
    if False:
        return 10

    def short_fun(a=1):
        if False:
            print('Hello World!')
        pass
    sig = oinspect._render_signature(signature(short_fun), short_fun.__name__)
    assert sig == 'short_fun(a=1)'

def test_render_signature_long():
    if False:
        i = 10
        return i + 15
    from typing import Optional

    def long_function(a_really_long_parameter: int, and_another_long_one: bool=False, let_us_make_sure_this_is_looong: Optional[str]=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass
    sig = oinspect._render_signature(signature(long_function), long_function.__name__)
    expected = 'long_function(\n    a_really_long_parameter: int,\n    and_another_long_one: bool = False,\n    let_us_make_sure_this_is_looong: Optional[str] = None,\n) -> bool'
    assert sig == expected