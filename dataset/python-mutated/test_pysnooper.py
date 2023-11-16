import io
import textwrap
import threading
import time
import types
import os
import sys
import zipfile
from pysnooper.utils import truncate
import pytest
import pysnooper
from pysnooper.variables import needs_parentheses
from .utils import assert_output, assert_sample_output, VariableEntry, CallEntry, LineEntry, ReturnEntry, OpcodeEntry, ReturnValueEntry, ExceptionEntry, ExceptionValueEntry, SourcePathEntry, CallEndedByExceptionEntry, ElapsedTimeEntry
from . import mini_toolbox

def test_string_io():
    if False:
        print('Hello World!')
    string_io = io.StringIO()

    @pysnooper.snoop(string_io, color=False)
    def my_function(foo):
        if False:
            for i in range(10):
                print('nop')
        x = 7
        y = 8
        return y + x
    result = my_function('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()))

def test_relative_time():
    if False:
        i = 10
        return i + 15
    snoop = pysnooper.snoop(relative_time=True, color=False)

    def foo(x):
        if False:
            for i in range(10):
                print('nop')
        if x == 0:
            bar1(x)
            qux()
            return
        with snoop:
            foo(x - 1)
            bar2(x)
            qux()
        int(4)
        bar3(9)
        return x

    @snoop
    def bar1(_x):
        if False:
            for i in range(10):
                print('nop')
        qux()

    @snoop
    def bar2(_x):
        if False:
            for i in range(10):
                print('nop')
        qux()

    @snoop
    def bar3(_x):
        if False:
            for i in range(10):
                print('nop')
        qux()

    def qux():
        if False:
            return 10
        time.sleep(0.1)
        return 9
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = foo(2)
    assert result == 2
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x', '2'), VariableEntry('bar1'), VariableEntry('bar2'), VariableEntry('bar3'), VariableEntry('foo'), VariableEntry('qux'), VariableEntry('snoop'), LineEntry('foo(x - 1)'), VariableEntry('x', '1'), VariableEntry('bar1'), VariableEntry('bar2'), VariableEntry('bar3'), VariableEntry('foo'), VariableEntry('qux'), VariableEntry('snoop'), LineEntry('foo(x - 1)'), VariableEntry('_x', '0'), VariableEntry('qux'), CallEntry('def bar1(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(0.1), LineEntry('bar2(x)'), VariableEntry('_x', '1'), VariableEntry('qux'), CallEntry('def bar2(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(0.1), LineEntry('qux()'), LineEntry(source_regex='with snoop:', min_python_version=(3, 10)), ElapsedTimeEntry(0.4), VariableEntry('_x', '9'), VariableEntry('qux'), CallEntry('def bar3(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(0.1), LineEntry('bar2(x)'), VariableEntry('_x', '2'), VariableEntry('qux'), CallEntry('def bar2(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(0.1), LineEntry('qux()'), LineEntry(source_regex='with snoop:', min_python_version=(3, 10)), ElapsedTimeEntry(0.7), VariableEntry('_x', '9'), VariableEntry('qux'), CallEntry('def bar3(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(0.1)))

def test_thread_info():
    if False:
        print('Hello World!')

    @pysnooper.snoop(thread_info=True, color=False)
    def my_function(foo):
        if False:
            i = 10
            return i + 15
        x = 7
        y = 8
        return y + x
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function('baba')
    assert result == 15
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()))

def test_multi_thread_info():
    if False:
        return 10

    @pysnooper.snoop(thread_info=True, color=False)
    def my_function(foo):
        if False:
            i = 10
            return i + 15
        x = 7
        y = 8
        return y + x

    def parse_call_content(line):
        if False:
            while True:
                i = 10
        return line.split('{event:9} '.format(event='call'))[-1]
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        my_function('baba')
        t1 = threading.Thread(target=my_function, name='test123', args=['bubu'])
        t1.start()
        t1.join()
        t1 = threading.Thread(target=my_function, name='bibi', args=['bibi'])
        t1.start()
        t1.join()
    output = output_capturer.string_io.getvalue()
    calls = [line for line in output.split('\n') if 'call' in line]
    main_thread = calls[0]
    assert parse_call_content(main_thread) == parse_call_content(calls[1])
    assert parse_call_content(main_thread) == parse_call_content(calls[2])
    thread_info_regex = '([0-9]+-{name}+[ ]+)'
    assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):', thread_info_regex=thread_info_regex.format(name='MainThread')), LineEntry('x = 7', thread_info_regex=thread_info_regex.format(name='MainThread')), VariableEntry('x', '7'), LineEntry('y = 8', thread_info_regex=thread_info_regex.format(name='MainThread')), VariableEntry('y', '8'), LineEntry('return y + x', thread_info_regex=thread_info_regex.format(name='MainThread')), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry(), VariableEntry('foo', value_regex="u?'bubu'"), CallEntry('def my_function(foo):', thread_info_regex=thread_info_regex.format(name='test123')), LineEntry('x = 7', thread_info_regex=thread_info_regex.format(name='test123')), VariableEntry('x', '7'), LineEntry('y = 8', thread_info_regex=thread_info_regex.format(name='test123')), VariableEntry('y', '8'), LineEntry('return y + x', thread_info_regex=thread_info_regex.format(name='test123')), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry(), VariableEntry('foo', value_regex="u?'bibi'"), CallEntry('def my_function(foo):', thread_info_regex=thread_info_regex.format(name='bibi')), LineEntry('x = 7', thread_info_regex=thread_info_regex.format(name='bibi')), VariableEntry('x', '7'), LineEntry('y = 8', thread_info_regex=thread_info_regex.format(name='bibi')), VariableEntry('y', '8'), LineEntry('return y + x', thread_info_regex=thread_info_regex.format(name='bibi')), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()))

@pytest.mark.parametrize('normalize', (True, False))
def test_callable(normalize):
    if False:
        while True:
            i = 10
    string_io = io.StringIO()

    def write(msg):
        if False:
            print('Hello World!')
        string_io.write(msg)

    @pysnooper.snoop(write, normalize=normalize, color=False)
    def my_function(foo):
        if False:
            print('Hello World!')
        x = 7
        y = 8
        return y + x
    result = my_function('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_watch(normalize):
    if False:
        for i in range(10):
            print('nop')

    class Foo(object):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.x = 2

        def square(self):
            if False:
                while True:
                    i = 10
            self.x **= 2

    @pysnooper.snoop(watch=('foo.x', 'io.__name__', 'len(foo.__dict__["x"] * "abc")'), normalize=normalize, color=False)
    def my_function():
        if False:
            print('Hello World!')
        foo = Foo()
        for i in range(2):
            foo.square()
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('Foo'), VariableEntry('io.__name__', "'io'"), CallEntry('def my_function():'), LineEntry('foo = Foo()'), VariableEntry('foo'), VariableEntry('foo.x', '2'), VariableEntry('len(foo.__dict__["x"] * "abc")', '6'), LineEntry(), VariableEntry('i', '0'), LineEntry(), VariableEntry('foo.x', '4'), VariableEntry('len(foo.__dict__["x"] * "abc")', '12'), LineEntry(), VariableEntry('i', '1'), LineEntry(), VariableEntry('foo.x', '16'), VariableEntry('len(foo.__dict__["x"] * "abc")', '48'), LineEntry(), ReturnEntry(), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_watch_explode(normalize):
    if False:
        i = 10
        return i + 15

    class Foo:

        def __init__(self, x, y):
            if False:
                while True:
                    i = 10
            self.x = x
            self.y = y

    @pysnooper.snoop(watch_explode=('_d', '_point', 'lst + []'), normalize=normalize, color=False)
    def my_function():
        if False:
            for i in range(10):
                print('nop')
        _d = {'a': 1, 'b': 2, 'c': 'ignore'}
        _point = Foo(x=3, y=4)
        lst = [7, 8, 9]
        lst.append(10)
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('Foo'), CallEntry('def my_function():'), LineEntry(), VariableEntry('_d'), VariableEntry("_d['a']", '1'), VariableEntry("_d['b']", '2'), VariableEntry("_d['c']", "'ignore'"), LineEntry(), VariableEntry('_point'), VariableEntry('_point.x', '3'), VariableEntry('_point.y', '4'), LineEntry(), VariableEntry('lst'), VariableEntry('(lst + [])[0]', '7'), VariableEntry('(lst + [])[1]', '8'), VariableEntry('(lst + [])[2]', '9'), VariableEntry('lst + []'), LineEntry(), VariableEntry('lst'), VariableEntry('(lst + [])[3]', '10'), VariableEntry('lst + []'), ReturnEntry(), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_variables_classes(normalize):
    if False:
        print('Hello World!')

    class WithSlots(object):
        __slots__ = ('x', 'y')

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.x = 3
            self.y = 4

    @pysnooper.snoop(watch=(pysnooper.Keys('_d', exclude='c'), pysnooper.Attrs('_d'), pysnooper.Attrs('_s'), pysnooper.Indices('_lst')[-3:]), normalize=normalize, color=False)
    def my_function():
        if False:
            print('Hello World!')
        _d = {'a': 1, 'b': 2, 'c': 'ignore'}
        _s = WithSlots()
        _lst = list(range(1000))
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('WithSlots'), CallEntry('def my_function():'), LineEntry(), VariableEntry('_d'), VariableEntry("_d['a']", '1'), VariableEntry("_d['b']", '2'), LineEntry(), VariableEntry('_s'), VariableEntry('_s.x', '3'), VariableEntry('_s.y', '4'), LineEntry(), VariableEntry('_lst'), VariableEntry('_lst[997]', '997'), VariableEntry('_lst[998]', '998'), VariableEntry('_lst[999]', '999'), ReturnEntry(), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_single_watch_no_comma(normalize):
    if False:
        print('Hello World!')

    class Foo(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.x = 2

        def square(self):
            if False:
                print('Hello World!')
            self.x **= 2

    @pysnooper.snoop(watch='foo', normalize=normalize, color=False)
    def my_function():
        if False:
            while True:
                i = 10
        foo = Foo()
        for i in range(2):
            foo.square()
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('Foo'), CallEntry('def my_function():'), LineEntry('foo = Foo()'), VariableEntry('foo'), LineEntry(), VariableEntry('i', '0'), LineEntry(), LineEntry(), VariableEntry('i', '1'), LineEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_long_variable(normalize):
    if False:
        i = 10
        return i + 15

    @pysnooper.snoop(normalize=normalize, color=False)
    def my_function():
        if False:
            print('Hello World!')
        foo = list(range(1000))
        return foo
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result == list(range(1000))
    output = output_capturer.string_io.getvalue()
    regex = '^(?=.{100}$)\\[0, 1, 2, .*\\.\\.\\..*, 997, 998, 999\\]$'
    assert_output(output, (SourcePathEntry(), CallEntry('def my_function():'), LineEntry('foo = list(range(1000))'), VariableEntry('foo', value_regex=regex), LineEntry(), ReturnEntry(), ReturnValueEntry(value_regex=regex), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_long_variable_with_custom_max_variable_length(normalize):
    if False:
        return 10

    @pysnooper.snoop(max_variable_length=200, normalize=normalize, color=False)
    def my_function():
        if False:
            i = 10
            return i + 15
        foo = list(range(1000))
        return foo
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result == list(range(1000))
    output = output_capturer.string_io.getvalue()
    regex = '^(?=.{200}$)\\[0, 1, 2, .*\\.\\.\\..*, 997, 998, 999\\]$'
    assert_output(output, (SourcePathEntry(), CallEntry('def my_function():'), LineEntry('foo = list(range(1000))'), VariableEntry('foo', value_regex=regex), LineEntry(), ReturnEntry(), ReturnValueEntry(value_regex=regex), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_long_variable_with_infinite_max_variable_length(normalize):
    if False:
        return 10

    @pysnooper.snoop(max_variable_length=None, normalize=normalize, color=False)
    def my_function():
        if False:
            i = 10
            return i + 15
        foo = list(range(1000))
        return foo
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result == list(range(1000))
    output = output_capturer.string_io.getvalue()
    regex = '^(?=.{1000,100000}$)\\[0, 1, 2, [^.]+ 997, 998, 999\\]$'
    assert_output(output, (SourcePathEntry(), CallEntry('def my_function():'), LineEntry('foo = list(range(1000))'), VariableEntry('foo', value_regex=regex), LineEntry(), ReturnEntry(), ReturnValueEntry(value_regex=regex), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_repr_exception(normalize):
    if False:
        while True:
            i = 10

    class Bad(object):

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            1 / 0

    @pysnooper.snoop(normalize=normalize, color=False)
    def my_function():
        if False:
            for i in range(10):
                print('nop')
        bad = Bad()
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('Bad'), CallEntry('def my_function():'), LineEntry('bad = Bad()'), VariableEntry('bad', value='REPR FAILED'), ReturnEntry(), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_depth(normalize):
    if False:
        while True:
            i = 10
    string_io = io.StringIO()

    def f4(x4):
        if False:
            return 10
        result4 = x4 * 2
        return result4

    def f3(x3):
        if False:
            while True:
                i = 10
        result3 = f4(x3)
        return result3

    def f2(x2):
        if False:
            while True:
                i = 10
        result2 = f3(x2)
        return result2

    @pysnooper.snoop(string_io, depth=3, normalize=normalize, color=False)
    def f1(x1):
        if False:
            i = 10
            return i + 15
        result1 = f2(x1)
        return result1
    result = f1(10)
    assert result == 20
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry(), VariableEntry(), CallEntry('def f1(x1):'), LineEntry(), VariableEntry(), VariableEntry(), CallEntry('def f2(x2):'), LineEntry(), VariableEntry(), VariableEntry(), CallEntry('def f3(x3):'), LineEntry(), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('20'), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('20'), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('20'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_method_and_prefix(normalize):
    if False:
        i = 10
        return i + 15

    class Baz(object):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.x = 2

        @pysnooper.snoop(watch=('self.x',), prefix='ZZZ', normalize=normalize, color=False)
        def square(self):
            if False:
                print('Hello World!')
            foo = 7
            self.x **= 2
            return self
    baz = Baz()
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = baz.square()
    assert result is baz
    assert result.x == 4
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(prefix='ZZZ'), VariableEntry('self', prefix='ZZZ'), VariableEntry('self.x', '2', prefix='ZZZ'), CallEntry('def square(self):', prefix='ZZZ'), LineEntry('foo = 7', prefix='ZZZ'), VariableEntry('foo', '7', prefix='ZZZ'), LineEntry('self.x **= 2', prefix='ZZZ'), VariableEntry('self.x', '4', prefix='ZZZ'), LineEntry(prefix='ZZZ'), ReturnEntry(prefix='ZZZ'), ReturnValueEntry(prefix='ZZZ'), ElapsedTimeEntry(prefix='ZZZ')), prefix='ZZZ', normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_file_output(normalize):
    if False:
        while True:
            i = 10
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder:
        path = folder / 'foo.log'

        @pysnooper.snoop(path, normalize=normalize, color=False)
        def my_function(_foo):
            if False:
                while True:
                    i = 10
            x = 7
            y = 8
            return y + x
        result = my_function('baba')
        assert result == 15
        with path.open() as output_file:
            output = output_file.read()
        assert_output(output, (SourcePathEntry(), VariableEntry('_foo', value_regex="u?'baba'"), CallEntry('def my_function(_foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_confusing_decorator_lines(normalize):
    if False:
        return 10
    string_io = io.StringIO()

    def empty_decorator(function):
        if False:
            return 10
        return function

    @empty_decorator
    @pysnooper.snoop(string_io, normalize=normalize, depth=2, color=False)
    @empty_decorator
    @empty_decorator
    def my_function(foo):
        if False:
            i = 10
            return i + 15
        x = lambda bar: 7
        y = 8
        return y + x(foo)
    result = my_function('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry(), VariableEntry(), LineEntry(), VariableEntry(), LineEntry(), VariableEntry('bar', value_regex="u?'baba'"), CallEntry('x = lambda bar: 7'), LineEntry(), ReturnEntry(), ReturnValueEntry('7'), ReturnEntry(), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_lambda(normalize):
    if False:
        print('Hello World!')
    string_io = io.StringIO()
    my_function = pysnooper.snoop(string_io, normalize=normalize, color=False)(lambda x: x ** 2)
    result = my_function(7)
    assert result == 49
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x', '7'), CallEntry(source_regex='^my_function = pysnooper.*'), LineEntry(source_regex='^my_function = pysnooper.*'), ReturnEntry(source_regex='^my_function = pysnooper.*'), ReturnValueEntry('49'), ElapsedTimeEntry()), normalize=normalize)

def test_unavailable_source():
    if False:
        for i in range(10):
            print('nop')
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder, mini_toolbox.TempSysPathAdder(str(folder)):
        module_name = 'iaerojajsijf'
        python_file_path = folder / ('%s.py' % (module_name,))
        content = textwrap.dedent(u'\n            import pysnooper\n            @pysnooper.snoop(color=False)\n            def f(x):\n                return x\n        ')
        with python_file_path.open('w') as python_file:
            python_file.write(content)
        module = __import__(module_name)
        python_file_path.unlink()
        with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
            result = getattr(module, 'f')(7)
        assert result == 7
        output = output_capturer.output
        assert_output(output, (SourcePathEntry(), VariableEntry(stage='starting'), CallEntry('SOURCE IS UNAVAILABLE'), LineEntry('SOURCE IS UNAVAILABLE'), ReturnEntry('SOURCE IS UNAVAILABLE'), ReturnValueEntry('7'), ElapsedTimeEntry()))

def test_no_overwrite_by_default():
    if False:
        return 10
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder:
        path = folder / 'foo.log'
        with path.open('w') as output_file:
            output_file.write(u'lala')

        @pysnooper.snoop(str(path), color=False)
        def my_function(foo):
            if False:
                i = 10
                return i + 15
            x = 7
            y = 8
            return y + x
        result = my_function('baba')
        assert result == 15
        with path.open() as output_file:
            output = output_file.read()
        assert output.startswith('lala')
        shortened_output = output[4:]
        assert_output(shortened_output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()))

def test_overwrite():
    if False:
        for i in range(10):
            print('nop')
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder:
        path = folder / 'foo.log'
        with path.open('w') as output_file:
            output_file.write(u'lala')

        @pysnooper.snoop(str(path), overwrite=True, color=False)
        def my_function(foo):
            if False:
                print('Hello World!')
            x = 7
            y = 8
            return y + x
        result = my_function('baba')
        result = my_function('baba')
        assert result == 15
        with path.open() as output_file:
            output = output_file.read()
        assert 'lala' not in output
        assert_output(output, (SourcePathEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry(), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_function(foo):'), LineEntry('x = 7'), VariableEntry('x', '7'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + x'), ReturnEntry('return y + x'), ReturnValueEntry('15'), ElapsedTimeEntry()))

def test_error_in_overwrite_argument():
    if False:
        while True:
            i = 10
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder:
        with pytest.raises(Exception, match='can only be used when writing'):

            @pysnooper.snoop(overwrite=True, color=False)
            def my_function(foo):
                if False:
                    return 10
                x = 7
                y = 8
                return y + x

def test_needs_parentheses():
    if False:
        print('Hello World!')
    assert not needs_parentheses('x')
    assert not needs_parentheses('x.y')
    assert not needs_parentheses('x.y.z')
    assert not needs_parentheses('x.y.z[0]')
    assert not needs_parentheses('x.y.z[0]()')
    assert not needs_parentheses('x.y.z[0]()(3, 4 * 5)')
    assert not needs_parentheses('foo(x)')
    assert not needs_parentheses('foo(x+y)')
    assert not needs_parentheses('(x+y)')
    assert not needs_parentheses('[x+1 for x in ()]')
    assert needs_parentheses('x + y')
    assert needs_parentheses('x * y')
    assert needs_parentheses('x and y')
    assert needs_parentheses('x if z else y')

@pytest.mark.parametrize('normalize', (True, False))
def test_with_block(normalize):
    if False:
        i = 10
        return i + 15
    snoop = pysnooper.snoop(normalize=normalize, color=False)

    def foo(x):
        if False:
            return 10
        if x == 0:
            bar1(x)
            qux()
            return
        with snoop:
            foo(x - 1)
            bar2(x)
            qux()
        int(4)
        bar3(9)
        return x

    @snoop
    def bar1(_x):
        if False:
            print('Hello World!')
        qux()

    @snoop
    def bar2(_x):
        if False:
            for i in range(10):
                print('nop')
        qux()

    @snoop
    def bar3(_x):
        if False:
            print('Hello World!')
        qux()

    def qux():
        if False:
            i = 10
            return i + 15
        return 9
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = foo(2)
    assert result == 2
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x', '2'), VariableEntry('bar1'), VariableEntry('bar2'), VariableEntry('bar3'), VariableEntry('foo'), VariableEntry('qux'), VariableEntry('snoop'), LineEntry('foo(x - 1)'), VariableEntry('x', '1'), VariableEntry('bar1'), VariableEntry('bar2'), VariableEntry('bar3'), VariableEntry('foo'), VariableEntry('qux'), VariableEntry('snoop'), LineEntry('foo(x - 1)'), VariableEntry('_x', '0'), VariableEntry('qux'), CallEntry('def bar1(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(), LineEntry('bar2(x)'), VariableEntry('_x', '1'), VariableEntry('qux'), CallEntry('def bar2(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(), LineEntry('qux()'), LineEntry(source_regex='with snoop:', min_python_version=(3, 10)), ElapsedTimeEntry(), VariableEntry('_x', '9'), VariableEntry('qux'), CallEntry('def bar3(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(), LineEntry('bar2(x)'), VariableEntry('_x', '2'), VariableEntry('qux'), CallEntry('def bar2(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry(), LineEntry('qux()'), LineEntry(source_regex='with snoop:', min_python_version=(3, 10)), ElapsedTimeEntry(), VariableEntry('_x', '9'), VariableEntry('qux'), CallEntry('def bar3(_x):'), LineEntry('qux()'), ReturnEntry('qux()'), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_with_block_depth(normalize):
    if False:
        print('Hello World!')
    string_io = io.StringIO()

    def f4(x4):
        if False:
            print('Hello World!')
        result4 = x4 * 2
        return result4

    def f3(x3):
        if False:
            print('Hello World!')
        result3 = f4(x3)
        return result3

    def f2(x2):
        if False:
            while True:
                i = 10
        result2 = f3(x2)
        return result2

    def f1(x1):
        if False:
            i = 10
            return i + 15
        str(3)
        with pysnooper.snoop(string_io, depth=3, normalize=normalize, color=False):
            result1 = f2(x1)
        return result1
    result = f1(10)
    assert result == 20
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry(), VariableEntry(), VariableEntry(), VariableEntry(), LineEntry('result1 = f2(x1)'), VariableEntry(), VariableEntry(), CallEntry('def f2(x2):'), LineEntry(), VariableEntry(), VariableEntry(), CallEntry('def f3(x3):'), LineEntry(), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('20'), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('20'), VariableEntry(min_python_version=(3, 10)), LineEntry(source_regex='with pysnooper.snoop.*', min_python_version=(3, 10)), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_cellvars(normalize):
    if False:
        while True:
            i = 10
    string_io = io.StringIO()

    def f2(a):
        if False:
            print('Hello World!')

        def f3(a):
            if False:
                return 10
            x = 0
            x += 1

            def f4(a):
                if False:
                    print('Hello World!')
                y = x
                return 42
            return f4(a)
        return f3(a)

    def f1(a):
        if False:
            print('Hello World!')
        with pysnooper.snoop(string_io, depth=4, normalize=normalize, color=False):
            result1 = f2(a)
        return result1
    result = f1(42)
    assert result == 42
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry(), VariableEntry(), VariableEntry(), VariableEntry(), LineEntry('result1 = f2(a)'), VariableEntry(), CallEntry('def f2(a):'), LineEntry(), VariableEntry(), LineEntry(), VariableEntry('a'), CallEntry('def f3(a):'), LineEntry(), VariableEntry('x'), LineEntry(), VariableEntry('x'), LineEntry(), VariableEntry(), LineEntry(), VariableEntry(), VariableEntry('x'), CallEntry('def f4(a):'), LineEntry(), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry(), ReturnEntry(), ReturnValueEntry(), ReturnEntry(), ReturnValueEntry(), VariableEntry(min_python_version=(3, 10)), LineEntry(source_regex='with pysnooper.snoop.*', min_python_version=(3, 10)), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_var_order(normalize):
    if False:
        while True:
            i = 10
    string_io = io.StringIO()

    def f(one, two, three, four):
        if False:
            return 10
        five = None
        six = None
        seven = None
        (five, six, seven) = (5, 6, 7)
    with pysnooper.snoop(string_io, depth=2, normalize=normalize, color=False):
        result = f(1, 2, 3, 4)
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry(), VariableEntry(), VariableEntry(), LineEntry('result = f(1, 2, 3, 4)'), VariableEntry('one', '1'), VariableEntry('two', '2'), VariableEntry('three', '3'), VariableEntry('four', '4'), CallEntry('def f(one, two, three, four):'), LineEntry(), VariableEntry('five'), LineEntry(), VariableEntry('six'), LineEntry(), VariableEntry('seven'), LineEntry(), VariableEntry('five', '5'), VariableEntry('six', '6'), VariableEntry('seven', '7'), ReturnEntry(), ReturnValueEntry(), VariableEntry('result', 'None', min_python_version=(3, 10)), LineEntry(source_regex='with pysnooper.snoop.*', min_python_version=(3, 10)), ElapsedTimeEntry()), normalize=normalize)

def test_truncate():
    if False:
        for i in range(10):
            print('nop')
    max_length = 20
    for i in range(max_length * 2):
        string = i * 'a'
        truncated = truncate(string, max_length)
        if len(string) <= max_length:
            assert string == truncated
        else:
            assert truncated == 'aaaaaaaa...aaaaaaaaa'
            assert len(truncated) == max_length

def test_indentation():
    if False:
        for i in range(10):
            print('nop')
    from .samples import indentation, recursion
    assert_sample_output(indentation)
    assert_sample_output(recursion)

def test_exception():
    if False:
        print('Hello World!')
    from .samples import exception
    assert_sample_output(exception)

def test_generator():
    if False:
        i = 10
        return i + 15
    string_io = io.StringIO()
    original_tracer = sys.gettrace()
    original_tracer_active = lambda : sys.gettrace() is original_tracer

    @pysnooper.snoop(string_io, color=False)
    def f(x1):
        if False:
            return 10
        assert not original_tracer_active()
        x2 = (yield x1)
        assert not original_tracer_active()
        x3 = 'foo'
        assert not original_tracer_active()
        x4 = (yield 2)
        assert not original_tracer_active()
        return
    assert original_tracer_active()
    generator = f(0)
    assert original_tracer_active()
    first_item = next(generator)
    assert original_tracer_active()
    assert first_item == 0
    second_item = generator.send('blabla')
    assert original_tracer_active()
    assert second_item == 2
    with pytest.raises(StopIteration) as exc_info:
        generator.send('looloo')
    assert original_tracer_active()
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x1', '0'), VariableEntry(), CallEntry(), LineEntry(), VariableEntry(), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('0'), ElapsedTimeEntry(), VariableEntry('x1', '0'), VariableEntry(), VariableEntry(), VariableEntry(), CallEntry(), VariableEntry('x2', "'blabla'"), LineEntry(), LineEntry(), VariableEntry('x3', "'foo'"), LineEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry('2'), ElapsedTimeEntry(), VariableEntry('x1', '0'), VariableEntry(), VariableEntry(), VariableEntry(), VariableEntry(), VariableEntry(), CallEntry(), VariableEntry('x4', "'looloo'"), LineEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry(None), ElapsedTimeEntry()))

@pytest.mark.parametrize('normalize', (True, False))
def test_custom_repr(normalize):
    if False:
        return 10
    string_io = io.StringIO()

    def large(l):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(l, list) and len(l) > 5

    def print_list_size(l):
        if False:
            for i in range(10):
                print('nop')
        return 'list(size={})'.format(len(l))

    def print_dict(d):
        if False:
            while True:
                i = 10
        return 'dict(keys={})'.format(sorted(list(d.keys())))

    def evil_condition(x):
        if False:
            i = 10
            return i + 15
        return large(x) or isinstance(x, dict)

    @pysnooper.snoop(string_io, custom_repr=((large, print_list_size), (dict, print_dict), (evil_condition, lambda x: 'I am evil')), normalize=normalize, color=False)
    def sum_to_x(x):
        if False:
            print('Hello World!')
        l = list(range(x))
        a = {'1': 1, '2': 2}
        return sum(l)
    result = sum_to_x(10000)
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x', '10000'), CallEntry(), LineEntry(), VariableEntry('l', 'list(size=10000)'), LineEntry(), VariableEntry('a', "dict(keys=['1', '2'])"), LineEntry(), ReturnEntry(), ReturnValueEntry('49995000'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_custom_repr_single(normalize):
    if False:
        i = 10
        return i + 15
    string_io = io.StringIO()

    @pysnooper.snoop(string_io, custom_repr=(list, lambda l: 'foofoo!'), normalize=normalize, color=False)
    def sum_to_x(x):
        if False:
            print('Hello World!')
        l = list(range(x))
        return 7
    result = sum_to_x(10000)
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('x', '10000'), CallEntry(), LineEntry(), VariableEntry('l', 'foofoo!'), LineEntry(), ReturnEntry(), ReturnValueEntry('7'), ElapsedTimeEntry()), normalize=normalize)

def test_disable():
    if False:
        i = 10
        return i + 15
    string_io = io.StringIO()

    def my_function(foo):
        if False:
            print('Hello World!')
        x = 7
        y = 8
        return x + y
    with mini_toolbox.TempValueSetter((pysnooper.tracer, 'DISABLED'), True):
        tracer = pysnooper.snoop(string_io, color=False)
        with tracer:
            result = my_function('baba')
        my_decorated_function = tracer(my_function)
        my_decorated_function('booboo')
    output = string_io.getvalue()
    assert not output

@pytest.mark.parametrize('normalize', (True, False))
def test_class(normalize):
    if False:
        for i in range(10):
            print('nop')
    string_io = io.StringIO()

    @pysnooper.snoop(string_io, normalize=normalize, color=False)
    class MyClass(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = 7

        def my_method(self, foo):
            if False:
                while True:
                    i = 10
            y = 8
            return y + self.x
    instance = MyClass()
    result = instance.my_method('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('self', value_regex='u?.+MyClass object'), CallEntry('def __init__(self):'), LineEntry('self.x = 7'), ReturnEntry('self.x = 7'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('self', value_regex='u?.+MyClass object'), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_method(self, foo):'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + self.x'), ReturnEntry('return y + self.x'), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_class_with_decorated_method(normalize):
    if False:
        print('Hello World!')
    string_io = io.StringIO()

    def decorator(function):
        if False:
            return 10

        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            result = function(*args, **kwargs)
            return result
        return wrapper

    @pysnooper.snoop(string_io, normalize=normalize, color=False)
    class MyClass(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = 7

        @decorator
        def my_method(self, foo):
            if False:
                while True:
                    i = 10
            y = 8
            return y + self.x
    instance = MyClass()
    result = instance.my_method('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('self', value_regex='u?.+MyClass object'), CallEntry('def __init__(self):'), LineEntry('self.x = 7'), ReturnEntry('self.x = 7'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('args', value_regex="\\(<.+>, 'baba'\\)"), VariableEntry('kwargs', value_regex='\\{\\}'), VariableEntry('function', value_regex='u?.+my_method'), CallEntry('def wrapper(*args, **kwargs):'), LineEntry('result = function(*args, **kwargs)'), VariableEntry('result', '15'), LineEntry('return result'), ReturnEntry('return result'), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_class_with_decorated_method_and_snoop_applied_to_method(normalize):
    if False:
        i = 10
        return i + 15
    string_io = io.StringIO()

    def decorator(function):
        if False:
            while True:
                i = 10

        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            result = function(*args, **kwargs)
            return result
        return wrapper

    @pysnooper.snoop(string_io, normalize=normalize, color=False)
    class MyClass(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = 7

        @decorator
        @pysnooper.snoop(string_io, normalize=normalize, color=False)
        def my_method(self, foo):
            if False:
                i = 10
                return i + 15
            y = 8
            return y + self.x
    instance = MyClass()
    result = instance.my_method('baba')
    assert result == 15
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def __init__(self):'), LineEntry('self.x = 7'), ReturnEntry('self.x = 7'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('args', value_regex="u?\\(<.+>, 'baba'\\)"), VariableEntry('kwargs', value_regex='u?\\{\\}'), VariableEntry('function', value_regex='u?.*my_method'), CallEntry('def wrapper(*args, **kwargs):'), LineEntry('result = function(*args, **kwargs)'), SourcePathEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), VariableEntry('foo', value_regex="u?'baba'"), CallEntry('def my_method(self, foo):'), LineEntry('y = 8'), VariableEntry('y', '8'), LineEntry('return y + self.x'), ReturnEntry('return y + self.x'), ReturnValueEntry('15'), ElapsedTimeEntry(), VariableEntry('result', '15'), LineEntry('return result'), ReturnEntry('return result'), ReturnValueEntry('15'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_class_with_property(normalize):
    if False:
        for i in range(10):
            print('nop')
    string_io = io.StringIO()

    @pysnooper.snoop(string_io, normalize=normalize, color=False)
    class MyClass(object):

        def __init__(self):
            if False:
                return 10
            self._x = 0

        def plain_method(self):
            if False:
                while True:
                    i = 10
            pass

        @property
        def x(self):
            if False:
                for i in range(10):
                    print('nop')
            self.plain_method()
            return self._x

        @x.setter
        def x(self, value):
            if False:
                print('Hello World!')
            self.plain_method()
            self._x = value

        @x.deleter
        def x(self):
            if False:
                i = 10
                return i + 15
            self.plain_method()
            del self._x
    instance = MyClass()
    result = instance.x
    assert result == instance._x
    instance.x = 1
    assert instance._x == 1
    del instance.x
    with pytest.raises(AttributeError):
        instance._x
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def __init__(self):'), LineEntry('self._x = 0'), ReturnEntry('self._x = 0'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def plain_method(self):'), LineEntry('pass'), ReturnEntry('pass'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def plain_method(self):'), LineEntry('pass'), ReturnEntry('pass'), ReturnValueEntry('None'), ElapsedTimeEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def plain_method(self):'), LineEntry('pass'), ReturnEntry('pass'), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

@pytest.mark.parametrize('normalize', (True, False))
def test_snooping_on_class_does_not_cause_base_class_to_be_snooped(normalize):
    if False:
        return 10
    string_io = io.StringIO()

    class UnsnoopedBaseClass(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.method_on_base_class_was_called = False

        def method_on_base_class(self):
            if False:
                return 10
            self.method_on_base_class_was_called = True

    @pysnooper.snoop(string_io, normalize=normalize, color=False)
    class MyClass(UnsnoopedBaseClass):

        def method_on_child_class(self):
            if False:
                while True:
                    i = 10
            self.method_on_base_class()
    instance = MyClass()
    assert not instance.method_on_base_class_was_called
    instance.method_on_child_class()
    assert instance.method_on_base_class_was_called
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), VariableEntry('self', value_regex='u?.*MyClass object'), CallEntry('def method_on_child_class(self):'), LineEntry('self.method_on_base_class()'), ReturnEntry('self.method_on_base_class()'), ReturnValueEntry('None'), ElapsedTimeEntry()), normalize=normalize)

def test_normalize():
    if False:
        i = 10
        return i + 15
    string_io = io.StringIO()

    class A:

        def __init__(self, a):
            if False:
                return 10
            self.a = a

    @pysnooper.snoop(string_io, normalize=True, color=False)
    def add():
        if False:
            i = 10
            return i + 15
        a = A(19)
        b = A(22)
        res = a.a + b.a
        return res
    add()
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry('test_pysnooper.py'), VariableEntry('A', value_regex='<class .*\\.A.?>'), CallEntry('def add():'), LineEntry('a = A(19)'), VariableEntry('a', value_regex='<.*\\.A (?:object|instance)>'), LineEntry('b = A(22)'), VariableEntry('b', value_regex='<.*\\.A (?:object|instance)>'), LineEntry('res = a.a + b.a'), VariableEntry('res', value='41'), LineEntry('return res'), ReturnEntry('return res'), ReturnValueEntry('41'), ElapsedTimeEntry()))

def test_normalize_prefix():
    if False:
        return 10
    string_io = io.StringIO()
    _prefix = 'ZZZZ'

    class A:

        def __init__(self, a):
            if False:
                while True:
                    i = 10
            self.a = a

    @pysnooper.snoop(string_io, normalize=True, prefix=_prefix, color=False)
    def add():
        if False:
            i = 10
            return i + 15
        a = A(19)
        b = A(22)
        res = a.a + b.a
        return res
    add()
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry('test_pysnooper.py', prefix=_prefix), VariableEntry('A', value_regex='<class .*\\.A.?>', prefix=_prefix), CallEntry('def add():', prefix=_prefix), LineEntry('a = A(19)', prefix=_prefix), VariableEntry('a', value_regex='<.*\\.A (?:object|instance)>', prefix=_prefix), LineEntry('b = A(22)', prefix=_prefix), VariableEntry('b', value_regex='<.*\\.A (?:object|instance)>', prefix=_prefix), LineEntry('res = a.a + b.a', prefix=_prefix), VariableEntry('res', value='41', prefix=_prefix), LineEntry('return res', prefix=_prefix), ReturnEntry('return res', prefix=_prefix), ReturnValueEntry('41', prefix=_prefix), ElapsedTimeEntry(prefix=_prefix)))

def test_normalize_thread_info():
    if False:
        return 10
    string_io = io.StringIO()

    class A:

        def __init__(self, a):
            if False:
                i = 10
                return i + 15
            self.a = a

    @pysnooper.snoop(string_io, normalize=True, thread_info=True, color=False)
    def add():
        if False:
            for i in range(10):
                print('nop')
        a = A(19)
        b = A(22)
        res = a.a + b.a
        return res
    with pytest.raises(NotImplementedError):
        add()

def test_exception():
    if False:
        print('Hello World!')
    string_io = io.StringIO()

    @pysnooper.snoop(string_io, color=False)
    def f():
        if False:
            i = 10
            return i + 15
        x = 8
        raise MemoryError
    with pytest.raises(MemoryError):
        f()
    output = string_io.getvalue()
    assert_output(output, (SourcePathEntry(), CallEntry(), LineEntry(), VariableEntry(), LineEntry(), ExceptionEntry(), ExceptionValueEntry('MemoryError'), CallEndedByExceptionEntry(), ElapsedTimeEntry()))

def test_exception_on_entry():
    if False:
        print('Hello World!')

    @pysnooper.snoop(color=False)
    def f(x):
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(TypeError):
        f()

def test_valid_zipfile():
    if False:
        i = 10
        return i + 15
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder, mini_toolbox.TempSysPathAdder(str(folder)):
        module_name = 'my_valid_zip_module'
        zip_name = 'valid.zip'
        zip_base_path = mini_toolbox.pathlib.Path('ansible/modules')
        python_file_path = folder / zip_name / zip_base_path / ('%s.py' % module_name)
        os.makedirs(str(folder / zip_name / zip_base_path))
        try:
            sys.path.insert(0, str(folder / zip_name / zip_base_path))
            content = textwrap.dedent(u'\n                import pysnooper\n                @pysnooper.snoop(color=False)\n                def f(x):\n                    return x\n            ')
            python_file_path.write_text(content)
            module = __import__(module_name)
            with zipfile.ZipFile(str(folder / 'foo_bar.zip'), 'w') as myZipFile:
                myZipFile.write(str(folder / zip_name / zip_base_path / ('%s.py' % module_name)), '%s/%s.py' % (zip_base_path, module_name), zipfile.ZIP_DEFLATED)
            python_file_path.unlink()
            folder.joinpath(zip_name).rename(folder.joinpath('%s.delete' % zip_name))
            folder.joinpath('foo_bar.zip').rename(folder.joinpath(zip_name))
            with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
                result = getattr(module, 'f')(7)
            assert result == 7
            output = output_capturer.output
            assert_output(output, (SourcePathEntry(), VariableEntry(stage='starting'), CallEntry('def f(x):'), LineEntry('return x'), ReturnEntry('return x'), ReturnValueEntry('7'), ElapsedTimeEntry()))
        finally:
            sys.path.remove(str(folder / zip_name / zip_base_path))

def test_invalid_zipfile():
    if False:
        while True:
            i = 10
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder, mini_toolbox.TempSysPathAdder(str(folder)):
        module_name = 'my_invalid_zip_module'
        zip_name = 'invalid.zip'
        zip_base_path = mini_toolbox.pathlib.Path('invalid/modules/path')
        python_file_path = folder / zip_name / zip_base_path / ('%s.py' % module_name)
        os.makedirs(str(folder / zip_name / zip_base_path))
        try:
            sys.path.insert(0, str(folder / zip_name / zip_base_path))
            content = textwrap.dedent(u'\n                import pysnooper\n                @pysnooper.snoop(color=False)\n                def f(x):\n                    return x\n            ')
            python_file_path.write_text(content)
            module = __import__(module_name)
            with zipfile.ZipFile(str(folder / 'foo_bar.zip'), 'w') as myZipFile:
                myZipFile.write(str(folder / zip_name / zip_base_path / ('%s.py' % module_name)), str(zip_base_path / ('%s.py' % (module_name,))), zipfile.ZIP_DEFLATED)
            python_file_path.unlink()
            folder.joinpath(zip_name).rename(folder.joinpath('%s.delete' % zip_name))
            folder.joinpath('foo_bar.zip').rename(folder.joinpath(zip_name))
            with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
                result = getattr(module, 'f')(7)
            assert result == 7
            output = output_capturer.output
            assert_output(output, (SourcePathEntry(), VariableEntry(stage='starting'), CallEntry('SOURCE IS UNAVAILABLE'), LineEntry('SOURCE IS UNAVAILABLE'), ReturnEntry('SOURCE IS UNAVAILABLE'), ReturnValueEntry('7'), ElapsedTimeEntry()))
        finally:
            sys.path.remove(str(folder / zip_name / zip_base_path))

def test_valid_damaged_zipfile():
    if False:
        while True:
            i = 10
    with mini_toolbox.create_temp_folder(prefix='pysnooper') as folder, mini_toolbox.TempSysPathAdder(str(folder)):
        module_name = 'my_damaged_module'
        zip_name = 'damaged.zip'
        zip_base_path = mini_toolbox.pathlib.Path('ansible/modules')
        python_file_path = folder / zip_name / zip_base_path / ('%s.py' % module_name)
        os.makedirs(str(folder / zip_name / zip_base_path))
        try:
            sys.path.insert(0, str(folder / zip_name / zip_base_path))
            content = textwrap.dedent(u'\n                import pysnooper\n                @pysnooper.snoop(color=False)\n                def f(x):\n                    return x\n            ')
            python_file_path.write_text(content)
            module = __import__(module_name)
            python_file_path.unlink()
            folder.joinpath(zip_name).rename(folder.joinpath('%s.delete' % zip_name))
            folder.joinpath(zip_name).write_text(u'I am not a zip file')
            with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
                result = getattr(module, 'f')(7)
            assert result == 7
            output = output_capturer.output
            assert_output(output, (SourcePathEntry(), VariableEntry(stage='starting'), CallEntry('SOURCE IS UNAVAILABLE'), LineEntry('SOURCE IS UNAVAILABLE'), ReturnEntry('SOURCE IS UNAVAILABLE'), ReturnValueEntry('7'), ElapsedTimeEntry()))
        finally:
            sys.path.remove(str(folder / zip_name / zip_base_path))