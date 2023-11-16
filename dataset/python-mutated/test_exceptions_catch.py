import asyncio
import site
import sys
import sysconfig
import types
import pytest
from loguru import logger

@pytest.mark.parametrize('diagnose', [False, True])
def test_caret_not_masked(writer, diagnose):
    if False:
        while True:
            i = 10
    logger.add(writer, backtrace=True, diagnose=diagnose, colorize=False, format='')

    @logger.catch
    def f(n):
        if False:
            return 10
        1 / n
        f(n - 1)
    f(30)
    assert sum((line.startswith('> ') for line in writer.read().splitlines())) == 1

@pytest.mark.parametrize('diagnose', [False, True])
def test_no_caret_if_no_backtrace(writer, diagnose):
    if False:
        while True:
            i = 10
    logger.add(writer, backtrace=False, diagnose=diagnose, colorize=False, format='')

    @logger.catch
    def f(n):
        if False:
            return 10
        1 / n
        f(n - 1)
    f(30)
    assert sum((line.startswith('> ') for line in writer.read().splitlines())) == 0

@pytest.mark.parametrize('encoding', ['ascii', 'UTF8', None, 'unknown-encoding', '', object()])
def test_sink_encoding(writer, encoding):
    if False:
        return 10

    class Writer:

        def __init__(self, encoding):
            if False:
                i = 10
                return i + 15
            self.encoding = encoding
            self.output = ''

        def write(self, message):
            if False:
                print('Hello World!')
            self.output += message
    writer = Writer(encoding)
    logger.add(writer, backtrace=True, diagnose=True, colorize=False, format='', catch=False)

    def foo(a, b):
        if False:
            print('Hello World!')
        a / b

    def bar(c):
        if False:
            print('Hello World!')
        foo(c, 0)
    try:
        bar(4)
    except ZeroDivisionError:
        logger.exception('')
    assert writer.output.endswith('ZeroDivisionError: division by zero\n')

def test_file_sink_ascii_encoding(tmp_path):
    if False:
        i = 10
        return i + 15
    file = tmp_path / 'test.log'
    logger.add(file, format='', encoding='ascii', errors='backslashreplace', catch=False)
    a = '天'
    try:
        '天' * a
    except Exception:
        logger.exception('')
    logger.remove()
    result = file.read_text('ascii')
    assert result.count('"\\u5929" * a') == 1
    assert result.count("-> '\\u5929'") == 1

def test_file_sink_utf8_encoding(tmp_path):
    if False:
        return 10
    file = tmp_path / 'test.log'
    logger.add(file, format='', encoding='utf8', errors='strict', catch=False)
    a = '天'
    try:
        '天' * a
    except Exception:
        logger.exception('')
    logger.remove()
    result = file.read_text('utf8')
    assert result.count('"天" * a') == 1
    assert result.count("└ '天'") == 1

def test_has_sys_real_prefix(writer, monkeypatch):
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as context:
        context.setattr(sys, 'real_prefix', '/foo/bar/baz', raising=False)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_no_sys_real_prefix(writer, monkeypatch):
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as context:
        context.delattr(sys, 'real_prefix', raising=False)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_has_site_getsitepackages(writer, monkeypatch):
    if False:
        while True:
            i = 10
    with monkeypatch.context() as context:
        context.setattr(site, 'getsitepackages', lambda : ['foo', 'bar', 'baz'], raising=False)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_no_site_getsitepackages(writer, monkeypatch):
    if False:
        return 10
    with monkeypatch.context() as context:
        context.delattr(site, 'getsitepackages', raising=False)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_user_site_is_path(writer, monkeypatch):
    if False:
        print('Hello World!')
    with monkeypatch.context() as context:
        context.setattr(site, 'USER_SITE', '/foo/bar/baz')
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_user_site_is_none(writer, monkeypatch):
    if False:
        print('Hello World!')
    with monkeypatch.context() as context:
        context.setattr(site, 'USER_SITE', None)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_sysconfig_get_path_return_path(writer, monkeypatch):
    if False:
        return 10
    with monkeypatch.context() as context:
        context.setattr(sysconfig, 'get_path', lambda *a, **k: '/foo/bar/baz')
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_sysconfig_get_path_return_none(writer, monkeypatch):
    if False:
        return 10
    with monkeypatch.context() as context:
        context.setattr(sysconfig, 'get_path', lambda *a, **k: None)
        logger.add(writer, backtrace=False, diagnose=True, colorize=False, format='')
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception('')
        assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_no_exception(writer):
    if False:
        i = 10
        return i + 15
    logger.add(writer, backtrace=False, diagnose=False, colorize=False, format='{message}')
    logger.exception('No Error.')
    assert writer.read() in ('No Error.\nNoneType\n', 'No Error.\nNoneType: None\n')

def test_exception_is_none():
    if False:
        i = 10
        return i + 15
    err = object()

    def writer(msg):
        if False:
            print('Hello World!')
        nonlocal err
        err = msg.record['exception']
    logger.add(writer)
    logger.error('No exception')
    assert err is None

def test_exception_is_tuple():
    if False:
        for i in range(10):
            print('nop')
    exception = None

    def writer(msg):
        if False:
            while True:
                i = 10
        nonlocal exception
        exception = msg.record['exception']
    logger.add(writer, catch=False)
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception('Exception')
        reference = sys.exc_info()
    (t_1, v_1, tb_1) = exception
    (t_2, v_2, tb_2) = (x for x in exception)
    (t_3, v_3, tb_3) = (exception[0], exception[1], exception[2])
    (t_4, v_4, tb_4) = (exception.type, exception.value, exception.traceback)
    assert isinstance(exception, tuple)
    assert len(exception) == 3
    assert exception == reference
    assert reference == exception
    assert not exception != reference
    assert not reference != exception
    assert all((t == ZeroDivisionError for t in (t_1, t_2, t_3, t_4)))
    assert all((isinstance(v, ZeroDivisionError) for v in (v_1, v_2, v_3, v_4)))
    assert all((isinstance(tb, types.TracebackType) for tb in (tb_1, tb_2, tb_3, tb_4)))

@pytest.mark.parametrize('exception', [ZeroDivisionError, ArithmeticError, (ValueError, ZeroDivisionError)])
def test_exception_not_raising(writer, exception):
    if False:
        while True:
            i = 10
    logger.add(writer)

    @logger.catch(exception)
    def a():
        if False:
            return 10
        1 / 0
    a()
    assert writer.read().endswith('ZeroDivisionError: division by zero\n')

@pytest.mark.parametrize('exception', [ValueError, (SyntaxError, TypeError)])
def test_exception_raising(writer, exception):
    if False:
        i = 10
        return i + 15
    logger.add(writer)

    @logger.catch(exception=exception)
    def a():
        if False:
            i = 10
            return i + 15
        1 / 0
    with pytest.raises(ZeroDivisionError):
        a()
    assert writer.read() == ''

@pytest.mark.parametrize('exclude', [ZeroDivisionError, ArithmeticError, (ValueError, ZeroDivisionError)])
@pytest.mark.parametrize('exception', [BaseException, ZeroDivisionError])
def test_exclude_exception_raising(writer, exclude, exception):
    if False:
        while True:
            i = 10
    logger.add(writer)

    @logger.catch(exception, exclude=exclude)
    def a():
        if False:
            i = 10
            return i + 15
        1 / 0
    with pytest.raises(ZeroDivisionError):
        a()
    assert writer.read() == ''

@pytest.mark.parametrize('exclude', [ValueError, (SyntaxError, TypeError)])
@pytest.mark.parametrize('exception', [BaseException, ZeroDivisionError])
def test_exclude_exception_not_raising(writer, exclude, exception):
    if False:
        i = 10
        return i + 15
    logger.add(writer)

    @logger.catch(exception, exclude=exclude)
    def a():
        if False:
            for i in range(10):
                print('nop')
        1 / 0
    a()
    assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_reraise(writer):
    if False:
        print('Hello World!')
    logger.add(writer)

    @logger.catch(reraise=True)
    def a():
        if False:
            print('Hello World!')
        1 / 0
    with pytest.raises(ZeroDivisionError):
        a()
    assert writer.read().endswith('ZeroDivisionError: division by zero\n')

def test_onerror(writer):
    if False:
        for i in range(10):
            print('nop')
    is_error_valid = False
    logger.add(writer, format='{message}')

    def onerror(error):
        if False:
            for i in range(10):
                print('nop')
        nonlocal is_error_valid
        logger.info('Called after logged message')
        (_, exception, _) = sys.exc_info()
        is_error_valid = error == exception and isinstance(error, ZeroDivisionError)

    @logger.catch(onerror=onerror)
    def a():
        if False:
            i = 10
            return i + 15
        1 / 0
    a()
    assert is_error_valid
    assert writer.read().endswith('ZeroDivisionError: division by zero\nCalled after logged message\n')

def test_onerror_with_reraise(writer):
    if False:
        while True:
            i = 10
    called = False
    logger.add(writer, format='{message}')

    def onerror(_):
        if False:
            return 10
        nonlocal called
        called = True
    with pytest.raises(ZeroDivisionError):
        with logger.catch(onerror=onerror, reraise=True):
            1 / 0
    assert called

def test_decorate_function(writer):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, format='{message}', diagnose=False, backtrace=False, colorize=False)

    @logger.catch
    def a(x):
        if False:
            while True:
                i = 10
        return 100 / x
    assert a(50) == 2
    assert writer.read() == ''

def test_decorate_coroutine(writer):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, format='{message}', diagnose=False, backtrace=False, colorize=False)

    @logger.catch
    async def foo(a, b):
        return a + b
    result = asyncio.run(foo(100, 5))
    assert result == 105
    assert writer.read() == ''

def test_decorate_generator(writer):
    if False:
        return 10

    @logger.catch
    def foo(x, y, z):
        if False:
            print('Hello World!')
        yield x
        yield y
        return z
    f = foo(1, 2, 3)
    assert next(f) == 1
    assert next(f) == 2
    with pytest.raises(StopIteration, match='3'):
        next(f)

def test_decorate_generator_with_error():
    if False:
        while True:
            i = 10

    @logger.catch
    def foo():
        if False:
            while True:
                i = 10
        for i in range(3):
            1 / (2 - i)
            yield i
    assert list(foo()) == [0, 1]

def test_default_with_function():
    if False:
        i = 10
        return i + 15

    @logger.catch(default=42)
    def foo():
        if False:
            return 10
        1 / 0
    assert foo() == 42

def test_default_with_generator():
    if False:
        print('Hello World!')

    @logger.catch(default=42)
    def foo():
        if False:
            while True:
                i = 10
        yield (1 / 0)
    with pytest.raises(StopIteration, match='42'):
        next(foo())

def test_default_with_coroutine():
    if False:
        for i in range(10):
            print('nop')

    @logger.catch(default=42)
    async def foo():
        return 1 / 0
    assert asyncio.run(foo()) == 42

def test_error_when_decorating_class_without_parentheses():
    if False:
        return 10
    with pytest.raises(TypeError):

        @logger.catch
        class Foo:
            pass

def test_error_when_decorating_class_with_parentheses():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):

        @logger.catch()
        class Foo:
            pass