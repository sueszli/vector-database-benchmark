import asyncio
import logging
import os
import pathlib
import sys
import pytest
from loguru import logger
message = 'test message'
expected = message + '\n'
repetitions = pytest.mark.parametrize('rep', [0, 1, 2])

def log(sink, rep=1):
    if False:
        while True:
            i = 10
    logger.debug("This shouldn't be printed.")
    i = logger.add(sink, format='{message}')
    for _ in range(rep):
        logger.debug(message)
    logger.remove(i)
    logger.debug("This shouldn't be printed neither.")

async def async_log(sink, rep=1):
    logger.debug("This shouldn't be printed.")
    i = logger.add(sink, format='{message}')
    for _ in range(rep):
        logger.debug(message)
    await logger.complete()
    logger.remove(i)
    logger.debug("This shouldn't be printed neither.")

@repetitions
def test_stdout_sink(rep, capsys):
    if False:
        while True:
            i = 10
    log(sys.stdout, rep)
    (out, err) = capsys.readouterr()
    assert out == expected * rep
    assert err == ''

@repetitions
def test_stderr_sink(rep, capsys):
    if False:
        print('Hello World!')
    log(sys.stderr, rep)
    (out, err) = capsys.readouterr()
    assert out == ''
    assert err == expected * rep

@repetitions
def test_devnull(rep):
    if False:
        i = 10
        return i + 15
    log(os.devnull, rep)

@repetitions
@pytest.mark.parametrize('sink_from_path', [str, pathlib.Path, lambda path: open(path, 'a'), lambda path: pathlib.Path(path).open('a')])
def test_file_sink(rep, sink_from_path, tmp_path):
    if False:
        return 10
    file = tmp_path / 'test.log'
    sink = sink_from_path(str(file))
    log(sink, rep)
    assert file.read_text() == expected * rep

@repetitions
def test_file_sink_folder_creation(rep, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    file = tmp_path.joinpath('some', 'sub', 'folder', 'not', 'existing', 'test.log')
    log(file, rep)
    assert file.read_text() == expected * rep

@repetitions
def test_function_sink(rep):
    if False:
        return 10
    a = []

    def func(log_message):
        if False:
            i = 10
            return i + 15
        a.append(log_message)
    log(func, rep)
    assert a == [expected] * rep

@repetitions
def test_coroutine_sink(capsys, rep):
    if False:
        while True:
            i = 10

    async def async_print(msg):
        await asyncio.sleep(0.01)
        print(msg, end='')
        await asyncio.sleep(0.01)
    asyncio.run(async_log(async_print, rep))
    (out, err) = capsys.readouterr()
    assert err == ''
    assert out == expected * rep

@repetitions
def test_file_object_sink(rep):
    if False:
        return 10

    class A:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.out = ''

        def write(self, m):
            if False:
                for i in range(10):
                    print('nop')
            self.out += m
    a = A()
    log(a, rep)
    assert a.out == expected * rep

@repetitions
def test_standard_handler_sink(rep):
    if False:
        while True:
            i = 10
    out = []

    class H(logging.Handler):

        def emit(self, record):
            if False:
                print('Hello World!')
            out.append(record.getMessage() + '\n')
    h = H()
    log(h, rep)
    assert out == [expected] * rep

@repetitions
def test_flush(rep):
    if False:
        while True:
            i = 10
    flushed = []
    out = []

    class A:

        def write(self, m):
            if False:
                i = 10
                return i + 15
            out.append(m)

        def flush(self):
            if False:
                while True:
                    i = 10
            flushed.append(out[-1])
    log(A(), rep)
    assert flushed == [expected] * rep

def test_file_sink_ascii_encoding(tmp_path):
    if False:
        while True:
            i = 10
    file = tmp_path / 'test.log'
    logger.add(file, encoding='ascii', format='{message}', errors='backslashreplace', catch=False)
    logger.info('天')
    logger.remove()
    assert file.read_text('ascii') == '\\u5929\n'

def test_file_sink_utf8_encoding(tmp_path):
    if False:
        print('Hello World!')
    file = tmp_path / 'test.log'
    logger.add(file, encoding='utf8', format='{message}', errors='strict', catch=False)
    logger.info('天')
    logger.remove()
    assert file.read_text('utf8') == '天\n'

def test_file_sink_default_encoding(tmp_path):
    if False:
        i = 10
        return i + 15
    file = tmp_path / 'test.log'
    logger.add(file, format='{message}', errors='strict', catch=False)
    logger.info('天')
    logger.remove()
    assert file.read_text('utf8') == '天\n'

def test_disabled_logger_in_sink(sink_with_logger):
    if False:
        print('Hello World!')
    sink = sink_with_logger(logger)
    logger.disable('tests.conftest')
    logger.add(sink, format='{message}')
    logger.info('Disabled test')
    assert sink.out == 'Disabled test\n'

@pytest.mark.parametrize('flush', [123, None])
def test_custom_sink_invalid_flush(capsys, flush):
    if False:
        while True:
            i = 10

    class Sink:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.flush = flush

        def write(self, message):
            if False:
                while True:
                    i = 10
            print(message, end='')
    logger.add(Sink(), format='{message}')
    logger.info('Test')
    (out, err) = capsys.readouterr()
    assert out == 'Test\n'
    assert err == ''

@pytest.mark.parametrize('stop', [123, None])
def test_custom_sink_invalid_stop(capsys, stop):
    if False:
        print('Hello World!')

    class Sink:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.stop = stop

        def write(self, message):
            if False:
                i = 10
                return i + 15
            print(message, end='')
    logger.add(Sink(), format='{message}')
    logger.info('Test')
    logger.remove()
    (out, err) = capsys.readouterr()
    assert out == 'Test\n'
    assert err == ''

@pytest.mark.parametrize('complete', [123, None, lambda : None])
def test_custom_sink_invalid_complete(capsys, complete):
    if False:
        return 10

    class Sink:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.complete = complete

        def write(self, message):
            if False:
                return 10
            print(message, end='')

    async def worker():
        logger.info('Test')
        await logger.complete()
    logger.add(Sink(), format='{message}')
    asyncio.run(worker())
    (out, err) = capsys.readouterr()
    assert out == 'Test\n'
    assert err == ''

@pytest.mark.parametrize('sink', [123, sys, object(), int])
def test_invalid_sink(sink):
    if False:
        return 10
    with pytest.raises(TypeError):
        log(sink, '')

def test_deprecated_start_and_stop(writer):
    if False:
        i = 10
        return i + 15
    with pytest.warns(DeprecationWarning):
        i = logger.start(writer, format='{message}')
    logger.debug('Test')
    assert writer.read() == 'Test\n'
    writer.clear()
    with pytest.warns(DeprecationWarning):
        logger.stop(i)
    logger.debug('Test')
    assert writer.read() == ''