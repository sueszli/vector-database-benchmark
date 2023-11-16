import pickle
import re
import sys
import time
import pytest
from loguru import logger
from .conftest import default_threading_excepthook

class NotPicklable:

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        raise pickle.PicklingError('You shall not serialize me!')

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        pass

class NotPicklableTypeError:

    def __getstate__(self):
        if False:
            return 10
        raise TypeError('You shall not serialize me!')

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        pass

class NotUnpicklable:

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return '...'

    def __setstate__(self, state):
        if False:
            return 10
        raise pickle.UnpicklingError('You shall not de-serialize me!')

class NotUnpicklableTypeError:

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return '...'

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        raise TypeError('You shall not de-serialize me!')

class NotWritable:

    def write(self, message):
        if False:
            return 10
        if 'fail' in message.record['extra']:
            raise RuntimeError('You asked me to fail...')
        print(message, end='')

def test_enqueue():
    if False:
        print('Hello World!')
    x = []

    def sink(message):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        x.append(message)
    logger.add(sink, format='{message}', enqueue=True)
    logger.debug('Test')
    assert len(x) == 0
    logger.complete()
    assert len(x) == 1
    assert x[0] == 'Test\n'

def test_enqueue_with_exception():
    if False:
        print('Hello World!')
    x = []

    def sink(message):
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        x.append(message)
    logger.add(sink, format='{message}', enqueue=True)
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception('Error')
    assert len(x) == 0
    logger.complete()
    assert len(x) == 1
    lines = x[0].splitlines()
    assert lines[0] == 'Error'
    assert lines[-1] == 'ZeroDivisionError: division by zero'

def test_caught_exception_queue_put(writer, capsys):
    if False:
        i = 10
        return i + 15
    logger.add(writer, enqueue=True, catch=True, format='{message}')
    logger.info("It's fine")
    logger.bind(broken=NotPicklable()).info('Bye bye...')
    logger.info("It's fine again")
    logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert writer.read() == "It's fine\nIt's fine again\n"
    assert out == ''
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert re.match('Record was: \\{.*Bye bye.*\\}', lines[1])
    assert lines[-2].endswith('PicklingError: You shall not serialize me!')
    assert lines[-1] == '--- End of logging error ---'

def test_caught_exception_queue_get(writer, capsys):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, enqueue=True, catch=True, format='{message}')
    logger.info("It's fine")
    logger.bind(broken=NotUnpicklable()).info('Bye bye...')
    logger.info("It's fine again")
    logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert writer.read() == "It's fine\nIt's fine again\n"
    assert out == ''
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert lines[1] == 'Record was: None'
    assert lines[-2].endswith('UnpicklingError: You shall not de-serialize me!')
    assert lines[-1] == '--- End of logging error ---'

def test_caught_exception_sink_write(capsys):
    if False:
        while True:
            i = 10
    logger.add(NotWritable(), enqueue=True, catch=True, format='{message}')
    logger.info("It's fine")
    logger.bind(fail=True).info('Bye bye...')
    logger.info("It's fine again")
    logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert out == "It's fine\nIt's fine again\n"
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert re.match('Record was: \\{.*Bye bye.*\\}', lines[1])
    assert lines[-2] == 'RuntimeError: You asked me to fail...'
    assert lines[-1] == '--- End of logging error ---'

def test_not_caught_exception_queue_put(writer, capsys):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, enqueue=True, catch=False, format='{message}')
    logger.info("It's fine")
    with pytest.raises(pickle.PicklingError, match='You shall not serialize me!'):
        logger.bind(broken=NotPicklable()).info('Bye bye...')
    logger.remove()
    (out, err) = capsys.readouterr()
    assert writer.read() == "It's fine\n"
    assert out == ''
    assert err == ''

def test_not_caught_exception_queue_get(writer, capsys):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, enqueue=True, catch=False, format='{message}')
    with default_threading_excepthook():
        logger.info("It's fine")
        logger.bind(broken=NotUnpicklable()).info('Bye bye...')
        logger.info("It's fine again")
        logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert writer.read() == "It's fine\nIt's fine again\n"
    assert out == ''
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert lines[1] == 'Record was: None'
    assert lines[-2].endswith('UnpicklingError: You shall not de-serialize me!')
    assert lines[-1] == '--- End of logging error ---'

def test_not_caught_exception_sink_write(capsys):
    if False:
        return 10
    logger.add(NotWritable(), enqueue=True, catch=False, format='{message}')
    with default_threading_excepthook():
        logger.info("It's fine")
        logger.bind(fail=True).info('Bye bye...')
        logger.info("It's fine again")
        logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert out == "It's fine\nIt's fine again\n"
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert re.match('Record was: \\{.*Bye bye.*\\}', lines[1])
    assert lines[-2] == 'RuntimeError: You asked me to fail...'
    assert lines[-1] == '--- End of logging error ---'

def test_not_caught_exception_sink_write_then_complete(capsys):
    if False:
        print('Hello World!')
    logger.add(NotWritable(), enqueue=True, catch=False, format='{message}')
    with default_threading_excepthook():
        logger.bind(fail=True).info('Bye bye...')
        logger.complete()
        logger.complete()
        logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert out == ''
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert re.match('Record was: \\{.*Bye bye.*\\}', lines[1])
    assert lines[-2] == 'RuntimeError: You asked me to fail...'
    assert lines[-1] == '--- End of logging error ---'

def test_not_caught_exception_queue_get_then_complete(writer, capsys):
    if False:
        for i in range(10):
            print('nop')
    logger.add(writer, enqueue=True, catch=False, format='{message}')
    with default_threading_excepthook():
        logger.bind(broken=NotUnpicklable()).info('Bye bye...')
        logger.complete()
        logger.complete()
        logger.remove()
    (out, err) = capsys.readouterr()
    lines = err.strip().splitlines()
    assert writer.read() == ''
    assert out == ''
    assert lines[0] == '--- Logging error in Loguru Handler #0 ---'
    assert lines[1] == 'Record was: None'
    assert lines[-2].endswith('UnpicklingError: You shall not de-serialize me!')
    assert lines[-1] == '--- End of logging error ---'

def test_wait_for_all_messages_enqueued(capsys):
    if False:
        print('Hello World!')

    def slow_sink(message):
        if False:
            print('Hello World!')
        time.sleep(0.01)
        sys.stderr.write(message)
    logger.add(slow_sink, enqueue=True, catch=False, format='{message}')
    for i in range(10):
        logger.info(i)
    logger.complete()
    (out, err) = capsys.readouterr()
    assert out == ''
    assert err == ''.join(('%d\n' % i for i in range(10)))

@pytest.mark.parametrize('exception_value', [NotPicklable(), NotPicklableTypeError()])
def test_logging_not_picklable_exception(exception_value):
    if False:
        print('Hello World!')
    exception = None

    def sink(message):
        if False:
            while True:
                i = 10
        nonlocal exception
        exception = message.record['exception']
    logger.add(sink, enqueue=True, catch=False)
    try:
        raise ValueError(exception_value)
    except Exception:
        logger.exception('Oups')
    logger.remove()
    (type_, value, traceback_) = exception
    assert type_ is ValueError
    assert value is None
    assert traceback_ is None

@pytest.mark.parametrize('exception_value', [NotUnpicklable(), NotUnpicklableTypeError()])
def test_logging_not_unpicklable_exception(exception_value):
    if False:
        while True:
            i = 10
    exception = None

    def sink(message):
        if False:
            return 10
        nonlocal exception
        exception = message.record['exception']
    logger.add(sink, enqueue=True, catch=False)
    try:
        raise ValueError(exception_value)
    except Exception:
        logger.exception('Oups')
    logger.remove()
    (type_, value, traceback_) = exception
    assert type_ is ValueError
    assert value is None
    assert traceback_ is None