import json
import re
import sys
from loguru import logger

class JsonSink:

    def __init__(self):
        if False:
            print('Hello World!')
        self.message = None
        self.dict = None
        self.json = None

    def write(self, message):
        if False:
            i = 10
            return i + 15
        self.message = message
        self.dict = message.record
        self.json = json.loads(message)

def test_serialize():
    if False:
        while True:
            i = 10
    sink = JsonSink()
    logger.add(sink, format='{level} {message}', serialize=True)
    logger.debug('Test')
    assert sink.json['text'] == 'DEBUG Test\n'
    assert sink.dict['message'] == sink.json['record']['message'] == 'Test'
    assert set(sink.dict.keys()) == set(sink.json['record'].keys())

def test_serialize_non_ascii_characters():
    if False:
        while True:
            i = 10
    sink = JsonSink()
    logger.add(sink, format='{level.icon} {message}', serialize=True)
    logger.debug('天')
    assert re.search('"message": "([^\\"]+)"', sink.message).group(1) == '天'
    assert re.search('"text": "([^\\"]+)"', sink.message).group(1) == '🐞 天\\n'
    assert re.search('"icon": "([^\\"]+)"', sink.message).group(1) == '🐞'
    assert sink.json['text'] == '🐞 天\n'
    assert sink.dict['message'] == sink.json['record']['message'] == '天'

def test_serialize_exception():
    if False:
        print('Hello World!')
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception('Error')
    lines = sink.json['text'].splitlines()
    assert lines[0] == 'Error'
    assert lines[-1] == 'ZeroDivisionError: division by zero'
    assert sink.json['record']['exception'] == {'type': 'ZeroDivisionError', 'value': 'division by zero', 'traceback': True}

def test_serialize_exception_without_context():
    if False:
        while True:
            i = 10
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)
    logger.exception('No Error')
    lines = sink.json['text'].splitlines()
    assert lines[0] == 'No Error'
    assert lines[-1] == 'NoneType' if sys.version_info < (3, 5, 3) else 'NoneType: None'
    assert sink.json['record']['exception'] == {'type': None, 'value': None, 'traceback': False}

def test_serialize_exception_none_tuple():
    if False:
        print('Hello World!')
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)
    logger.opt(exception=(None, None, None)).error('No Error')
    lines = sink.json['text'].splitlines()
    assert lines[0] == 'No Error'
    assert lines[-1] == 'NoneType' if sys.version_info < (3, 5, 3) else 'NoneType: None'
    assert sink.json['record']['exception'] == {'type': None, 'value': None, 'traceback': False}

def test_serialize_exception_instance():
    if False:
        return 10
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)
    logger.opt(exception=ZeroDivisionError('Oops')).error('Failure')
    lines = sink.json['text'].splitlines()
    assert lines[0] == 'Failure'
    assert lines[-1] == 'ZeroDivisionError: Oops'
    assert sink.json['record']['exception'] == {'type': 'ZeroDivisionError', 'value': 'Oops', 'traceback': False}

def test_serialize_with_catch_decorator():
    if False:
        while True:
            i = 10
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)

    @logger.catch
    def foo():
        if False:
            print('Hello World!')
        1 / 0
    foo()
    lines = sink.json['text'].splitlines()
    assert lines[0].startswith('An error has been caught')
    assert lines[-1] == 'ZeroDivisionError: division by zero'
    assert bool(sink.json['record']['exception'])

def test_serialize_with_record_option():
    if False:
        return 10
    sink = JsonSink()
    logger.add(sink, format='{message}', serialize=True, catch=False)
    logger.opt(record=True).info('Test', foo=123)
    assert sink.json['text'] == 'Test\n'
    assert sink.dict['extra'] == {'foo': 123}

def test_serialize_not_serializable():
    if False:
        return 10
    sink = JsonSink()
    logger.add(sink, format='{message}', catch=False, serialize=True)
    not_serializable = object()
    logger.bind(not_serializable=not_serializable).debug('Test')
    assert sink.dict['extra']['not_serializable'] == not_serializable
    assert bool(sink.json['record']['extra']['not_serializable'])