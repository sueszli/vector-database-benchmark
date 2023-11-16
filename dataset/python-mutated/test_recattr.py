import re
import loguru._recattrs as recattrs
from loguru import logger

def test_patch_record_file(writer):
    if False:
        i = 10
        return i + 15

    def patch(record):
        if False:
            i = 10
            return i + 15
        record['file'].name = '456'
        record['file'].path = '123/456'
    logger.add(writer, format='{file} {file.name} {file.path}')
    logger.patch(patch).info('Test')
    assert writer.read() == '456 456 123/456\n'

def test_patch_record_thread(writer):
    if False:
        for i in range(10):
            print('nop')

    def patch(record):
        if False:
            return 10
        record['thread'].id = 111
        record['thread'].name = 'Thread-111'
    logger.add(writer, format='{thread} {thread.name} {thread.id}')
    logger.patch(patch).info('Test')
    assert writer.read() == '111 Thread-111 111\n'

def test_patch_record_process(writer):
    if False:
        while True:
            i = 10

    def patch(record):
        if False:
            return 10
        record['process'].id = 123
        record['process'].name = 'Process-123'
    logger.add(writer, format='{process} {process.name} {process.id}')
    logger.patch(patch).info('Test')
    assert writer.read() == '123 Process-123 123\n'

def test_patch_record_exception(writer):
    if False:
        i = 10
        return i + 15

    def patch(record):
        if False:
            while True:
                i = 10
        (type_, value, traceback) = record['exception']
        record['exception'] = (type_, value, None)
    logger.add(writer, format='')
    try:
        1 / 0
    except ZeroDivisionError:
        logger.patch(patch).exception('Error')
    assert writer.read() == '\nZeroDivisionError: division by zero\n'

def test_level_repr():
    if False:
        return 10
    level = recattrs.RecordLevel('FOO', 123, '!!')
    assert repr(level) == "(name='FOO', no=123, icon='!!')"

def test_file_repr():
    if False:
        i = 10
        return i + 15
    file_ = recattrs.RecordFile('foo.txt', 'path/foo.txt')
    assert repr(file_) == "(name='foo.txt', path='path/foo.txt')"

def test_thread_repr():
    if False:
        return 10
    thread = recattrs.RecordThread(98765, 'thread-1')
    assert repr(thread) == "(id=98765, name='thread-1')"

def test_process_repr():
    if False:
        return 10
    process = recattrs.RecordProcess(12345, 'process-1')
    assert repr(process) == "(id=12345, name='process-1')"

def test_exception_repr():
    if False:
        print('Hello World!')
    exception = recattrs.RecordException(ValueError, ValueError('Nope'), None)
    regex = "\\(type=<class 'ValueError'>, value=ValueError\\('Nope',?\\), traceback=None\\)"
    assert re.fullmatch(regex, repr(exception))