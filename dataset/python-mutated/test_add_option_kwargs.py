import pytest
from loguru import logger

def test_file_mode_a(tmp_path):
    if False:
        print('Hello World!')
    file = tmp_path / 'test.log'
    file.write_text('base\n')
    logger.add(file, format='{message}', mode='a')
    logger.debug('msg')
    assert file.read_text() == 'base\nmsg\n'

def test_file_mode_w(tmp_path):
    if False:
        return 10
    file = tmp_path / 'test.log'
    file.write_text('base\n')
    logger.add(file, format='{message}', mode='w')
    logger.debug('msg')
    assert file.read_text() == 'msg\n'

def test_file_auto_buffering(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    dummy_filepath = tmp_path / 'dummy.txt'
    with open(str(dummy_filepath), buffering=-1, mode='w') as dummy_file:
        dummy_file.write('.' * 127)
        if dummy_filepath.read_text() != '':
            pytest.skip('Size buffer for text files is too small.')
        dummy_file.write('.' * (65536 - 127))
        if dummy_filepath.read_text() == '':
            pytest.skip('Size buffer for text files is too big.')
    filepath = tmp_path / 'test.log'
    logger.add(filepath, format='{message}', buffering=-1)
    logger.debug('A short message.')
    assert filepath.read_text() == ''
    logger.debug('A long message' + '.' * 65536)
    assert filepath.read_text() != ''

def test_file_line_buffering(tmp_path):
    if False:
        while True:
            i = 10
    filepath = tmp_path / 'test.log'
    logger.add(filepath, format=lambda _: '{message}', buffering=1)
    logger.debug('Without newline')
    assert filepath.read_text() == ''
    logger.debug('With newline\n')
    assert filepath.read_text() != ''

def test_invalid_function_kwargs():
    if False:
        for i in range(10):
            print('nop')

    def function(message):
        if False:
            return 10
        pass
    with pytest.raises(TypeError, match='add\\(\\) got an unexpected keyword argument'):
        logger.add(function, b='X')

def test_invalid_file_object_kwargs():
    if False:
        for i in range(10):
            print('nop')

    class Writer:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.out = ''

        def write(self, m):
            if False:
                while True:
                    i = 10
            pass
    writer = Writer()
    with pytest.raises(TypeError, match='add\\(\\) got an unexpected keyword argument'):
        logger.add(writer, format='{message}', kw1='1', kw2='2')

def test_invalid_file_kwargs():
    if False:
        return 10
    with pytest.raises(TypeError, match='.*keyword argument;*'):
        logger.add('file.log', nope=123)

def test_invalid_coroutine_kwargs():
    if False:
        i = 10
        return i + 15

    async def foo():
        pass
    with pytest.raises(TypeError, match='add\\(\\) got an unexpected keyword argument'):
        logger.add(foo, nope=123)