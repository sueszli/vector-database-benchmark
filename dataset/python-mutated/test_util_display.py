"""Tests util functions."""
import pytest
from sphinx.testing.util import strip_escseq
from sphinx.util import logging
from sphinx.util.display import SkipProgressMessage, display_chunk, progress_message, status_iterator

def test_display_chunk():
    if False:
        for i in range(10):
            print('nop')
    assert display_chunk('hello') == 'hello'
    assert display_chunk(['hello']) == 'hello'
    assert display_chunk(['hello', 'sphinx', 'world']) == 'hello .. world'
    assert display_chunk(('hello',)) == 'hello'
    assert display_chunk(('hello', 'sphinx', 'world')) == 'hello .. world'

@pytest.mark.sphinx('dummy')
def test_status_iterator_length_0(app, status, warning):
    if False:
        print('Hello World!')
    logging.setup(app, status, warning)
    status.seek(0)
    status.truncate(0)
    yields = list(status_iterator(['hello', 'sphinx', 'world'], 'testing ... '))
    output = strip_escseq(status.getvalue())
    assert 'testing ... hello sphinx world \n' in output
    assert yields == ['hello', 'sphinx', 'world']

@pytest.mark.sphinx('dummy')
def test_status_iterator_verbosity_0(app, status, warning):
    if False:
        print('Hello World!')
    logging.setup(app, status, warning)
    status.seek(0)
    status.truncate(0)
    yields = list(status_iterator(['hello', 'sphinx', 'world'], 'testing ... ', length=3, verbosity=0))
    output = strip_escseq(status.getvalue())
    assert 'testing ... [ 33%] hello\r' in output
    assert 'testing ... [ 67%] sphinx\r' in output
    assert 'testing ... [100%] world\r\n' in output
    assert yields == ['hello', 'sphinx', 'world']

@pytest.mark.sphinx('dummy')
def test_status_iterator_verbosity_1(app, status, warning):
    if False:
        for i in range(10):
            print('nop')
    logging.setup(app, status, warning)
    status.seek(0)
    status.truncate(0)
    yields = list(status_iterator(['hello', 'sphinx', 'world'], 'testing ... ', length=3, verbosity=1))
    output = strip_escseq(status.getvalue())
    assert 'testing ... [ 33%] hello\n' in output
    assert 'testing ... [ 67%] sphinx\n' in output
    assert 'testing ... [100%] world\n\n' in output
    assert yields == ['hello', 'sphinx', 'world']

def test_progress_message(app, status, warning):
    if False:
        for i in range(10):
            print('nop')
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)
    with progress_message('testing'):
        logger.info('blah ', nonl=True)
    output = strip_escseq(status.getvalue())
    assert 'testing... blah done\n' in output
    with progress_message('testing'):
        raise SkipProgressMessage('Reason: %s', 'error')
    output = strip_escseq(status.getvalue())
    assert 'testing... skipped\nReason: error\n' in output
    try:
        with progress_message('testing'):
            raise
    except Exception:
        pass
    output = strip_escseq(status.getvalue())
    assert 'testing... failed\n' in output

    @progress_message('testing')
    def func():
        if False:
            return 10
        logger.info('in func ', nonl=True)
    func()
    output = strip_escseq(status.getvalue())
    assert 'testing... in func done\n' in output