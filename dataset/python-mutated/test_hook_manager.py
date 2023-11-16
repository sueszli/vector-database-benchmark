import logging
import sys
from contextlib import contextmanager
from threading import Condition
from unittest import mock
import click
import pytest
from alive_progress.core.hook_manager import buffered_hook_manager
from alive_progress.utils.terminal import get_term

@contextmanager
def install_hook(hook_manager=None):
    if False:
        print('Hello World!')
    if hook_manager is None:
        hook_manager = hook('nice {}! ')
    hook_manager.install()
    yield
    hook_manager.uninstall()

def hook(header):
    if False:
        while True:
            i = 10
    return buffered_hook_manager(header, lambda : 35, Condition(), get_term())

@pytest.fixture(params=[('ok', 'nice 35! ok\n'), ('ok  ', 'nice 35! ok\n'), ('  ok', 'nice 35!   ok\n'), ('  ok  ', 'nice 35!   ok\n')])
def print_data(request):
    if False:
        i = 10
        return i + 15
    yield request.param

def test_hook_manager_captures_stdout(print_data, capsys):
    if False:
        i = 10
        return i + 15
    (out, expected) = print_data
    with install_hook():
        print(out)
    assert capsys.readouterr().out == expected

def test_hook_manager_captures_bytes_stdout(print_data, capsys):
    if False:
        return 10
    (out, expected) = print_data
    with install_hook():
        click.echo(out)
    assert capsys.readouterr().out == expected

def _hook_manager_captures_logging(capsys):
    if False:
        for i in range(10):
            print('nop')
    import sys
    logging.basicConfig(stream=sys.stderr)
    logger = logging.getLogger('?name?')
    with install_hook():
        logger.error('oops')
    assert capsys.readouterr().err == 'nice! ERROR:?name?:oops\n'

def test_hook_manager_captures_multiple_lines(capsys):
    if False:
        i = 10
        return i + 15
    with install_hook():
        print('ok1\nok2')
    assert capsys.readouterr().out == 'nice 35! ok1\n         ok2\n'

def test_hook_manager_can_be_disabled(capsys):
    if False:
        return 10
    with install_hook(hook('')):
        print('ok')
    assert capsys.readouterr().out == 'ok\n'

def test_hook_manager_flush(capsys):
    if False:
        for i in range(10):
            print('nop')
    hook_manager = hook('')
    with install_hook(hook_manager):
        print('ok', end='')
        assert capsys.readouterr().out == ''
        hook_manager.flush_buffers()
        assert capsys.readouterr().out == 'ok\n'
    hook_manager.flush_buffers()
    assert capsys.readouterr().out == ''

def test_hook_manager_do_clear_line_on_stdout():
    if False:
        print('Hello World!')
    term = get_term()
    hook_manager = buffered_hook_manager('', None, Condition(), term)
    m_clear = mock.Mock()
    with install_hook(hook_manager), mock.patch.dict(term.__dict__, clear_line=m_clear):
        print('some')
    m_clear.assert_called()

def test_hook_manager_do_not_flicker_screen_when_logging():
    if False:
        for i in range(10):
            print('nop')
    logging.basicConfig()
    logger = logging.getLogger()
    term = get_term()
    hook_manager = buffered_hook_manager('', None, Condition(), term)
    m_clear = mock.Mock()
    with install_hook(hook_manager), mock.patch.dict(term.__dict__, clear_line=m_clear):
        logger.error('oops')
    m_clear.assert_not_called()

@pytest.fixture
def handlers():
    if False:
        while True:
            i = 10
    handlers = (logging.StreamHandler(sys.stderr), logging.StreamHandler(sys.stdout), logging.FileHandler('/dev/null', delay=True))
    [logging.root.addHandler(h) for h in handlers]
    yield handlers
    [logging.root.removeHandler(h) for h in handlers]

def test_install(handlers):
    if False:
        return 10
    hook_manager = hook('')
    with mock.patch('logging.StreamHandler.setStream') as mock_set_stream:
        hook_manager.install()
    mock_set_stream.assert_has_calls(tuple((mock.call(mock.ANY) for _ in handlers)))

def test_uninstall(handlers):
    if False:
        print('Hello World!')
    hook_manager = hook('')
    hook_manager.install()
    with mock.patch('logging.StreamHandler.setStream') as mock_set_stream:
        hook_manager.uninstall()
    mock_set_stream.assert_has_calls(tuple((mock.call(mock.ANY) for _ in handlers)))