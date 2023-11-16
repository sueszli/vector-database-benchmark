import time
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def refresher():
    if False:
        return 10
    from mycli.completion_refresher import CompletionRefresher
    return CompletionRefresher()

def test_ctor(refresher):
    if False:
        for i in range(10):
            print('nop')
    'Refresher object should contain a few handlers.\n\n    :param refresher:\n    :return:\n\n    '
    assert len(refresher.refreshers) > 0
    actual_handlers = list(refresher.refreshers.keys())
    expected_handlers = ['databases', 'schemata', 'tables', 'users', 'functions', 'special_commands', 'show_commands', 'keywords']
    assert expected_handlers == actual_handlers

def test_refresh_called_once(refresher):
    if False:
        return 10
    '\n\n    :param refresher:\n    :return:\n    '
    callbacks = Mock()
    sqlexecute = Mock()
    with patch.object(refresher, '_bg_refresh') as bg_refresh:
        actual = refresher.refresh(sqlexecute, callbacks)
        time.sleep(1)
        assert len(actual) == 1
        assert len(actual[0]) == 4
        assert actual[0][3] == 'Auto-completion refresh started in the background.'
        bg_refresh.assert_called_with(sqlexecute, callbacks, {})

def test_refresh_called_twice(refresher):
    if False:
        return 10
    'If refresh is called a second time, it should be restarted.\n\n    :param refresher:\n    :return:\n\n    '
    callbacks = Mock()
    sqlexecute = Mock()

    def dummy_bg_refresh(*args):
        if False:
            i = 10
            return i + 15
        time.sleep(3)
    refresher._bg_refresh = dummy_bg_refresh
    actual1 = refresher.refresh(sqlexecute, callbacks)
    time.sleep(1)
    assert len(actual1) == 1
    assert len(actual1[0]) == 4
    assert actual1[0][3] == 'Auto-completion refresh started in the background.'
    actual2 = refresher.refresh(sqlexecute, callbacks)
    time.sleep(1)
    assert len(actual2) == 1
    assert len(actual2[0]) == 4
    assert actual2[0][3] == 'Auto-completion refresh restarted.'

def test_refresh_with_callbacks(refresher):
    if False:
        while True:
            i = 10
    'Callbacks must be called.\n\n    :param refresher:\n\n    '
    callbacks = [Mock()]
    sqlexecute_class = Mock()
    sqlexecute = Mock()
    with patch('mycli.completion_refresher.SQLExecute', sqlexecute_class):
        refresher.refreshers = {}
        refresher.refresh(sqlexecute, callbacks)
        time.sleep(1)
        assert callbacks[0].call_count == 1