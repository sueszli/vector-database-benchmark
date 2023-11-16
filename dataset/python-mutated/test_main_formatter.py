from unittest.mock import Mock, patch
import pytest
from streamlink.plugin import Plugin
from streamlink_cli.main import get_formatter
from streamlink_cli.utils import datetime

@pytest.fixture(scope='module')
def plugin():
    if False:
        print('Hello World!')

    class FakePlugin(Plugin):
        __module__ = 'FAKE'

        def _get_streams(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    plugin = FakePlugin(Mock(), 'https://foo/bar')
    plugin.id = 'ID'
    plugin.author = 'AUTHOR'
    plugin.category = 'CATEGORY'
    plugin.title = 'TITLE'
    return plugin

@pytest.mark.parametrize(('formatterinput', 'expected'), [('{url}', 'https://foo/bar'), ('{plugin}', 'FAKE'), ('{id}', 'ID'), ('{author}', 'AUTHOR'), ('{category}', 'CATEGORY'), ('{game}', 'CATEGORY'), ('{title}', 'TITLE'), ('{time}', '2000-01-01_00-00-00'), ('{time:%Y}', '2000')])
@patch('streamlink_cli.utils.datetime.now', Mock(return_value=datetime(2000, 1, 1, 0, 0, 0, 0)))
@patch('streamlink_cli.main.args', Mock(url='https://foo/bar'))
def test_get_formatter(plugin, formatterinput, expected):
    if False:
        while True:
            i = 10
    formatter = get_formatter(plugin)
    assert formatter.title(formatterinput) == expected