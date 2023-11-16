import pytest
from ulauncher.api.shared.query import Query
from ulauncher.modes.shortcuts.ShortcutResult import ShortcutResult

class TestShortcutResult:

    @pytest.fixture(autouse=True)
    def OpenAction(self, mocker):
        if False:
            print('Hello World!')
        return mocker.patch('ulauncher.modes.shortcuts.ShortcutResult.OpenAction')

    @pytest.fixture(autouse=True)
    def run_script(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.modes.shortcuts.ShortcutResult.run_script')

    @pytest.fixture
    def result(self):
        if False:
            while True:
                i = 10
        return ShortcutResult(keyword='kw', name='name', cmd='https://site/?q=%s', icon='icon_path')

    def test_keyword(self, result):
        if False:
            for i in range(10):
                print('nop')
        assert result.keyword == 'kw'

    def test_name(self, result):
        if False:
            print('Hello World!')
        assert result.name == 'name'

    def test_get_description(self, result):
        if False:
            print('Hello World!')
        assert result.get_description(Query('kw test')) == 'https://site/?q=test'
        assert result.get_description(Query('keyword test')) == 'https://site/?q=...'
        assert result.get_description(Query('goo')) == 'https://site/?q=...'

    def test_icon(self, result):
        if False:
            while True:
                i = 10
        assert isinstance(result.icon, str)

    def test_on_activation(self, result, OpenAction):
        if False:
            return 10
        result = result.on_activation(Query('kw test'))
        OpenAction.assert_called_once_with('https://site/?q=test')
        assert not isinstance(result, str)

    def test_on_activation__default_search(self, result, OpenAction):
        if False:
            while True:
                i = 10
        result.is_default_search = True
        result = result.on_activation(Query('search query'))
        OpenAction.assert_called_once_with('https://site/?q=search query')
        assert not isinstance(result, str)

    def test_on_activation__run_without_arguments(self, result, OpenAction):
        if False:
            i = 10
            return i + 15
        result.run_without_argument = True
        result = result.on_activation(Query('kw'))
        OpenAction.assert_called_once_with('https://site/?q=%s')
        assert not isinstance(result, str)

    def test_on_activation__misspelled_kw(self, result, OpenAction):
        if False:
            while True:
                i = 10
        assert result.on_activation(Query('keyword query')) == 'kw '
        assert not OpenAction.called

    def test_on_activation__run_file(self, run_script):
        if False:
            print('Hello World!')
        result = ShortcutResult(keyword='kw', name='name', cmd='/usr/bin/something/%s', icon='icon_path')
        result.on_activation(Query('kw query'))
        run_script.assert_called_once_with('/usr/bin/something/query', 'query')