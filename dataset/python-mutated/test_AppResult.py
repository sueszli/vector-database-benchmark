import shutil
from pathlib import Path
import pytest
from gi.repository import Gio
from ulauncher.api.shared.query import Query
from ulauncher.modes.apps.AppResult import AppResult
from ulauncher.utils.json_utils import json_load
ENTRIES_DIR = Path(__file__).parent.joinpath('mock_desktop_entries').resolve()

class TestAppResult:

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        Path('/tmp/ulauncher-test').mkdir(parents=True, exist_ok=True)

    def teardown_class(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree('/tmp/ulauncher-test')

    @pytest.fixture(autouse=True)
    def patch_DesktopAppInfo_new(self, mocker):
        if False:
            i = 10
            return i + 15

        def mkappinfo(app_id):
            if False:
                print('Hello World!')
            return Gio.DesktopAppInfo.new_from_filename(f'{ENTRIES_DIR}/{app_id}')
        return mocker.patch('ulauncher.modes.apps.AppResult.Gio.DesktopAppInfo.new', new=mkappinfo)

    @pytest.fixture(autouse=True)
    def patch_DesktopAppInfo_get_all(self, mocker):
        if False:
            while True:
                i = 10

        def get_all_appinfo():
            if False:
                print('Hello World!')
            return map(Gio.DesktopAppInfo.new, ['trueapp.desktop', 'falseapp.desktop'])
        return mocker.patch('ulauncher.modes.apps.AppResult.Gio.DesktopAppInfo.get_all', new=get_all_appinfo)

    @pytest.fixture
    def app1(self):
        if False:
            print('Hello World!')
        return AppResult.from_id('trueapp.desktop')

    @pytest.fixture
    def app2(self):
        if False:
            i = 10
            return i + 15
        return AppResult.from_id('falseapp.desktop')

    @pytest.fixture(autouse=True)
    def app_starts(self, mocker):
        if False:
            return 10
        app_starts = json_load('/tmp/ulauncher-test/app_starts.json')
        app_starts.update({'falseapp.desktop': 3000, 'trueapp.desktop': 765})
        return mocker.patch('ulauncher.modes.apps.AppResult.app_starts', new=app_starts)

    def test_get_name(self, app1):
        if False:
            while True:
                i = 10
        assert app1.name == 'TrueApp - Full Name'

    def test_get_description(self, app1):
        if False:
            while True:
                i = 10
        assert app1.get_description(Query('q')) == 'Your own yes-man'

    def test_icon(self, app1):
        if False:
            print('Hello World!')
        assert app1.icon == 'dialog-yes'

    def test_search_score(self, app1):
        if False:
            print('Hello World!')
        assert app1.search_score('true') > app1.search_score('trivago')

    def test_on_activation(self, app1, mocker, app_starts):
        if False:
            print('Hello World!')
        launch_app = mocker.patch('ulauncher.modes.apps.AppResult.launch_app')
        assert app1.on_activation(Query('query')) is launch_app.return_value
        launch_app.assert_called_with('trueapp.desktop')
        assert app_starts.get('trueapp.desktop') == 766

    def test_get_most_frequent(self):
        if False:
            for i in range(10):
                print('nop')
        assert len(AppResult.get_most_frequent()) == 2
        assert AppResult.get_most_frequent()[0].name == 'FalseApp - Full Name'