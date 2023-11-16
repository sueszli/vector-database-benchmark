from types import SimpleNamespace
import pytest
from ulauncher.modes.file_browser.FileBrowserMode import FileBrowserMode

class MockDirEntry:

    def __init__(self, name, atime, is_file=True):
        if False:
            i = 10
            return i + 15
        self.name = name
        self._atime = atime
        self._is_file = is_file

    def is_file(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_file

    def stat(self):
        if False:
            i = 10
            return i + 15
        return SimpleNamespace(st_atime=self._atime)

class TestFileBrowserMode:

    @pytest.fixture(autouse=True)
    def scandir(self, mocker):
        if False:
            while True:
                i = 10
        sd = mocker.patch('ulauncher.modes.file_browser.FileBrowserMode.os.scandir')
        sd.return_value = [MockDirEntry('a', 1655837759), MockDirEntry('D', 1655839002), MockDirEntry('B', 1655839892), MockDirEntry('c', 1655837959)]
        return sd

    @pytest.fixture
    def mode(self):
        if False:
            return 10
        return FileBrowserMode()

    def test_is_enabled(self, mode):
        if False:
            while True:
                i = 10
        assert mode.is_enabled('~/Downloads')
        assert mode.is_enabled('~')
        assert mode.is_enabled('$USER/Videos')
        assert mode.is_enabled('/usr/bin')
        assert mode.is_enabled('/')
        assert mode.is_enabled(' /foo/bar')
        assert not mode.is_enabled('test')
        assert not mode.is_enabled('+')
        assert not mode.is_enabled(' ')

    def test_list_files(self, mode):
        if False:
            for i in range(10):
                print('nop')
        assert mode.list_files('path') == ['a', 'B', 'c', 'D']
        assert mode.list_files('path', sort_by_atime=True) == ['B', 'D', 'c', 'a']

    def test_filter_dot_files(self, mode):
        if False:
            while True:
                i = 10
        assert mode.filter_dot_files(['a', '.b', 'c', '.d']) == ['a', 'c']

    def test_handle_query__path_from_q_exists__dir_listing_rendered(self):
        if False:
            i = 10
            return i + 15
        flattened_results = [str(r.path) for r in FileBrowserMode().handle_query('/tmp/')]
        assert flattened_results == ['/tmp/B', '/tmp/D', '/tmp/c', '/tmp/a']

    def test_handle_query__invalid_path__empty_list_rendered(self, mode):
        if False:
            print('Hello World!')
        assert mode.handle_query('~~') == []