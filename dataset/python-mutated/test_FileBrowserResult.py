from ulauncher.modes.file_browser.FileBrowserResult import FileBrowserResult

class TestFileBrowserResult:

    def test_get_name(self):
        if False:
            while True:
                i = 10
        assert FileBrowserResult('/tmp/dir').name == 'dir'

    def test_icon(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(FileBrowserResult('/tmp/').icon, str)