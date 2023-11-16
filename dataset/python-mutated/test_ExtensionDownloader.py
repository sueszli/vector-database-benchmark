from unittest import mock
import pytest
from ulauncher.modes.extensions.ExtensionDb import ExtensionDb, ExtensionRecord
from ulauncher.modes.extensions.ExtensionDownloader import ExtensionDownloader

class TestExtensionDownloader:

    @pytest.fixture
    def ext_db(self):
        if False:
            for i in range(10):
                print('nop')
        return mock.create_autospec(ExtensionDb)

    @pytest.fixture
    def downloader(self, ext_db):
        if False:
            i = 10
            return i + 15
        return ExtensionDownloader(ext_db)

    @pytest.fixture(autouse=True)
    def remote(self, mocker):
        if False:
            i = 10
            return i + 15
        remote = mocker.patch('ulauncher.modes.extensions.ExtensionDownloader.ExtensionRemote').return_value
        remote.extension_id = 'com.github.ulauncher.ulauncher-timer'
        remote.get_download_url.return_value = 'https://github.com/Ulauncher/ulauncher-timer/archive/master.tar.gz'
        remote.get_compatible_hash.return_value = '64e106c'
        return remote

    def test_check_update__returns_new_version(self, downloader, ext_db):
        if False:
            while True:
                i = 10
        ext_id = 'com.github.ulauncher.ulauncher-timer'
        ext_db.get.return_value = ExtensionRecord(id=ext_id, url='https://github.com/Ulauncher/ulauncher-timer', updated_at='2017-01-01')
        assert downloader.check_update(ext_id) == (True, '64e106c')