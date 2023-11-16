import os
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
try:
    import lxml
except ImportError:
    lxml = None

class URLPreviewTests(unittest.HomeserverTestCase):
    if not lxml:
        skip = 'url preview feature requires lxml'

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        config = self.default_config()
        config['url_preview_enabled'] = True
        config['max_spider_size'] = 9999999
        config['url_preview_ip_range_blacklist'] = ('192.168.1.1', '1.0.0.0/8', '3fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff', '2001:800::/21')
        self.storage_path = self.mktemp()
        self.media_store_path = self.mktemp()
        os.mkdir(self.storage_path)
        os.mkdir(self.media_store_path)
        config['media_store_path'] = self.media_store_path
        provider_config = {'module': 'synapse.media.storage_provider.FileStorageProviderBackend', 'store_local': True, 'store_synchronous': False, 'store_remote': True, 'config': {'directory': self.storage_path}}
        config['media_storage_providers'] = [provider_config]
        return self.setup_test_homeserver(config=config)

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        media_repo = hs.get_media_repository()
        assert media_repo.url_previewer is not None
        self.url_previewer = media_repo.url_previewer

    def test_all_urls_allowed(self) -> None:
        if False:
            print('Hello World!')
        self.assertFalse(self.url_previewer._is_url_blocked('http://matrix.org'))
        self.assertFalse(self.url_previewer._is_url_blocked('https://matrix.org'))
        self.assertFalse(self.url_previewer._is_url_blocked('http://localhost:8000'))
        self.assertFalse(self.url_previewer._is_url_blocked('http://user:pass@matrix.org'))

    @override_config({'url_preview_url_blacklist': [{'username': 'user'}, {'scheme': 'http', 'netloc': 'matrix.org'}]})
    def test_blocked_url(self) -> None:
        if False:
            print('Hello World!')
        self.assertTrue(self.url_previewer._is_url_blocked('http://matrix.org'))
        self.assertFalse(self.url_previewer._is_url_blocked('https://matrix.org'))
        self.assertTrue(self.url_previewer._is_url_blocked('http://user:pass@example.com'))
        self.assertTrue(self.url_previewer._is_url_blocked('http://user@example.com'))

    @override_config({'url_preview_url_blacklist': [{'netloc': '*.example.com'}]})
    def test_glob_blocked_url(self) -> None:
        if False:
            while True:
                i = 10
        self.assertTrue(self.url_previewer._is_url_blocked('http://foo.example.com'))
        self.assertTrue(self.url_previewer._is_url_blocked('http://.example.com'))
        self.assertFalse(self.url_previewer._is_url_blocked('https://example.com'))

    @override_config({'url_preview_url_blacklist': [{'netloc': '^.+\\.example\\.com'}]})
    def test_regex_blocked_urL(self) -> None:
        if False:
            return 10
        self.assertTrue(self.url_previewer._is_url_blocked('http://foo.example.com'))
        self.assertFalse(self.url_previewer._is_url_blocked('http://.example.com'))
        self.assertFalse(self.url_previewer._is_url_blocked('https://example.com'))