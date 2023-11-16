from typing import Dict
from twisted.test.proto_helpers import MemoryReactor
from twisted.web.resource import Resource
from synapse.media._base import FileInfo
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.test_utils import SMALL_PNG
from tests.unittest import override_config

class MediaDomainBlockingTests(unittest.HomeserverTestCase):
    remote_media_id = 'doesnotmatter'
    remote_server_name = 'evil.com'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        file_id = 'abcdefg12345'
        file_info = FileInfo(server_name=self.remote_server_name, file_id=file_id)
        with hs.get_media_repository().media_storage.store_into_file(file_info) as (f, fname, finish):
            f.write(SMALL_PNG)
            self.get_success(finish())
        self.get_success(self.store.store_cached_remote_media(origin=self.remote_server_name, media_id=self.remote_media_id, media_type='image/png', media_length=1, time_now_ms=clock.time_msec(), upload_name='test.png', filesystem_id=file_id))

    def create_resource_dict(self) -> Dict[str, Resource]:
        if False:
            return 10
        return {'/_matrix/media': self.hs.get_media_repository_resource()}

    @override_config({'prevent_media_downloads_from': ['evil.com']})
    def test_cannot_download_blocked_media(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests to ensure that remote media which is blocked cannot be downloaded.\n        '
        response = self.make_request('GET', f'/_matrix/media/v3/download/evil.com/{self.remote_media_id}', shorthand=False)
        self.assertEqual(response.code, 404)

    @override_config({'prevent_media_downloads_from': ['not-listed.com']})
    def test_remote_media_normally_unblocked(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests to ensure that remote media is normally able to be downloaded\n        when no domain block is in place.\n        '
        response = self.make_request('GET', f'/_matrix/media/v3/download/evil.com/{self.remote_media_id}', shorthand=False)
        self.assertEqual(response.code, 200)

    @override_config({'prevent_media_downloads_from': ['evil.com'], 'dynamic_thumbnails': True})
    def test_cannot_download_blocked_media_thumbnail(self) -> None:
        if False:
            return 10
        '\n        Same test as test_cannot_download_blocked_media but for thumbnails.\n        '
        response = self.make_request('GET', f'/_matrix/media/v3/thumbnail/evil.com/{self.remote_media_id}?width=100&height=100', shorthand=False, content={'width': 100, 'height': 100})
        self.assertEqual(response.code, 404)

    @override_config({'prevent_media_downloads_from': ['not-listed.com'], 'dynamic_thumbnails': True})
    def test_remote_media_thumbnail_normally_unblocked(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Same test as test_remote_media_normally_unblocked but for thumbnails.\n        '
        response = self.make_request('GET', f'/_matrix/media/v3/thumbnail/evil.com/{self.remote_media_id}?width=100&height=100', shorthand=False)
        self.assertEqual(response.code, 200)