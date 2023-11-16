import os
import shutil
import tempfile
from binascii import unhexlify
from io import BytesIO
from typing import Any, BinaryIO, ClassVar, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock
from urllib import parse
import attr
from parameterized import parameterized, parameterized_class
from PIL import Image as Image
from typing_extensions import Literal
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.test.proto_helpers import MemoryReactor
from twisted.web.resource import Resource
from synapse.api.errors import Codes
from synapse.events import EventBase
from synapse.http.types import QueryParams
from synapse.logging.context import make_deferred_yieldable
from synapse.media._base import FileInfo, ThumbnailInfo
from synapse.media.filepath import MediaFilePaths
from synapse.media.media_storage import MediaStorage, ReadableFileWrapper
from synapse.media.storage_provider import FileStorageProviderBackend
from synapse.module_api import ModuleApi
from synapse.module_api.callbacks.spamchecker_callbacks import load_legacy_spam_checkers
from synapse.rest import admin
from synapse.rest.client import login
from synapse.rest.media.thumbnail_resource import ThumbnailResource
from synapse.server import HomeServer
from synapse.types import JsonDict, RoomAlias
from synapse.util import Clock
from tests import unittest
from tests.server import FakeChannel
from tests.test_utils import SMALL_PNG
from tests.utils import default_config

class MediaStorageTests(unittest.HomeserverTestCase):
    needs_threadpool = True

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.test_dir = tempfile.mkdtemp(prefix='synapse-tests-')
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.primary_base_path = os.path.join(self.test_dir, 'primary')
        self.secondary_base_path = os.path.join(self.test_dir, 'secondary')
        hs.config.media.media_store_path = self.primary_base_path
        storage_providers = [FileStorageProviderBackend(hs, self.secondary_base_path)]
        self.filepaths = MediaFilePaths(self.primary_base_path)
        self.media_storage = MediaStorage(hs, self.primary_base_path, self.filepaths, storage_providers)

    def test_ensure_media_is_in_local_cache(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        media_id = 'some_media_id'
        test_body = 'Test\n'
        rel_path = self.filepaths.local_media_filepath_rel(media_id)
        secondary_path = os.path.join(self.secondary_base_path, rel_path)
        os.makedirs(os.path.dirname(secondary_path))
        with open(secondary_path, 'w') as f:
            f.write(test_body)
        file_info = FileInfo(None, media_id)
        x = defer.ensureDeferred(self.media_storage.ensure_media_is_in_local_cache(file_info))
        self.wait_on_thread(x)
        local_path = self.get_success(x)
        self.assertTrue(os.path.exists(local_path))
        self.assertEqual(os.path.commonprefix([self.primary_base_path, local_path]), self.primary_base_path)
        with open(local_path) as f:
            body = f.read()
        self.assertEqual(test_body, body)

@attr.s(auto_attribs=True, slots=True, frozen=True)
class _TestImage:
    """An image for testing thumbnailing with the expected results

    Attributes:
        data: The raw image to thumbnail
        content_type: The type of the image as a content type, e.g. "image/png"
        extension: The extension associated with the format, e.g. ".png"
        expected_cropped: The expected bytes from cropped thumbnailing, or None if
            test should just check for success.
        expected_scaled: The expected bytes from scaled thumbnailing, or None if
            test should just check for a valid image returned.
        expected_found: True if the file should exist on the server, or False if
            a 404/400 is expected.
        unable_to_thumbnail: True if we expect the thumbnailing to fail (400), or
            False if the thumbnailing should succeed or a normal 404 is expected.
        is_inline: True if we expect the file to be served using an inline
            Content-Disposition or False if we expect an attachment.
    """
    data: bytes
    content_type: bytes
    extension: bytes
    expected_cropped: Optional[bytes] = None
    expected_scaled: Optional[bytes] = None
    expected_found: bool = True
    unable_to_thumbnail: bool = False
    is_inline: bool = True

@parameterized_class(('test_image',), [(_TestImage(SMALL_PNG, b'image/png', b'.png', unhexlify(b'89504e470d0a1a0a0000000d4948445200000020000000200806000000737a7af40000001a49444154789cedc101010000008220ffaf6e484001000000ef0610200001194334ee0000000049454e44ae426082'), unhexlify(b'89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082')),), (_TestImage(unhexlify(b'89504e470d0a1a0a0000000d4948445200000001000000010100000000376ef9240000000274524e5300010194fdae0000000a49444154789c636800000082008177cd72b60000000049454e44ae426082'), b'image/png', b'.png'),), (_TestImage(unhexlify(b'524946461a000000574542505650384c0d0000002f00000010071011118888fe0700'), b'image/webp', b'.webp'),), (_TestImage(b'', b'image/gif', b'.gif', expected_found=False, unable_to_thumbnail=True),), (_TestImage(b'<?xml version="1.0"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n\n<svg xmlns="http://www.w3.org/2000/svg"\n     width="400" height="400">\n  <circle cx="100" cy="100" r="50" stroke="black"\n    stroke-width="5" fill="red" />\n</svg>', b'image/svg', b'.svg', expected_found=False, unable_to_thumbnail=True, is_inline=False),)])
class MediaRepoTests(unittest.HomeserverTestCase):
    test_image: ClassVar[_TestImage]
    hijack_auth = True
    user_id = '@test:user'

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        self.fetches: List[Tuple['Deferred[Tuple[bytes, Tuple[int, Dict[bytes, List[bytes]]]]]', str, str, Optional[QueryParams]]] = []

        def get_file(destination: str, path: str, output_stream: BinaryIO, args: Optional[QueryParams]=None, retry_on_dns_fail: bool=True, max_size: Optional[int]=None, ignore_backoff: bool=False) -> 'Deferred[Tuple[int, Dict[bytes, List[bytes]]]]':
            if False:
                i = 10
                return i + 15
            'A mock for MatrixFederationHttpClient.get_file.'

            def write_to(r: Tuple[bytes, Tuple[int, Dict[bytes, List[bytes]]]]) -> Tuple[int, Dict[bytes, List[bytes]]]:
                if False:
                    i = 10
                    return i + 15
                (data, response) = r
                output_stream.write(data)
                return response
            d: Deferred[Tuple[bytes, Tuple[int, Dict[bytes, List[bytes]]]]] = Deferred()
            self.fetches.append((d, destination, path, args))
            d_after_callback = d.addCallback(write_to)
            return make_deferred_yieldable(d_after_callback)
        client = Mock()
        client.get_file = get_file
        self.storage_path = self.mktemp()
        self.media_store_path = self.mktemp()
        os.mkdir(self.storage_path)
        os.mkdir(self.media_store_path)
        config = self.default_config()
        config['media_store_path'] = self.media_store_path
        config['max_image_pixels'] = 2000000
        provider_config = {'module': 'synapse.media.storage_provider.FileStorageProviderBackend', 'store_local': True, 'store_synchronous': False, 'store_remote': True, 'config': {'directory': self.storage_path}}
        config['media_storage_providers'] = [provider_config]
        hs = self.setup_test_homeserver(config=config, federation_http_client=client)
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        self.media_repo = hs.get_media_repository()
        self.media_id = 'example.com/12345'

    def create_resource_dict(self) -> Dict[str, Resource]:
        if False:
            return 10
        resources = super().create_resource_dict()
        resources['/_matrix/media'] = self.hs.get_media_repository_resource()
        return resources

    def _req(self, content_disposition: Optional[bytes], include_content_type: bool=True) -> FakeChannel:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('GET', f'/_matrix/media/v3/download/{self.media_id}', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(len(self.fetches), 1)
        self.assertEqual(self.fetches[0][1], 'example.com')
        self.assertEqual(self.fetches[0][2], '/_matrix/media/r0/download/' + self.media_id)
        self.assertEqual(self.fetches[0][3], {'allow_remote': 'false', 'timeout_ms': '20000'})
        headers = {b'Content-Length': [b'%d' % len(self.test_image.data)]}
        if include_content_type:
            headers[b'Content-Type'] = [self.test_image.content_type]
        if content_disposition:
            headers[b'Content-Disposition'] = [content_disposition]
        self.fetches[0][0].callback((self.test_image.data, (len(self.test_image.data), headers)))
        self.pump()
        self.assertEqual(channel.code, 200)
        return channel

    def test_handle_missing_content_type(self) -> None:
        if False:
            print('Hello World!')
        channel = self._req(b'attachment; filename=out' + self.test_image.extension, include_content_type=False)
        headers = channel.headers
        self.assertEqual(channel.code, 200)
        self.assertEqual(headers.getRawHeaders(b'Content-Type'), [b'application/octet-stream'])

    def test_disposition_filename_ascii(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        If the filename is filename=<ascii> then Synapse will decode it as an\n        ASCII string, and use filename= in the response.\n        '
        channel = self._req(b'attachment; filename=out' + self.test_image.extension)
        headers = channel.headers
        self.assertEqual(headers.getRawHeaders(b'Content-Type'), [self.test_image.content_type])
        self.assertEqual(headers.getRawHeaders(b'Content-Disposition'), [(b'inline' if self.test_image.is_inline else b'attachment') + b'; filename=out' + self.test_image.extension])

    def test_disposition_filenamestar_utf8escaped(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        If the filename is filename=*utf8''<utf8 escaped> then Synapse will\n        correctly decode it as the UTF-8 string, and use filename* in the\n        response.\n        "
        filename = parse.quote('☃'.encode()).encode('ascii')
        channel = self._req(b"attachment; filename*=utf-8''" + filename + self.test_image.extension)
        headers = channel.headers
        self.assertEqual(headers.getRawHeaders(b'Content-Type'), [self.test_image.content_type])
        self.assertEqual(headers.getRawHeaders(b'Content-Disposition'), [(b'inline' if self.test_image.is_inline else b'attachment') + b"; filename*=utf-8''" + filename + self.test_image.extension])

    def test_disposition_none(self) -> None:
        if False:
            return 10
        '\n        If there is no filename, Content-Disposition should only\n        be a disposition type.\n        '
        channel = self._req(None)
        headers = channel.headers
        self.assertEqual(headers.getRawHeaders(b'Content-Type'), [self.test_image.content_type])
        self.assertEqual(headers.getRawHeaders(b'Content-Disposition'), [b'inline' if self.test_image.is_inline else b'attachment'])

    def test_thumbnail_crop(self) -> None:
        if False:
            return 10
        'Test that a cropped remote thumbnail is available.'
        self._test_thumbnail('crop', self.test_image.expected_cropped, expected_found=self.test_image.expected_found, unable_to_thumbnail=self.test_image.unable_to_thumbnail)

    def test_thumbnail_scale(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that a scaled remote thumbnail is available.'
        self._test_thumbnail('scale', self.test_image.expected_scaled, expected_found=self.test_image.expected_found, unable_to_thumbnail=self.test_image.unable_to_thumbnail)

    def test_invalid_type(self) -> None:
        if False:
            print('Hello World!')
        'An invalid thumbnail type is never available.'
        self._test_thumbnail('invalid', None, expected_found=False, unable_to_thumbnail=self.test_image.unable_to_thumbnail)

    @unittest.override_config({'thumbnail_sizes': [{'width': 32, 'height': 32, 'method': 'scale'}]})
    def test_no_thumbnail_crop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Override the config to generate only scaled thumbnails, but request a cropped one.\n        '
        self._test_thumbnail('crop', None, expected_found=False, unable_to_thumbnail=self.test_image.unable_to_thumbnail)

    @unittest.override_config({'thumbnail_sizes': [{'width': 32, 'height': 32, 'method': 'crop'}]})
    def test_no_thumbnail_scale(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Override the config to generate only cropped thumbnails, but request a scaled one.\n        '
        self._test_thumbnail('scale', None, expected_found=False, unable_to_thumbnail=self.test_image.unable_to_thumbnail)

    def test_thumbnail_repeated_thumbnail(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that fetching the same thumbnail works, and deleting the on disk\n        thumbnail regenerates it.\n        '
        self._test_thumbnail('scale', self.test_image.expected_scaled, expected_found=self.test_image.expected_found, unable_to_thumbnail=self.test_image.unable_to_thumbnail)
        if not self.test_image.expected_found:
            return
        params = '?width=32&height=32&method=scale'
        channel = self.make_request('GET', f'/_matrix/media/v3/thumbnail/{self.media_id}{params}', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 200)
        if self.test_image.expected_scaled:
            self.assertEqual(channel.result['body'], self.test_image.expected_scaled, channel.result['body'])
        (origin, media_id) = self.media_id.split('/')
        info = self.get_success(self.store.get_cached_remote_media(origin, media_id))
        assert info is not None
        file_id = info.filesystem_id
        thumbnail_dir = self.media_repo.filepaths.remote_media_thumbnail_dir(origin, file_id)
        shutil.rmtree(thumbnail_dir, ignore_errors=True)
        channel = self.make_request('GET', f'/_matrix/media/v3/thumbnail/{self.media_id}{params}', shorthand=False, await_result=False)
        self.pump()
        self.assertEqual(channel.code, 200)
        if self.test_image.expected_scaled:
            self.assertEqual(channel.result['body'], self.test_image.expected_scaled, channel.result['body'])

    def _test_thumbnail(self, method: str, expected_body: Optional[bytes], expected_found: bool, unable_to_thumbnail: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Test the given thumbnailing method works as expected.\n\n        Args:\n            method: The thumbnailing method to use (crop, scale).\n            expected_body: The expected bytes from thumbnailing, or None if\n                test should just check for a valid image.\n            expected_found: True if the file should exist on the server, or False if\n                a 404/400 is expected.\n            unable_to_thumbnail: True if we expect the thumbnailing to fail (400), or\n                False if the thumbnailing should succeed or a normal 404 is expected.\n        '
        params = '?width=32&height=32&method=' + method
        channel = self.make_request('GET', f'/_matrix/media/r0/thumbnail/{self.media_id}{params}', shorthand=False, await_result=False)
        self.pump()
        headers = {b'Content-Length': [b'%d' % len(self.test_image.data)], b'Content-Type': [self.test_image.content_type]}
        self.fetches[0][0].callback((self.test_image.data, (len(self.test_image.data), headers)))
        self.pump()
        if expected_found:
            self.assertEqual(channel.code, 200)
            self.assertEqual(channel.headers.getRawHeaders(b'Cross-Origin-Resource-Policy'), [b'cross-origin'])
            if expected_body is not None:
                self.assertEqual(channel.result['body'], expected_body, channel.result['body'])
            else:
                Image.open(BytesIO(channel.result['body']))
        elif unable_to_thumbnail:
            self.assertEqual(channel.code, 400)
            self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': "Cannot find any thumbnails for the requested media ('/_matrix/media/r0/thumbnail/example.com/12345'). This might mean the media is not a supported_media_format=(image/jpeg, image/jpg, image/webp, image/gif, image/png) or that thumbnailing failed for some other reason. (Dynamic thumbnails are disabled on this server.)"})
        else:
            self.assertEqual(channel.code, 404)
            self.assertEqual(channel.json_body, {'errcode': 'M_NOT_FOUND', 'error': "Not found '/_matrix/media/r0/thumbnail/example.com/12345'"})

    @parameterized.expand([('crop', 16), ('crop', 64), ('scale', 16), ('scale', 64)])
    def test_same_quality(self, method: str, desired_size: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that choosing between thumbnails with the same quality rating succeeds.\n\n        We are not particular about which thumbnail is chosen.'
        content_type = self.test_image.content_type.decode()
        media_repo = self.hs.get_media_repository()
        thumbnail_resouce = ThumbnailResource(self.hs, media_repo, media_repo.media_storage)
        self.assertIsNotNone(thumbnail_resouce._select_thumbnail(desired_width=desired_size, desired_height=desired_size, desired_method=method, desired_type=content_type, thumbnail_infos=[ThumbnailInfo(width=32, height=32, method=method, type=content_type, length=256), ThumbnailInfo(width=32, height=32, method=method, type=content_type, length=256)], file_id=f'image{self.test_image.extension.decode()}', url_cache=False, server_name=None))

    def test_x_robots_tag_header(self) -> None:
        if False:
            return 10
        '\n        Tests that the `X-Robots-Tag` header is present, which informs web crawlers\n        to not index, archive, or follow links in media.\n        '
        channel = self._req(b'attachment; filename=out' + self.test_image.extension)
        headers = channel.headers
        self.assertEqual(headers.getRawHeaders(b'X-Robots-Tag'), [b'noindex, nofollow, noarchive, noimageindex'])

    def test_cross_origin_resource_policy_header(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that the Cross-Origin-Resource-Policy header is set to "cross-origin"\n        allowing web clients to embed media from the downloads API.\n        '
        channel = self._req(b'attachment; filename=out' + self.test_image.extension)
        headers = channel.headers
        self.assertEqual(headers.getRawHeaders(b'Cross-Origin-Resource-Policy'), [b'cross-origin'])

class TestSpamCheckerLegacy:
    """A spam checker module that rejects all media that includes the bytes
    `evil`.

    Uses the legacy Spam-Checker API.
    """

    def __init__(self, config: Dict[str, Any], api: ModuleApi) -> None:
        if False:
            while True:
                i = 10
        self.config = config
        self.api = api

    @staticmethod
    def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        return config

    async def check_event_for_spam(self, event: EventBase) -> Union[bool, str]:
        return False

    async def user_may_invite(self, inviter_userid: str, invitee_userid: str, room_id: str) -> bool:
        return True

    async def user_may_create_room(self, userid: str) -> bool:
        return True

    async def user_may_create_room_alias(self, userid: str, room_alias: RoomAlias) -> bool:
        return True

    async def user_may_publish_room(self, userid: str, room_id: str) -> bool:
        return True

    async def check_media_file_for_spam(self, file_wrapper: ReadableFileWrapper, file_info: FileInfo) -> bool:
        buf = BytesIO()
        await file_wrapper.write_chunks_to(buf.write)
        return b'evil' in buf.getvalue()

class SpamCheckerTestCaseLegacy(unittest.HomeserverTestCase):
    servlets = [login.register_servlets, admin.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.user = self.register_user('user', 'pass')
        self.tok = self.login('user', 'pass')
        load_legacy_spam_checkers(hs)

    def create_resource_dict(self) -> Dict[str, Resource]:
        if False:
            for i in range(10):
                print('nop')
        resources = super().create_resource_dict()
        resources['/_matrix/media'] = self.hs.get_media_repository_resource()
        return resources

    def default_config(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        config = default_config('test')
        config.update({'spam_checker': [{'module': TestSpamCheckerLegacy.__module__ + '.TestSpamCheckerLegacy', 'config': {}}]})
        return config

    def test_upload_innocent(self) -> None:
        if False:
            i = 10
            return i + 15
        'Attempt to upload some innocent data that should be allowed.'
        self.helper.upload_media(SMALL_PNG, tok=self.tok, expect_code=200)

    def test_upload_ban(self) -> None:
        if False:
            while True:
                i = 10
        'Attempt to upload some data that includes bytes "evil", which should\n        get rejected by the spam checker.\n        '
        data = b'Some evil data'
        self.helper.upload_media(data, tok=self.tok, expect_code=400)
EVIL_DATA = b'Some evil data'
EVIL_DATA_EXPERIMENT = b'Some evil data to trigger the experimental tuple API'

class SpamCheckerTestCase(unittest.HomeserverTestCase):
    servlets = [login.register_servlets, admin.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user = self.register_user('user', 'pass')
        self.tok = self.login('user', 'pass')
        hs.get_module_api().register_spam_checker_callbacks(check_media_file_for_spam=self.check_media_file_for_spam)

    def create_resource_dict(self) -> Dict[str, Resource]:
        if False:
            return 10
        resources = super().create_resource_dict()
        resources['/_matrix/media'] = self.hs.get_media_repository_resource()
        return resources

    async def check_media_file_for_spam(self, file_wrapper: ReadableFileWrapper, file_info: FileInfo) -> Union[Codes, Literal['NOT_SPAM'], Tuple[Codes, JsonDict]]:
        buf = BytesIO()
        await file_wrapper.write_chunks_to(buf.write)
        if buf.getvalue() == EVIL_DATA:
            return Codes.FORBIDDEN
        elif buf.getvalue() == EVIL_DATA_EXPERIMENT:
            return (Codes.FORBIDDEN, {})
        else:
            return 'NOT_SPAM'

    def test_upload_innocent(self) -> None:
        if False:
            i = 10
            return i + 15
        'Attempt to upload some innocent data that should be allowed.'
        self.helper.upload_media(SMALL_PNG, tok=self.tok, expect_code=200)

    def test_upload_ban(self) -> None:
        if False:
            while True:
                i = 10
        'Attempt to upload some data that includes bytes "evil", which should\n        get rejected by the spam checker.\n        '
        self.helper.upload_media(EVIL_DATA, tok=self.tok, expect_code=400)
        self.helper.upload_media(EVIL_DATA_EXPERIMENT, tok=self.tok, expect_code=400)