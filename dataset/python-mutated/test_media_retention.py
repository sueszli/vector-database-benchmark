import io
from typing import Iterable, Optional
from matrix_common.types.mxc_uri import MXCUri
from twisted.test.proto_helpers import MemoryReactor
from synapse.rest import admin
from synapse.rest.client import login, register, room
from synapse.server import HomeServer
from synapse.types import UserID
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
from tests.utils import MockClock

class MediaRetentionTestCase(unittest.HomeserverTestCase):
    ONE_DAY_IN_MS = 24 * 60 * 60 * 1000
    THIRTY_DAYS_IN_MS = 30 * ONE_DAY_IN_MS
    servlets = [room.register_servlets, login.register_servlets, register.register_servlets, admin.register_servlets_for_client_rest_resource]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        return self.setup_test_homeserver(clock=MockClock())

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.remote_server_name = 'remote.homeserver'
        self.store = hs.get_datastores().main
        test_user_id = self.register_user('alice', 'password')
        media_repository = hs.get_media_repository()
        test_media_content = b'example string'

        def _create_media_and_set_attributes(last_accessed_ms: Optional[int], is_quarantined: Optional[bool]=False, is_protected: Optional[bool]=False) -> MXCUri:
            if False:
                i = 10
                return i + 15
            mxc_uri: MXCUri = self.get_success(media_repository.create_content(media_type='text/plain', upload_name=None, content=io.BytesIO(test_media_content), content_length=len(test_media_content), auth_user=UserID.from_string(test_user_id)))
            if last_accessed_ms is not None:
                self.get_success(self.store.update_cached_last_access_time(local_media=(mxc_uri.media_id,), remote_media=(), time_ms=last_accessed_ms))
            if is_quarantined:
                self.get_success(self.store.quarantine_media_by_id(server_name=self.hs.config.server.server_name, media_id=mxc_uri.media_id, quarantined_by='@theadmin:test'))
            if is_protected:
                self.get_success(self.store.mark_local_media_as_safe(media_id=mxc_uri.media_id, safe=True))
            return mxc_uri

        def _cache_remote_media_and_set_attributes(media_id: str, last_accessed_ms: Optional[int], is_quarantined: Optional[bool]=False) -> MXCUri:
            if False:
                i = 10
                return i + 15
            self.get_success(self.store.store_cached_remote_media(origin=self.remote_server_name, media_id=media_id, media_type='text/plain', media_length=1, time_now_ms=clock.time_msec(), upload_name='testfile.txt', filesystem_id='abcdefg12345'))
            if last_accessed_ms is not None:
                self.get_success(hs.get_datastores().main.update_cached_last_access_time(local_media=(), remote_media=((self.remote_server_name, media_id),), time_ms=last_accessed_ms))
            if is_quarantined:
                self.get_success(self.store.quarantine_media_by_id(server_name=self.remote_server_name, media_id=media_id, quarantined_by='@theadmin:test'))
            return MXCUri(self.remote_server_name, media_id)
        self.local_recently_accessed_media = _create_media_and_set_attributes(last_accessed_ms=self.THIRTY_DAYS_IN_MS)
        self.local_not_recently_accessed_media = _create_media_and_set_attributes(last_accessed_ms=self.ONE_DAY_IN_MS)
        self.local_not_recently_accessed_quarantined_media = _create_media_and_set_attributes(last_accessed_ms=self.ONE_DAY_IN_MS, is_quarantined=True)
        self.local_not_recently_accessed_protected_media = _create_media_and_set_attributes(last_accessed_ms=self.ONE_DAY_IN_MS, is_protected=True)
        self.local_never_accessed_media = _create_media_and_set_attributes(last_accessed_ms=None)
        self.remote_recently_accessed_media = _cache_remote_media_and_set_attributes(media_id='a', last_accessed_ms=self.THIRTY_DAYS_IN_MS)
        self.remote_not_recently_accessed_media = _cache_remote_media_and_set_attributes(media_id='b', last_accessed_ms=self.ONE_DAY_IN_MS)
        self.remote_not_recently_accessed_quarantined_media = _cache_remote_media_and_set_attributes(media_id='c', last_accessed_ms=self.ONE_DAY_IN_MS, is_quarantined=True)

    @override_config({'media_retention': {'local_media_lifetime': '30d'}})
    def test_local_media_retention(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that local media that have not been accessed recently is purged, while\n        cached remote media is unaffected.\n        '
        self.reactor.advance(31 * 24 * 60 * 60)
        self._assert_if_mxc_uris_purged(purged=[self.local_not_recently_accessed_media, self.local_never_accessed_media], not_purged=[self.local_recently_accessed_media, self.local_not_recently_accessed_quarantined_media, self.local_not_recently_accessed_protected_media, self.remote_recently_accessed_media, self.remote_not_recently_accessed_media, self.remote_not_recently_accessed_quarantined_media])

    @override_config({'media_retention': {'remote_media_lifetime': '30d'}})
    def test_remote_media_cache_retention(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that entries from the remote media cache that have not been accessed\n        recently is purged, while local media is unaffected.\n        '
        self.reactor.advance(31 * 24 * 60 * 60)
        self._assert_if_mxc_uris_purged(purged=[self.remote_not_recently_accessed_media], not_purged=[self.remote_recently_accessed_media, self.local_recently_accessed_media, self.local_not_recently_accessed_media, self.local_not_recently_accessed_quarantined_media, self.local_not_recently_accessed_protected_media, self.remote_not_recently_accessed_quarantined_media, self.local_never_accessed_media])

    def _assert_if_mxc_uris_purged(self, purged: Iterable[MXCUri], not_purged: Iterable[MXCUri]) -> None:
        if False:
            i = 10
            return i + 15

        def _assert_mxc_uri_purge_state(mxc_uri: MXCUri, expect_purged: bool) -> None:
            if False:
                return 10
            'Given an MXC URI, assert whether it has been purged or not.'
            if mxc_uri.server_name == self.hs.config.server.server_name:
                found_media = bool(self.get_success(self.store.get_local_media(mxc_uri.media_id)))
            else:
                found_media = bool(self.get_success(self.store.get_cached_remote_media(mxc_uri.server_name, mxc_uri.media_id)))
            if expect_purged:
                self.assertFalse(found_media, msg=f'{mxc_uri} unexpectedly not purged')
            else:
                self.assertTrue(found_media, msg=f'{mxc_uri} unexpectedly purged')
        for mxc_uri in purged:
            _assert_mxc_uri_purge_state(mxc_uri, expect_purged=True)
        for mxc_uri in not_purged:
            _assert_mxc_uri_purge_state(mxc_uri, expect_purged=False)