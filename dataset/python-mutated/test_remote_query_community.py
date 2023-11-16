import asyncio
import random
import string
import time
from asyncio import sleep
from binascii import unhexlify
from operator import attrgetter
from os import urandom
from unittest.mock import Mock, patch
import pytest
from ipv8.keyvault.crypto import default_eccrypto
from pony.orm import db_session
from pony.orm.dbapiprovider import OperationalError
from tribler.core.components.ipv8.adapters_tests import TriblerTestBase
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import NEW
from tribler.core.components.metadata_store.db.serialization import CHANNEL_THUMBNAIL, CHANNEL_TORRENT, REGULAR_TORRENT
from tribler.core.components.metadata_store.db.store import MetadataStore
from tribler.core.components.metadata_store.remote_query_community.remote_query_community import RemoteQueryCommunity, sanitize_query
from tribler.core.components.metadata_store.remote_query_community.settings import RemoteQueryCommunitySettings
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.unicode import hexlify
from tribler.core.utilities.utilities import random_infohash

def random_string():
    if False:
        while True:
            i = 10
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=100))

def add_random_torrent(metadata_cls, name='test', channel=None, seeders=None, leechers=None, last_check=None):
    if False:
        for i in range(10):
            print('nop')
    d = {'infohash': random_infohash(), 'title': name, 'tags': '', 'size': 1234, 'status': NEW}
    if channel:
        d.update({'origin_id': channel.id_})
    torrent_metadata = metadata_cls.from_dict(d)
    torrent_metadata.sign()
    if seeders:
        torrent_metadata.health.seeders = seeders
    if leechers:
        torrent_metadata.health.leechers = leechers
    if last_check:
        torrent_metadata.health.last_check = last_check

class BasicRemoteQueryCommunity(RemoteQueryCommunity):
    community_id = unhexlify('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')

class TestRemoteQueryCommunity(TriblerTestBase):
    """
    Unit tests for the base RemoteQueryCommunity which do not need a real Session.
    """

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        random.seed(123)
        super().__init__(methodName)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(456)
        super().setUp()
        self.count = 0
        self.metadata_store_set = set()
        self.initialize(BasicRemoteQueryCommunity, 2)

    async def tearDown(self):
        for metadata_store in self.metadata_store_set:
            metadata_store.shutdown()
        await super().tearDown()

    def create_node(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        metadata_store = MetadataStore(Path(self.temporary_directory()) / f'{self.count}.db', Path(self.temporary_directory()), default_eccrypto.generate_key('curve25519'), disable_sync=True)
        self.metadata_store_set.add(metadata_store)
        kwargs['metadata_store'] = metadata_store
        kwargs['rqc_settings'] = RemoteQueryCommunitySettings()
        node = super().create_node(*args, **kwargs)
        self.count += 1
        return node

    def channel_metadata(self, i):
        if False:
            print('Hello World!')
        return self.nodes[i].overlay.mds.ChannelMetadata

    def torrent_metadata(self, i):
        if False:
            return 10
        return self.nodes[i].overlay.mds.TorrentMetadata

    async def test_remote_select(self):
        """
        Test querying metadata entries from a remote machine
        """
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        self.nodes[1].overlay.rqc_settings.max_channel_query_back = 0
        with db_session:
            channel = mds0.ChannelMetadata.create_channel('ubuntu channel', 'ubuntu')
            for i in range(20):
                add_random_torrent(mds0.TorrentMetadata, name=f'ubuntu {i}', channel=channel, seeders=2 * i, leechers=i, last_check=int(time.time()) + i)
        kwargs_dict = {'txt_filter': 'ubuntu*', 'metadata_type': [REGULAR_TORRENT]}
        callback = Mock()
        self.nodes[1].overlay.send_remote_select(self.nodes[0].my_peer, **kwargs_dict, processing_callback=callback)
        await self.deliver_messages(timeout=0.5)
        callback.assert_called()
        with db_session:
            torrents0 = sorted(mds0.get_entries(**kwargs_dict), key=attrgetter('infohash'))
            torrents1 = sorted(mds1.get_entries(**kwargs_dict), key=attrgetter('infohash'))
            self.assertEqual(len(torrents0), len(torrents1))
            self.assertEqual(len(torrents0), 20)
            for (t0, t1) in zip(torrents0, torrents1):
                assert t0.health.seeders == t1.health.seeders
                assert t0.health.leechers == t1.health.leechers
                assert t0.health.last_check == t1.health.last_check
        kwargs_dict = {'txt_filter': 'ubuntu*', 'origin_id': 352127}
        callback = Mock()
        self.nodes[1].overlay.send_remote_select(self.nodes[0].my_peer, **kwargs_dict, processing_callback=callback)
        await self.deliver_messages(timeout=0.5)
        callback.assert_called()

    async def test_remote_select_query_back(self):
        """
        Test querying back preview contents for previously unknown channels.
        """
        num_channels = 5
        max_received_torrents_per_channel_query_back = 4
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        with db_session:
            for _ in range(0, num_channels):
                chan = mds0.ChannelMetadata.create_channel('channel', '')
                for i in range(0, max_received_torrents_per_channel_query_back):
                    torrent = mds0.TorrentMetadata(origin_id=chan.id_, infohash=random_infohash())
                    torrent.health.seeders = i
        peer = self.nodes[0].my_peer
        kwargs_dict = {'metadata_type': [CHANNEL_TORRENT]}
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=0.5)
        with db_session:
            received_channels = list(mds1.ChannelMetadata.select(lambda g: g.title == 'channel'))
            assert len(received_channels) == num_channels
            received_torrents = list(mds1.TorrentMetadata.select(lambda g: g.metadata_type == REGULAR_TORRENT))
            assert num_channels * max_received_torrents_per_channel_query_back == len(received_torrents)
            seeders = {t.health.seeders for t in received_torrents}
            assert seeders == set(range(max_received_torrents_per_channel_query_back))

    async def test_push_back_entry_update(self):
        """
        Test pushing back update for an entry.
        Scenario: both hosts 0 and 1 have metadata entries for the same channel,
        but host 1's version was created later (its timestamp is higher).
        When host 1 queries -> host 0 for channel info, host 0 sends it back.
        Upon receiving the response, host 1 sees that it has a newer version of the channel entry,
        so it pushes it back to host 0.
        """
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        fake_key = default_eccrypto.generate_key('curve25519')
        with db_session:
            chan = mds0.ChannelMetadata(infohash=random_infohash(), title='foo', sign_with=fake_key)
            chan_payload_old = chan._payload_class.from_signed_blob(chan.serialized())
            chan.timestamp = chan.timestamp + 1
            chan.sign(key=fake_key)
            chan_payload_updated = chan._payload_class.from_signed_blob(chan.serialized())
            chan.delete()
            mds0.ChannelMetadata.from_payload(chan_payload_old)
            mds1.ChannelMetadata.from_payload(chan_payload_updated)
            assert mds0.ChannelMetadata.get(timestamp=chan_payload_old.timestamp)
        peer = self.nodes[0].my_peer
        kwargs_dict = {'metadata_type': [CHANNEL_TORRENT]}
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=0.5)
        with db_session:
            assert mds0.ChannelMetadata.get(timestamp=chan_payload_updated.timestamp)

    async def test_push_entry_update(self):
        """
        Test if sending back information on updated version of a metadata entry works
        """

    @pytest.mark.timeout(10)
    async def test_remote_select_torrents(self):
        """
        Test dropping packets that go over the response limit for a remote select.

        """
        peer = self.nodes[0].my_peer
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        with db_session:
            chan = mds0.ChannelMetadata.create_channel(random_string(), '')
            torrent_infohash = random_infohash()
            torrent = mds0.TorrentMetadata(origin_id=chan.id_, infohash=torrent_infohash, title='title1')
            torrent.sign()
        callback_called = asyncio.Event()
        processing_results = []

        def callback(_, results):
            if False:
                i = 10
                return i + 15
            processing_results.extend(results)
            callback_called.set()
        self.nodes[1].overlay.send_remote_select(peer, metadata_type=[REGULAR_TORRENT], infohash=torrent_infohash, processing_callback=callback)
        await callback_called.wait()
        assert len(processing_results) == 1
        obj = processing_results[0].md_obj
        assert isinstance(obj, mds1.TorrentMetadata)
        assert obj.title == 'title1'
        assert obj.health.seeders == 0
        with db_session:
            torrent = mds0.TorrentMetadata.get(infohash=torrent_infohash)
            torrent.timestamp += 1
            torrent.title = 'title2'
            torrent.sign()
        processing_results = []
        callback_called.clear()
        self.nodes[1].overlay.send_remote_select(peer, metadata_type=[REGULAR_TORRENT], infohash=torrent_infohash, processing_callback=callback)
        await callback_called.wait()
        assert len(processing_results) == 1
        obj = processing_results[0].md_obj
        assert isinstance(obj, mds1.TorrentMetadata)
        assert obj.health.seeders == 0

    async def test_remote_select_packets_limit(self):
        """
        Test dropping packets that go over the response limit for a remote select.

        """
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        self.nodes[1].overlay.rqc_settings.max_channel_query_back = 0
        with db_session:
            for _ in range(0, 100):
                mds0.ChannelMetadata.create_channel(random_string(), '')
        peer = self.nodes[0].my_peer
        kwargs_dict = {'metadata_type': [CHANNEL_TORRENT]}
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        self.assertTrue(self.nodes[1].overlay.request_cache._identifiers)
        await self.deliver_messages(timeout=1.5)
        with db_session:
            received_channels = list(mds1.ChannelMetadata.select())
            received_channels_count = len(received_channels)
            assert 40 < received_channels_count < 60
            self.assertFalse(self.nodes[1].overlay.request_cache._identifiers)

    def test_sanitize_query(self):
        if False:
            return 10
        req_response_list = [({'first': None, 'last': None}, {'first': 0, 'last': 100}), ({'first': 123, 'last': None}, {'first': 123, 'last': 223}), ({'first': None, 'last': 1000}, {'first': 0, 'last': 100}), ({'first': 100, 'last': None}, {'first': 100, 'last': 200}), ({'first': 123}, {'first': 123, 'last': 223}), ({'last': 123}, {'first': 0, 'last': 100}), ({}, {'first': 0, 'last': 100})]
        for (req, resp) in req_response_list:
            assert sanitize_query(req) == resp

    def test_sanitize_query_binary_fields(self):
        if False:
            while True:
                i = 10
        for field in ('infohash', 'channel_pk'):
            field_in_b = b'0' * 20
            field_in_hex = hexlify(field_in_b)
            assert sanitize_query({field: field_in_hex})[field] == field_in_b

    async def test_unknown_query_attribute(self):
        rqc_node1 = self.nodes[0].overlay
        rqc_node2 = self.nodes[1].overlay
        rqc_node2.send_remote_select(rqc_node1.my_peer, **{'new_attribute': 'some_value'})
        await self.deliver_messages(timeout=0.1)
        rqc_node2.send_remote_select(rqc_node1.my_peer, **{'infohash': b'0' * 20, 'foo': 'bar'})
        await self.deliver_messages(timeout=0.1)

    async def test_process_rpc_query_match_many(self):
        """
        Check if a correct query with a match in our database returns a result.
        """
        with db_session:
            channel = self.channel_metadata(0).create_channel('a channel', '')
            add_random_torrent(self.torrent_metadata(0), name='a torrent', channel=channel)
        results = await self.overlay(0).process_rpc_query({})
        self.assertEqual(2, len(results))
        (channel_md, torrent_md) = results if isinstance(results[0], self.channel_metadata(0)) else results[::-1]
        self.assertEqual('a channel', channel_md.title)
        self.assertEqual('a torrent', torrent_md.title)

    async def test_process_rpc_query_match_one(self):
        """
        Check if a correct query with one match in our database returns one result.
        """
        with db_session:
            self.channel_metadata(0).create_channel('a channel', '')
        results = await self.overlay(0).process_rpc_query({})
        self.assertEqual(1, len(results))
        (channel_md,) = results
        self.assertEqual('a channel', channel_md.title)

    async def test_process_rpc_query_match_none(self):
        """
        Check if a correct query with no match in our database returns no result.
        """
        results = await self.overlay(0).process_rpc_query({})
        self.assertEqual(0, len(results))

    def test_parse_parameters_match_empty_json(self):
        if False:
            print('Hello World!')
        '\n        Check if processing an empty request causes a ValueError (JSONDecodeError) to be raised.\n        '
        with self.assertRaises(ValueError):
            self.overlay(0).parse_parameters(b'')

    def test_parse_parameters_match_illegal_json(self):
        if False:
            while True:
                i = 10
        '\n        Check if processing a request with illegal JSON causes a UnicodeDecodeError to be raised.\n        '
        with self.assertRaises(UnicodeDecodeError):
            self.overlay(0).parse_parameters(b'{"akey":\x80}')

    async def test_process_rpc_query_match_invalid_json(self):
        """
        Check if processing a request with invalid JSON causes a ValueError to be raised.
        """
        with db_session:
            self.channel_metadata(0).create_channel('a channel', '')
        query = b'{"id_":' + b'1' * 200 + b'}'
        with self.assertRaises(ValueError):
            parameters = self.overlay(0).parse_parameters(query)
            await self.overlay(0).process_rpc_query(parameters)

    async def test_process_rpc_query_match_invalid_key(self):
        """
        Check if processing a request with invalid flags causes a UnicodeDecodeError to be raised.
        """
        with self.assertRaises(TypeError):
            parameters = self.overlay(0).parse_parameters(b'{"bla":":("}')
            await self.overlay(0).process_rpc_query(parameters)

    async def test_process_rpc_query_no_column(self):
        """
        Check if processing a request with no database columns causes an OperationalError.
        """
        with self.assertRaises(OperationalError):
            parameters = self.overlay(0).parse_parameters(b'{"txt_filter":{"key":"bla"}}')
            await self.overlay(0).process_rpc_query(parameters)

    async def test_remote_query_big_response(self):
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        value = urandom(20000)
        with db_session:
            mds1.ChannelThumbnail(binary_data=value)
        kwargs_dict = {'metadata_type': [CHANNEL_THUMBNAIL]}
        callback = Mock()
        self.nodes[0].overlay.send_remote_select(self.nodes[1].my_peer, **kwargs_dict, processing_callback=callback)
        await self.deliver_messages(timeout=0.5)
        callback.assert_called()
        with db_session:
            torrents0 = mds0.get_entries(**kwargs_dict)
            torrents1 = mds1.get_entries(**kwargs_dict)
            self.assertEqual(len(torrents0), len(torrents1))

    async def test_remote_select_query_back_thumbs_and_descriptions(self):
        """
        Test querying back preview thumbnail and description for previously unknown and updated channels.
        """
        mds0 = self.nodes[0].overlay.mds
        mds1 = self.nodes[1].overlay.mds
        with db_session:
            chan = mds0.ChannelMetadata.create_channel('channel', '')
            mds0.ChannelThumbnail(public_key=chan.public_key, origin_id=chan.id_, binary_data=urandom(2000), data_type='image/png', status=NEW)
            mds0.ChannelDescription(public_key=chan.public_key, origin_id=chan.id_, json_text='{"description_text": "foobar"}', status=NEW)
            chan.commit_all_channels()
            chan_v = chan.timestamp
        peer = self.nodes[0].my_peer
        kwargs_dict = {'metadata_type': [CHANNEL_TORRENT]}
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=0.5)
        with db_session:
            assert mds1.ChannelMetadata.get(lambda g: g.title == 'channel')
            assert mds1.ChannelThumbnail.get()
            assert mds1.ChannelDescription.get()
        with db_session:
            thumb = mds0.ChannelThumbnail.get()
            new_pic_bytes = urandom(2500)
            thumb.update_properties({'binary_data': new_pic_bytes})
            descr = mds0.ChannelDescription.get()
            descr.update_properties({'json_text': '{"description_text": "yummy"}'})
            chan = mds0.ChannelMetadata.get()
            chan.commit_all_channels()
            chan_v2 = chan.timestamp
            assert chan_v2 > chan_v
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=1)
        with db_session:
            assert mds1.ChannelMetadata.get(lambda g: g.title == 'channel')
            assert mds1.ChannelThumbnail.get().binary_data == new_pic_bytes
            assert mds1.ChannelDescription.get().json_text == '{"description_text": "yummy"}'
        with db_session:
            mds1.ChannelThumbnail.get().delete()
            mds1.ChannelDescription.get().delete()
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=1)
        with db_session:
            mds1.ChannelThumbnail.get()
            mds1.ChannelDescription.get()
        with db_session:
            chan = mds0.ChannelMetadata.get()
            mds0.TorrentMetadata(public_key=chan.public_key, origin_id=chan.id_, infohash=random_infohash(), status=NEW)
            chan.commit_all_channels()
            chan_v3 = chan.timestamp
            assert chan_v3 > chan_v2
        self.nodes[0].overlay.eva_send_binary = Mock()
        self.nodes[1].overlay.send_remote_select(peer, **kwargs_dict)
        await self.deliver_messages(timeout=1)
        self.nodes[0].overlay.eva_send_binary.assert_not_called()
        with db_session:
            assert mds1.ChannelMetadata.get(lambda g: g.title == 'channel').timestamp == chan_v3

    async def test_drop_silent_peer(self):
        self.nodes[1].overlay.rqc_settings.max_channel_query_back = 0
        kwargs_dict = {'txt_filter': 'ubuntu*'}
        basic_path = 'tribler.core.components.metadata_store.remote_query_community.remote_query_community'
        with self.overlay(1).request_cache.passthrough():
            with patch(basic_path + '.RemoteQueryCommunity._on_remote_select_basic'):
                self.nodes[1].overlay.network.remove_peer = Mock()
                self.nodes[1].overlay.send_remote_select(self.nodes[0].my_peer, **kwargs_dict)
                await sleep(0.0)
                self.nodes[1].overlay.network.remove_peer.assert_called()

    async def test_dont_drop_silent_peer_on_empty_response(self):
        self.nodes[1].overlay.rqc_settings.max_channel_query_back = 0
        was_called = []

        async def mock_on_remote_select_response(*_, **__):
            was_called.append(True)
            return []
        kwargs_dict = {'txt_filter': 'ubuntu*'}
        self.nodes[1].overlay.network.remove_peer = Mock()
        self.nodes[1].overlay.mds.process_compressed_mdblob_threaded = mock_on_remote_select_response
        self.nodes[1].overlay.send_remote_select(self.nodes[0].my_peer, **kwargs_dict)
        await self.deliver_messages()
        assert was_called
        self.nodes[1].overlay.network.remove_peer.assert_not_called()

    async def test_remote_select_force_eva(self):
        with db_session:
            for _ in range(0, 10):
                self.nodes[1].overlay.mds.ChannelThumbnail(binary_data=urandom(500))
        kwargs_dict = {'metadata_type': [CHANNEL_THUMBNAIL]}
        self.nodes[1].overlay.eva.send_binary = Mock()
        self.nodes[0].overlay.send_remote_select(self.nodes[1].my_peer, **kwargs_dict, force_eva_response=True)
        await self.deliver_messages(timeout=0.5)
        self.nodes[1].overlay.eva.send_binary.assert_called_once()

    async def test_multiple_parallel_request(self):
        peer_a = self.nodes[0].my_peer
        a = self.nodes[0].overlay
        b = self.nodes[1].overlay
        with db_session:
            add_random_torrent(a.mds.TorrentMetadata, name='foo')
            add_random_torrent(a.mds.TorrentMetadata, name='bar')
        callback1 = Mock()
        kwargs1 = {'txt_filter': 'foo', 'metadata_type': [REGULAR_TORRENT]}
        b.send_remote_select(peer_a, **kwargs1, processing_callback=callback1)
        callback2 = Mock()
        kwargs2 = {'txt_filter': 'bar', 'metadata_type': [REGULAR_TORRENT]}
        b.send_remote_select(peer_a, **kwargs2, processing_callback=callback2)
        original_get_entries = MetadataStore.get_entries

        def slow_get_entries(self, *args, **kwargs):
            if False:
                return 10
            time.sleep(0.1)
            return original_get_entries(self, *args, **kwargs)
        with patch.object(a, 'logger') as logger, patch.object(MetadataStore, 'get_entries', slow_get_entries):
            await self.deliver_messages(timeout=0.5)
        torrents1 = list(b.mds.get_entries(**kwargs1))
        torrents2 = list(b.mds.get_entries(**kwargs2))
        assert callback1.called and callback2.called
        assert bool(torrents1) != bool(torrents2)
        warnings = [call.args[0] for call in logger.warning.call_args_list]
        assert len([msg for msg in warnings if msg.startswith('Ignore remote query')]) == 1