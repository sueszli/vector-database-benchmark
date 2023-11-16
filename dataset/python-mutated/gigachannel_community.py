import time
import uuid
from binascii import unhexlify
from collections import defaultdict
from dataclasses import dataclass
from random import sample
from anyio import Event, create_task_group, move_on_after
from ipv8.types import Peer
from pony.orm import db_session
from tribler.core import notifications
from tribler.core.components.ipv8.discovery_booster import DiscoveryBooster
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT
from tribler.core.components.metadata_store.remote_query_community.payload_checker import ObjState
from tribler.core.components.metadata_store.remote_query_community.remote_query_community import RemoteQueryCommunity
from tribler.core.components.metadata_store.utils import NoChannelSourcesException
from tribler.core.utilities.notifier import Notifier
from tribler.core.utilities.simpledefs import CHANNELS_VIEW_UUID
from tribler.core.utilities.unicode import hexlify
minimal_blob_size = 200
maximum_payload_size = 1024
max_entries = maximum_payload_size // minimal_blob_size
max_search_peers = 5
happy_eyeballs_delay = 0.3
max_address_cache_lifetime = 5.0

@dataclass
class ChannelEntry:
    timestamp: float
    channel_version: int

class ChannelsPeersMapping:

    def __init__(self, max_peers_per_channel=10):
        if False:
            i = 10
            return i + 15
        self.max_peers_per_channel = max_peers_per_channel
        self._channels_dict = defaultdict(set)
        self._peers_channels = defaultdict(set)

    def add(self, peer: Peer, channel_pk: bytes, channel_id: int):
        if False:
            print('Hello World!')
        id_tuple = (channel_pk, channel_id)
        channel_peers = self._channels_dict[id_tuple]
        channel_peers.add(peer)
        self._peers_channels[peer].add(id_tuple)
        if len(channel_peers) > self.max_peers_per_channel:
            removed_peer = min(channel_peers, key=lambda x: x.last_response)
            channel_peers.remove(removed_peer)
            self._peers_channels[removed_peer].remove(id_tuple)
            if not self._peers_channels[removed_peer]:
                self._peers_channels.pop(removed_peer)

    def remove_peer(self, peer):
        if False:
            return 10
        for id_tuple in self._peers_channels[peer]:
            self._channels_dict[id_tuple].discard(peer)
            if not self._channels_dict[id_tuple]:
                self._channels_dict.pop(id_tuple)
        self._peers_channels.pop(peer)

    def get_last_seen_peers_for_channel(self, channel_pk: bytes, channel_id: int, limit=None):
        if False:
            for i in range(10):
                print('nop')
        id_tuple = (channel_pk, channel_id)
        channel_peers = self._channels_dict.get(id_tuple, [])
        return sorted(channel_peers, key=lambda x: x.last_response, reverse=True)[0:limit]

class GigaChannelCommunity(RemoteQueryCommunity):
    community_id = unhexlify('d3512d0ff816d8ac672eab29a9c1a3a32e17cb13')

    def create_introduction_response(self, lan_socket_address, socket_address, identifier, introduction=None, extra_bytes=b'', prefix=None, new_style=False):
        if False:
            print('Hello World!')
        return super().create_introduction_response(lan_socket_address, socket_address, identifier, introduction=introduction, prefix=prefix, new_style=new_style)

    def __init__(self, *args, notifier: Notifier=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.notifier = notifier
        self.queried_peers = set()
        self.address_cache = {}
        self.address_cache_created_at = time.time()
        self.discovery_booster = DiscoveryBooster()
        self.discovery_booster.apply(self)
        self.channels_peers = ChannelsPeersMapping()

    def guess_address(self, interface):
        if False:
            i = 10
            return i + 15
        now = time.time()
        cache_lifetime = now - self.address_cache_created_at
        if cache_lifetime > max_address_cache_lifetime:
            self.address_cache.clear()
            self.address_cache_created_at = now
        result = self.address_cache.get(interface)
        if result is not None:
            return result
        result = super().guess_address(interface)
        self.address_cache[interface] = result
        return result

    def get_random_peers(self, sample_size=None):
        if False:
            for i in range(10):
                print('nop')
        all_peers = self.get_peers()
        if sample_size is not None and sample_size < len(all_peers):
            return sample(all_peers, sample_size)
        return all_peers

    def introduction_response_callback(self, peer, dist, payload):
        if False:
            i = 10
            return i + 15
        if peer.address in self.network.blacklist or peer.mid in self.queried_peers or peer.mid in self.network.blacklist_mids:
            return
        if len(self.queried_peers) >= self.settings.queried_peers_limit:
            self.queried_peers.clear()
        self.queried_peers.add(peer.mid)
        self.send_remote_select_subscribed_channels(peer)

    def send_remote_select_subscribed_channels(self, peer):
        if False:
            print('Hello World!')

        def on_packet_callback(_, processing_results):
            if False:
                i = 10
                return i + 15
            with db_session:
                for c in (r.md_obj for r in processing_results if r.md_obj.metadata_type == CHANNEL_TORRENT):
                    self.mds.vote_bump(c.public_key, c.id_, peer.public_key.key_to_bin()[10:])
                    self.channels_peers.add(peer, c.public_key, c.id_)
            results = [r.md_obj.to_simple_dict() for r in processing_results if r.obj_state == ObjState.NEW_OBJECT and r.md_obj.metadata_type == CHANNEL_TORRENT and (r.md_obj.origin_id == 0)]
            if self.notifier and results:
                self.notifier[notifications.channel_discovered]({'results': results, 'uuid': str(CHANNELS_VIEW_UUID)})
        request_dict = {'metadata_type': [CHANNEL_TORRENT], 'subscribed': True, 'attribute_ranges': (('num_entries', 1, None),), 'complete_channel': True}
        self.send_remote_select(peer, **request_dict, processing_callback=on_packet_callback)

    async def remote_select_channel_contents(self, **kwargs):
        peers_to_query = self.get_known_subscribed_peers_for_node(kwargs['channel_pk'], kwargs['origin_id'])
        if not peers_to_query:
            raise NoChannelSourcesException()
        result = []
        async with create_task_group() as tg:
            got_at_least_one_response = Event()

            async def _send_remote_select(peer):
                request = self.send_remote_select(peer, force_eva_response=True, **kwargs)
                await request.processing_results
                if result or got_at_least_one_response.is_set():
                    return
                result.extend(request.processing_results.result())
                got_at_least_one_response.set()
            for peer in peers_to_query:
                if got_at_least_one_response.is_set():
                    break
                tg.start_soon(_send_remote_select, peer)
                with move_on_after(happy_eyeballs_delay):
                    await got_at_least_one_response.wait()
            await got_at_least_one_response.wait()
            tg.cancel_scope.cancel()
        request_results = [r.md_obj.to_simple_dict() for r in result]
        return request_results

    def send_search_request(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        request_uuid = uuid.uuid4()

        def notify_gui(request, processing_results):
            if False:
                print('Hello World!')
            results = [r.md_obj.to_simple_dict() for r in processing_results if r.obj_state in (ObjState.NEW_OBJECT, ObjState.UPDATED_LOCAL_VERSION)]
            if self.notifier:
                self.notifier[notifications.remote_query_results]({'results': results, 'uuid': str(request_uuid), 'peer': hexlify(request.peer.mid)})
        if 'channel_pk' in kwargs and 'origin_id' in kwargs:
            peers_to_query = self.get_known_subscribed_peers_for_node(kwargs['channel_pk'], kwargs['origin_id'], self.settings.max_mapped_query_peers)
        else:
            peers_to_query = self.get_random_peers(self.rqc_settings.max_query_peers)
        for p in peers_to_query:
            self.send_remote_select(p, **kwargs, processing_callback=notify_gui)
        return (request_uuid, peers_to_query)

    def get_known_subscribed_peers_for_node(self, node_pk, node_id, limit=None):
        if False:
            return 10
        root_id = node_id
        with db_session:
            node = self.mds.ChannelNode.get(public_key=node_pk, id_=node_id)
            if node:
                root_id = next((node.id_ for node in node.get_parent_nodes() if node.origin_id == 0), node.origin_id)
        return self.channels_peers.get_last_seen_peers_for_channel(node_pk, root_id, limit)

    def _on_query_timeout(self, request_cache):
        if False:
            print('Hello World!')
        if not request_cache.peer_responded:
            self.channels_peers.remove_peer(request_cache.peer)
        super()._on_query_timeout(request_cache)

class GigaChannelTestnetCommunity(GigaChannelCommunity):
    """
    This community defines a testnet for the giga channels, used for testing purposes.
    """
    community_id = unhexlify('ad8cece0dfdb0e03344b59a4d31a38fe9812da9d')