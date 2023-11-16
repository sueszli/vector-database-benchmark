import json
import struct
import time
from asyncio import Future
from binascii import unhexlify
from itertools import count
from typing import Any, Dict, List, Optional, Set
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.lazy_payload import VariablePayload, vp_compile
from ipv8.requestcache import NumberCache, RandomNumberCache, RequestCache
from pony.orm import db_session
from pony.orm.dbapiprovider import OperationalError
from tribler.core.components.database.db.layers.knowledge_data_access_layer import ResourceType
from tribler.core.components.ipv8.eva.protocol import EVAProtocol
from tribler.core.components.ipv8.eva.result import TransferResult
from tribler.core.components.ipv8.tribler_community import TriblerCommunity
from tribler.core.components.knowledge.community.knowledge_validator import is_valid_resource
from tribler.core.components.metadata_store.db.orm_bindings.channel_metadata import LZ4_EMPTY_ARCHIVE, entries_to_chunk
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, COLLECTION_NODE, REGULAR_TORRENT
from tribler.core.components.metadata_store.db.store import MetadataStore
from tribler.core.components.metadata_store.remote_query_community.payload_checker import ObjState
from tribler.core.components.metadata_store.remote_query_community.settings import RemoteQueryCommunitySettings
from tribler.core.components.metadata_store.utils import RequestTimeoutException
from tribler.core.utilities.pony_utils import run_threaded
from tribler.core.utilities.unicode import hexlify
BINARY_FIELDS = ('infohash', 'channel_pk')

def sanitize_query(query_dict: Dict[str, Any], cap=100) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    sanitized_dict = dict(query_dict)
    first = sanitized_dict.get('first', None)
    last = sanitized_dict.get('last', None)
    first = first or 0
    last = last if last is not None and last <= first + cap else first + cap
    sanitized_dict.update({'first': first, 'last': last})
    for field in BINARY_FIELDS:
        value = sanitized_dict.get(field)
        if value is not None:
            sanitized_dict[field] = unhexlify(value)
    return sanitized_dict

def convert_to_json(parameters):
    if False:
        print('Hello World!')
    sanitized = dict(parameters)
    if 'metadata_type' in sanitized:
        sanitized['metadata_type'] = [int(mt) for mt in sanitized['metadata_type'] if mt]
    for field in BINARY_FIELDS:
        value = parameters.get(field)
        if value is not None:
            sanitized[field] = hexlify(value)
    if 'origin_id' in parameters:
        sanitized['origin_id'] = int(parameters['origin_id'])
    return json.dumps(sanitized)

@vp_compile
class RemoteSelectPayload(VariablePayload):
    msg_id = 201
    format_list = ['I', 'varlenH']
    names = ['id', 'json']

@vp_compile
class RemoteSelectPayloadEva(RemoteSelectPayload):
    msg_id = 209

@vp_compile
class SelectResponsePayload(VariablePayload):
    msg_id = 202
    format_list = ['I', 'raw']
    names = ['id', 'raw_blob']

class SelectRequest(RandomNumberCache):

    def __init__(self, request_cache, prefix, request_kwargs, peer, processing_callback=None, timeout_callback=None):
        if False:
            i = 10
            return i + 15
        super().__init__(request_cache, prefix)
        self.request_kwargs = request_kwargs
        self.processing_callback = processing_callback
        self.packets_limit = 10
        self.peer = peer
        self.peer_responded = False
        self.timeout_callback = timeout_callback

    def on_timeout(self):
        if False:
            i = 10
            return i + 15
        if self.timeout_callback is not None:
            self.timeout_callback(self)

class EvaSelectRequest(SelectRequest):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.packets_limit = 1
        self.processing_results = Future()
        self.register_future(self.processing_results, on_timeout=RequestTimeoutException())

class PushbackWindow(NumberCache):

    def __init__(self, request_cache, prefix, original_request_id):
        if False:
            print('Hello World!')
        super().__init__(request_cache, prefix, original_request_id)
        self.packets_limit = 10

    def on_timeout(self):
        if False:
            print('Hello World!')
        pass

class RemoteQueryCommunity(TriblerCommunity):
    """
    Community for general purpose SELECT-like queries into remote Channels database
    """

    def __init__(self, my_peer, endpoint, network, rqc_settings: RemoteQueryCommunitySettings=None, metadata_store=None, tribler_db=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(my_peer, endpoint, network=network, **kwargs)
        self.rqc_settings = rqc_settings
        self.mds: MetadataStore = metadata_store
        self.tribler_db = tribler_db
        self.request_cache = RequestCache()
        self.add_message_handler(RemoteSelectPayload, self.on_remote_select)
        self.add_message_handler(RemoteSelectPayloadEva, self.on_remote_select_eva)
        self.add_message_handler(SelectResponsePayload, self.on_remote_select_response)
        self.eva = EVAProtocol(self, self.on_receive, self.on_send_complete, self.on_error)
        self.remote_queries_in_progress = 0
        self.next_remote_query_num = count().__next__

    async def on_receive(self, result: TransferResult):
        self.logger.debug(f'EVA data received: peer {hexlify(result.peer.mid)}, info {result.info}')
        packet = (result.peer.address, result.data)
        self.on_packet(packet)

    async def on_send_complete(self, result: TransferResult):
        self.logger.debug(f'EVA outgoing transfer complete: peer {hexlify(result.peer.mid)},  info {result.info}')

    async def on_error(self, peer, exception):
        self.logger.warning(f'EVA transfer error:{exception.__class__.__name__}:{exception}, Peer: {hexlify(peer.mid)}')

    def send_remote_select(self, peer, processing_callback=None, force_eva_response=False, **kwargs):
        if False:
            return 10
        request_class = EvaSelectRequest if force_eva_response else SelectRequest
        request = request_class(self.request_cache, hexlify(peer.mid), kwargs, peer, processing_callback=processing_callback, timeout_callback=self._on_query_timeout)
        self.request_cache.add(request)
        self.logger.debug(f'Select to {hexlify(peer.mid)} with ({kwargs})')
        args = (request.number, convert_to_json(kwargs).encode('utf8'))
        if force_eva_response:
            self.ez_send(peer, RemoteSelectPayloadEva(*args))
        else:
            self.ez_send(peer, RemoteSelectPayload(*args))
        return request

    def should_limit_rate_for_query(self, sanitized_parameters: Dict[str, Any]) -> bool:
        if False:
            while True:
                i = 10
        return 'txt_filter' in sanitized_parameters

    async def process_rpc_query_rate_limited(self, sanitized_parameters: Dict[str, Any]) -> List:
        query_num = self.next_remote_query_num()
        if self.remote_queries_in_progress and self.should_limit_rate_for_query(sanitized_parameters):
            self.logger.warning(f'Ignore remote query {query_num} as another one is already processing. The ignored query: {sanitized_parameters}')
            return []
        self.logger.info(f'Process remote query {query_num}: {sanitized_parameters}')
        self.remote_queries_in_progress += 1
        t = time.time()
        try:
            return await self.process_rpc_query(sanitized_parameters)
        finally:
            self.remote_queries_in_progress -= 1
            self.logger.info(f'Remote query {query_num} processed in {time.time() - t} seconds: {sanitized_parameters}')

    async def process_rpc_query(self, sanitized_parameters: Dict[str, Any]) -> List:
        """
        Retrieve the result of a database query from a third party, encoded as raw JSON bytes (through `dumps`).
        :raises TypeError: if the JSON contains invalid keys.
        :raises ValueError: if no JSON could be decoded.
        :raises pony.orm.dbapiprovider.OperationalError: if an illegal query was performed.
        """
        if self.tribler_db:
            tags = sanitized_parameters.pop('tags', None)
            infohash_set = await run_threaded(self.tribler_db.instance, self.search_for_tags, tags)
            if infohash_set:
                sanitized_parameters['infohash_set'] = {bytes.fromhex(s) for s in infohash_set}
        return await self.mds.get_entries_threaded(**sanitized_parameters)

    @db_session
    def search_for_tags(self, tags: Optional[List[str]]) -> Optional[Set[str]]:
        if False:
            print('Hello World!')
        if not tags or not self.tribler_db:
            return None
        valid_tags = {tag for tag in tags if is_valid_resource(tag)}
        result = self.tribler_db.knowledge.get_subjects_intersection(subjects_type=ResourceType.TORRENT, objects=valid_tags, predicate=ResourceType.TAG, case_sensitive=False)
        return result

    def send_db_results(self, peer, request_payload_id, db_results, force_eva_response=False):
        if False:
            for i in range(10):
                print('nop')
        if len(db_results) == 0:
            self.ez_send(peer, SelectResponsePayload(request_payload_id, LZ4_EMPTY_ARCHIVE))
            return
        index = 0
        while index < len(db_results):
            transfer_size = self.eva.settings.binary_size_limit if force_eva_response else self.rqc_settings.maximum_payload_size
            (data, index) = entries_to_chunk(db_results, transfer_size, start_index=index, include_health=True)
            payload = SelectResponsePayload(request_payload_id, data)
            if force_eva_response or len(data) > self.rqc_settings.maximum_payload_size:
                self.eva.send_binary(peer, struct.pack('>i', request_payload_id), self.ezr_pack(payload.msg_id, payload))
            else:
                self.ez_send(peer, payload)

    @lazy_wrapper(RemoteSelectPayloadEva)
    async def on_remote_select_eva(self, peer, request_payload):
        await self._on_remote_select_basic(peer, request_payload, force_eva_response=True)

    @lazy_wrapper(RemoteSelectPayload)
    async def on_remote_select(self, peer, request_payload):
        await self._on_remote_select_basic(peer, request_payload)

    def parse_parameters(self, json_bytes: bytes) -> Dict[str, Any]:
        if False:
            return 10
        parameters = json.loads(json_bytes)
        return sanitize_query(parameters, self.rqc_settings.max_response_size)

    async def _on_remote_select_basic(self, peer, request_payload, force_eva_response=False):
        try:
            sanitized_parameters = self.parse_parameters(request_payload.json)
            db_results = await self.process_rpc_query_rate_limited(sanitized_parameters)
            if db_results and (not self.request_cache.has(hexlify(peer.mid), request_payload.id)):
                self.request_cache.add(PushbackWindow(self.request_cache, hexlify(peer.mid), request_payload.id))
            self.send_db_results(peer, request_payload.id, db_results, force_eva_response)
        except (OperationalError, TypeError, ValueError) as error:
            self.logger.error(f'Remote select. The error occurred: {error}')

    @lazy_wrapper(SelectResponsePayload)
    async def on_remote_select_response(self, peer, response_payload):
        """
        Match the the response that we received from the network to a query cache
        and process it by adding the corresponding entries to the MetadataStore database.
        This processes both direct responses and pushback (updates) responses
        """
        self.logger.debug(f'Response from {hexlify(peer.mid)}')
        request = self.request_cache.get(hexlify(peer.mid), response_payload.id)
        if request is None:
            return
        if request.packets_limit > 1:
            request.packets_limit -= 1
        else:
            self.request_cache.pop(hexlify(peer.mid), response_payload.id)
        processing_results = await self.mds.process_compressed_mdblob_threaded(response_payload.raw_blob)
        self.logger.debug(f'Response result: {processing_results}')
        if isinstance(request, EvaSelectRequest) and (not request.processing_results.done()):
            request.processing_results.set_result(processing_results)
        if isinstance(request, SelectRequest) and self.rqc_settings.push_updates_back_enabled:
            newer_entities = [r.md_obj for r in processing_results if r.obj_state == ObjState.LOCAL_VERSION_NEWER]
            self.send_db_results(peer, response_payload.id, newer_entities)
        if self.rqc_settings.channel_query_back_enabled:
            for result in processing_results:
                if result.obj_state == ObjState.NEW_OBJECT and result.md_obj.metadata_type in (CHANNEL_TORRENT, COLLECTION_NODE):
                    request_dict = {'metadata_type': [COLLECTION_NODE, REGULAR_TORRENT], 'channel_pk': result.md_obj.public_key, 'origin_id': result.md_obj.id_, 'first': 0, 'last': self.rqc_settings.max_channel_query_back}
                    self.send_remote_select(peer=peer, **request_dict)
                for dep_query_dict in result.missing_deps:
                    self.send_remote_select(peer=peer, **dep_query_dict)
        if isinstance(request, SelectRequest) and request.processing_callback:
            request.processing_callback(request, processing_results)
        if isinstance(request, SelectRequest):
            request.peer_responded = True

    def _on_query_timeout(self, request_cache):
        if False:
            while True:
                i = 10
        if not request_cache.peer_responded:
            self.logger.debug('Remote query timeout, deleting peer: %s %s %s', str(request_cache.peer.address), hexlify(request_cache.peer.mid), str(request_cache.request_kwargs))
            self.network.remove_peer(request_cache.peer)

    async def unload(self):
        await self.eva.shutdown()
        await self.request_cache.shutdown()
        await super().unload()