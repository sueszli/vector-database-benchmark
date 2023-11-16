import zlib
from collections import OrderedDict, defaultdict
import asyncio
import os
import time
from typing import Tuple, Dict, TYPE_CHECKING, Optional, Union, Set, Callable
from datetime import datetime
import functools
import aiorpcx
from aiorpcx import ignore_after
from .crypto import sha256, sha256d
from . import bitcoin, util
from . import ecc
from .ecc import sig_string_from_r_and_s, der_sig_from_sig_string
from . import constants
from .util import bfh, log_exceptions, ignore_exceptions, chunks, OldTaskGroup, UnrelatedTransactionException, error_text_bytes_to_safe_str
from . import transaction
from .bitcoin import make_op_return, DummyAddress
from .transaction import PartialTxOutput, match_script_against_template, Sighash
from .logging import Logger
from .lnrouter import RouteEdge
from .lnonion import new_onion_packet, OnionFailureCode, calc_hops_data_for_payment, process_onion_packet, OnionPacket, construct_onion_error, obfuscate_onion_error, OnionRoutingFailure, ProcessedOnionPacket, UnsupportedOnionPacketVersion, InvalidOnionMac, InvalidOnionPubkey, OnionFailureCodeMetaFlag
from .lnchannel import Channel, RevokeAndAck, RemoteCtnTooFarInFuture, ChannelState, PeerState, ChanCloseOption, CF_ANNOUNCE_CHANNEL
from . import lnutil
from .lnutil import Outpoint, LocalConfig, RECEIVED, UpdateAddHtlc, ChannelConfig, RemoteConfig, OnlyPubkeyKeypair, ChannelConstraints, RevocationStore, funding_output_script, get_per_commitment_secret_from_seed, secret_to_pubkey, PaymentFailure, LnFeatures, LOCAL, REMOTE, HTLCOwner, ln_compare_features, privkey_to_pubkey, MIN_FINAL_CLTV_DELTA_ACCEPTED, LightningPeerConnectionClosed, HandshakeFailed, RemoteMisbehaving, ShortChannelID, IncompatibleLightningFeatures, derive_payment_secret_from_payment_preimage, ChannelType, LNProtocolWarning, validate_features, IncompatibleOrInsaneFeatures
from .lnutil import FeeUpdate, channel_id_from_funding_tx, PaymentFeeBudget
from .lntransport import LNTransport, LNTransportBase
from .lnmsg import encode_msg, decode_msg, UnknownOptionalMsgType, FailedToParseMsg
from .interface import GracefulDisconnect
from .lnrouter import fee_for_edge_msat
from .json_db import StoredDict
from .invoices import PR_PAID
from .simple_config import FEE_LN_ETA_TARGET
from .trampoline import decode_routing_info
if TYPE_CHECKING:
    from .lnworker import LNGossip, LNWallet
    from .lnrouter import LNPaymentRoute
    from .transaction import PartialTransaction
LN_P2P_NETWORK_TIMEOUT = 20

class Peer(Logger):
    LOGGING_SHORTCUT = 'P'
    ORDERED_MESSAGES = ('accept_channel', 'funding_signed', 'funding_created', 'accept_channel', 'closing_signed')
    SPAMMY_MESSAGES = ('ping', 'pong', 'channel_announcement', 'node_announcement', 'channel_update')
    DELAY_INC_MSG_PROCESSING_SLEEP = 0.01

    def __init__(self, lnworker: Union['LNGossip', 'LNWallet'], pubkey: bytes, transport: LNTransportBase, *, is_channel_backup=False):
        if False:
            return 10
        self.lnworker = lnworker
        self.network = lnworker.network
        self.asyncio_loop = self.network.asyncio_loop
        self.is_channel_backup = is_channel_backup
        self._sent_init = False
        self._received_init = False
        self.initialized = self.asyncio_loop.create_future()
        self.got_disconnected = asyncio.Event()
        self.querying = asyncio.Event()
        self.transport = transport
        self.pubkey = pubkey
        self.privkey = self.transport.privkey
        self.features = self.lnworker.features
        self.their_features = LnFeatures(0)
        self.node_ids = [self.pubkey, privkey_to_pubkey(self.privkey)]
        assert self.node_ids[0] != self.node_ids[1]
        self.last_message_time = 0
        self.pong_event = asyncio.Event()
        self.reply_channel_range = asyncio.Queue()
        self.gossip_queue = asyncio.Queue()
        self.ordered_message_queues = defaultdict(asyncio.Queue)
        self.temp_id_to_id = {}
        self.funding_created_sent = set()
        self.funding_signed_sent = set()
        self.shutdown_received = {}
        self.channel_reestablish_msg = defaultdict(self.asyncio_loop.create_future)
        self.orphan_channel_updates = OrderedDict()
        Logger.__init__(self)
        self.taskgroup = OldTaskGroup()
        self.received_htlcs_pending_removal = set()
        self.received_htlc_removed_event = asyncio.Event()
        self._htlc_switch_iterstart_event = asyncio.Event()
        self._htlc_switch_iterdone_event = asyncio.Event()
        self._received_revack_event = asyncio.Event()
        self.received_commitsig_event = asyncio.Event()
        self.downstream_htlc_resolved_event = asyncio.Event()
        self.jit_failures = {}

    def send_message(self, message_name: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert util.get_running_loop() == util.get_asyncio_loop(), f'this must be run on the asyncio thread!'
        assert type(message_name) is str
        if message_name not in self.SPAMMY_MESSAGES:
            self.logger.debug(f'Sending {message_name.upper()}')
        if message_name.upper() != 'INIT' and (not self.is_initialized()):
            raise Exception('tried to send message before we are initialized')
        raw_msg = encode_msg(message_name, **kwargs)
        self._store_raw_msg_if_local_update(raw_msg, message_name=message_name, channel_id=kwargs.get('channel_id'))
        self.transport.send_bytes(raw_msg)

    def _store_raw_msg_if_local_update(self, raw_msg: bytes, *, message_name: str, channel_id: Optional[bytes]):
        if False:
            while True:
                i = 10
        is_commitment_signed = message_name == 'commitment_signed'
        if not (message_name.startswith('update_') or is_commitment_signed):
            return
        assert channel_id
        chan = self.get_channel_by_id(channel_id)
        if not chan:
            raise Exception(f'channel {channel_id.hex()} not found for peer {self.pubkey.hex()}')
        chan.hm.store_local_update_raw_msg(raw_msg, is_commitment_signed=is_commitment_signed)
        if is_commitment_signed:
            self.lnworker.save_channel(chan)

    def maybe_set_initialized(self):
        if False:
            for i in range(10):
                print('nop')
        if self.initialized.done():
            return
        if self._sent_init and self._received_init:
            self.initialized.set_result(True)

    def is_initialized(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.initialized.done() and (not self.initialized.cancelled()) and (self.initialized.exception() is None) and (self.initialized.result() is True)

    async def initialize(self):
        if isinstance(self.transport, LNTransport):
            await self.transport.handshake()
        self.logger.info(f'handshake done for {self.transport.peer_addr or self.pubkey.hex()}')
        features = self.features.for_init_message()
        b = int.bit_length(features)
        flen = b // 8 + int(bool(b % 8))
        self.send_message('init', gflen=0, flen=flen, features=features, init_tlvs={'networks': {'chains': constants.net.rev_genesis_bytes()}})
        self._sent_init = True
        self.maybe_set_initialized()

    @property
    def channels(self) -> Dict[bytes, Channel]:
        if False:
            return 10
        return self.lnworker.channels_for_peer(self.pubkey)

    def get_channel_by_id(self, channel_id: bytes) -> Optional[Channel]:
        if False:
            while True:
                i = 10
        chan = self.lnworker.get_channel_by_id(channel_id)
        if not chan:
            return None
        if chan.node_id != self.pubkey:
            return None
        return chan

    def diagnostic_name(self):
        if False:
            i = 10
            return i + 15
        return self.lnworker.__class__.__name__ + ', ' + self.transport.name()

    async def ping_if_required(self):
        if time.time() - self.last_message_time > 30:
            self.send_message('ping', num_pong_bytes=4, byteslen=4)
            self.pong_event.clear()
            await self.pong_event.wait()

    def process_message(self, message: bytes):
        if False:
            for i in range(10):
                print('nop')
        try:
            (message_type, payload) = decode_msg(message)
        except UnknownOptionalMsgType as e:
            self.logger.info(f'received unknown message from peer. ignoring: {e!r}')
            return
        except FailedToParseMsg as e:
            self.logger.info(f'failed to parse message from peer. disconnecting. msg_type={e.msg_type_name}({e.msg_type_int}). exc={e!r}')
            raise GracefulDisconnect() from e
        self.last_message_time = time.time()
        if message_type not in self.SPAMMY_MESSAGES:
            self.logger.debug(f'Received {message_type.upper()}')
        if self.is_channel_backup is True and message_type != 'init':
            return
        if message_type in self.ORDERED_MESSAGES:
            chan_id = payload.get('channel_id') or payload['temporary_channel_id']
            self.ordered_message_queues[chan_id].put_nowait((message_type, payload))
        else:
            if message_type not in ('error', 'warning') and 'channel_id' in payload:
                chan = self.get_channel_by_id(payload['channel_id'])
                if chan is None:
                    self.logger.info(f"Received {message_type} for unknown channel {payload['channel_id'].hex()}")
                    return
                args = (chan, payload)
            else:
                args = (payload,)
            try:
                f = getattr(self, 'on_' + message_type)
            except AttributeError:
                return
            if message_type in ['node_announcement', 'channel_announcement', 'channel_update']:
                payload['raw'] = message
            execution_result = f(*args)
            if asyncio.iscoroutinefunction(f):
                asyncio.ensure_future(self.taskgroup.spawn(execution_result))

    def on_warning(self, payload):
        if False:
            return 10
        chan_id = payload.get('channel_id')
        err_bytes = payload['data']
        is_known_chan_id = chan_id in self.channels or chan_id in self.temp_id_to_id
        self.logger.info(f'remote peer sent warning [DO NOT TRUST THIS MESSAGE]: {error_text_bytes_to_safe_str(err_bytes)}. chan_id={chan_id.hex()}. is_known_chan_id={is_known_chan_id!r}')

    def on_error(self, payload):
        if False:
            print('Hello World!')
        chan_id = payload.get('channel_id')
        err_bytes = payload['data']
        is_known_chan_id = chan_id in self.channels or chan_id in self.temp_id_to_id
        self.logger.info(f'remote peer sent error [DO NOT TRUST THIS MESSAGE]: {error_text_bytes_to_safe_str(err_bytes)}. chan_id={chan_id.hex()}. is_known_chan_id={is_known_chan_id!r}')
        if chan_id in self.channels:
            self.schedule_force_closing(chan_id)
            self.ordered_message_queues[chan_id].put_nowait((None, {'error': err_bytes}))
        elif chan_id in self.temp_id_to_id:
            chan_id = self.temp_id_to_id[chan_id] or chan_id
            self.ordered_message_queues[chan_id].put_nowait((None, {'error': err_bytes}))
        elif chan_id == bytes(32):
            for cid in self.channels:
                self.schedule_force_closing(cid)
                self.ordered_message_queues[cid].put_nowait((None, {'error': err_bytes}))
        else:
            return
        raise GracefulDisconnect

    async def send_warning(self, channel_id: bytes, message: str=None, *, close_connection=False):
        """Sends a warning and disconnects if close_connection.

        Note:
        * channel_id is the temporary channel id when the channel id is not yet available

        A sending node:
        MAY set channel_id to all zero if the warning is not related to a specific channel.

        when failure was caused by an invalid signature check:
        * SHOULD include the raw, hex-encoded transaction in reply to a funding_created,
          funding_signed, closing_signed, or commitment_signed message.
        """
        assert isinstance(channel_id, bytes)
        encoded_data = b'' if not message else message.encode('ascii')
        self.send_message('warning', channel_id=channel_id, data=encoded_data, len=len(encoded_data))
        if close_connection:
            raise GracefulDisconnect

    async def send_error(self, channel_id: bytes, message: str=None, *, force_close_channel=False):
        """Sends an error message and force closes the channel.

        Note:
        * channel_id is the temporary channel id when the channel id is not yet available

        A sending node:
        * SHOULD send error for protocol violations or internal errors that make channels
          unusable or that make further communication unusable.
        * SHOULD send error with the unknown channel_id in reply to messages of type
          32-255 related to unknown channels.
        * MUST fail the channel(s) referred to by the error message.
        * MAY set channel_id to all zero to indicate all channels.

        when failure was caused by an invalid signature check:
        * SHOULD include the raw, hex-encoded transaction in reply to a funding_created,
          funding_signed, closing_signed, or commitment_signed message.
        """
        assert isinstance(channel_id, bytes)
        encoded_data = b'' if not message else message.encode('ascii')
        self.send_message('error', channel_id=channel_id, data=encoded_data, len=len(encoded_data))
        if force_close_channel:
            if channel_id in self.channels:
                self.schedule_force_closing(channel_id)
            elif channel_id == bytes(32):
                for cid in self.channels:
                    self.schedule_force_closing(cid)
        raise GracefulDisconnect

    def on_ping(self, payload):
        if False:
            print('Hello World!')
        l = payload['num_pong_bytes']
        self.send_message('pong', byteslen=l)

    def on_pong(self, payload):
        if False:
            i = 10
            return i + 15
        self.pong_event.set()

    async def wait_for_message(self, expected_name: str, channel_id: bytes):
        q = self.ordered_message_queues[channel_id]
        (name, payload) = await util.wait_for2(q.get(), LN_P2P_NETWORK_TIMEOUT)
        if (err_bytes := payload.get('error')) is not None:
            err_text = error_text_bytes_to_safe_str(err_bytes)
            raise GracefulDisconnect(f'remote peer sent error [DO NOT TRUST THIS MESSAGE]: {err_text}')
        if name != expected_name:
            raise Exception(f"Received unexpected '{name}'")
        return payload

    def on_init(self, payload):
        if False:
            return 10
        if self._received_init:
            self.logger.info('ALREADY INITIALIZED BUT RECEIVED INIT')
            return
        _their_features = int.from_bytes(payload['features'], byteorder='big')
        _their_features |= int.from_bytes(payload['globalfeatures'], byteorder='big')
        try:
            self.their_features = validate_features(_their_features)
        except IncompatibleOrInsaneFeatures as e:
            raise GracefulDisconnect(f'remote sent insane features: {repr(e)}')
        try:
            self.features = ln_compare_features(self.features, self.their_features)
        except IncompatibleLightningFeatures as e:
            self.initialized.set_exception(e)
            raise GracefulDisconnect(f'{str(e)}')
        their_networks = payload['init_tlvs'].get('networks')
        if their_networks:
            their_chains = list(chunks(their_networks['chains'], 32))
            if constants.net.rev_genesis_bytes() not in their_chains:
                raise GracefulDisconnect(f'no common chain found with remote. (they sent: {their_chains})')
        self.lnworker.on_peer_successfully_established(self)
        self._received_init = True
        self.maybe_set_initialized()

    def on_node_announcement(self, payload):
        if False:
            print('Hello World!')
        if not self.lnworker.uses_trampoline():
            self.gossip_queue.put_nowait(('node_announcement', payload))

    def on_channel_announcement(self, payload):
        if False:
            print('Hello World!')
        if not self.lnworker.uses_trampoline():
            self.gossip_queue.put_nowait(('channel_announcement', payload))

    def on_channel_update(self, payload):
        if False:
            while True:
                i = 10
        self.maybe_save_remote_update(payload)
        if not self.lnworker.uses_trampoline():
            self.gossip_queue.put_nowait(('channel_update', payload))

    def maybe_save_remote_update(self, payload):
        if False:
            i = 10
            return i + 15
        if not self.channels:
            return
        for chan in self.channels.values():
            if payload['short_channel_id'] in [chan.short_channel_id, chan.get_local_scid_alias()]:
                chan.set_remote_update(payload)
                self.logger.info(f'saved remote channel_update gossip msg for chan {chan.get_id_for_log()}')
                break
        else:
            short_channel_id = ShortChannelID(payload['short_channel_id'])
            self.logger.info(f'received orphan channel update {short_channel_id}')
            self.orphan_channel_updates[short_channel_id] = payload
            while len(self.orphan_channel_updates) > 25:
                self.orphan_channel_updates.popitem(last=False)

    def on_announcement_signatures(self, chan: Channel, payload):
        if False:
            print('Hello World!')
        h = chan.get_channel_announcement_hash()
        node_signature = payload['node_signature']
        bitcoin_signature = payload['bitcoin_signature']
        if not ecc.verify_signature(chan.config[REMOTE].multisig_key.pubkey, bitcoin_signature, h):
            raise Exception('bitcoin_sig invalid in announcement_signatures')
        if not ecc.verify_signature(self.pubkey, node_signature, h):
            raise Exception('node_sig invalid in announcement_signatures')
        chan.config[REMOTE].announcement_node_sig = node_signature
        chan.config[REMOTE].announcement_bitcoin_sig = bitcoin_signature
        self.lnworker.save_channel(chan)
        self.maybe_send_announcement_signatures(chan, is_reply=True)

    def handle_disconnect(func):
        if False:
            return 10

        @functools.wraps(func)
        async def wrapper_func(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except GracefulDisconnect as e:
                self.logger.log(e.log_level, f'Disconnecting: {repr(e)}')
            except (LightningPeerConnectionClosed, IncompatibleLightningFeatures, aiorpcx.socks.SOCKSError) as e:
                self.logger.info(f'Disconnecting: {repr(e)}')
            finally:
                self.close_and_cleanup()
        return wrapper_func

    @ignore_exceptions
    @log_exceptions
    @handle_disconnect
    async def main_loop(self):
        async with self.taskgroup as group:
            await group.spawn(self._message_loop())
            await group.spawn(self.htlc_switch())
            await group.spawn(self.query_gossip())
            await group.spawn(self.process_gossip())
            await group.spawn(self.send_own_gossip())

    async def process_gossip(self):
        while True:
            await asyncio.sleep(5)
            if not self.network.lngossip:
                continue
            chan_anns = []
            chan_upds = []
            node_anns = []
            while True:
                (name, payload) = await self.gossip_queue.get()
                if name == 'channel_announcement':
                    chan_anns.append(payload)
                elif name == 'channel_update':
                    chan_upds.append(payload)
                elif name == 'node_announcement':
                    node_anns.append(payload)
                else:
                    raise Exception('unknown message')
                if self.gossip_queue.empty():
                    break
            if self.network.lngossip:
                await self.network.lngossip.process_gossip(chan_anns, node_anns, chan_upds)

    async def send_own_gossip(self):
        if self.lnworker == self.lnworker.network.lngossip:
            return
        await asyncio.sleep(10)
        while True:
            public_channels = [chan for chan in self.lnworker.channels.values() if chan.is_public()]
            if public_channels:
                alias = self.lnworker.config.LIGHTNING_NODE_ALIAS
                self.send_node_announcement(alias)
                for chan in public_channels:
                    if chan.is_open() and chan.peer_state == PeerState.GOOD:
                        self.maybe_send_channel_announcement(chan)
            await asyncio.sleep(600)

    async def query_gossip(self):
        try:
            await util.wait_for2(self.initialized, LN_P2P_NETWORK_TIMEOUT)
        except Exception as e:
            raise GracefulDisconnect(f'Failed to initialize: {e!r}') from e
        if self.lnworker == self.lnworker.network.lngossip:
            try:
                (ids, complete) = await util.wait_for2(self.get_channel_range(), LN_P2P_NETWORK_TIMEOUT)
            except asyncio.TimeoutError as e:
                raise GracefulDisconnect('query_channel_range timed out') from e
            self.logger.info('Received {} channel ids. (complete: {})'.format(len(ids), complete))
            await self.lnworker.add_new_ids(ids)
            while True:
                todo = self.lnworker.get_ids_to_query()
                if not todo:
                    await asyncio.sleep(1)
                    continue
                await self.get_short_channel_ids(todo)

    async def get_channel_range(self):
        first_block = constants.net.BLOCK_HEIGHT_FIRST_LIGHTNING_CHANNELS
        num_blocks = self.lnworker.network.get_local_height() - first_block
        self.query_channel_range(first_block, num_blocks)
        intervals = []
        ids = set()
        while True:
            (index, num, complete, _ids) = await self.reply_channel_range.get()
            ids.update(_ids)
            intervals.append((index, index + num))
            intervals.sort()
            while len(intervals) > 1:
                (a, b) = intervals[0]
                (c, d) = intervals[1]
                if not (a <= c and a <= b and (c <= d)):
                    raise Exception(f'insane reply_channel_range intervals {(a, b, c, d)}')
                if b >= c:
                    intervals = [(a, d)] + intervals[2:]
                else:
                    break
            if len(intervals) == 1 and complete:
                (a, b) = intervals[0]
                if a <= first_block and b >= first_block + num_blocks:
                    break
        return (ids, complete)

    def request_gossip(self, timestamp=0):
        if False:
            while True:
                i = 10
        if timestamp == 0:
            self.logger.info('requesting whole channel graph')
        else:
            self.logger.info(f'requesting channel graph since {datetime.fromtimestamp(timestamp).ctime()}')
        self.send_message('gossip_timestamp_filter', chain_hash=constants.net.rev_genesis_bytes(), first_timestamp=timestamp, timestamp_range=b'\xff' * 4)

    def query_channel_range(self, first_block, num_blocks):
        if False:
            print('Hello World!')
        self.logger.info(f'query channel range {first_block} {num_blocks}')
        self.send_message('query_channel_range', chain_hash=constants.net.rev_genesis_bytes(), first_blocknum=first_block, number_of_blocks=num_blocks)

    def decode_short_ids(self, encoded):
        if False:
            print('Hello World!')
        if encoded[0] == 0:
            decoded = encoded[1:]
        elif encoded[0] == 1:
            decoded = zlib.decompress(encoded[1:])
        else:
            raise Exception(f'decode_short_ids: unexpected first byte: {encoded[0]}')
        ids = [decoded[i:i + 8] for i in range(0, len(decoded), 8)]
        return ids

    def on_reply_channel_range(self, payload):
        if False:
            while True:
                i = 10
        first = payload['first_blocknum']
        num = payload['number_of_blocks']
        complete = bool(int.from_bytes(payload['sync_complete'], 'big'))
        encoded = payload['encoded_short_ids']
        ids = self.decode_short_ids(encoded)
        self.reply_channel_range.put_nowait((first, num, complete, ids))

    async def get_short_channel_ids(self, ids):
        self.logger.info(f'Querying {len(ids)} short_channel_ids')
        assert not self.querying.is_set()
        self.query_short_channel_ids(ids)
        await self.querying.wait()
        self.querying.clear()

    def query_short_channel_ids(self, ids, compressed=True):
        if False:
            while True:
                i = 10
        ids = sorted(ids)
        s = b''.join(ids)
        encoded = zlib.compress(s) if compressed else s
        prefix = b'\x01' if compressed else b'\x00'
        self.send_message('query_short_channel_ids', chain_hash=constants.net.rev_genesis_bytes(), len=1 + len(encoded), encoded_short_ids=prefix + encoded)

    async def _message_loop(self):
        try:
            await util.wait_for2(self.initialize(), LN_P2P_NETWORK_TIMEOUT)
        except (OSError, asyncio.TimeoutError, HandshakeFailed) as e:
            raise GracefulDisconnect(f'initialize failed: {repr(e)}') from e
        async for msg in self.transport.read_messages():
            self.process_message(msg)
            if self.DELAY_INC_MSG_PROCESSING_SLEEP:
                await asyncio.sleep(self.DELAY_INC_MSG_PROCESSING_SLEEP)

    def on_reply_short_channel_ids_end(self, payload):
        if False:
            for i in range(10):
                print('nop')
        self.querying.set()

    def close_and_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self.transport:
                self.transport.close()
        except Exception:
            pass
        self.lnworker.peer_closed(self)
        self.got_disconnected.set()

    def is_shutdown_anysegwit(self):
        if False:
            for i in range(10):
                print('nop')
        return self.features.supports(LnFeatures.OPTION_SHUTDOWN_ANYSEGWIT_OPT)

    def is_channel_type(self):
        if False:
            i = 10
            return i + 15
        return self.features.supports(LnFeatures.OPTION_CHANNEL_TYPE_OPT)

    def accepts_zeroconf(self):
        if False:
            i = 10
            return i + 15
        return self.features.supports(LnFeatures.OPTION_ZEROCONF_OPT)

    def is_upfront_shutdown_script(self):
        if False:
            print('Hello World!')
        return self.features.supports(LnFeatures.OPTION_UPFRONT_SHUTDOWN_SCRIPT_OPT)

    def upfront_shutdown_script_from_payload(self, payload, msg_identifier: str) -> Optional[bytes]:
        if False:
            print('Hello World!')
        if msg_identifier not in ['accept', 'open']:
            raise ValueError("msg_identifier must be either 'accept' or 'open'")
        uss_tlv = payload[msg_identifier + '_channel_tlvs'].get('upfront_shutdown_script')
        if uss_tlv and self.is_upfront_shutdown_script():
            upfront_shutdown_script = uss_tlv['shutdown_scriptpubkey']
        else:
            upfront_shutdown_script = b''
        self.logger.info(f'upfront shutdown script received: {upfront_shutdown_script}')
        return upfront_shutdown_script

    def make_local_config(self, funding_sat: int, push_msat: int, initiator: HTLCOwner, channel_type: ChannelType) -> LocalConfig:
        if False:
            for i in range(10):
                print('nop')
        channel_seed = os.urandom(32)
        initial_msat = funding_sat * 1000 - push_msat if initiator == LOCAL else push_msat
        upfront_shutdown_script = b''
        assert channel_type & channel_type.OPTION_STATIC_REMOTEKEY
        wallet = self.lnworker.wallet
        assert wallet.txin_type == 'p2wpkh'
        addr = wallet.get_new_sweep_address_for_channel()
        static_remotekey = bytes.fromhex(wallet.get_public_key(addr))
        dust_limit_sat = bitcoin.DUST_LIMIT_P2PKH
        reserve_sat = max(funding_sat // 100, dust_limit_sat)
        local_config = LocalConfig.from_seed(channel_seed=channel_seed, static_remotekey=static_remotekey, upfront_shutdown_script=upfront_shutdown_script, to_self_delay=self.network.config.LIGHTNING_TO_SELF_DELAY_CSV, dust_limit_sat=dust_limit_sat, max_htlc_value_in_flight_msat=funding_sat * 1000, max_accepted_htlcs=30, initial_msat=initial_msat, reserve_sat=reserve_sat, funding_locked_received=False, current_commitment_signature=None, current_htlc_signatures=b'', htlc_minimum_msat=1, announcement_node_sig=b'', announcement_bitcoin_sig=b'')
        local_config.validate_params(funding_sat=funding_sat, config=self.network.config, peer_features=self.features)
        return local_config

    def temporarily_reserve_funding_tx_change_address(func):
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        async def wrapper(self: 'Peer', *args, **kwargs):
            funding_tx = kwargs['funding_tx']
            wallet = self.lnworker.wallet
            change_addresses = [txout.address for txout in funding_tx.outputs() if wallet.is_change(txout.address)]
            for addr in change_addresses:
                wallet.set_reserved_state_of_address(addr, reserved=True)
            try:
                return await func(self, *args, **kwargs)
            finally:
                for addr in change_addresses:
                    self.lnworker.wallet.set_reserved_state_of_address(addr, reserved=False)
        return wrapper

    @temporarily_reserve_funding_tx_change_address
    async def channel_establishment_flow(self, *, funding_tx: 'PartialTransaction', funding_sat: int, push_msat: int, public: bool, zeroconf: bool=False, temp_channel_id: bytes, opening_fee: int=None) -> Tuple[Channel, 'PartialTransaction']:
        """Implements the channel opening flow.

        -> open_channel message
        <- accept_channel message
        -> funding_created message
        <- funding_signed message

        Channel configurations are initialized in this method.
        """
        await util.wait_for2(self.initialized, LN_P2P_NETWORK_TIMEOUT)
        if self.lnworker.uses_trampoline() and (not self.lnworker.is_trampoline_peer(self.pubkey)):
            raise Exception('Not a trampoline node: ' + str(self.their_features))
        if public and (not self.lnworker.config.EXPERIMENTAL_LN_FORWARD_PAYMENTS):
            raise Exception('Cannot create public channels')
        channel_flags = CF_ANNOUNCE_CHANNEL if public else 0
        feerate = self.lnworker.current_feerate_per_kw()
        open_channel_tlvs = {}
        assert self.their_features.supports(LnFeatures.OPTION_STATIC_REMOTEKEY_OPT)
        our_channel_type = ChannelType(ChannelType.OPTION_STATIC_REMOTEKEY)
        if zeroconf:
            our_channel_type |= ChannelType(ChannelType.OPTION_ZEROCONF)
        if self.is_channel_type():
            open_channel_tlvs['channel_type'] = {'type': our_channel_type.to_bytes_minimal()}
        local_config = self.make_local_config(funding_sat, push_msat, LOCAL, our_channel_type)
        open_channel_tlvs['upfront_shutdown_script'] = {'shutdown_scriptpubkey': local_config.upfront_shutdown_script}
        if opening_fee:
            open_channel_tlvs['channel_opening_fee'] = {'channel_opening_fee': opening_fee}
        per_commitment_secret_first = get_per_commitment_secret_from_seed(local_config.per_commitment_secret_seed, RevocationStore.START_INDEX)
        per_commitment_point_first = secret_to_pubkey(int.from_bytes(per_commitment_secret_first, 'big'))
        self.temp_id_to_id[temp_channel_id] = None
        self.send_message('open_channel', temporary_channel_id=temp_channel_id, chain_hash=constants.net.rev_genesis_bytes(), funding_satoshis=funding_sat, push_msat=push_msat, dust_limit_satoshis=local_config.dust_limit_sat, feerate_per_kw=feerate, max_accepted_htlcs=local_config.max_accepted_htlcs, funding_pubkey=local_config.multisig_key.pubkey, revocation_basepoint=local_config.revocation_basepoint.pubkey, htlc_basepoint=local_config.htlc_basepoint.pubkey, payment_basepoint=local_config.payment_basepoint.pubkey, delayed_payment_basepoint=local_config.delayed_basepoint.pubkey, first_per_commitment_point=per_commitment_point_first, to_self_delay=local_config.to_self_delay, max_htlc_value_in_flight_msat=local_config.max_htlc_value_in_flight_msat, channel_flags=channel_flags, channel_reserve_satoshis=local_config.reserve_sat, htlc_minimum_msat=local_config.htlc_minimum_msat, open_channel_tlvs=open_channel_tlvs)
        payload = await self.wait_for_message('accept_channel', temp_channel_id)
        self.logger.debug(f'received accept_channel for temp_channel_id={temp_channel_id.hex()}. payload={payload!r}')
        remote_per_commitment_point = payload['first_per_commitment_point']
        funding_txn_minimum_depth = payload['minimum_depth']
        if not zeroconf and funding_txn_minimum_depth <= 0:
            raise Exception(f'minimum depth too low, {funding_txn_minimum_depth}')
        if funding_txn_minimum_depth > 30:
            raise Exception(f'minimum depth too high, {funding_txn_minimum_depth}')
        upfront_shutdown_script = self.upfront_shutdown_script_from_payload(payload, 'accept')
        accept_channel_tlvs = payload.get('accept_channel_tlvs')
        their_channel_type = accept_channel_tlvs.get('channel_type') if accept_channel_tlvs else None
        if their_channel_type:
            their_channel_type = ChannelType.from_bytes(their_channel_type['type'], byteorder='big').discard_unknown_and_check()
            if open_channel_tlvs.get('channel_type') is not None and their_channel_type != our_channel_type:
                raise Exception('Channel type is not the one that we sent.')
        remote_config = RemoteConfig(payment_basepoint=OnlyPubkeyKeypair(payload['payment_basepoint']), multisig_key=OnlyPubkeyKeypair(payload['funding_pubkey']), htlc_basepoint=OnlyPubkeyKeypair(payload['htlc_basepoint']), delayed_basepoint=OnlyPubkeyKeypair(payload['delayed_payment_basepoint']), revocation_basepoint=OnlyPubkeyKeypair(payload['revocation_basepoint']), to_self_delay=payload['to_self_delay'], dust_limit_sat=payload['dust_limit_satoshis'], max_htlc_value_in_flight_msat=payload['max_htlc_value_in_flight_msat'], max_accepted_htlcs=payload['max_accepted_htlcs'], initial_msat=push_msat, reserve_sat=payload['channel_reserve_satoshis'], htlc_minimum_msat=payload['htlc_minimum_msat'], next_per_commitment_point=remote_per_commitment_point, current_per_commitment_point=None, upfront_shutdown_script=upfront_shutdown_script, announcement_node_sig=b'', announcement_bitcoin_sig=b'')
        ChannelConfig.cross_validate_params(local_config=local_config, remote_config=remote_config, funding_sat=funding_sat, is_local_initiator=True, initial_feerate_per_kw=feerate, config=self.network.config, peer_features=self.features)
        redeem_script = funding_output_script(local_config, remote_config)
        funding_address = bitcoin.redeem_script_to_address('p2wsh', redeem_script)
        funding_output = PartialTxOutput.from_address_and_value(funding_address, funding_sat)
        funding_tx.replace_output_address(DummyAddress.CHANNEL, funding_address)
        has_onchain_backup = self.lnworker and self.lnworker.has_recoverable_channels()
        if has_onchain_backup:
            backup_data = self.lnworker.cb_data(self.pubkey)
            dummy_scriptpubkey = make_op_return(backup_data)
            for o in funding_tx.outputs():
                if o.scriptpubkey == dummy_scriptpubkey:
                    encrypted_data = self.lnworker.encrypt_cb_data(backup_data, funding_address)
                    assert len(encrypted_data) == len(backup_data)
                    o.scriptpubkey = make_op_return(encrypted_data)
                    break
            else:
                raise Exception('op_return output not found in funding tx')
        funding_tx.set_rbf(False)
        if not funding_tx.is_segwit():
            raise Exception('Funding transaction is not segwit')
        funding_txid = funding_tx.txid()
        assert funding_txid
        funding_index = funding_tx.outputs().index(funding_output)
        (channel_id, funding_txid_bytes) = channel_id_from_funding_tx(funding_txid, funding_index)
        outpoint = Outpoint(funding_txid, funding_index)
        constraints = ChannelConstraints(flags=channel_flags, capacity=funding_sat, is_initiator=True, funding_txn_minimum_depth=funding_txn_minimum_depth)
        storage = self.create_channel_storage(channel_id, outpoint, local_config, remote_config, constraints, our_channel_type)
        chan = Channel(storage, lnworker=self.lnworker, initial_feerate=feerate)
        chan.storage['funding_inputs'] = [txin.prevout.to_json() for txin in funding_tx.inputs()]
        chan.storage['has_onchain_backup'] = has_onchain_backup
        if isinstance(self.transport, LNTransport):
            chan.add_or_update_peer_addr(self.transport.peer_addr)
        (sig_64, _) = chan.sign_next_commitment()
        self.temp_id_to_id[temp_channel_id] = channel_id
        self.send_message('funding_created', temporary_channel_id=temp_channel_id, funding_txid=funding_txid_bytes, funding_output_index=funding_index, signature=sig_64)
        self.funding_created_sent.add(channel_id)
        payload = await self.wait_for_message('funding_signed', channel_id)
        self.logger.info('received funding_signed')
        remote_sig = payload['signature']
        try:
            chan.receive_new_commitment(remote_sig, [])
        except LNProtocolWarning as e:
            await self.send_warning(channel_id, message=str(e), close_connection=True)
        chan.open_with_first_pcp(remote_per_commitment_point, remote_sig)
        chan.set_state(ChannelState.OPENING)
        if zeroconf:
            chan.set_state(ChannelState.FUNDED)
            self.send_channel_ready(chan)
        self.lnworker.add_new_channel(chan)
        return (chan, funding_tx)

    def create_channel_storage(self, channel_id, outpoint, local_config, remote_config, constraints, channel_type):
        if False:
            while True:
                i = 10
        chan_dict = {'node_id': self.pubkey.hex(), 'channel_id': channel_id.hex(), 'short_channel_id': None, 'funding_outpoint': outpoint, 'remote_config': remote_config, 'local_config': local_config, 'constraints': constraints, 'remote_update': None, 'state': ChannelState.PREOPENING.name, 'onion_keys': {}, 'data_loss_protect_remote_pcp': {}, 'log': {}, 'fail_htlc_reasons': {}, 'unfulfilled_htlcs': {}, 'revocation_store': {}, 'channel_type': channel_type}
        return StoredDict(chan_dict, None, [])

    async def on_open_channel(self, payload):
        """Implements the channel acceptance flow.

        <- open_channel message
        -> accept_channel message
        <- funding_created message
        -> funding_signed message

        Channel configurations are initialized in this method.
        """
        if self.lnworker.has_recoverable_channels():
            raise Exception('not accepting channels')
        if payload['chain_hash'] != constants.net.rev_genesis_bytes():
            raise Exception('wrong chain_hash')
        funding_sat = payload['funding_satoshis']
        push_msat = payload['push_msat']
        feerate = payload['feerate_per_kw']
        temp_chan_id = payload['temporary_channel_id']
        self.temp_id_to_id[temp_chan_id] = None
        open_channel_tlvs = payload.get('open_channel_tlvs')
        channel_type = open_channel_tlvs.get('channel_type') if open_channel_tlvs else None
        channel_opening_fee = open_channel_tlvs.get('channel_opening_fee') if open_channel_tlvs else None
        if channel_opening_fee:
            pass
        if self.is_channel_type() and channel_type is None:
            raise Exception("sender has advertized option_channel_type, but hasn't sent the channel type")
        elif self.is_channel_type() and channel_type is not None:
            channel_type = ChannelType.from_bytes(channel_type['type'], byteorder='big').discard_unknown_and_check()
            if not channel_type.complies_with_features(self.features):
                raise Exception("sender has sent a channel type we don't support")
        local_config = self.make_local_config(funding_sat, push_msat, REMOTE, channel_type)
        upfront_shutdown_script = self.upfront_shutdown_script_from_payload(payload, 'open')
        remote_config = RemoteConfig(payment_basepoint=OnlyPubkeyKeypair(payload['payment_basepoint']), multisig_key=OnlyPubkeyKeypair(payload['funding_pubkey']), htlc_basepoint=OnlyPubkeyKeypair(payload['htlc_basepoint']), delayed_basepoint=OnlyPubkeyKeypair(payload['delayed_payment_basepoint']), revocation_basepoint=OnlyPubkeyKeypair(payload['revocation_basepoint']), to_self_delay=payload['to_self_delay'], dust_limit_sat=payload['dust_limit_satoshis'], max_htlc_value_in_flight_msat=payload['max_htlc_value_in_flight_msat'], max_accepted_htlcs=payload['max_accepted_htlcs'], initial_msat=funding_sat * 1000 - push_msat, reserve_sat=payload['channel_reserve_satoshis'], htlc_minimum_msat=payload['htlc_minimum_msat'], next_per_commitment_point=payload['first_per_commitment_point'], current_per_commitment_point=None, upfront_shutdown_script=upfront_shutdown_script, announcement_node_sig=b'', announcement_bitcoin_sig=b'')
        ChannelConfig.cross_validate_params(local_config=local_config, remote_config=remote_config, funding_sat=funding_sat, is_local_initiator=False, initial_feerate_per_kw=feerate, config=self.network.config, peer_features=self.features)
        channel_flags = ord(payload['channel_flags'])
        per_commitment_secret_first = get_per_commitment_secret_from_seed(local_config.per_commitment_secret_seed, RevocationStore.START_INDEX)
        per_commitment_point_first = secret_to_pubkey(int.from_bytes(per_commitment_secret_first, 'big'))
        is_zeroconf = channel_type & channel_type.OPTION_ZEROCONF
        if is_zeroconf and (not self.network.config.ZEROCONF_TRUSTED_NODE.startswith(self.pubkey.hex())):
            raise Exception(f'not accepting zeroconf from node {self.pubkey}')
        min_depth = 0 if is_zeroconf else 3
        accept_channel_tlvs = {'upfront_shutdown_script': {'shutdown_scriptpubkey': local_config.upfront_shutdown_script}}
        if self.is_channel_type():
            accept_channel_tlvs['channel_type'] = {'type': channel_type.to_bytes_minimal()}
        self.send_message('accept_channel', temporary_channel_id=temp_chan_id, dust_limit_satoshis=local_config.dust_limit_sat, max_htlc_value_in_flight_msat=local_config.max_htlc_value_in_flight_msat, channel_reserve_satoshis=local_config.reserve_sat, htlc_minimum_msat=local_config.htlc_minimum_msat, minimum_depth=min_depth, to_self_delay=local_config.to_self_delay, max_accepted_htlcs=local_config.max_accepted_htlcs, funding_pubkey=local_config.multisig_key.pubkey, revocation_basepoint=local_config.revocation_basepoint.pubkey, payment_basepoint=local_config.payment_basepoint.pubkey, delayed_payment_basepoint=local_config.delayed_basepoint.pubkey, htlc_basepoint=local_config.htlc_basepoint.pubkey, first_per_commitment_point=per_commitment_point_first, accept_channel_tlvs=accept_channel_tlvs)
        funding_created = await self.wait_for_message('funding_created', temp_chan_id)
        funding_idx = funding_created['funding_output_index']
        funding_txid = funding_created['funding_txid'][::-1].hex()
        (channel_id, funding_txid_bytes) = channel_id_from_funding_tx(funding_txid, funding_idx)
        constraints = ChannelConstraints(flags=channel_flags, capacity=funding_sat, is_initiator=False, funding_txn_minimum_depth=min_depth)
        outpoint = Outpoint(funding_txid, funding_idx)
        chan_dict = self.create_channel_storage(channel_id, outpoint, local_config, remote_config, constraints, channel_type)
        chan = Channel(chan_dict, lnworker=self.lnworker, initial_feerate=feerate, opening_fee=channel_opening_fee)
        chan.storage['init_timestamp'] = int(time.time())
        if isinstance(self.transport, LNTransport):
            chan.add_or_update_peer_addr(self.transport.peer_addr)
        remote_sig = funding_created['signature']
        try:
            chan.receive_new_commitment(remote_sig, [])
        except LNProtocolWarning as e:
            await self.send_warning(channel_id, message=str(e), close_connection=True)
        (sig_64, _) = chan.sign_next_commitment()
        self.send_message('funding_signed', channel_id=channel_id, signature=sig_64)
        self.temp_id_to_id[temp_chan_id] = channel_id
        self.funding_signed_sent.add(chan.channel_id)
        chan.open_with_first_pcp(payload['first_per_commitment_point'], remote_sig)
        chan.set_state(ChannelState.OPENING)
        if is_zeroconf:
            chan.set_state(ChannelState.FUNDED)
            self.send_channel_ready(chan)
        self.lnworker.add_new_channel(chan)

    async def request_force_close(self, channel_id: bytes):
        """Try to trigger the remote peer to force-close."""
        await self.initialized
        latest_point = secret_to_pubkey(42)
        self.send_message('channel_reestablish', channel_id=channel_id, next_commitment_number=0, next_revocation_number=0, your_last_per_commitment_secret=0, my_current_per_commitment_point=latest_point)
        self.send_message('error', channel_id=channel_id, data=b'', len=0)

    def schedule_force_closing(self, channel_id: bytes):
        if False:
            for i in range(10):
                print('nop')
        " wrapper of lnworker's method, that raises if channel is not with this peer "
        channels_with_peer = list(self.channels.keys())
        channels_with_peer.extend(self.temp_id_to_id.values())
        if channel_id not in channels_with_peer:
            raise ValueError(f'channel {channel_id.hex()} does not belong to this peer')
        chan = self.channels.get(channel_id)
        if not chan:
            self.logger.warning(f'tried to force-close channel {channel_id.hex()} but it is not in self.channels yet')
        if ChanCloseOption.LOCAL_FCLOSE in chan.get_close_options():
            self.lnworker.schedule_force_closing(channel_id)
        else:
            self.logger.info(f'tried to force-close channel {chan.get_id_for_log()} but close option is not allowed. chan.get_state()={chan.get_state()!r}')

    def on_channel_reestablish(self, chan, msg):
        if False:
            i = 10
            return i + 15
        their_next_local_ctn = msg['next_commitment_number']
        their_oldest_unrevoked_remote_ctn = msg['next_revocation_number']
        their_local_pcp = msg.get('my_current_per_commitment_point')
        their_claim_of_our_last_per_commitment_secret = msg.get('your_last_per_commitment_secret')
        self.logger.info(f'channel_reestablish ({chan.get_id_for_log()}): received channel_reestablish with (their_next_local_ctn={their_next_local_ctn}, their_oldest_unrevoked_remote_ctn={their_oldest_unrevoked_remote_ctn})')
        if their_next_local_ctn < 0:
            raise RemoteMisbehaving(f'channel reestablish: their_next_local_ctn < 0')
        if their_oldest_unrevoked_remote_ctn < 0:
            raise RemoteMisbehaving(f'channel reestablish: their_oldest_unrevoked_remote_ctn < 0')
        oldest_unrevoked_local_ctn = chan.get_oldest_unrevoked_ctn(LOCAL)
        latest_local_ctn = chan.get_latest_ctn(LOCAL)
        next_local_ctn = chan.get_next_ctn(LOCAL)
        oldest_unrevoked_remote_ctn = chan.get_oldest_unrevoked_ctn(REMOTE)
        latest_remote_ctn = chan.get_latest_ctn(REMOTE)
        next_remote_ctn = chan.get_next_ctn(REMOTE)
        we_are_ahead = False
        they_are_ahead = False
        we_must_resend_revoke_and_ack = False
        if next_remote_ctn != their_next_local_ctn:
            if their_next_local_ctn == latest_remote_ctn and chan.hm.is_revack_pending(REMOTE):
                pass
            else:
                self.logger.warning(f'channel_reestablish ({chan.get_id_for_log()}): expected remote ctn {next_remote_ctn}, got {their_next_local_ctn}')
                if their_next_local_ctn < next_remote_ctn:
                    we_are_ahead = True
                else:
                    they_are_ahead = True
        if oldest_unrevoked_local_ctn != their_oldest_unrevoked_remote_ctn:
            if oldest_unrevoked_local_ctn - 1 == their_oldest_unrevoked_remote_ctn:
                we_must_resend_revoke_and_ack = True
            else:
                self.logger.warning(f'channel_reestablish ({chan.get_id_for_log()}): expected local ctn {oldest_unrevoked_local_ctn}, got {their_oldest_unrevoked_remote_ctn}')
                if their_oldest_unrevoked_remote_ctn < oldest_unrevoked_local_ctn:
                    we_are_ahead = True
                else:
                    they_are_ahead = True
        assert self.features.supports(LnFeatures.OPTION_DATA_LOSS_PROTECT_OPT)

        def are_datalossprotect_fields_valid() -> bool:
            if False:
                while True:
                    i = 10
            if their_local_pcp is None or their_claim_of_our_last_per_commitment_secret is None:
                return False
            if their_oldest_unrevoked_remote_ctn > 0:
                (our_pcs, __) = chan.get_secret_and_point(LOCAL, their_oldest_unrevoked_remote_ctn - 1)
            else:
                assert their_oldest_unrevoked_remote_ctn == 0
                our_pcs = bytes(32)
            if our_pcs != their_claim_of_our_last_per_commitment_secret:
                self.logger.error(f'channel_reestablish ({chan.get_id_for_log()}): (DLP) local PCS mismatch: {our_pcs.hex()} != {their_claim_of_our_last_per_commitment_secret.hex()}')
                return False
            assert chan.is_static_remotekey_enabled()
            return True
        if not are_datalossprotect_fields_valid():
            raise RemoteMisbehaving('channel_reestablish: data loss protect fields invalid')
        fut = self.channel_reestablish_msg[chan.channel_id]
        if they_are_ahead:
            self.logger.warning(f'channel_reestablish ({chan.get_id_for_log()}): remote is ahead of us! They should force-close. Remote PCP: {their_local_pcp.hex()}')
            chan.set_data_loss_protect_remote_pcp(their_next_local_ctn - 1, their_local_pcp)
            chan.set_state(ChannelState.WE_ARE_TOXIC)
            self.lnworker.save_channel(chan)
            chan.peer_state = PeerState.BAD
            fut.set_exception(RemoteMisbehaving('remote ahead of us'))
        elif we_are_ahead:
            self.logger.warning(f'channel_reestablish ({chan.get_id_for_log()}): we are ahead of remote! trying to force-close.')
            self.schedule_force_closing(chan.channel_id)
            fut.set_exception(RemoteMisbehaving('we are ahead of remote'))
        else:
            fut.set_result((we_must_resend_revoke_and_ack, their_next_local_ctn))

    async def reestablish_channel(self, chan: Channel):
        await self.initialized
        chan_id = chan.channel_id
        if chan.should_request_force_close:
            chan.set_state(ChannelState.REQUESTED_FCLOSE)
            await self.request_force_close(chan_id)
            chan.should_request_force_close = False
            return
        assert ChannelState.PREOPENING < chan.get_state() < ChannelState.FORCE_CLOSING
        if chan.peer_state != PeerState.DISCONNECTED:
            self.logger.info(f'reestablish_channel was called but channel {chan.get_id_for_log()} already in peer_state {chan.peer_state!r}')
            return
        chan.peer_state = PeerState.REESTABLISHING
        util.trigger_callback('channel', self.lnworker.wallet, chan)
        oldest_unrevoked_local_ctn = chan.get_oldest_unrevoked_ctn(LOCAL)
        latest_local_ctn = chan.get_latest_ctn(LOCAL)
        next_local_ctn = chan.get_next_ctn(LOCAL)
        oldest_unrevoked_remote_ctn = chan.get_oldest_unrevoked_ctn(REMOTE)
        latest_remote_ctn = chan.get_latest_ctn(REMOTE)
        next_remote_ctn = chan.get_next_ctn(REMOTE)
        chan.hm.discard_unsigned_remote_updates()
        assert chan.is_static_remotekey_enabled()
        (latest_secret, latest_point) = chan.get_secret_and_point(LOCAL, 0)
        if oldest_unrevoked_remote_ctn == 0:
            last_rev_secret = 0
        else:
            last_rev_index = oldest_unrevoked_remote_ctn - 1
            last_rev_secret = chan.revocation_store.retrieve_secret(RevocationStore.START_INDEX - last_rev_index)
        self.send_message('channel_reestablish', channel_id=chan_id, next_commitment_number=next_local_ctn, next_revocation_number=oldest_unrevoked_remote_ctn, your_last_per_commitment_secret=last_rev_secret, my_current_per_commitment_point=latest_point)
        self.logger.info(f'channel_reestablish ({chan.get_id_for_log()}): sent channel_reestablish with (next_local_ctn={next_local_ctn}, oldest_unrevoked_remote_ctn={oldest_unrevoked_remote_ctn})')
        fut = self.channel_reestablish_msg[chan_id]
        await fut
        (we_must_resend_revoke_and_ack, their_next_local_ctn) = fut.result()

        def replay_updates_and_commitsig():
            if False:
                return 10
            unacked = chan.hm.get_unacked_local_updates()
            replayed_msgs = []
            for (ctn, messages) in unacked.items():
                if ctn < their_next_local_ctn:
                    continue
                for raw_upd_msg in messages:
                    self.transport.send_bytes(raw_upd_msg)
                    replayed_msgs.append(raw_upd_msg)
            self.logger.info(f'channel_reestablish ({chan.get_id_for_log()}): replayed {len(replayed_msgs)} unacked messages. {[decode_msg(raw_upd_msg)[0] for raw_upd_msg in replayed_msgs]}')

        def resend_revoke_and_ack():
            if False:
                i = 10
                return i + 15
            (last_secret, last_point) = chan.get_secret_and_point(LOCAL, oldest_unrevoked_local_ctn - 1)
            (next_secret, next_point) = chan.get_secret_and_point(LOCAL, oldest_unrevoked_local_ctn + 1)
            self.send_message('revoke_and_ack', channel_id=chan.channel_id, per_commitment_secret=last_secret, next_per_commitment_point=next_point)
        was_revoke_last = chan.hm.was_revoke_last()
        if we_must_resend_revoke_and_ack and (not was_revoke_last):
            self.logger.info(f'channel_reestablish ({chan.get_id_for_log()}): replaying a revoke_and_ack first.')
            resend_revoke_and_ack()
        replay_updates_and_commitsig()
        if we_must_resend_revoke_and_ack and was_revoke_last:
            self.logger.info(f'channel_reestablish ({chan.get_id_for_log()}): replaying a revoke_and_ack last.')
            resend_revoke_and_ack()
        chan.peer_state = PeerState.GOOD
        if chan.is_funded():
            chan_just_became_ready = their_next_local_ctn == next_local_ctn == 1
            if chan_just_became_ready or self.features.supports(LnFeatures.OPTION_SCID_ALIAS_OPT):
                self.send_channel_ready(chan)
        self.maybe_send_announcement_signatures(chan)
        util.trigger_callback('channel', self.lnworker.wallet, chan)
        if chan.get_state() == ChannelState.SHUTDOWN:
            await self.send_shutdown(chan)

    def send_channel_ready(self, chan: Channel):
        if False:
            return 10
        assert chan.is_funded()
        if chan.sent_channel_ready:
            return
        channel_id = chan.channel_id
        per_commitment_secret_index = RevocationStore.START_INDEX - 1
        second_per_commitment_point = secret_to_pubkey(int.from_bytes(get_per_commitment_secret_from_seed(chan.config[LOCAL].per_commitment_secret_seed, per_commitment_secret_index), 'big'))
        channel_ready_tlvs = {}
        if self.features.supports(LnFeatures.OPTION_SCID_ALIAS_OPT):
            channel_ready_tlvs['short_channel_id'] = {'alias': chan.get_local_scid_alias(create_new_if_needed=True)}
        self.send_message('channel_ready', channel_id=channel_id, second_per_commitment_point=second_per_commitment_point, channel_ready_tlvs=channel_ready_tlvs)
        chan.sent_channel_ready = True
        self.maybe_mark_open(chan)

    def on_channel_ready(self, chan: Channel, payload):
        if False:
            print('Hello World!')
        self.logger.info(f'on_channel_ready. channel: {chan.channel_id.hex()}')
        scid_alias = payload.get('channel_ready_tlvs', {}).get('short_channel_id', {}).get('alias')
        if scid_alias:
            chan.save_remote_scid_alias(scid_alias)
        if not chan.config[LOCAL].funding_locked_received:
            their_next_point = payload['second_per_commitment_point']
            chan.config[REMOTE].next_per_commitment_point = their_next_point
            chan.config[LOCAL].funding_locked_received = True
            self.lnworker.save_channel(chan)
        self.maybe_mark_open(chan)

    def send_node_announcement(self, alias: str):
        if False:
            i = 10
            return i + 15
        timestamp = int(time.time())
        node_id = privkey_to_pubkey(self.privkey)
        features = self.features.for_node_announcement()
        b = int.bit_length(features)
        flen = b // 8 + int(bool(b % 8))
        rgb_color = bytes.fromhex('000000')
        alias = bytes(alias, 'utf8')
        alias += bytes(32 - len(alias))
        addresses = b''
        raw_msg = encode_msg('node_announcement', flen=flen, features=features, timestamp=timestamp, rgb_color=rgb_color, node_id=node_id, alias=alias, addrlen=len(addresses), addresses=addresses)
        h = sha256d(raw_msg[64 + 2:])
        signature = ecc.ECPrivkey(self.privkey).sign(h, sig_string_from_r_and_s)
        (message_type, payload) = decode_msg(raw_msg)
        payload['signature'] = signature
        raw_msg = encode_msg(message_type, **payload)
        self.transport.send_bytes(raw_msg)

    def maybe_send_channel_announcement(self, chan: Channel):
        if False:
            for i in range(10):
                print('nop')
        node_sigs = [chan.config[REMOTE].announcement_node_sig, chan.config[LOCAL].announcement_node_sig]
        bitcoin_sigs = [chan.config[REMOTE].announcement_bitcoin_sig, chan.config[LOCAL].announcement_bitcoin_sig]
        if not bitcoin_sigs[0] or not bitcoin_sigs[1]:
            return
        (raw_msg, is_reverse) = chan.construct_channel_announcement_without_sigs()
        if is_reverse:
            node_sigs.reverse()
            bitcoin_sigs.reverse()
        (message_type, payload) = decode_msg(raw_msg)
        payload['node_signature_1'] = node_sigs[0]
        payload['node_signature_2'] = node_sigs[1]
        payload['bitcoin_signature_1'] = bitcoin_sigs[0]
        payload['bitcoin_signature_2'] = bitcoin_sigs[1]
        raw_msg = encode_msg(message_type, **payload)
        self.transport.send_bytes(raw_msg)

    def maybe_mark_open(self, chan: Channel):
        if False:
            for i in range(10):
                print('nop')
        if not chan.sent_channel_ready:
            return
        if not chan.config[LOCAL].funding_locked_received:
            return
        self.mark_open(chan)

    def mark_open(self, chan: Channel):
        if False:
            print('Hello World!')
        assert chan.is_funded()
        old_state = chan.get_state()
        if old_state == ChannelState.OPEN:
            return
        if old_state != ChannelState.FUNDED:
            self.logger.info(f'cannot mark open ({chan.get_id_for_log()}), current state: {repr(old_state)}')
            return
        assert chan.config[LOCAL].funding_locked_received
        chan.set_state(ChannelState.OPEN)
        util.trigger_callback('channel', self.lnworker.wallet, chan)
        pending_channel_update = self.orphan_channel_updates.get(chan.short_channel_id)
        if pending_channel_update:
            chan.set_remote_update(pending_channel_update)
        self.logger.info(f'CHANNEL OPENING COMPLETED ({chan.get_id_for_log()})')
        forwarding_enabled = self.network.config.EXPERIMENTAL_LN_FORWARD_PAYMENTS
        if forwarding_enabled and chan.short_channel_id:
            self.logger.info(f'sending channel update for outgoing edge ({chan.get_id_for_log()})')
            chan_upd = chan.get_outgoing_gossip_channel_update()
            self.transport.send_bytes(chan_upd)

    def maybe_send_announcement_signatures(self, chan: Channel, is_reply=False):
        if False:
            return 10
        if not chan.is_public():
            return
        if chan.sent_announcement_signatures:
            return
        if not is_reply and chan.config[REMOTE].announcement_node_sig:
            return
        h = chan.get_channel_announcement_hash()
        bitcoin_signature = ecc.ECPrivkey(chan.config[LOCAL].multisig_key.privkey).sign(h, sig_string_from_r_and_s)
        node_signature = ecc.ECPrivkey(self.privkey).sign(h, sig_string_from_r_and_s)
        self.send_message('announcement_signatures', channel_id=chan.channel_id, short_channel_id=chan.short_channel_id, node_signature=node_signature, bitcoin_signature=bitcoin_signature)
        chan.config[LOCAL].announcement_node_sig = node_signature
        chan.config[LOCAL].announcement_bitcoin_sig = bitcoin_signature
        self.lnworker.save_channel(chan)
        chan.sent_announcement_signatures = True

    def on_update_fail_htlc(self, chan: Channel, payload):
        if False:
            return 10
        htlc_id = payload['id']
        reason = payload['reason']
        self.logger.info(f'on_update_fail_htlc. chan {chan.short_channel_id}. htlc_id {htlc_id}')
        chan.receive_fail_htlc(htlc_id, error_bytes=reason)
        self.maybe_send_commitment(chan)

    def maybe_send_commitment(self, chan: Channel) -> bool:
        if False:
            return 10
        assert util.get_running_loop() == util.get_asyncio_loop(), f'this must be run on the asyncio thread!'
        if chan.is_closed():
            return False
        if chan.hm.is_revack_pending(REMOTE):
            return False
        if not chan.has_pending_changes(REMOTE):
            return False
        self.logger.info(f'send_commitment. chan {chan.short_channel_id}. ctn: {chan.get_next_ctn(REMOTE)}.')
        (sig_64, htlc_sigs) = chan.sign_next_commitment()
        self.send_message('commitment_signed', channel_id=chan.channel_id, signature=sig_64, num_htlcs=len(htlc_sigs), htlc_signature=b''.join(htlc_sigs))
        return True

    def create_onion_for_route(self, *, route: 'LNPaymentRoute', amount_msat: int, total_msat: int, payment_hash: bytes, min_final_cltv_delta: int, payment_secret: bytes, trampoline_onion: Optional[OnionPacket]=None):
        if False:
            print('Hello World!')
        route[0].node_features |= self.features
        local_height = self.network.get_local_height()
        final_cltv_abs = local_height + min_final_cltv_delta
        (hops_data, amount_msat, cltv_abs) = calc_hops_data_for_payment(route, amount_msat, final_cltv_abs=final_cltv_abs, total_msat=total_msat, payment_secret=payment_secret)
        num_hops = len(hops_data)
        self.logger.info(f'lnpeer.pay len(route)={len(route)}')
        for i in range(len(route)):
            self.logger.info(f'  {i}: edge={route[i].short_channel_id} hop_data={hops_data[i]!r}')
        assert final_cltv_abs <= cltv_abs, (final_cltv_abs, cltv_abs)
        session_key = os.urandom(32)
        if trampoline_onion:
            self.logger.info(f'adding trampoline onion to final payload')
            trampoline_payload = hops_data[num_hops - 2].payload
            trampoline_payload['trampoline_onion_packet'] = {'version': trampoline_onion.version, 'public_key': trampoline_onion.public_key, 'hops_data': trampoline_onion.hops_data, 'hmac': trampoline_onion.hmac}
            if (t_hops_data := trampoline_onion._debug_hops_data):
                t_route = trampoline_onion._debug_route
                assert t_route is not None
                self.logger.info(f'lnpeer.pay len(t_route)={len(t_route)}')
                for i in range(len(t_route)):
                    self.logger.info(f'  {i}: t_node={t_route[i].end_node.hex()} hop_data={t_hops_data[i]!r}')
        payment_path_pubkeys = [x.node_id for x in route]
        onion = new_onion_packet(payment_path_pubkeys, session_key, hops_data, associated_data=payment_hash)
        self.logger.info(f'starting payment. len(route)={len(hops_data)}.')
        if cltv_abs > local_height + lnutil.NBLOCK_CLTV_DELTA_TOO_FAR_INTO_FUTURE:
            raise PaymentFailure(f'htlc expiry too far into future. (in {cltv_abs - local_height} blocks)')
        return (onion, amount_msat, cltv_abs, session_key)

    def send_htlc(self, *, chan: Channel, payment_hash: bytes, amount_msat: int, cltv_abs: int, onion: OnionPacket, session_key: Optional[bytes]=None) -> UpdateAddHtlc:
        if False:
            for i in range(10):
                print('nop')
        htlc = UpdateAddHtlc(amount_msat=amount_msat, payment_hash=payment_hash, cltv_abs=cltv_abs, timestamp=int(time.time()))
        htlc = chan.add_htlc(htlc)
        if session_key:
            chan.set_onion_key(htlc.htlc_id, session_key)
        self.logger.info(f'starting payment. htlc: {htlc}')
        self.send_message('update_add_htlc', channel_id=chan.channel_id, id=htlc.htlc_id, cltv_expiry=htlc.cltv_abs, amount_msat=htlc.amount_msat, payment_hash=htlc.payment_hash, onion_routing_packet=onion.to_bytes())
        self.maybe_send_commitment(chan)
        return htlc

    def pay(self, *, route: 'LNPaymentRoute', chan: Channel, amount_msat: int, total_msat: int, payment_hash: bytes, min_final_cltv_delta: int, payment_secret: bytes, trampoline_onion: Optional[OnionPacket]=None) -> UpdateAddHtlc:
        if False:
            while True:
                i = 10
        assert amount_msat > 0, 'amount_msat is not greater zero'
        assert len(route) > 0
        if not chan.can_send_update_add_htlc():
            raise PaymentFailure('Channel cannot send update_add_htlc')
        (onion, amount_msat, cltv_abs, session_key) = self.create_onion_for_route(route=route, amount_msat=amount_msat, total_msat=total_msat, payment_hash=payment_hash, min_final_cltv_delta=min_final_cltv_delta, payment_secret=payment_secret, trampoline_onion=trampoline_onion)
        htlc = self.send_htlc(chan=chan, payment_hash=payment_hash, amount_msat=amount_msat, cltv_abs=cltv_abs, onion=onion, session_key=session_key)
        return htlc

    def send_revoke_and_ack(self, chan: Channel):
        if False:
            print('Hello World!')
        if chan.is_closed():
            return
        self.logger.info(f'send_revoke_and_ack. chan {chan.short_channel_id}. ctn: {chan.get_oldest_unrevoked_ctn(LOCAL)}')
        rev = chan.revoke_current_commitment()
        self.lnworker.save_channel(chan)
        self.send_message('revoke_and_ack', channel_id=chan.channel_id, per_commitment_secret=rev.per_commitment_secret, next_per_commitment_point=rev.next_per_commitment_point)
        self.maybe_send_commitment(chan)

    def on_commitment_signed(self, chan: Channel, payload):
        if False:
            for i in range(10):
                print('nop')
        if chan.peer_state == PeerState.BAD:
            return
        self.logger.info(f'on_commitment_signed. chan {chan.short_channel_id}. ctn: {chan.get_next_ctn(LOCAL)}.')
        if not chan.has_pending_changes(LOCAL):
            raise RemoteMisbehaving('received commitment_signed without pending changes')
        if chan.hm.is_revack_pending(LOCAL):
            raise RemoteMisbehaving('received commitment_signed before we revoked previous ctx')
        data = payload['htlc_signature']
        htlc_sigs = list(chunks(data, 64))
        chan.receive_new_commitment(payload['signature'], htlc_sigs)
        self.send_revoke_and_ack(chan)
        self.received_commitsig_event.set()
        self.received_commitsig_event.clear()

    def on_update_fulfill_htlc(self, chan: Channel, payload):
        if False:
            return 10
        preimage = payload['payment_preimage']
        payment_hash = sha256(preimage)
        htlc_id = payload['id']
        self.logger.info(f'on_update_fulfill_htlc. chan {chan.short_channel_id}. htlc_id {htlc_id}')
        chan.receive_htlc_settle(preimage, htlc_id)
        self.lnworker.save_preimage(payment_hash, preimage)
        self.maybe_send_commitment(chan)

    def on_update_fail_malformed_htlc(self, chan: Channel, payload):
        if False:
            print('Hello World!')
        htlc_id = payload['id']
        failure_code = payload['failure_code']
        self.logger.info(f'on_update_fail_malformed_htlc. chan {chan.get_id_for_log()}. htlc_id {htlc_id}. failure_code={failure_code}')
        if failure_code & OnionFailureCodeMetaFlag.BADONION == 0:
            self.schedule_force_closing(chan.channel_id)
            raise RemoteMisbehaving(f'received update_fail_malformed_htlc with unexpected failure code: {failure_code}')
        reason = OnionRoutingFailure(code=failure_code, data=payload['sha256_of_onion'])
        chan.receive_fail_htlc(htlc_id, error_bytes=None, reason=reason)
        self.maybe_send_commitment(chan)

    def on_update_add_htlc(self, chan: Channel, payload):
        if False:
            while True:
                i = 10
        payment_hash = payload['payment_hash']
        htlc_id = payload['id']
        cltv_abs = payload['cltv_expiry']
        amount_msat_htlc = payload['amount_msat']
        onion_packet = payload['onion_routing_packet']
        htlc = UpdateAddHtlc(amount_msat=amount_msat_htlc, payment_hash=payment_hash, cltv_abs=cltv_abs, timestamp=int(time.time()), htlc_id=htlc_id)
        self.logger.info(f'on_update_add_htlc. chan {chan.short_channel_id}. htlc={str(htlc)}')
        if chan.get_state() != ChannelState.OPEN:
            raise RemoteMisbehaving(f'received update_add_htlc while chan.get_state() != OPEN. state was {chan.get_state()!r}')
        if cltv_abs > bitcoin.NLOCKTIME_BLOCKHEIGHT_MAX:
            self.schedule_force_closing(chan.channel_id)
            raise RemoteMisbehaving(f'received update_add_htlc with cltv_abs={cltv_abs!r} > BLOCKHEIGHT_MAX')
        chan.receive_htlc(htlc, onion_packet)
        util.trigger_callback('htlc_added', chan, htlc, RECEIVED)

    def maybe_forward_htlc(self, *, incoming_chan: Channel, htlc: UpdateAddHtlc, processed_onion: ProcessedOnionPacket) -> Tuple[bytes, int]:
        if False:
            for i in range(10):
                print('nop')

        def log_fail_reason(reason: str):
            if False:
                return 10
            self.logger.debug(f'maybe_forward_htlc. will FAIL HTLC: inc_chan={incoming_chan.get_id_for_log()}. {reason}. inc_htlc={str(htlc)}. onion_payload={processed_onion.hop_data.payload}')
        forwarding_enabled = self.network.config.EXPERIMENTAL_LN_FORWARD_PAYMENTS
        if not forwarding_enabled:
            log_fail_reason('forwarding is disabled')
            raise OnionRoutingFailure(code=OnionFailureCode.PERMANENT_CHANNEL_FAILURE, data=b'')
        chain = self.network.blockchain()
        if chain.is_tip_stale():
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_NODE_FAILURE, data=b'')
        try:
            _next_chan_scid = processed_onion.hop_data.payload['short_channel_id']['short_channel_id']
            next_chan_scid = ShortChannelID(_next_chan_scid)
        except Exception:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        try:
            next_amount_msat_htlc = processed_onion.hop_data.payload['amt_to_forward']['amt_to_forward']
        except Exception:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        try:
            next_cltv_abs = processed_onion.hop_data.payload['outgoing_cltv_value']['outgoing_cltv_value']
        except Exception:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        next_chan = self.lnworker.get_channel_by_short_id(next_chan_scid)
        if self.lnworker.features.supports(LnFeatures.OPTION_ZEROCONF_OPT):
            next_peer = self.lnworker.get_peer_by_scid_alias(next_chan_scid)
        else:
            next_peer = None
        if not next_chan and next_peer and next_peer.accepts_zeroconf():
            for next_chan in next_peer.channels.values():
                if next_chan.can_pay(next_amount_msat_htlc):
                    break
            else:

                async def wrapped_callback():
                    coro = self.lnworker.open_channel_just_in_time(next_peer, next_amount_msat_htlc, next_cltv_abs, htlc.payment_hash, processed_onion.next_packet)
                    try:
                        await coro
                    except OnionRoutingFailure as e:
                        self.jit_failures[next_chan_scid.hex()] = e
                asyncio.ensure_future(wrapped_callback())
                return (next_chan_scid, -1)
        local_height = chain.height()
        if next_chan is None:
            log_fail_reason(f'cannot find next_chan {next_chan_scid}')
            raise OnionRoutingFailure(code=OnionFailureCode.UNKNOWN_NEXT_PEER, data=b'')
        outgoing_chan_upd = next_chan.get_outgoing_gossip_channel_update(scid=next_chan_scid)[2:]
        outgoing_chan_upd_len = len(outgoing_chan_upd).to_bytes(2, byteorder='big')
        outgoing_chan_upd_message = outgoing_chan_upd_len + outgoing_chan_upd
        if not next_chan.can_send_update_add_htlc():
            log_fail_reason(f'next_chan {next_chan.get_id_for_log()} cannot send ctx updates. chan state {next_chan.get_state()!r}, peer state: {next_chan.peer_state!r}')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_CHANNEL_FAILURE, data=outgoing_chan_upd_message)
        if not next_chan.can_pay(next_amount_msat_htlc):
            log_fail_reason(f'transient error (likely due to insufficient funds): not next_chan.can_pay(amt)')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_CHANNEL_FAILURE, data=outgoing_chan_upd_message)
        if htlc.cltv_abs - next_cltv_abs < next_chan.forwarding_cltv_delta:
            log_fail_reason(f'INCORRECT_CLTV_EXPIRY. htlc.cltv_abs={htlc.cltv_abs!r} - next_cltv_abs={next_cltv_abs!r} < next_chan.forwarding_cltv_delta={next_chan.forwarding_cltv_delta!r}')
            data = htlc.cltv_abs.to_bytes(4, byteorder='big') + outgoing_chan_upd_message
            raise OnionRoutingFailure(code=OnionFailureCode.INCORRECT_CLTV_EXPIRY, data=data)
        if htlc.cltv_abs - lnutil.MIN_FINAL_CLTV_DELTA_ACCEPTED <= local_height or next_cltv_abs <= local_height:
            raise OnionRoutingFailure(code=OnionFailureCode.EXPIRY_TOO_SOON, data=outgoing_chan_upd_message)
        if max(htlc.cltv_abs, next_cltv_abs) > local_height + lnutil.NBLOCK_CLTV_DELTA_TOO_FAR_INTO_FUTURE:
            raise OnionRoutingFailure(code=OnionFailureCode.EXPIRY_TOO_FAR, data=b'')
        forwarding_fees = fee_for_edge_msat(forwarded_amount_msat=next_amount_msat_htlc, fee_base_msat=next_chan.forwarding_fee_base_msat, fee_proportional_millionths=next_chan.forwarding_fee_proportional_millionths)
        if htlc.amount_msat - next_amount_msat_htlc < forwarding_fees:
            data = next_amount_msat_htlc.to_bytes(8, byteorder='big') + outgoing_chan_upd_message
            raise OnionRoutingFailure(code=OnionFailureCode.FEE_INSUFFICIENT, data=data)
        if self._maybe_refuse_to_forward_htlc_that_corresponds_to_payreq_we_created(htlc.payment_hash):
            log_fail_reason(f'RHASH corresponds to payreq we created')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_NODE_FAILURE, data=b'')
        self.logger.info(f'maybe_forward_htlc. will forward HTLC: inc_chan={incoming_chan.short_channel_id}. inc_htlc={str(htlc)}. next_chan={next_chan.get_id_for_log()}.')
        next_peer = self.lnworker.peers.get(next_chan.node_id)
        if next_peer is None:
            log_fail_reason(f'next_peer offline ({next_chan.node_id.hex()})')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_CHANNEL_FAILURE, data=outgoing_chan_upd_message)
        try:
            next_htlc = next_peer.send_htlc(chan=next_chan, payment_hash=htlc.payment_hash, amount_msat=next_amount_msat_htlc, cltv_abs=next_cltv_abs, onion=processed_onion.next_packet)
        except BaseException as e:
            log_fail_reason(f'error sending message to next_peer={next_chan.node_id.hex()}')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_CHANNEL_FAILURE, data=outgoing_chan_upd_message)
        return (next_chan_scid, next_htlc.htlc_id)

    @log_exceptions
    async def maybe_forward_trampoline(self, *, payment_hash: bytes, inc_cltv_abs: int, outer_onion: ProcessedOnionPacket, trampoline_onion: ProcessedOnionPacket):
        forwarding_enabled = self.network.config.EXPERIMENTAL_LN_FORWARD_PAYMENTS
        forwarding_trampoline_enabled = self.network.config.EXPERIMENTAL_LN_FORWARD_TRAMPOLINE_PAYMENTS
        if not (forwarding_enabled and forwarding_trampoline_enabled):
            self.logger.info(f'trampoline forwarding is disabled. failing htlc.')
            raise OnionRoutingFailure(code=OnionFailureCode.PERMANENT_CHANNEL_FAILURE, data=b'')
        payload = trampoline_onion.hop_data.payload
        payment_data = payload.get('payment_data')
        try:
            payment_secret = payment_data['payment_secret'] if payment_data else os.urandom(32)
            outgoing_node_id = payload['outgoing_node_id']['outgoing_node_id']
            amt_to_forward = payload['amt_to_forward']['amt_to_forward']
            out_cltv_abs = payload['outgoing_cltv_value']['outgoing_cltv_value']
            if 'invoice_features' in payload:
                self.logger.info('forward_trampoline: legacy')
                next_trampoline_onion = None
                invoice_features = payload['invoice_features']['invoice_features']
                invoice_routing_info = payload['invoice_routing_info']['invoice_routing_info']
                r_tags = decode_routing_info(invoice_routing_info)
                self.logger.info(f'r_tags {r_tags}')
            else:
                self.logger.info('forward_trampoline: end-to-end')
                invoice_features = LnFeatures.BASIC_MPP_OPT
                next_trampoline_onion = trampoline_onion.next_packet
                r_tags = []
        except Exception as e:
            self.logger.exception('')
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        if self._maybe_refuse_to_forward_htlc_that_corresponds_to_payreq_we_created(payment_hash):
            self.logger.debug(f'maybe_forward_trampoline. will FAIL HTLC(s). RHASH corresponds to payreq we created. payment_hash.hex()={payment_hash.hex()!r}')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_NODE_FAILURE, data=b'')
        total_msat = outer_onion.hop_data.payload['payment_data']['total_msat']
        budget = PaymentFeeBudget(fee_msat=total_msat - amt_to_forward, cltv=inc_cltv_abs - out_cltv_abs)
        self.logger.info(f'trampoline forwarding. budget={budget}')
        self.logger.info(f'trampoline forwarding. inc_cltv_abs={inc_cltv_abs!r}, out_cltv_abs={out_cltv_abs!r}')
        local_height_of_onion_creator = self.network.get_local_height() - 1
        cltv_budget_for_rest_of_route = out_cltv_abs - local_height_of_onion_creator
        if budget.fee_msat < 1000:
            raise OnionRoutingFailure(code=OnionFailureCode.TRAMPOLINE_FEE_INSUFFICIENT, data=b'')
        if budget.cltv < 576:
            raise OnionRoutingFailure(code=OnionFailureCode.TRAMPOLINE_EXPIRY_TOO_SOON, data=b'')
        next_peer = self.lnworker.peers.get(outgoing_node_id)
        if next_peer and next_peer.accepts_zeroconf():
            self.logger.info(f'JIT: found next_peer')
            for next_chan in next_peer.channels.values():
                if next_chan.can_pay(amt_to_forward):
                    self.logger.info(f'jit: next_chan can pay')
                    break
            else:
                scid_alias = self.lnworker._scid_alias_of_node(next_peer.pubkey)
                route = [RouteEdge(start_node=next_peer.pubkey, end_node=outgoing_node_id, short_channel_id=scid_alias, fee_base_msat=0, fee_proportional_millionths=0, cltv_delta=144, node_features=0)]
                (next_onion, amount_msat, cltv_abs, session_key) = self.create_onion_for_route(route=route, amount_msat=amt_to_forward, total_msat=amt_to_forward, payment_hash=payment_hash, min_final_cltv_delta=cltv_budget_for_rest_of_route, payment_secret=payment_secret, trampoline_onion=next_trampoline_onion)
                await self.lnworker.open_channel_just_in_time(next_peer, amt_to_forward, cltv_abs, payment_hash, next_onion)
                return
        try:
            await self.lnworker.pay_to_node(node_pubkey=outgoing_node_id, payment_hash=payment_hash, payment_secret=payment_secret, amount_to_pay=amt_to_forward, min_final_cltv_delta=cltv_budget_for_rest_of_route, r_tags=r_tags, invoice_features=invoice_features, fwd_trampoline_onion=next_trampoline_onion, budget=budget, attempts=1)
        except OnionRoutingFailure as e:
            raise
        except PaymentFailure as e:
            self.logger.debug(f'maybe_forward_trampoline. PaymentFailure for payment_hash.hex()={payment_hash.hex()!r}, payment_secret.hex()={payment_secret.hex()!r}: {e!r}')
            raise OnionRoutingFailure(code=OnionFailureCode.UNKNOWN_NEXT_PEER, data=b'')

    def _maybe_refuse_to_forward_htlc_that_corresponds_to_payreq_we_created(self, payment_hash: bytes) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the HTLC should be failed.\n        We must not forward HTLCs with a matching payment_hash to a payment request we created.\n        Example attack:\n        - Bob creates payment request with HASH1, for 1 BTC; and gives the payreq to Alice\n        - Alice sends htlc A->B->C, for 1 sat, with HASH1\n        - Bob must not release the preimage of HASH1\n        '
        payment_info = self.lnworker.get_payment_info(payment_hash)
        is_our_payreq = payment_info and payment_info.direction == RECEIVED
        return bool(is_our_payreq and self.lnworker.get_preimage(payment_hash))

    def maybe_fulfill_htlc(self, *, chan: Channel, htlc: UpdateAddHtlc, processed_onion: ProcessedOnionPacket, onion_packet_bytes: bytes) -> Tuple[Optional[bytes], Optional[Callable]]:
        if False:
            for i in range(10):
                print('nop')
        'As a final recipient of an HTLC, decide if we should fulfill it.\n        Return (preimage, forwarding_callback) with at most a single element not None\n        '

        def log_fail_reason(reason: str):
            if False:
                while True:
                    i = 10
            self.logger.info(f'maybe_fulfill_htlc. will FAIL HTLC: chan {chan.short_channel_id}. {reason}. htlc={str(htlc)}. onion_payload={processed_onion.hop_data.payload}')
        try:
            amt_to_forward = processed_onion.hop_data.payload['amt_to_forward']['amt_to_forward']
        except Exception:
            log_fail_reason(f"'amt_to_forward' missing from onion")
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        chain = self.network.blockchain()
        if chain.is_tip_stale():
            log_fail_reason(f'our chain tip is stale')
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_NODE_FAILURE, data=b'')
        local_height = chain.height()
        exc_incorrect_or_unknown_pd = OnionRoutingFailure(code=OnionFailureCode.INCORRECT_OR_UNKNOWN_PAYMENT_DETAILS, data=amt_to_forward.to_bytes(8, byteorder='big') + local_height.to_bytes(4, byteorder='big'))
        if local_height + MIN_FINAL_CLTV_DELTA_ACCEPTED > htlc.cltv_abs:
            log_fail_reason(f'htlc.cltv_abs is unreasonably close')
            raise exc_incorrect_or_unknown_pd
        try:
            cltv_abs_from_onion = processed_onion.hop_data.payload['outgoing_cltv_value']['outgoing_cltv_value']
        except Exception:
            log_fail_reason(f"'outgoing_cltv_value' missing from onion")
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_PAYLOAD, data=b'\x00\x00\x00')
        if cltv_abs_from_onion > htlc.cltv_abs:
            log_fail_reason(f'cltv_abs_from_onion != htlc.cltv_abs')
            raise OnionRoutingFailure(code=OnionFailureCode.FINAL_INCORRECT_CLTV_EXPIRY, data=htlc.cltv_abs.to_bytes(4, byteorder='big'))
        try:
            total_msat = processed_onion.hop_data.payload['payment_data']['total_msat']
        except Exception:
            log_fail_reason(f"'total_msat' missing from onion")
            raise exc_incorrect_or_unknown_pd
        if chan.opening_fee:
            channel_opening_fee = chan.opening_fee['channel_opening_fee']
            total_msat -= channel_opening_fee
            amt_to_forward -= channel_opening_fee
        else:
            channel_opening_fee = 0
        if amt_to_forward > htlc.amount_msat:
            log_fail_reason(f'amt_to_forward != htlc.amount_msat')
            raise OnionRoutingFailure(code=OnionFailureCode.FINAL_INCORRECT_HTLC_AMOUNT, data=htlc.amount_msat.to_bytes(8, byteorder='big'))
        try:
            payment_secret_from_onion = processed_onion.hop_data.payload['payment_data']['payment_secret']
        except Exception:
            log_fail_reason(f"'payment_secret' missing from onion")
            raise exc_incorrect_or_unknown_pd
        from .lnworker import RecvMPPResolution
        mpp_resolution = self.lnworker.check_mpp_status(payment_secret=payment_secret_from_onion, short_channel_id=chan.short_channel_id, htlc=htlc, expected_msat=total_msat)
        if mpp_resolution == RecvMPPResolution.WAITING:
            return (None, None)
        elif mpp_resolution == RecvMPPResolution.EXPIRED:
            log_fail_reason(f'MPP_TIMEOUT')
            raise OnionRoutingFailure(code=OnionFailureCode.MPP_TIMEOUT, data=b'')
        elif mpp_resolution == RecvMPPResolution.FAILED:
            log_fail_reason(f'mpp_resolution is FAILED')
            raise exc_incorrect_or_unknown_pd
        elif mpp_resolution == RecvMPPResolution.ACCEPTED:
            pass
        else:
            raise Exception(f'unexpected mpp_resolution={mpp_resolution!r}')
        payment_hash = htlc.payment_hash
        if processed_onion.trampoline_onion_packet:
            trampoline_onion = self.process_onion_packet(processed_onion.trampoline_onion_packet, payment_hash=payment_hash, onion_packet_bytes=onion_packet_bytes, is_trampoline=True)
            if trampoline_onion.are_we_final:
                (preimage, cb) = self.maybe_fulfill_htlc(chan=chan, htlc=htlc, processed_onion=trampoline_onion, onion_packet_bytes=onion_packet_bytes)
                if preimage:
                    return (preimage, None)
                else:
                    return (None, cb)
            else:
                callback = lambda : self.maybe_forward_trampoline(payment_hash=payment_hash, inc_cltv_abs=htlc.cltv_abs, outer_onion=processed_onion, trampoline_onion=trampoline_onion)
                return (None, callback)
        info = self.lnworker.get_payment_info(payment_hash)
        if info is None:
            log_fail_reason(f'no payment_info found for RHASH {htlc.payment_hash.hex()}')
            raise exc_incorrect_or_unknown_pd
        preimage = self.lnworker.get_preimage(payment_hash)
        expected_payment_secrets = [self.lnworker.get_payment_secret(htlc.payment_hash)]
        if preimage:
            expected_payment_secrets.append(derive_payment_secret_from_payment_preimage(preimage))
        if payment_secret_from_onion not in expected_payment_secrets:
            log_fail_reason(f'incorrect payment secret {payment_secret_from_onion.hex()} != {expected_payment_secrets[0].hex()}')
            raise exc_incorrect_or_unknown_pd
        invoice_msat = info.amount_msat
        if channel_opening_fee:
            invoice_msat -= channel_opening_fee
        if not (invoice_msat is None or invoice_msat <= total_msat <= 2 * invoice_msat):
            log_fail_reason(f'total_msat={total_msat} too different from invoice_msat={invoice_msat}')
            raise exc_incorrect_or_unknown_pd
        hold_invoice_callback = self.lnworker.hold_invoice_callbacks.get(payment_hash)
        if hold_invoice_callback and (not preimage):
            return (None, lambda : hold_invoice_callback(payment_hash))
        if not preimage:
            self.logger.info(f'missing preimage and no hold invoice callback {payment_hash.hex()}')
            raise exc_incorrect_or_unknown_pd
        chan.opening_fee = None
        self.logger.info(f'maybe_fulfill_htlc. will FULFILL HTLC: chan {chan.short_channel_id}. htlc={str(htlc)}')
        return (preimage, None)

    def fulfill_htlc(self, chan: Channel, htlc_id: int, preimage: bytes):
        if False:
            return 10
        self.logger.info(f'_fulfill_htlc. chan {chan.short_channel_id}. htlc_id {htlc_id}')
        assert chan.can_send_ctx_updates(), f'cannot send updates: {chan.short_channel_id}'
        assert chan.hm.is_htlc_irrevocably_added_yet(htlc_proposer=REMOTE, htlc_id=htlc_id)
        self.received_htlcs_pending_removal.add((chan, htlc_id))
        chan.settle_htlc(preimage, htlc_id)
        self.send_message('update_fulfill_htlc', channel_id=chan.channel_id, id=htlc_id, payment_preimage=preimage)

    def fail_htlc(self, *, chan: Channel, htlc_id: int, error_bytes: bytes):
        if False:
            print('Hello World!')
        self.logger.info(f'fail_htlc. chan {chan.short_channel_id}. htlc_id {htlc_id}.')
        assert chan.can_send_ctx_updates(), f'cannot send updates: {chan.short_channel_id}'
        self.received_htlcs_pending_removal.add((chan, htlc_id))
        chan.fail_htlc(htlc_id)
        self.send_message('update_fail_htlc', channel_id=chan.channel_id, id=htlc_id, len=len(error_bytes), reason=error_bytes)

    def fail_malformed_htlc(self, *, chan: Channel, htlc_id: int, reason: OnionRoutingFailure):
        if False:
            print('Hello World!')
        self.logger.info(f'fail_malformed_htlc. chan {chan.short_channel_id}. htlc_id {htlc_id}.')
        assert chan.can_send_ctx_updates(), f'cannot send updates: {chan.short_channel_id}'
        if not (reason.code & OnionFailureCodeMetaFlag.BADONION and len(reason.data) == 32):
            raise Exception(f"unexpected reason when sending 'update_fail_malformed_htlc': {reason!r}")
        self.received_htlcs_pending_removal.add((chan, htlc_id))
        chan.fail_htlc(htlc_id)
        self.send_message('update_fail_malformed_htlc', channel_id=chan.channel_id, id=htlc_id, sha256_of_onion=reason.data, failure_code=reason.code)

    def on_revoke_and_ack(self, chan: Channel, payload):
        if False:
            i = 10
            return i + 15
        if chan.peer_state == PeerState.BAD:
            return
        self.logger.info(f'on_revoke_and_ack. chan {chan.short_channel_id}. ctn: {chan.get_oldest_unrevoked_ctn(REMOTE)}')
        rev = RevokeAndAck(payload['per_commitment_secret'], payload['next_per_commitment_point'])
        chan.receive_revocation(rev)
        self.lnworker.save_channel(chan)
        self.maybe_send_commitment(chan)
        self._received_revack_event.set()
        self._received_revack_event.clear()

    def on_update_fee(self, chan: Channel, payload):
        if False:
            i = 10
            return i + 15
        feerate = payload['feerate_per_kw']
        chan.update_fee(feerate, False)

    def maybe_update_fee(self, chan: Channel):
        if False:
            return 10
        '\n        called when our fee estimates change\n        '
        if not chan.can_send_ctx_updates():
            return
        feerate_per_kw = self.lnworker.current_feerate_per_kw()
        if not chan.constraints.is_initiator:
            if constants.net is not constants.BitcoinRegtest:
                chan_feerate = chan.get_latest_feerate(LOCAL)
                ratio = chan_feerate / feerate_per_kw
                if ratio < 0.5:
                    self.logger.warning(f'({chan.get_id_for_log()}) feerate is {chan_feerate} sat/kw, current recommended feerate is {feerate_per_kw} sat/kw, consider force closing!')
            return
        chan_fee = chan.get_next_feerate(REMOTE)
        if feerate_per_kw < chan_fee / 2:
            self.logger.info('FEES HAVE FALLEN')
        elif feerate_per_kw > chan_fee * 2:
            self.logger.info('FEES HAVE RISEN')
        elif chan.get_latest_ctn(REMOTE) == 0:
            self.logger.info('updating fee to bump remote ctn')
            if feerate_per_kw == chan_fee:
                feerate_per_kw += 1
        else:
            return
        self.logger.info(f'(chan: {chan.get_id_for_log()}) current pending feerate {chan_fee}. new feerate {feerate_per_kw}')
        chan.update_fee(feerate_per_kw, True)
        self.send_message('update_fee', channel_id=chan.channel_id, feerate_per_kw=feerate_per_kw)
        self.maybe_send_commitment(chan)

    @log_exceptions
    async def close_channel(self, chan_id: bytes):
        chan = self.channels[chan_id]
        self.shutdown_received[chan_id] = self.asyncio_loop.create_future()
        await self.send_shutdown(chan)
        payload = await self.shutdown_received[chan_id]
        try:
            txid = await self._shutdown(chan, payload, is_local=True)
            self.logger.info(f'({chan.get_id_for_log()}) Channel closed {txid}')
        except asyncio.TimeoutError:
            txid = chan.unconfirmed_closing_txid
            self.logger.info(f'({chan.get_id_for_log()}) did not send closing_signed, {txid}')
            if txid is None:
                raise Exception('The remote peer did not send their final signature. The channel may not have been be closed')
        return txid

    async def on_shutdown(self, chan: Channel, payload):
        their_scriptpubkey = payload['scriptpubkey']
        their_upfront_scriptpubkey = chan.config[REMOTE].upfront_shutdown_script
        if self.is_upfront_shutdown_script() and their_upfront_scriptpubkey:
            if not their_scriptpubkey == their_upfront_scriptpubkey:
                await self.send_warning(chan.channel_id, "remote didn't use upfront shutdown script it commited to in channel opening", close_connection=True)
        elif self.is_shutdown_anysegwit() and match_script_against_template(their_scriptpubkey, transaction.SCRIPTPUBKEY_TEMPLATE_ANYSEGWIT):
            pass
        elif match_script_against_template(their_scriptpubkey, transaction.SCRIPTPUBKEY_TEMPLATE_WITNESS_V0):
            pass
        else:
            await self.send_warning(chan.channel_id, f'scriptpubkey in received shutdown message does not conform to any template: {their_scriptpubkey.hex()}', close_connection=True)
        chan_id = chan.channel_id
        if chan_id in self.shutdown_received:
            self.shutdown_received[chan_id].set_result(payload)
        else:
            chan = self.channels[chan_id]
            await self.send_shutdown(chan)
            txid = await self._shutdown(chan, payload, is_local=False)
            self.logger.info(f'({chan.get_id_for_log()}) Channel closed by remote peer {txid}')

    def can_send_shutdown(self, chan: Channel):
        if False:
            while True:
                i = 10
        if chan.get_state() >= ChannelState.OPENING:
            return True
        if chan.constraints.is_initiator and chan.channel_id in self.funding_created_sent:
            return True
        if not chan.constraints.is_initiator and chan.channel_id in self.funding_signed_sent:
            return True
        return False

    async def send_shutdown(self, chan: Channel):
        if not self.can_send_shutdown(chan):
            raise Exception('cannot send shutdown')
        if chan.config[LOCAL].upfront_shutdown_script:
            scriptpubkey = chan.config[LOCAL].upfront_shutdown_script
        else:
            scriptpubkey = bfh(bitcoin.address_to_script(chan.sweep_address))
        assert scriptpubkey
        chan.set_can_send_ctx_updates(False)
        while chan.has_pending_changes(REMOTE):
            await asyncio.sleep(0.1)
        self.send_message('shutdown', channel_id=chan.channel_id, len=len(scriptpubkey), scriptpubkey=scriptpubkey)
        chan.set_state(ChannelState.SHUTDOWN)
        chan.set_can_send_ctx_updates(True)

    def get_shutdown_fee_range(self, chan, closing_tx, is_local):
        if False:
            i = 10
            return i + 15
        ' return the closing fee and fee range we initially try to enforce '
        config = self.network.config
        our_fee = None
        if config.TEST_SHUTDOWN_FEE:
            our_fee = config.TEST_SHUTDOWN_FEE
        else:
            fee_rate_per_kb = config.eta_target_to_fee(FEE_LN_ETA_TARGET)
            if fee_rate_per_kb is None:
                fee_rate_per_kb = self.network.config.fee_per_kb()
            if fee_rate_per_kb is not None:
                our_fee = fee_rate_per_kb * closing_tx.estimated_size() // 1000
            max_fee = chan.get_latest_fee(LOCAL if is_local else REMOTE)
            if our_fee is None:
                self.logger.warning(f'got no fee estimates for co-op close! falling back to chan.get_latest_fee')
                our_fee = max_fee
            our_fee = min(our_fee, max_fee)
        if config.TEST_SHUTDOWN_LEGACY:
            our_fee_range = None
        elif config.TEST_SHUTDOWN_FEE_RANGE:
            our_fee_range = config.TEST_SHUTDOWN_FEE_RANGE
        else:
            our_fee_range = {'min_fee_satoshis': our_fee // 2, 'max_fee_satoshis': our_fee * 2}
        self.logger.info(f'Our fee range: {our_fee_range} and fee: {our_fee}')
        return (our_fee, our_fee_range)

    @log_exceptions
    async def _shutdown(self, chan: Channel, payload, *, is_local: bool):
        while chan.has_unsettled_htlcs():
            self.logger.info(f'(chan: {chan.short_channel_id}) waiting for htlcs to settle...')
            await asyncio.sleep(1)
        chan.set_can_send_ctx_updates(False)
        their_scriptpubkey = payload['scriptpubkey']
        if chan.config[LOCAL].upfront_shutdown_script:
            our_scriptpubkey = chan.config[LOCAL].upfront_shutdown_script
        else:
            our_scriptpubkey = bfh(bitcoin.address_to_script(chan.sweep_address))
        assert our_scriptpubkey
        (dummy_sig, dummy_tx) = chan.make_closing_tx(our_scriptpubkey, their_scriptpubkey, fee_sat=0)
        our_sig = None
        closing_tx = None
        is_initiator = chan.constraints.is_initiator
        (our_fee, our_fee_range) = self.get_shutdown_fee_range(chan, dummy_tx, is_local)

        def send_closing_signed(our_fee, our_fee_range, drop_remote):
            if False:
                print('Hello World!')
            nonlocal our_sig, closing_tx
            if our_fee_range:
                closing_signed_tlvs = {'fee_range': our_fee_range}
            else:
                closing_signed_tlvs = {}
            (our_sig, closing_tx) = chan.make_closing_tx(our_scriptpubkey, their_scriptpubkey, fee_sat=our_fee, drop_remote=drop_remote)
            self.logger.info(f'Sending fee range: {closing_signed_tlvs} and fee: {our_fee}')
            self.send_message('closing_signed', channel_id=chan.channel_id, fee_satoshis=our_fee, signature=our_sig, closing_signed_tlvs=closing_signed_tlvs)

        def verify_signature(tx, sig):
            if False:
                while True:
                    i = 10
            their_pubkey = chan.config[REMOTE].multisig_key.pubkey
            preimage_hex = tx.serialize_preimage(0)
            pre_hash = sha256d(bfh(preimage_hex))
            return ecc.verify_signature(their_pubkey, sig, pre_hash)

        async def receive_closing_signed():
            nonlocal our_sig, closing_tx
            try:
                cs_payload = await self.wait_for_message('closing_signed', chan.channel_id)
            except asyncio.exceptions.TimeoutError:
                self.schedule_force_closing(chan.channel_id)
                raise Exception('closing_signed not received, force closing.')
            their_fee = cs_payload['fee_satoshis']
            their_fee_range = cs_payload['closing_signed_tlvs'].get('fee_range')
            their_sig = cs_payload['signature']
            (our_sig, closing_tx) = chan.make_closing_tx(our_scriptpubkey, their_scriptpubkey, fee_sat=their_fee, drop_remote=False)
            if verify_signature(closing_tx, their_sig):
                drop_remote = False
            else:
                (our_sig, closing_tx) = chan.make_closing_tx(our_scriptpubkey, their_scriptpubkey, fee_sat=their_fee, drop_remote=True)
                if verify_signature(closing_tx, their_sig):
                    drop_remote = True
                else:
                    raise Exception('failed to verify their signature')
            to_remote_set = closing_tx.get_output_idxs_from_scriptpubkey(their_scriptpubkey.hex())
            if not drop_remote and to_remote_set:
                to_remote_idx = to_remote_set.pop()
                to_remote_amount = closing_tx.outputs()[to_remote_idx].value
                transaction.check_scriptpubkey_template_and_dust(their_scriptpubkey, to_remote_amount)
            return (their_fee, their_fee_range, their_sig, drop_remote)

        def choose_new_fee(our_fee, our_fee_range, their_fee, their_fee_range, their_previous_fee):
            if False:
                while True:
                    i = 10
            assert our_fee != their_fee
            fee_range_sent = our_fee_range and (is_initiator or their_previous_fee is not None)
            if our_fee_range and their_fee_range and (not is_initiator) and (not self.network.config.TEST_SHUTDOWN_FEE_RANGE):
                our_fee_range['max_fee_satoshis'] = max(their_fee_range['max_fee_satoshis'], our_fee_range['max_fee_satoshis'])
                our_fee_range['min_fee_satoshis'] = min(their_fee_range['min_fee_satoshis'], our_fee_range['min_fee_satoshis'])
            if fee_range_sent and our_fee_range['min_fee_satoshis'] <= their_fee <= our_fee_range['max_fee_satoshis']:
                our_fee = their_fee
            elif our_fee_range and their_fee_range:
                overlap_min = max(our_fee_range['min_fee_satoshis'], their_fee_range['min_fee_satoshis'])
                overlap_max = min(our_fee_range['max_fee_satoshis'], their_fee_range['max_fee_satoshis'])
                if overlap_min > overlap_max:
                    self.schedule_force_closing(chan.channel_id)
                    raise Exception('There is no overlap between between their and our fee range.')
                if is_initiator:
                    if not overlap_min <= their_fee <= overlap_max:
                        self.schedule_force_closing(chan.channel_id)
                        raise Exception('Their fee is not in the overlap region, we force closed.')
                    our_fee = their_fee
                else:
                    if fee_range_sent:
                        self.schedule_force_closing(chan.channel_id)
                        raise Exception('Expected the same fee as ours, we force closed.')
                    our_fee = (overlap_min + overlap_max) // 2
            else:
                if their_previous_fee and (not min(our_fee, their_previous_fee) < their_fee < max(our_fee, their_previous_fee)):
                    raise Exception('Their fee is not between our last sent and their last sent fee.')
                if abs(their_fee - our_fee) < 2:
                    our_fee = their_fee
                else:
                    our_fee = (our_fee + their_fee) // 2
            return (our_fee, our_fee_range)
        their_fee = None
        drop_remote = False
        if is_initiator:
            send_closing_signed(our_fee, our_fee_range, drop_remote)
        while True:
            their_previous_fee = their_fee
            (their_fee, their_fee_range, their_sig, drop_remote) = await receive_closing_signed()
            if our_fee == their_fee:
                break
            (our_fee, our_fee_range) = choose_new_fee(our_fee, our_fee_range, their_fee, their_fee_range, their_previous_fee)
            if not is_initiator and our_fee == their_fee:
                break
            send_closing_signed(our_fee, our_fee_range, drop_remote)
            if is_initiator and our_fee == their_fee:
                break
        if not is_initiator:
            send_closing_signed(our_fee, our_fee_range, drop_remote)
        closing_tx.add_signature_to_txin(txin_idx=0, signing_pubkey=chan.config[LOCAL].multisig_key.pubkey.hex(), sig=(der_sig_from_sig_string(our_sig) + Sighash.to_sigbytes(Sighash.ALL)).hex())
        closing_tx.add_signature_to_txin(txin_idx=0, signing_pubkey=chan.config[REMOTE].multisig_key.pubkey.hex(), sig=(der_sig_from_sig_string(their_sig) + Sighash.to_sigbytes(Sighash.ALL)).hex())
        try:
            self.lnworker.wallet.adb.add_transaction(closing_tx)
        except UnrelatedTransactionException:
            pass
        chan.set_state(ChannelState.CLOSING)
        await self.network.try_broadcasting(closing_tx, 'closing')
        return closing_tx.txid()

    async def htlc_switch(self):
        await self.initialized
        while True:
            await self.ping_if_required()
            self._htlc_switch_iterdone_event.set()
            self._htlc_switch_iterdone_event.clear()
            async with ignore_after(0.1):
                async with OldTaskGroup(wait=any) as group:
                    await group.spawn(self._received_revack_event.wait())
                    await group.spawn(self.downstream_htlc_resolved_event.wait())
            self._htlc_switch_iterstart_event.set()
            self._htlc_switch_iterstart_event.clear()
            self._maybe_cleanup_received_htlcs_pending_removal()
            for (chan_id, chan) in self.channels.items():
                if not chan.can_send_ctx_updates():
                    continue
                self.maybe_send_commitment(chan)
                done = set()
                unfulfilled = chan.unfulfilled_htlcs
                for (htlc_id, (local_ctn, remote_ctn, onion_packet_hex, forwarding_info)) in unfulfilled.items():
                    if forwarding_info:
                        forwarding_info = tuple(forwarding_info)
                        self.lnworker.downstream_htlc_to_upstream_peer_map[forwarding_info] = self.pubkey
                    if not chan.hm.is_htlc_irrevocably_added_yet(htlc_proposer=REMOTE, htlc_id=htlc_id):
                        continue
                    htlc = chan.hm.get_htlc_by_id(REMOTE, htlc_id)
                    error_reason = None
                    error_bytes = None
                    preimage = None
                    fw_info = None
                    onion_packet_bytes = bytes.fromhex(onion_packet_hex)
                    onion_packet = None
                    try:
                        onion_packet = OnionPacket.from_bytes(onion_packet_bytes)
                    except OnionRoutingFailure as e:
                        error_reason = e
                    else:
                        try:
                            (preimage, fw_info, error_bytes) = self.process_unfulfilled_htlc(chan=chan, htlc=htlc, forwarding_info=forwarding_info, onion_packet_bytes=onion_packet_bytes, onion_packet=onion_packet)
                        except OnionRoutingFailure as e:
                            error_bytes = construct_onion_error(e, onion_packet.public_key, our_onion_private_key=self.privkey)
                        if error_bytes:
                            error_bytes = obfuscate_onion_error(error_bytes, onion_packet.public_key, our_onion_private_key=self.privkey)
                    if fw_info:
                        unfulfilled[htlc_id] = (local_ctn, remote_ctn, onion_packet_hex, fw_info)
                        self.lnworker.downstream_htlc_to_upstream_peer_map[fw_info] = self.pubkey
                    elif preimage or error_reason or error_bytes:
                        if preimage:
                            self.lnworker.set_request_status(htlc.payment_hash, PR_PAID)
                            if not self.lnworker.enable_htlc_settle:
                                continue
                            self.fulfill_htlc(chan, htlc.htlc_id, preimage)
                        elif error_bytes:
                            self.fail_htlc(chan=chan, htlc_id=htlc.htlc_id, error_bytes=error_bytes)
                        else:
                            self.fail_malformed_htlc(chan=chan, htlc_id=htlc.htlc_id, reason=error_reason)
                        done.add(htlc_id)
                for htlc_id in done:
                    (local_ctn, remote_ctn, onion_packet_hex, forwarding_info) = unfulfilled.pop(htlc_id)
                    if forwarding_info:
                        forwarding_info = tuple(forwarding_info)
                        self.lnworker.downstream_htlc_to_upstream_peer_map.pop(forwarding_info, None)
                self.maybe_send_commitment(chan)

    def _maybe_cleanup_received_htlcs_pending_removal(self) -> None:
        if False:
            i = 10
            return i + 15
        done = set()
        for (chan, htlc_id) in self.received_htlcs_pending_removal:
            if chan.hm.is_htlc_irrevocably_removed_yet(htlc_proposer=REMOTE, htlc_id=htlc_id):
                done.add((chan, htlc_id))
        if done:
            for key in done:
                self.received_htlcs_pending_removal.remove(key)
            self.received_htlc_removed_event.set()
            self.received_htlc_removed_event.clear()

    async def wait_one_htlc_switch_iteration(self) -> None:
        """Waits until the HTLC switch does a full iteration or the peer disconnects,
        whichever happens first.
        """

        async def htlc_switch_iteration():
            await self._htlc_switch_iterstart_event.wait()
            await self._htlc_switch_iterdone_event.wait()
        async with OldTaskGroup(wait=any) as group:
            await group.spawn(htlc_switch_iteration())
            await group.spawn(self.got_disconnected.wait())

    def process_unfulfilled_htlc(self, *, chan: Channel, htlc: UpdateAddHtlc, forwarding_info: Tuple[str, int], onion_packet_bytes: bytes, onion_packet: OnionPacket) -> Tuple[Optional[bytes], Union[bool, None, Tuple[str, int]], Optional[bytes]]:
        if False:
            while True:
                i = 10
        '\n        return (preimage, fw_info, error_bytes) with at most a single element that is not None\n        raise an OnionRoutingFailure if we need to fail the htlc\n        '
        payment_hash = htlc.payment_hash
        processed_onion = self.process_onion_packet(onion_packet, payment_hash=payment_hash, onion_packet_bytes=onion_packet_bytes)
        if processed_onion.are_we_final:
            if not forwarding_info:
                (preimage, forwarding_callback) = self.maybe_fulfill_htlc(chan=chan, htlc=htlc, processed_onion=processed_onion, onion_packet_bytes=onion_packet_bytes)
                if forwarding_callback:
                    payment_secret = processed_onion.hop_data.payload['payment_data']['payment_secret']
                    payment_key = payment_hash + payment_secret
                    if not self.lnworker.enable_htlc_forwarding:
                        return (None, None, None)
                    elif payment_key in self.lnworker.final_onion_forwardings:
                        self.logger.info(f'we are already forwarding this.')
                    else:
                        self.lnworker.final_onion_forwardings.add(payment_key)
                        self.lnworker.final_onion_forwarding_failures.pop(payment_key, None)

                        async def wrapped_callback():
                            forwarding_coro = forwarding_callback()
                            try:
                                await forwarding_coro
                            except OnionRoutingFailure as e:
                                self.lnworker.final_onion_forwarding_failures[payment_key] = e
                            finally:
                                self.lnworker.final_onion_forwardings.remove(payment_key)
                        asyncio.ensure_future(wrapped_callback())
                    fw_info = (payment_key.hex(), -1)
                    return (None, fw_info, None)
            else:
                payment_key_outer_onion = bytes.fromhex(forwarding_info[0])
                preimage = self.lnworker.get_preimage(payment_hash)
                payment_secret_inner_onion = self.lnworker.get_payment_secret(payment_hash)
                payment_key_inner_onion = payment_hash + payment_secret_inner_onion
                for payment_key in [payment_key_inner_onion, payment_key_outer_onion]:
                    error_reason = self.lnworker.final_onion_forwarding_failures.get(payment_key)
                    if error_reason:
                        self.logger.info(f'trampoline forwarding failure: {error_reason.code_name()}')
                        raise error_reason
        elif not forwarding_info:
            if not self.lnworker.enable_htlc_forwarding:
                return (None, None, None)
            (next_chan_id, next_htlc_id) = self.maybe_forward_htlc(incoming_chan=chan, htlc=htlc, processed_onion=processed_onion)
            fw_info = (next_chan_id.hex(), next_htlc_id)
            return (None, fw_info, None)
        else:
            preimage = self.lnworker.get_preimage(payment_hash)
            (next_chan_id_hex, htlc_id) = forwarding_info
            next_chan = self.lnworker.get_channel_by_short_id(bytes.fromhex(next_chan_id_hex))
            if next_chan:
                (error_bytes, error_reason) = next_chan.pop_fail_htlc_reason(htlc_id)
                if error_bytes:
                    return (None, None, error_bytes)
                if error_reason:
                    raise error_reason
            if htlc_id == -1:
                error_reason = self.jit_failures.pop(next_chan_id_hex, None)
                if error_reason:
                    raise error_reason
        if preimage:
            return (preimage, None, None)
        return (None, None, None)

    def process_onion_packet(self, onion_packet: OnionPacket, *, payment_hash: bytes, onion_packet_bytes: bytes, is_trampoline: bool=False) -> ProcessedOnionPacket:
        if False:
            while True:
                i = 10
        failure_data = sha256(onion_packet_bytes)
        try:
            processed_onion = process_onion_packet(onion_packet, associated_data=payment_hash, our_onion_private_key=self.privkey, is_trampoline=is_trampoline)
        except UnsupportedOnionPacketVersion:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_VERSION, data=failure_data)
        except InvalidOnionPubkey:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_KEY, data=failure_data)
        except InvalidOnionMac:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_HMAC, data=failure_data)
        except Exception as e:
            self.logger.info(f'error processing onion packet: {e!r}')
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_VERSION, data=failure_data)
        if self.network.config.TEST_FAIL_HTLCS_AS_MALFORMED:
            raise OnionRoutingFailure(code=OnionFailureCode.INVALID_ONION_VERSION, data=failure_data)
        if self.network.config.TEST_FAIL_HTLCS_WITH_TEMP_NODE_FAILURE:
            raise OnionRoutingFailure(code=OnionFailureCode.TEMPORARY_NODE_FAILURE, data=b'')
        return processed_onion