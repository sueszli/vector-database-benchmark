from __future__ import annotations
import logging
import random
import socket
import struct
import sys
import time
from abc import ABCMeta, abstractmethod
from asyncio import DatagramProtocol, Future, TimeoutError, ensure_future, get_event_loop
from typing import List, TYPE_CHECKING
import async_timeout
from aiohttp import ClientResponseError, ClientSession, ClientTimeout
from ipv8.taskmanager import TaskManager
from tribler.core.components.socks_servers.socks5.aiohttp_connector import Socks5Connector
from tribler.core.components.socks_servers.socks5.client import Socks5Client
from tribler.core.components.torrent_checker.torrent_checker import DHT
from tribler.core.components.torrent_checker.torrent_checker.dataclasses import HealthInfo, TrackerResponse
from tribler.core.components.torrent_checker.torrent_checker.utils import filter_non_exceptions, gather_coros
from tribler.core.utilities.tracker_utils import add_url_params, parse_tracker_url
from tribler.core.utilities.utilities import INT32_MAX, bdecode_compat
if TYPE_CHECKING:
    from tribler.core.components.libtorrent.download_manager.download_manager import DownloadManager
TRACKER_ACTION_CONNECT = 0
TRACKER_ACTION_ANNOUNCE = 1
TRACKER_ACTION_SCRAPE = 2
UDP_TRACKER_INIT_CONNECTION_ID = 4497486125440
MAX_INFOHASHES_IN_SCRAPE = 60

class TrackerSession(TaskManager):
    __meta__ = ABCMeta

    def __init__(self, tracker_type, tracker_url, tracker_address, announce_page, timeout):
        if False:
            print('Hello World!')
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.tracker_type = tracker_type
        self.tracker_url = tracker_url
        self.tracker_address = tracker_address
        self.announce_page = announce_page
        self.timeout = timeout
        self.infohash_list = []
        self.last_contact = None
        self.is_initiated = False
        self.is_finished = False
        self.is_failed = False

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}[{self.tracker_type}, {self.tracker_url}]'

    async def cleanup(self):
        await self.shutdown_task_manager()
        self.infohash_list = None

    def has_infohash(self, infohash):
        if False:
            i = 10
            return i + 15
        return infohash in self.infohash_list

    def add_infohash(self, infohash):
        if False:
            print('Hello World!')
        '\n        Adds an infohash into this session.\n        :param infohash: The infohash to be added.\n        '
        assert not self.is_initiated, 'Must not add request to an initiated session.'
        assert not self.has_infohash(infohash), 'Must not add duplicate requests'
        if len(self.infohash_list) < MAX_INFOHASHES_IN_SCRAPE:
            self.infohash_list.append(infohash)

    def failed(self, msg=None):
        if False:
            print('Hello World!')
        '\n        This method handles everything that needs to be done when one step\n        in the session has failed and thus no data can be obtained.\n        '
        if not self.is_failed:
            self.is_failed = True
            result_msg = f'{self.tracker_type} tracker failed for url {self.tracker_url}'
            if msg:
                result_msg += f' (error: {msg})'
            raise ValueError(result_msg)

    @abstractmethod
    async def connect_to_tracker(self) -> TrackerResponse:
        """Does some work when a connection has been established."""

class HttpTrackerSession(TrackerSession):

    def __init__(self, tracker_url, tracker_address, announce_page, timeout, proxy):
        if False:
            print('Hello World!')
        super().__init__('http', tracker_url, tracker_address, announce_page, timeout)
        self._session = ClientSession(connector=Socks5Connector(proxy) if proxy else None, raise_for_status=True, timeout=ClientTimeout(total=self.timeout))

    async def connect_to_tracker(self) -> TrackerResponse:
        url = add_url_params('http://%s:%s%s' % (self.tracker_address[0], self.tracker_address[1], self.announce_page.replace('announce', 'scrape')), {'info_hash': self.infohash_list})
        self.is_initiated = True
        self.last_contact = int(time.time())
        try:
            self._logger.debug('%s HTTP SCRAPE message sent: %s', self, url)
            async with self._session:
                async with self._session.get(url.encode('ascii').decode('utf-8')) as response:
                    body = await response.read()
        except UnicodeEncodeError as e:
            raise e
        except ClientResponseError as e:
            self._logger.warning('%s HTTP SCRAPE error response code %s', self, e.status)
            self.failed(msg=f'error code {e.status}')
        except Exception as e:
            self.failed(msg=str(e))
        return self._process_scrape_response(body)

    def _process_scrape_response(self, body) -> TrackerResponse:
        if False:
            while True:
                i = 10
        '\n        This function handles the response body of an HTTP result from an HTTP tracker\n        '
        if body is None:
            self.failed(msg='no response body')
        response_dict = bdecode_compat(body)
        if not response_dict:
            self.failed(msg='no valid response')
        health_list: List[HealthInfo] = []
        now = int(time.time())
        unprocessed_infohashes = set(self.infohash_list)
        files = response_dict.get(b'files')
        if isinstance(files, dict):
            for (infohash, file_info) in files.items():
                seeders = leechers = 0
                if isinstance(file_info, dict):
                    seeders = file_info.get(b'complete', 0)
                    leechers = file_info.get(b'incomplete', 0)
                unprocessed_infohashes.discard(infohash)
                health_list.append(HealthInfo(infohash, seeders, leechers, last_check=now, self_checked=True))
        elif b'failure reason' in response_dict:
            self._logger.info('%s Failure as reported by tracker [%s]', self, repr(response_dict[b'failure reason']))
            self.failed(msg=repr(response_dict[b'failure reason']))
        health_list.extend((HealthInfo(infohash=infohash, last_check=now, self_checked=True) for infohash in unprocessed_infohashes))
        self.is_finished = True
        return TrackerResponse(url=self.tracker_url, torrent_health_list=health_list)

    async def cleanup(self):
        """
        Cleans the session by cancelling all deferreds and closing sockets.
        :return: A deferred that fires once the cleanup is done.
        """
        await self._session.close()
        await super().cleanup()

class UdpSocketManager(DatagramProtocol):
    """
    The UdpSocketManager ensures that the network packets are forwarded to the right UdpTrackerSession.
    """

    def __init__(self):
        if False:
            return 10
        self._logger = logging.getLogger(self.__class__.__name__)
        self.tracker_sessions = {}
        self.transport = None
        self.proxy_transports = {}

    def connection_made(self, transport):
        if False:
            while True:
                i = 10
        self.transport = transport

    async def send_request(self, data, tracker_session):
        transport = self.transport
        proxy = tracker_session.proxy
        if proxy:
            transport = self.proxy_transports.get(proxy, Socks5Client(proxy, self.datagram_received))
            if not transport.associated:
                await transport.associate_udp()
            if proxy not in self.proxy_transports:
                self.proxy_transports[proxy] = transport
        host = tracker_session.ip_address or tracker_session.tracker_address[0]
        try:
            transport.sendto(data, (host, tracker_session.port))
            f = self.tracker_sessions[tracker_session.transaction_id] = Future()
            return await f
        except OSError as e:
            self._logger.warning('Unable to write data to %s:%d - %s', tracker_session.ip_address, tracker_session.port, e)
            return RuntimeError('Unable to write to socket - ' + str(e))

    def datagram_received(self, data, _):
        if False:
            i = 10
            return i + 15
        if data and len(data) >= 4:
            transaction_id = struct.unpack_from('!i', data, 4)[0]
            if transaction_id in self.tracker_sessions:
                session = self.tracker_sessions.pop(transaction_id)
                if not session.done():
                    session.set_result(data)

class UdpTrackerSession(TrackerSession):
    """
    The UDPTrackerSession makes a connection with a UDP tracker and queries
    seeders and leechers for one or more infohashes. It handles the message serialization
    and communication with the torrent checker by making use of Deferred (asynchronously).
    """
    _active_session_dict = dict()

    def __init__(self, tracker_url, tracker_address, announce_page, timeout, proxy, socket_mgr):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('udp', tracker_url, tracker_address, announce_page, timeout)
        self._logger.setLevel(logging.INFO)
        self._connection_id = 0
        self.transaction_id = 0
        self.port = tracker_address[1]
        self.ip_address = None
        self.socket_mgr = socket_mgr
        self.proxy = proxy
        self._connection_id = UDP_TRACKER_INIT_CONNECTION_ID
        self.action = TRACKER_ACTION_CONNECT
        self.generate_transaction_id()

    def generate_transaction_id(self):
        if False:
            i = 10
            return i + 15
        '\n        Generates a unique transaction id and stores this in the _active_session_dict set.\n        '
        while True:
            transaction_id = random.randint(0, INT32_MAX)
            if transaction_id not in UdpTrackerSession._active_session_dict.items():
                UdpTrackerSession._active_session_dict[self] = transaction_id
                self.transaction_id = transaction_id
                break

    def remove_transaction_id(self):
        if False:
            print('Hello World!')
        '\n        Removes an session and its corresponding id from the _active_session_dict set and the socket manager.\n        :param session: The session that needs to be removed from the set.\n        '
        if self in UdpTrackerSession._active_session_dict:
            del UdpTrackerSession._active_session_dict[self]
        if self.socket_mgr and self.transaction_id in self.socket_mgr.tracker_sessions:
            self.socket_mgr.tracker_sessions.pop(self.transaction_id)

    async def cleanup(self):
        """
        Cleans the session by cancelling all deferreds and closing sockets.
        :return: A deferred that fires once the cleanup is done.
        """
        await super().cleanup()
        self.remove_transaction_id()

    async def connect_to_tracker(self) -> TrackerResponse:
        """
        Connects to the tracker and starts querying for seed and leech data.
        :return: A dictionary containing seed/leech information per infohash
        """
        self.is_initiated = True
        await self.cancel_pending_task('result')
        await self.cancel_pending_task('resolve')
        try:
            async with async_timeout.timeout(self.timeout):
                if not self.proxy:
                    coro = get_event_loop().getaddrinfo(self.tracker_address[0], 0, family=socket.AF_INET)
                    if isinstance(coro, Future):
                        infos = await coro
                    else:
                        infos = await self.register_anonymous_task('resolve', ensure_future(coro))
                    self.ip_address = infos[0][-1][0]
                await self.connect()
                return await self.scrape()
        except TimeoutError:
            self.failed(msg='request timed out')
        except socket.gaierror as e:
            self.failed(msg=str(e))

    async def connect(self):
        """
        Creates a connection message and calls the socket manager to send it.
        """
        if not self.socket_mgr.transport:
            self.failed(msg='UDP socket transport not ready')
        message = struct.pack('!qii', self._connection_id, self.action, self.transaction_id)
        response = await self.socket_mgr.send_request(message, self)
        if len(response) < 16:
            self._logger.error('%s Invalid response for UDP CONNECT: %s', self, repr(response))
            self.failed(msg='invalid response size')
        (action, transaction_id) = struct.unpack_from('!ii', response, 0)
        if action != self.action or transaction_id != self.transaction_id:
            errmsg_length = len(response) - 8
            (error_message,) = struct.unpack_from('!' + str(errmsg_length) + 's', response, 8)
            self._logger.info('%s Error response for UDP CONNECT [%s]: %s', self, repr(response), repr(error_message))
            self.failed(msg=error_message.decode('utf8', errors='ignore'))
        self._connection_id = struct.unpack_from('!q', response, 8)[0]
        self.action = TRACKER_ACTION_SCRAPE
        self.generate_transaction_id()
        self.last_contact = int(time.time())

    async def scrape(self) -> TrackerResponse:
        if sys.version_info.major > 2:
            infohash_list = self.infohash_list
        else:
            infohash_list = [str(infohash) for infohash in self.infohash_list]
        fmt = '!qii' + '20s' * len(self.infohash_list)
        message = struct.pack(fmt, self._connection_id, self.action, self.transaction_id, *infohash_list)
        response = await self.socket_mgr.send_request(message, self)
        if len(response) < 8:
            self._logger.info('%s Invalid response for UDP SCRAPE: %s', self, repr(response))
            self.failed('invalid message size')
        (action, transaction_id) = struct.unpack_from('!ii', response, 0)
        if action != self.action or transaction_id != self.transaction_id:
            errmsg_length = len(response) - 8
            (error_message,) = struct.unpack_from('!' + str(errmsg_length) + 's', response, 8)
            self._logger.info('%s Error response for UDP SCRAPE: [%s] [%s]', self, repr(response), repr(error_message))
            self.failed(msg=error_message.decode('utf8', errors='ignore'))
        if len(response) - 8 != len(self.infohash_list) * 12:
            self._logger.info('%s UDP SCRAPE response mismatch: %s', self, len(response))
            self.failed(msg='invalid response size')
        offset = 8
        response_list = []
        now = int(time.time())
        for infohash in self.infohash_list:
            (complete, _downloaded, incomplete) = struct.unpack_from('!iii', response, offset)
            offset += 12
            response_list.append(HealthInfo(infohash, seeders=complete, leechers=incomplete, last_check=now, self_checked=True))
        self.remove_transaction_id()
        self.last_contact = int(time.time())
        self.is_finished = True
        return TrackerResponse(url=self.tracker_url, torrent_health_list=response_list)

class FakeDHTSession(TrackerSession):
    """
    Fake TrackerSession that manages DHT requests
    """

    def __init__(self, download_manager: DownloadManager, timeout: float):
        if False:
            print('Hello World!')
        super().__init__(DHT, DHT, DHT, DHT, timeout)
        self.download_manager = download_manager

    async def connect_to_tracker(self) -> TrackerResponse:
        health_list = []
        now = int(time.time())
        for infohash in self.infohash_list:
            metainfo = await self.download_manager.get_metainfo(infohash, timeout=self.timeout, raise_errors=True)
            health = HealthInfo(infohash, seeders=metainfo[b'seeders'], leechers=metainfo[b'leechers'], last_check=now, self_checked=True)
            health_list.append(health)
        return TrackerResponse(url=DHT, torrent_health_list=health_list)

class FakeBep33DHTSession(FakeDHTSession):
    """
    Fake session for a BEP33 lookup.
    """

    async def connect_to_tracker(self) -> TrackerResponse:
        coros = [self.download_manager.dht_health_manager.get_health(infohash, timeout=self.timeout) for infohash in self.infohash_list]
        results = await gather_coros(coros)
        return TrackerResponse(url=DHT, torrent_health_list=filter_non_exceptions(results))

def create_tracker_session(tracker_url, timeout, proxy, socket_manager) -> TrackerSession:
    if False:
        print('Hello World!')
    '\n    Creates a tracker session with the given tracker URL.\n    :param tracker_url: The given tracker URL.\n    :param timeout: The timeout for the session.\n    :return: The tracker session.\n    '
    (tracker_type, tracker_address, announce_page) = parse_tracker_url(tracker_url)
    if tracker_type == 'udp':
        return UdpTrackerSession(tracker_url, tracker_address, announce_page, timeout, proxy, socket_manager)
    return HttpTrackerSession(tracker_url, tracker_address, announce_page, timeout, proxy)