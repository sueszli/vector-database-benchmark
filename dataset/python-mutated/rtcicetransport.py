import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from aioice import Candidate, Connection, ConnectionClosed
from pyee.asyncio import AsyncIOEventEmitter
from .exceptions import InvalidStateError
from .rtcconfiguration import RTCIceServer
STUN_REGEX = re.compile('(?P<scheme>stun|stuns)\\:(?P<host>[^?:]+)(\\:(?P<port>[0-9]+?))?(\\?transport=.*)?')
TURN_REGEX = re.compile('(?P<scheme>turn|turns)\\:(?P<host>[^?:]+)(\\:(?P<port>[0-9]+?))?(\\?transport=(?P<transport>.*))?')
logger = logging.getLogger(__name__)

@dataclass
class RTCIceCandidate:
    """
    The :class:`RTCIceCandidate` interface represents a candidate Interactive
    Connectivity Establishment (ICE) configuration which may be used to
    establish an RTCPeerConnection.
    """
    component: int
    foundation: str
    ip: str
    port: int
    priority: int
    protocol: str
    type: str
    relatedAddress: Optional[str] = None
    relatedPort: Optional[int] = None
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None
    tcpType: Optional[str] = None

@dataclass
class RTCIceParameters:
    """
    The :class:`RTCIceParameters` dictionary includes the ICE username
    fragment and password and other ICE-related parameters.
    """
    usernameFragment: Optional[str] = None
    'ICE username fragment.'
    password: Optional[str] = None
    'ICE password.'
    iceLite: bool = False

def candidate_from_aioice(x: Candidate) -> RTCIceCandidate:
    if False:
        i = 10
        return i + 15
    return RTCIceCandidate(component=x.component, foundation=x.foundation, ip=x.host, port=x.port, priority=x.priority, protocol=x.transport, relatedAddress=x.related_address, relatedPort=x.related_port, tcpType=x.tcptype, type=x.type)

def candidate_to_aioice(x: RTCIceCandidate) -> Candidate:
    if False:
        i = 10
        return i + 15
    return Candidate(component=x.component, foundation=x.foundation, host=x.ip, port=x.port, priority=x.priority, related_address=x.relatedAddress, related_port=x.relatedPort, transport=x.protocol, tcptype=x.tcpType, type=x.type)

def connection_kwargs(servers: List[RTCIceServer]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    kwargs: Dict[str, Any] = {}
    for server in servers:
        if isinstance(server.urls, list):
            uris = server.urls
        else:
            uris = [server.urls]
        for uri in uris:
            parsed = parse_stun_turn_uri(uri)
            if parsed['scheme'] == 'stun':
                if 'stun_server' in kwargs:
                    continue
                kwargs['stun_server'] = (parsed['host'], parsed['port'])
            elif parsed['scheme'] in ['turn', 'turns']:
                if 'turn_server' in kwargs:
                    continue
                if parsed['scheme'] == 'turn' and parsed['transport'] not in ['udp', 'tcp']:
                    continue
                elif parsed['scheme'] == 'turns' and parsed['transport'] != 'tcp':
                    continue
                if server.credentialType != 'password':
                    continue
                kwargs['turn_server'] = (parsed['host'], parsed['port'])
                kwargs['turn_ssl'] = parsed['scheme'] == 'turns'
                kwargs['turn_transport'] = parsed['transport']
                kwargs['turn_username'] = server.username
                kwargs['turn_password'] = server.credential
    return kwargs

def parse_stun_turn_uri(uri: str) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    if uri.startswith('stun'):
        match = STUN_REGEX.fullmatch(uri)
    elif uri.startswith('turn'):
        match = TURN_REGEX.fullmatch(uri)
    else:
        raise ValueError('malformed uri: invalid scheme')
    if not match:
        raise ValueError('malformed uri')
    parsed: Dict[str, Any] = match.groupdict()
    if parsed['port']:
        parsed['port'] = int(parsed['port'])
    elif parsed['scheme'] in ['stuns', 'turns']:
        parsed['port'] = 5349
    else:
        parsed['port'] = 3478
    if parsed['scheme'] == 'turn' and (not parsed['transport']):
        parsed['transport'] = 'udp'
    elif parsed['scheme'] == 'turns' and (not parsed['transport']):
        parsed['transport'] = 'tcp'
    return parsed

class RTCIceGatherer(AsyncIOEventEmitter):
    """
    The :class:`RTCIceGatherer` interface gathers local host, server reflexive
    and relay candidates, as well as enabling the retrieval of local
    Interactive Connectivity Establishment (ICE) parameters which can be
    exchanged in signaling.
    """

    def __init__(self, iceServers: Optional[List[RTCIceServer]]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        if iceServers is None:
            iceServers = self.getDefaultIceServers()
        ice_kwargs = connection_kwargs(iceServers)
        self._connection = Connection(ice_controlling=False, **ice_kwargs)
        self._remote_candidates_end = False
        self.__state = 'new'

    @property
    def state(self) -> str:
        if False:
            while True:
                i = 10
        '\n        The current state of the ICE gatherer.\n        '
        return self.__state

    async def gather(self) -> None:
        """
        Gather ICE candidates.
        """
        if self.__state == 'new':
            self.__setState('gathering')
            await self._connection.gather_candidates()
            self.__setState('completed')

    @classmethod
    def getDefaultIceServers(cls) -> List[RTCIceServer]:
        if False:
            while True:
                i = 10
        '\n        Return the list of default :class:`RTCIceServer`.\n        '
        return [RTCIceServer('stun:stun.l.google.com:19302')]

    def getLocalCandidates(self) -> List[RTCIceCandidate]:
        if False:
            return 10
        '\n        Retrieve the list of valid local candidates associated with the ICE\n        gatherer.\n        '
        return [candidate_from_aioice(x) for x in self._connection.local_candidates]

    def getLocalParameters(self) -> RTCIceParameters:
        if False:
            return 10
        '\n        Retrieve the ICE parameters of the ICE gatherer.\n\n        :rtype: RTCIceParameters\n        '
        return RTCIceParameters(usernameFragment=self._connection.local_username, password=self._connection.local_password)

    def __setState(self, state: str) -> None:
        if False:
            print('Hello World!')
        self.__state = state
        self.emit('statechange')

class RTCIceTransport(AsyncIOEventEmitter):
    """
    The :class:`RTCIceTransport` interface allows an application access to
    information about the Interactive Connectivity Establishment (ICE)
    transport over which packets are sent and received.

    :param gatherer: An :class:`RTCIceGatherer`.
    """

    def __init__(self, gatherer: RTCIceGatherer) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.__iceGatherer = gatherer
        self.__monitor_task: Optional[asyncio.Future[None]] = None
        self.__start: Optional[asyncio.Event] = None
        self.__state = 'new'
        self._connection = gatherer._connection
        self._role_set = False
        self._recv = self._connection.recv
        self._send = self._connection.send

    @property
    def iceGatherer(self) -> RTCIceGatherer:
        if False:
            print('Hello World!')
        '\n        The ICE gatherer passed in the constructor.\n        '
        return self.__iceGatherer

    @property
    def role(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        "\n        The current role of the ICE transport.\n\n        Either `'controlling'` or `'controlled'`.\n        "
        if self._connection.ice_controlling:
            return 'controlling'
        else:
            return 'controlled'

    @property
    def state(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The current state of the ICE transport.\n        '
        return self.__state

    async def addRemoteCandidate(self, candidate: Optional[RTCIceCandidate]) -> None:
        """
        Add a remote candidate.

        :param candidate: The new candidate or `None` to signal end of candidates.
        """
        if not self.__iceGatherer._remote_candidates_end:
            if candidate is None:
                self.__iceGatherer._remote_candidates_end = True
                await self._connection.add_remote_candidate(None)
            else:
                await self._connection.add_remote_candidate(candidate_to_aioice(candidate))

    def getRemoteCandidates(self) -> List[RTCIceCandidate]:
        if False:
            while True:
                i = 10
        '\n        Retrieve the list of candidates associated with the remote\n        :class:`RTCIceTransport`.\n        '
        return [candidate_from_aioice(x) for x in self._connection.remote_candidates]

    async def start(self, remoteParameters: RTCIceParameters) -> None:
        """
        Initiate connectivity checks.

        :param remoteParameters: The :class:`RTCIceParameters` associated with
                                  the remote :class:`RTCIceTransport`.
        """
        if self.state == 'closed':
            raise InvalidStateError('RTCIceTransport is closed')
        if self.__start is not None:
            await self.__start.wait()
            return
        self.__start = asyncio.Event()
        self.__monitor_task = asyncio.ensure_future(self._monitor())
        self.__setState('checking')
        self._connection.remote_is_lite = remoteParameters.iceLite
        self._connection.remote_username = remoteParameters.usernameFragment
        self._connection.remote_password = remoteParameters.password
        try:
            await self._connection.connect()
        except ConnectionError:
            self.__setState('failed')
        else:
            self.__setState('completed')
        self.__start.set()

    async def stop(self) -> None:
        """
        Irreversibly stop the :class:`RTCIceTransport`.
        """
        if self.state != 'closed':
            self.__setState('closed')
            await self._connection.close()
            if self.__monitor_task is not None:
                await self.__monitor_task
                self.__monitor_task = None

    async def _monitor(self) -> None:
        while True:
            event = await self._connection.get_event()
            if isinstance(event, ConnectionClosed):
                if self.state == 'completed':
                    self.__setState('failed')
                return

    def __log_debug(self, msg: str, *args) -> None:
        if False:
            while True:
                i = 10
        logger.debug(f'RTCIceTransport(%s) {msg}', self.role, *args)

    def __setState(self, state: str) -> None:
        if False:
            return 10
        if state != self.__state:
            self.__log_debug('- %s -> %s', self.__state, state)
            self.__state = state
            self.emit('statechange')
            if state == 'closed':
                self.iceGatherer.remove_all_listeners()
                self.remove_all_listeners()