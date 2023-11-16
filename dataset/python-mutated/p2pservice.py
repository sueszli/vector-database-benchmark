import ipaddress
import itertools
import logging
import socket
import random
import time
from collections import deque
from threading import Lock
from typing import Any, Callable, Dict, List
from golem_messages import message
from golem_messages.datastructures import p2p as dt_p2p
from golem_messages.datastructures import tasks as dt_tasks
from golem.config.active import P2P_SEEDS
from golem.core import simplechallenge
from golem.core.variables import MAX_CONNECT_SOCKET_ADDRESSES
from golem.core.common import node_info_str
from golem.diag.service import DiagnosticsProvider
from golem.model import KnownHosts, db
from golem.network.p2p.peersession import PeerSession, PeerSessionInfo
from golem.network.transport import tcpnetwork
from golem.network.transport import tcpserver
from golem.network.transport.network import ProtocolFactory, SessionFactory
from golem.ranking.manager.gossip_manager import GossipManager
from .peerkeeper import PeerKeeper, key_distance
logger = logging.getLogger(__name__)
LAST_MESSAGE_BUFFER_LEN = 5
RECONNECT_WITH_SEED_THRESHOLD = 30
SOLVE_CHALLENGE = True
FORWARD_NEIGHBORS_COUNT = 3
FORWARD_BATCH_SIZE = 12
BASE_DIFFICULTY = 5
HISTORY_LEN = 5
TASK_INTERVAL = 10
PEERS_INTERVAL = 30
FORWARD_INTERVAL = 2
RANDOM_DISCONNECT_INTERVAL = 5 * 60
RANDOM_DISCONNECT_FRACTION = 0.1
MAX_STORED_HOSTS = 100

class P2PService(tcpserver.PendingConnectionsServer, DiagnosticsProvider):

    def __init__(self, node, config_desc, keys_auth, connect_to_known_hosts=True):
        if False:
            i = 10
            return i + 15
        'Create new P2P Server. Listen on port for connections and\n           connect to other peers. Keeps up-to-date list of peers information\n           and optimal number of open connections.\n        :param Node node: Information about this node\n        :param ClientConfigDescriptor config_desc: configuration options\n        :param KeysAuth keys_auth: authorization manager\n        '
        network = tcpnetwork.TCPNetwork(ProtocolFactory(tcpnetwork.BroadcastProtocol, self, SessionFactory(PeerSession)), config_desc.use_ipv6, limit_connection_rate=True)
        tcpserver.PendingConnectionsServer.__init__(self, config_desc, network)
        self.node = node
        self.keys_auth = keys_auth
        self.peer_keeper = PeerKeeper(keys_auth.key_id)
        self.task_server = None
        self.metadata_manager = None
        self.resource_port = 0
        self.suggested_address = {}
        self.suggested_conn_reverse = {}
        self.gossip_keeper = GossipManager()
        self.manager_session = None
        self.metadata_providers: Dict[str, Callable[[], Any]] = {}
        self.node_name = self.config_desc.node_name
        self.last_message_time_threshold = self.config_desc.p2p_session_timeout
        self.last_message_buffer_len = LAST_MESSAGE_BUFFER_LEN
        self.last_time_tried_connect_with_seed = 0
        self.reconnect_with_seed_threshold = RECONNECT_WITH_SEED_THRESHOLD
        self.should_solve_challenge = SOLVE_CHALLENGE
        self.challenge_history = deque(maxlen=HISTORY_LEN)
        self.last_challenge = ''
        self.base_difficulty = BASE_DIFFICULTY
        self.connect_to_known_hosts = connect_to_known_hosts
        self.peers = {}
        self.peer_order = []
        self.incoming_peers = {}
        self.free_peers = []
        self.seeds = set()
        self.used_seeds = set()
        self.bootstrap_seeds = P2P_SEEDS
        self._peer_lock = Lock()
        try:
            self.__remove_redundant_hosts_from_db()
            self._sync_seeds()
        except Exception as exc:
            logger.error('Error reading seed addresses: {}'.format(exc))
        now = time.time()
        self.last_peers_request = now
        self.last_tasks_request = now
        self.last_refresh_peers = now
        self.last_forward_request = now
        self.last_random_disconnect = now
        self.last_seeds_sync = time.time()
        self.last_messages = []
        random.seed()

    def _listening_established(self, port):
        if False:
            print('Hello World!')
        super(P2PService, self)._listening_established(port)
        self.node.p2p_prv_port = port

    def connect_to_network(self):
        if False:
            return 10
        logger.debug('Connecting to seeds')
        self.connect_to_seeds()
        if not self.connect_to_known_hosts:
            return
        logger.debug('Connecting to known hosts')
        for host in KnownHosts.select().where(KnownHosts.is_seed == False).limit(self.config_desc.opt_peer_num):
            ip_address = host.ip_address
            port = host.port
            logger.debug('Connecting to %s:%s ...', ip_address, port)
            try:
                socket_address = tcpnetwork.SocketAddress(ip_address, port)
                self.connect(socket_address)
                logger.debug('Connected!')
            except Exception as exc:
                logger.error('Cannot connect to host {}:{}: {}'.format(ip_address, port, exc))

    def connect_to_seeds(self):
        if False:
            return 10
        self.last_time_tried_connect_with_seed = time.time()
        if not self.connect_to_known_hosts:
            return
        for _ in range(len(self.seeds)):
            (ip_address, port) = self._get_next_random_seed()
            logger.debug('Connecting to %s:%s ...', ip_address, port)
            try:
                socket_address = tcpnetwork.SocketAddress(ip_address, port)
                self.connect(socket_address)
            except Exception as exc:
                logger.error('Cannot connect to seed %s:%s: %s', ip_address, port, exc)
                continue
            logger.debug('Connected!')
            break

    def connect(self, socket_address):
        if False:
            while True:
                i = 10
        if not self.active:
            return
        connect_info = tcpnetwork.TCPConnectInfo([socket_address], self.__connection_established, P2PService.__connection_failure)
        self.network.connect(connect_info)

    def disconnect(self):
        if False:
            print('Hello World!')
        peers = dict(self.peers)
        for peer in peers.values():
            peer.dropped()

    def new_connection(self, session):
        if False:
            i = 10
            return i + 15
        if self.active:
            session.start()
        else:
            session.disconnect(message.base.Disconnect.REASON.NoMoreMessages)

    def add_known_peer(self, node, ip_address, port, metadata=None):
        if False:
            i = 10
            return i + 15
        is_seed = node.is_super_node() if node else False
        try:
            with db.transaction():
                (host, _) = KnownHosts.get_or_create(ip_address=ip_address, port=port, defaults={'is_seed': is_seed})
                host.last_connected = time.time()
                host.metadata = metadata or {}
                host.save()
            self.__remove_redundant_hosts_from_db()
            self._sync_seeds()
        except Exception as err:
            logger.error("Couldn't add known peer %s:%s - %s", ip_address, port, err)

    def set_metadata_manager(self, metadata_manager):
        if False:
            return 10
        self.metadata_manager = metadata_manager

    def interpret_metadata(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.metadata_manager.interpret_metadata(*args, **kwargs)

    def sync_network(self):
        if False:
            i = 10
            return i + 15
        'Get information about new tasks and new peers in the network.\n           Remove excess information about peers\n        '
        super().sync_network(timeout=self.last_message_time_threshold)
        now = time.time()
        if self.task_server and now - self.last_tasks_request > TASK_INTERVAL:
            self.last_tasks_request = now
            self._send_get_tasks()
        if now - self.last_peers_request > PEERS_INTERVAL:
            self.last_peers_request = now
            self.__sync_free_peers()
            self.__sync_peer_keeper()
            self.__send_get_peers()
        if now - self.last_forward_request > FORWARD_INTERVAL:
            self.last_forward_request = now
            self._sync_forward_requests()
        self.__remove_old_peers()
        if now - self.last_random_disconnect > RANDOM_DISCONNECT_INTERVAL:
            self.last_random_disconnect = now
            self._disconnect_random_peers()
        self._sync_pending()
        if now - self.last_seeds_sync > self.reconnect_with_seed_threshold:
            self._sync_seeds()
        if len(self.peers) == 0:
            delta = now - self.last_time_tried_connect_with_seed
            if delta > self.reconnect_with_seed_threshold:
                self.connect_to_seeds()

    def get_diagnostics(self, output_format):
        if False:
            return 10
        peer_data = []
        for peer in self.peers.values():
            peer = PeerSessionInfo(peer).get_simplified_repr()
            peer_data.append(peer)
        return self._format_diagnostics(peer_data, output_format)

    def get_estimated_network_size(self) -> int:
        if False:
            return 10
        size = self.peer_keeper.get_estimated_network_size()
        logger.info('Estimated network size: %r', size)
        return size

    @staticmethod
    def get_performance_percentile_rank(perf: float, env_id: str) -> float:
        if False:
            for i in range(10):
                print('nop')
        hosts_perf = [host.metadata['performance'].get(env_id, -1.0) for host in KnownHosts.select() if 'performance' in host.metadata]
        if not hosts_perf:
            logger.warning('Cannot compute percentile rank. No host performance info is available')
            return 1.0
        rank = sum((1 for x in hosts_perf if x < perf)) / len(hosts_perf)
        logger.info(f'Performance for env `{env_id}`: rank({perf}) = {rank}')
        return rank

    def ping_peers(self, interval):
        if False:
            print('Hello World!')
        ' Send ping to all peers with whom this peer has open connection\n        :param int interval: will send ping only if time from last ping\n                             was longer than interval\n        '
        for p in list(self.peers.values()):
            p.ping(interval)

    def find_peer(self, key_id):
        if False:
            print('Hello World!')
        ' Find peer with given id on list of active connections\n        :param key_id: id of a searched peer\n        :return None|PeerSession: connection to a given peer or None\n        '
        return self.peers.get(key_id)

    def get_peers(self):
        if False:
            while True:
                i = 10
        ' Return all open connection to other peers that this node keeps\n        :return dict: dictionary of peers sessions\n        '
        return self.peers

    def add_peer(self, peer: PeerSession):
        if False:
            while True:
                i = 10
        ' Add a new open connection with a peer to the list of peers\n        :param peer: peer session with given peer\n        '
        key_id = peer.key_id
        logger.info('Adding peer. node=%s, address=%s:%s', node_info_str(peer.node_name, key_id), peer.address, peer.port)
        with self._peer_lock:
            self.peers[key_id] = peer
            self.peer_order.append(key_id)
        try:
            self.pending_sessions.remove(peer)
        except KeyError:
            pass

    def add_to_peer_keeper(self, peer_info):
        if False:
            i = 10
            return i + 15
        ' Add information about peer to the peer keeper\n        :param Node peer_info: information about new peer\n        '
        peer_to_ping_info = self.peer_keeper.add_peer(peer_info)
        if peer_to_ping_info and peer_to_ping_info.key in self.peers:
            peer_to_ping = self.peers[peer_to_ping_info.key]
            if peer_to_ping:
                peer_to_ping.ping(0)

    def pong_received(self, key_num):
        if False:
            i = 10
            return i + 15
        ' React to pong received from other node\n        :param key_num: public key of a ping sender\n        :return:\n        '
        self.peer_keeper.pong_received(key_num)

    def try_to_add_peer(self, peer_info: dt_p2p.Peer, force=False):
        if False:
            for i in range(10):
                print('nop')
        ' Add peer to inner peer information\n        :param force: add or overwrite existing data\n        '
        key_id = peer_info['node'].key
        node_name = peer_info['node'].node_name
        if not self._is_address_valid(peer_info['address'], peer_info['port']):
            return
        if not (force or self.__is_new_peer(key_id)):
            return
        logger.info('Adding peer to incoming. node=%s, address=%s:%s', node_info_str(node_name, key_id), peer_info['address'], peer_info['port'])
        self.incoming_peers[key_id] = {'address': peer_info['address'], 'port': peer_info['port'], 'node': peer_info['node'], 'node_name': node_name, 'conn_trials': 0}
        if key_id not in self.free_peers:
            self.free_peers.append(key_id)
        logger.debug(self.incoming_peers)

    def remove_peer(self, peer_session):
        if False:
            while True:
                i = 10
        ' Remove given peer session\n        :param PeerSession peer_session: remove peer session\n        '
        self.remove_pending_conn(peer_session.conn_id)
        peer_id = peer_session.key_id
        stored_session = self.peers.get(peer_id)
        if stored_session == peer_session:
            self.remove_peer_by_id(peer_id)

    def remove_peer_by_id(self, peer_id):
        if False:
            for i in range(10):
                print('nop')
        ' Remove peer session with peer that has given id\n        :param str peer_id:\n        '
        with self._peer_lock:
            peer = self.peers.pop(peer_id, None)
            self.incoming_peers.pop(peer_id, None)
            self.suggested_address.pop(peer_id, None)
            self.suggested_conn_reverse.pop(peer_id, None)
            if peer_id in self.free_peers:
                self.free_peers.remove(peer_id)
            if peer_id in self.peer_order:
                self.peer_order.remove(peer_id)
        if not peer:
            logger.info("Can't remove peer {}, unknown peer".format(peer_id))

    def refresh_peer(self, peer):
        if False:
            print('Hello World!')
        self.remove_peer(peer)
        self.try_to_add_peer({'address': peer.address, 'port': peer.port, 'node': peer.node_info, 'node_name': peer.node_name}, force=True)

    def enough_peers(self):
        if False:
            while True:
                i = 10
        'Inform whether peer has optimal or more open connections with\n           other peers\n        :return bool: True if peer has enough open connections with other\n                      peers, False otherwise\n        '
        with self._peer_lock:
            return len(self.peers) >= self.config_desc.opt_peer_num

    def set_last_message(self, type_, client_key_id, t, msg, address, port):
        if False:
            print('Hello World!')
        'Add given message to last message buffer and inform peer keeper\n           about it\n        :param int type_: message time\n        :param client_key_id: public key of a message sender\n        :param float t: time of receiving message\n        :param Message msg: received message\n        :param str address: sender address\n        :param int port: sender port\n        '
        self.peer_keeper.set_last_message_time(client_key_id)
        if len(self.last_messages) >= self.last_message_buffer_len:
            self.last_messages = self.last_messages[-(self.last_message_buffer_len - 1):]
        self.last_messages.append([type_, t, address, port, msg])

    def get_last_messages(self):
        if False:
            i = 10
            return i + 15
        ' Return list of a few recent messages\n        :return list: last messages\n        '
        return self.last_messages

    def manager_session_disconnect(self, uid):
        if False:
            return 10
        ' Remove manager session\n        '
        self.manager_session = None

    def change_config(self, config_desc):
        if False:
            print('Hello World!')
        ' Change configuration descriptor.\n        If node_name was changed, send hello to all peers to update node_name.\n        If listening port is changed, than stop listening on old port and start\n        listening on a new one. If seed address is changed, connect to a new\n        seed.\n        Change configuration for resource server.\n        :param ClientConfigDescriptor config_desc: new config descriptor\n        '
        tcpserver.TCPServer.change_config(self, config_desc)
        self.node_name = config_desc.node_name
        self.last_message_time_threshold = self.config_desc.p2p_session_timeout
        for peer in list(self.peers.values()):
            if peer.port == self.config_desc.seed_port and peer.address == self.config_desc.seed_host:
                return
        if self.config_desc.seed_host and self.config_desc.seed_port:
            try:
                socket_address = tcpnetwork.SocketAddress(self.config_desc.seed_host, self.config_desc.seed_port)
                self.connect(socket_address)
            except ipaddress.AddressValueError as err:
                logger.error('Invalid seed address: ' + str(err))

    def change_address(self, th_dict_repr):
        if False:
            i = 10
            return i + 15
        ' Change peer address in task header dictionary representation\n        :param dict th_dict_repr: task header dictionary representation\n                                  that should be changed\n        '
        try:
            id_ = th_dict_repr['task_owner']['key']
            if self.peers[id_]:
                th_dict_repr['task_owner']['pub_addr'] = self.peers[id_].address
                th_dict_repr['task_owner']['pub_port'] = self.peers[id_].port
        except KeyError as err:
            logger.error('Wrong task representation: {}'.format(err))

    def check_solution(self, solution, challenge, difficulty):
        if False:
            i = 10
            return i + 15
        "\n        Check whether solution is valid for given challenge and it's difficulty\n        :param str solution: solution to check\n        :param str challenge: solved puzzle\n        :param int difficulty: difficulty of a challenge\n        :return boolean: true if challenge has been correctly solved,\n                         false otherwise\n        "
        return simplechallenge.accept_challenge(challenge, solution, difficulty)

    def solve_challenge(self, key_id, challenge, difficulty):
        if False:
            i = 10
            return i + 15
        ' Solve challenge with given difficulty for a node with key_id\n        :param str key_id: key id of a node that has send this challenge\n        :param str challenge: puzzle to solve\n        :param int difficulty: difficulty of challenge\n        :return str: solution of a challenge\n        '
        self.challenge_history.append([key_id, challenge])
        (solution, time_) = simplechallenge.solve_challenge(challenge, difficulty)
        logger.debug('Solved challenge with difficulty %r in %r sec', difficulty, time_)
        return solution

    def get_peers_degree(self):
        if False:
            i = 10
            return i + 15
        ' Return peers degree level\n        :return dict: dictionary where peers ids are keys and their\n                      degrees are values\n        '
        return {peer.key_id: peer.degree for peer in list(self.peers.values())}

    def get_key_id(self):
        if False:
            return 10
        ' Return node public key in a form of an id '
        return self.peer_keeper.key_num

    def set_suggested_address(self, client_key_id, addr, port):
        if False:
            i = 10
            return i + 15
        'Set suggested address for peer. This node will be used as first\n           for connection attempt\n        :param str client_key_id: peer public key\n        :param str addr: peer suggested address\n        :param int port: peer suggested port\n                         [this argument is ignored right now]\n        :return:\n        '
        self.suggested_address[client_key_id] = addr

    def get_socket_addresses(self, node_info, prv_port=None, pub_port=None):
        if False:
            i = 10
            return i + 15
        ' Change node info into tcp addresses. Adds a suggested address.\n        :param Node node_info: node information\n        :param prv_port: private port that should be used\n        :param pub_port: public port that should be used\n        :return:\n        '
        prv_port = prv_port or node_info.p2p_prv_port
        pub_port = pub_port or node_info.p2p_pub_port
        socket_addresses = super().get_socket_addresses(node_info=node_info, prv_port=prv_port, pub_port=pub_port)
        address = self.suggested_address.get(node_info.key, None)
        if not address:
            return socket_addresses
        if self._is_address_valid(address, prv_port):
            socket_address = tcpnetwork.SocketAddress(address, prv_port)
            self._prepend_address(socket_addresses, socket_address)
        if self._is_address_valid(address, pub_port):
            socket_address = tcpnetwork.SocketAddress(address, pub_port)
            self._prepend_address(socket_addresses, socket_address)
        return socket_addresses[:MAX_CONNECT_SOCKET_ADDRESSES]

    def add_metadata_provider(self, name: str, provider: Callable[[], Any]):
        if False:
            i = 10
            return i + 15
        self.metadata_providers[name] = provider

    def remove_metadata_provider(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.metadata_providers.pop(name, None)

    def get_node_metadata(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        ' Get metadata about node to be sent in `Hello` message '
        return {name: provider() for (name, provider) in self.metadata_providers.items()}

    def send_find_nodes(self, peers_to_find):
        if False:
            i = 10
            return i + 15
        'Kademlia find node function. Send find node request\n           to the closest neighbours\n         of a sought node\n        :param dict peers_to_find: list of nodes that should be find with\n                                   their closest neighbours list\n        '
        for (node_key_id, neighbours) in peers_to_find.items():
            for neighbour in neighbours:
                peer = self.peers.get(neighbour.key)
                if peer:
                    peer.send_find_node(node_key_id)

    def find_node(self, node_key_id, alpha=None) -> List[dt_p2p.Peer]:
        if False:
            print('Hello World!')
        'Kademlia find node function. Find closest neighbours of a node\n           with given public key\n        :param node_key_id: public key of a sought node\n        :param alpha: number of neighbours to find\n        :return list: list of information about closest neighbours\n        '
        alpha = alpha or self.peer_keeper.concurrency
        if node_key_id is None:
            sessions: List[PeerSession] = [peer_session for peer_session in self.peers.values() if self._is_address_valid(peer_session.address, peer_session.listen_port)]
            alpha = min(alpha, len(sessions))
            neighbours: List[PeerSession] = random.sample(sessions, alpha)

            def _mapper_session(session: PeerSession) -> dt_p2p.Peer:
                if False:
                    for i in range(10):
                        print('nop')
                return dt_p2p.Peer({'address': session.address, 'port': session.listen_port, 'node': session.node_info})
            return [_mapper_session(session) for session in neighbours]
        node_neighbours: List[dt_p2p.Node] = self.peer_keeper.neighbours(node_key_id, alpha)

        def _mapper(peer: dt_p2p.Node) -> dt_p2p.Peer:
            if False:
                for i in range(10):
                    print('nop')
            return dt_p2p.Peer({'address': peer.prv_addr, 'port': peer.prv_port, 'node': peer})
        return [_mapper(peer) for peer in node_neighbours if self._is_address_valid(peer.prv_addr, peer.prv_port)]

    def get_own_tasks_headers(self):
        if False:
            return 10
        ' Return a list of a known tasks headers\n        :return list: list of task header\n        '
        return self.task_server.get_own_tasks_headers()

    def get_others_tasks_headers(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return a list of a known tasks headers\n        :return list: list of task header\n        '
        return self.task_server.get_others_tasks_headers()

    def add_task_header(self, task_header: dt_tasks.TaskHeader):
        if False:
            while True:
                i = 10
        ' Add new task header to a list of known task headers\n        :param dict th_dict_repr: new task header dictionary representation\n        :return bool: True if a task header was in a right format,\n                      False otherwise\n        '
        return self.task_server.add_task_header(task_header)

    def remove_task_header(self, task_id) -> bool:
        if False:
            return 10
        ' Remove header of a task with given id from a list of a known tasks\n        :param str task_id: id of a task that should be removed\n        :return: False if task was already removed\n        '
        return self.task_server.remove_task_header(task_id)

    def remove_task(self, task_id):
        if False:
            while True:
                i = 10
        ' Ask all peers to remove information about given task\n        :param str task_id: id of a task that should be removed\n        '
        for p in list(self.peers.values()):
            p.send_remove_task(task_id)

    def send_remove_task_container(self, msg_remove_task):
        if False:
            for i in range(10):
                print('nop')
        for p in list(self.peers.values()):
            p.send(message.p2p.RemoveTaskContainer(remove_tasks=[msg_remove_task]))

    def want_to_start_task_session(self, key_id, node_info, conn_id, super_node_info=None):
        if False:
            for i in range(10):
                print('nop')
        'Inform peer with public key <key_id> that node from node info wants\n           to start task session with him. If peer with given id is on a list\n           of peers that this message will be send directly. Otherwise all\n           peers will receive a request to pass this message.\n        :param str key_id: key id of a node that should open a task session\n        :param Node node_info: information about node that requested session\n        :param str conn_id: connection id for reference\n        :param Node|None super_node_info: *Default: None* information about\n                                          node with public ip that took part\n                                          in message transport\n        '
        if not self.task_server.task_connections_helper.is_new_conn_request(key_id, node_info):
            self.task_server.remove_pending_conn(conn_id)
            self.task_server.remove_responses(conn_id)
            return
        if super_node_info is None and self.node.is_super_node():
            super_node_info = self.node
        connected_peer = self.peers.get(key_id)
        if connected_peer:
            if node_info.key == self.node.key:
                self.suggested_conn_reverse[key_id] = True
            connected_peer.send_want_to_start_task_session(node_info, conn_id, super_node_info)
            logger.debug('Starting task session with %s', key_id)
            return
        msg_snd = False
        peers = list(self.peers.values())
        distances = sorted((p for p in peers if p.key_id != node_info.key and p.verified), key=lambda p: key_distance(key_id, p.key_id))
        for peer in distances[:FORWARD_NEIGHBORS_COUNT]:
            self.task_server.task_connections_helper.forward_queue_put(peer, key_id, node_info, conn_id, super_node_info)
            msg_snd = True
        if msg_snd and node_info.key == self.node.key:
            self.task_server.add_forwarded_session_request(key_id, conn_id)
        if not msg_snd and node_info.key == self.get_key_id():
            self.task_server.task_connections_helper.cannot_start_task_session(conn_id)

    def send_gossip(self, gossip, send_to):
        if False:
            while True:
                i = 10
        ' send gossip to given peers\n        :param list gossip: list of gossips that should be sent\n        :param list send_to: list of ids of peers that should receive gossip\n        '
        for peer_id in send_to:
            peer = self.find_peer(peer_id)
            if peer is not None:
                peer.send_gossip(gossip)

    def hear_gossip(self, gossip):
        if False:
            print('Hello World!')
        ' Add newly heard gossip to the gossip list\n        :param list gossip: list of gossips from one peer\n        '
        self.gossip_keeper.add_gossip(gossip)

    def pop_gossips(self):
        if False:
            return 10
        ' Return all gathered gossips and clear gossip buffer\n        :return list: list of all gossips\n        '
        return self.gossip_keeper.pop_gossips()

    def send_stop_gossip(self):
        if False:
            i = 10
            return i + 15
        ' Send stop gossip message to all peers '
        for peer in list(self.peers.values()):
            peer.send_stop_gossip()

    def stop_gossip(self, id_):
        if False:
            print('Hello World!')
        ' Register that peer with given id has stopped gossiping\n        :param str id_: id of a string that has stopped gossiping\n        '
        self.gossip_keeper.register_that_peer_stopped_gossiping(id_)

    def pop_stop_gossip_form_peers(self):
        if False:
            print('Hello World!')
        " Return set of all peers that has stopped gossiping\n        :return set: set of peers id's\n        "
        return self.gossip_keeper.pop_peers_that_stopped_gossiping()

    def push_local_rank(self, node_id, loc_rank):
        if False:
            print('Hello World!')
        ' Send local rank to peers\n        :param str node_id: id of anode that this opinion is about\n        :param list loc_rank: opinion about this node\n        :return:\n        '
        for peer in list(self.peers.values()):
            peer.send_loc_rank(node_id, loc_rank)

    def safe_neighbour_loc_rank(self, neigh_id, about_id, rank):
        if False:
            i = 10
            return i + 15
        '\n        Add local rank from neighbour to the collection\n        :param str neigh_id: id of a neighbour - opinion giver\n        :param str about_id: opinion is about a node with this id\n        :param list rank: opinion that node <neigh_id> has about\n                          node <about_id>\n        :return:\n        '
        self.gossip_keeper.add_neighbour_loc_rank(neigh_id, about_id, rank)

    def pop_neighbours_loc_ranks(self):
        if False:
            print('Hello World!')
        'Return all local ranks that was collected in that round\n           and clear the rank list\n        :return list: list of all neighbours local rank sent to this node\n        '
        return self.gossip_keeper.pop_neighbour_loc_ranks()

    def _set_conn_established(self):
        if False:
            print('Hello World!')
        self.conn_established_for_type.update({P2PConnTypes.Start: self.__connection_established})

    def _set_conn_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn_failure_for_type.update({P2PConnTypes.Start: P2PService.__connection_failure})

    def _set_conn_final_failure(self):
        if False:
            print('Hello World!')
        self.conn_final_failure_for_type.update({P2PConnTypes.Start: P2PService.__connection_final_failure})

    def _get_difficulty(self, key_id):
        if False:
            for i in range(10):
                print('nop')
        return self.base_difficulty

    def _get_challenge(self, key_id):
        if False:
            print('Hello World!')
        self.last_challenge = simplechallenge.create_challenge(self.challenge_history, self.last_challenge)
        return self.last_challenge

    def __send_get_peers(self):
        if False:
            return 10
        for p in list(self.peers.values()):
            p.send_get_peers()

    def _send_get_tasks(self):
        if False:
            return 10
        for p in list(self.peers.values()):
            p.send_get_tasks()

    def __connection_established(self, protocol: tcpnetwork.BroadcastProtocol, conn_id: str):
        if False:
            for i in range(10):
                print('nop')
        peer_conn = protocol.transport.getPeer()
        ip_address = peer_conn.host
        port = peer_conn.port
        protocol.conn_id = conn_id
        self._mark_connected(conn_id, ip_address, port)
        logger.debug('Connection to peer established. %s: %s, conn_id %s', ip_address, port, conn_id)

    @staticmethod
    def __connection_failure(conn_id: str):
        if False:
            print('Hello World!')
        logger.debug('Connection to peer failure %s.', conn_id)

    @staticmethod
    def __connection_final_failure(conn_id: str):
        if False:
            for i in range(10):
                print('nop')
        logger.debug("Can't connect to peer %s.", conn_id)

    def __is_new_peer(self, id_):
        if False:
            return 10
        return id_ not in self.incoming_peers and (not self.__is_connected_peer(id_))

    def __is_connected_peer(self, id_):
        if False:
            while True:
                i = 10
        return id_ in self.peers or int(id_, 16) == self.get_key_id()

    def __remove_old_peers(self):
        if False:
            print('Hello World!')
        for peer in list(self.peers.values()):
            delta = time.time() - peer.last_message_time
            if delta > self.last_message_time_threshold:
                self.remove_peer(peer)
                peer.disconnect(message.base.Disconnect.REASON.Timeout)

    def _sync_forward_requests(self):
        if False:
            while True:
                i = 10
        helper = self.task_server.task_connections_helper
        entries = helper.forward_queue_get(FORWARD_BATCH_SIZE)
        for entry in entries:
            (peer, args) = (entry[0](), entry[1])
            if peer:
                peer.send_set_task_session(*args)

    def __sync_free_peers(self):
        if False:
            while True:
                i = 10
        while self.free_peers and (not self.enough_peers()):
            peer_id = random.choice(self.free_peers)
            self.free_peers.remove(peer_id)
            if not self.__is_connected_peer(peer_id):
                peer = self.incoming_peers[peer_id]
                node = peer['node']
                self.incoming_peers[peer_id]['conn_trials'] += 1
                self._add_pending_request(P2PConnTypes.Start, node, prv_port=node.p2p_prv_port, pub_port=node.p2p_pub_port, args={})

    def __sync_peer_keeper(self):
        if False:
            print('Hello World!')
        self.__remove_sessions_to_end_from_peer_keeper()
        peers_to_find: Dict[int, List[dt_p2p.Node]] = self.peer_keeper.sync()
        self.__remove_sessions_to_end_from_peer_keeper()
        if peers_to_find:
            self.send_find_nodes(peers_to_find)

    def _sync_seeds(self, known_hosts=None):
        if False:
            while True:
                i = 10
        self.last_seeds_sync = time.time()
        if not known_hosts:
            known_hosts = KnownHosts.select().where(KnownHosts.is_seed)

        def _resolve_hostname(host, port):
            if False:
                return 10
            try:
                port = int(port)
            except ValueError:
                logger.info('Invalid seed: %s:%s. Ignoring.', host, port)
                return
            if not (host and port):
                logger.debug('Ignoring incomplete seed. host=%r port=%r', host, port)
                return
            try:
                for addrinfo in socket.getaddrinfo(host, port):
                    yield addrinfo[4]
            except OSError as e:
                logger.error("Can't resolve %s:%s. %s", host, port, e)
        self.seeds = set()
        ip_address = self.config_desc.seed_host or ''
        port = self.config_desc.seed_port
        for hostport in itertools.chain(((kh.ip_address, kh.port) for kh in known_hosts if kh.is_seed), self.bootstrap_seeds, ((ip_address, port),), (cs.split(':', 1) for cs in self.config_desc.seeds.split(None))):
            self.seeds.update(_resolve_hostname(*hostport))

    def _get_next_random_seed(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            for seed in random.sample(self.seeds, k=len(self.seeds)):
                if seed not in self.used_seeds:
                    self.used_seeds.add(seed)
                    return seed
            self.used_seeds = set()

    def __remove_sessions_to_end_from_peer_keeper(self):
        if False:
            print('Hello World!')
        for node in self.peer_keeper.sessions_to_end:
            self.remove_peer_by_id(node.key)
        self.peer_keeper.sessions_to_end = []

    def _disconnect_random_peers(self) -> None:
        if False:
            while True:
                i = 10
        peers = list(self.peers.values())
        if len(peers) < self.config_desc.opt_peer_num:
            return
        logger.info('Disconnecting random peers')
        for peer in random.sample(peers, k=int(len(peers) * RANDOM_DISCONNECT_FRACTION)):
            logger.info('Disconnecting peer %r', peer.key_id)
            self.remove_peer(peer)
            peer.disconnect(message.base.Disconnect.REASON.Refresh)

    @staticmethod
    def __remove_redundant_hosts_from_db():
        if False:
            print('Hello World!')
        to_delete = KnownHosts.select().order_by(KnownHosts.last_connected.desc()).offset(MAX_STORED_HOSTS)
        KnownHosts.delete().where(KnownHosts.id << to_delete).execute()

class P2PConnTypes(object):
    """ P2P Connection Types that allows to choose right reaction  """
    Start = 1