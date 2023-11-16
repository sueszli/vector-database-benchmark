from __future__ import absolute_import, division
import collections
import copy
import logging
import random
import socket
import threading
import time
import weakref
try:
    import selectors
except ImportError:
    from kafka.vendor import selectors34 as selectors
from kafka.vendor import six
from kafka.cluster import ClusterMetadata
from kafka.conn import BrokerConnection, ConnectionStates, collect_hosts, get_ip_port_afi
from kafka import errors as Errors
from kafka.future import Future
from kafka.metrics import AnonMeasurable
from kafka.metrics.stats import Avg, Count, Rate
from kafka.metrics.stats.rate import TimeUnit
from kafka.protocol.metadata import MetadataRequest
from kafka.util import Dict, WeakMethod
from kafka.vendor import socketpair
from kafka.version import __version__
if six.PY2:
    ConnectionError = None
log = logging.getLogger('kafka.client')

class KafkaClient(object):
    """
    A network client for asynchronous request/response network I/O.

    This is an internal class used to implement the user-facing producer and
    consumer clients.

    This class is not thread-safe!

    Attributes:
        cluster (:any:`ClusterMetadata`): Local cache of cluster metadata, retrieved
            via MetadataRequests during :meth:`~kafka.KafkaClient.poll`.

    Keyword Arguments:
        bootstrap_servers: 'host[:port]' string (or list of 'host[:port]'
            strings) that the client should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default port is 9092. If no servers are
            specified, will default to localhost:9092.
        client_id (str): a name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client. Also
            submitted to GroupCoordinator for logging with respect to
            consumer group administration. Default: 'kafka-python-{version}'
        reconnect_backoff_ms (int): The amount of time in milliseconds to
            wait before attempting to reconnect to a given host.
            Default: 50.
        reconnect_backoff_max_ms (int): The maximum amount of time in
            milliseconds to backoff/wait when reconnecting to a broker that has
            repeatedly failed to connect. If provided, the backoff per host
            will increase exponentially for each consecutive connection
            failure, up to this maximum. Once the maximum is reached,
            reconnection attempts will continue periodically with this fixed
            rate. To avoid connection storms, a randomization factor of 0.2
            will be applied to the backoff resulting in a random range between
            20% below and 20% above the computed value. Default: 1000.
        request_timeout_ms (int): Client request timeout in milliseconds.
            Default: 30000.
        connections_max_idle_ms: Close idle connections after the number of
            milliseconds specified by this config. The broker closes idle
            connections after connections.max.idle.ms, so this avoids hitting
            unexpected socket disconnected errors on the client.
            Default: 540000
        retry_backoff_ms (int): Milliseconds to backoff when retrying on
            errors. Default: 100.
        max_in_flight_requests_per_connection (int): Requests are pipelined
            to kafka brokers up to this number of maximum requests per
            broker connection. Default: 5.
        receive_buffer_bytes (int): The size of the TCP receive buffer
            (SO_RCVBUF) to use when reading data. Default: None (relies on
            system defaults). Java client defaults to 32768.
        send_buffer_bytes (int): The size of the TCP send buffer
            (SO_SNDBUF) to use when sending data. Default: None (relies on
            system defaults). Java client defaults to 131072.
        socket_options (list): List of tuple-arguments to socket.setsockopt
            to apply to broker connection sockets. Default:
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        metadata_max_age_ms (int): The period of time in milliseconds after
            which we force a refresh of metadata even if we haven't seen any
            partition leadership changes to proactively discover any new
            brokers or partitions. Default: 300000
        security_protocol (str): Protocol used to communicate with brokers.
            Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            Default: PLAINTEXT.
        ssl_context (ssl.SSLContext): Pre-configured SSLContext for wrapping
            socket connections. If provided, all other ssl_* configurations
            will be ignored. Default: None.
        ssl_check_hostname (bool): Flag to configure whether SSL handshake
            should verify that the certificate matches the broker's hostname.
            Default: True.
        ssl_cafile (str): Optional filename of CA file to use in certificate
            verification. Default: None.
        ssl_certfile (str): Optional filename of file in PEM format containing
            the client certificate, as well as any CA certificates needed to
            establish the certificate's authenticity. Default: None.
        ssl_keyfile (str): Optional filename containing the client private key.
            Default: None.
        ssl_password (str): Optional password to be used when loading the
            certificate chain. Default: None.
        ssl_crlfile (str): Optional filename containing the CRL to check for
            certificate expiration. By default, no CRL check is done. When
            providing a file, only the leaf certificate will be checked against
            this CRL. The CRL can only be checked with Python 3.4+ or 2.7.9+.
            Default: None.
        ssl_ciphers (str): optionally set the available ciphers for ssl
            connections. It should be a string in the OpenSSL cipher list
            format. If no cipher can be selected (because compile-time options
            or other configuration forbids use of all the specified ciphers),
            an ssl.SSLError will be raised. See ssl.SSLContext.set_ciphers
        api_version (tuple): Specify which Kafka API version to use. If set
            to None, KafkaClient will attempt to infer the broker version by
            probing various APIs. Example: (0, 10, 2). Default: None
        api_version_auto_timeout_ms (int): number of milliseconds to throw a
            timeout exception from the constructor when checking the broker
            api version. Only applies if api_version is None
        selector (selectors.BaseSelector): Provide a specific selector
            implementation to use for I/O multiplexing.
            Default: selectors.DefaultSelector
        metrics (kafka.metrics.Metrics): Optionally provide a metrics
            instance for capturing network IO stats. Default: None.
        metric_group_prefix (str): Prefix for metric names. Default: ''
        sasl_mechanism (str): Authentication mechanism when security_protocol
            is configured for SASL_PLAINTEXT or SASL_SSL. Valid values are:
            PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256, SCRAM-SHA-512.
        sasl_plain_username (str): username for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_plain_password (str): password for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_kerberos_service_name (str): Service name to include in GSSAPI
            sasl mechanism handshake. Default: 'kafka'
        sasl_kerberos_domain_name (str): kerberos domain name to use in GSSAPI
            sasl mechanism handshake. Default: one of bootstrap servers
        sasl_oauth_token_provider (AbstractTokenProvider): OAuthBearer token provider
            instance. (See kafka.oauth.abstract). Default: None
    """
    DEFAULT_CONFIG = {'bootstrap_servers': 'localhost', 'bootstrap_topics_filter': set(), 'client_id': 'kafka-python-' + __version__, 'request_timeout_ms': 30000, 'wakeup_timeout_ms': 3000, 'connections_max_idle_ms': 9 * 60 * 1000, 'reconnect_backoff_ms': 50, 'reconnect_backoff_max_ms': 1000, 'max_in_flight_requests_per_connection': 5, 'receive_buffer_bytes': None, 'send_buffer_bytes': None, 'socket_options': [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)], 'sock_chunk_bytes': 4096, 'sock_chunk_buffer_count': 1000, 'retry_backoff_ms': 100, 'metadata_max_age_ms': 300000, 'security_protocol': 'PLAINTEXT', 'ssl_context': None, 'ssl_check_hostname': True, 'ssl_cafile': None, 'ssl_certfile': None, 'ssl_keyfile': None, 'ssl_password': None, 'ssl_crlfile': None, 'ssl_ciphers': None, 'api_version': None, 'api_version_auto_timeout_ms': 2000, 'selector': selectors.DefaultSelector, 'metrics': None, 'metric_group_prefix': '', 'sasl_mechanism': None, 'sasl_plain_username': None, 'sasl_plain_password': None, 'sasl_kerberos_service_name': 'kafka', 'sasl_kerberos_domain_name': None, 'sasl_oauth_token_provider': None}

    def __init__(self, **configs):
        if False:
            while True:
                i = 10
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]
        self._closed = False
        (self._wake_r, self._wake_w) = socket.socketpair()
        self._selector = self.config['selector']()
        self.cluster = ClusterMetadata(**self.config)
        self._topics = set()
        self._metadata_refresh_in_progress = False
        self._conns = Dict()
        self._api_versions = None
        self._connecting = set()
        self._sending = set()
        self._refresh_on_disconnects = True
        self._last_bootstrap = 0
        self._bootstrap_fails = 0
        self._wake_r.setblocking(False)
        self._wake_w.settimeout(self.config['wakeup_timeout_ms'] / 1000.0)
        self._wake_lock = threading.Lock()
        self._lock = threading.RLock()
        self._pending_completion = collections.deque()
        self._selector.register(self._wake_r, selectors.EVENT_READ)
        self._idle_expiry_manager = IdleConnectionManager(self.config['connections_max_idle_ms'])
        self._sensors = None
        if self.config['metrics']:
            self._sensors = KafkaClientMetrics(self.config['metrics'], self.config['metric_group_prefix'], weakref.proxy(self._conns))
        self._num_bootstrap_hosts = len(collect_hosts(self.config['bootstrap_servers']))
        if self.config['api_version'] is None:
            check_timeout = self.config['api_version_auto_timeout_ms'] / 1000
            self.config['api_version'] = self.check_version(timeout=check_timeout)

    def _can_bootstrap(self):
        if False:
            return 10
        effective_failures = self._bootstrap_fails // self._num_bootstrap_hosts
        backoff_factor = 2 ** effective_failures
        backoff_ms = min(self.config['reconnect_backoff_ms'] * backoff_factor, self.config['reconnect_backoff_max_ms'])
        backoff_ms *= random.uniform(0.8, 1.2)
        next_at = self._last_bootstrap + backoff_ms / 1000.0
        now = time.time()
        if next_at > now:
            return False
        return True

    def _can_connect(self, node_id):
        if False:
            while True:
                i = 10
        if node_id not in self._conns:
            if self.cluster.broker_metadata(node_id):
                return True
            return False
        conn = self._conns[node_id]
        return conn.disconnected() and (not conn.blacked_out())

    def _conn_state_change(self, node_id, sock, conn):
        if False:
            i = 10
            return i + 15
        with self._lock:
            if conn.connecting():
                if node_id not in self._connecting:
                    self._connecting.add(node_id)
                try:
                    self._selector.register(sock, selectors.EVENT_WRITE, conn)
                except KeyError:
                    self._selector.modify(sock, selectors.EVENT_WRITE, conn)
                if self.cluster.is_bootstrap(node_id):
                    self._last_bootstrap = time.time()
            elif conn.connected():
                log.debug('Node %s connected', node_id)
                if node_id in self._connecting:
                    self._connecting.remove(node_id)
                try:
                    self._selector.modify(sock, selectors.EVENT_READ, conn)
                except KeyError:
                    self._selector.register(sock, selectors.EVENT_READ, conn)
                if self._sensors:
                    self._sensors.connection_created.record()
                self._idle_expiry_manager.update(node_id)
                if self.cluster.is_bootstrap(node_id):
                    self._bootstrap_fails = 0
                else:
                    for node_id in list(self._conns.keys()):
                        if self.cluster.is_bootstrap(node_id):
                            self._conns.pop(node_id).close()
            elif conn.state is ConnectionStates.DISCONNECTED:
                if node_id in self._connecting:
                    self._connecting.remove(node_id)
                try:
                    self._selector.unregister(sock)
                except KeyError:
                    pass
                if self._sensors:
                    self._sensors.connection_closed.record()
                idle_disconnect = False
                if self._idle_expiry_manager.is_expired(node_id):
                    idle_disconnect = True
                self._idle_expiry_manager.remove(node_id)
                if node_id not in self._conns:
                    pass
                elif self.cluster.is_bootstrap(node_id):
                    self._bootstrap_fails += 1
                elif self._refresh_on_disconnects and (not self._closed) and (not idle_disconnect):
                    log.warning('Node %s connection failed -- refreshing metadata', node_id)
                    self.cluster.request_update()

    def maybe_connect(self, node_id, wakeup=True):
        if False:
            while True:
                i = 10
        'Queues a node for asynchronous connection during the next .poll()'
        if self._can_connect(node_id):
            self._connecting.add(node_id)
            if wakeup:
                self.wakeup()
            return True
        return False

    def _should_recycle_connection(self, conn):
        if False:
            print('Hello World!')
        if not conn.disconnected():
            return False
        broker = self.cluster.broker_metadata(conn.node_id)
        if broker is None:
            return False
        (host, _, afi) = get_ip_port_afi(broker.host)
        if conn.host != host or conn.port != broker.port:
            log.info('Broker metadata change detected for node %s from %s:%s to %s:%s', conn.node_id, conn.host, conn.port, broker.host, broker.port)
            return True
        return False

    def _maybe_connect(self, node_id):
        if False:
            while True:
                i = 10
        'Idempotent non-blocking connection attempt to the given node id.'
        with self._lock:
            conn = self._conns.get(node_id)
            if conn is None:
                broker = self.cluster.broker_metadata(node_id)
                assert broker, 'Broker id %s not in current metadata' % (node_id,)
                log.debug('Initiating connection to node %s at %s:%s', node_id, broker.host, broker.port)
                (host, port, afi) = get_ip_port_afi(broker.host)
                cb = WeakMethod(self._conn_state_change)
                conn = BrokerConnection(host, broker.port, afi, state_change_callback=cb, node_id=node_id, **self.config)
                self._conns[node_id] = conn
            elif self._should_recycle_connection(conn):
                self._conns.pop(node_id)
                return False
            elif conn.connected():
                return True
            conn.connect()
            return conn.connected()

    def ready(self, node_id, metadata_priority=True):
        if False:
            while True:
                i = 10
        'Check whether a node is connected and ok to send more requests.\n\n        Arguments:\n            node_id (int): the id of the node to check\n            metadata_priority (bool): Mark node as not-ready if a metadata\n                refresh is required. Default: True\n\n        Returns:\n            bool: True if we are ready to send to the given node\n        '
        self.maybe_connect(node_id)
        return self.is_ready(node_id, metadata_priority=metadata_priority)

    def connected(self, node_id):
        if False:
            while True:
                i = 10
        'Return True iff the node_id is connected.'
        conn = self._conns.get(node_id)
        if conn is None:
            return False
        return conn.connected()

    def _close(self):
        if False:
            return 10
        if not self._closed:
            self._closed = True
            self._wake_r.close()
            self._wake_w.close()
            self._selector.close()

    def close(self, node_id=None):
        if False:
            print('Hello World!')
        'Close one or all broker connections.\n\n        Arguments:\n            node_id (int, optional): the id of the node to close\n        '
        with self._lock:
            if node_id is None:
                self._close()
                conns = list(self._conns.values())
                self._conns.clear()
                for conn in conns:
                    conn.close()
            elif node_id in self._conns:
                self._conns.pop(node_id).close()
            else:
                log.warning('Node %s not found in current connection list; skipping', node_id)
                return

    def __del__(self):
        if False:
            return 10
        self._close()

    def is_disconnected(self, node_id):
        if False:
            print('Hello World!')
        'Check whether the node connection has been disconnected or failed.\n\n        A disconnected node has either been closed or has failed. Connection\n        failures are usually transient and can be resumed in the next ready()\n        call, but there are cases where transient failures need to be caught\n        and re-acted upon.\n\n        Arguments:\n            node_id (int): the id of the node to check\n\n        Returns:\n            bool: True iff the node exists and is disconnected\n        '
        conn = self._conns.get(node_id)
        if conn is None:
            return False
        return conn.disconnected()

    def connection_delay(self, node_id):
        if False:
            while True:
                i = 10
        '\n        Return the number of milliseconds to wait, based on the connection\n        state, before attempting to send data. When disconnected, this respects\n        the reconnect backoff time. When connecting, returns 0 to allow\n        non-blocking connect to finish. When connected, returns a very large\n        number to handle slow/stalled connections.\n\n        Arguments:\n            node_id (int): The id of the node to check\n\n        Returns:\n            int: The number of milliseconds to wait.\n        '
        conn = self._conns.get(node_id)
        if conn is None:
            return 0
        return conn.connection_delay()

    def is_ready(self, node_id, metadata_priority=True):
        if False:
            print('Hello World!')
        'Check whether a node is ready to send more requests.\n\n        In addition to connection-level checks, this method also is used to\n        block additional requests from being sent during a metadata refresh.\n\n        Arguments:\n            node_id (int): id of the node to check\n            metadata_priority (bool): Mark node as not-ready if a metadata\n                refresh is required. Default: True\n\n        Returns:\n            bool: True if the node is ready and metadata is not refreshing\n        '
        if not self._can_send_request(node_id):
            return False
        if metadata_priority:
            if self._metadata_refresh_in_progress:
                return False
            if self.cluster.ttl() == 0:
                return False
        return True

    def _can_send_request(self, node_id):
        if False:
            print('Hello World!')
        conn = self._conns.get(node_id)
        if not conn:
            return False
        return conn.connected() and conn.can_send_more()

    def send(self, node_id, request, wakeup=True):
        if False:
            print('Hello World!')
        'Send a request to a specific node. Bytes are placed on an\n        internal per-connection send-queue. Actual network I/O will be\n        triggered in a subsequent call to .poll()\n\n        Arguments:\n            node_id (int): destination node\n            request (Struct): request object (not-encoded)\n            wakeup (bool): optional flag to disable thread-wakeup\n\n        Raises:\n            AssertionError: if node_id is not in current cluster metadata\n\n        Returns:\n            Future: resolves to Response struct or Error\n        '
        conn = self._conns.get(node_id)
        if not conn or not self._can_send_request(node_id):
            self.maybe_connect(node_id, wakeup=wakeup)
            return Future().failure(Errors.NodeNotReadyError(node_id))
        future = conn.send(request, blocking=False)
        self._sending.add(conn)
        if wakeup:
            self.wakeup()
        return future

    def poll(self, timeout_ms=None, future=None):
        if False:
            i = 10
            return i + 15
        'Try to read and write to sockets.\n\n        This method will also attempt to complete node connections, refresh\n        stale metadata, and run previously-scheduled tasks.\n\n        Arguments:\n            timeout_ms (int, optional): maximum amount of time to wait (in ms)\n                for at least one response. Must be non-negative. The actual\n                timeout will be the minimum of timeout, request timeout and\n                metadata timeout. Default: request_timeout_ms\n            future (Future, optional): if provided, blocks until future.is_done\n\n        Returns:\n            list: responses received (can be empty)\n        '
        if future is not None:
            timeout_ms = 100
        elif timeout_ms is None:
            timeout_ms = self.config['request_timeout_ms']
        elif not isinstance(timeout_ms, (int, float)):
            raise TypeError('Invalid type for timeout: %s' % type(timeout_ms))
        responses = []
        while True:
            with self._lock:
                if self._closed:
                    break
                for node_id in list(self._connecting):
                    self._maybe_connect(node_id)
                metadata_timeout_ms = self._maybe_refresh_metadata()
                if future is not None and future.is_done:
                    timeout = 0
                else:
                    idle_connection_timeout_ms = self._idle_expiry_manager.next_check_ms()
                    timeout = min(timeout_ms, metadata_timeout_ms, idle_connection_timeout_ms, self.config['request_timeout_ms'])
                    if self.in_flight_request_count() == 0:
                        timeout = min(timeout, self.config['retry_backoff_ms'])
                    timeout = max(0, timeout)
                self._poll(timeout / 1000)
            responses.extend(self._fire_pending_completed_requests())
            if future is None or future.is_done:
                break
        return responses

    def _register_send_sockets(self):
        if False:
            while True:
                i = 10
        while self._sending:
            conn = self._sending.pop()
            try:
                key = self._selector.get_key(conn._sock)
                events = key.events | selectors.EVENT_WRITE
                self._selector.modify(key.fileobj, events, key.data)
            except KeyError:
                self._selector.register(conn._sock, selectors.EVENT_WRITE, conn)

    def _poll(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        processed = set()
        self._register_send_sockets()
        start_select = time.time()
        ready = self._selector.select(timeout)
        end_select = time.time()
        if self._sensors:
            self._sensors.select_time.record((end_select - start_select) * 1000000000)
        for (key, events) in ready:
            if key.fileobj is self._wake_r:
                self._clear_wake_fd()
                continue
            if events & selectors.EVENT_WRITE:
                conn = key.data
                if conn.connecting():
                    conn.connect()
                elif conn.send_pending_requests_v2():
                    if key.events ^ selectors.EVENT_WRITE:
                        self._selector.modify(key.fileobj, key.events ^ selectors.EVENT_WRITE, key.data)
                    else:
                        self._selector.unregister(key.fileobj)
            if not events & selectors.EVENT_READ:
                continue
            conn = key.data
            processed.add(conn)
            if not conn.in_flight_requests:
                try:
                    unexpected_data = key.fileobj.recv(1)
                    if unexpected_data:
                        log.warning('Protocol out of sync on %r, closing', conn)
                except socket.error:
                    pass
                conn.close(Errors.KafkaConnectionError('Socket EVENT_READ without in-flight-requests'))
                continue
            self._idle_expiry_manager.update(conn.node_id)
            self._pending_completion.extend(conn.recv())
        if self.config['security_protocol'] in ('SSL', 'SASL_SSL'):
            for conn in self._conns.values():
                if conn not in processed and conn.connected() and conn._sock.pending():
                    self._pending_completion.extend(conn.recv())
        for conn in six.itervalues(self._conns):
            if conn.requests_timed_out():
                log.warning('%s timed out after %s ms. Closing connection.', conn, conn.config['request_timeout_ms'])
                conn.close(error=Errors.RequestTimedOutError('Request timed out after %s ms' % conn.config['request_timeout_ms']))
        if self._sensors:
            self._sensors.io_time.record((time.time() - end_select) * 1000000000)
        self._maybe_close_oldest_connection()

    def in_flight_request_count(self, node_id=None):
        if False:
            i = 10
            return i + 15
        'Get the number of in-flight requests for a node or all nodes.\n\n        Arguments:\n            node_id (int, optional): a specific node to check. If unspecified,\n                return the total for all nodes\n\n        Returns:\n            int: pending in-flight requests for the node, or all nodes if None\n        '
        if node_id is not None:
            conn = self._conns.get(node_id)
            if conn is None:
                return 0
            return len(conn.in_flight_requests)
        else:
            return sum([len(conn.in_flight_requests) for conn in list(self._conns.values())])

    def _fire_pending_completed_requests(self):
        if False:
            while True:
                i = 10
        responses = []
        while True:
            try:
                (response, future) = self._pending_completion.popleft()
            except IndexError:
                break
            future.success(response)
            responses.append(response)
        return responses

    def least_loaded_node(self):
        if False:
            return 10
        'Choose the node with fewest outstanding requests, with fallbacks.\n\n        This method will prefer a node with an existing connection and no\n        in-flight-requests. If no such node is found, a node will be chosen\n        randomly from disconnected nodes that are not "blacked out" (i.e.,\n        are not subject to a reconnect backoff). If no node metadata has been\n        obtained, will return a bootstrap node (subject to exponential backoff).\n\n        Returns:\n            node_id or None if no suitable node was found\n        '
        nodes = [broker.nodeId for broker in self.cluster.brokers()]
        random.shuffle(nodes)
        inflight = float('inf')
        found = None
        for node_id in nodes:
            conn = self._conns.get(node_id)
            connected = conn is not None and conn.connected()
            blacked_out = conn is not None and conn.blacked_out()
            curr_inflight = len(conn.in_flight_requests) if conn is not None else 0
            if connected and curr_inflight == 0:
                return node_id
            elif not blacked_out and curr_inflight < inflight:
                inflight = curr_inflight
                found = node_id
        return found

    def set_topics(self, topics):
        if False:
            return 10
        'Set specific topics to track for metadata.\n\n        Arguments:\n            topics (list of str): topics to check for metadata\n\n        Returns:\n            Future: resolves after metadata request/response\n        '
        if set(topics).difference(self._topics):
            future = self.cluster.request_update()
        else:
            future = Future().success(set(topics))
        self._topics = set(topics)
        return future

    def add_topic(self, topic):
        if False:
            return 10
        'Add a topic to the list of topics tracked via metadata.\n\n        Arguments:\n            topic (str): topic to track\n\n        Returns:\n            Future: resolves after metadata request/response\n        '
        if topic in self._topics:
            return Future().success(set(self._topics))
        self._topics.add(topic)
        return self.cluster.request_update()

    def _maybe_refresh_metadata(self, wakeup=False):
        if False:
            while True:
                i = 10
        'Send a metadata request if needed.\n\n        Returns:\n            int: milliseconds until next refresh\n        '
        ttl = self.cluster.ttl()
        wait_for_in_progress_ms = self.config['request_timeout_ms'] if self._metadata_refresh_in_progress else 0
        metadata_timeout = max(ttl, wait_for_in_progress_ms)
        if metadata_timeout > 0:
            return metadata_timeout
        node_id = self.least_loaded_node()
        if node_id is None:
            log.debug('Give up sending metadata request since no node is available')
            return self.config['reconnect_backoff_ms']
        if self._can_send_request(node_id):
            topics = list(self._topics)
            if not topics and self.cluster.is_bootstrap(node_id):
                topics = list(self.config['bootstrap_topics_filter'])
            if self.cluster.need_all_topic_metadata or not topics:
                topics = [] if self.config['api_version'] < (0, 10) else None
            api_version = 0 if self.config['api_version'] < (0, 10) else 1
            request = MetadataRequest[api_version](topics)
            log.debug('Sending metadata request %s to node %s', request, node_id)
            future = self.send(node_id, request, wakeup=wakeup)
            future.add_callback(self.cluster.update_metadata)
            future.add_errback(self.cluster.failed_update)
            self._metadata_refresh_in_progress = True

            def refresh_done(val_or_error):
                if False:
                    while True:
                        i = 10
                self._metadata_refresh_in_progress = False
            future.add_callback(refresh_done)
            future.add_errback(refresh_done)
            return self.config['request_timeout_ms']
        if self._connecting:
            return self.config['reconnect_backoff_ms']
        if self.maybe_connect(node_id, wakeup=wakeup):
            log.debug('Initializing connection to node %s for metadata request', node_id)
            return self.config['reconnect_backoff_ms']
        return float('inf')

    def get_api_versions(self):
        if False:
            return 10
        'Return the ApiVersions map, if available.\n\n        Note: A call to check_version must previously have succeeded and returned\n        version 0.10.0 or later\n\n        Returns: a map of dict mapping {api_key : (min_version, max_version)},\n        or None if ApiVersion is not supported by the kafka cluster.\n        '
        return self._api_versions

    def check_version(self, node_id=None, timeout=2, strict=False):
        if False:
            while True:
                i = 10
        'Attempt to guess the version of a Kafka broker.\n\n        Note: It is possible that this method blocks longer than the\n            specified timeout. This can happen if the entire cluster\n            is down and the client enters a bootstrap backoff sleep.\n            This is only possible if node_id is None.\n\n        Returns: version tuple, i.e. (0, 10), (0, 9), (0, 8, 2), ...\n\n        Raises:\n            NodeNotReadyError (if node_id is provided)\n            NoBrokersAvailable (if node_id is None)\n            UnrecognizedBrokerVersion: please file bug if seen!\n            AssertionError (if strict=True): please file bug if seen!\n        '
        self._lock.acquire()
        end = time.time() + timeout
        while time.time() < end:
            try_node = node_id or self.least_loaded_node()
            if try_node is None:
                self._lock.release()
                raise Errors.NoBrokersAvailable()
            self._maybe_connect(try_node)
            conn = self._conns[try_node]
            self._refresh_on_disconnects = False
            try:
                remaining = end - time.time()
                version = conn.check_version(timeout=remaining, strict=strict, topics=list(self.config['bootstrap_topics_filter']))
                if version >= (0, 10, 0):
                    self._api_versions = conn.get_api_versions()
                self._lock.release()
                return version
            except Errors.NodeNotReadyError:
                if node_id is not None:
                    self._lock.release()
                    raise
            finally:
                self._refresh_on_disconnects = True
        else:
            self._lock.release()
            raise Errors.NoBrokersAvailable()

    def wakeup(self):
        if False:
            while True:
                i = 10
        with self._wake_lock:
            try:
                self._wake_w.sendall(b'x')
            except socket.timeout:
                log.warning('Timeout to send to wakeup socket!')
                raise Errors.KafkaTimeoutError()
            except socket.error:
                log.warning('Unable to send to wakeup socket!')

    def _clear_wake_fd(self):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                self._wake_r.recv(1024)
            except socket.error:
                break

    def _maybe_close_oldest_connection(self):
        if False:
            return 10
        expired_connection = self._idle_expiry_manager.poll_expired_connection()
        if expired_connection:
            (conn_id, ts) = expired_connection
            idle_ms = (time.time() - ts) * 1000
            log.info('Closing idle connection %s, last active %d ms ago', conn_id, idle_ms)
            self.close(node_id=conn_id)

    def bootstrap_connected(self):
        if False:
            return 10
        'Return True if a bootstrap node is connected'
        for node_id in self._conns:
            if not self.cluster.is_bootstrap(node_id):
                continue
            if self._conns[node_id].connected():
                return True
        else:
            return False
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

class IdleConnectionManager(object):

    def __init__(self, connections_max_idle_ms):
        if False:
            print('Hello World!')
        if connections_max_idle_ms > 0:
            self.connections_max_idle = connections_max_idle_ms / 1000
        else:
            self.connections_max_idle = float('inf')
        self.next_idle_close_check_time = None
        self.update_next_idle_close_check_time(time.time())
        self.lru_connections = OrderedDict()

    def update(self, conn_id):
        if False:
            return 10
        if conn_id in self.lru_connections:
            del self.lru_connections[conn_id]
        self.lru_connections[conn_id] = time.time()

    def remove(self, conn_id):
        if False:
            i = 10
            return i + 15
        if conn_id in self.lru_connections:
            del self.lru_connections[conn_id]

    def is_expired(self, conn_id):
        if False:
            while True:
                i = 10
        if conn_id not in self.lru_connections:
            return None
        return time.time() >= self.lru_connections[conn_id] + self.connections_max_idle

    def next_check_ms(self):
        if False:
            print('Hello World!')
        now = time.time()
        if not self.lru_connections:
            return float('inf')
        elif self.next_idle_close_check_time <= now:
            return 0
        else:
            return int((self.next_idle_close_check_time - now) * 1000)

    def update_next_idle_close_check_time(self, ts):
        if False:
            while True:
                i = 10
        self.next_idle_close_check_time = ts + self.connections_max_idle

    def poll_expired_connection(self):
        if False:
            for i in range(10):
                print('nop')
        if time.time() < self.next_idle_close_check_time:
            return None
        if not len(self.lru_connections):
            return None
        oldest_conn_id = None
        oldest_ts = None
        if OrderedDict is dict:
            for (conn_id, ts) in self.lru_connections.items():
                if oldest_conn_id is None or ts < oldest_ts:
                    oldest_conn_id = conn_id
                    oldest_ts = ts
        else:
            (oldest_conn_id, oldest_ts) = next(iter(self.lru_connections.items()))
        self.update_next_idle_close_check_time(oldest_ts)
        if time.time() >= oldest_ts + self.connections_max_idle:
            return (oldest_conn_id, oldest_ts)
        else:
            return None

class KafkaClientMetrics(object):

    def __init__(self, metrics, metric_group_prefix, conns):
        if False:
            while True:
                i = 10
        self.metrics = metrics
        self.metric_group_name = metric_group_prefix + '-metrics'
        self.connection_closed = metrics.sensor('connections-closed')
        self.connection_closed.add(metrics.metric_name('connection-close-rate', self.metric_group_name, 'Connections closed per second in the window.'), Rate())
        self.connection_created = metrics.sensor('connections-created')
        self.connection_created.add(metrics.metric_name('connection-creation-rate', self.metric_group_name, 'New connections established per second in the window.'), Rate())
        self.select_time = metrics.sensor('select-time')
        self.select_time.add(metrics.metric_name('select-rate', self.metric_group_name, 'Number of times the I/O layer checked for new I/O to perform per second'), Rate(sampled_stat=Count()))
        self.select_time.add(metrics.metric_name('io-wait-time-ns-avg', self.metric_group_name, 'The average length of time the I/O thread spent waiting for a socket ready for reads or writes in nanoseconds.'), Avg())
        self.select_time.add(metrics.metric_name('io-wait-ratio', self.metric_group_name, 'The fraction of time the I/O thread spent waiting.'), Rate(time_unit=TimeUnit.NANOSECONDS))
        self.io_time = metrics.sensor('io-time')
        self.io_time.add(metrics.metric_name('io-time-ns-avg', self.metric_group_name, 'The average length of time for I/O per select call in nanoseconds.'), Avg())
        self.io_time.add(metrics.metric_name('io-ratio', self.metric_group_name, 'The fraction of time the I/O thread spent doing I/O'), Rate(time_unit=TimeUnit.NANOSECONDS))
        metrics.add_metric(metrics.metric_name('connection-count', self.metric_group_name, 'The current number of active connections.'), AnonMeasurable(lambda config, now: len(conns)))