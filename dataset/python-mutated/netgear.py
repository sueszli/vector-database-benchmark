"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""
import os
import time
import string
import secrets
import numpy as np
import logging as log
from threading import Thread
from collections import deque
from os.path import expanduser
from .helper import logger_handler, generate_auth_certificates, check_WriteAccess, check_open_port, import_dependency_safe, logcurr_vidgear_ver
zmq = import_dependency_safe('zmq', pkg_name='pyzmq', error='silent', min_version='4.0')
if not zmq is None:
    from zmq import ssh
    from zmq import auth
    from zmq.auth.thread import ThreadAuthenticator
    from zmq.error import ZMQError
simplejpeg = import_dependency_safe('simplejpeg', error='silent', min_version='1.6.1')
paramiko = import_dependency_safe('paramiko', error='silent')
logger = log.getLogger('NetGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class NetGear:
    """
    NetGear is exclusively designed to transfer video frames synchronously and asynchronously between interconnecting systems over the network in real-time.

    NetGear implements a high-level wrapper around PyZmQ python library that contains python bindings for ZeroMQ - a high-performance asynchronous distributed messaging library
    that provides a message queue, but unlike message-oriented middleware, its system can run without a dedicated message broker.

    NetGear also supports real-time Frame Compression capabilities for optimizing performance while sending the frames directly over the network, by encoding the frame before sending
    it and decoding it on the client's end automatically in real-time.

    !!! info
        NetGear API now internally implements robust *Lazy Pirate pattern* (auto-reconnection) for its synchronous messaging patterns _(i.e. `zmq.PAIR` & `zmq.REQ/zmq.REP`)_
        at both Server and Client ends, where its API instead of doing a blocking receive, will:

        * Poll the socket and receive from it only when it's sure a reply has arrived.
        * Attempt to reconnect, if no reply has arrived within a timeout period.
        * Abandon the connection if there is still no reply after several requests.

    NetGear as of now seamlessly supports three ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_

    _whereas the supported protocol are: `tcp` and `ipc`_.

    ??? tip "Modes of Operation"

        * **Primary Modes**

            NetGear API primarily has two modes of operations:

            * **Send Mode:** _which employs `send()` function to send video frames over the network in real-time._

            * **Receive Mode:** _which employs `recv()` function to receive frames, sent over the network with *Send Mode* in real-time. The mode sends back confirmation when the
            frame is received successfully in few patterns._

        * **Exclusive Modes**

            In addition to these primary modes, NetGear API offers applications-specific Exclusive Modes:

            * **Multi-Servers Mode:** _In this exclusive mode, NetGear API robustly **handles multiple servers at once**, thereby providing seamless access to frames and unidirectional
            data transfer from multiple Servers/Publishers across the network in real-time._

            * **Multi-Clients Mode:** _In this exclusive mode, NetGear API robustly **handles multiple clients at once**, thereby providing seamless access to frames and unidirectional
            data transfer to multiple Client/Consumers across the network in real-time._

            * **Bidirectional Mode:** _This exclusive mode **provides seamless support for bidirectional data transmission between between Server and Client along with video frames**._

            * **Secure Mode:** _In this exclusive mode, NetGear API **provides easy access to powerful, smart & secure ZeroMQ's Security Layers** that enables strong encryption on
            data, and unbreakable authentication between the Server and Client with the help of custom certificates/keys that brings cheap, standardized privacy and authentication
            for distributed systems over the network._
    """

    def __init__(self, address=None, port=None, protocol=None, pattern=0, receive_mode=False, logging=False, **options):
        if False:
            i = 10
            return i + 15
        "\n        This constructor method initializes the object state and attributes of the NetGear class.\n\n        Parameters:\n            address (str): sets the valid network address of the Server/Client.\n            port (str): sets the valid Network Port of the Server/Client.\n            protocol (str): sets the valid messaging protocol between Server/Client.\n            pattern (int): sets the supported messaging pattern(flow of communication) between Server/Client\n            receive_mode (bool): select the Netgear's Mode of operation.\n            logging (bool): enables/disables logging.\n            options (dict): provides the flexibility to alter various NetGear internal properties.\n        "
        logcurr_vidgear_ver(logging=logging)
        import_dependency_safe('zmq' if zmq is None else '', min_version='4.0', pkg_name='pyzmq')
        import_dependency_safe('simplejpeg' if simplejpeg is None else '', error='log', min_version='1.6.1')
        self.__logging = True if logging else False
        valid_messaging_patterns = {0: (zmq.PAIR, zmq.PAIR), 1: (zmq.REQ, zmq.REP), 2: (zmq.PUB, zmq.SUB)}
        msg_pattern = None
        if isinstance(pattern, int) and pattern in valid_messaging_patterns.keys():
            msg_pattern = valid_messaging_patterns[pattern]
        else:
            pattern = 0
            msg_pattern = valid_messaging_patterns[pattern]
            self.__logging and logger.warning('Wrong pattern value, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information.')
        self.__pattern = pattern
        if protocol is None or not protocol in ['tcp', 'ipc']:
            protocol = 'tcp'
            self.__logging and logger.warning('Protocol is not supported or not provided. Defaulting to `tcp` protocol!')
        self.__msg_flag = 0
        self.__msg_copy = False
        self.__msg_track = False
        self.__ssh_tunnel_mode = None
        self.__ssh_tunnel_pwd = None
        self.__ssh_tunnel_keyfile = None
        self.__paramiko_present = False if paramiko is None else True
        self.__multiserver_mode = False
        self.__multiclient_mode = False
        self.__bi_mode = False
        valid_security_mech = {0: 'Grasslands', 1: 'StoneHouse', 2: 'IronHouse'}
        self.__secure_mode = 0
        auth_cert_dir = ''
        self.__auth_publickeys_dir = ''
        self.__auth_secretkeys_dir = ''
        overwrite_cert = False
        custom_cert_location = ''
        self.__jpeg_compression = True if not simplejpeg is None else False
        self.__jpeg_compression_quality = 90
        self.__jpeg_compression_fastdct = True
        self.__jpeg_compression_fastupsample = False
        self.__jpeg_compression_colorspace = 'BGR'
        self.__ex_compression_params = None
        self.__return_data = None
        self.__id = ''.join((secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)))
        self.__terminate = False
        if pattern < 2:
            self.__poll = zmq.Poller()
            self.__max_retries = 3
            self.__request_timeout = 4000
        else:
            self.__subscriber_timeout = None
        options = {str(k).strip(): v for (k, v) in options.items()}
        for (key, value) in options.items():
            if key == 'multiserver_mode' and isinstance(value, bool):
                if pattern > 0:
                    self.__multiserver_mode = value
                else:
                    self.__multiserver_mode = False
                    logger.critical('Multi-Server Mode is disabled!')
                    raise ValueError('[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.'.format(pattern))
            elif key == 'multiclient_mode' and isinstance(value, bool):
                if pattern > 0:
                    self.__multiclient_mode = value
                else:
                    self.__multiclient_mode = False
                    logger.critical('Multi-Client Mode is disabled!')
                    raise ValueError('[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Client Mode is enabled. Kindly refer Docs for more Information.'.format(pattern))
            elif key == 'bidirectional_mode' and isinstance(value, bool):
                if pattern < 2:
                    self.__bi_mode = value
                else:
                    self.__bi_mode = False
                    logger.warning('Bidirectional data transmission is disabled!')
                    raise ValueError('[NetGear:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!'.format(pattern))
            elif key == 'secure_mode' and isinstance(value, int) and (value in valid_security_mech):
                self.__secure_mode = value
            elif key == 'custom_cert_location' and isinstance(value, str):
                custom_cert_location = os.path.abspath(value)
                assert os.path.isdir(custom_cert_location), '[NetGear:ERROR] :: `custom_cert_location` value must be the path to a valid directory!'
                assert check_WriteAccess(custom_cert_location, is_windows=True if os.name == 'nt' else False, logging=self.__logging), "[NetGear:ERROR] :: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(value)
            elif key == 'overwrite_cert' and isinstance(value, bool):
                overwrite_cert = value
            elif key == 'ssh_tunnel_mode' and isinstance(value, str):
                self.__ssh_tunnel_mode = value.strip()
            elif key == 'ssh_tunnel_pwd' and isinstance(value, str):
                self.__ssh_tunnel_pwd = value
            elif key == 'ssh_tunnel_keyfile' and isinstance(value, str):
                self.__ssh_tunnel_keyfile = value if os.path.isfile(value) else None
                if self.__ssh_tunnel_keyfile is None:
                    logger.warning('Discarded invalid or non-existential SSH Tunnel Key-file at {}!'.format(value))
            elif key == 'jpeg_compression' and (not simplejpeg is None) and isinstance(value, (bool, str)):
                if isinstance(value, str) and value.strip().upper() in ['RGB', 'BGR', 'RGBX', 'BGRX', 'XBGR', 'XRGB', 'GRAY', 'RGBA', 'BGRA', 'ABGR', 'ARGB', 'CMYK']:
                    self.__jpeg_compression_colorspace = value.strip().upper()
                    self.__jpeg_compression = True
                else:
                    self.__jpeg_compression = value
            elif key == 'jpeg_compression_quality' and isinstance(value, (int, float)):
                if value >= 10 and value <= 100:
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning('Skipped invalid `jpeg_compression_quality` value!')
            elif key == 'jpeg_compression_fastdct' and isinstance(value, bool):
                self.__jpeg_compression_fastdct = value
            elif key == 'jpeg_compression_fastupsample' and isinstance(value, bool):
                self.__jpeg_compression_fastupsample = value
            elif key == 'max_retries' and isinstance(value, int) and (pattern < 2):
                if value >= 0:
                    self.__max_retries = value
                else:
                    logger.warning('Invalid `max_retries` value skipped!')
            elif key == 'request_timeout' and isinstance(value, int) and (pattern < 2):
                if value >= 4:
                    self.__request_timeout = value * 1000
                else:
                    logger.warning('Invalid `request_timeout` value skipped!')
            elif key == 'subscriber_timeout' and isinstance(value, int) and (pattern == 2):
                if value > 0:
                    self.__subscriber_timeout = value * 1000
                else:
                    logger.warning('Invalid `request_timeout` value skipped!')
            elif key == 'flag' and isinstance(value, int):
                self.__msg_flag = value
            elif key == 'copy' and isinstance(value, bool):
                self.__msg_copy = value
            elif key == 'track' and isinstance(value, bool):
                self.__msg_track = value
            else:
                pass
        if self.__secure_mode:
            if overwrite_cert:
                if not receive_mode:
                    self.__logging and logger.warning('Overwriting ZMQ Authentication certificates over previous ones!')
                else:
                    overwrite_cert = False
                    self.__logging and logger.critical("Overwriting ZMQ Authentication certificates is disabled for Client's end!")
            try:
                if custom_cert_location:
                    (auth_cert_dir, self.__auth_secretkeys_dir, self.__auth_publickeys_dir) = generate_auth_certificates(custom_cert_location, overwrite=overwrite_cert, logging=logging)
                else:
                    (auth_cert_dir, self.__auth_secretkeys_dir, self.__auth_publickeys_dir) = generate_auth_certificates(os.path.join(expanduser('~'), '.vidgear'), overwrite=overwrite_cert, logging=logging)
                self.__logging and logger.debug('`{}` is the default location for storing ZMQ authentication certificates/keys.'.format(auth_cert_dir))
            except Exception as e:
                logger.exception(str(e))
                self.__secure_mode = 0
                logger.critical('ZMQ Security Mechanism is disabled for this connection due to errors!')
        if not self.__ssh_tunnel_mode is None:
            if receive_mode:
                logger.error('SSH Tunneling cannot be enabled for Client-end!')
            else:
                ssh_address = self.__ssh_tunnel_mode
                (ssh_address, ssh_port) = ssh_address.split(':') if ':' in ssh_address else [ssh_address, '22']
                if '47' in ssh_port:
                    self.__ssh_tunnel_mode = self.__ssh_tunnel_mode.replace(':47', '')
                else:
                    (ssh_user, ssh_ip) = ssh_address.split('@') if '@' in ssh_address else ['', ssh_address]
                    assert check_open_port(ssh_ip, port=int(ssh_port)), '[NetGear:ERROR] :: Host `{}` is not available for SSH Tunneling at port-{}!'.format(ssh_address, ssh_port)
        if self.__multiclient_mode and self.__multiserver_mode:
            raise ValueError('[NetGear:ERROR] :: Multi-Client and Multi-Server Mode cannot be enabled simultaneously!')
        elif self.__multiserver_mode or self.__multiclient_mode:
            if self.__bi_mode:
                self.__logging and logger.debug('Bidirectional Data Transmission is also enabled for this connection!')
            if self.__ssh_tunnel_mode:
                raise ValueError('[NetGear:ERROR] :: SSH Tunneling and {} Mode cannot be enabled simultaneously. Kindly refer docs!'.format('Multi-Server' if self.__multiserver_mode else 'Multi-Client'))
        elif self.__bi_mode:
            self.__logging and logger.debug('Bidirectional Data Transmission is enabled for this connection!')
        elif self.__ssh_tunnel_mode:
            self.__logging and logger.debug('SSH Tunneling is enabled for host:`{}` with `{}` back-end.'.format(self.__ssh_tunnel_mode, 'paramiko' if self.__paramiko_present else 'pexpect'))
        self.__msg_context = zmq.Context.instance()
        self.__receive_mode = receive_mode
        if self.__receive_mode:
            if address is None:
                address = '*'
            if self.__multiserver_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError('[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Server ports while Multi-Server mode is enabled. For more information refer VidGear docs.')
                else:
                    logger.debug('Enabling Multi-Server Mode at PORTS: {}!'.format(port))
                self.__port_buffer = []
            elif self.__multiclient_mode:
                if port is None:
                    raise ValueError('[NetGear:ERROR] :: Kindly provide a unique & valid port value at Client-end. For more information refer VidGear docs.')
                else:
                    logger.debug('Enabling Multi-Client Mode at PORT: {} on this device!'.format(port))
                self.__port = port
            elif port is None:
                port = '5555'
            try:
                if self.__secure_mode > 0:
                    z_auth = ThreadAuthenticator(self.__msg_context)
                    z_auth.start()
                    z_auth.allow(str(address))
                    if self.__secure_mode == 2:
                        z_auth.configure_curve(domain='*', location=self.__auth_publickeys_dir)
                    else:
                        z_auth.configure_curve(domain='*', location=auth.CURVE_ALLOW_ANY)
                self.__msg_socket = self.__msg_context.socket(msg_pattern[1])
                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)
                if self.__secure_mode > 0:
                    server_secret_file = os.path.join(self.__auth_secretkeys_dir, 'server.key_secret')
                    (server_public, server_secret) = auth.load_certificate(server_secret_file)
                    self.__msg_socket.curve_secretkey = server_secret
                    self.__msg_socket.curve_publickey = server_public
                    self.__msg_socket.curve_server = True
                if self.__pattern == 2:
                    self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE, '')
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(zmq.RCVTIMEO, self.__subscriber_timeout)
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(zmq.LINGER, 0)
                if self.__multiserver_mode:
                    for pt in port:
                        self.__msg_socket.bind(protocol + '://' + str(address) + ':' + str(pt))
                else:
                    self.__msg_socket.bind(protocol + '://' + str(address) + ':' + str(port))
                if pattern < 2:
                    if self.__multiserver_mode:
                        self.__connection_address = []
                        for pt in port:
                            self.__connection_address.append(protocol + '://' + str(address) + ':' + str(pt))
                    else:
                        self.__connection_address = protocol + '://' + str(address) + ':' + str(port)
                    self.__msg_pattern = msg_pattern[1]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    self.__logging and logger.debug('Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.'.format(self.__max_retries, self.__request_timeout / 1000))
                else:
                    self.__logging and self.__subscriber_timeout and logger.debug('Timeout: {} secs is enabled for this system.'.format(self.__subscriber_timeout / 1000))
            except Exception as e:
                logger.exception(str(e))
                if self.__secure_mode:
                    logger.critical('Failed to activate Secure Mode: `{}` for this connection!'.format(valid_security_mech[self.__secure_mode]))
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError('[NetGear:ERROR] :: Receive Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.'.format('Multi-Server' if self.__multiserver_mode else 'Multi-Client', protocol + '://' + str(address) + ':' + str(port), pattern))
                else:
                    if self.__bi_mode:
                        logger.critical('Failed to activate Bidirectional Mode for this connection!')
                    raise RuntimeError('[NetGear:ERROR] :: Receive Mode failed to bind address: {} and pattern: {}! Kindly recheck all parameters.'.format(protocol + '://' + str(address) + ':' + str(port), pattern))
            self.__logging and logger.debug('Threaded Queue Mode is enabled by default for this connection.')
            self.__queue = deque(maxlen=96)
            self.__thread = Thread(target=self.__recv_handler, name='NetGear', args=())
            self.__thread.daemon = True
            self.__thread.start()
            if self.__logging:
                logger.debug('Successfully Binded to address: {} with pattern: {}.'.format(protocol + '://' + str(address) + ':' + str(port), pattern))
                if self.__jpeg_compression:
                    logger.debug('JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.'.format(self.__jpeg_compression_colorspace, self.__jpeg_compression_quality, 'enabled' if self.__jpeg_compression_fastdct else 'disabled', 'enabled' if self.__jpeg_compression_fastupsample else 'disabled'))
                if self.__secure_mode:
                    logger.debug('Successfully enabled ZMQ Security Mechanism: `{}` for this connection.'.format(valid_security_mech[self.__secure_mode]))
                logger.debug('Multi-threaded Receive Mode is successfully enabled.')
                logger.debug('Unique System ID is {}.'.format(self.__id))
                logger.debug('Receive Mode is now activated.')
        else:
            if address is None:
                address = 'localhost'
            if self.__multiserver_mode:
                if port is None:
                    raise ValueError('[NetGear:ERROR] :: Kindly provide a unique & valid port value at Server-end. For more information refer VidGear docs.')
                else:
                    logger.debug('Enabling Multi-Server Mode at PORT: {} on this device!'.format(port))
                self.__port = port
            elif self.__multiclient_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError('[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Client ports while Multi-Client mode is enabled. For more information refer VidGear docs.')
                else:
                    logger.debug('Enabling Multi-Client Mode at PORTS: {}!'.format(port))
                self.__port_buffer = []
            elif port is None:
                port = '5555'
            try:
                if self.__secure_mode > 0:
                    z_auth = ThreadAuthenticator(self.__msg_context)
                    z_auth.start()
                    z_auth.allow(str(address))
                    if self.__secure_mode == 2:
                        z_auth.configure_curve(domain='*', location=self.__auth_publickeys_dir)
                    else:
                        z_auth.configure_curve(domain='*', location=auth.CURVE_ALLOW_ANY)
                self.__msg_socket = self.__msg_context.socket(msg_pattern[0])
                if self.__pattern == 1:
                    self.__msg_socket.REQ_RELAXED = True
                    self.__msg_socket.REQ_CORRELATE = True
                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)
                if self.__secure_mode > 0:
                    client_secret_file = os.path.join(self.__auth_secretkeys_dir, 'client.key_secret')
                    (client_public, client_secret) = auth.load_certificate(client_secret_file)
                    self.__msg_socket.curve_secretkey = client_secret
                    self.__msg_socket.curve_publickey = client_public
                    server_public_file = os.path.join(self.__auth_publickeys_dir, 'server.key')
                    (server_public, _) = auth.load_certificate(server_public_file)
                    self.__msg_socket.curve_serverkey = server_public
                if self.__multiclient_mode:
                    for pt in port:
                        self.__msg_socket.connect(protocol + '://' + str(address) + ':' + str(pt))
                elif self.__ssh_tunnel_mode:
                    ssh.tunnel_connection(self.__msg_socket, protocol + '://' + str(address) + ':' + str(port), self.__ssh_tunnel_mode, keyfile=self.__ssh_tunnel_keyfile, password=self.__ssh_tunnel_pwd, paramiko=self.__paramiko_present)
                else:
                    self.__msg_socket.connect(protocol + '://' + str(address) + ':' + str(port))
                if pattern < 2:
                    if self.__multiclient_mode:
                        self.__connection_address = []
                        for pt in port:
                            self.__connection_address.append(protocol + '://' + str(address) + ':' + str(pt))
                    else:
                        self.__connection_address = protocol + '://' + str(address) + ':' + str(port)
                    self.__msg_pattern = msg_pattern[0]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    self.__logging and logger.debug('Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.'.format(self.__max_retries, self.__request_timeout / 1000))
            except Exception as e:
                logger.exception(str(e))
                if self.__secure_mode:
                    logger.critical('Failed to activate Secure Mode: `{}` for this connection!'.format(valid_security_mech[self.__secure_mode]))
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError('[NetGear:ERROR] :: Send Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.'.format('Multi-Server' if self.__multiserver_mode else 'Multi-Client', protocol + '://' + str(address) + ':' + str(port), pattern))
                else:
                    if self.__bi_mode:
                        logger.critical('Failed to activate Bidirectional Mode for this connection!')
                    if self.__ssh_tunnel_mode:
                        logger.critical('Failed to initiate SSH Tunneling Mode for this server with `{}` back-end!'.format('paramiko' if self.__paramiko_present else 'pexpect'))
                    raise RuntimeError('[NetGear:ERROR] :: Send Mode failed to connect address: {} and pattern: {}! Kindly recheck all parameters.'.format(protocol + '://' + str(address) + ':' + str(port), pattern))
            if self.__logging:
                logger.debug('Successfully connected to address: {} with pattern: {}.'.format(protocol + '://' + str(address) + ':' + str(port), pattern))
                if self.__jpeg_compression:
                    logger.debug('JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.'.format(self.__jpeg_compression_colorspace, self.__jpeg_compression_quality, 'enabled' if self.__jpeg_compression_fastdct else 'disabled', 'enabled' if self.__jpeg_compression_fastupsample else 'disabled'))
                if self.__secure_mode:
                    logger.debug('Enabled ZMQ Security Mechanism: `{}` for this connection.'.format(valid_security_mech[self.__secure_mode]))
                logger.debug('Unique System ID is {}.'.format(self.__id))
                logger.debug('Send Mode is successfully activated and ready to send data.')

    def __recv_handler(self):
        if False:
            i = 10
            return i + 15
        '\n        A threaded receiver handler, that keep iterating data from ZMQ socket to a internally monitored deque,\n        until the thread is terminated, or socket disconnects.\n        '
        frame = None
        while not self.__terminate:
            if len(self.__queue) >= 96:
                time.sleep(1e-06)
                continue
            if self.__pattern < 2:
                socks = dict(self.__poll.poll(self.__request_timeout * 3))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag | zmq.DONTWAIT)
                else:
                    logger.critical('No response from Server(s), Reconnecting again...')
                    self.__msg_socket.close(linger=0)
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1
                    if not self.__max_retries:
                        if self.__multiserver_mode:
                            logger.error('All Servers seems to be offline, Abandoning!')
                        else:
                            logger.error('Server seems to be offline, Abandoning!')
                        self.__terminate = True
                        continue
                    try:
                        self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                        if isinstance(self.__connection_address, list):
                            for _connection in self.__connection_address:
                                self.__msg_socket.bind(_connection)
                        else:
                            self.__msg_socket.bind(self.__connection_address)
                    except Exception as e:
                        logger.exception(str(e))
                        self.__terminate = True
                        raise RuntimeError('API failed to restart the Client-end!')
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    continue
            else:
                try:
                    msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        logger.critical('Connection Timeout. Exiting!')
                        self.__terminate = True
                        self.__queue.append(None)
                        break
            if msg_json['terminate_flag']:
                if self.__multiserver_mode:
                    if msg_json['port'] in self.__port_buffer:
                        if self.__pattern == 1:
                            self.__msg_socket.send_string('Termination signal successfully received at client!')
                        self.__port_buffer.remove(msg_json['port'])
                        self.__logging and logger.warning('Termination signal received from Server at port: {}!'.format(msg_json['port']))
                    if not self.__port_buffer:
                        logger.critical('Termination signal received from all Servers!!!')
                        self.__terminate = True
                else:
                    if self.__pattern == 1:
                        self.__msg_socket.send_string("Termination signal successfully received at Client's end!")
                    self.__terminate = True
                    self.__logging and logger.critical('Termination signal received from server!')
                continue
            msg_data = self.__msg_socket.recv(flags=self.__msg_flag | zmq.DONTWAIT, copy=self.__msg_copy, track=self.__msg_track)
            if self.__pattern < 2:
                if self.__bi_mode or self.__multiclient_mode:
                    if not self.__return_data is None and isinstance(self.__return_data, np.ndarray):
                        return_data = np.copy(self.__return_data)
                        if not return_data.flags['C_CONTIGUOUS']:
                            return_data = np.ascontiguousarray(return_data, dtype=return_data.dtype)
                        if self.__jpeg_compression:
                            if self.__jpeg_compression_colorspace == 'GRAY':
                                if return_data.ndim == 2:
                                    return_data = return_data[:, :, np.newaxis]
                                return_data = simplejpeg.encode_jpeg(return_data, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, fastdct=self.__jpeg_compression_fastdct)
                            else:
                                return_data = simplejpeg.encode_jpeg(return_data, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, colorsubsampling='422', fastdct=self.__jpeg_compression_fastdct)
                        return_dict = dict(port=self.__port) if self.__multiclient_mode else dict()
                        return_dict.update(dict(return_type=type(self.__return_data).__name__, compression={'dct': self.__jpeg_compression_fastdct, 'ups': self.__jpeg_compression_fastupsample, 'colorspace': self.__jpeg_compression_colorspace} if self.__jpeg_compression else False, array_dtype=str(self.__return_data.dtype) if not self.__jpeg_compression else '', array_shape=self.__return_data.shape if not self.__jpeg_compression else '', data=None))
                        self.__msg_socket.send_json(return_dict, self.__msg_flag | zmq.SNDMORE)
                        self.__msg_socket.send(return_data, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track)
                    else:
                        return_dict = dict(port=self.__port) if self.__multiclient_mode else dict()
                        return_dict.update(dict(return_type=type(self.__return_data).__name__, data=self.__return_data))
                        self.__msg_socket.send_json(return_dict, self.__msg_flag)
                else:
                    self.__msg_socket.send_string('Data received on device: {} !'.format(self.__id))
            elif self.__return_data:
                logger.warning('`return_data` is disabled for this pattern!')
            if msg_json['compression']:
                frame = simplejpeg.decode_jpeg(msg_data, colorspace=msg_json['compression']['colorspace'], fastdct=self.__jpeg_compression_fastdct or msg_json['compression']['dct'], fastupsample=self.__jpeg_compression_fastupsample or msg_json['compression']['ups'])
                if frame is None:
                    self.__terminate = True
                    raise RuntimeError('[NetGear:ERROR] :: Received compressed JPEG frame decoding failed')
                if msg_json['compression']['colorspace'] == 'GRAY' and frame.ndim == 3:
                    frame = np.squeeze(frame, axis=2)
            else:
                frame_buffer = np.frombuffer(msg_data, dtype=msg_json['dtype'])
                frame = frame_buffer.reshape(msg_json['shape'])
            if self.__multiserver_mode:
                if not msg_json['port'] in self.__port_buffer:
                    self.__port_buffer.append(msg_json['port'])
                if msg_json['message']:
                    self.__queue.append((msg_json['port'], msg_json['message'], frame))
                else:
                    self.__queue.append((msg_json['port'], frame))
            elif self.__bi_mode:
                if msg_json['message']:
                    self.__queue.append((msg_json['message'], frame))
                else:
                    self.__queue.append((None, frame))
            else:
                self.__queue.append(frame)

    def recv(self, return_data=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        A Receiver end method, that extracts received frames synchronously from monitored deque, while maintaining a\n        fixed-length frame buffer in the memory, and blocks the thread if the deque is full.\n\n        Parameters:\n            return_data (any): inputs return data _(of any datatype)_, for sending back to Server.\n\n        **Returns:** A n-dimensional numpy array.\n        '
        if not self.__receive_mode:
            self.__terminate = True
            raise ValueError('[NetGear:ERROR] :: `recv()` function cannot be used while receive_mode is disabled. Kindly refer vidgear docs!')
        if (self.__bi_mode or self.__multiclient_mode) and (not return_data is None):
            self.__return_data = return_data
        while not self.__terminate:
            try:
                if len(self.__queue) > 0:
                    return self.__queue.popleft()
                else:
                    time.sleep(1e-05)
                    continue
            except KeyboardInterrupt:
                self.__terminate = True
                break
        return None

    def send(self, frame, message=None):
        if False:
            while True:
                i = 10
        '\n        A Server end method, that sends the data and frames over the network to Client(s).\n\n        Parameters:\n            frame (numpy.ndarray): inputs numpy array(frame).\n            message (any): input for sending additional data _(of any datatype except `numpy.ndarray`)_ to Client(s).\n\n        **Returns:** Data _(of any datatype)_ in selected exclusive modes, otherwise None-type.\n\n        '
        if self.__receive_mode:
            self.__terminate = True
            raise ValueError('[NetGear:ERROR] :: `send()` function cannot be used while receive_mode is enabled. Kindly refer vidgear docs!')
        if not message is None and isinstance(message, np.ndarray):
            logger.warning('Skipped unsupported `message` of datatype: {}!'.format(type(message).__name__))
            message = None
        exit_flag = True if frame is None or self.__terminate else False
        if not exit_flag and (not frame.flags['C_CONTIGUOUS']):
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)
        if self.__jpeg_compression:
            if self.__jpeg_compression_colorspace == 'GRAY':
                if frame.ndim == 2:
                    frame = np.expand_dims(frame, axis=2)
                frame = simplejpeg.encode_jpeg(frame, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, fastdct=self.__jpeg_compression_fastdct)
            else:
                frame = simplejpeg.encode_jpeg(frame, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, colorsubsampling='422', fastdct=self.__jpeg_compression_fastdct)
        msg_dict = dict(port=self.__port) if self.__multiserver_mode else dict()
        msg_dict.update(dict(terminate_flag=exit_flag, compression={'dct': self.__jpeg_compression_fastdct, 'ups': self.__jpeg_compression_fastupsample, 'colorspace': self.__jpeg_compression_colorspace} if self.__jpeg_compression else False, message=message, pattern=str(self.__pattern), dtype=str(frame.dtype) if not self.__jpeg_compression else '', shape=frame.shape if not self.__jpeg_compression else ''))
        self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
        self.__msg_socket.send(frame, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track)
        if self.__pattern < 2:
            if self.__bi_mode or self.__multiclient_mode:
                recvd_data = None
                socks = dict(self.__poll.poll(self.__request_timeout))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    recv_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                else:
                    logger.critical('No response from Client, Reconnecting again...')
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1
                    if not self.__max_retries:
                        if self.__multiclient_mode:
                            logger.error('All Clients failed to respond on multiple attempts.')
                        else:
                            logger.error('Client failed to respond on multiple attempts.')
                        self.__terminate = True
                        raise RuntimeError('[NetGear:ERROR] :: Client(s) seems to be offline, Abandoning.')
                    self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                    if isinstance(self.__connection_address, list):
                        for _connection in self.__connection_address:
                            self.__msg_socket.connect(_connection)
                    elif self.__ssh_tunnel_mode:
                        ssh.tunnel_connection(self.__msg_socket, self.__connection_address, self.__ssh_tunnel_mode, keyfile=self.__ssh_tunnel_keyfile, password=self.__ssh_tunnel_pwd, paramiko=self.__paramiko_present)
                    else:
                        self.__msg_socket.connect(self.__connection_address)
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    return None
                if self.__multiclient_mode and (not recv_json['port'] in self.__port_buffer):
                    self.__port_buffer.append(recv_json['port'])
                if recv_json['return_type'] == 'ndarray':
                    recv_array = self.__msg_socket.recv(flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track)
                    if recv_json['compression']:
                        recvd_data = simplejpeg.decode_jpeg(recv_array, colorspace=recv_json['compression']['colorspace'], fastdct=self.__jpeg_compression_fastdct or recv_json['compression']['dct'], fastupsample=self.__jpeg_compression_fastupsample or recv_json['compression']['ups'])
                        if recvd_data is None:
                            self.__terminate = True
                            raise RuntimeError('[NetGear:ERROR] :: Received compressed frame `{}` decoding failed with flag: {}.'.format(recv_json['compression'], self.__ex_compression_params))
                        if recv_json['compression']['colorspace'] == 'GRAY' and recvd_data.ndim == 3:
                            recvd_data = np.squeeze(recvd_data, axis=2)
                    else:
                        recvd_data = np.frombuffer(recv_array, dtype=recv_json['array_dtype']).reshape(recv_json['array_shape'])
                else:
                    recvd_data = recv_json['data']
                return (recv_json['port'], recvd_data) if self.__multiclient_mode else recvd_data
            else:
                socks = dict(self.__poll.poll(self.__request_timeout))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    recv_confirmation = self.__msg_socket.recv()
                else:
                    logger.critical('No response from Client, Reconnecting again...')
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1
                    if not self.__max_retries:
                        logger.error('Client failed to respond on repeated attempts.')
                        self.__terminate = True
                        raise RuntimeError('[NetGear:ERROR] :: Client seems to be offline, Abandoning!')
                    self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                    if self.__ssh_tunnel_mode:
                        ssh.tunnel_connection(self.__msg_socket, self.__connection_address, self.__ssh_tunnel_mode, keyfile=self.__ssh_tunnel_keyfile, password=self.__ssh_tunnel_pwd, paramiko=self.__paramiko_present)
                    else:
                        self.__msg_socket.connect(self.__connection_address)
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    return None
                self.__logging and logger.debug(recv_confirmation)

    def close(self, kill=False):
        if False:
            print('Hello World!')
        '\n        Safely terminates the threads, and NetGear resources.\n\n        Parameters:\n            kill (bool): Kills ZMQ context instead of graceful exiting in receive mode.\n        '
        self.__logging and logger.debug('Terminating various {} Processes.'.format('Receive Mode' if self.__receive_mode else 'Send Mode'))
        if self.__receive_mode:
            if not self.__queue is None and self.__queue:
                self.__queue.clear()
            self.__terminate = True
            self.__logging and logger.debug('Terminating. Please wait...')
            if self.__thread is not None:
                if self.__thread.is_alive() and kill:
                    logger.warning('Thread still running...Killing it forcefully!')
                    self.__msg_context.destroy()
                    self.__thread.join()
                else:
                    self.__thread.join()
                    self.__msg_socket.close(linger=0)
                self.__thread = None
            self.__logging and logger.debug('Terminated Successfully!')
        else:
            self.__terminate = True
            kill and logger.warning('`kill` parmeter is only available in the receive mode.')
            if self.__pattern < 2 and (not self.__max_retries) or (self.__multiclient_mode and (not self.__port_buffer)):
                try:
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                except ZMQError:
                    pass
                finally:
                    return
            if self.__multiserver_mode:
                term_dict = dict(terminate_flag=True, port=self.__port)
            else:
                term_dict = dict(terminate_flag=True)
            try:
                if self.__multiclient_mode:
                    for _ in self.__port_buffer:
                        self.__msg_socket.send_json(term_dict)
                else:
                    self.__msg_socket.send_json(term_dict)
                if self.__pattern < 2:
                    self.__logging and logger.debug('Terminating. Please wait...')
                    if self.__msg_socket.poll(self.__request_timeout // 5, zmq.POLLIN):
                        self.__msg_socket.recv()
            except Exception as e:
                if not isinstance(e, ZMQError):
                    logger.exception(str(e))
            finally:
                self.__msg_socket.setsockopt(zmq.LINGER, 0)
                self.__msg_socket.close()
                self.__logging and logger.debug('Terminated Successfully!')