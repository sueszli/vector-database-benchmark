import struct
import socket
import time
from scapy.contrib.automotive import log_automotive
from scapy.fields import ByteEnumField, ConditionalField, IntField, MayEnd, StrFixedLenField, XByteEnumField, XByteField, XIntField, XShortEnumField, XShortField, XStrField
from scapy.packet import Packet, bind_layers, bind_bottom_up
from scapy.supersocket import StreamSocket
from scapy.layers.inet import TCP, UDP
from scapy.contrib.automotive.uds import UDS
from scapy.data import MTU
from typing import Any, Union, Tuple, Optional

class DoIP(Packet):
    """
    Implementation of the DoIP (ISO 13400) protocol. DoIP packets can be sent
    via UDP and TCP. Depending on the payload type, the correct connection
    need to be chosen:

    +--------------+--------------------------------------------------------------+-----------------+
    | Payload Type | Payload Type Name                                            | Connection Kind |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0000       | Generic DoIP header negative acknowledge                     | UDP / TCP       |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0001       | Vehicle Identification request message                       | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0002       | Vehicle identification request message with EID              | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0003       | Vehicle identification request message with VIN              | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0004       | Vehicle announcement message/vehicle identification response | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0005       | Routing activation request                                   | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0006       | Routing activation response                                  | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0007       | Alive Check request                                          | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x0008       | Alive Check response                                         | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x4001       | IP entity status request                                     | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x4002       | DoIP entity status response                                  | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x4003       | Diagnostic power mode information request                    | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x4004       | Diagnostic power mode information response                   | UDP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x8001       | Diagnostic message                                           | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x8002       | Diagnostic message positive acknowledgement                  | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+
    | 0x8003       | Diagnostic message negative acknowledgement                  | TCP             |
    +--------------+--------------------------------------------------------------+-----------------+

    Example with UDP:
        >>> socket = L3RawSocket(iface="eth0")
        >>> resp = socket.sr1(IP(dst="169.254.117.238")/UDP(dport=13400)/DoIP(payload_type=1))

    Example with TCP:
        >>> socket = DoIPSocket("169.254.117.238")
        >>> pkt = DoIP(payload_type=0x8001, source_address=0xe80, target_address=0x1000) / UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)

    Example with UDS:
        >>> socket = UDS_DoIPSocket("169.254.117.238")
        >>> pkt = UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)
    """
    payload_types = {0: 'Generic DoIP header NACK', 1: 'Vehicle identification request', 2: 'Vehicle identification request with EID', 3: 'Vehicle identification request with VIN', 4: 'Vehicle announcement message/vehicle identification response message', 5: 'Routing activation request', 6: 'Routing activation response', 7: 'Alive check request', 8: 'Alive check response', 16385: 'DoIP entity status request', 16386: 'DoIP entity status response', 16387: 'Diagnostic power mode information request', 16388: 'Diagnostic power mode information response', 32769: 'Diagnostic message', 32770: 'Diagnostic message ACK', 32771: 'Diagnostic message NACK'}
    name = 'DoIP'
    fields_desc = [XByteField('protocol_version', 2), XByteField('inverse_version', 253), XShortEnumField('payload_type', 0, payload_types), IntField('payload_length', None), ConditionalField(ByteEnumField('nack', 0, {0: 'Incorrect pattern format', 1: 'Unknown payload type', 2: 'Message too large', 3: 'Out of memory', 4: 'Invalid payload length'}), lambda p: p.payload_type in [0]), ConditionalField(StrFixedLenField('vin', b'', 17), lambda p: p.payload_type in [3, 4]), ConditionalField(XShortField('logical_address', 0), lambda p: p.payload_type in [4]), ConditionalField(StrFixedLenField('eid', b'', 6), lambda p: p.payload_type in [2, 4]), ConditionalField(StrFixedLenField('gid', b'', 6), lambda p: p.payload_type in [4]), ConditionalField(MayEnd(XByteEnumField('further_action', 0, {0: 'No further action required', 1: 'Reserved by ISO 13400', 2: 'Reserved by ISO 13400', 3: 'Reserved by ISO 13400', 4: 'Reserved by ISO 13400', 5: 'Reserved by ISO 13400', 6: 'Reserved by ISO 13400', 7: 'Reserved by ISO 13400', 8: 'Reserved by ISO 13400', 9: 'Reserved by ISO 13400', 10: 'Reserved by ISO 13400', 11: 'Reserved by ISO 13400', 12: 'Reserved by ISO 13400', 13: 'Reserved by ISO 13400', 14: 'Reserved by ISO 13400', 15: 'Reserved by ISO 13400', 16: 'Routing activation required to initiate central security'})), lambda p: p.payload_type in [4]), ConditionalField(XByteEnumField('vin_gid_status', 0, {0: 'VIN and/or GID are synchronized', 1: 'Reserved by ISO 13400', 2: 'Reserved by ISO 13400', 3: 'Reserved by ISO 13400', 4: 'Reserved by ISO 13400', 5: 'Reserved by ISO 13400', 6: 'Reserved by ISO 13400', 7: 'Reserved by ISO 13400', 8: 'Reserved by ISO 13400', 9: 'Reserved by ISO 13400', 10: 'Reserved by ISO 13400', 11: 'Reserved by ISO 13400', 12: 'Reserved by ISO 13400', 13: 'Reserved by ISO 13400', 14: 'Reserved by ISO 13400', 15: 'Reserved by ISO 13400', 16: 'Incomplete: VIN and GID are NOT synchronized'}), lambda p: p.payload_type in [4]), ConditionalField(XShortField('source_address', 0), lambda p: p.payload_type in [5, 8, 32769, 32770, 32771]), ConditionalField(XByteEnumField('activation_type', 0, {0: 'Default', 1: 'WWH-OBD', 224: 'Central security', 22: 'Default', 278: 'Diagnostic', 57366: 'Central security'}), lambda p: p.payload_type in [5]), ConditionalField(XShortField('logical_address_tester', 0), lambda p: p.payload_type in [6]), ConditionalField(XShortField('logical_address_doip_entity', 0), lambda p: p.payload_type in [6]), ConditionalField(XByteEnumField('routing_activation_response', 0, {0: 'Routing activation denied due to unknown source address.', 1: 'Routing activation denied because all concurrently supported TCP_DATA sockets are registered and active.', 2: 'Routing activation denied because an SA different from the table connection entry was received on the already activated TCP_DATA socket.', 3: 'Routing activation denied because the SA is already registered and active on a different TCP_DATA socket.', 4: 'Routing activation denied due to missing authentication.', 5: 'Routing activation denied due to rejected confirmation.', 6: 'Routing activation denied due to unsupported routing activation type.', 7: 'Routing activation denied because the specified activation type requires a secure TLS TCP_DATA socket.', 8: 'Reserved by ISO 13400.', 9: 'Reserved by ISO 13400.', 10: 'Reserved by ISO 13400.', 11: 'Reserved by ISO 13400.', 12: 'Reserved by ISO 13400.', 13: 'Reserved by ISO 13400.', 14: 'Reserved by ISO 13400.', 15: 'Reserved by ISO 13400.', 16: 'Routing successfully activated.', 17: 'Routing will be activated; confirmation required.'}), lambda p: p.payload_type in [6]), ConditionalField(XIntField('reserved_iso', 0), lambda p: p.payload_type in [5, 6]), ConditionalField(XStrField('reserved_oem', b''), lambda p: p.payload_type in [5, 6]), ConditionalField(XByteEnumField('diagnostic_power_mode', 0, {0: 'not ready', 1: 'ready', 2: 'not supported'}), lambda p: p.payload_type in [16388]), ConditionalField(ByteEnumField('node_type', 0, {0: 'DoIP gateway', 1: 'DoIP node'}), lambda p: p.payload_type in [16386]), ConditionalField(XByteField('max_open_sockets', 1), lambda p: p.payload_type in [16386]), ConditionalField(XByteField('cur_open_sockets', 0), lambda p: p.payload_type in [16386]), ConditionalField(IntField('max_data_size', 0), lambda p: p.payload_type in [16386]), ConditionalField(XShortField('target_address', 0), lambda p: p.payload_type in [32769, 32770, 32771]), ConditionalField(XByteEnumField('ack_code', 0, {0: 'ACK'}), lambda p: p.payload_type in [32770]), ConditionalField(ByteEnumField('nack_code', 0, {0: 'Reserved by ISO 13400', 1: 'Reserved by ISO 13400', 2: 'Invalid source address', 3: 'Unknown target address', 4: 'Diagnostic message too large', 5: 'Out of memory', 6: 'Target unreachable', 7: 'Unknown network', 8: 'Transport protocol error'}), lambda p: p.payload_type in [32771]), ConditionalField(XStrField('previous_msg', b''), lambda p: p.payload_type in [32770, 32771])]

    def answers(self, other):
        if False:
            print('Hello World!')
        'DEV: true if self is an answer from other'
        if isinstance(other, type(self)):
            if self.payload_type == 0:
                return 1
            matches = [(4, 1), (4, 2), (4, 3), (6, 5), (8, 7), (16386, 16385), (16388, 16387), (32769, 32769), (32771, 32769)]
            if (self.payload_type, other.payload_type) in matches:
                if self.payload_type == 32769:
                    return self.payload.answers(other.payload)
                return 1
        return 0

    def hashret(self):
        if False:
            return 10
        if self.payload_type in [32769, 32770, 32771]:
            return bytes(self)[:2] + struct.pack('H', self.target_address ^ self.source_address)
        return bytes(self)[:2]

    def post_build(self, pkt, pay):
        if False:
            return 10
        "\n        This will set the Field 'payload_length' to the correct value.\n        "
        if self.payload_length is None:
            pkt = pkt[:4] + struct.pack('!I', len(pay) + len(pkt) - 8) + pkt[8:]
        return pkt + pay

    def extract_padding(self, s):
        if False:
            return 10
        if self.payload_type == 32769:
            return (s[:self.payload_length - 4], None)
        else:
            return (b'', None)

class DoIPSocket(StreamSocket):
    """ Custom StreamSocket for DoIP communication. This sockets automatically
    sends a routing activation request as soon as a TCP connection is
    established.

    :param ip: IP address of destination
    :param port: destination port, usually 13400
    :param activate_routing: If true, routing activation request is
                             automatically sent
    :param source_address: DoIP source address
    :param target_address: DoIP target address, this is automatically
                           determined if routing activation request is sent
    :param activation_type: This allows to set a different activation type for
                            the routing activation request
    :param reserved_oem: Optional parameter to set value for reserved_oem field
                         of routing activation request

    Example:
        >>> socket = DoIPSocket("169.254.0.131")
        >>> pkt = DoIP(payload_type=0x8001, source_address=0xe80, target_address=0x1000) / UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)
    """

    def __init__(self, ip='127.0.0.1', port=13400, activate_routing=True, source_address=3712, target_address=0, activation_type=0, reserved_oem=b''):
        if False:
            return 10
        self.ip = ip
        self.port = port
        self.source_address = source_address
        self.buffer = b''
        self._init_socket()
        if activate_routing:
            self._activate_routing(source_address, target_address, activation_type, reserved_oem)

    def recv(self, x=MTU, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.buffer:
            len_data = self.buffer[:8]
        else:
            len_data = self.ins.recv(8, socket.MSG_PEEK)
            if len(len_data) != 8:
                return None
        len_int = struct.unpack('>I', len_data[4:8])[0]
        len_int += 8
        self.buffer += self.ins.recv(len_int - len(self.buffer))
        if len(self.buffer) != len_int:
            return None
        pkt = self.basecls(self.buffer, **kwargs)
        self.buffer = b''
        return pkt

    def _init_socket(self, sock_family=socket.AF_INET):
        if False:
            while True:
                i = 10
        s = socket.socket(sock_family, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        addrinfo = socket.getaddrinfo(self.ip, self.port, proto=socket.IPPROTO_TCP)
        s.connect(addrinfo[0][-1])
        StreamSocket.__init__(self, s, DoIP)

    def _activate_routing(self, source_address, target_address, activation_type, reserved_oem=b''):
        if False:
            print('Hello World!')
        resp = self.sr1(DoIP(payload_type=5, activation_type=activation_type, source_address=source_address, reserved_oem=reserved_oem), verbose=False, timeout=1)
        if resp and resp.payload_type == 6 and (resp.routing_activation_response == 16):
            self.target_address = target_address or resp.logical_address_doip_entity
            log_automotive.info('Routing activation successful! Target address set to: 0x%x', self.target_address)
        else:
            log_automotive.error('Routing activation failed! Response: %s', repr(resp))

class DoIPSocket6(DoIPSocket):
    """ Custom StreamSocket for DoIP communication over IPv6.
    This sockets automatically sends a routing activation request as soon as
    a TCP connection is established.

    :param ip: IPv6 address of destination
    :param port: destination port, usually 13400
    :param activate_routing: If true, routing activation request is
                             automatically sent
    :param source_address: DoIP source address
    :param target_address: DoIP target address, this is automatically
                           determined if routing activation request is sent
    :param activation_type: This allows to set a different activation type for
                            the routing activation request
    :param reserved_oem: Optional parameter to set value for reserved_oem field
                         of routing activation request

    Example:
        >>> socket = DoIPSocket6("2001:16b8:3f0e:2f00:21a:37ff:febf:edb9")
        >>> socket_link_local = DoIPSocket6("fe80::30e8:80ff:fe07:6d43%eth1")
        >>> pkt = DoIP(payload_type=0x8001, source_address=0xe80, target_address=0x1000) / UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)
    """

    def __init__(self, ip='::1', port=13400, activate_routing=True, source_address=3712, target_address=0, activation_type=0, reserved_oem=b''):
        if False:
            i = 10
            return i + 15
        self.ip = ip
        self.port = port
        self.source_address = source_address
        self.buffer = b''
        super(DoIPSocket6, self)._init_socket(socket.AF_INET6)
        if activate_routing:
            super(DoIPSocket6, self)._activate_routing(source_address, target_address, activation_type, reserved_oem)

class UDS_DoIPSocket(DoIPSocket):
    """
    Application-Layer socket for DoIP endpoints. This socket takes care about
    the encapsulation of UDS packets into DoIP packets.

    Example:
        >>> socket = UDS_DoIPSocket("169.254.117.238")
        >>> pkt = UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)
    """

    def send(self, x):
        if False:
            print('Hello World!')
        if isinstance(x, UDS):
            pkt = DoIP(payload_type=32769, source_address=self.source_address, target_address=self.target_address) / x
        else:
            pkt = x
        try:
            x.sent_time = time.time()
        except AttributeError:
            pass
        return super(UDS_DoIPSocket, self).send(pkt)

    def recv(self, x=MTU, **kwargs):
        if False:
            return 10
        pkt = super(UDS_DoIPSocket, self).recv(x, **kwargs)
        if pkt and pkt.payload_type == 32769:
            return pkt.payload
        else:
            return pkt

class UDS_DoIPSocket6(DoIPSocket6, UDS_DoIPSocket):
    """
    Application-Layer socket for DoIP endpoints. This socket takes care about
    the encapsulation of UDS packets into DoIP packets.

    Example:
        >>> socket = UDS_DoIPSocket6("2001:16b8:3f0e:2f00:21a:37ff:febf:edb9")
        >>> pkt = UDS() / UDS_RDBI(identifiers=[0x1000])
        >>> resp = socket.sr1(pkt, timeout=1)
    """
    pass
bind_bottom_up(UDP, DoIP, sport=13400)
bind_bottom_up(UDP, DoIP, dport=13400)
bind_layers(UDP, DoIP, sport=13400, dport=13400)
bind_layers(TCP, DoIP, sport=13400)
bind_layers(TCP, DoIP, dport=13400)
bind_layers(DoIP, UDS, payload_type=32769)