import struct
import array
from impacket.ImpactPacket import Header, array_frombytes
from impacket.IP6_Address import IP6_Address
from impacket.IP6_Extension_Headers import IP6_Extension_Header
from impacket import LOG

class IP6(Header):
    ethertype = 34525
    HEADER_SIZE = 40
    IP_PROTOCOL_VERSION = 6

    def __init__(self, buffer=None):
        if False:
            print('Hello World!')
        Header.__init__(self, IP6.HEADER_SIZE)
        self.set_ip_v(IP6.IP_PROTOCOL_VERSION)
        if buffer:
            self.load_header(buffer)

    def contains(self, aHeader):
        if False:
            for i in range(10):
                print('nop')
        Header.contains(self, aHeader)
        if isinstance(aHeader, IP6_Extension_Header):
            self.set_next_header(aHeader.get_header_type())

    def get_header_size(self):
        if False:
            i = 10
            return i + 15
        return IP6.HEADER_SIZE

    def __str__(self):
        if False:
            return 10
        protocol_version = self.get_ip_v()
        traffic_class = self.get_traffic_class()
        flow_label = self.get_flow_label()
        payload_length = self.get_payload_length()
        next_header = self.get_next_header()
        hop_limit = self.get_hop_limit()
        source_address = self.get_ip_src()
        destination_address = self.get_ip_dst()
        s = 'Protocol version: ' + str(protocol_version) + '\n'
        s += 'Traffic class: ' + str(traffic_class) + '\n'
        s += 'Flow label: ' + str(flow_label) + '\n'
        s += 'Payload length: ' + str(payload_length) + '\n'
        s += 'Next header: ' + str(next_header) + '\n'
        s += 'Hop limit: ' + str(hop_limit) + '\n'
        s += 'Source address: ' + source_address.as_string() + '\n'
        s += 'Destination address: ' + destination_address.as_string() + '\n'
        return s

    def get_pseudo_header(self):
        if False:
            return 10
        source_address = self.get_ip_src().as_bytes()
        destination_address = self.get_ip_dst().as_bytes()
        reserved_bytes = [0, 0, 0]
        upper_layer_packet_length = self.get_payload_length()
        upper_layer_protocol_number = self.get_next_header()
        next_header = self.child()
        while isinstance(next_header, IP6_Extension_Header):
            upper_layer_packet_length -= next_header.get_header_size()
            upper_layer_protocol_number = next_header.get_next_header()
            next_header = next_header.child()
        pseudo_header = array.array('B')
        pseudo_header.extend(source_address)
        pseudo_header.extend(destination_address)
        array_frombytes(pseudo_header, struct.pack('!L', upper_layer_packet_length))
        pseudo_header.fromlist(reserved_bytes)
        array_frombytes(pseudo_header, struct.pack('B', upper_layer_protocol_number))
        return pseudo_header

    def get_ip_v(self):
        if False:
            print('Hello World!')
        return (self.get_byte(0) & 240) >> 4

    def get_traffic_class(self):
        if False:
            print('Hello World!')
        return (self.get_byte(0) & 15) << 4 | (self.get_byte(1) & 240) >> 4

    def get_flow_label(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.get_byte(1) & 15) << 16 | self.get_byte(2) << 8 | self.get_byte(3)

    def get_payload_length(self):
        if False:
            return 10
        return self.get_byte(4) << 8 | self.get_byte(5)

    def get_next_header(self):
        if False:
            print('Hello World!')
        return self.get_byte(6)

    def get_hop_limit(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_byte(7)

    def get_ip_src(self):
        if False:
            return 10
        address = IP6_Address(self.get_bytes()[8:24])
        return address

    def get_ip_dst(self):
        if False:
            for i in range(10):
                print('nop')
        address = IP6_Address(self.get_bytes()[24:40])
        return address

    def set_ip_v(self, version):
        if False:
            for i in range(10):
                print('nop')
        if version != 6:
            raise Exception('set_ip_v - version != 6')
        b = self.get_byte(0) & 15
        b |= version << 4
        self.set_byte(0, b)

    def set_traffic_class(self, traffic_class):
        if False:
            while True:
                i = 10
        b0 = self.get_byte(0) & 240
        b1 = self.get_byte(1) & 15
        b0 |= (traffic_class & 240) >> 4
        b1 |= (traffic_class & 15) << 4
        self.set_byte(0, b0)
        self.set_byte(1, b1)

    def set_flow_label(self, flow_label):
        if False:
            while True:
                i = 10
        b1 = self.get_byte(1) & 240
        b1 |= (flow_label & 983040) >> 16
        self.set_byte(1, b1)
        self.set_byte(2, (flow_label & 65280) >> 8)
        self.set_byte(3, flow_label & 255)

    def set_payload_length(self, payload_length):
        if False:
            for i in range(10):
                print('nop')
        self.set_byte(4, (payload_length & 65280) >> 8)
        self.set_byte(5, payload_length & 255)

    def set_next_header(self, next_header):
        if False:
            return 10
        self.set_byte(6, next_header)

    def set_hop_limit(self, hop_limit):
        if False:
            for i in range(10):
                print('nop')
        self.set_byte(7, hop_limit)

    def set_ip_src(self, source_address):
        if False:
            return 10
        address = IP6_Address(source_address)
        bytes = self.get_bytes()
        bytes[8:24] = address.as_bytes()
        self.set_bytes(bytes)

    def set_ip_dst(self, destination_address):
        if False:
            while True:
                i = 10
        address = IP6_Address(destination_address)
        bytes = self.get_bytes()
        bytes[24:40] = address.as_bytes()
        self.set_bytes(bytes)

    def get_protocol_version(self):
        if False:
            print('Hello World!')
        LOG.warning('deprecated soon')
        return self.get_ip_v()

    def get_source_address(self):
        if False:
            i = 10
            return i + 15
        LOG.warning('deprecated soon')
        return self.get_ip_src()

    def get_destination_address(self):
        if False:
            return 10
        LOG.warning('deprecated soon')
        return self.get_ip_dst()

    def set_protocol_version(self, version):
        if False:
            for i in range(10):
                print('nop')
        LOG.warning('deprecated soon')
        self.set_ip_v(version)

    def set_source_address(self, source_address):
        if False:
            i = 10
            return i + 15
        LOG.warning('deprecated soon')
        self.set_ip_src(source_address)

    def set_destination_address(self, destination_address):
        if False:
            for i in range(10):
                print('nop')
        LOG.warning('deprecated soon')
        self.set_ip_dst(destination_address)