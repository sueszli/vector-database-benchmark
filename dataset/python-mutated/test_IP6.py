import unittest
from impacket import IP6, ImpactDecoder

class TestIP6(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.binary_packet = [100, 130, 70, 5, 5, 220, 17, 1, 254, 128, 0, 0, 0, 0, 0, 0, 120, 248, 137, 209, 48, 255, 37, 107, 255, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3]

    def test_decoding(self):
        if False:
            while True:
                i = 10
        'Test IP6 Packet decoding.'
        d = ImpactDecoder.IP6Decoder()
        parsed_packet = d.decode(self.binary_packet)
        protocol_version = parsed_packet.get_ip_v()
        traffic_class = parsed_packet.get_traffic_class()
        flow_label = parsed_packet.get_flow_label()
        payload_length = parsed_packet.get_payload_length()
        next_header = parsed_packet.get_next_header()
        hop_limit = parsed_packet.get_hop_limit()
        source_address = parsed_packet.get_ip_src()
        destination_address = parsed_packet.get_ip_dst()
        self.assertEqual(protocol_version, 6, 'IP6 parsing - Incorrect protocol version')
        self.assertEqual(traffic_class, 72, 'IP6 parsing - Incorrect traffic class')
        self.assertEqual(flow_label, 148997, 'IP6 parsing - Incorrect flow label')
        self.assertEqual(payload_length, 1500, 'IP6 parsing - Incorrect payload length')
        self.assertEqual(next_header, 17, 'IP6 parsing - Incorrect next header')
        self.assertEqual(hop_limit, 1, 'IP6 parsing - Incorrect hop limit')
        self.assertEqual(source_address.as_string(), 'FE80::78F8:89D1:30FF:256B', 'IP6 parsing - Incorrect source address')
        self.assertEqual(destination_address.as_string(), 'FF02::1:3', 'IP6 parsing - Incorrect destination address')

    def test_creation(self):
        if False:
            print('Hello World!')
        'Test IP6 Packet creation.'
        crafted_packet = IP6.IP6()
        crafted_packet.set_traffic_class(72)
        crafted_packet.set_flow_label(148997)
        crafted_packet.set_payload_length(1500)
        crafted_packet.set_next_header(17)
        crafted_packet.set_hop_limit(1)
        crafted_packet.set_ip_src('FE80::78F8:89D1:30FF:256B')
        crafted_packet.set_ip_dst('FF02::1:3')
        crafted_buffer = crafted_packet.get_bytes().tolist()
        self.assertEqual(crafted_buffer, self.binary_packet, 'IP6 creation - Buffer mismatch')
if __name__ == '__main__':
    unittest.main(verbosity=1)