import unittest
from impacket import IP6, ImpactDecoder, ICMP6

class TestICMP6(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.packet_list = self.generate_icmp6_constructed_packets()
        self.message_description_list = ['Echo Request', 'Echo Reply', 'Parameter problem - Erroneous header field', 'Parameter problem - Unrecognized Next Header', 'Parameter problem - Unrecognized IP6 Option', 'Destination unreachable - No route to destination', 'Destination unreachable - Administratively prohibited', 'Destination unreachable - Beyond scope of source address', 'Destination unreachable - Address unreachable ', 'Destination unreachable - Port unreachable', 'Destination unreachable - Src addr failed due to policy', 'Destination unreachable - Reject route', 'Time exceeded - Hop limit exceeded in transit', 'Time exceeded - Fragment reassembly time exceeded', 'Packet too big']
        self.reference_data_list = [[128, 0, 162, 166, 0, 1, 0, 2, 254, 86, 136], [129, 0, 161, 166, 0, 1, 0, 2, 254, 86, 136], [4, 0, 30, 168, 0, 0, 0, 2, 254, 86, 136], [4, 1, 30, 167, 0, 0, 0, 2, 254, 86, 136], [4, 2, 30, 166, 0, 0, 0, 2, 254, 86, 136], [1, 0, 33, 170, 0, 0, 0, 0, 254, 86, 136], [1, 1, 33, 169, 0, 0, 0, 0, 254, 86, 136], [1, 2, 33, 168, 0, 0, 0, 0, 254, 86, 136], [1, 3, 33, 167, 0, 0, 0, 0, 254, 86, 136], [1, 4, 33, 166, 0, 0, 0, 0, 254, 86, 136], [1, 5, 33, 165, 0, 0, 0, 0, 254, 86, 136], [1, 6, 33, 164, 0, 0, 0, 0, 254, 86, 136], [3, 0, 31, 170, 0, 0, 0, 0, 254, 86, 136], [3, 1, 31, 169, 0, 0, 0, 0, 254, 86, 136], [2, 0, 27, 150, 0, 0, 5, 20, 254, 86, 136]]

    def encapsulate_icmp6_packet_in_ip6_packet(self, icmp6_packet):
        if False:
            for i in range(10):
                print('nop')
        ip6_packet = IP6.IP6()
        ip6_packet.set_traffic_class(0)
        ip6_packet.set_flow_label(0)
        ip6_packet.set_hop_limit(1)
        ip6_packet.set_ip_src('FE80::78F8:89D1:30FF:256B')
        ip6_packet.set_ip_dst('FF02::1')
        ip6_packet.contains(icmp6_packet)
        ip6_packet.set_next_header(ip6_packet.child().get_ip_protocol_number())
        ip6_packet.set_payload_length(ip6_packet.child().get_size())
        icmp6_packet.calculate_checksum()
        return ip6_packet

    def compare_icmp6_packet_with_reference_buffer(self, icmp6_packet, reference_buffer, test_fail_message):
        if False:
            for i in range(10):
                print('nop')
        ip6_packet = self.encapsulate_icmp6_packet_in_ip6_packet(icmp6_packet)
        icmp6_header_buffer = ip6_packet.child().get_bytes().tolist()
        icmp6_payload_buffer = icmp6_packet.child().get_bytes().tolist()
        generated_buffer = icmp6_header_buffer + icmp6_payload_buffer
        self.assertEqual(generated_buffer, reference_buffer, test_fail_message)

    def generate_icmp6_constructed_packets(self):
        if False:
            for i in range(10):
                print('nop')
        packet_list = []
        arbitrary_data = [254, 86, 136]
        echo_id = 1
        echo_sequence_number = 2
        icmp6_packet = ICMP6.ICMP6.Echo_Request(echo_id, echo_sequence_number, arbitrary_data)
        packet_list.append(icmp6_packet)
        icmp6_packet = ICMP6.ICMP6.Echo_Reply(echo_id, echo_sequence_number, arbitrary_data)
        packet_list.append(icmp6_packet)
        originating_packet_data = arbitrary_data
        for code in range(0, 3):
            problem_pointer = 2
            icmp6_packet = ICMP6.ICMP6.Parameter_Problem(code, problem_pointer, originating_packet_data)
            packet_list.append(icmp6_packet)
        for code in range(0, 7):
            icmp6_packet = ICMP6.ICMP6.Destination_Unreachable(code, originating_packet_data)
            packet_list.append(icmp6_packet)
        for code in range(0, 2):
            icmp6_packet = ICMP6.ICMP6.Time_Exceeded(code, originating_packet_data)
            packet_list.append(icmp6_packet)
        icmp6_packet = ICMP6.ICMP6.Packet_Too_Big(1300, originating_packet_data)
        packet_list.append(icmp6_packet)
        return packet_list

    def test_message_construction(self):
        if False:
            return 10
        for (packet, reference, msg) in zip(self.packet_list, self.reference_data_list, self.message_description_list):
            self.compare_icmp6_packet_with_reference_buffer(packet, reference, 'ICMP6 creation of ' + msg + ' - Buffer mismatch')

    def test_message_decoding(self):
        if False:
            i = 10
            return i + 15
        d = ImpactDecoder.ICMP6Decoder()
        msg_types = [ICMP6.ICMP6.ECHO_REQUEST, ICMP6.ICMP6.ECHO_REPLY, ICMP6.ICMP6.PARAMETER_PROBLEM, ICMP6.ICMP6.PARAMETER_PROBLEM, ICMP6.ICMP6.PARAMETER_PROBLEM, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.DESTINATION_UNREACHABLE, ICMP6.ICMP6.TIME_EXCEEDED, ICMP6.ICMP6.TIME_EXCEEDED, ICMP6.ICMP6.PACKET_TOO_BIG]
        msg_codes = [0, 0, ICMP6.ICMP6.ERRONEOUS_HEADER_FIELD_ENCOUNTERED, ICMP6.ICMP6.UNRECOGNIZED_NEXT_HEADER_TYPE_ENCOUNTERED, ICMP6.ICMP6.UNRECOGNIZED_IPV6_OPTION_ENCOUNTERED, ICMP6.ICMP6.NO_ROUTE_TO_DESTINATION, ICMP6.ICMP6.ADMINISTRATIVELY_PROHIBITED, ICMP6.ICMP6.BEYOND_SCOPE_OF_SOURCE_ADDRESS, ICMP6.ICMP6.ADDRESS_UNREACHABLE, ICMP6.ICMP6.PORT_UNREACHABLE, ICMP6.ICMP6.SOURCE_ADDRESS_FAILED_INGRESS_EGRESS_POLICY, ICMP6.ICMP6.REJECT_ROUTE_TO_DESTINATION, ICMP6.ICMP6.HOP_LIMIT_EXCEEDED_IN_TRANSIT, ICMP6.ICMP6.FRAGMENT_REASSEMBLY_TIME_EXCEEDED, 0]
        for i in range(0, len(self.reference_data_list)):
            p = d.decode(self.reference_data_list[i])
            self.assertEqual(p.get_type(), msg_types[i], self.message_description_list[i] + ' - Msg type mismatch')
            self.assertEqual(p.get_code(), msg_codes[i], self.message_description_list[i] + ' - Msg code mismatch')
            if i in range(0, 2):
                self.assertEqual(p.get_echo_id(), 1, self.message_description_list[i] + ' - ID mismatch')
                self.assertEqual(p.get_echo_sequence_number(), 2, self.message_description_list[i] + ' - Sequence number mismatch')
                self.assertEqual(p.get_echo_arbitrary_data().tolist(), [254, 86, 136], self.message_description_list[i] + ' - Arbitrary data mismatch')
            if i in range(2, 5):
                self.assertEqual(p.get_parm_problem_pointer(), 2, self.message_description_list[i] + ' - Pointer mismatch')
            if i in range(5, 15):
                self.assertEqual(p.get_originating_packet_data().tolist(), [254, 86, 136], self.message_description_list[i] + ' - Originating packet data mismatch')
            if i in range(14, 15):
                self.assertEqual(p.get_mtu(), 1300, self.message_description_list[i] + ' - MTU mismatch')
if __name__ == '__main__':
    unittest.main(verbosity=1)