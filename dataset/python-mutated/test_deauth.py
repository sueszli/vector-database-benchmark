""" This module tests the deauth module in extensions """
import collections
import unittest
from collections import defaultdict
import mock
import scapy.layers.dot11 as dot11
import wifiphisher.common.constants as constants
import wifiphisher.extensions.deauth as deauth

class TestDeauth(unittest.TestCase):
    """ Tests Deauth class """

    def setUp(self):
        if False:
            while True:
                i = 10
        ' Set up the tests '
        essid = dot11.Dot11Elt(ID='SSID', info='')
        rates = dot11.Dot11Elt(ID='Rates', info='\x03\x12\x96\x18$0H`')
        dsset = dot11.Dot11Elt(ID='DSset', info='\x06')
        self.packet = dot11.RadioTap() / dot11.Dot11() / essid / rates / dsset
        custom_tuple = collections.namedtuple('test', 'target_ap_bssid target_ap_channel rogue_ap_mac args target_ap_essid is_freq_hop_allowed')
        self.target_channel = '6'
        self.target_bssid = 'BB:BB:BB:BB:BB:BB'
        self.rogue_mac = 'CC:CC:CC:CC:CC:CC'
        self.target_essid = 'Evil'
        self.args = mock.Mock()
        self.args.deauth_essid = False
        self.args.channel_monitor = False
        self.args.deauth_channels = []
        data0 = custom_tuple(self.target_bssid, self.target_channel, self.rogue_mac, self.args, self.target_essid, True)
        data1 = custom_tuple(None, self.target_channel, self.rogue_mac, self.args, self.target_essid, True)
        self.deauth_obj0 = deauth.Deauth(data0)
        self.deauth_obj1 = deauth.Deauth(data1)
        self.deauth_obj0._deauth_bssids = dict()
        self.deauth_obj1._deauth_bssids = dict()

    def test_craft_packet_normal_expected(self):
        if False:
            print('Hello World!')
        '\n        Test _craft_packet method when given all the normal arguments and\n        expecting normal results\n        '
        sender = '00:00:00:00:00:00'
        receiver = '11:11:11:11:11:11'
        bssid = '00:00:00:00:00:00'
        result = self.deauth_obj0._craft_packet(sender, receiver, bssid)
        message0 = 'Failed to craft a packet for disassociation'
        message1 = 'Failed to craft a packet for deauthentication'
        self.assertEqual(result[0].addr1, receiver, message0)
        self.assertEqual(result[0].addr2, sender, message0)
        self.assertEqual(result[0].addr3, bssid, message0)
        self.assertEqual(result[1].addr1, receiver, message1)
        self.assertEqual(result[1].addr2, sender, message1)
        self.assertEqual(result[1].addr3, bssid, message1)

    def test_get_packet_broadcast(self):
        if False:
            return 10
        '\n        Test get_packet method for crafting the broadcast frame\n        '
        sender = '00:00:00:00:00:00'
        receiver = '11:11:11:11:11:11'
        essid = dot11.Dot11Elt(ID='SSID', info='')
        rates = dot11.Dot11Elt(ID='Rates', info='\x03\x12\x96\x18$0H`')
        dsset = dot11.Dot11Elt(ID='DSset', info='\x06')
        packet = dot11.RadioTap() / dot11.Dot11() / dot11.Dot11Beacon() / essid / rates / dsset
        packet.addr1 = receiver
        packet.addr2 = sender
        packet.addr3 = self.target_bssid
        packet.FCfield = 0
        pkts_to_send = self.deauth_obj0.get_packet(packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(self.target_channel in pkts_to_send, True, message0)
        result = pkts_to_send[self.target_channel]
        self.assertEqual(result[0].subtype, 10, message1)
        self.assertEqual(result[0].addr1, constants.WIFI_BROADCAST, message1)
        self.assertEqual(result[0].addr2, self.target_bssid, message1)
        self.assertEqual(result[0].addr3, self.target_bssid, message1)
        self.assertEqual(result[1].subtype, 12, message1)
        self.assertEqual(result[1].addr1, constants.WIFI_BROADCAST, message1)
        self.assertEqual(result[1].addr2, self.target_bssid, message1)
        self.assertEqual(result[1].addr3, self.target_bssid, message1)

    def test_get_packet_second_run_non_releavent_client_empty(self):
        if False:
            while True:
                i = 10
        '\n        Test get_packet method for the second time when given a packet which\n        is not related to the target access point and --essid is not used.\n        The expected result are an channel list containing target channel and\n        an empty packet list\n        '
        sender0 = '00:00:00:00:00:00'
        receiver0 = '11:11:11:11:11:11'
        bssid0 = '22:22:22:22:22:22:22'
        sender1 = '33:33:33:33:33:33'
        receiver1 = '44:44:44:44:44:44'
        bssid1 = '55:55:55:55:55:55'
        self.packet.addr1 = receiver0
        self.packet.addr2 = sender0
        self.packet.addr3 = bssid0
        self.deauth_obj0.get_packet(self.packet)
        self.packet.addr1 = receiver1
        self.packet.addr2 = sender1
        self.packet.addr3 = bssid1
        result = self.deauth_obj0.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    def test_get_packet_second_run_our_ap_empty(self):
        if False:
            return 10
        '\n        Test get_packet method for the second time when given a packet which\n        is from our own rouge ap to the target access point and --essid is\n        not used. The expected result are an channel list containing target\n        channel and an empty packet list\n        '
        sender0 = '00:00:00:00:00:00'
        receiver0 = '11:11:11:11:11:11'
        bssid0 = '22:22:22:22:22:22:22'
        sender1 = '33:33:33:33:33:33'
        receiver1 = '44:44:44:44:44:44'
        bssid1 = self.rogue_mac
        self.packet.addr1 = receiver0
        self.packet.addr2 = sender0
        self.packet.addr3 = bssid0
        self.deauth_obj0.get_packet(self.packet)
        self.packet.addr1 = receiver1
        self.packet.addr2 = sender1
        self.packet.addr3 = bssid1
        result = self.deauth_obj0.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    def test_get_packet_multiple_clients_multiple_packets(self):
        if False:
            i = 10
            return i + 15
        '\n        Test get_packet method when run multiple times with valid cleints.\n        --essid is not used. The expected result are the channel of the\n        target AP followed by the broadcast packet for the target AP and\n        all the client packets\n        '
        sender0 = self.target_bssid
        receiver0 = '11:11:11:11:11:11'
        bssid0 = self.target_bssid
        sender1 = '33:33:33:33:33:33'
        receiver1 = self.target_bssid
        bssid1 = self.target_bssid
        self.packet.addr1 = receiver0
        self.packet.addr2 = sender0
        self.packet.addr3 = bssid0
        self.deauth_obj0._deauth_bssids[self.target_bssid] = self.target_channel
        pkts_to_send0 = self.deauth_obj0.get_packet(self.packet)
        result0 = pkts_to_send0[self.target_channel]
        self.packet.addr1 = receiver1
        self.packet.addr2 = sender1
        self.packet.addr3 = bssid1
        pkts_to_send1 = self.deauth_obj0.get_packet(self.packet)
        result1 = pkts_to_send1[self.target_channel]
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(self.target_channel in pkts_to_send0, True, message0)
        self.assertEqual(result0[0].subtype, 10, message1)
        self.assertEqual(result0[0].addr1, self.target_bssid, message1)
        self.assertEqual(result0[0].addr2, receiver0, message1)
        self.assertEqual(result0[0].addr3, self.target_bssid, message1)
        self.assertEqual(result0[1].subtype, 12, message1)
        self.assertEqual(result0[1].addr1, self.target_bssid, message1)
        self.assertEqual(result0[1].addr2, receiver0, message1)
        self.assertEqual(result0[1].addr3, self.target_bssid, message1)
        self.assertEqual(result0[2].subtype, 10, message1)
        self.assertEqual(result0[2].addr1, receiver0, message1)
        self.assertEqual(result0[2].addr2, self.target_bssid, message1)
        self.assertEqual(result0[2].addr3, self.target_bssid, message1)
        self.assertEqual(result0[3].subtype, 12, message1)
        self.assertEqual(result0[3].addr1, receiver0, message1)
        self.assertEqual(result0[3].addr2, self.target_bssid, message1)
        self.assertEqual(result0[3].addr3, self.target_bssid, message1)
        self.assertEqual(result1[4].subtype, 10, message1)
        self.assertEqual(result1[4].addr1, sender1, message1)
        self.assertEqual(result1[4].addr2, self.target_bssid, message1)
        self.assertEqual(result1[4].addr3, self.target_bssid, message1)
        self.assertEqual(result1[5].subtype, 12, message1)
        self.assertEqual(result1[5].addr1, sender1, message1)
        self.assertEqual(result1[5].addr2, self.target_bssid, message1)
        self.assertEqual(result1[5].addr3, self.target_bssid, message1)
        self.assertEqual(result1[6].subtype, 10, message1)
        self.assertEqual(result1[6].addr1, self.target_bssid, message1)
        self.assertEqual(result1[6].addr2, sender1, message1)
        self.assertEqual(result1[6].addr3, self.target_bssid, message1)
        self.assertEqual(result1[7].subtype, 12, message1)
        self.assertEqual(result1[7].addr1, self.target_bssid, message1)
        self.assertEqual(result1[7].addr2, sender1, message1)
        self.assertEqual(result1[7].addr3, self.target_bssid, message1)

    def test_get_packet_essid_flag_client_client_packet(self):
        if False:
            print('Hello World!')
        '\n        Test get_packet method when --essid flag is given. A new\n        client is given as input and the proper packets and the\n        clients channel is expected\n        '
        sender = '22:22:22:22:22:22'
        receiver = '11:11:11:11:11:11'
        bssid = receiver
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        self.deauth_obj1._deauth_bssids[bssid] = self.target_channel
        pkts_to_send = self.deauth_obj1.get_packet(self.packet)
        result = pkts_to_send[self.target_channel]
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(self.target_channel in pkts_to_send, True, message0)
        self.assertEqual(result[0].subtype, 10, message1)
        self.assertEqual(result[0].addr1, sender, message1)
        self.assertEqual(result[0].addr2, receiver, message1)
        self.assertEqual(result[0].addr3, bssid, message1)
        self.assertEqual(result[1].subtype, 12, message1)
        self.assertEqual(result[1].addr1, sender, message1)
        self.assertEqual(result[1].addr2, receiver, message1)
        self.assertEqual(result[1].addr3, bssid, message1)
        self.assertEqual(result[2].subtype, 10, message1)
        self.assertEqual(result[2].addr1, receiver, message1)
        self.assertEqual(result[2].addr2, sender, message1)
        self.assertEqual(result[2].addr3, bssid, message1)
        self.assertEqual(result[3].subtype, 12, message1)
        self.assertEqual(result[3].addr1, receiver, message1)
        self.assertEqual(result[3].addr2, sender, message1)
        self.assertEqual(result[3].addr3, bssid, message1)

    def test_get_packet_essid_flag_our_own_ap_empty_list(self):
        if False:
            i = 10
            return i + 15
        '\n        Test get_packet method when --essid flag is given. Our own\n        client is given as input. An empty list for both channel and\n        packets\n        '
        sender = '00:00:00:00:00:00'
        receiver = self.rogue_mac
        bssid = self.rogue_mac
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        result = self.deauth_obj1.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    @mock.patch('wifiphisher.extensions.deauth.ord')
    def test_get_packet_essid_flag_malformed0_channel_empty_list(self, mock_ord):
        if False:
            i = 10
            return i + 15
        '\n        Test get_packet method when --essid flag is given. This is the\n        case when a packet is malformed in the channel section. An empty\n        list for both channel and packets. This test the TypeError case\n        '
        mock_ord.side_effect = TypeError
        sender = '00:00:00:00:00:00'
        receiver = '11:11:11:11:11:11'
        bssid = '22:22:22:22:22:22:22'
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        result = self.deauth_obj1.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    @mock.patch('wifiphisher.extensions.deauth.ord')
    def test_get_packet_essid_flag_malformed1_channel_empty_list(self, mock_ord):
        if False:
            print('Hello World!')
        '\n        Test get_packet method when --essid flag is given. This is the\n        case when a packet is malformed in the channel section. An empty\n        list for both channel and packets. This tests the IndexError case\n        '
        mock_ord.side_effect = IndexError
        sender = '00:00:00:00:00:00'
        receiver = '11:11:11:11:11:11'
        bssid = '22:22:22:22:22:22:22'
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        result = self.deauth_obj1.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    @mock.patch('wifiphisher.extensions.deauth.ord')
    def test_get_packet_essid_flag_malformed2_channel_empty_list(self, mock_ord):
        if False:
            i = 10
            return i + 15
        '\n        Test get_packet method when --essid flag is given. This is the\n        case when a packet is malformed in the channel section. In this case\n        the channel reported is out of range and an empty list for both\n        channel and packets\n        '
        mock_ord.return_value = 200
        sender = '33:33:33:33:33:33'
        receiver = '11:11:11:11:11:11'
        bssid = '22:22:22:22:22:22:22'
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        result = self.deauth_obj1.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    def test_add_client_invalid_sender_none(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test _add_client when the given sender is in the non_client_address.\n        The expected output is None\n        '
        sender = constants.WIFI_INVALID
        receiver = '11:11:11:11:11:11'
        bssid = receiver
        result = self.deauth_obj0._add_clients(sender, receiver, bssid)
        self.assertIsNone(result)

    def test_add_client_invalid_receiver_none(self):
        if False:
            print('Hello World!')
        '\n        Test _add_client when the given receiver is in the non_client_address.\n        The expected output is None\n        '
        sender = '11:11:11:11:11:11'
        receiver = constants.WIFI_INVALID
        bssid = sender
        result = self.deauth_obj0._add_clients(sender, receiver, bssid)
        self.assertIsNone(result)

    def test_add_client_invalid_sender_receiver_none(self):
        if False:
            print('Hello World!')
        '\n        Test _add_client when the given sender and receiver are in the\n        non_client_address. The expected output is None\n        '
        sender = constants.WIFI_INVALID
        receiver = constants.WIFI_INVALID
        bssid = '22:22:22:22:22:22:22'
        result = self.deauth_obj0._add_clients(sender, receiver, bssid)
        self.assertIsNone(result)

    def test_add_client_irrelevent_sender_receiver_none(self):
        if False:
            i = 10
            return i + 15
        '\n        Test _add_client when neither sender nor receiver is the\n        BSSID. The expected output is None\n        '
        sender = '11:11:11:11:11:11'
        receiver = '33:33:33:33:33:33'
        bssid = '22:22:22:22:22:22:22'
        result = self.deauth_obj0._add_clients(sender, receiver, bssid)
        self.assertIsNone(result)

    def test_add_client_receiver_is_bssid_packets(self):
        if False:
            while True:
                i = 10
        '\n        Test _add_client when the given receiver is the bssid. The\n        expected output is proper packets for both sender and receiver\n        '
        sender = '22:22:22:22:22:22'
        receiver = '11:11:11:11:11:11'
        bssid = receiver
        result = self.deauth_obj1._add_clients(sender, receiver, bssid)
        message0 = 'Failed to return the correct client'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], sender, message0)
        self.assertEqual(result[1][0].subtype, 10, message1)
        self.assertEqual(result[1][0].addr1, sender, message1)
        self.assertEqual(result[1][0].addr2, receiver, message1)
        self.assertEqual(result[1][0].addr3, bssid, message1)
        self.assertEqual(result[1][1].subtype, 12, message1)
        self.assertEqual(result[1][1].addr1, sender, message1)
        self.assertEqual(result[1][1].addr2, receiver, message1)
        self.assertEqual(result[1][1].addr3, bssid, message1)
        self.assertEqual(result[1][2].subtype, 10, message1)
        self.assertEqual(result[1][2].addr1, receiver, message1)
        self.assertEqual(result[1][2].addr2, sender, message1)
        self.assertEqual(result[1][2].addr3, bssid, message1)
        self.assertEqual(result[1][3].subtype, 12, message1)
        self.assertEqual(result[1][3].addr1, receiver, message1)
        self.assertEqual(result[1][3].addr2, sender, message1)
        self.assertEqual(result[1][3].addr3, bssid, message1)

    def test_add_client_sender_is_bssid_packets(self):
        if False:
            while True:
                i = 10
        '\n        Test _add_client when the given sender is the bssid. The\n        expected output is proper packets for both sender and receiver\n        '
        sender = '22:22:22:22:22:22'
        receiver = '11:11:11:11:11:11'
        bssid = sender
        result = self.deauth_obj1._add_clients(sender, receiver, bssid)
        message0 = 'Failed to return the correct client'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], receiver, message0)
        self.assertEqual(result[1][0].subtype, 10, message1)
        self.assertEqual(result[1][0].addr1, sender, message1)
        self.assertEqual(result[1][0].addr2, receiver, message1)
        self.assertEqual(result[1][0].addr3, bssid, message1)
        self.assertEqual(result[1][1].subtype, 12, message1)
        self.assertEqual(result[1][1].addr1, sender, message1)
        self.assertEqual(result[1][1].addr2, receiver, message1)
        self.assertEqual(result[1][1].addr3, bssid, message1)
        self.assertEqual(result[1][2].subtype, 10, message1)
        self.assertEqual(result[1][2].addr1, receiver, message1)
        self.assertEqual(result[1][2].addr2, sender, message1)
        self.assertEqual(result[1][2].addr3, bssid, message1)
        self.assertEqual(result[1][3].subtype, 12, message1)
        self.assertEqual(result[1][3].addr1, receiver, message1)
        self.assertEqual(result[1][3].addr2, sender, message1)
        self.assertEqual(result[1][3].addr3, bssid, message1)

    def test_send_output_no_client_proper(self):
        if False:
            print('Hello World!')
        '\n        Test send_output method when no client has been detected.\n        The expected result is an empty message list\n        '
        message = 'Failed to send the proper output'
        self.assertEqual(self.deauth_obj1.send_output(), [], message)

    def test_send_output_single_client_proper(self):
        if False:
            return 10
        '\n        Test send_output method when a client has been already\n        detected. The expected result is the proper output\n        containing that client\n        '
        sender = '44:44:44:44:44:44'
        receiver = '55:55:55:55:55:55'
        bssid = receiver
        self.packet.addr1 = receiver
        self.packet.addr2 = sender
        self.packet.addr3 = bssid
        self.deauth_obj1._deauth_bssids[bssid] = self.target_channel
        self.deauth_obj1.get_packet(self.packet)
        actual = self.deauth_obj1.send_output()
        expected = 'DEAUTH/DISAS - {}'.format(sender)
        message = 'Failed to send the proper output'
        self.assertEqual(expected, actual[0], message)

    def test_send_output_multiple_client_proper(self):
        if False:
            print('Hello World!')
        '\n        Test send_output method when multiple client has been already\n        detected. The expected result is the proper output\n        containing that clients\n        '
        sender0 = '22:22:22:22:22:22'
        receiver0 = '11:11:11:11:11:11'
        bssid0 = receiver0
        sender1 = '33:33:33:33:33:33'
        receiver1 = '44:44:44:44:44:44'
        bssid1 = sender1
        self.packet.addr1 = receiver0
        self.packet.addr2 = sender0
        self.packet.addr3 = bssid0
        self.deauth_obj1._deauth_bssids[bssid0] = self.target_channel
        self.deauth_obj1.get_packet(self.packet)
        self.packet.addr1 = receiver1
        self.packet.addr2 = sender1
        self.packet.addr3 = bssid1
        self.deauth_obj1._deauth_bssids[bssid1] = self.target_channel
        self.deauth_obj1.get_packet(self.packet)
        actual = self.deauth_obj1.send_output()
        expected0 = 'DEAUTH/DISAS - {}'.format(sender0)
        expected1 = 'DEAUTH/DISAS - {}'.format(receiver1)
        self.assertIn(expected0, actual)
        self.assertIn(expected1, actual)

    def test_send_channels_non_frenzy_target_channel(self):
        if False:
            while True:
                i = 10
        "\n        Test send_channels method when --essid is not given. The\n        expected result is the target AP's channel\n        "
        actual = self.deauth_obj0.send_channels()
        message = "Failed to send target AP's channel"
        expected = [self.target_channel]
        self.assertEqual(expected, actual, message)

    def test_send_channels_frenzy_all_channels(self):
        if False:
            i = 10
            return i + 15
        '\n        Test send_channels method when --essid is given. The expected\n        result is all channels\n        '
        actual = self.deauth_obj1.send_channels()
        message = 'Failed to send all the channels'
        expected = [str(ch) for ch in range(1, 14)]
        self.assertEqual(expected, actual, message)

    def test_extract_bssid_to_ds_0_from_ds_1_addr2(self):
        if False:
            return 10
        '\n        Test _extract_bssid when to_ds is 1 and from_ds is 0.\n        The case should return packet.addr2\n        '
        self.packet.FCfield = 2
        self.packet.addr1 = '11:11:11:11:11:11'
        self.packet.addr2 = '22:22:22:22:22:22'
        self.packet.addr3 = '33:33:33:33:33:33'
        message = 'Fail to get correct BSSID as address 2'
        actual = self.deauth_obj0._extract_bssid(self.packet)
        expected = self.packet.addr2
        self.assertEqual(expected, actual, message)

    def test_extract_bssid_to_ds_1_from_ds_0_addr1(self):
        if False:
            print('Hello World!')
        '\n        Test _extract_bssid when to_ds is 1 and from_ds is 0.\n        The case should return packet.addr2\n        '
        self.packet.FCfield = 1
        self.packet.addr1 = '11:11:11:11:11:11'
        self.packet.addr2 = '22:22:22:22:22:22'
        self.packet.addr3 = '33:33:33:33:33:33'
        message = 'Fail to get correct BSSID as address 1'
        actual = self.deauth_obj0._extract_bssid(self.packet)
        expected = self.packet.addr1
        self.assertEqual(expected, actual, message)

    def test_extract_bssid_to_ds_0_from_ds_0_addr3(self):
        if False:
            i = 10
            return i + 15
        '\n        Test _extract_bssid when to_ds is 0 and from_ds is 0.\n        The case should return packet.addr3\n        '
        self.packet.FCfield = 0
        self.packet.addr1 = '11:11:11:11:11:11'
        self.packet.addr2 = '22:22:22:22:22:22'
        self.packet.addr3 = '33:33:33:33:33:33'
        message = 'Fail to get correct BSSID as address 3'
        actual = self.deauth_obj0._extract_bssid(self.packet)
        expected = self.packet.addr3
        self.assertEqual(expected, actual, message)

    def test_get_packet_to_ds_1_from_ds_1_empty(self):
        if False:
            while True:
                i = 10
        '\n        Drop the WDS frame in get_packet\n        '
        self.packet.FCfield = 3
        result = self.deauth_obj0.get_packet(self.packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    def test_get_packet_address_malform_empty(self):
        if False:
            return 10
        '\n        Drop the frame if the address is malformed\n        '
        packet = mock.Mock(spec=[])
        result = self.deauth_obj0.get_packet(packet)
        message0 = 'Failed to return an correct channel'
        message1 = 'Failed to return an correct packets'
        self.assertEqual(result[0], [], message0)
        self.assertEqual(result[1], [], message1)

    def test_is_target_target_ap_bssid_true(self):
        if False:
            while True:
                i = 10
        '\n        Get the target attacking bssid for the speficic ESSID\n        when --essid is not used\n        '
        essid = dot11.Dot11Elt(ID='SSID', info='Evil')
        packet = dot11.RadioTap() / dot11.Dot11() / dot11.Dot11Beacon() / essid
        packet.addr3 = '99:99:99:99:99:99'
        self.deauth_obj0._data.args.deauth_essid = 'Evil'
        result = self.deauth_obj0._is_target(packet)
        expected = True
        message = 'Fail to check the attacking essid: ' + self.target_essid
        self.assertEqual(result, expected, message)

    def test_is_target_essid_non_decodable_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assign essid to a constant when it is utf-8 non-decodable\n        '
        essid = dot11.Dot11Elt(ID='SSID', info='\x99\x873')
        packet = dot11.RadioTap() / dot11.Dot11() / dot11.Dot11Beacon() / essid
        packet.addr3 = '99:99:99:99:99:99'
        result = self.deauth_obj0._is_target(packet)
        expected = False
        message = 'Fail to raise the UnicodeDecodeError for non-printable essid'
        self.assertEqual(result, expected, message)

    def test_channel_deauth(self):
        if False:
            print('Hello World!')
        '\n        Test that we are deauthing on the right channels each time.\n        '
        result = self.deauth_obj0.send_channels()
        expected = [str(self.deauth_obj0._data.target_ap_channel)]
        message = 'Fail to receive right channels'
        self.assertEqual(result, expected, message)
        self.deauth_obj1._data.args.deauth_channels = [1, 2, 3, 4]
        result = self.deauth_obj1.send_channels()
        expected = ['1', '2', '3', '4']
        message = 'Fail to receive right channels'
        self.assertEqual(result, expected, message)