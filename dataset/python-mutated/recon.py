"""Handles all reconnaissance operations."""
from logging import getLogger
from threading import Thread
from time import sleep, strftime
import scapy
import scapy.layers.dot11 as dot11
import wifiphisher.common.globals as universal
from wifiphisher.common.constants import LOCS_DIR, NON_CLIENT_ADDRESSES
from wifiphisher.common.interfaces import NetworkManager
LOGGER = getLogger(__name__)

class AccessPoint(object):
    """Represents an access point."""

    def __init__(self, ssid, bssid, channel, encryption, capture_file=False):
        if False:
            for i in range(10):
                print('nop')
        'Initialize class with all the given arguments.'
        self.name = ssid
        self.mac_address = bssid
        self.channel = channel
        self.encryption = encryption
        self.signal_strength = None
        self.client_count = 0
        self._clients = set()
        if capture_file:
            with open(capture_file, 'a') as _file:
                _file.write('{bssid} {ssid}\n'.format(bssid=bssid, ssid=ssid))

    def add_client(self, client):
        if False:
            i = 10
            return i + 15
        'Add client to access point.'
        if client not in self._clients:
            self._clients.add(client)
            self.client_count += 1

class AccessPointFinder(object):
    """Finds all the available access point."""

    def __init__(self, ap_interface, network_manager):
        if False:
            print('Hello World!')
        'Initialize class with all the given arguments.'
        self._interface = ap_interface
        self.observed_access_points = list()
        self._capture_file = False
        self._should_continue = True
        self._hidden_networks = list()
        self._sniff_packets_thread = Thread(target=self._sniff_packets)
        self._channel_hop_thread = Thread(target=self._channel_hop)
        self._network_manager = network_manager

    def _process_packets(self, packet):
        if False:
            while True:
                i = 10
        'Process a RadioTap packet to find access points.'
        if packet.haslayer(dot11.Dot11Beacon):
            if hasattr(packet.payload, 'info'):
                if not packet.info or b'\x00' in packet.info:
                    if packet.addr3 not in self._hidden_networks:
                        self._hidden_networks.append(packet.addr3)
                else:
                    self._create_ap_with_info(packet)
        elif packet.haslayer(dot11.Dot11ProbeResp):
            if packet.addr3 in self._hidden_networks:
                self._create_ap_with_info(packet)
        elif packet.haslayer(dot11.Dot11):
            self._find_clients(packet)

    def _create_ap_with_info(self, packet):
        if False:
            i = 10
            return i + 15
        'Create and add an access point using the extracted information.\n\n        Access points which are malformed or not in 2G channel list are\n        excluded.\n        '
        elt_section = packet[dot11.Dot11Elt]
        try:
            channel = str(ord(packet[dot11.Dot11Elt][2].info))
            if int(channel) not in universal.ALL_2G_CHANNELS:
                return
        except (TypeError, IndexError):
            return
        mac_address = packet.addr3
        name = None
        encryption_type = None
        non_decodable_name = '<contains non-printable chars>'
        rssi = get_rssi(packet.notdecoded)
        new_signal_strength = calculate_signal_strength(rssi)
        try:
            name = elt_section.info.decode('utf8')
        except UnicodeDecodeError:
            name = non_decodable_name
        for access_point in self.observed_access_points:
            if mac_address == access_point.mac_address:
                current_signal_strength = access_point.signal_strength
                signal_difference = new_signal_strength - current_signal_strength
                if signal_difference > 5:
                    access_point.signal_strength = new_signal_strength
                return None
        encryption_type = find_encryption_type(packet)
        access_point = AccessPoint(name, mac_address, channel, encryption_type, capture_file=self._capture_file)
        access_point.signal_strength = new_signal_strength
        self.observed_access_points.append(access_point)

    def _sniff_packets(self):
        if False:
            return 10
        'Sniff packets one at a time until otherwise set.'
        while self._should_continue:
            dot11.sniff(iface=self._interface, prn=self._process_packets, count=1, store=0)

    def capture_aps(self):
        if False:
            i = 10
            return i + 15
        'Create Lure10 capture file.'
        self._capture_file = '{LOCS_DIR}area_{time}'.format(LOCS_DIR=LOCS_DIR, time=strftime('%Y%m%d_%H%M%S'))
        LOGGER.info('Create lure10-capture file %s', self._capture_file)

    def find_all_access_points(self):
        if False:
            i = 10
            return i + 15
        'Find all the visible and hidden access points.'
        self._sniff_packets_thread.start()
        self._channel_hop_thread.start()

    def stop_finding_access_points(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop looking for access points.'
        self._should_continue = False
        wait_time = 10
        self._channel_hop_thread.join(wait_time)
        self._sniff_packets_thread.join(wait_time)

    def _channel_hop(self):
        if False:
            print('Hello World!')
        "Change the interface's channel every three seconds.\n\n        .. note: The channel range is between 1 to 13\n        "
        while self._should_continue:
            for channel in universal.ALL_2G_CHANNELS:
                if self._should_continue:
                    self._network_manager.set_interface_channel(self._interface, channel)
                    sleep(3)
                else:
                    break

    def _find_clients(self, packet):
        if False:
            return 10
        'Find and add if a client is discovered.'
        receiver = packet.addr1
        sender = packet.addr2
        if sender and receiver:
            receiver_identifier = receiver[:8]
            sender_identifier = sender[:8]
        else:
            return None
        if (receiver_identifier, sender_identifier) not in NON_CLIENT_ADDRESSES:
            for access_point in self.observed_access_points:
                access_point_mac = access_point.mac_address
                if access_point_mac == receiver:
                    access_point.add_client(sender)
                elif access_point_mac == sender:
                    access_point.add_client(receiver)

    def get_sorted_access_points(self):
        if False:
            i = 10
            return i + 15
        'Return all access points sorted based on signal strength.'
        return sorted(self.observed_access_points, key=lambda ap: ap.signal_strength, reverse=True)

def get_rssi(non_decoded_packet):
    if False:
        for i in range(10):
            print('nop')
    'Return the rssi value of the packet.'
    try:
        return -(256 - max(ord(non_decoded_packet[-4:-3]), ord(non_decoded_packet[-2:-1])))
    except TypeError:
        return -100

def calculate_signal_strength(rssi):
    if False:
        i = 10
        return i + 15
    'Calculate the signal strength of access point.'
    signal_strength = 0
    if rssi >= -50:
        signal_strength = 100
    else:
        signal_strength = 2 * (rssi + 100)
    return signal_strength

def find_encryption_type(packet):
    if False:
        i = 10
        return i + 15
    'Return the encryption type of the access point.\n\n    .. note: Possible return values are WPA2, WPA, WEP, OPEN,\n        WPA2/WPS and WPA/WPS\n    '
    encryption_info = packet.sprintf('%Dot11Beacon.cap%')
    elt_section = packet[dot11.Dot11Elt]
    encryption_type = None
    found_wps = False
    try:
        while isinstance(elt_section, dot11.Dot11Elt) or (not encryption_type and (not found_wps)):
            if elt_section.ID == 48:
                encryption_type = 'WPA2'
            elif elt_section.ID == 221 and elt_section.info.startswith(b'\x00P\xf2\x01\x01\x00'):
                encryption_type = 'WPA'
            if elt_section.ID == 221 and elt_section.info.startswith(b'\x00P\xf2\x04'):
                found_wps = True
            elt_section = elt_section.payload
            if not encryption_type:
                if 'privacy' in encryption_info:
                    encryption_type = 'WEP'
                else:
                    encryption_type = 'OPEN'
    except AttributeError:
        pass
    if encryption_type != 'WEP' and found_wps:
        encryption_type += '/WPS'
    return encryption_type