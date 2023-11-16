"""
Extension that interacts with roguehostapd to print relevant information. For example,
information regarding automatic association attacks.
"""
from collections import defaultdict
import wifiphisher.common.constants as constants

class Roguehostapdinfo(object):
    """
    Handles for printing KARMA attack information
    """

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup the class with all the given arguments.\n\n        :param self: A roguehostapdinfo object.\n        :param data: Shared data from main engine\n        :type self: roguehostapdinfo\n        :type data: dictionary\n        :return: None\n        :rtype: None\n        '
        self._data = data
        self._packets_to_send = defaultdict(list)
        self._mac2ssid_dict = defaultdict()
        self._known_beacon_ssids = self._get_known_beacon_ssids()

    def get_packet(self, packet):
        if False:
            i = 10
            return i + 15
        '\n        :param self: A roguehostapdinfo object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: roguehostapdinfo\n        :type packet: scapy.layers.RadioTap\n        :return: empty list\n        :rtype: list\n        '
        return self._packets_to_send

    def _get_known_beacon_ssids(self):
        if False:
            print('Hello World!')
        '\n        :param self: A roguehostapdinfo object\n        :type self: roguehostapdinfo\n        :return: None\n        :rtype: None\n        '
        known_beacons_ssids = set()
        if self._data.args.known_beacons:
            area_file = constants.KNOWN_WLANS_FILE
            with open(area_file) as _file:
                for line in _file:
                    if line.startswith('!'):
                        continue
                    essid = line.rstrip()
                    known_beacons_ssids.add(essid)
        return known_beacons_ssids

    def send_output(self):
        if False:
            i = 10
            return i + 15
        '\n        Send the output the extension manager\n        :param self: A roguehostapdinfo object.\n        :type self: roguehostapdinfo\n        :return: A list with the password checking information\n        :rtype: list\n        ..note: In each packet we ask roguehostapd whether there are victims\n        associated to rogue AP\n        '
        info = []
        ssid_mac_list = self._data.roguehostapd.get_karma_data()
        try:
            (mac_list, ssid_list) = list(zip(*ssid_mac_list))
        except ValueError:
            mac_list = []
            ssid_list = []
        pop_macs = []
        for mac in self._mac2ssid_dict:
            if mac not in mac_list:
                pop_macs.append(mac)
        for key in pop_macs:
            self._mac2ssid_dict.pop(key)
        for (idx, mac) in enumerate(mac_list):
            if mac not in self._mac2ssid_dict:
                self._mac2ssid_dict[mac] = ssid_list[idx]
        macssid_pairs = list(self._mac2ssid_dict.items())
        for (mac, ssid) in macssid_pairs:
            if ssid == self._data.target_ap_essid:
                outputstr = 'Victim ' + mac + " probed for WLAN with ESSID: '" + ssid + "' (Evil Twin)"
            elif ssid not in self._known_beacon_ssids:
                outputstr = 'Victim ' + mac + " probed for WLAN with ESSID: '" + ssid + "' (KARMA)"
            else:
                outputstr = 'Victim ' + mac + " probed for WLAN with ESSID: '" + ssid + "' (Known Beacons)"
            info.append(outputstr)
        return info

    def send_channels(self):
        if False:
            while True:
                i = 10
        "\n        Send channels to subscribe\n        :param self: A roguehostapdinfo object.\n        :type self: roguehostapdinfo\n        :return: empty list\n        :rtype: list\n        ..note: we don't need to send frames in this extension\n        "
        return [self._data.target_ap_channel]

    def on_exit(self):
        if False:
            return 10
        '\n        Free all the resources regarding to this module\n        :param self: A roguehostapdinfo object.\n        :type self: roguehostapdinfo\n        :return: None\n        :rtype: None\n        '
        pass