"""
This module was made to match MAC address with vendors
"""
import wifiphisher.common.constants as constants

class MACMatcher(object):
    """
    This class handles Organizationally Unique Identifiers (OUIs).
    The original data comes from http://standards.ieee.org/regauth/
    oui/oui.tx

    .. seealso:: http://standards.ieee.org/faqs/OUI.html
    """

    def __init__(self, mac_vendor_file):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup the class with all the given arguments\n\n        :param self: A MACMatcher object\n        :param mac_vendor_file: The path of the vendor file\n        :type self: MACMatcher\n        :type mac_vendor_file: string\n        :return: None\n        :rtype: None\n        '
        self._mac_to_vendor = {}
        self._vendor_file = mac_vendor_file
        self._get_vendor_information()

    def _get_vendor_information(self):
        if False:
            i = 10
            return i + 15
        '\n        Read and process all the data in the vendor file\n\n        :param self: A MACMatcher object\n        :type self: MACMatcher\n        :return: None\n        :rtype: None\n        '
        with open(self._vendor_file, 'r') as _file:
            for line in _file:
                if not line.startswith('#'):
                    separated_line = line.rstrip('\n').split('|')
                    mac_identifier = separated_line[0]
                    vendor = separated_line[1]
                    logo = separated_line[2]
                    self._mac_to_vendor[mac_identifier] = (vendor, logo)

    def get_vendor_name(self, mac_address):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the matched vendor name for the given MAC address\n        or Unknown if no match is found\n\n        :param self: A MACMatcher object\n        :param mac_address: MAC address of device\n        :type self: MACMatcher\n        :type mac_address: string\n        :return: The vendor name of the device if MAC address is found\n                 and Unknown otherwise\n        :rtype: string\n        '
        if mac_address is None:
            return None
        mac_identifier = mac_address.replace(':', '').upper()[0:6]
        try:
            vendor = self._mac_to_vendor[mac_identifier][0]
            return vendor
        except KeyError:
            return 'Unknown'

    def get_vendor_logo_path(self, mac_address):
        if False:
            return 10
        '\n        Return the the full path of the logo in the filesystem for the\n        given MAC address or None if no match is found\n\n        :param self: A MACMatcher object\n        :param mac_address: MAC address of the device\n        :type self: MACMatcher\n        :type mac_address: string\n        :return: The full path of the logo if MAC address if found and\n                 None otherwise\n        :rtype: string or None\n        '
        if mac_address is None:
            return None
        mac_identifier = mac_address.replace(':', '').upper()[0:6]
        if mac_identifier in self._mac_to_vendor:
            logo = self._mac_to_vendor[mac_identifier][1]
            logo_path = constants.LOGOS_DIR + logo
            if logo:
                return logo_path
            else:
                return None

    def unbind(self):
        if False:
            print('Hello World!')
        '\n        Unloads mac to vendor mapping from memory and therefore you can\n        not use MACMatcher instance once this method is called\n\n        :param self: A MACMatcher object\n        :type self: MACMatcher\n        :return: None\n        :rtype: None\n        '
        del self._mac_to_vendor