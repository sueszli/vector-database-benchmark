"""Module to keep track the victims connected to the rogue AP."""
import time
import wifiphisher.common.constants as constants
from wifiphisher.common.macmatcher import MACMatcher as macmatcher

class Victim(object):
    """Resembles a Victim (i.e. a connected device to the rogue AP)."""

    def __init__(self, vmac_address, ip_address):
        if False:
            return 10
        'Create a Victim object.'
        self.vmac_address = vmac_address
        self.ip_address = ip_address
        self.os = ''
        self.vendor = ''
        self.timestamp = time.time()

    def associate_victim_mac_to_vendor(self, vmac_address):
        if False:
            for i in range(10):
                print('nop')
        'Find the victims vendor by its mac address.\n\n        Receives a victims mac address as input, finds the corresponding vendor\n        by using a macmacther object and then accesses the victim\n        dictionary and changes the vendor for the victim with the\n        corresponding mac address\n\n        :param self: empty Victim instance\n        :type self: Victim\n        :param vmac_address: mac address of the victim\n        :type vmac_address: str\n\n        '
        macmacther_instance = macmatcher(constants.MAC_PREFIX_FILE)
        vendor = macmacther_instance.get_vendor_name(vmac_address)
        victims_instance = Victims.get_instance()
        if vmac_address in victims_instance.victims_dic:
            victims_instance.victims_dic[vmac_address].vendor = vendor
        else:
            raise Exception('Error: No such mac address exists in dictionary')

    def assign_ip_to_victim(self, vmac_address, ip_address):
        if False:
            i = 10
            return i + 15
        'Update the ip address of the victim by mac address.'
        victims_instance = Victims.get_instance()
        if vmac_address in victims_instance.victims_dic:
            victims_instance.victims_dic[vmac_address].ip_address = ip_address
        else:
            raise Exception('Error: No such mac address exists in dictionary')

class Victims:
    """Singleton class that manages all of the victims."""
    __instance = None

    @staticmethod
    def get_instance():
        if False:
            return 10
        'Return the instance of the class or create new if none exists.'
        if Victims.__instance is None:
            Victims()
        return Victims.__instance

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        if Victims.__instance:
            raise Exception('Error: Victims class is a singleton!')
        else:
            Victims.__instance = self
            self.victims_dic = {}
            self.url_file = open(constants.URL_TO_OS_FILE, 'r')

    def add_to_victim_dic(self, victim_obj):
        if False:
            return 10
        'Add new victim to the dictionary.'
        self.victims_dic[victim_obj.vmac_address] = victim_obj

    def get_print_representation(self):
        if False:
            return 10
        'Return dic with five most recent victims in order to be printed.\n\n        :param self: Victims instance\n        :type self: Victims\n        :rtype str\n\n        '
        mac_timestamp = {}
        sorted_mac_timestamp = []
        most_recent_dic = {}
        max_victim_counter = 0
        for value in list(self.victims_dic.values()):
            mac_timestamp[value.vmac_address] = value.timestamp
        sorted_mac_timestamp = sorted(list(mac_timestamp.items()), key=lambda p: float(p[1]))
        for item in reversed(sorted_mac_timestamp):
            if max_victim_counter >= 5:
                return most_recent_dic
            victim_obj = self.victims_dic[item[0]]
            victim_value = '\t' + victim_obj.ip_address + '\t' + victim_obj.vendor + '\t' + victim_obj.os
            most_recent_dic[victim_obj.vmac_address] = victim_value
            max_victim_counter += 1
        return most_recent_dic

    def associate_victim_ip_to_os(self, ip_address, url):
        if False:
            while True:
                i = 10
        'Find and update Victims os based on the url it requests.\n\n        Receives a victims ip address and request as input, finds the\n        corresponding os by reading the initial requests file and then accesses\n        the victim dictionary and changes the os for the victim with the\n        corresponding ip address.\n\n        :param self: Victims instance\n        :type self: Victims\n        :param ip_address: ip address of the victim\n        :type ip_address: str\n        :param url: request of the victim\n        :type url: str\n\n        '
        self.url_file.seek(0)
        for line in self.url_file:
            line = line.split('|')
            url_check = line[1].strip()
            os = line[0].strip()
            if url_check in url:
                for key in self.victims_dic:
                    if ip_address == self.victims_dic[key].ip_address:
                        self.victims_dic[key].os = os