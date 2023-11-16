import subprocess
import aj
import yaml
import os
from os.path import isfile, join
from jadi import component
from aj.plugins.network.api import NetworkManager
from .ip import *

@component(NetworkManager)
class UbuntuNetworkManager(NetworkManager):
    """
    Use of Netplan for Ubuntu version above 18.04
    """
    path = '/etc/netplan'

    @classmethod
    def __verify__(cls):
        if False:
            i = 10
            return i + 15
        '\n        Verify if this manager is relevant. Use netplan for Ubuntu > 18.\n\n        :return: bool\n        :rtype: bool\n        '
        check_prior_ubuntu = False
        if 'Ubuntu' in aj.platform_string:
            ubuntu_version = int(aj.platform_string[7:9])
            check_prior_ubuntu = ubuntu_version >= 18
        return aj.platform in ['debian'] and check_prior_ubuntu

    def __init__(self, context):
        if False:
            i = 10
            return i + 15
        NetworkManager.__init__(self, context)

    def get_config(self):
        if False:
            print('Hello World!')
        '\n        Parse the content of /etc/netplan.\n\n        :return: List of iface informations, one iface per dict\n        :rtype: list of dict\n        '
        ifaces = []
        netplan_files = [join(self.path, f) for f in os.listdir(self.path) if isfile(join(self.path, f))]
        for path in netplan_files:
            try:
                with open(path, 'r') as netplan_config:
                    config = yaml.load(netplan_config, Loader=yaml.SafeLoader) or {}
                    network_config = config.get('network', {})
                    ethernet_config = network_config.get('ethernets', {})
            except KeyError:
                continue
            for key in ethernet_config:
                addresses = ethernet_config[key].get('adresses', None)
                if addresses is None:
                    (ip, mask) = ifconfig_get_ip4_mask(key)
                    gateway = ifconfig_get_gateway(key)
                else:
                    (ip, mask) = ethernet_config[key]['addresses'][0].split('/')
                    gateway = ethernet_config[key].get('gateway4', None)
                iface = {'name': key, 'family': None, 'addressing': None, 'address': ip, 'mask': mask, 'gateway': gateway, 'hwaddress': None, 'mtu': None, 'scope': None, 'metric': None, 'client': None, 'pre_up_script': None, 'pre_down_script': None, 'up_script': None, 'down_script': None, 'post_up_script': None, 'post_down_script': None}
                ifaces.append(iface)
        return ifaces

    def set_config(self, config):
        if False:
            while True:
                i = 10
        '\n        Set the new config in the config file through augeas.\n\n        :param config: List of iface informations, one dict per iface\n        :type config: list of dict\n        '
        raise NotImplementedError

    def get_state(self, iface):
        if False:
            i = 10
            return i + 15
        '\n        Get ip and status for an iface.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        :return: Ip and status\n        :rtype: dict\n        '
        return {'address': ifconfig_get_ip(iface), 'up': ifconfig_get_up(iface)}

    def up(self, iface):
        if False:
            i = 10
            return i + 15
        '\n        Bring an iface up.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        ifconfig_up(iface)

    def down(self, iface):
        if False:
            for i in range(10):
                print('nop')
        '\n        Bring an iface down.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        ifconfig_down(iface)

    def get_hostname(self):
        if False:
            print('Hello World!')
        '\n        Get hostname value.\n\n        :return: Hostname\n        :rtype: string\n        '
        return subprocess.check_output('hostname', encoding='utf-8')

    def set_hostname(self, value):
        if False:
            while True:
                i = 10
        '\n        Write new hostname in /etc/hostname.\n\n        :param value: Hostname name\n        :type value: string\n        '
        with open('/etc/hostname', 'w') as f:
            f.write(value)
        subprocess.check_call(['hostname', value])