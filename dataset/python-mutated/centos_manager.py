import os
import subprocess
from jadi import component
import aj
from aj.plugins.augeas.api import Augeas
from aj.plugins.network.api import NetworkManager
from .ifconfig import ifconfig_up, ifconfig_down, ifconfig_get_ip, ifconfig_get_up

@component(NetworkManager)
class CentOSNetworkManager(NetworkManager):
    path = '/etc/sysconfig/network-scripts'
    aug_path = '/files' + path

    @classmethod
    def __verify__(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify if this manager is relevant.\n\n        :return: bool\n        :rtype: bool\n        '
        return aj.platform in ['centos']

    def __init__(self, context):
        if False:
            i = 10
            return i + 15
        NetworkManager.__init__(self, context)

    def get_augeas(self, iface):
        if False:
            while True:
                i = 10
        '\n        Read the content of interfaces config file through augeas.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        :return: Augeas object\n        :rtype: augeas\n        '
        aug = Augeas(modules=[{'name': 'Shellvars', 'lens': 'Shellvars.lns', 'incl': [os.path.join(self.path, 'ifcfg-' + iface)]}])
        aug.load()
        return aug

    def get_config(self):
        if False:
            print('Hello World!')
        '\n        Parse the content of interface config file through augeas.\n\n        :return: List of iface informations, one iface per dict\n        :rtype: list of dict\n        '
        ifaces = []
        for file in os.listdir(self.path):
            if file.startswith('ifcfg-'):
                name = file.split('-')[1]
                aug_path = os.path.join(self.aug_path, file)
                aug = self.get_augeas(name)
                iface = {'name': name, 'family': 'inet6' if bool(aug.get(aug_path + '/IPV6INIT')) else 'inet', 'addressing': aug.get(aug_path + '/BOOTPROTO') or 'static', 'address': aug.get(aug_path + '/IPADDR'), 'mask': aug.get(aug_path + '/NETMASK'), 'gateway': aug.get(aug_path + '/GATEWAY') if bool(aug.get(aug_path + '/IPV6INIT')) else aug.get(aug_path + '/IPV6_DEFAULTGW'), 'hwaddress': aug.get(aug_path + '/HWADDR'), 'dhcpClient': aug.get(aug_path + '/DHCP_HOSTNAME')}
                ifaces.append(iface)
        return ifaces

    def set_config(self, config):
        if False:
            while True:
                i = 10
        '\n        Set the new config in the config file through augeas.\n\n        :param config: List of iface informations, one dict per iface\n        :type config: list of dict\n        '
        for (index, iface) in enumerate(config):
            aug = self.get_augeas(iface['name'])
            file = f'ifcfg-{iface}'
            aug_path = os.path.join(self.aug_path, file)
            if iface['family'] == 'inet':
                aug.remove(aug_path + '/IPV6INIT')
                aug.remove(aug_path + '/IPV6ADDR')
                aug.remove(aug_path + '/IPV6_DEFAULTGW')
                aug.setd(aug_path + '/IPADDR', iface['address'])
                aug.setd(aug_path + '/NETMASK', iface['mask'])
                aug.setd(aug_path + '/GATEWAY', iface['gateway'])
            else:
                aug.remove(aug_path + '/IPADDR')
                aug.remove(aug_path + '/NETMASK')
                aug.remove(aug_path + '/GATEWAY')
                aug.setd(aug_path + '/IPV6INIT', 'yes')
                aug.setd(aug_path + '/IPV6ADDR', iface['address'])
                aug.setd(aug_path + '/IPV6_DEFAULTGW', iface['gateway'])
            aug.setd(aug_path + '/BOOTPROTO', iface['method'])
            aug.setd(aug_path + '/HWADDR', iface['hwaddress'])
            aug.setd(aug_path + '/DHCP_HOSTNAME', iface['dhcpClient'])
            aug.save()

    def get_state(self, iface):
        if False:
            return 10
        '\n        Get ip and status for an iface.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        :return: Ip and status\n        :rtype: dict\n        '
        return {'address': ifconfig_get_ip(iface), 'up': ifconfig_get_up(iface)}

    def up(self, iface):
        if False:
            while True:
                i = 10
        '\n        Bring an iface up.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        ifconfig_up(iface)

    def down(self, iface):
        if False:
            return 10
        '\n        Bring an iface down.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        ifconfig_down(iface)

    def get_hostname(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get hostname value.\n\n        :return: Hostname\n        :rtype: string\n        '
        return subprocess.check_output('hostname', encoding='utf-8')

    def set_hostname(self, value):
        if False:
            print('Hello World!')
        '\n        Write new hostname in /etc/hostname.\n\n        :param value: Hostname name\n        :type value: string\n        '
        with open('/etc/hostname', 'w') as f:
            f.write(value)
        subprocess.check_call(['hostname', value])