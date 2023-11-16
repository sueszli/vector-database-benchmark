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
        """
        Verify if this manager is relevant.

        :return: bool
        :rtype: bool
        """

        return aj.platform in ['centos']

    def __init__(self, context):
        NetworkManager.__init__(self, context)

    def get_augeas(self, iface):
        """
        Read the content of interfaces config file through augeas.

        :param iface: Network interface, e.g. eth0
        :type iface: string
        :return: Augeas object
        :rtype: augeas
        """

        aug = Augeas(modules=[{
            'name': 'Shellvars',
            'lens': 'Shellvars.lns',
            'incl': [
                os.path.join(self.path, 'ifcfg-' + iface),
            ]
        }])
        aug.load()
        return aug

    def get_config(self):
        """
        Parse the content of interface config file through augeas.

        :return: List of iface informations, one iface per dict
        :rtype: list of dict
        """

        ifaces = []
        for file in os.listdir(self.path):
            if file.startswith('ifcfg-'):
                name = file.split('-')[1]
                aug_path = os.path.join(self.aug_path, file)
                aug = self.get_augeas(name)
                iface = {
                    'name': name,
                    'family': 'inet6' if bool(aug.get(aug_path + '/IPV6INIT')) else 'inet',
                    'addressing': aug.get(aug_path + '/BOOTPROTO') or 'static',
                    'address': aug.get(aug_path + '/IPADDR'),
                    'mask': aug.get(aug_path + '/NETMASK'),
                    'gateway': aug.get(aug_path + '/GATEWAY') if bool(aug.get(aug_path + '/IPV6INIT')) else aug.get(aug_path + '/IPV6_DEFAULTGW'),
                    'hwaddress': aug.get(aug_path + '/HWADDR'),
                    'dhcpClient': aug.get(aug_path + '/DHCP_HOSTNAME'),
                }
                ifaces.append(iface)
        return ifaces

    def set_config(self, config):
        """
        Set the new config in the config file through augeas.

        :param config: List of iface informations, one dict per iface
        :type config: list of dict
        """

        for index, iface in enumerate(config):
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
        """
        Get ip and status for an iface.

        :param iface: Network interface, e.g. eth0
        :type iface: string
        :return: Ip and status
        :rtype: dict
        """

        return {
            'address': ifconfig_get_ip(iface),
            'up': ifconfig_get_up(iface),
        }

    def up(self, iface):
        """
        Bring an iface up.

        :param iface: Network interface, e.g. eth0
        :type iface: string
        """

        ifconfig_up(iface)

    def down(self, iface):
        """
        Bring an iface down.

        :param iface: Network interface, e.g. eth0
        :type iface: string
        """

        ifconfig_down(iface)

    def get_hostname(self):
        """
        Get hostname value.

        :return: Hostname
        :rtype: string
        """

        return subprocess.check_output('hostname', encoding='utf-8')

    def set_hostname(self, value):
        """
        Write new hostname in /etc/hostname.

        :param value: Hostname name
        :type value: string
        """

        with open('/etc/hostname', 'w') as f:
            f.write(value)
        subprocess.check_call(['hostname', value])
