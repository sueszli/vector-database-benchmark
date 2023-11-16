import subprocess
import aj
from jadi import component
from aj.plugins.augeas.api import Augeas
from aj.plugins.network.api import NetworkManager
if subprocess.call(['which', 'ifconfig']) == 0:
    from .ifconfig import ifconfig_up, ifconfig_down, ifconfig_get_ip, ifconfig_get_up
else:
    from .ip import ifconfig_up, ifconfig_down, ifconfig_get_ip, ifconfig_get_up

@component(NetworkManager)
class DebianNetworkManager(NetworkManager):
    path = '/etc/network/interfaces'
    aug_path = '/files' + path

    @classmethod
    def __verify__(cls):
        if False:
            while True:
                i = 10
        '\n        Verify if this manager is relevant. Use the same manager for Debian\n        and Ubuntu < 18.\n\n        :return: bool\n        :rtype: bool\n        '
        check_prior_ubuntu = True
        if 'Ubuntu' in aj.platform_string:
            ubuntu_version = int(aj.platform_string[7:9])
            check_prior_ubuntu = ubuntu_version < 18
        return aj.platform in ['debian'] and check_prior_ubuntu

    def __init__(self, context):
        if False:
            for i in range(10):
                print('nop')
        NetworkManager.__init__(self, context)

    def get_augeas(self):
        if False:
            print('Hello World!')
        '\n        Read the content of /etc/network/interfaces through augeas.\n\n        :return: Augeas object\n        :rtype: augeas\n        '
        aug = Augeas(modules=[{'name': 'Interfaces', 'lens': 'Interfaces.lns', 'incl': [self.path, self.path + '.d/*']}])
        aug.load()
        return aug

    def get_config(self):
        if False:
            i = 10
            return i + 15
        '\n        Parse the content of /etc/network/interface through augeas.\n\n        :return: List of iface informations, one iface per dict\n        :rtype: list of dict\n        '
        aug = self.get_augeas()
        ifaces = []
        for path in aug.match(self.aug_path + '/iface[*]'):
            iface = {'name': aug.get(path), 'family': aug.get(path + '/family'), 'addressing': aug.get(path + '/method'), 'address': aug.get(path + '/address'), 'mask': aug.get(path + '/netmask'), 'gateway': aug.get(path + '/gateway'), 'hwaddress': aug.get(path + '/hwaddress'), 'mtu': aug.get(path + '/mtu'), 'scope': aug.get(path + '/scope'), 'metric': aug.get(path + '/metric'), 'client': aug.get(path + '/client'), 'pre_up_script': aug.get(path + '/pre-up'), 'pre_down_script': aug.get(path + '/pre-down'), 'up_script': aug.get(path + '/up'), 'down_script': aug.get(path + '/down'), 'post_up_script': aug.get(path + '/post-up'), 'post_down_script': aug.get(path + '/post-down')}
            ifaces.append(iface)
        return ifaces

    def set_config(self, config):
        if False:
            while True:
                i = 10
        '\n        Set the new config in the config file through augeas.\n\n        :param config: List of iface informations, one dict per iface\n        :type config: list of dict\n        '
        aug = self.get_augeas()
        for (index, iface) in enumerate(config):
            path = f'{self.aug_path}/iface[{index + 1}]'
            aug.setd(path + '/family', iface['family'])
            aug.setd(path + '/method', iface['addressing'])
            aug.setd(path + '/address', iface['address'])
            aug.setd(path + '/netmask', iface['mask'])
            aug.setd(path + '/gateway', iface['gateway'])
            aug.setd(path + '/hwaddress', iface['hwaddress'])
            aug.setd(path + '/mtu', iface['mtu'])
            aug.setd(path + '/scope', iface['scope'])
            aug.setd(path + '/metric', iface['metric'])
            aug.setd(path + '/client', iface['client'])
            aug.setd(path + '/pre-up', iface['pre_up_script'])
            aug.setd(path + '/pre-down', iface['pre_down_script'])
            aug.setd(path + '/up', iface['up_script'])
            aug.setd(path + '/down', iface['down_script'])
            aug.setd(path + '/post-up', iface['post_up_script'])
            aug.setd(path + '/post-down', iface['post_down_script'])
        aug.save()

    def get_state(self, iface):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        '\n        Write new hostname in /etc/hostname.\n\n        :param value: Hostname name\n        :type value: string\n        '
        with open('/etc/hostname', 'w') as f:
            f.write(value)
        subprocess.check_call(['hostname', value])