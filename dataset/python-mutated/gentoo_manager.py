import subprocess
from jadi import component
import aj
from aj.plugins.augeas.api import Augeas
from aj.plugins.network.api import NetworkManager
from .ifconfig import ifconfig_get_ip, ifconfig_get_up

@component(NetworkManager)
class GentooNetworkManager(NetworkManager):
    path = '/etc/conf.d/net'
    aug_path = '/files' + path

    @classmethod
    def __verify__(cls):
        if False:
            return 10
        '\n        Verify if this manager is relevant.\n\n        :return: bool\n        :rtype: bool\n        '
        return aj.platform in ['gentoo']

    def __init__(self, context):
        if False:
            for i in range(10):
                print('nop')
        NetworkManager.__init__(self, context)

    def get_augeas(self):
        if False:
            while True:
                i = 10
        '\n        Read the content of interfaces config files through augeas.\n\n        :return: Augeas object\n        :rtype: augeas\n        '
        aug = Augeas(modules=[{'name': 'Shellvars', 'lens': 'Shellvars.lns', 'incl': [self.path]}])
        aug.load()
        return aug

    def get_config(self):
        if False:
            i = 10
            return i + 15
        '\n        Parse the content of interfaces config files through augeas.\n\n        :return: List of iface informations, one iface per dict\n        :rtype: list of dict\n        '
        ifaces = []
        aug = self.get_augeas()
        for key in aug.match(f'{self.aug_path}/*'):
            if 'config_' not in key:
                continue
            iface_name = key.split('_')[-1]
            iface = {'name': iface_name, 'family': 'inet'}
            value = aug.get(key).strip('"')
            if value == 'dhcp':
                iface['addressing'] = 'dhcp'
            else:
                iface['addressing'] = 'static'
                tokens = value.split()
                iface['address'] = tokens.pop(0)
                while len(tokens):
                    key = tokens.pop(0)
                    value = tokens.pop(0)
                    if key == 'netmask':
                        iface['mask'] = value
            ifaces.append(iface)
            route_key = f'{self.aug_path}/routes_{iface_name}'
            if aug.match(route_key):
                routes = aug.get(route_key).strip('"').splitlines()
                for route in routes:
                    if route.strip().startswith('default via'):
                        iface['gateway'] = route.split()[-1]
        return ifaces

    def set_config(self, config):
        if False:
            return 10
        '\n        Set the new config in the config file through augeas.\n\n        :param config: List of iface informations, one dict per iface\n        :type config: list of dict\n        '
        aug = self.get_augeas()
        for iface in config:
            if iface['addressing'] == 'dhcp':
                value = 'dhcp'
            else:
                value = iface['address']
                if iface.get('mask', None):
                    value += f" netmask {iface['mask']}"
            aug.set(f"{self.aug_path}/config_{iface['name']}", f'"{value}"')
            route_key = f"{self.aug_path}/routes_{iface['name']}"
            if aug.match(route_key):
                routes = aug.get(route_key).strip('"').splitlines()
                routes = [route for route in routes if 'default via' not in route]
            else:
                routes = []
            if iface.get('gateway', None):
                routes.append(f"default via {iface['gateway']}")
            route_join = '\n'.join(routes)
            aug.set(route_key, f'"{route_join}"')
        aug.save()

    def get_state(self, iface):
        if False:
            while True:
                i = 10
        '\n        Get ip and status for an iface.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        :return: Ip and status\n        :rtype: dict\n        '
        return {'address': ifconfig_get_ip(iface), 'up': ifconfig_get_up(iface)}

    def up(self, iface):
        if False:
            for i in range(10):
                print('nop')
        '\n        Bring an iface up.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        subprocess.call([f'/etc/init.d/net.{iface}', 'restart'])
        subprocess.call(['rc-update', 'add', f'net.{iface}', 'default'])

    def down(self, iface):
        if False:
            for i in range(10):
                print('nop')
        '\n        Bring an iface down.\n\n        :param iface: Network interface, e.g. eth0\n        :type iface: string\n        '
        subprocess.call([f'/etc/init.d/net.{iface}', 'stop'])
        subprocess.call(['rc-update', 'delete', f'net.{iface}', 'default'])

    def get_hostname(self):
        if False:
            for i in range(10):
                print('nop')
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