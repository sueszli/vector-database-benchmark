"""Manage autodiscover Glances server (thk to the ZeroConf protocol)."""
import socket
import sys
from glances.globals import BSD
from glances.logger import logger
try:
    from zeroconf import __version__ as __zeroconf_version, ServiceBrowser, ServiceInfo, Zeroconf
    zeroconf_tag = True
except ImportError:
    zeroconf_tag = False
if zeroconf_tag:
    zeroconf_min_version = (0, 17, 0)
    zeroconf_version = tuple([int(num) for num in __zeroconf_version.split('.')])
    logger.debug('Zeroconf version {} detected.'.format(__zeroconf_version))
    if zeroconf_version < zeroconf_min_version:
        logger.critical('Please install zeroconf 0.17 or higher.')
        sys.exit(1)
zeroconf_type = '_%s._tcp.local.' % 'glances'

class AutoDiscovered(object):
    """Class to manage the auto discovered servers dict."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._server_list = []

    def get_servers_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current server list (list of dict).'
        return self._server_list

    def set_server(self, server_pos, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Set the key to the value for the server_pos (position in the list).'
        self._server_list[server_pos][key] = value

    def add_server(self, name, ip, port):
        if False:
            return 10
        'Add a new server to the list.'
        new_server = {'key': name, 'name': name.split(':')[0], 'ip': ip, 'port': port, 'username': 'glances', 'password': '', 'status': 'UNKNOWN', 'type': 'DYNAMIC'}
        self._server_list.append(new_server)
        logger.debug('Updated servers list (%s servers): %s' % (len(self._server_list), self._server_list))

    def remove_server(self, name):
        if False:
            i = 10
            return i + 15
        'Remove a server from the dict.'
        for i in self._server_list:
            if i['key'] == name:
                try:
                    self._server_list.remove(i)
                    logger.debug('Remove server %s from the list' % name)
                    logger.debug('Updated servers list (%s servers): %s' % (len(self._server_list), self._server_list))
                except ValueError:
                    logger.error('Cannot remove server %s from the list' % name)

class GlancesAutoDiscoverListener(object):
    """Zeroconf listener for Glances server."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.servers = AutoDiscovered()

    def get_servers_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current server list (list of dict).'
        return self.servers.get_servers_list()

    def set_server(self, server_pos, key, value):
        if False:
            return 10
        'Set the key to the value for the server_pos (position in the list).'
        self.servers.set_server(server_pos, key, value)

    def add_service(self, zeroconf, srv_type, srv_name):
        if False:
            i = 10
            return i + 15
        'Method called when a new Zeroconf client is detected.\n\n        Note: the return code will never be used\n\n        :return: True if the zeroconf client is a Glances server\n        '
        if srv_type != zeroconf_type:
            return False
        logger.debug('Check new Zeroconf server: %s / %s' % (srv_type, srv_name))
        info = zeroconf.get_service_info(srv_type, srv_name)
        if info and (info.addresses or info.parsed_addresses):
            address = info.addresses[0] if info.addresses else info.parsed_addresses[0]
            new_server_ip = socket.inet_ntoa(address)
            new_server_port = info.port
            self.servers.add_server(srv_name, new_server_ip, new_server_port)
            logger.info('New Glances server detected (%s from %s:%s)' % (srv_name, new_server_ip, new_server_port))
        else:
            logger.warning('New Glances server detected, but failed to be get Zeroconf ServiceInfo ')
        return True

    def remove_service(self, zeroconf, srv_type, srv_name):
        if False:
            while True:
                i = 10
        'Remove the server from the list.'
        self.servers.remove_server(srv_name)
        logger.info('Glances server %s removed from the autodetect list' % srv_name)

class GlancesAutoDiscoverServer(object):
    """Implementation of the Zeroconf protocol (server side for the Glances client)."""

    def __init__(self, args=None):
        if False:
            print('Hello World!')
        if zeroconf_tag:
            logger.info('Init autodiscover mode (Zeroconf protocol)')
            try:
                self.zeroconf = Zeroconf()
            except socket.error as e:
                logger.error('Cannot start Zeroconf (%s)' % e)
                self.zeroconf_enable_tag = False
            else:
                self.listener = GlancesAutoDiscoverListener()
                self.browser = ServiceBrowser(self.zeroconf, zeroconf_type, self.listener)
                self.zeroconf_enable_tag = True
        else:
            logger.error('Cannot start autodiscover mode (Zeroconf lib is not installed)')
            self.zeroconf_enable_tag = False

    def get_servers_list(self):
        if False:
            while True:
                i = 10
        'Return the current server list (dict of dict).'
        if zeroconf_tag and self.zeroconf_enable_tag:
            return self.listener.get_servers_list()
        else:
            return []

    def set_server(self, server_pos, key, value):
        if False:
            i = 10
            return i + 15
        'Set the key to the value for the server_pos (position in the list).'
        if zeroconf_tag and self.zeroconf_enable_tag:
            self.listener.set_server(server_pos, key, value)

    def close(self):
        if False:
            i = 10
            return i + 15
        if zeroconf_tag and self.zeroconf_enable_tag:
            self.zeroconf.close()

class GlancesAutoDiscoverClient(object):
    """Implementation of the zeroconf protocol (client side for the Glances server)."""

    def __init__(self, hostname, args=None):
        if False:
            i = 10
            return i + 15
        if zeroconf_tag:
            zeroconf_bind_address = args.bind_address
            try:
                self.zeroconf = Zeroconf()
            except socket.error as e:
                logger.error('Cannot start zeroconf: {}'.format(e))
            if not BSD:
                try:
                    if zeroconf_bind_address == '0.0.0.0':
                        zeroconf_bind_address = self.find_active_ip_address()
                except KeyError:
                    pass
            zeroconf_bind_address = socket.gethostbyname(zeroconf_bind_address)
            address_family = socket.getaddrinfo(zeroconf_bind_address, args.port)[0][0]
            try:
                self.info = ServiceInfo(zeroconf_type, '{}:{}.{}'.format(hostname, args.port, zeroconf_type), address=socket.inet_pton(address_family, zeroconf_bind_address), port=args.port, weight=0, priority=0, properties={}, server=hostname)
            except TypeError:
                self.info = ServiceInfo(zeroconf_type, name='{}:{}.{}'.format(hostname, args.port, zeroconf_type), addresses=[socket.inet_pton(address_family, zeroconf_bind_address)], port=args.port, weight=0, priority=0, properties={}, server=hostname)
            try:
                self.zeroconf.register_service(self.info)
            except Exception as e:
                logger.error('Error while announcing Glances server: {}'.format(e))
            else:
                print('Announce the Glances server on the LAN (using {} IP address)'.format(zeroconf_bind_address))
        else:
            logger.error('Cannot announce Glances server on the network: zeroconf library not found.')

    @staticmethod
    def find_active_ip_address():
        if False:
            i = 10
            return i + 15
        'Try to find the active IP addresses.'
        import netifaces
        gateway_itf = netifaces.gateways()['default'][netifaces.AF_INET][1]
        return netifaces.ifaddresses(gateway_itf)[netifaces.AF_INET][0]['addr']

    def close(self):
        if False:
            return 10
        if zeroconf_tag:
            self.zeroconf.unregister_service(self.info)
            self.zeroconf.close()