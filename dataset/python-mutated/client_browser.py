"""Manage the Glances client browser (list of Glances server)."""
import ujson
import socket
import threading
from glances.globals import Fault, ProtocolError, ServerProxy
from glances.client import GlancesClient, GlancesClientTransport
from glances.logger import logger, LOG_FILENAME
from glances.password_list import GlancesPasswordList as GlancesPassword
from glances.static_list import GlancesStaticServer
from glances.autodiscover import GlancesAutoDiscoverServer
from glances.outputs.glances_curses_browser import GlancesCursesBrowser

class GlancesClientBrowser(object):
    """This class creates and manages the TCP client browser (servers list)."""

    def __init__(self, config=None, args=None):
        if False:
            for i in range(10):
                print('nop')
        self.args = args
        self.config = config
        self.static_server = None
        self.password = None
        self.load()
        if not self.args.disable_autodiscover:
            self.autodiscover_server = GlancesAutoDiscoverServer()
        else:
            self.autodiscover_server = None
        self.screen = GlancesCursesBrowser(args=self.args)

    def load(self):
        if False:
            return 10
        'Load server and password list from the configuration file.'
        self.static_server = GlancesStaticServer(config=self.config)
        self.password = GlancesPassword(config=self.config)

    def get_servers_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current server list (list of dict).\n\n        Merge of static + autodiscover servers list.\n        '
        ret = []
        if self.args.browser:
            ret = self.static_server.get_servers_list()
            if self.autodiscover_server is not None:
                ret = self.static_server.get_servers_list() + self.autodiscover_server.get_servers_list()
        return ret

    def __get_uri(self, server):
        if False:
            print('Hello World!')
        'Return the URI for the given server dict.'
        if server['password'] != '':
            if server['status'] == 'PROTECTED':
                clear_password = self.password.get_password(server['name'])
                if clear_password is not None:
                    server['password'] = self.password.get_hash(clear_password)
            return 'http://{}:{}@{}:{}'.format(server['username'], server['password'], server['ip'], server['port'])
        else:
            return 'http://{}:{}'.format(server['ip'], server['port'])

    def __update_stats(self, server):
        if False:
            while True:
                i = 10
        'Update stats for the given server (picked from the server list)'
        uri = self.__get_uri(server)
        t = GlancesClientTransport()
        t.set_timeout(3)
        try:
            s = ServerProxy(uri, transport=t)
        except Exception as e:
            logger.warning("Client browser couldn't create socket ({})".format(e))
        else:
            try:
                cpu_percent = 100 - ujson.loads(s.getCpu())['idle']
                server['cpu_percent'] = '{:.1f}'.format(cpu_percent)
                server['mem_percent'] = ujson.loads(s.getMem())['percent']
                server['hr_name'] = ujson.loads(s.getSystem())['hr_name']
            except (socket.error, Fault, KeyError) as e:
                logger.debug('Error while grabbing stats form server ({})'.format(e))
                server['status'] = 'OFFLINE'
            except ProtocolError as e:
                if e.errcode == 401:
                    server['password'] = None
                    server['status'] = 'PROTECTED'
                else:
                    server['status'] = 'OFFLINE'
                logger.debug('Cannot grab stats from server ({} {})'.format(e.errcode, e.errmsg))
            else:
                server['status'] = 'ONLINE'
                try:
                    load_min5 = ujson.loads(s.getLoad())['min5']
                    server['load_min5'] = '{:.2f}'.format(load_min5)
                except Exception as e:
                    logger.warning('Error while grabbing stats form server ({})'.format(e))
        return server

    def __display_server(self, server):
        if False:
            while True:
                i = 10
        'Connect and display the given server'
        logger.debug('Selected server {}'.format(server))
        self.screen.display_popup('Connect to {}:{}'.format(server['name'], server['port']), duration=1)
        if server['password'] is None:
            clear_password = self.password.get_password(server['name'])
            if clear_password is None or self.get_servers_list()[self.screen.active_server]['status'] == 'PROTECTED':
                clear_password = self.screen.display_popup('Password needed for {}: '.format(server['name']), is_input=True)
            if clear_password is not None:
                self.set_in_selected('password', self.password.get_hash(clear_password))
        logger.info('Connect Glances client to the {} server'.format(server['key']))
        args_server = self.args
        args_server.client = server['ip']
        args_server.port = server['port']
        args_server.username = server['username']
        args_server.password = server['password']
        client = GlancesClient(config=self.config, args=args_server, return_to_browser=True)
        if not client.login():
            self.screen.display_popup("Sorry, cannot connect to '{}'\nSee '{}' for more details".format(server['name'], LOG_FILENAME))
            self.set_in_selected('status', 'OFFLINE')
        else:
            connection_type = client.serve_forever()
            try:
                logger.debug('Disconnect Glances client from the {} server'.format(server['key']))
            except IndexError:
                pass
            else:
                if connection_type == 'snmp':
                    self.set_in_selected('status', 'SNMP')
                else:
                    self.set_in_selected('status', 'ONLINE')
        self.screen.active_server = None

    def __serve_forever(self):
        if False:
            while True:
                i = 10
        'Main client loop.'
        thread_list = {}
        while not self.screen.is_end:
            logger.debug('Iter through the following server list: {}'.format(self.get_servers_list()))
            for v in self.get_servers_list():
                key = v['key']
                thread = thread_list.get(key, None)
                if thread is None or thread.is_alive() is False:
                    thread = threading.Thread(target=self.__update_stats, args=[v])
                    thread_list[key] = thread
                    thread.start()
            if self.screen.active_server is None:
                self.screen.update(self.get_servers_list())
            else:
                self.__display_server(self.get_servers_list()[self.screen.active_server])
        for thread in thread_list.values():
            thread.join()

    def serve_forever(self):
        if False:
            i = 10
            return i + 15
        'Wrapper to the serve_forever function.\n\n        This function will restore the terminal to a sane state\n        before re-raising the exception and generating a traceback.\n        '
        try:
            return self.__serve_forever()
        finally:
            self.end()

    def set_in_selected(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Set the (key, value) for the selected server in the list.'
        if self.screen.active_server >= len(self.static_server.get_servers_list()):
            self.autodiscover_server.set_server(self.screen.active_server - len(self.static_server.get_servers_list()), key, value)
        else:
            self.static_server.set_server(self.screen.active_server, key, value)

    def end(self):
        if False:
            while True:
                i = 10
        'End of the client browser session.'
        self.screen.end()