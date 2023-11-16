import logging
import time
import random
import socket
import sys
import gevent
import gevent.pool
from gevent.server import StreamServer
import util
from util import helper
from Config import config
from .FileRequest import FileRequest
from Peer import PeerPortchecker
from Site import SiteManager
from Connection import ConnectionServer
from Plugin import PluginManager
from Debug import Debug

@PluginManager.acceptPlugins
class FileServer(ConnectionServer):

    def __init__(self, ip=config.fileserver_ip, port=config.fileserver_port, ip_type=config.fileserver_ip_type):
        if False:
            for i in range(10):
                print('nop')
        self.site_manager = SiteManager.site_manager
        self.portchecker = PeerPortchecker.PeerPortchecker(self)
        self.log = logging.getLogger('FileServer')
        self.ip_type = ip_type
        self.ip_external_list = []
        self.supported_ip_types = ['ipv4']
        if helper.getIpType(ip) == 'ipv6' or self.isIpv6Supported():
            self.supported_ip_types.append('ipv6')
        if ip_type == 'ipv6' or (ip_type == 'dual' and 'ipv6' in self.supported_ip_types):
            ip = ip.replace('*', '::')
        else:
            ip = ip.replace('*', '0.0.0.0')
        if config.tor == 'always':
            port = config.tor_hs_port
            config.fileserver_port = port
        elif port == 0:
            (port_range_from, port_range_to) = list(map(int, config.fileserver_port_range.split('-')))
            port = self.getRandomPort(ip, port_range_from, port_range_to)
            config.fileserver_port = port
            if not port:
                raise Exception("Can't find bindable port")
            if not config.tor == 'always':
                config.saveValue('fileserver_port', port)
                config.arguments.fileserver_port = port
        ConnectionServer.__init__(self, ip, port, self.handleRequest)
        self.log.debug('Supported IP types: %s' % self.supported_ip_types)
        if ip_type == 'dual' and ip == '::':
            try:
                self.log.debug('Binding proxy to %s:%s' % ('::', self.port))
                self.stream_server_proxy = StreamServer(('0.0.0.0', self.port), self.handleIncomingConnection, spawn=self.pool, backlog=100)
            except Exception as err:
                self.log.info('StreamServer proxy create error: %s' % Debug.formatException(err))
        self.port_opened = {}
        self.sites = self.site_manager.sites
        self.last_request = time.time()
        self.files_parsing = {}
        self.ui_server = None

    def getRandomPort(self, ip, port_range_from, port_range_to):
        if False:
            while True:
                i = 10
        self.log.info('Getting random port in range %s-%s...' % (port_range_from, port_range_to))
        tried = []
        for bind_retry in range(100):
            port = random.randint(port_range_from, port_range_to)
            if port in tried:
                continue
            tried.append(port)
            sock = helper.createSocket(ip)
            try:
                sock.bind((ip, port))
                success = True
            except Exception as err:
                self.log.warning('Error binding to port %s: %s' % (port, err))
                success = False
            sock.close()
            if success:
                self.log.info('Found unused random port: %s' % port)
                return port
            else:
                time.sleep(0.1)
        return False

    def isIpv6Supported(self):
        if False:
            i = 10
            return i + 15
        if config.tor == 'always':
            return True
        ipv6_testip = 'fcec:ae97:8902:d810:6c92:ec67:efb2:3ec5'
        try:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            sock.connect((ipv6_testip, 80))
            local_ipv6 = sock.getsockname()[0]
            if local_ipv6 == '::1':
                self.log.debug('IPv6 not supported, no local IPv6 address')
                return False
            else:
                self.log.debug('IPv6 supported on IP %s' % local_ipv6)
                return True
        except socket.error as err:
            self.log.warning('IPv6 not supported: %s' % err)
            return False
        except Exception as err:
            self.log.error('IPv6 check error: %s' % err)
            return False

    def listenProxy(self):
        if False:
            return 10
        try:
            self.stream_server_proxy.serve_forever()
        except Exception as err:
            if err.errno == 98:
                self.log.debug('StreamServer proxy listen error: %s' % err)
            else:
                self.log.info('StreamServer proxy listen error: %s' % err)

    def handleRequest(self, connection, message):
        if False:
            print('Hello World!')
        if config.verbose:
            if 'params' in message:
                self.log.debug('FileRequest: %s %s %s %s' % (str(connection), message['cmd'], message['params'].get('site'), message['params'].get('inner_path')))
            else:
                self.log.debug('FileRequest: %s %s' % (str(connection), message['cmd']))
        req = FileRequest(self, connection)
        req.route(message['cmd'], message.get('req_id'), message.get('params'))
        if not self.has_internet and (not connection.is_private_ip):
            self.has_internet = True
            self.onInternetOnline()

    def onInternetOnline(self):
        if False:
            return 10
        self.log.info('Internet online')
        gevent.spawn(self.checkSites, check_files=False, force_port_check=True)

    def reload(self):
        if False:
            for i in range(10):
                print('nop')
        global FileRequest
        import imp
        FileRequest = imp.load_source('FileRequest', 'src/File/FileRequest.py').FileRequest

    def portCheck(self):
        if False:
            return 10
        if config.offline:
            self.log.info('Offline mode: port check disabled')
            res = {'ipv4': None, 'ipv6': None}
            self.port_opened = res
            return res
        if config.ip_external:
            for ip_external in config.ip_external:
                SiteManager.peer_blacklist.append((ip_external, self.port))
            ip_external_types = set([helper.getIpType(ip) for ip in config.ip_external])
            res = {'ipv4': 'ipv4' in ip_external_types, 'ipv6': 'ipv6' in ip_external_types}
            self.ip_external_list = config.ip_external
            self.port_opened.update(res)
            self.log.info('Server port opened based on configuration ipv4: %s, ipv6: %s' % (res['ipv4'], res['ipv6']))
            return res
        self.port_opened = {}
        if self.ui_server:
            self.ui_server.updateWebsocket()
        if 'ipv6' in self.supported_ip_types:
            res_ipv6_thread = gevent.spawn(self.portchecker.portCheck, self.port, 'ipv6')
        else:
            res_ipv6_thread = None
        res_ipv4 = self.portchecker.portCheck(self.port, 'ipv4')
        if not res_ipv4['opened'] and config.tor != 'always':
            if self.portchecker.portOpen(self.port):
                res_ipv4 = self.portchecker.portCheck(self.port, 'ipv4')
        if res_ipv6_thread is None:
            res_ipv6 = {'ip': None, 'opened': None}
        else:
            res_ipv6 = res_ipv6_thread.get()
            if res_ipv6['opened'] and (not helper.getIpType(res_ipv6['ip']) == 'ipv6'):
                self.log.info('Invalid IPv6 address from port check: %s' % res_ipv6['ip'])
                res_ipv6['opened'] = False
        self.ip_external_list = []
        for res_ip in [res_ipv4, res_ipv6]:
            if res_ip['ip'] and res_ip['ip'] not in self.ip_external_list:
                self.ip_external_list.append(res_ip['ip'])
                SiteManager.peer_blacklist.append((res_ip['ip'], self.port))
        self.log.info('Server port opened ipv4: %s, ipv6: %s' % (res_ipv4['opened'], res_ipv6['opened']))
        res = {'ipv4': res_ipv4['opened'], 'ipv6': res_ipv6['opened']}
        interface_ips = helper.getInterfaceIps('ipv4')
        if 'ipv6' in self.supported_ip_types:
            interface_ips += helper.getInterfaceIps('ipv6')
        for ip in interface_ips:
            if not helper.isPrivateIp(ip) and ip not in self.ip_external_list:
                self.ip_external_list.append(ip)
                res[helper.getIpType(ip)] = True
                SiteManager.peer_blacklist.append((ip, self.port))
                self.log.debug('External ip found on interfaces: %s' % ip)
        self.port_opened.update(res)
        if self.ui_server:
            self.ui_server.updateWebsocket()
        return res

    def checkSite(self, site, check_files=False):
        if False:
            return 10
        if site.isServing():
            site.announce(mode='startup')
            site.update(check_files=check_files)
            site.sendMyHashfield()
            site.updateHashfield()

    @util.Noparallel()
    def checkSites(self, check_files=False, force_port_check=False):
        if False:
            for i in range(10):
                print('nop')
        self.log.debug('Checking sites...')
        s = time.time()
        sites_checking = False
        if not self.port_opened or force_port_check:
            if len(self.sites) <= 2:
                sites_checking = True
                for (address, site) in list(self.sites.items()):
                    gevent.spawn(self.checkSite, site, check_files)
            self.portCheck()
            if not self.port_opened['ipv4']:
                self.tor_manager.startOnions()
        if not sites_checking:
            check_pool = gevent.pool.Pool(5)
            for site in sorted(list(self.sites.values()), key=lambda site: site.settings.get('modified', 0), reverse=True):
                if not site.isServing():
                    continue
                check_thread = check_pool.spawn(self.checkSite, site, check_files)
                time.sleep(2)
                if site.settings.get('modified', 0) < time.time() - 60 * 60 * 24:
                    check_thread.join(timeout=5)
        self.log.debug('Checksites done in %.3fs' % (time.time() - s))

    def cleanupSites(self):
        if False:
            for i in range(10):
                print('nop')
        import gc
        startup = True
        time.sleep(5 * 60)
        peers_protected = set([])
        while 1:
            self.log.debug('Running site cleanup, connections: %s, internet: %s, protected peers: %s' % (len(self.connections), self.has_internet, len(peers_protected)))
            for (address, site) in list(self.sites.items()):
                if not site.isServing():
                    continue
                if not startup:
                    site.cleanupPeers(peers_protected)
                time.sleep(1)
            peers_protected = set([])
            for (address, site) in list(self.sites.items()):
                if not site.isServing():
                    continue
                if site.peers:
                    with gevent.Timeout(10, exception=False):
                        site.announcer.announcePex()
                if site.content_updated is False:
                    site.update()
                elif site.bad_files:
                    site.retryBadFiles()
                if time.time() - site.settings.get('modified', 0) < 60 * 60 * 24 * 7:
                    connected_num = site.needConnections(check_site_on_reconnect=True)
                    if connected_num < config.connected_limit:
                        peers_protected.update([peer.key for peer in site.getConnectedPeers()])
                time.sleep(1)
            site = None
            gc.collect()
            startup = False
            time.sleep(60 * 20)

    def announceSite(self, site):
        if False:
            print('Hello World!')
        site.announce(mode='update', pex=False)
        active_site = time.time() - site.settings.get('modified', 0) < 24 * 60 * 60
        if site.settings['own'] or active_site:
            site.needConnections(check_site_on_reconnect=True)
        site.sendMyHashfield(3)
        site.updateHashfield(3)

    def announceSites(self):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(5 * 60)
        while 1:
            config.loadTrackersFile()
            s = time.time()
            for (address, site) in list(self.sites.items()):
                if not site.isServing():
                    continue
                gevent.spawn(self.announceSite, site).join(timeout=10)
                time.sleep(1)
            taken = time.time() - s
            sleep = max(0, 60 * 20 / len(config.trackers) - taken)
            self.log.debug('Site announce tracker done in %.3fs, sleeping for %.3fs...' % (taken, sleep))
            time.sleep(sleep)

    def wakeupWatcher(self):
        if False:
            while True:
                i = 10
        last_time = time.time()
        last_my_ips = socket.gethostbyname_ex('')[2]
        while 1:
            time.sleep(30)
            is_time_changed = time.time() - max(self.last_request, last_time) > 60 * 3
            if is_time_changed:
                self.log.info('Wakeup detected: time warp from %0.f to %0.f (%0.f sleep seconds), acting like startup...' % (last_time, time.time(), time.time() - last_time))
            my_ips = socket.gethostbyname_ex('')[2]
            is_ip_changed = my_ips != last_my_ips
            if is_ip_changed:
                self.log.info('IP change detected from %s to %s' % (last_my_ips, my_ips))
            if is_time_changed or is_ip_changed:
                self.checkSites(check_files=False, force_port_check=True)
            last_time = time.time()
            last_my_ips = my_ips

    def start(self, check_sites=True):
        if False:
            return 10
        if self.stopping:
            return False
        ConnectionServer.start(self)
        try:
            self.stream_server.start()
        except Exception as err:
            self.log.error('Error listening on: %s:%s: %s' % (self.ip, self.port, err))
        self.sites = self.site_manager.list()
        if config.debug:
            from Debug import DebugReloader
            DebugReloader.watcher.addCallback(self.reload)
        if check_sites:
            gevent.spawn(self.checkSites)
        thread_announce_sites = gevent.spawn(self.announceSites)
        thread_cleanup_sites = gevent.spawn(self.cleanupSites)
        thread_wakeup_watcher = gevent.spawn(self.wakeupWatcher)
        ConnectionServer.listen(self)
        self.log.debug('Stopped.')

    def stop(self):
        if False:
            print('Hello World!')
        if self.running and self.portchecker.upnp_port_opened:
            self.log.debug('Closing port %d' % self.port)
            try:
                self.portchecker.portClose(self.port)
                self.log.info('Closed port via upnp.')
            except Exception as err:
                self.log.info('Failed at attempt to use upnp to close port: %s' % err)
        return ConnectionServer.stop(self)