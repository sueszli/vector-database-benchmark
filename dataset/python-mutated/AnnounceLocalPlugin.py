import time
import gevent
from Plugin import PluginManager
from Config import config
from . import BroadcastServer

@PluginManager.registerTo('SiteAnnouncer')
class SiteAnnouncerPlugin(object):

    def announce(self, force=False, *args, **kwargs):
        if False:
            print('Hello World!')
        local_announcer = self.site.connection_server.local_announcer
        thread = None
        if local_announcer and (force or time.time() - local_announcer.last_discover > 5 * 60):
            thread = gevent.spawn(local_announcer.discover, force=force)
        back = super(SiteAnnouncerPlugin, self).announce(*args, force=force, **kwargs)
        if thread:
            thread.join()
        return back

class LocalAnnouncer(BroadcastServer.BroadcastServer):

    def __init__(self, server, listen_port):
        if False:
            while True:
                i = 10
        super(LocalAnnouncer, self).__init__('zeronet', listen_port=listen_port)
        self.server = server
        self.sender_info['peer_id'] = self.server.peer_id
        self.sender_info['port'] = self.server.port
        self.sender_info['broadcast_port'] = listen_port
        self.sender_info['rev'] = config.rev
        self.known_peers = {}
        self.last_discover = 0

    def discover(self, force=False):
        if False:
            i = 10
            return i + 15
        self.log.debug('Sending discover request (force: %s)' % force)
        self.last_discover = time.time()
        if force:
            self.known_peers = {}
        for (peer_id, known_peer) in list(self.known_peers.items()):
            if time.time() - known_peer['found'] > 20 * 60:
                del self.known_peers[peer_id]
                self.log.debug('Timeout, removing from known_peers: %s' % peer_id)
        self.broadcast({'cmd': 'discoverRequest', 'params': {}}, port=self.listen_port)

    def actionDiscoverRequest(self, sender, params):
        if False:
            print('Hello World!')
        back = {'cmd': 'discoverResponse', 'params': {'sites_changed': self.server.site_manager.sites_changed}}
        if sender['peer_id'] not in self.known_peers:
            self.known_peers[sender['peer_id']] = {'added': time.time(), 'sites_changed': 0, 'updated': 0, 'found': time.time()}
            self.log.debug('Got discover request from unknown peer %s (%s), time to refresh known peers' % (sender['ip'], sender['peer_id']))
            gevent.spawn_later(1.0, self.discover)
        return back

    def actionDiscoverResponse(self, sender, params):
        if False:
            i = 10
            return i + 15
        if sender['peer_id'] in self.known_peers:
            self.known_peers[sender['peer_id']]['found'] = time.time()
        if params['sites_changed'] != self.known_peers.get(sender['peer_id'], {}).get('sites_changed'):
            return {'cmd': 'siteListRequest'}
        else:
            for site in self.server.sites.values():
                peer = site.peers.get('%s:%s' % (sender['ip'], sender['port']))
                if peer:
                    peer.found('local')

    def actionSiteListRequest(self, sender, params):
        if False:
            for i in range(10):
                print('nop')
        back = []
        sites = list(self.server.sites.values())
        site_groups = [sites[i:i + 100] for i in range(0, len(sites), 100)]
        for site_group in site_groups:
            res = {}
            res['sites_changed'] = self.server.site_manager.sites_changed
            res['sites'] = [site.address_hash for site in site_group]
            back.append({'cmd': 'siteListResponse', 'params': res})
        return back

    def actionSiteListResponse(self, sender, params):
        if False:
            return 10
        s = time.time()
        peer_sites = set(params['sites'])
        num_found = 0
        added_sites = []
        for site in self.server.sites.values():
            if site.address_hash in peer_sites:
                added = site.addPeer(sender['ip'], sender['port'], source='local')
                num_found += 1
                if added:
                    site.worker_manager.onPeers()
                    site.updateWebsocket(peers_added=1)
                    added_sites.append(site)
        if sender['peer_id'] not in self.known_peers:
            self.known_peers[sender['peer_id']] = {'added': time.time()}
        self.known_peers[sender['peer_id']]['sites_changed'] = params['sites_changed']
        self.known_peers[sender['peer_id']]['updated'] = time.time()
        self.known_peers[sender['peer_id']]['found'] = time.time()
        self.log.debug('Tracker result: Discover from %s response parsed in %.3fs, found: %s added: %s of %s' % (sender['ip'], time.time() - s, num_found, added_sites, len(peer_sites)))

@PluginManager.registerTo('FileServer')
class FileServerPlugin(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(FileServerPlugin, self).__init__(*args, **kwargs)
        if config.broadcast_port and config.tor != 'always' and (not config.disable_udp):
            self.local_announcer = LocalAnnouncer(self, config.broadcast_port)
        else:
            self.local_announcer = None

    def start(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.local_announcer:
            gevent.spawn(self.local_announcer.start)
        return super(FileServerPlugin, self).start(*args, **kwargs)

    def stop(self):
        if False:
            return 10
        if self.local_announcer:
            self.local_announcer.stop()
        res = super(FileServerPlugin, self).stop()
        return res

@PluginManager.registerTo('ConfigPlugin')
class ConfigPlugin(object):

    def createArguments(self):
        if False:
            return 10
        group = self.parser.add_argument_group('AnnounceLocal plugin')
        group.add_argument('--broadcast_port', help='UDP broadcasting port for local peer discovery', default=1544, type=int, metavar='port')
        return super(ConfigPlugin, self).createArguments()