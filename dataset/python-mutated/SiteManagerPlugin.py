import logging
import re
import time
from Config import config
from Plugin import PluginManager
allow_reload = False
log = logging.getLogger('ZeronamePlugin')

@PluginManager.registerTo('SiteManager')
class SiteManagerPlugin(object):
    site_zeroname = None
    db_domains = {}
    db_domains_modified = None

    def load(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SiteManagerPlugin, self).load(*args, **kwargs)
        if not self.get(config.bit_resolver):
            self.need(config.bit_resolver)

    def isBitDomain(self, address):
        if False:
            for i in range(10):
                print('nop')
        return re.match('(.*?)([A-Za-z0-9_-]+\\.bit)$', address)

    def resolveBitDomain(self, domain):
        if False:
            return 10
        domain = domain.lower()
        if not self.site_zeroname:
            self.site_zeroname = self.need(config.bit_resolver)
        site_zeroname_modified = self.site_zeroname.content_manager.contents.get('content.json', {}).get('modified', 0)
        if not self.db_domains or self.db_domains_modified != site_zeroname_modified:
            self.site_zeroname.needFile('data/names.json', priority=10)
            s = time.time()
            try:
                self.db_domains = self.site_zeroname.storage.loadJson('data/names.json')
            except Exception as err:
                log.error('Error loading names.json: %s' % err)
            log.debug('Domain db with %s entries loaded in %.3fs (modification: %s -> %s)' % (len(self.db_domains), time.time() - s, self.db_domains_modified, site_zeroname_modified))
            self.db_domains_modified = site_zeroname_modified
        return self.db_domains.get(domain)

    def resolveDomain(self, domain):
        if False:
            print('Hello World!')
        return self.resolveBitDomain(domain) or super(SiteManagerPlugin, self).resolveDomain(domain)

    def isDomain(self, address):
        if False:
            while True:
                i = 10
        return self.isBitDomain(address) or super(SiteManagerPlugin, self).isDomain(address)

@PluginManager.registerTo('ConfigPlugin')
class ConfigPlugin(object):

    def createArguments(self):
        if False:
            print('Hello World!')
        group = self.parser.add_argument_group('Zeroname plugin')
        group.add_argument('--bit_resolver', help='ZeroNet site to resolve .bit domains', default='1Name2NXVi1RDPDgf5617UoW7xA6YrhM9F', metavar='address')
        return super(ConfigPlugin, self).createArguments()