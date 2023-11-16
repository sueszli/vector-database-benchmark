import logging
import json
import time
import binascii
import gevent
import util
from Crypt import CryptBitcoin
from Plugin import PluginManager
from Config import config
from util import helper
from Debug import Debug

@PluginManager.acceptPlugins
class User(object):

    def __init__(self, master_address=None, master_seed=None, data={}):
        if False:
            print('Hello World!')
        if master_seed:
            self.master_seed = master_seed
            self.master_address = CryptBitcoin.privatekeyToAddress(self.master_seed)
        elif master_address:
            self.master_address = master_address
            self.master_seed = data.get('master_seed')
        else:
            self.master_seed = CryptBitcoin.newSeed()
            self.master_address = CryptBitcoin.privatekeyToAddress(self.master_seed)
        self.sites = data.get('sites', {})
        self.certs = data.get('certs', {})
        self.settings = data.get('settings', {})
        self.delayed_save_thread = None
        self.log = logging.getLogger('User:%s' % self.master_address)

    @util.Noparallel(queue=True, ignore_class=True)
    def save(self):
        if False:
            for i in range(10):
                print('nop')
        s = time.time()
        users = json.load(open('%s/users.json' % config.data_dir))
        if self.master_address not in users:
            users[self.master_address] = {}
        user_data = users[self.master_address]
        if self.master_seed:
            user_data['master_seed'] = self.master_seed
        user_data['sites'] = self.sites
        user_data['certs'] = self.certs
        user_data['settings'] = self.settings
        helper.atomicWrite('%s/users.json' % config.data_dir, helper.jsonDumps(users).encode('utf8'))
        self.log.debug('Saved in %.3fs' % (time.time() - s))
        self.delayed_save_thread = None

    def saveDelayed(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.delayed_save_thread:
            self.delayed_save_thread = gevent.spawn_later(5, self.save)

    def getAddressAuthIndex(self, address):
        if False:
            return 10
        return int(binascii.hexlify(address.encode()), 16)

    @util.Noparallel()
    def generateAuthAddress(self, address):
        if False:
            return 10
        s = time.time()
        address_id = self.getAddressAuthIndex(address)
        auth_privatekey = CryptBitcoin.hdPrivatekey(self.master_seed, address_id)
        self.sites[address] = {'auth_address': CryptBitcoin.privatekeyToAddress(auth_privatekey), 'auth_privatekey': auth_privatekey}
        self.saveDelayed()
        self.log.debug('Added new site: %s in %.3fs' % (address, time.time() - s))
        return self.sites[address]

    def getSiteData(self, address, create=True):
        if False:
            for i in range(10):
                print('nop')
        if address not in self.sites:
            if not create:
                return {'auth_address': None, 'auth_privatekey': None}
            self.generateAuthAddress(address)
        return self.sites[address]

    def deleteSiteData(self, address):
        if False:
            i = 10
            return i + 15
        if address in self.sites:
            del self.sites[address]
            self.saveDelayed()
            self.log.debug('Deleted site: %s' % address)

    def setSiteSettings(self, address, settings):
        if False:
            while True:
                i = 10
        site_data = self.getSiteData(address)
        site_data['settings'] = settings
        self.saveDelayed()
        return site_data

    def getNewSiteData(self):
        if False:
            return 10
        import random
        bip32_index = random.randrange(2 ** 256) % 100000000
        site_privatekey = CryptBitcoin.hdPrivatekey(self.master_seed, bip32_index)
        site_address = CryptBitcoin.privatekeyToAddress(site_privatekey)
        if site_address in self.sites:
            raise Exception('Random error: site exist!')
        self.getSiteData(site_address)
        self.sites[site_address]['privatekey'] = site_privatekey
        self.save()
        return (site_address, bip32_index, self.sites[site_address])

    def getAuthAddress(self, address, create=True):
        if False:
            print('Hello World!')
        cert = self.getCert(address)
        if cert:
            return cert['auth_address']
        else:
            return self.getSiteData(address, create)['auth_address']

    def getAuthPrivatekey(self, address, create=True):
        if False:
            for i in range(10):
                print('nop')
        cert = self.getCert(address)
        if cert:
            return cert['auth_privatekey']
        else:
            return self.getSiteData(address, create)['auth_privatekey']

    def addCert(self, auth_address, domain, auth_type, auth_user_name, cert_sign):
        if False:
            i = 10
            return i + 15
        auth_privatekey = [site['auth_privatekey'] for site in list(self.sites.values()) if site['auth_address'] == auth_address][0]
        cert_node = {'auth_address': auth_address, 'auth_privatekey': auth_privatekey, 'auth_type': auth_type, 'auth_user_name': auth_user_name, 'cert_sign': cert_sign}
        if self.certs.get(domain) and self.certs[domain] != cert_node:
            return False
        elif self.certs.get(domain) == cert_node:
            return None
        else:
            self.certs[domain] = cert_node
            self.save()
            return True

    def deleteCert(self, domain):
        if False:
            print('Hello World!')
        del self.certs[domain]

    def setCert(self, address, domain):
        if False:
            for i in range(10):
                print('nop')
        site_data = self.getSiteData(address)
        if domain:
            site_data['cert'] = domain
        elif 'cert' in site_data:
            del site_data['cert']
        self.saveDelayed()
        return site_data

    def getCert(self, address):
        if False:
            for i in range(10):
                print('nop')
        site_data = self.getSiteData(address, create=False)
        if not site_data or 'cert' not in site_data:
            return None
        return self.certs.get(site_data['cert'])

    def getCertUserId(self, address):
        if False:
            while True:
                i = 10
        site_data = self.getSiteData(address, create=False)
        if not site_data or 'cert' not in site_data:
            return None
        cert = self.certs.get(site_data['cert'])
        if cert:
            return cert['auth_user_name'] + '@' + site_data['cert']