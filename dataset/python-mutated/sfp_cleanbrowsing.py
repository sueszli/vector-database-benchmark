import dns.resolver
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_cleanbrowsing(SpiderFootPlugin):
    meta = {'name': 'CleanBrowsing.org', 'summary': 'Check if a host would be blocked by CleanBrowsing.org DNS content filters.', 'flags': [], 'useCases': ['Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://cleanbrowsing.org/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://cleanbrowsing.org/guides/', 'https://cleanbrowsing.org/filters/', 'https://cleanbrowsing.org/how-it-works', 'https://cleanbrowsing.org/web-filtering-for-shools-and-cipa-compliance', 'https://cleanbrowsing.org/getting-started'], 'favIcon': 'https://cleanbrowsing.org/favicon-new.ico', 'logo': 'https://cleanbrowsing.org/images/logos/CleanBrowsing-logo-large-2019-Orange-II.png', 'description': 'You get to decide what type of content is allowed in your home or network via our DNS-based content filtering service. Parents can protect their kids from adult content, schools can be CIPA compliant and businesses can block malicious domains and gain visibility into their network.\nCleanBrowsing is a DNS-based content filtering service that offers a safe way to browse the web without surprises. It intercepts domain requests and filter sites that should be blocked, based on your requirements. Our free family filter, for example, blocks adult content, while still allowing Google, Youtube, Bing, DuckDuckGo and the rest of the web to load safely.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['INTERNET_NAME', 'AFFILIATE_INTERNET_NAME', 'CO_HOSTED_SITE']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['BLACKLISTED_INTERNET_NAME', 'BLACKLISTED_AFFILIATE_INTERNET_NAME', 'BLACKLISTED_COHOST', 'MALICIOUS_INTERNET_NAME', 'MALICIOUS_AFFILIATE_INTERNET_NAME', 'MALICIOUS_COHOST']

    def queryFamilyDNS(self, qaddr):
        if False:
            for i in range(10):
                print('nop')
        res = dns.resolver.Resolver()
        res.nameservers = ['185.228.168.168', '185.228.168.169']
        try:
            addrs = res.resolve(qaddr)
            self.debug(f'Addresses returned: {addrs}')
        except Exception:
            self.debug(f'Unable to resolve {qaddr}')
            return False
        if addrs:
            return True
        return False

    def queryAdultDNS(self, qaddr):
        if False:
            print('Hello World!')
        res = dns.resolver.Resolver()
        res.nameservers = ['185.228.168.10', '185.228.169.11']
        try:
            addrs = res.resolve(qaddr)
            self.debug(f'Addresses returned: {addrs}')
        except Exception:
            self.debug(f'Unable to resolve {qaddr}')
            return False
        if addrs:
            return True
        return False

    def querySecurityDNS(self, qaddr):
        if False:
            while True:
                i = 10
        res = dns.resolver.Resolver()
        res.nameservers = ['185.228.168.9', '185.228.169.9']
        try:
            addrs = res.resolve(qaddr)
            self.debug(f'Addresses returned: {addrs}')
        except Exception:
            self.debug(f'Unable to resolve {qaddr}')
            return False
        if addrs:
            return True
        return False

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {event.module}')
        if eventData in self.results:
            return
        self.results[eventData] = True
        if eventName == 'INTERNET_NAME':
            malicious_type = 'MALICIOUS_INTERNET_NAME'
            blacklist_type = 'BLACKLISTED_INTERNET_NAME'
        elif eventName == 'AFFILIATE_INTERNET_NAME':
            malicious_type = 'MALICIOUS_AFFILIATE_INTERNET_NAME'
            blacklist_type = 'BLACKLISTED_AFFILIATE_INTERNET_NAME'
        elif eventName == 'CO_HOSTED_SITE':
            malicious_type = 'MALICIOUS_COHOST'
            blacklist_type = 'BLACKLISTED_COHOST'
        else:
            self.debug(f'Unexpected event type {eventName}, skipping')
            return
        if not self.sf.resolveHost(eventData) and (not self.sf.resolveHost6(eventData)):
            return
        family = self.queryFamilyDNS(eventData)
        adult = self.queryAdultDNS(eventData)
        security = self.querySecurityDNS(eventData)
        if family and adult and security:
            return
        self.debug(f'{eventData} was blocked by CleanBrowsing DNS')
        if not security:
            evt = SpiderFootEvent(blacklist_type, f'CleanBrowsing DNS - Security [{eventData}]', self.__name__, event)
            self.notifyListeners(evt)
            evt = SpiderFootEvent(malicious_type, f'CleanBrowsing DNS - Security [{eventData}]', self.__name__, event)
            self.notifyListeners(evt)
        elif not adult:
            evt = SpiderFootEvent(blacklist_type, f'CleanBrowsing DNS - Adult [{eventData}]', self.__name__, event)
            self.notifyListeners(evt)
        elif not family:
            evt = SpiderFootEvent(blacklist_type, f'CleanBrowsing DNS - Family [{eventData}]', self.__name__, event)
            self.notifyListeners(evt)