import dns.resolver
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_cloudflaredns(SpiderFootPlugin):
    meta = {'name': 'CloudFlare DNS', 'summary': 'Check if a host would be blocked by CloudFlare DNS.', 'flags': [], 'useCases': ['Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://www.cloudflare.com/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://developers.cloudflare.com/1.1.1.1/1.1.1.1-for-families/'], 'favIcon': 'https://www.cloudflare.com/img/favicon/favicon-32x32.png', 'logo': 'https://www.cloudflare.com/img/logo-web-badges/cf-logo-on-white-bg.svg', 'description': '1.1.1.1 for Families is the easiest way to add a layer of protection to your home network and protect it from malware and adult content. 1.1.1.1 for Families leverages Cloudflare’s global network to ensure that it is fast and secure around the world.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['INTERNET_NAME', 'AFFILIATE_INTERNET_NAME', 'CO_HOSTED_SITE']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['BLACKLISTED_INTERNET_NAME', 'BLACKLISTED_AFFILIATE_INTERNET_NAME', 'BLACKLISTED_COHOST', 'MALICIOUS_INTERNET_NAME', 'MALICIOUS_AFFILIATE_INTERNET_NAME', 'MALICIOUS_COHOST']

    def queryFamilyDNS(self, qaddr):
        if False:
            print('Hello World!')
        res = dns.resolver.Resolver()
        res.nameservers = ['1.1.1.3', '1.0.0.3']
        try:
            return res.resolve(qaddr)
        except Exception:
            self.debug(f'Unable to resolve {qaddr}')
        return None

    def queryMalwareDNS(self, qaddr):
        if False:
            i = 10
            return i + 15
        res = dns.resolver.Resolver()
        res.nameservers = ['1.1.1.2', '1.0.0.2']
        try:
            return res.resolve(qaddr)
        except Exception:
            self.debug(f'Unable to resolve {qaddr}')
        return None

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {event.module}')
        if eventData in self.results:
            return
        self.results[eventData] = True
        if eventName == 'INTERNET_NAME':
            e = 'BLACKLISTED_INTERNET_NAME'
        elif eventName == 'AFFILIATE_INTERNET_NAME':
            e = 'BLACKLISTED_AFFILIATE_INTERNET_NAME'
        elif eventName == 'CO_HOSTED_SITE':
            e = 'BLACKLISTED_COHOST'
        else:
            self.debug(f'Unexpected event type {eventName}, skipping')
            return
        family = self.sf.normalizeDNS(self.queryFamilyDNS(eventData))
        malware = self.sf.normalizeDNS(self.queryMalwareDNS(eventData))
        if not family or not malware:
            return
        if '0.0.0.0' not in family and '0.0.0.0' not in malware:
            return
        if '0.0.0.0' not in malware:
            self.debug(f'{eventData} blocked by CloudFlare Family DNS')
            evt = SpiderFootEvent(e, f'CloudFlare - Family [{eventData}]', self.__name__, event)
            self.notifyListeners(evt)
            return
        self.debug(f'{eventData} blocked by CloudFlare Malware DNS')
        evt = SpiderFootEvent(e, f'CloudFlare - Malware [{eventData}]', self.__name__, event)
        self.notifyListeners(evt)
        if eventName == 'INTERNET_NAME':
            e = 'MALICIOUS_INTERNET_NAME'
        elif eventName == 'AFFILIATE_INTERNET_NAME':
            e = 'MALICIOUS_AFFILIATE_INTERNET_NAME'
        elif eventName == 'CO_HOSTED_SITE':
            e = 'MALICIOUS_COHOST'
        else:
            self.debug(f'Unexpected event type {eventName}, skipping')
        evt = SpiderFootEvent(e, f'CloudFlare - Malware [{eventData}]', self.__name__, event)
        self.notifyListeners(evt)