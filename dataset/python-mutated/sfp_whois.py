import ipwhois
import netaddr
import whois
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_whois(SpiderFootPlugin):
    meta = {'name': 'Whois', 'summary': 'Perform a WHOIS look-up on domain names and owned netblocks.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries']}
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
            i = 10
            return i + 15
        return ['DOMAIN_NAME', 'DOMAIN_NAME_PARENT', 'NETBLOCK_OWNER', 'NETBLOCKV6_OWNER', 'CO_HOSTED_SITE_DOMAIN', 'AFFILIATE_DOMAIN_NAME', 'SIMILARDOMAIN']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['DOMAIN_WHOIS', 'NETBLOCK_WHOIS', 'DOMAIN_REGISTRAR', 'CO_HOSTED_SITE_DOMAIN_WHOIS', 'AFFILIATE_DOMAIN_WHOIS', 'SIMILARDOMAIN_WHOIS']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName.startswith('DOMAIN_NAME'):
            typ = 'DOMAIN_WHOIS'
        elif eventName.startswith('NETBLOCK'):
            typ = 'NETBLOCK_WHOIS'
        elif eventName.startswith('AFFILIATE_DOMAIN_NAME'):
            typ = 'AFFILIATE_DOMAIN_WHOIS'
        elif eventName.startswith('CO_HOSTED_SITE_DOMAIN'):
            typ = 'CO_HOSTED_SITE_DOMAIN_WHOIS'
        elif eventName == 'SIMILARDOMAIN':
            typ = 'SIMILARDOMAIN_WHOIS'
        else:
            self.error(f'Invalid event type: {eventName}')
            return
        data = None
        if eventName in ['NETBLOCK_OWNER', 'NETBLOCKV6_OWNER']:
            try:
                netblock = netaddr.IPNetwork(eventData)
            except Exception as e:
                self.error(f'Invalid netblock {eventData}: {e}')
                return
            ip = netblock[0]
            self.debug(f'Sending RDAP query for IP address: {ip}')
            try:
                r = ipwhois.IPWhois(ip)
                data = str(r.lookup_rdap(depth=1))
            except Exception as e:
                self.error(f'Unable to perform WHOIS query on {ip}: {e}')
        else:
            self.debug(f'Sending WHOIS query for domain: {eventData}')
            try:
                whoisdata = whois.whois(eventData)
                data = str(whoisdata.text)
            except Exception as e:
                self.error(f'Unable to perform WHOIS query on {eventData}: {e}')
        if not data:
            self.error(f'No WHOIS record for {eventData}')
            return
        if len(str(data)) < 250:
            self.error(f'WHOIS data ({len(data)} bytes) is smaller than 250 bytes. Throttling from WHOIS server is probably happening. Ignoring response.')
            return
        rawevt = SpiderFootEvent(typ, data, self.__name__, event)
        self.notifyListeners(rawevt)
        if eventName.startswith('DOMAIN_NAME'):
            if whoisdata:
                registrar = whoisdata.get('registrar')
                if registrar:
                    evt = SpiderFootEvent('DOMAIN_REGISTRAR', registrar, self.__name__, event)
                    self.notifyListeners(evt)