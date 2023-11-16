import dns.resolver
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnscommonsrv(SpiderFootPlugin):
    meta = {'name': 'DNS Common SRV', 'summary': 'Attempts to identify hostnames through brute-forcing common DNS SRV records.', 'flags': ['slow'], 'useCases': ['Footprint', 'Investigate'], 'categories': ['DNS']}
    opts = {}
    optdescs = {}
    events = None
    commonsrv = ['_ldap._tcp', '_gc._msdcs', '_ldap._tcp.pdc._msdcs', '_ldap._tcp.gc._msdcs', '_kerberos._tcp.dc._msdcs', '_kerberos._tcp', '_kerberos._udp', '_kerberos-master._tcp', '_kerberos-master._udp', '_kpasswd._tcp', '_kpasswd._udp', '_ntp._udp', '_sip._tcp', '_sip._udp', '_sip._tls', '_sips._tcp', '_stun._tcp', '_stun._udp', '_stuns._tcp', '_turn._tcp', '_turn._udp', '_turns._tcp', '_jabber._tcp', '_xmpp-client._tcp', '_xmpp-server._tcp']

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.events = self.tempStorage()
        self.__dataSource__ = 'DNS'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['INTERNET_NAME', 'DOMAIN_NAME']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['INTERNET_NAME', 'AFFILIATE_INTERNET_NAME']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_dnscommonsrv':
            self.debug(f'Ignoring {eventName}, from self.')
            return
        eventDataHash = self.sf.hashstring(eventData)
        parentEvent = event
        if eventDataHash in self.events:
            return
        self.events[eventDataHash] = True
        res = dns.resolver.Resolver()
        if self.opts.get('_dnsserver', '') != '':
            res.nameservers = [self.opts['_dnsserver']]
        self.debug('Iterating through possible SRV records.')
        for srv in self.commonsrv:
            if self.checkForStop():
                return
            name = srv + '.' + eventData
            if self.sf.hashstring(name) in self.events:
                continue
            try:
                answers = res.query(name, 'SRV', timeout=10)
            except Exception:
                answers = []
            if not answers:
                continue
            evt = SpiderFootEvent('DNS_SRV', name, self.__name__, parentEvent)
            self.notifyListeners(evt)
            for a in answers:
                tgt_clean = a.target.to_text().rstrip('.')
                if self.getTarget().matches(tgt_clean):
                    evt_type = 'INTERNET_NAME'
                else:
                    evt_type = 'AFFILIATE_INTERNET_NAME'
                evt = SpiderFootEvent(evt_type, tgt_clean, self.__name__, parentEvent)
                self.notifyListeners(evt)