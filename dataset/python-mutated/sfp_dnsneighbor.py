import ipaddress
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnsneighbor(SpiderFootPlugin):
    meta = {'name': 'DNS Look-aside', 'summary': 'Attempt to reverse-resolve the IP addresses next to your target to see if they are related.', 'flags': [], 'useCases': ['Footprint', 'Investigate'], 'categories': ['DNS']}
    opts = {'lookasidebits': 4, 'validatereverse': True}
    optdescs = {'validatereverse': 'Validate that reverse-resolved hostnames still resolve back to that IP before considering them as aliases of your target.', 'lookasidebits': 'If look-aside is enabled, the netmask size (in CIDR notation) to check. Default is 4 bits (16 hosts).'}
    events = None
    domresults = None
    hostresults = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.events = self.tempStorage()
        self.domresults = self.tempStorage()
        self.hostresults = self.tempStorage()
        self.__dataSource__ = 'DNS'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['IP_ADDRESS']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['AFFILIATE_IPADDR', 'IP_ADDRESS']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventDataHash = self.sf.hashstring(eventData)
        addrs = None
        parentEvent = event
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventDataHash in self.events:
            return
        self.events[eventDataHash] = True
        try:
            address = ipaddress.ip_address(eventData)
            netmask = address.max_prefixlen - min(address.max_prefixlen, max(1, int(self.opts.get('lookasidebits'))))
            network = ipaddress.ip_network(f'{eventData}/{netmask}', strict=False)
        except ValueError:
            self.error(f'Invalid IP address received: {eventData}')
            return
        self.debug(f'Lookaside max: {network.network_address}, min: {network.broadcast_address}')
        for ip in network:
            sip = str(ip)
            self.debug('Attempting look-aside lookup of: ' + sip)
            if self.checkForStop():
                return
            if sip in self.hostresults or sip == eventData:
                continue
            addrs = self.sf.resolveIP(sip)
            if not addrs:
                self.debug('Look-aside resolve for ' + sip + ' failed.')
                continue
            if self.getTarget().matches(sip):
                affil = False
            else:
                affil = True
                for a in addrs:
                    if self.getTarget().matches(a):
                        affil = False
            self.events[sip] = True
            ev = self.processHost(sip, parentEvent, affil)
            if not ev:
                continue
            for addr in addrs:
                if self.checkForStop():
                    return
                if addr == sip:
                    continue
                if self.sf.validIP(addr) or self.sf.validIP6(addr):
                    parent = parentEvent
                else:
                    parent = ev
                if self.getTarget().matches(addr):
                    self.processHost(addr, parent, False)
                else:
                    self.processHost(addr, parent, True)

    def processHost(self, host, parentEvent, affiliate=None):
        if False:
            while True:
                i = 10
        parentHash = self.sf.hashstring(parentEvent.data)
        if host not in self.hostresults:
            self.hostresults[host] = [parentHash]
        else:
            if parentHash in self.hostresults[host] or parentEvent.data == host:
                self.debug('Skipping host, ' + host + ', already processed.')
                return None
            self.hostresults[host] = self.hostresults[host] + [parentHash]
        self.debug('Found host: ' + host)
        if affiliate is None:
            affil = True
            if self.getTarget().matches(host):
                affil = False
            elif not self.sf.validIP(host) and (not self.sf.validIP6(host)):
                hostips = self.sf.resolveHost(host)
                if hostips:
                    for hostip in hostips:
                        if self.getTarget().matches(hostip):
                            affil = False
                            break
                hostips6 = self.sf.resolveHost6(host)
                if hostips6:
                    for hostip in hostips6:
                        if self.getTarget().matches(hostip):
                            affil = False
                            break
        else:
            affil = affiliate
        if not self.sf.validIP(host):
            return None
        if affil:
            htype = 'AFFILIATE_IPADDR'
        else:
            htype = 'IP_ADDRESS'
        if not htype:
            return None
        evt = SpiderFootEvent(htype, host, self.__name__, parentEvent)
        self.notifyListeners(evt)
        return evt