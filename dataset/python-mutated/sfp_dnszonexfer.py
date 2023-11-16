import re
import dns.query
import dns.zone
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnszonexfer(SpiderFootPlugin):
    meta = {'name': 'DNS Zone Transfer', 'summary': 'Attempts to perform a full DNS zone transfer.', 'flags': [], 'useCases': ['Footprint', 'Investigate'], 'categories': ['DNS']}
    opts = {'timeout': 30}
    optdescs = {'timeout': 'Timeout in seconds'}
    events = None

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
            i = 10
            return i + 15
        return ['PROVIDER_DNS']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['RAW_DNS_RECORDS', 'INTERNET_NAME']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventDataHash = self.sf.hashstring(eventData)
        parentEvent = event
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_dnszonexfer':
            self.debug(f'Ignoring {eventName}, from self.')
            return
        if eventDataHash in self.events:
            self.debug('Skipping duplicate event for ' + eventData)
            return
        self.events[eventDataHash] = True
        res = dns.resolver.Resolver()
        if self.opts.get('_dnsserver', '') != '':
            res.nameservers = [self.opts['_dnsserver']]
        if not self.sf.validIP(eventData) and (not self.sf.validIP6(eventData)):
            nsips = self.sf.resolveHost(eventData)
            if not nsips:
                return
            if not nsips:
                self.error("Couldn't resolve the name server, so not attempting zone transfer.")
                return
            for n in nsips:
                if self.sf.validIP(n):
                    nsip = n
                    break
        else:
            nsip = eventData
        for name in self.getTarget().getNames():
            self.debug('Trying for name: ' + name)
            try:
                ret = list()
                z = dns.zone.from_xfr(dns.query.xfr(nsip, name, timeout=int(self.opts['timeout'])))
                names = list(z.nodes.keys())
                for n in names:
                    ret.append(z[n].to_text(n))
                evt = SpiderFootEvent('RAW_DNS_RECORDS', '\n'.join(ret), self.__name__, parentEvent)
                self.notifyListeners(evt)
                for row in ret:
                    pat = re.compile('^(\\S+)\\.?\\s+\\d+\\s+IN\\s+[AC].*', re.IGNORECASE | re.DOTALL)
                    grps = re.findall(pat, row)
                    if len(grps) > 0:
                        for strdata in grps:
                            self.debug('Matched: ' + strdata)
                            if strdata.endswith('.'):
                                strdata = strdata[:-1]
                            else:
                                strdata = strdata + '.' + name
                            evt = SpiderFootEvent('INTERNET_NAME', strdata, self.__name__, parentEvent)
                            self.notifyListeners(evt)
            except Exception as e:
                self.info(f'Unable to perform DNS zone transfer for {eventData} ({name}): {e}')