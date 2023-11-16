import importlib
import random
import threading
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnsbrute(SpiderFootPlugin):
    meta = {'name': 'DNS Brute-forcer', 'summary': 'Attempts to identify hostnames through brute-forcing common names and iterations.', 'flags': [], 'useCases': ['Footprint', 'Investigate'], 'categories': ['DNS']}
    opts = {'skipcommonwildcard': True, 'domainonly': True, 'commons': True, 'top10000': False, 'numbersuffix': True, 'numbersuffixlimit': True, '_maxthreads': 100}
    optdescs = {'skipcommonwildcard': "If wildcard DNS is detected, don't bother brute-forcing.", 'domainonly': 'Only attempt to brute-force names on domain names, not hostnames (some hostnames are also sub-domains).', 'commons': 'Try a list of about 750 common hostnames/sub-domains.', 'top10000': 'Try a further 10,000 common hostnames/sub-domains. Will make the scan much slower.', 'numbersuffix': 'For any host found, try appending 1, 01, 001, -1, -01, -001, 2, 02, etc. (up to 10)', 'numbersuffixlimit': 'Limit using the number suffixes for hosts that have already been resolved? If disabled this will significantly extend the duration of scans.', '_maxthreads': 'Maximum threads'}
    events = None
    sublist = None
    lock = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.sublist = self.tempStorage()
        self.events = self.tempStorage()
        self.__dataSource__ = 'DNS'
        self.lock = threading.Lock()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]
        if self.opts['commons']:
            with importlib.resources.open_text('spiderfoot.dicts', 'subdomains.txt') as f:
                for s in f.readlines():
                    s = s.strip()
                    self.sublist[s] = True
        if self.opts['top10000']:
            with importlib.resources.open_text('spiderfoot.dicts', 'subdomains-10000.txt') as f:
                for s in f.readlines():
                    s = s.strip()
                    self.sublist[s] = True

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        ret = ['DOMAIN_NAME']
        if not self.opts['domainonly'] or self.opts['numbersuffix']:
            ret.append('INTERNET_NAME')
        return ret

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['INTERNET_NAME']

    def tryHost(self, name):
        if False:
            while True:
                i = 10
        try:
            if self.sf.resolveHost(name) or self.sf.resolveHost6(name):
                with self.lock:
                    self.hostResults[name] = True
        except Exception:
            with self.lock:
                self.hostResults[name] = False

    def tryHostWrapper(self, hostList, sourceEvent):
        if False:
            i = 10
            return i + 15
        self.hostResults = dict()
        running = True
        i = 0
        t = []
        self.info('Spawning threads to check hosts: ' + str(hostList))
        for name in hostList:
            tn = 'thread_sfp_dnsbrute_' + str(random.SystemRandom().randint(1, 999999999))
            t.append(threading.Thread(name=tn, target=self.tryHost, args=(name,)))
            t[i].start()
            i += 1
        while running:
            found = False
            for rt in threading.enumerate():
                if rt.name.startswith('thread_sfp_dnsbrute_'):
                    found = True
            if not found:
                running = False
            time.sleep(0.05)
        for res in self.hostResults:
            if self.hostResults.get(res, False):
                self.sendEvent(sourceEvent, res)

    def sendEvent(self, source, result):
        if False:
            return 10
        self.info('Found a brute-forced host: ' + result)
        evt = SpiderFootEvent('INTERNET_NAME', result, self.__name__, source)
        self.notifyListeners(evt)

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventDataHash = self.sf.hashstring(eventData)
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_dnsbrute':
            return
        if eventDataHash in self.events:
            return
        self.events[eventDataHash] = True
        if eventName == 'INTERNET_NAME' and (not self.getTarget().matches(eventData, includeChildren=False)):
            if not self.opts['numbersuffix']:
                return
            if self.checkForStop():
                return
            (h, dom) = eventData.split('.', 1)
            wildcard = self.sf.checkDnsWildcard(dom)
            if self.opts['skipcommonwildcard'] and wildcard:
                self.debug('Wildcard DNS detected on ' + dom + ' so skipping host iteration.')
                return
            dom = '.' + dom
            nextsubs = dict()
            for i in range(10):
                nextsubs[h + str(i) + dom] = True
                nextsubs[h + '0' + str(i) + dom] = True
                nextsubs[h + '00' + str(i) + dom] = True
                nextsubs[h + '-' + str(i) + dom] = True
                nextsubs[h + '-0' + str(i) + dom] = True
                nextsubs[h + '-00' + str(i) + dom] = True
            self.tryHostWrapper(list(nextsubs.keys()), event)
            return
        if not self.getTarget().matches(eventData, includeChildren=False):
            return
        self.debug('Iterating through possible sub-domains.')
        wildcard = self.sf.checkDnsWildcard(eventData)
        if self.opts['skipcommonwildcard'] and wildcard:
            self.debug('Wildcard DNS detected.')
            return
        targetList = list()
        for sub in self.sublist:
            if self.checkForStop():
                return
            name = f'{sub}.{eventData}'
            if len(targetList) <= self.opts['_maxthreads']:
                targetList.append(name)
            else:
                self.tryHostWrapper(targetList, event)
                targetList = list()
        if len(targetList) > 0:
            self.tryHostWrapper(targetList, event)
        if self.opts['numbersuffix'] and (not self.opts['numbersuffixlimit']):
            nextsubs = dict()
            dom = '.' + eventData
            for s in self.sublist:
                if self.checkForStop():
                    return
                for i in range(10):
                    nextsubs[s + str(i) + dom] = True
                    nextsubs[s + '0' + str(i) + dom] = True
                    nextsubs[s + '00' + str(i) + dom] = True
                    nextsubs[s + '-' + str(i) + dom] = True
                    nextsubs[s + '-0' + str(i) + dom] = True
                    nextsubs[s + '-00' + str(i) + dom] = True
                if len(list(nextsubs.keys())) >= self.opts['_maxthreads']:
                    self.tryHostWrapper(list(nextsubs.keys()), event)
                    nextsubs = dict()
            if len(nextsubs) > 0:
                self.tryHostWrapper(list(nextsubs.keys()), event)