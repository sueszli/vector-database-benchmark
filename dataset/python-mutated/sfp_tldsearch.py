import random
import threading
import time
import dns.resolver
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_tldsearch(SpiderFootPlugin):
    meta = {'name': 'TLD Searcher', 'summary': 'Search all Internet TLDs for domains with the same name as the target (this can be very slow.)', 'flags': ['slow'], 'useCases': ['Footprint'], 'categories': ['DNS']}
    opts = {'activeonly': False, 'skipwildcards': True, '_maxthreads': 50}
    optdescs = {'activeonly': 'Only report domains that have content (try to fetch the page)?', 'skipwildcards': 'Skip TLDs and sub-TLDs that have wildcard DNS.', '_maxthreads': 'Maximum threads'}
    results = None
    tldResults = dict()
    lock = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'DNS'
        self.lock = threading.Lock()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['INTERNET_NAME']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['SIMILARDOMAIN']

    def tryTld(self, target, tld):
        if False:
            i = 10
            return i + 15
        resolver = dns.resolver.Resolver()
        resolver.timeout = 1
        resolver.lifetime = 1
        resolver.search = list()
        if self.opts.get('_dnsserver', '') != '':
            resolver.nameservers = [self.opts['_dnsserver']]
        if self.opts['skipwildcards'] and self.sf.checkDnsWildcard(tld):
            return
        try:
            if not self.sf.resolveHost(target) and (not self.sf.resolveHost6(target)):
                with self.lock:
                    self.tldResults[target] = False
            else:
                with self.lock:
                    self.tldResults[target] = True
        except Exception:
            with self.lock:
                self.tldResults[target] = False

    def tryTldWrapper(self, tldList, sourceEvent):
        if False:
            while True:
                i = 10
        self.tldResults = dict()
        running = True
        t = []
        self.info(f'Spawning threads to check TLDs: {tldList}')
        for (i, pair) in enumerate(tldList):
            (domain, tld) = pair
            tn = 'thread_sfp_tldsearch_' + str(random.SystemRandom().randint(0, 999999999))
            t.append(threading.Thread(name=tn, target=self.tryTld, args=(domain, tld)))
            t[i].start()
        while running:
            found = False
            for rt in threading.enumerate():
                if rt.name.startswith('thread_sfp_tldsearch_'):
                    found = True
            if not found:
                running = False
            time.sleep(0.1)
        for res in self.tldResults:
            if self.getTarget().matches(res, includeParents=True, includeChildren=True):
                continue
            if self.tldResults[res] and res not in self.results:
                self.sendEvent(sourceEvent, res)

    def sendEvent(self, source, result):
        if False:
            return 10
        self.info("Found a TLD with the target's name: " + result)
        self.results[result] = True
        if self.opts['activeonly']:
            if self.checkForStop():
                return
            pageContent = self.sf.fetchUrl('http://' + result, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], noLog=True, verify=False)
            if pageContent['content'] is not None:
                evt = SpiderFootEvent('SIMILARDOMAIN', result, self.__name__, source)
                self.notifyListeners(evt)
        else:
            evt = SpiderFootEvent('SIMILARDOMAIN', result, self.__name__, source)
            self.notifyListeners(evt)

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        keyword = self.sf.domainKeyword(eventData, self.opts['_internettlds'])
        if not keyword:
            self.error(f'Failed to extract keyword from {eventData}')
            return
        self.debug(f'Keyword extracted from {eventData}: {keyword}')
        if keyword in self.results:
            return
        self.results[keyword] = True
        targetList = list()
        for tld in self.opts['_internettlds']:
            if type(tld) != str:
                tld = str(tld.strip(), errors='ignore')
            else:
                tld = tld.strip()
            if tld.startswith('//') or len(tld) == 0:
                continue
            if tld.startswith('!') or tld.startswith('*') or tld.startswith('..'):
                continue
            if tld.endswith('.arpa'):
                continue
            tryDomain = keyword + '.' + tld
            if self.checkForStop():
                return
            if len(targetList) <= self.opts['_maxthreads']:
                targetList.append([tryDomain, tld])
            else:
                self.tryTldWrapper(targetList, event)
                targetList = list()
        if len(targetList) > 0:
            self.tryTldWrapper(targetList, event)