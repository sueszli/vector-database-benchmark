from spiderfoot import SpiderFootEvent, SpiderFootPlugin
nearchars = {'a': ['4', 's'], 'b': ['v', 'n'], 'c': ['x', 'v'], 'd': ['s', 'f'], 'e': ['w', 'r'], 'f': ['d', 'g'], 'g': ['f', 'h'], 'h': ['g', 'j', 'n'], 'i': ['o', 'u', '1'], 'j': ['k', 'h', 'i'], 'k': ['l', 'j'], 'l': ['i', '1', 'k'], 'm': ['n'], 'n': ['m'], 'o': ['p', 'i', '0'], 'p': ['o', 'q'], 'r': ['t', 'e'], 's': ['a', 'd', '5'], 't': ['7', 'y', 'z', 'r'], 'u': ['v', 'i', 'y', 'z'], 'v': ['u', 'c', 'b'], 'w': ['v', 'vv', 'q', 'e'], 'x': ['z', 'y', 'c'], 'y': ['z', 'x'], 'z': ['y', 'x'], '0': ['o'], '1': ['l'], '2': ['5'], '3': ['e'], '4': ['a'], '5': ['s'], '6': ['b'], '7': ['t'], '8': ['b'], '9': []}
pairs = {'oo': ['00'], 'll': ['l1l', 'l1l', '111', '11'], '11': ['ll', 'lll', 'l1l', '1l1']}

class sfp_similar(SpiderFootPlugin):
    meta = {'name': 'Similar Domain Finder', 'summary': 'Search various sources to identify similar looking domain names, for instance squatted domains.', 'flags': [], 'useCases': ['Footprint', 'Investigate'], 'categories': ['DNS']}
    opts = {'skipwildcards': True}
    optdescs = {'skipwildcards': 'Skip TLDs and sub-TLDs that have wildcard DNS.'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'DNS'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['SIMILARDOMAIN']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventData = event.data
        dom = self.sf.domainKeyword(eventData, self.opts['_internettlds'])
        if not dom:
            return
        tld = '.' + eventData.split(dom + '.')[-1]
        self.debug(f'Keyword extracted from {eventData}: {dom}')
        if dom in self.results:
            return
        self.results[dom] = True
        if self.opts['skipwildcards'] and self.sf.checkDnsWildcard(tld[1:]):
            return
        domlist = list()
        pos = 0
        for c in dom:
            if c not in nearchars:
                continue
            if len(nearchars[c]) == 0:
                continue
            npos = pos + 1
            for xc in nearchars[c]:
                newdom = dom[0:pos] + xc + dom[npos:len(dom)]
                domlist.append(newdom)
            pos += 1
        for p in pairs:
            if p in dom:
                for r in pairs[p]:
                    domlist.append(dom.replace(p, r))
        for c in nearchars:
            domlist.append(dom + c)
            domlist.append(c + dom)
        for (pos, c) in enumerate(dom):
            domlist.append(dom[0:pos] + c + c + dom[pos + 1:len(dom)])
        for d in domlist:
            try:
                for domain in [f'{d}{tld}', f'www.{d}{tld}']:
                    if self.sf.resolveHost(domain) or self.sf.resolveHost6(domain):
                        self.debug(f'Resolved {domain}')
                        evt = SpiderFootEvent('SIMILARDOMAIN', f'{d}{tld}', self.__name__, event)
                        self.notifyListeners(evt)
                        break
            except Exception:
                continue