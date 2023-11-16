import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_h1nobbdde(SpiderFootPlugin):
    meta = {'name': 'HackerOne (Unofficial)', 'summary': 'Check external vulnerability scanning/reporting service h1.nobbd.de to see if the target is listed.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Leaks, Dumps and Breaches'], 'dataSource': {'website': 'http://www.nobbd.de/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['http://www.nobbd.de/index.php#projekte', 'https://twitter.com/disclosedh1'], 'favIcon': 'http://www.nobbd.de/favicon.ico', 'logo': 'http://www.nobbd.de/favicon.ico', 'description': 'Unofficial Bug Monitoring platform for HackerOne.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['VULNERABILITY_DISCLOSURE']

    def queryOBB(self, qry):
        if False:
            print('Hello World!')
        ret = list()
        url = 'http://h1.nobbd.de/search.php?q=' + qry
        res = self.sf.fetchUrl(url, timeout=30, useragent=self.opts['_useragent'])
        if res['content'] is None:
            self.debug('No content returned from h1.nobbd.de')
            return None
        try:
            rx = re.compile('<a class="title" href=.(.[^"]+).*?title=.(.[^"\']+)', re.IGNORECASE | re.DOTALL)
            for m in rx.findall(str(res['content'])):
                if qry in m[1]:
                    ret.append(m[1] + '\n<SFURL>' + m[0] + '</SFURL>')
        except Exception as e:
            self.error(f'Error processing response from h1.nobbd.de: {e}')
            return None
        return ret

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        data = list()
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        obb = self.queryOBB(eventData)
        if obb:
            data.extend(obb)
        for n in data:
            e = SpiderFootEvent('VULNERABILITY_DISCLOSURE', n, self.__name__, event)
            self.notifyListeners(e)