import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_arin(SpiderFootPlugin):
    meta = {'name': 'ARIN', 'summary': 'Queries ARIN registry for contact information.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries'], 'dataSource': {'website': 'https://www.arin.net/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://www.arin.net/resources/', 'https://www.arin.net/reference/', 'https://www.arin.net/participate/', 'https://www.arin.net/resources/guide/request/', 'https://www.arin.net/resources/registry/transfers/', 'https://www.arin.net/resources/guide/ipv6/'], 'favIcon': 'https://www.arin.net/img/favicon.ico', 'logo': 'https://www.arin.net/img/logo-stnd.svg', 'description': 'ARIN is a nonprofit, member-based organization that administers IP addresses & ASNs in support of the operation and growth of the Internet.\nEstablished in December 1997 as a Regional Internet Registry, the American Registry for Internet Numbers (ARIN) is responsible for the management and distribution of Internet number resources such as Internet Protocol (IP) addresses and Autonomous System Numbers (ASNs). ARIN manages these resources within its service region, which is comprised of Canada, the United States, and many Caribbean and North Atlantic islands.'}}
    opts = {}
    optdescs = {}
    results = None
    currentEventSrc = None
    keywords = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        self.currentEventSrc = None
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['DOMAIN_NAME', 'HUMAN_NAME']

    def producedEvents(self):
        if False:
            return 10
        return ['RAW_RIR_DATA', 'HUMAN_NAME']

    def fetchRir(self, url):
        if False:
            print('Hello World!')
        head = {'Accept': 'application/json'}
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], headers=head)
        if res['content'] is not None and res['code'] != '404':
            return res
        return None

    def query(self, qtype, value):
        if False:
            while True:
                i = 10
        url = 'https://whois.arin.net/rest/'
        if qtype == 'domain':
            url += 'pocs;domain=@' + value
        try:
            if qtype == 'name':
                (fname, lname) = value.split(' ', 1)
                if fname.endswith(','):
                    t = fname
                    fname = lname
                    lname = t
                url += 'pocs;first=' + fname + ';last=' + lname
        except Exception as e:
            self.debug("Couldn't process name: " + value + ' (' + str(e) + ')')
            return None
        if qtype == 'contact':
            url = value
        res = self.fetchRir(url)
        if not res:
            self.debug('No info found/available for ' + value + ' at ARIN.')
            return None
        try:
            data = json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, self.currentEventSrc)
        self.notifyListeners(evt)
        return data

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.currentEventSrc = event
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName == 'DOMAIN_NAME':
            ret = self.query('domain', eventData)
            if not ret:
                return
            if 'pocs' in ret:
                if 'pocRef' in ret['pocs']:
                    ref = list()
                    if type(ret['pocs']['pocRef']) == dict:
                        ref = [ret['pocs']['pocRef']]
                    else:
                        ref = ret['pocs']['pocRef']
                    for p in ref:
                        name = p['@name']
                        if ',' in name:
                            sname = name.split(', ', 1)
                            name = sname[1] + ' ' + sname[0]
                        evt = SpiderFootEvent('HUMAN_NAME', name, self.__name__, self.currentEventSrc)
                        self.notifyListeners(evt)
                        self.query('contact', p['$'])
        if eventName == 'HUMAN_NAME':
            ret = self.query('name', eventData)
            if not ret:
                return
            if 'pocs' in ret:
                if 'pocRef' in ret['pocs']:
                    ref = list()
                    if type(ret['pocs']['pocRef']) == dict:
                        ref = [ret['pocs']['pocRef']]
                    else:
                        ref = ret['pocs']['pocRef']
                    for p in ref:
                        self.query('contact', p['$'])