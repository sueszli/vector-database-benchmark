import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ipapico(SpiderFootPlugin):
    meta = {'name': 'ipapi.co', 'summary': 'Queries ipapi.co to identify geolocation of IP Addresses using ipapi.co API', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Real World'], 'dataSource': {'website': 'https://ipapi.co/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://ipapi.co/api/'], 'favIcon': 'https://ipapi.co/static/images/favicon.b64f1de785e1.ico', 'logo': 'https://ipapi.co/static/images/favicon.34f0ec468301.png', 'description': 'Powerful & Simple REST API for IP Address Geolocation.ipapi.co provides a REST API to find the location of an IP address.'}}
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
            print('Hello World!')
        return ['IP_ADDRESS', 'IPV6_ADDRESS']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['GEOINFO', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            return 10
        queryString = f'https://ipapi.co/{qry}/json/'
        res = self.sf.fetchUrl(queryString, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        time.sleep(1.5)
        if res['content'] is None:
            self.info(f'No ipapi.co data found for {qry}')
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
        return None

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.query(eventData)
        if data is None:
            self.info('No results returned from ipapi.co')
            return
        if data.get('country'):
            location = ', '.join(filter(None, [data.get('city'), data.get('region'), data.get('region_code'), data.get('country_name'), data.get('country')]))
            evt = SpiderFootEvent('GEOINFO', location, self.__name__, event)
            self.notifyListeners(evt)
            evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
            self.notifyListeners(evt)