import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ipapicom(SpiderFootPlugin):
    meta = {'name': 'ipapi.com', 'summary': 'Queries ipapi.com to identify geolocation of IP Addresses using ipapi.com API', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Real World'], 'dataSource': {'website': 'https://ipapi.com/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://ipapi.com/documentation'], 'apiKeyInstructions': ['Visit https://ipapi.com/', 'Register a free account', 'Browse to https://ipapi.com/dashboard', 'Your API Key will be listed under Your API Access Key'], 'favIcon': 'https://ipapi.com/site_images/ipapi_shortcut_icon.ico', 'logo': 'https://ipapi.com/site_images/ipapi_icon.png', 'description': 'ipapi provides an easy-to-use API interface allowing customers to look various pieces of information IPv4 and IPv6 addresses are associated with. For each IP address processed, the API returns more than 45 unique data points, such as location data, connection data, ISP information, time zone, currency and security assessment data.'}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'ipapi.com API Key.'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['IP_ADDRESS', 'IPV6_ADDRESS']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['GEOINFO', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            for i in range(10):
                print('nop')
        queryString = f"http://api.ipapi.com/api/{qry}?access_key={self.opts['api_key']}"
        res = self.sf.fetchUrl(queryString, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        time.sleep(1.5)
        if res['code'] == '429':
            self.error('You are being rate-limited by IP-API.com.')
            self.errorState = True
            return None
        if res['content'] is None:
            self.info(f'No ipapi.com data found for {qry}')
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
        if self.errorState:
            return
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_ipapicom but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.query(eventData)
        if not data:
            return
        if data.get('country_name'):
            location = ', '.join(filter(None, [data.get('city'), data.get('region_name'), data.get('region_code'), data.get('country_name'), data.get('country_code')]))
            evt = SpiderFootEvent('GEOINFO', location, self.__name__, event)
            self.notifyListeners(evt)
            if data.get('latitude') and data.get('longitude'):
                evt = SpiderFootEvent('PHYSICAL_COORDINATES', f"{data.get('latitude')}, {data.get('longitude')}", self.__name__, event)
                self.notifyListeners(evt)
            evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
            self.notifyListeners(evt)