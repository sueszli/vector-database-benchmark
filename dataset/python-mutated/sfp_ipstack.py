import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ipstack(SpiderFootPlugin):
    meta = {'name': 'ipstack', 'summary': 'Identifies the physical location of IP addresses identified using ipstack.com.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Real World'], 'dataSource': {'website': 'https://ipstack.com/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://ipstack.com/documentation', 'https://ipstack.com/faq'], 'apiKeyInstructions': ['Visit https://ipstack.com/product', "Click on 'Get Free API Key'", "Click on 'Dashboard'", "The API key is listed under 'Your API Access Key'"], 'favIcon': 'https://ipstack.com/ipstack_images/ipstack_logo.svg', 'logo': 'https://ipstack.com/ipstack_images/ipstack_logo.svg', 'description': 'Locate and identify website visitors by IP address.\nipstack offers one of the leading IP to geolocation APIS and global IP database services worldwide.'}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'Ipstack.com API key.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['IP_ADDRESS']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['GEOINFO']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.errorState:
            return
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_ipstack but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        res = self.sf.fetchUrl('http://api.ipstack.com/' + eventData + '?access_key=' + self.opts['api_key'], timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if res['content'] is None:
            self.info('No GeoIP info found for ' + eventData)
        try:
            hostip = json.loads(res['content'])
            if 'success' in hostip and hostip['success'] is False:
                self.error('Invalid ipstack.com API key or usage limits exceeded.')
                self.errorState = True
                return
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return
        geoinfo = hostip.get('country_name')
        if geoinfo:
            self.info(f'Found GeoIP for {eventData}: {geoinfo}')
            evt = SpiderFootEvent('GEOINFO', geoinfo, self.__name__, event)
            self.notifyListeners(evt)