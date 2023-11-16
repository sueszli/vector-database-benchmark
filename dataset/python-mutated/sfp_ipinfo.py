import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ipinfo(SpiderFootPlugin):
    meta = {'name': 'IPInfo.io', 'summary': 'Identifies the physical location of IP addresses identified using ipinfo.io.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Real World'], 'dataSource': {'website': 'https://ipinfo.io', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://ipinfo.io/developers'], 'apiKeyInstructions': ['Visit https://ipinfo.io/', 'Sign up for a free account', 'Navigate to https://ipinfo.io/account', "The API key is listed above 'is your access token'"], 'favIcon': 'https://ipinfo.io/static/favicon-96x96.png?v3', 'logo': 'https://ipinfo.io/static/deviceicons/android-icon-96x96.png', 'description': 'The Trusted Source for IP Address Data.\nWith IPinfo, you can pinpoint your users’ locations, customize their experiences, prevent fraud, ensure compliance, and so much more.'}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'Ipinfo.io access token.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['IP_ADDRESS', 'IPV6_ADDRESS']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['GEOINFO']

    def queryIP(self, ip):
        if False:
            for i in range(10):
                print('nop')
        headers = {'Authorization': 'Bearer ' + self.opts['api_key']}
        res = self.sf.fetchUrl('https://ipinfo.io/' + ip + '/json', timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], headers=headers)
        if res['code'] == '429':
            self.error('You are being rate-limited by ipinfo.io.')
            self.errorState = True
            return None
        if res['content'] is None:
            self.info('No GeoIP info found for ' + ip)
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
        return None

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_ipinfo but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.queryIP(eventData)
        if data is None:
            return
        if 'country' not in data:
            return
        location = ', '.join([_f for _f in [data.get('city'), data.get('region'), data.get('country')] if _f])
        self.info('Found GeoIP for ' + eventData + ': ' + location)
        evt = SpiderFootEvent('GEOINFO', location, self.__name__, event)
        self.notifyListeners(evt)