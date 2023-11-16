import json
import urllib.parse
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ipregistry(SpiderFootPlugin):
    meta = {'name': 'ipregistry', 'summary': 'Query the ipregistry.co database for reputation and geo-location.', 'flags': ['apikey'], 'useCases': ['Passive', 'Footprint', 'Investigate'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://ipregistry.co/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://ipregistry.co/docs'], 'apiKeyInstructions': ['Visit https://dashboard.ipregistry.co/signup', 'Register a free account', "Click on 'API Keys' in left navbar", "Click on 'Click to reveal API key' for existing Default key"], 'favIcon': 'https://cdn.ipregistry.co/icons/favicon-32x32.png', 'logo': 'https://ipregistry.co/assets/ipregistry.svg', 'description': 'Ipregistry is a trusted and in-depth IP Geolocation and Threat detections source of information that canbenefit publishers, ad networks, retailers, financial services, e-commerce stores and more.'}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'Ipregistry API Key.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=None):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        if userOpts:
            self.opts.update(userOpts)

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['IP_ADDRESS', 'IPV6_ADDRESS']

    def producedEvents(self):
        if False:
            return 10
        return ['GEOINFO', 'MALICIOUS_IPADDR', 'PHYSICAL_COORDINATES', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            return 10
        qs = urllib.parse.urlencode({'key': self.opts['api_key']})
        res = self.sf.fetchUrl(f'https://api.ipregistry.co/{qry}?{qs}', timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['content'] is None:
            self.info(f"No {self.meta['name']} info found for {qry}")
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f"Error processing JSON response from {self.meta['name']}: {e}")
        return None

    def emit(self, etype, data, pevent):
        if False:
            return 10
        evt = SpiderFootEvent(etype, data, self.__name__, pevent)
        self.notifyListeners(evt)
        return evt

    def generate_location_events(self, location, pevent):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(location, dict):
            return
        physical_location = None
        country = location.get('country')
        if isinstance(country, dict):
            country_name = country.get('name')
        else:
            country_name = None
        region = location.get('region')
        if isinstance(region, dict):
            region_name = region.get('name')
        else:
            region_name = None
        latitude = location.get('latitude')
        longitude = location.get('longitude')
        if latitude and longitude:
            physical_location = f'{latitude}, {longitude}'
        geo_info = ', '.join([_f for _f in [location.get('city'), region_name, location.get('postal'), country_name] if _f])
        if geo_info:
            self.emit('GEOINFO', geo_info, pevent)
        if physical_location:
            self.emit('PHYSICAL_COORDINATES', physical_location, pevent)

    def generate_security_events(self, security, pevent):
        if False:
            while True:
                i = 10
        if not isinstance(security, dict):
            return
        malicious = any((security.get(k) for k in ('is_abuser', 'is_attacker', 'is_threat')))
        if malicious:
            self.emit('MALICIOUS_IPADDR', f'ipregistry [{pevent.data}]', pevent)

    def generate_events(self, data, pevent):
        if False:
            print('Hello World!')
        if not isinstance(data, dict):
            return
        self.generate_location_events(data.get('location'), pevent)
        self.generate_security_events(data.get('security'), pevent)

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self.errorState:
            return
        self.debug(f'Received event, {event.eventType}, from {event.module}')
        if self.opts['api_key'] == '':
            self.error(f'You enabled {self.__class__.__name__} but did not set an API key!')
            self.errorState = True
            return
        if event.data in self.results:
            self.debug(f'Skipping {event.data}, already checked.')
            return
        self.results[event.data] = True
        if event.eventType in ('IP_ADDRESS', 'IPV6_ADDRESS'):
            data = self.query(event.data)
            self.generate_events(data, event)
            self.emit('RAW_RIR_DATA', json.dumps(data), event)