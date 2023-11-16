import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_openstreetmap(SpiderFootPlugin):
    meta = {'name': 'OpenStreetMap', 'summary': 'Retrieves latitude/longitude coordinates for physical addresses from OpenStreetMap API.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Real World'], 'dataSource': {'website': 'https://www.openstreetmap.org/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://wiki.openstreetmap.org/wiki/API', 'https://wiki.openstreetmap.org/wiki/API_v0.6'], 'favIcon': 'https://www.openstreetmap.org/assets/osm_logo-b7061f13a03615f787a7e0e56a0db5252eb2a217ab063183e78526a8cc10989b.svg', 'logo': 'https://www.openstreetmap.org/assets/osm_logo-b7061f13a03615f787a7e0e56a0db5252eb2a217ab063183e78526a8cc10989b.svg', 'description': 'OpenStreetMap powers map data on thousands of web sites, mobile apps, and hardware devices.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['PHYSICAL_ADDRESS']

    def producedEvents(self):
        if False:
            return 10
        return ['PHYSICAL_COORDINATES']

    def query(self, qry):
        if False:
            while True:
                i = 10
        params = {'q': qry.encode('raw_unicode_escape').decode('ascii', errors='replace'), 'format': 'json', 'polygon': '0', 'addressdetails': '0'}
        res = self.sf.fetchUrl('https://nominatim.openstreetmap.org/search?' + urllib.parse.urlencode(params), timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['content'] is None:
            self.info('No location info found for ' + qry)
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
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        address = eventData
        if address.lower().startswith('po box'):
            self.debug('Skipping PO BOX address')
            return
        rx1 = re.compile('^(c/o|care of|attn:|attention:)\\s+[0-9a-z\\s\\.]', flags=re.IGNORECASE)
        address = re.sub(rx1, '', address)
        rx2 = re.compile('^(Level|Floor|Suite|Room)\\s+[0-9a-z]+,', flags=re.IGNORECASE)
        address = re.sub(rx2, '', address)
        data = self.query(eventData)
        time.sleep(1)
        if data is None:
            self.debug('Found no results for ' + eventData)
            return
        self.info('Found ' + str(len(data)) + ' matches for ' + eventData)
        for location in data:
            try:
                lat = location.get('lat')
                lon = location.get('lon')
            except Exception as e:
                self.debug('Failed to get lat/lon: ' + str(e))
                continue
            if not lat or not lon:
                continue
            coords = str(lat) + ',' + str(lon)
            self.debug('Found coordinates: ' + coords)
            evt = SpiderFootEvent('PHYSICAL_COORDINATES', coords, self.__name__, event)
            self.notifyListeners(evt)