import re
import urllib.error
import urllib.parse
import urllib.request
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ahmia(SpiderFootPlugin):
    meta = {'name': 'Ahmia', 'flags': ['tor'], 'summary': "Search Tor 'Ahmia' search engine for mentions of the target.", 'useCases': ['Footprint', 'Investigate'], 'categories': ['Search Engines'], 'dataSource': {'website': 'https://ahmia.fi/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://ahmia.fi/documentation/', 'https://github.com/ahmia/', 'http://msydqstlz2kzerdg.onion/', 'https://ahmia.fi/stats'], 'favIcon': 'https://ahmia.fi/static/images/favicon.ico', 'logo': 'https://ahmia.fi/static/images/ahmiafi_black.png', 'description': 'Ahmia searches hidden services on the Tor network. To access these hidden services,you need the Tor browser bundle. Abuse material is not allowed on Ahmia. See our service blacklist and report abuse material if you find it in the index. It will be removed as soon as possible.\nContributors to Ahmia believe that the Tor network is an important and resilient distributed platform for anonymity and privacy worldwide. By providing a search engine for what many call the "deep web" or "dark net", Ahmia makes hidden services accessible to a wide range of people, not just Tor network early adopters.'}}
    opts = {'fetchlinks': True, 'fullnames': True}
    optdescs = {'fetchlinks': 'Fetch the darknet pages (via TOR, if enabled) to verify they mention your target.', 'fullnames': 'Search for human names?'}
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
            return 10
        return ['DOMAIN_NAME', 'HUMAN_NAME', 'EMAILADDR']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['DARKNET_MENTION_URL', 'DARKNET_MENTION_CONTENT']

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if not self.opts['fullnames'] and eventName == 'HUMAN_NAME':
            self.debug(f'Skipping HUMAN_NAME: {eventData}')
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        params = urllib.parse.urlencode({'q': eventData})
        data = self.sf.fetchUrl(f'https://ahmia.fi/search/?{params}', useragent=self.opts['_useragent'], timeout=15)
        if not data:
            self.info(f'No results for {eventData} returned from Ahmia.fi.')
            return
        content = data.get('content')
        if not content:
            self.info(f'No results for {eventData} returned from Ahmia.fi.')
            return
        links = re.findall('redirect_url=(.[^"]+)"', content, re.IGNORECASE | re.DOTALL)
        if not links:
            self.info(f'No results for {eventData} returned from Ahmia.fi.')
            return
        for link in links:
            if self.checkForStop():
                return
            if link in self.results:
                continue
            self.results[link] = True
            self.debug(f'Found a darknet mention: {link}')
            if not self.sf.urlFQDN(link).endswith('.onion'):
                continue
            if not self.opts['fetchlinks']:
                evt = SpiderFootEvent('DARKNET_MENTION_URL', link, self.__name__, event)
                self.notifyListeners(evt)
                continue
            res = self.sf.fetchUrl(link, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], verify=False)
            if res['content'] is None:
                self.debug(f'Ignoring {link} as no data returned')
                continue
            if eventData not in res['content']:
                self.debug(f'Ignoring {link} as no mention of {eventData}')
                continue
            evt = SpiderFootEvent('DARKNET_MENTION_URL', link, self.__name__, event)
            self.notifyListeners(evt)
            try:
                startIndex = res['content'].index(eventData) - 120
                endIndex = startIndex + len(eventData) + 240
            except Exception:
                self.debug(f"String '{eventData}' not found in content.")
                continue
            wdata = res['content'][startIndex:endIndex]
            evt = SpiderFootEvent('DARKNET_MENTION_CONTENT', f'...{wdata}...', self.__name__, evt)
            self.notifyListeners(evt)