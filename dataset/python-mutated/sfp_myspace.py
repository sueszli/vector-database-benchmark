import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_myspace(SpiderFootPlugin):
    meta = {'name': 'MySpace', 'summary': 'Gather username and location from MySpace.com profiles.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Social Media'], 'dataSource': {'website': 'https://myspace.com/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://www.programmableweb.com/api/myspace'], 'favIcon': 'https://x.myspacecdn.com/new/common/images/favicons/favicon.ico', 'logo': 'https://x.myspacecdn.com/new/common/images/favicons/114-Retina-iPhone.png', 'description': "Myspace is a place where people come to connect, discover, and share.\nThrough an open design, compelling editorial features, and analytics-based recommendations, Myspace creates a creative community of people who connect around mutual affinity and inspiration for the purpose of shaping, sharing, and discovering what's next."}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.__dataSource__ = 'MySpace.com'
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['EMAILADDR', 'SOCIAL_MEDIA']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['SOCIAL_MEDIA', 'GEOINFO']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName == 'EMAILADDR':
            email = eventData
            res = self.sf.fetchUrl('https://myspace.com/search/people?q=' + email, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
            if res['content'] is None:
                self.error(f'Could not fetch MySpace content for {email}')
                return
            profiles = re.findall('<a href="/[a-zA-Z0-9_]+">[^<]+</a></h6>', str(res['content']))
            if not profiles:
                self.debug(f'No profiles found for e-mail: {email}')
                return
            profile = profiles[0]
            try:
                matches = re.findall('<a href=\\"\\/([a-zA-Z0-9_]+)\\".*[\\&; :\\"\\#\\*\\(\\"\\\'\\;\\,\\>\\.\\?\\!]+' + email + '[\\&; :\\"\\#\\*\\)\\"\\\'\\;\\,\\<\\.\\?\\!]+', profile, re.IGNORECASE)
            except Exception:
                self.debug('Malformed e-mail address, skipping.')
                return
            if not matches:
                self.debug('No concrete match for that e-mail.')
                return
            name = matches[0]
            e = SpiderFootEvent('SOCIAL_MEDIA', f'MySpace: <SFURL>https://myspace.com/{name}</SFURL>', self.__name__, event)
            self.notifyListeners(e)
        if eventName == 'SOCIAL_MEDIA':
            try:
                network = eventData.split(': ')[0]
                url = eventData.split(': ')[1].replace('<SFURL>', '').replace('</SFURL>', '')
            except Exception as e:
                self.debug(f'Unable to parse SOCIAL_MEDIA: {eventData} ({e})')
                return
            if network != 'MySpace':
                self.debug(f'Skipping social network profile, {url}, as not a MySpace profile')
                return
            res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
            if res['content'] is None:
                return
            data = re.findall('<div class="location_[^"]+" data-display-text="(.+?)"', res['content'])
            if not data:
                return
            location = data[0]
            if len(location) < 5 or len(location) > 100:
                self.debug('Skipping likely invalid location.')
                return
            e = SpiderFootEvent('GEOINFO', location, self.__name__, event)
            self.notifyListeners(e)