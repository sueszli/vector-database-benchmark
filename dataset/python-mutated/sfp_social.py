import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin
regexps = dict({'LinkedIn (Individual)': list(['.*linkedin.com/in/([a-zA-Z0-9_]+$)']), 'LinkedIn (Company)': list(['.*linkedin.com/company/([a-zA-Z0-9_]+$)']), 'Github': list(['.*github.com/([a-zA-Z0-9_]+)\\/']), 'Google+': list(['.*plus.google.com/([0-9]+$)']), 'Bitbucket': list(['.*bitbucket.org/([a-zA-Z0-9_]+)\\/']), 'Gitlab': list(['.*gitlab.com/([a-zA-Z0-9_]+)\\/']), 'Facebook': list(['.*facebook.com/([a-zA-Z0-9_]+$)']), 'MySpace': list(['https?://myspace.com/([a-zA-Z0-9_\\.]+$)']), 'YouTube': list(['.*youtube.com/([a-zA-Z0-9_]+$)']), 'Twitter': list(['.*twitter.com/([a-zA-Z0-9_]{1,15}$)', '.*twitter.com/#!/([a-zA-Z0-9_]{1,15}$)']), 'SlideShare': list(['.*slideshare.net/([a-zA-Z0-9_]+$)']), 'Instagram': list(['.*instagram.com/([a-zA-Z0-9_\\.]+)/?$'])})

class sfp_social(SpiderFootPlugin):
    meta = {'name': 'Social Network Identifier', 'summary': 'Identify presence on social media networks such as LinkedIn, Twitter and others.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Social Media']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['LINKED_URL_EXTERNAL']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['SOCIAL_MEDIA', 'USERNAME']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in list(self.results.keys()):
            return
        self.results[eventData] = True
        for regexpGrp in list(regexps.keys()):
            for regex in regexps[regexpGrp]:
                bits = re.match(regex, eventData, re.IGNORECASE)
                if not bits:
                    continue
                self.info(f'Matched {regexpGrp} in {eventData}')
                evt = SpiderFootEvent('SOCIAL_MEDIA', f'{regexpGrp}: <SFURL>{eventData}</SFURL>', self.__name__, event)
                self.notifyListeners(evt)
                if regexpGrp != 'Google+':
                    un = bits.group(1)
                    evt = SpiderFootEvent('USERNAME', str(un), self.__name__, event)
                    self.notifyListeners(evt)