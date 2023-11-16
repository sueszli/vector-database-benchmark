import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ethereum(SpiderFootPlugin):
    meta = {'name': 'Ethereum Address Extractor', 'summary': 'Identify ethereum addresses in scraped webpages.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['TARGET_WEB_CONTENT']

    def producedEvents(self):
        if False:
            return 10
        return ['ETHEREUM_ADDRESS']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        sourceData = self.sf.hashstring(eventData)
        if sourceData in self.results:
            return
        self.results[sourceData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        matches = re.findall('[\\s:=\\>](0x[a-fA-F0-9]{40})', eventData)
        for m in matches:
            self.debug('Ethereum address match: ' + m)
            evt = SpiderFootEvent('ETHEREUM_ADDRESS', m, self.__name__, event)
            self.notifyListeners(evt)