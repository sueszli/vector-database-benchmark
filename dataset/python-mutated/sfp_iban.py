from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_iban(SpiderFootPlugin):
    meta = {'name': 'IBAN Number Extractor', 'summary': 'Identify International Bank Account Numbers (IBANs) in any data.', 'flags': ['errorprone'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['TARGET_WEB_CONTENT', 'DARKNET_MENTION_CONTENT', 'LEAKSITE_CONTENT']

    def producedEvents(self):
        if False:
            return 10
        return ['IBAN_NUMBER']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        ibans = SpiderFootHelpers.extractIbansFromText(eventData)
        for ibanNumber in set(ibans):
            self.info(f'Found IBAN number: {ibanNumber}')
            evt = SpiderFootEvent('IBAN_NUMBER', ibanNumber, self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)