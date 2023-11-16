from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_creditcard(SpiderFootPlugin):
    meta = {'name': 'Credit Card Number Extractor', 'summary': 'Identify Credit Card Numbers in any data', 'flags': ['errorprone'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['DARKNET_MENTION_CONTENT', 'LEAKSITE_CONTENT']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['CREDIT_CARD_NUMBER']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        creditCards = SpiderFootHelpers.extractCreditCardsFromText(eventData)
        for creditCard in set(creditCards):
            self.info(f'Found credit card number: {creditCard}')
            evt = SpiderFootEvent('CREDIT_CARD_NUMBER', creditCard, self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)