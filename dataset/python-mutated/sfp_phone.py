import phonenumbers
from phonenumbers import carrier
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_phone(SpiderFootPlugin):
    meta = {'name': 'Phone Number Extractor', 'summary': 'Identify phone numbers in scraped webpages.', 'flags': [], 'useCases': ['Passive', 'Footprint', 'Investigate'], 'categories': ['Content Analysis']}
    opts = {}
    results = None
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['TARGET_WEB_CONTENT', 'DOMAIN_WHOIS', 'NETBLOCK_WHOIS', 'PHONE_NUMBER']

    def producedEvents(self):
        if False:
            return 10
        return ['PHONE_NUMBER', 'PROVIDER_TELCO']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        sourceData = self.sf.hashstring(eventData)
        if sourceData in self.results:
            return
        self.results[sourceData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName in ['TARGET_WEB_CONTENT', 'DOMAIN_WHOIS', 'NETBLOCK_WHOIS']:
            content = eventData.replace('.', '-')
            for match in phonenumbers.PhoneNumberMatcher(content, region=None):
                n = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
                evt = SpiderFootEvent('PHONE_NUMBER', n, self.__name__, event)
                if event.moduleDataSource:
                    evt.moduleDataSource = event.moduleDataSource
                else:
                    evt.moduleDataSource = 'Unknown'
                self.notifyListeners(evt)
        if eventName == 'PHONE_NUMBER':
            try:
                number = phonenumbers.parse(eventData)
            except Exception as e:
                self.debug(f'Error parsing phone number: {e}')
                return
            try:
                number_carrier = carrier.name_for_number(number, 'en')
            except Exception as e:
                self.debug(f'Error retrieving phone number carrier: {e}')
                return
            if not number_carrier:
                self.debug(f'No carrier information found for {eventData}')
                return
            evt = SpiderFootEvent('PROVIDER_TELCO', number_carrier, self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)