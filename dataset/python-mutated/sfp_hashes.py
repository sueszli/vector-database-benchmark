from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_hashes(SpiderFootPlugin):
    meta = {'name': 'Hash Extractor', 'summary': 'Identify MD5 and SHA hashes in web content, files and more.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['TARGET_WEB_CONTENT', 'BASE64_DATA', 'LEAKSITE_CONTENT', 'RAW_DNS_RECORDS', 'RAW_FILE_META_DATA']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['HASH']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        hashes = SpiderFootHelpers.extractHashesFromText(eventData)
        for hashtup in hashes:
            (hashalgo, hashval) = hashtup
            evt = SpiderFootEvent('HASH', f'[{hashalgo}] {hashval}', self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)