from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_intfiles(SpiderFootPlugin):
    meta = {'name': 'Interesting File Finder', 'summary': 'Identifies potential files of interest, e.g. office documents, zip files.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Crawling and Scanning']}
    opts = {'fileexts': ['doc', 'docx', 'ppt', 'pptx', 'pdf', 'xls', 'xlsx', 'zip']}
    optdescs = {'fileexts': 'File extensions of files you consider interesting.'}
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
            while True:
                i = 10
        return ['LINKED_URL_INTERNAL']

    def producedEvents(self):
        if False:
            return 10
        return ['INTERESTING_FILE']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            return
        self.results[eventData] = True
        for fileExt in self.opts['fileexts']:
            if '.' + fileExt.lower() in eventData.lower():
                evt = SpiderFootEvent('INTERESTING_FILE', eventData, self.__name__, event)
                self.notifyListeners(evt)