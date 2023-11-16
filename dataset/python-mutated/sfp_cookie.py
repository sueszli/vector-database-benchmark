import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_cookie(SpiderFootPlugin):
    meta = {'name': 'Cookie Extractor', 'summary': 'Extract Cookies from HTTP headers.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['WEBSERVER_HTTPHEADERS']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['TARGET_WEB_COOKIE']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventSource = event.actualSource
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventSource in self.results:
            return
        self.results[eventSource] = True
        fqdn = self.sf.urlFQDN(eventSource)
        if not self.getTarget().matches(fqdn):
            self.debug(f'Not collecting cookies from external sites. Ignoring HTTP headers from {fqdn}')
            return
        try:
            data = json.loads(eventData)
        except Exception:
            self.error('Received HTTP headers from another module in an unexpected format.')
            return
        cookie = data.get('cookie')
        if cookie:
            evt = SpiderFootEvent('TARGET_WEB_COOKIE', cookie, self.__name__, event)
            self.notifyListeners(evt)