import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin
regexps = dict({'URL_JAVASCRIPT': list(['text/javascript', '<script ']), 'URL_FORM': list(['<form ', 'method=[PG]', '<input ']), 'URL_PASSWORD': list(['<input.*type=["\']*password']), 'URL_UPLOAD': list(['type=["\']*file']), 'URL_JAVA_APPLET': list(['<applet ']), 'URL_FLASH': list(['\\.swf[ \\\'\\"]'])})

class sfp_pageinfo(SpiderFootPlugin):
    meta = {'name': 'Page Information', 'summary': 'Obtain information about web pages (do they take passwords, do they contain forms, etc.)', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
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
            while True:
                i = 10
        return ['TARGET_WEB_CONTENT']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['URL_STATIC', 'URL_JAVASCRIPT', 'URL_FORM', 'URL_PASSWORD', 'URL_UPLOAD', 'URL_JAVA_APPLET', 'URL_FLASH', 'PROVIDER_JAVASCRIPT']

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        if 'sfp_spider' not in event.module:
            self.debug('Ignoring web content from ' + event.module)
            return
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventSource = event.actualSource
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if not self.getTarget().matches(self.sf.urlFQDN(eventSource)):
            self.debug('Not gathering page info for external site ' + eventSource)
            return
        if '.css?' in eventSource or eventSource.endswith('.css'):
            self.debug('Not attempting to match CSS content.')
            return
        if '.js?' in eventSource or eventSource.endswith('.js'):
            self.debug('Not attempting to match JS content.')
            return
        if eventSource in self.results:
            self.debug('Already checked this page for a page type, skipping.')
            return
        self.results[eventSource] = list()
        for regexpGrp in regexps:
            if regexpGrp in self.results[eventSource]:
                continue
            for regex in regexps[regexpGrp]:
                rx = re.compile(regex, re.IGNORECASE)
                matches = re.findall(rx, eventData)
                if len(matches) > 0 and regexpGrp not in self.results[eventSource]:
                    self.info('Matched ' + regexpGrp + ' in content from ' + eventSource)
                    self.results[eventSource] = self.results[eventSource] + [regexpGrp]
                    evt = SpiderFootEvent(regexpGrp, eventSource, self.__name__, event)
                    self.notifyListeners(evt)
        if len(self.results[eventSource]) == 0:
            self.info('Treating ' + eventSource + ' as URL_STATIC')
            evt = SpiderFootEvent('URL_STATIC', eventSource, self.__name__, event)
            self.notifyListeners(evt)
        pat = re.compile('<script.*src=[\'"]?([^\'">]*)', re.IGNORECASE)
        matches = re.findall(pat, eventData)
        if len(matches) > 0:
            for match in matches:
                if '://' not in match:
                    continue
                if not self.sf.urlFQDN(match):
                    continue
                if self.getTarget().matches(self.sf.urlFQDN(match)):
                    continue
                self.debug(f'Externally hosted JavaScript found at: {match}')
                evt = SpiderFootEvent('PROVIDER_JAVASCRIPT', match, self.__name__, event)
                self.notifyListeners(evt)