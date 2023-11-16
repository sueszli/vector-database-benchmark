import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin
regexps = dict({'jQuery': list(['jquery']), 'YUI': list(['\\/yui\\/', 'yui\\-', 'yui\\.']), 'Prototype': list(['\\/prototype\\/', 'prototype\\-', 'prototype\\.js']), 'ZURB Foundation': list(['\\/foundation\\/', 'foundation\\-', 'foundation\\.js']), 'Bootstrap': list(['\\/bootstrap\\/', 'bootstrap\\-', 'bootstrap\\.js']), 'ExtJS': list(['[\\\'\\"\\=]ext\\.js', 'extjs', '\\/ext\\/*\\.js']), 'Mootools': list(['\\/mootools\\/', 'mootools\\-', 'mootools\\.js']), 'Dojo': list(['\\/dojo\\/', '[\\\'\\"\\=]dojo\\-', '[\\\'\\"\\=]dojo\\.js']), 'Wordpress': list(['\\/wp-includes\\/', '\\/wp-content\\/'])})

class sfp_webframework(SpiderFootPlugin):
    meta = {'name': 'Web Framework Identifier', 'summary': 'Identify the usage of popular web frameworks like jQuery, YUI and others.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Content Analysis']}
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
            return 10
        return ['TARGET_WEB_CONTENT']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['URL_WEB_FRAMEWORK']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventSource = event.actualSource
        if srcModuleName != 'sfp_spider':
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventSource not in self.results:
            self.results[eventSource] = list()
        if not self.getTarget().matches(self.sf.urlFQDN(eventSource)):
            self.debug('Not collecting web content information for external sites.')
            return
        for regexpGrp in list(regexps.keys()):
            if regexpGrp in self.results[eventSource]:
                continue
            for regex in regexps[regexpGrp]:
                pat = re.compile(regex, re.IGNORECASE)
                matches = re.findall(pat, eventData)
                if len(matches) > 0 and regexpGrp not in self.results[eventSource]:
                    self.info('Matched ' + regexpGrp + ' in content from ' + eventSource)
                    self.results[eventSource] = self.results[eventSource] + [regexpGrp]
                    evt = SpiderFootEvent('URL_WEB_FRAMEWORK', regexpGrp, self.__name__, event)
                    self.notifyListeners(evt)