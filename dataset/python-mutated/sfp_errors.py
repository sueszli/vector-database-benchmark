import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin
regexps = dict({'PHP Error': ['PHP pase error', 'PHP warning', 'PHP error', 'unexpected T_VARIABLE', 'warning: failed opening', 'include_path='], 'Generic Error': ['Internal Server Error', 'Incorrect syntax'], 'Oracle Error': ['ORA-\\d+', 'TNS:.?no listen'], 'ASP Error': ['NET_SessionId'], 'MySQL Error': ['mysql_query\\(', 'mysql_connect\\('], 'ODBC Error': ['\\[ODBC SQL']})

class sfp_errors(SpiderFootPlugin):
    meta = {'name': 'Error String Extractor', 'summary': 'Identify common error messages in content like SQL errors, etc.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Content Analysis']}
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
            while True:
                i = 10
        return ['TARGET_WEB_CONTENT']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['ERROR_MESSAGE']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if srcModuleName != 'sfp_spider':
            return
        eventSource = event.actualSource
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventSource not in list(self.results.keys()):
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
                    evt = SpiderFootEvent('ERROR_MESSAGE', regexpGrp, self.__name__, event)
                    self.notifyListeners(evt)