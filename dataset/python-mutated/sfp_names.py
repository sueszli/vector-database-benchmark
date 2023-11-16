import re
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_names(SpiderFootPlugin):
    meta = {'name': 'Human Name Extractor', 'summary': 'Attempt to identify human names in fetched content.', 'flags': ['errorprone'], 'useCases': ['Footprint', 'Passive'], 'categories': ['Content Analysis']}
    opts = {'algolimit': 75, 'emailtoname': True, 'filterjscss': True}
    optdescs = {'algolimit': "A value between 0-100 to tune the sensitivity of the name finder. Less than 40 will give you a lot of junk, over 50 and you'll probably miss things but will have less false positives.", 'emailtoname': 'Convert e-mail addresses in the form of firstname.surname@target to names?', 'filterjscss': 'Filter out names that originated from CSS/JS content. Enabling this avoids detection of popular Javascript and web framework author names.'}
    results = None
    d = None
    n = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.d = SpiderFootHelpers.dictionaryWordsFromWordlists()
        self.n = SpiderFootHelpers.humanNamesFromWordlists()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['TARGET_WEB_CONTENT', 'EMAILADDR', 'DOMAIN_WHOIS', 'NETBLOCK_WHOIS', 'RAW_RIR_DATA', 'RAW_FILE_META_DATA']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['HUMAN_NAME']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName == 'TARGET_WEB_CONTENT':
            url = event.actualSource
            if url is not None:
                if self.opts['filterjscss'] and ('.js' in url or '.css' in url):
                    self.debug('Ignoring web content from CSS/JS.')
                    return
        if eventName == 'EMAILADDR' and self.opts['emailtoname']:
            potential_name = eventData.split('@')[0]
            if '.' not in potential_name:
                return
            name = ' '.join(map(str.capitalize, potential_name.split('.')))
            if re.search('[0-9]', name):
                return
            evt = SpiderFootEvent('HUMAN_NAME', name, self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)
            return
        if eventName == 'RAW_RIR_DATA':
            if srcModuleName not in ['sfp_builtwith', 'sfp_clearbit', 'sfp_emailcrawlr', 'sfp_fullcontact', 'sfp_github', 'sfp_hunter', 'sfp_opencorporates', 'sfp_slideshare', 'sfp_jsonwhoiscom', 'sfp_twitter', 'sfp_gravatar', 'sfp_keybase']:
                self.debug('Ignoring RAW_RIR_DATA from untrusted module.')
                return
        rx = re.compile("([A-Z][a-z�������������]+)\\s+.?.?\\s?([A-Z][�������������a-zA-Z\\'\\-]+)")
        m = re.findall(rx, eventData)
        for r in m:
            p = 0
            notindict = False
            first = r[0].lower()
            if first[len(first) - 2] == "'" or first[len(first) - 1] == "'":
                continue
            secondOrig = r[1].replace("'s", '')
            secondOrig = secondOrig.rstrip("'")
            second = r[1].lower().replace("'s", '')
            second = second.rstrip("'")
            if first not in self.d and second not in self.d:
                self.debug(f'Both first and second names are not in the dictionary, so high chance of name: ({first}:{second}).')
                p += 75
                notindict = True
            else:
                self.debug(first + ' was found or ' + second + ' was found in dictionary.')
            if first in self.n:
                p += 50
            if len(first) == 2 or len(second) == 2:
                p -= 50
            if not notindict:
                if first in self.d and second not in self.d:
                    p -= 20
                if first not in self.d and second in self.d:
                    p -= 40
            name = r[0] + ' ' + secondOrig
            self.debug('Name of ' + name + ' has score: ' + str(p))
            if p >= self.opts['algolimit']:
                evt = SpiderFootEvent('HUMAN_NAME', name, self.__name__, event)
                if event.moduleDataSource:
                    evt.moduleDataSource = event.moduleDataSource
                else:
                    evt.moduleDataSource = 'Unknown'
                self.notifyListeners(evt)