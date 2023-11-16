import re
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_skymem(SpiderFootPlugin):
    meta = {'name': 'Skymem', 'summary': 'Look up e-mail addresses on Skymem.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Search Engines'], 'dataSource': {'website': 'http://www.skymem.info/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['http://www.skymem.info/faq'], 'favIcon': 'https://www.google.com/s2/favicons?domain=http://www.skymem.info/', 'logo': '', 'description': 'Find email addresses of companies and people.'}}
    results = None
    opts = {}
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['INTERNET_NAME', 'DOMAIN_NAME']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['EMAILADDR', 'EMAILADDR_GENERIC']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        res = self.sf.fetchUrl('http://www.skymem.info/srch?q=' + eventData, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if res['content'] is None:
            return
        emails = SpiderFootHelpers.extractEmailsFromText(res['content'])
        for email in emails:
            mailDom = email.lower().split('@')[1]
            if not self.getTarget().matches(mailDom):
                self.debug('Skipped address: ' + email)
                continue
            self.info('Found e-mail address: ' + email)
            if email not in self.results:
                if email.split('@')[0] in self.opts['_genericusers'].split(','):
                    evttype = 'EMAILADDR_GENERIC'
                else:
                    evttype = 'EMAILADDR'
                evt = SpiderFootEvent(evttype, email, self.__name__, event)
                self.notifyListeners(evt)
                self.results[email] = True
        domain_ids = re.findall('<a href="/domain/([a-z0-9]+)\\?p=', str(res['content']))
        if not domain_ids:
            return
        domain_id = domain_ids[0]
        for page in range(1, 21):
            res = self.sf.fetchUrl(f'http://www.skymem.info/domain/{domain_id}?p={page}', timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
            if res['content'] is None:
                break
            emails = SpiderFootHelpers.extractEmailsFromText(res['content'])
            for email in emails:
                mailDom = email.lower().split('@')[1]
                if not self.getTarget().matches(mailDom):
                    self.debug('Skipped address: ' + email)
                    continue
                self.info('Found e-mail address: ' + email)
                if email not in self.results:
                    if email.split('@')[0] in self.opts['_genericusers'].split(','):
                        evttype = 'EMAILADDR_GENERIC'
                    else:
                        evttype = 'EMAILADDR'
                    evt = SpiderFootEvent(evttype, email, self.__name__, event)
                    self.notifyListeners(evt)
                    self.results[email] = True
            max_page = 0
            pages = re.findall('/domain/' + domain_id + '\\?p=(\\d+)', str(res['content']))
            for p in pages:
                if int(p) >= max_page:
                    max_page = int(p)
            if page >= max_page:
                break