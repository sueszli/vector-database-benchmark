from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_pgp(SpiderFootPlugin):
    meta = {'name': 'PGP Key Servers', 'summary': 'Look up domains and e-mail addresses in PGP public key servers.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries']}
    results = None
    errorState = False
    opts = {'retrieve_keys': True, 'keyserver_search1': 'https://keyserver.ubuntu.com/pks/lookup?fingerprint=on&op=vindex&search=', 'keyserver_fetch1': 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=', 'keyserver_search2': 'http://the.earth.li:11371/pks/lookup?fingerprint=on&op=vindex&search=', 'keyserver_fetch2': 'http://the.earth.li:11371/pks/lookup?op=get&search='}
    optdescs = {'retrieve_keys': 'Retrieve PGP keys.', 'keyserver_search1': 'PGP public key server URL to find e-mail addresses on a domain. Domain will get appended.', 'keyserver_fetch1': 'PGP public key server URL to find the public key for an e-mail address. Email address will get appended.', 'keyserver_search2': 'Backup PGP public key server URL to find e-mail addresses on a domain. Domain will get appended.', 'keyserver_fetch2': 'Backup PGP public key server URL to find the public key for an e-mail address. Email address will get appended.'}

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['INTERNET_NAME', 'EMAILADDR', 'DOMAIN_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['EMAILADDR', 'EMAILADDR_GENERIC', 'AFFILIATE_EMAILADDR', 'PGP_KEY']

    def queryDomain(self, keyserver_search_url, qry):
        if False:
            while True:
                i = 10
        res = self.sf.fetchUrl(keyserver_search_url + qry, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if not res:
            return None
        if res['content'] is None:
            return None
        if res['code'] == '503':
            return None
        return res

    def queryEmail(self, keyserver_fetch_url, qry):
        if False:
            print('Hello World!')
        res = self.sf.fetchUrl(keyserver_fetch_url + qry, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if not res:
            return None
        if res['content'] is None:
            return None
        if res['code'] == '503':
            return None
        return res

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        eventData = event.data
        if self.errorState:
            return
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {event.module}')
        if not self.opts['keyserver_search1'] and (not self.opts['keyserver_search2']):
            self.error(f'You enabled {self.__class__.__name__} but did not set key server URLs')
            self.errorState = True
            return
        if eventName in ['DOMAIN_NAME', 'INTERNET_NAME']:
            res = self.queryDomain(self.opts['keyserver_search1'], eventData)
            if not res:
                res = self.queryDomain(self.opts['keyserver_search2'], eventData)
            if not res:
                return
            emails = SpiderFootHelpers.extractEmailsFromText(res['content'])
            self.info(f'Found {len(emails)} email addresses')
            for email in emails:
                if email.split('@')[0] in self.opts['_genericusers'].split(','):
                    evttype = 'EMAILADDR_GENERIC'
                else:
                    evttype = 'EMAILADDR'
                mailDom = email.lower().split('@')[1]
                if not self.getTarget().matches(mailDom):
                    evttype = 'AFFILIATE_EMAILADDR'
                self.debug(f'Found e-mail address: {email}')
                evt = SpiderFootEvent(evttype, email, self.__name__, event)
                self.notifyListeners(evt)
        if eventName == 'EMAILADDR' and self.opts['retrieve_keys']:
            res = self.queryEmail(self.opts['keyserver_fetch1'], eventData)
            if not res:
                res = self.queryEmail(self.opts['keyserver_fetch2'], eventData)
            if not res:
                return
            keys = SpiderFootHelpers.extractPgpKeysFromText(res['content'])
            self.info(f'Found {len(keys)} public PGP keys')
            for key in keys:
                self.debug(f'Found public key: {key}')
                evt = SpiderFootEvent('PGP_KEY', key, self.__name__, event)
                self.notifyListeners(evt)