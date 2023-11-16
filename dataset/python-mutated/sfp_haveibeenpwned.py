import json
import re
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_haveibeenpwned(SpiderFootPlugin):
    meta = {'name': 'HaveIBeenPwned', 'summary': 'Check HaveIBeenPwned.com for hacked e-mail addresses identified in breaches.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Leaks, Dumps and Breaches'], 'dataSource': {'website': 'https://haveibeenpwned.com/', 'model': 'COMMERCIAL_ONLY', 'references': ['https://haveibeenpwned.com/API/v3', 'https://haveibeenpwned.com/FAQs'], 'apiKeyInstructions': ['Visit https://haveibeenpwned.com/API/Key', 'Register an account', 'Visit https://haveibeenpwned.com/API/Key'], 'favIcon': 'https://haveibeenpwned.com/favicon.ico', 'logo': 'https://haveibeenpwned.com/favicon.ico', 'description': 'Check if you have an account that has been compromised in a data breach.'}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'HaveIBeenPwned.com API key.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['EMAILADDR', 'PHONE_NUMBER']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['EMAILADDR_COMPROMISED', 'PHONE_NUMBER_COMPROMISED', 'LEAKSITE_CONTENT', 'LEAKSITE_URL']

    def query(self, qry):
        if False:
            for i in range(10):
                print('nop')
        if self.opts['api_key']:
            version = '3'
        else:
            version = '2'
        url = f'https://haveibeenpwned.com/api/v{version}/breachedaccount/{qry}'
        hdrs = {'Accept': f'application/vnd.haveibeenpwned.v{version}+json'}
        retry = 0
        if self.opts['api_key']:
            hdrs['hibp-api-key'] = self.opts['api_key']
        while retry < 2:
            time.sleep(1.5)
            res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot', headers=hdrs)
            if res['code'] == '200':
                break
            if res['code'] == '404':
                return None
            if res['code'] == '429':
                time.sleep(2)
            retry += 1
            if res['code'] == '401':
                self.error('Failed to authenticate key with HaveIBeenPwned.com.')
                self.errorState = True
                return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f'Error processing JSON response from HaveIBeenPwned?: {e}')
        return None

    def queryPaste(self, qry):
        if False:
            for i in range(10):
                print('nop')
        url = f'https://haveibeenpwned.com/api/v3/pasteaccount/{qry}'
        headers = {'Accept': 'application/json', 'hibp-api-key': self.opts['api_key']}
        retry = 0
        while retry < 2:
            time.sleep(1.5)
            res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot', headers=headers)
            if res['code'] == '200':
                break
            if res['code'] == '404':
                return None
            if res['code'] == '429':
                time.sleep(2)
            retry += 1
            if res['code'] == '401':
                self.error('Failed to authenticate key with HaveIBeenPwned.com.')
                self.errorState = True
                return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f'Error processing JSON response from HaveIBeenPwned?: {e}')
        return None

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_haveibeenpwned but did not set an API key!')
            self.errorState = True
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.query(eventData)
        if data is not None:
            for n in data:
                try:
                    site = n['Name']
                except Exception as e:
                    self.debug(f'Unable to parse result from HaveIBeenPwned?: {e}')
                    continue
                if eventName == 'EMAILADDR':
                    e = SpiderFootEvent('EMAILADDR_COMPROMISED', eventData + ' [' + site + ']', self.__name__, event)
                else:
                    e = SpiderFootEvent('PHONE_NUMBER_COMPROMISED', eventData + ' [' + site + ']', self.__name__, event)
                self.notifyListeners(e)
        if eventName == 'PHONE_NUMBER':
            return
        pasteData = self.queryPaste(eventData)
        if pasteData is None:
            return
        sites = {'Pastebin': 'https://pastebin.com/', 'Pastie': 'http://pastie.org/p/', 'Slexy': 'https://slexy.org/view/', 'Ghostbin': 'https://ghostbin.com/paste/', 'JustPaste': 'https://justpaste.it/'}
        links = set()
        for n in pasteData:
            try:
                source = n.get('Source')
                site = source
                if source in sites:
                    site = f"{sites[n.get('Source')]}{n.get('Id')}"
                    links.add(site)
            except Exception as e:
                self.debug(f'Unable to parse result from HaveIBeenPwned?: {e}')
                continue
        for link in links:
            try:
                self.debug('Found a link: ' + link)
                if self.checkForStop():
                    return
                res = self.sf.fetchUrl(link, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
                if res['content'] is None:
                    self.debug(f'Ignoring {link} as no data returned')
                    continue
                if re.search('[^a-zA-Z\\-\\_0-9]' + re.escape(eventData) + '[^a-zA-Z\\-\\_0-9]', res['content'], re.IGNORECASE) is None:
                    continue
                evt1 = SpiderFootEvent('LEAKSITE_URL', link, self.__name__, event)
                self.notifyListeners(evt1)
                evt2 = SpiderFootEvent('LEAKSITE_CONTENT', res['content'], self.__name__, evt1)
                self.notifyListeners(evt2)
            except Exception as e:
                self.debug(f'Unable to parse result from HaveIBeenPwned?: {e}')
                continue