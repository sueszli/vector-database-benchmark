import re
from bs4 import BeautifulSoup
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnsdumpster(SpiderFootPlugin):
    meta = {'name': 'DNSDumpster', 'summary': "Passive subdomain enumeration using HackerTarget's DNSDumpster", 'useCases': ['Investigate', 'Footprint', 'Passive'], 'categories': ['Passive DNS'], 'dataSource': {'website': 'https://dnsdumpster.com/', 'model': 'FREE_NOAUTH_UNLIMITED', 'description': 'DNSdumpster.com is a FREE domain research tool that can discover hosts related to a domain.'}}
    opts = {}
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.debug('Setting up sfp_dnsdumpster')
        self.results = self.tempStorage()
        self.opts.update(userOpts)

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['DOMAIN_NAME', 'INTERNET_NAME']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['INTERNET_NAME', 'INTERNET_NAME_UNRESOLVED']

    def query(self, domain):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        url = 'https://dnsdumpster.com'
        res1 = self.sf.fetchUrl(url, useragent=self.opts.get('_useragent', 'Spiderfoot'))
        if res1['code'] not in ['200']:
            self.error(f'''Bad response code "{res1['code']}" from DNSDumpster''')
        else:
            self.debug(f'''Valid response code "{res1['code']}" from DNSDumpster''')
        html = BeautifulSoup(str(res1['content']), features='lxml')
        csrftoken = None
        csrfmiddlewaretoken = None
        try:
            for cookie in res1['headers'].get('set-cookie', '').split(';'):
                (k, v) = cookie.split('=', 1)
                if k == 'csrftoken':
                    csrftoken = str(v)
            csrfmiddlewaretoken = html.find('input', {'name': 'csrfmiddlewaretoken'}).attrs.get('value', None)
        except Exception:
            pass
        if not csrftoken or not csrfmiddlewaretoken:
            self.error('Error obtaining CSRF tokens')
            self.errorState = True
            return ret
        self.debug('Successfully obtained CSRF tokens')
        url = 'https://dnsdumpster.com/'
        subdomains = set()
        res2 = self.sf.fetchUrl(url, cookies={'csrftoken': csrftoken}, postData={'csrfmiddlewaretoken': csrfmiddlewaretoken, 'targetip': str(domain).lower(), 'user': 'free'}, headers={'origin': 'https://dnsdumpster.com', 'referer': 'https://dnsdumpster.com/'}, useragent=self.opts.get('_useragent', 'Spiderfoot'))
        if res2['code'] not in ['200']:
            self.error(f'''Bad response code "{res2['code']}" from DNSDumpster''')
            return ret
        html = BeautifulSoup(str(res2['content']), features='lxml')
        escaped_domain = re.escape(domain)
        match_pattern = re.compile('^[\\w\\.-]+\\.' + escaped_domain + '$')
        for subdomain in html.findAll(text=match_pattern):
            subdomains.add(str(subdomain).strip().lower())
        return list(subdomains)

    def sendEvent(self, source, host):
        if False:
            print('Hello World!')
        if self.sf.resolveHost(host) or self.sf.resolveHost6(host):
            e = SpiderFootEvent('INTERNET_NAME', host, self.__name__, source)
        else:
            e = SpiderFootEvent('INTERNET_NAME_UNRESOLVED', host, self.__name__, source)
        self.notifyListeners(e)

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        query = str(event.data).lower()
        self.debug(f'Received event, {event.eventType}, from {event.module}')
        target = self.getTarget()
        eventDataHash = self.sf.hashstring(query)
        if eventDataHash in self.results or (target.matches(query, includeParents=True) and (not target.matches(query, includeChildren=False))):
            self.debug(f'Skipping already-processed event, {event.eventType}, from {event.module}')
            return
        self.results[eventDataHash] = True
        for hostname in self.query(query):
            if target.matches(hostname, includeParents=True) and (not target.matches(hostname, includeChildren=False)):
                self.sendEvent(event, hostname)
            else:
                self.debug(f'Invalid subdomain: {hostname}')