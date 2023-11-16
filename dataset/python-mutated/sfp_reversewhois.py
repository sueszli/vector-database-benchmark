import re
from bs4 import BeautifulSoup
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_reversewhois(SpiderFootPlugin):
    meta = {'name': 'ReverseWhois', 'summary': 'Reverse Whois lookups using reversewhois.io.', 'useCases': ['Investigate', 'Passive'], 'categories': ['Search Engines'], 'dataSource': {'website': 'https://www.reversewhois.io/', 'model': 'FREE_NOAUTH_UNLIMITED', 'favIcon': 'https://www.reversewhois.io/dist/img/favicon-32x32.png', 'description': 'ReverseWhois is a free search engine to find domain names owned by an individual or company.\nSearch based on names or email addresses.'}}
    opts = {}
    optdescs = {}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['AFFILIATE_INTERNET_NAME', 'AFFILIATE_DOMAIN_NAME', 'DOMAIN_REGISTRAR']

    def query(self, qry):
        if False:
            print('Hello World!')
        url = f'https://reversewhois.io?searchterm={qry}'
        res = self.sf.fetchUrl(url, timeout=self.opts.get('_fetchtimeout', 30))
        if res['code'] not in ['200']:
            self.error('You may have exceeded ReverseWhois usage limits.')
            self.errorState = True
            return ([], [])
        html = BeautifulSoup(res['content'], features='lxml')
        date_regex = re.compile('\\d{4}-\\d{2}-\\d{2}')
        registrars = set()
        domains = set()
        for table_row in html.findAll('tr'):
            table_cells = table_row.findAll('td')
            try:
                if date_regex.match(table_cells[2].text.strip()):
                    domain = table_cells[1].text.strip().lower()
                    registrar = table_cells[-1].text.strip()
                    if domain:
                        domains.add(domain)
                    if registrar:
                        registrars.add(registrar)
            except IndexError:
                self.debug(f'Invalid row {table_row}')
                continue
        if not registrars and (not domains):
            self.info(f'No ReverseWhois info found for {qry}')
        return (list(domains), list(registrars))

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        (domains, registrars) = self.query(eventData)
        for domain in set(domains):
            if not self.getTarget().matches(domain, includeChildren=False):
                e = SpiderFootEvent('AFFILIATE_INTERNET_NAME', domain, self.__name__, event)
                self.notifyListeners(e)
                if self.sf.isDomain(domain, self.opts['_internettlds']):
                    evt = SpiderFootEvent('AFFILIATE_DOMAIN_NAME', domain, self.__name__, event)
                    self.notifyListeners(evt)
        for registrar in set(registrars):
            e = SpiderFootEvent('DOMAIN_REGISTRAR', registrar, self.__name__, event)
            self.notifyListeners(e)