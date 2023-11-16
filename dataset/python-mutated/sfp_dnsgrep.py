import json
import urllib.error
import urllib.parse
import urllib.request
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_dnsgrep(SpiderFootPlugin):
    meta = {'name': 'DNSGrep', 'summary': 'Obtain Passive DNS information from Rapid7 Sonar Project using DNSGrep API.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Passive DNS'], 'dataSource': {'website': 'https://opendata.rapid7.com/', 'model': 'FREE_AUTH_UNLIMITED', 'references': ['https://opendata.rapid7.com/apihelp/', 'https://www.rapid7.com/about/research'], 'apiKeyInstructions': ['Visit https://opendata.rapid7.com/apihelp/', 'Submit form requesting for access', 'After getting access, navigate to https://insight.rapid7.com/platform#/apiKeyManagement', 'Create an User Key', 'The API key will be listed after creation'], 'favIcon': 'https://www.rapid7.com/includes/img/favicon.ico', 'logo': 'https://www.rapid7.com/includes/img/Rapid7_logo.svg', 'description': 'Offering researchers and community members open access to data from Project Sonar, which conducts internet-wide surveys to gain insights into global exposure to common vulnerabilities.'}}
    opts = {'timeout': 30, 'dns_resolve': True}
    optdescs = {'timeout': 'Query timeout, in seconds.', 'dns_resolve': 'DNS resolve each identified domain.'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['INTERNET_NAME', 'INTERNET_NAME_UNRESOLVED', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            print('Hello World!')
        params = {'q': '.' + qry.encode('raw_unicode_escape').decode('ascii', errors='replace')}
        res = self.sf.fetchUrl('https://dns.bufferover.run/dns?' + urllib.parse.urlencode(params), timeout=self.opts['timeout'], useragent=self.opts['_useragent'])
        if res['content'] is None:
            self.info('No results found for ' + qry)
            return None
        if res['code'] != '200':
            self.debug('Error retrieving search results for ' + qry)
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f'Error processing JSON response from DNSGrep: {e}')
        return None

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        data = self.query(eventData)
        if data is None:
            self.info('No DNS records found for ' + eventData)
            return
        evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
        self.notifyListeners(evt)
        domains = list()
        fdns = data.get('FDNS_A')
        if fdns:
            for r in fdns:
                try:
                    (ip, domain) = r.split(',')
                except Exception:
                    continue
                domains.append(domain)
        rdns = data.get('RDNS')
        if rdns:
            for r in rdns:
                try:
                    (ip, domain) = r.split(',')
                except Exception:
                    continue
                domains.append(domain)
        for domain in domains:
            if domain in self.results:
                continue
            if not self.getTarget().matches(domain, includeParents=True):
                continue
            evt_type = 'INTERNET_NAME'
            if self.opts['dns_resolve'] and (not self.sf.resolveHost(domain)) and (not self.sf.resolveHost6(domain)):
                self.debug(f'Host {domain} could not be resolved')
                evt_type += '_UNRESOLVED'
            evt = SpiderFootEvent(evt_type, domain, self.__name__, event)
            self.notifyListeners(evt)