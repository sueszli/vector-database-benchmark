import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_zonefiles(SpiderFootPlugin):
    meta = {'name': 'ZoneFile.io', 'summary': 'Search ZoneFiles.io Domain query API for domain information.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Passive DNS'], 'dataSource': {'website': 'https://zonefiles.io', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://zonefiles.io/query-api/'], 'apiKeyInstructions': ['Visit https://zonefiles.io', 'Register an account', 'Visit https://zonefiles.io/profile/', "The API key is listed next to 'Your API token'"], 'favIcon': 'https://zonefiles.io/favicon.ico', 'logo': 'https://zonefiles.io/static/images/logo.png', 'description': 'You can fetch data for any domain name with our pay-as-you-go API.'}}
    opts = {'api_key': '', 'verify': True, 'delay': 1}
    optdescs = {'api_key': 'ZoneFiles.io API key.', 'verify': 'Verify specified domains still resolve to the identified IP address.', 'delay': 'Delay between requests, in seconds.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['RAW_RIR_DATA', 'IP_ADDRESS', 'PHONE_NUMBER', 'EMAILADDR', 'PROVIDER_DNS', 'SOFTWARE_USED']

    def queryDomain(self, qry):
        if False:
            i = 10
            return i + 15
        'Query a domain\n\n        Args:\n            qry (str): domain\n\n        Returns:\n            str: API response as JSON\n        '
        headers = {'Accept': 'application/json'}
        res = self.sf.fetchUrl(f"https://zonefiles.io/q/{self.opts['api_key']}/{qry}", headers=headers, timeout=30, useragent=self.opts['_useragent'])
        time.sleep(self.opts['delay'])
        return self.parseApiResponse(res)

    def parseApiResponse(self, res: dict):
        if False:
            i = 10
            return i + 15
        if not res:
            self.error('No response from ZoneFiles.')
            return None
        if res['code'] == '429':
            self.error('You are being rate-limited by ZoneFiles.')
            self.errorState = True
            return None
        if res['code'] == '500' or res['code'] == '502' or res['code'] == '503':
            self.error('ZoneFiles.io service is unavailable')
            self.errorState = True
            return None
        if res['code'] != '200':
            self.error('Failed to retrieve content from ZoneFiles.')
            self.errorState = True
            return None
        if not res['content']:
            return None
        try:
            json_data = json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        data = json_data.get('data')
        if not data:
            return None
        return data

    def handleEvent(self, event):
        if False:
            return 10
        if self.errorState:
            return
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {event.eventType}, from {event.module}')
        if self.opts['api_key'] == '':
            self.error(f'You enabled {self.__class__.__name__} but did not set an API key!')
            self.errorState = True
            return
        data = self.queryDomain(eventData)
        if not data:
            self.debug(f'No information found for domain {eventData}')
            return
        evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
        self.notifyListeners(evt)
        ip = data.get('ip')
        if ip:
            if self.opts['verify']:
                if self.sf.validateIP(eventData, str(ip)):
                    evt = SpiderFootEvent('IP_ADDRESS', str(ip), self.__name__, event)
                    self.notifyListeners(evt)
            else:
                evt = SpiderFootEvent('IP_ADDRESS', str(ip), self.__name__, event)
                self.notifyListeners(evt)
        dns = data.get('dns')
        if dns:
            for nameserver in set(dns.split(',')):
                evt = SpiderFootEvent('PROVIDER_DNS', nameserver, self.__name__, event)
                self.notifyListeners(evt)
        emails = data.get('emails')
        if emails:
            for email in set(emails.split(',')):
                mail_domain = email.lower().split('@')[1]
                if not self.getTarget().matches(mail_domain):
                    self.debug(f'Ignored affiliate email address: {email}')
                    continue
                self.info(f'Found e-mail address: {email}')
                evt_type = 'EMAILADDR'
                if email.split('@')[0] in self.opts['_genericusers'].split(','):
                    evt_type = 'EMAILADDR_GENERIC'
                evt = SpiderFootEvent(evt_type, email, self.__name__, event)
                self.notifyListeners(evt)
        phones = data.get('phones')
        if phones:
            for phone in set(phones.split(',')):
                if SpiderFootHelpers.validPhoneNumber(phone):
                    evt = SpiderFootEvent('PHONE_NUMBER', phone, self.__name__, event)
                    self.notifyListeners(evt)
        technologies = data.get('technologies')
        if technologies and isinstance(technologies, dict):
            for tech in technologies.keys():
                evt = SpiderFootEvent('SOFTWARE_USED', tech, self.__name__, event)
                self.notifyListeners(evt)