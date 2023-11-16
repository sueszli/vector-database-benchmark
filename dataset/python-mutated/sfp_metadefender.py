import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_metadefender(SpiderFootPlugin):
    meta = {'name': 'MetaDefender', 'summary': 'Search MetaDefender API for IP address and domain IP reputation.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://metadefender.opswat.com/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://onlinehelp.opswat.com/mdcloud/'], 'apiKeyInstructions': ['Visit https://metadefender.opswat.com/', 'Register a free account', 'Navigate to https://metadefender.opswat.com/account', "The API key is listed under 'API key'"], 'favIcon': 'https://mcl-cdn.opswat.com/1.40.3-729f31db/city/icons/icon-48x48.png?v=61be50566cce944a710aaa90ba6bbb8d', 'logo': 'https://mcl-cdn.opswat.com/1.40.3-729f31db/city/icons/icon-48x48.png?v=61be50566cce944a710aaa90ba6bbb8d', 'description': 'File Analysis - Analyzing binaries with 30+ anti-malware engines.\nHeuristic analysis to detect more unknown and targeted attacks.\nBinary vulnerability data assessment, IP/Domain reputation, Threat Intelligence Feeds'}}
    opts = {'api_key': '', 'delay': 6}
    optdescs = {'api_key': 'MetaDefender API key.', 'delay': 'Delay between requests, in seconds.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['IP_ADDRESS', 'INTERNET_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['MALICIOUS_IPADDR', 'MALICIOUS_INTERNET_NAME', 'BLACKLISTED_IPADDR', 'BLACKLISTED_INTERNET_NAME', 'GEOINFO']

    def queryDomain(self, qry):
        if False:
            i = 10
            return i + 15
        headers = {'Accept': 'application/json', 'apikey': self.opts['api_key']}
        res = self.sf.fetchUrl('https://api.metadefender.com/v4/domain/' + qry, headers=headers, timeout=15, useragent=self.opts['_useragent'])
        time.sleep(self.opts['delay'])
        return self.parseApiResponse(res)

    def queryIp(self, qry):
        if False:
            i = 10
            return i + 15
        headers = {'Accept': 'application/json', 'apikey': self.opts['api_key']}
        res = self.sf.fetchUrl('https://api.metadefender.com/v4/ip/' + qry, headers=headers, timeout=15, useragent=self.opts['_useragent'])
        time.sleep(self.opts['delay'])
        return self.parseApiResponse(res)

    def parseApiResponse(self, res: dict):
        if False:
            for i in range(10):
                print('nop')
        if not res:
            self.error('No response from MetaDefender.')
            return None
        if res['code'] == '401':
            self.error('Invalid MetaDefender API key')
            self.errorState = True
            return None
        if res['code'] == '429':
            self.error('You are being rate-limited by MetaDefender')
            self.errorState = True
            return None
        if res['code'] == '404':
            return None
        if res['content'] is None:
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
        return None

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        eventData = event.data
        if self.errorState:
            return
        if eventData in self.results:
            return
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_metadefender but did not set an API key!')
            self.errorState = True
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {event.module}')
        if eventName == 'IP_ADDRESS':
            data = self.queryIp(eventData)
            if data is None:
                self.debug('No matches found for ' + eventData)
                return
            geo_info = data.get('geo_info')
            if geo_info:
                location = ', '.join([_f for _f in [geo_info.get('city').get('name'), geo_info.get('country').get('name')] if _f])
                evt = SpiderFootEvent('GEOINFO', location, self.__name__, event)
                self.notifyListeners(evt)
            res = data.get('lookup_results')
            if not res:
                self.debug('No matches found for ' + eventData)
                return
            sources = res.get('sources')
            if not sources:
                self.debug('No matches found for ' + eventData)
                return
            for m in sources:
                if not m.get('assessment'):
                    continue
                if m['assessment'] != 'trustworthy':
                    continue
                provider = m.get('provider')
                evt = SpiderFootEvent('MALICIOUS_IPADDR', provider + ' [' + eventData + ']', self.__name__, event)
                self.notifyListeners(evt)
                evt = SpiderFootEvent('BLACKLISTED_IPADDR', provider + ' [' + eventData + ']', self.__name__, event)
                self.notifyListeners(evt)
        if eventName == 'INTERNET_NAME':
            data = self.queryDomain(eventData)
            if data is None:
                self.debug('No matches found for ' + eventData)
                return
            res = data.get('lookup_results')
            if not res:
                self.debug('No matches found for ' + eventData)
                return
            sources = res.get('sources')
            if not sources:
                self.debug('No matches found for ' + eventData)
                return
            for m in sources:
                if not m.get('assessment'):
                    continue
                if m['assessment'] == 'trustworthy':
                    continue
                provider = m.get('provider')
                evt = SpiderFootEvent('MALICIOUS_INTERNET_NAME', provider + ' [' + eventData + ']', self.__name__, event)
                self.notifyListeners(evt)
                evt = SpiderFootEvent('BLACKLISTED_INTERNET_NAME', provider + ' [' + eventData + ']', self.__name__, event)
                self.notifyListeners(evt)