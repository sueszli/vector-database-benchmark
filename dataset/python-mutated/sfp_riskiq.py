import base64
import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_riskiq(SpiderFootPlugin):
    meta = {'name': 'RiskIQ', 'summary': "Obtain information from RiskIQ's (formerly PassiveTotal) Passive DNS and Passive SSL databases.", 'flags': ['apikey'], 'useCases': ['Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://community.riskiq.com/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://info.riskiq.net/help', 'https://www.riskiq.com/resources/?type=training_videos', 'https://api.riskiq.net/api/concepts.html'], 'apiKeyInstructions': ['Visit https://community.riskiq.com/home', 'Register a free account', 'Navigate to https://community.riskiq.com/settings', "Click on 'Show' beside 'User'", "The API Key combination will be under 'Key' and 'Secret'"], 'favIcon': 'https://community.riskiq.com/static/assets/favicon.png', 'logo': 'https://community.riskiq.com/static/assets/favicon.png', 'description': 'RiskIQ Community brings petabytes of internet intelligence directly to your fingertips. Investigate threats by pivoting through attacker infrastructure data. Understand your digital assets that are internet-exposed, and map and monitor your external attack surface.'}}
    opts = {'api_key_login': '', 'api_key_password': '', 'verify': True, 'cohostsamedomain': False, 'maxcohost': 100}
    optdescs = {'api_key_login': 'RiskIQ login.', 'api_key_password': 'RiskIQ API Key.', 'verify': 'Verify co-hosts are valid by checking if they still resolve to the shared IP.', 'cohostsamedomain': 'Treat co-hosted sites on the same target domain as co-hosting?', 'maxcohost': 'Stop reporting co-hosted sites after this many are found, as it would likely indicate web hosting.'}
    results = None
    errorState = False
    cohostcount = 0

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.cohostcount = 0
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['INTERNET_NAME', 'IP_ADDRESS', 'DOMAIN_NAME', 'EMAILADDR']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['IP_ADDRESS', 'INTERNET_NAME', 'AFFILIATE_INTERNET_NAME', 'DOMAIN_NAME', 'AFFILIATE_DOMAIN_NAME', 'INTERNET_NAME_UNRESOLVED', 'CO_HOSTED_SITE', 'NETBLOCK_OWNER']

    def query(self, qry, qtype, opts=dict()):
        if False:
            i = 10
            return i + 15
        ret = None
        post = None
        if self.errorState:
            return None
        if qtype == 'PDNS':
            url = 'https://api.passivetotal.org/v2/dns/search/keyword'
            post = '{"query": "' + qry + '"}'
        if qtype == 'PSSL':
            url = 'https://api.passivetotal.org/v2/ssl-certificate/search'
            post = '{"field": "subjectCommonName", "query": "' + qry + '"}'
        if qtype == 'WHOIS':
            url = 'https://api.passivetotal.org/v2/whois/search'
            post = '{"field": "email", "query": "' + qry + '"}'
        api_key_login = self.opts['api_key_login']
        if type(api_key_login) == str:
            api_key_login = api_key_login.encode('utf-8')
        api_key_password = self.opts['api_key_password']
        if type(api_key_password) == str:
            api_key_password = api_key_password.encode('utf-8')
        cred = base64.b64encode(api_key_login + ':'.encode('utf-8') + api_key_password)
        headers = {'Authorization': 'Basic ' + cred.decode('utf-8'), 'Content-Type': 'application/json'}
        res = self.sf.fetchUrl(url, timeout=30, useragent='SpiderFoot', headers=headers, postData=post)
        if res['code'] in ['400', '429', '500', '403']:
            self.error('RiskIQ access seems to have been rejected or you have exceeded usage limits.')
            self.errorState = True
            return None
        if res['content'] is None:
            self.info('No RiskIQ info found for ' + qry)
            return None
        try:
            ret = json.loads(res['content'])
            if 'results' not in ret:
                self.info('No RiskIQ info found for ' + qry)
                return None
        except Exception as e:
            self.error(f'Invalid JSON returned by RiskIQ: {e}')
            return None
        return ret['results']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_riskiq':
            self.debug('Ignoring ' + eventName + ', from self.')
            return
        if self.opts['api_key_login'] == '' or self.opts['api_key_password'] == '':
            self.error('You enabled sfp_riskiq but did not set an credentials!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName == 'DOMAIN_NAME':
            ret = self.query(eventData, 'PSSL')
            if not ret:
                self.info(f'No RiskIQ passive SSL data found for {eventData}')
            else:
                for res in ret:
                    host = res.get('subjectCommonName')
                    if not host:
                        continue
                    if host == eventData:
                        continue
                    if self.getTarget().matches(host, includeChildren=True):
                        if self.sf.resolveHost(host) or self.sf.resolveHost6(host):
                            e = SpiderFootEvent('INTERNET_NAME', host, self.__name__, event)
                        else:
                            e = SpiderFootEvent('INTERNET_NAME_UNRESOLVED', host, self.__name__, event)
                        self.notifyListeners(e)
                        if self.sf.isDomain(host, self.opts['_internettlds']):
                            e = SpiderFootEvent('DOMAIN_NAME', host, self.__name__, event)
                            self.notifyListeners(e)
        if eventName == 'EMAILADDR':
            ret = self.query(eventData, 'WHOIS')
            if not ret:
                self.info('No RiskIQ passive DNS data found for ' + eventData)
                return
            for r in ret:
                if not eventData.endswith('@' + r['domain']):
                    if self.sf.validIP(r['domain']):
                        t = 'NETBLOCK_OWNER'
                    else:
                        t = 'AFFILIATE_INTERNET_NAME'
                    e = SpiderFootEvent(t, r['domain'], self.__name__, event)
                    self.notifyListeners(e)
                    if t == 'AFFILIATE_INTERNET_NAME' and self.sf.isDomain(r['domain'], self.opts['_internettlds']):
                        evt = SpiderFootEvent('AFFILIATE_DOMAIN_NAME', r['domain'], self.__name__, event)
                        self.notifyListeners(evt)
            return
        if eventName in ['IP_ADDRESS', 'INTERNET_NAME', 'DOMAIN_NAME']:
            ret = self.query(eventData, 'PDNS')
            if not ret:
                self.info('No RiskIQ passive DNS data found for ' + eventData)
                return
            cohosts = list()
            if eventName == 'IP_ADDRESS':
                for r in ret:
                    if r['focusPoint'].endswith('.'):
                        r['focusPoint'] = r['focusPoint'][:-1]
                    if not self.getTarget().matches(r['focusPoint']) and '*' not in r['focusPoint']:
                        cohosts.append(r['focusPoint'])
            if eventName in ['INTERNET_NAME', 'DOMAIN_NAME']:
                for r in ret:
                    if r['focusPoint'].endswith('.'):
                        r['focusPoint'] = r['focusPoint'][:-1]
                    if r['focusPoint'] != eventData and '*' not in r['focusPoint']:
                        cohosts.append(r['focusPoint'])
            for co in cohosts:
                if co == eventData:
                    continue
                if eventName == 'IP_ADDRESS' and (self.opts['verify'] and (not self.sf.validateIP(co, eventData))):
                    self.debug('Host no longer resolves to our IP.')
                    continue
                if not self.opts['cohostsamedomain']:
                    if self.getTarget().matches(co, includeParents=True):
                        if self.sf.resolveHost(co) or self.sf.resolveHost6(co):
                            e = SpiderFootEvent('INTERNET_NAME', co, self.__name__, event)
                        else:
                            e = SpiderFootEvent('INTERNET_NAME_UNRESOLVED', co, self.__name__, event)
                        self.notifyListeners(e)
                        if self.sf.isDomain(co, self.opts['_internettlds']):
                            e = SpiderFootEvent('DOMAIN_NAME', co, self.__name__, event)
                            self.notifyListeners(e)
                        continue
                if self.cohostcount < self.opts['maxcohost']:
                    e = SpiderFootEvent('CO_HOSTED_SITE', co, self.__name__, event)
                    self.notifyListeners(e)
                    self.cohostcount += 1