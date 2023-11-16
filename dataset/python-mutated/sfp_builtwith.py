import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_builtwith(SpiderFootPlugin):
    meta = {'name': 'BuiltWith', 'summary': "Query BuiltWith.com's Domain API for information about your target's web technology stack, e-mail addresses and more.", 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Search Engines'], 'dataSource': {'website': 'https://builtwith.com/', 'model': 'FREE_AUTH_LIMITED', 'references': ['https://api.builtwith.com/', 'https://kb.builtwith.com/', 'https://builtwith.com/screencast', 'https://builtwith.com/faq'], 'apiKeyInstructions': ['Visit https://api.builtwith.com/free-api', 'Register a free account', 'Navigate to https://api.builtwith.com/free-api', "The API key is listed under 'Your API Key'"], 'favIcon': 'https://d28rh9vvmrd65v.cloudfront.net/favicon.ico', 'logo': 'https://d28rh9vvmrd65v.cloudfront.net/favicon.ico', 'description': 'Build lists of websites from our database of 38,701+ web technologies and over a quarter of a billion websites showing which sites use shopping carts, analytics, hosting and many more. Filter by location, traffic, vertical and more.\nKnow your prospects platform before you talk to them. Improve your conversions with validated market adoption.\nGet advanced technology market share information and country based analytics for all web technologies.'}}
    opts = {'api_key': '', 'maxage': 30}
    optdescs = {'api_key': 'Builtwith.com Domain API key.', 'maxage': 'The maximum age of the data returned, in days, in order to be considered valid.'}
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
            print('Hello World!')
        return ['DOMAIN_NAME']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['INTERNET_NAME', 'EMAILADDR', 'EMAILADDR_GENERIC', 'RAW_RIR_DATA', 'WEBSERVER_TECHNOLOGY', 'PHONE_NUMBER', 'DOMAIN_NAME', 'CO_HOSTED_SITE', 'IP_ADDRESS', 'WEB_ANALYTICS_ID']

    def queryRelationships(self, t):
        if False:
            for i in range(10):
                print('nop')
        url = f"https://api.builtwith.com/rv1/api.json?LOOKUP={t}&KEY={self.opts['api_key']}"
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['code'] == '404':
            return None
        if not res['content']:
            return None
        try:
            return json.loads(res['content'])['Relationships']
        except Exception as e:
            self.error(f'Error processing JSON response from builtwith.com: {e}')
        return None

    def queryDomainInfo(self, t):
        if False:
            for i in range(10):
                print('nop')
        url = f"https://api.builtwith.com/rv1/api.json?LOOKUP={t}&KEY={self.opts['api_key']}"
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['code'] == '404':
            return None
        if not res['content']:
            return None
        try:
            return json.loads(res['content'])['Results'][0]
        except Exception as e:
            self.error(f'Error processing JSON response from builtwith.com: {e}')
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
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_builtwith but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.queryDomainInfo(eventData)
        if data is not None:
            if 'Meta' in data:
                if data['Meta'].get('Names', []):
                    for nb in data['Meta']['Names']:
                        e = SpiderFootEvent('RAW_RIR_DATA', 'Possible full name: ' + nb['Name'], self.__name__, event)
                        self.notifyListeners(e)
                        if nb.get('Email', None):
                            if SpiderFootHelpers.validEmail(nb['Email']):
                                if nb['Email'].split('@')[0] in self.opts['_genericusers'].split(','):
                                    evttype = 'EMAILADDR_GENERIC'
                                else:
                                    evttype = 'EMAILADDR'
                                e = SpiderFootEvent(evttype, nb['Email'], self.__name__, event)
                                self.notifyListeners(e)
                if data['Meta'].get('Emails', []):
                    for email in data['Meta']['Emails']:
                        if SpiderFootHelpers.validEmail(email):
                            if email.split('@')[0] in self.opts['_genericusers'].split(','):
                                evttype = 'EMAILADDR_GENERIC'
                            else:
                                evttype = 'EMAILADDR'
                            e = SpiderFootEvent(evttype, email, self.__name__, event)
                            self.notifyListeners(e)
                if data['Meta'].get('Telephones', []):
                    for phone in data['Meta']['Telephones']:
                        phone = phone.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')
                        e = SpiderFootEvent('PHONE_NUMBER', phone, self.__name__, event)
                        self.notifyListeners(e)
            if 'Paths' in data.get('Result', []):
                for p in data['Result']['Paths']:
                    if p.get('SubDomain', ''):
                        h = p['SubDomain'] + '.' + eventData
                        ev = SpiderFootEvent('INTERNET_NAME', h, self.__name__, event)
                        self.notifyListeners(ev)
                        if self.sf.isDomain(h, self.opts['_internettlds']):
                            ev = SpiderFootEvent('DOMAIN_NAME', h, self.__name__, event)
                            self.notifyListeners(ev)
                    else:
                        ev = None
                    for t in p.get('Technologies', []):
                        if ev:
                            src = ev
                        else:
                            src = event
                        agelimit = int(time.time() * 1000) - 86400000 * self.opts['maxage']
                        if t.get('LastDetected', 0) < agelimit:
                            self.debug('Data found too old, skipping.')
                            continue
                        e = SpiderFootEvent('WEBSERVER_TECHNOLOGY', t['Name'], self.__name__, src)
                        self.notifyListeners(e)
        data = self.queryRelationships(eventData)
        if data is None:
            return
        agelimit = int(time.time() * 1000) - 86400000 * self.opts['maxage']
        for r in data:
            if 'Domain' not in r or 'Identifiers' not in r:
                self.debug('Data returned not in the format requested.')
                continue
            if r['Domain'] != eventData:
                self.debug("Data returned doesn't match data requested, skipping.")
                continue
            for i in r['Identifiers']:
                if 'Last' not in i or 'Type' not in i or 'Value' not in i:
                    self.debug('Data returned not in the format requested.')
                    continue
                if i['Last'] < agelimit:
                    self.debug('Data found too old, skipping.')
                    continue
                evttype = None
                if i['Type'] == 'ip':
                    if self.sf.validIP(i['Value']):
                        val = i['Value']
                        evttype = 'IP_ADDRESS'
                    else:
                        val = i['Value'].strip('.')
                        if self.getTarget.matches(val):
                            evttype = 'INTERNET_NAME'
                        else:
                            evttype = 'CO_HOSTED_SITE'
                    e = SpiderFootEvent(evttype, val, self.__name__, event)
                    self.notifyListeners(e)
                    continue
                txt = i['Type'] + ': ' + str(i['Value'])
                e = SpiderFootEvent('WEB_ANALYTICS_ID', txt, self.__name__, event)
                self.notifyListeners(e)
                if i['Matches']:
                    for m in i['Matches']:
                        if 'Domain' not in m:
                            continue
                        evt = SpiderFootEvent('AFFILIATE_INTERNET_NAME', m['Domain'], self.__name__, e)
                        self.notifyListeners(evt)
                        if self.sf.isDomain(m['Domain'], self.opts['_internettlds']):
                            evt = SpiderFootEvent('AFFILIATE_DOMAIN_NAME', m['Domain'], self.__name__, e)
                            self.notifyListeners(evt)