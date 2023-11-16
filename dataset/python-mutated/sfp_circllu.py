import base64
import json
import re
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_circllu(SpiderFootPlugin):
    meta = {'name': 'CIRCL.LU', 'summary': "Obtain information from CIRCL.LU's Passive DNS and Passive SSL databases.", 'flags': ['apikey'], 'useCases': ['Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://www.circl.lu/', 'model': 'FREE_AUTH_UNLIMITED', 'references': ['https://www.circl.lu/services/passive-dns/', 'https://www.circl.lu/services/passive-ssl/', 'https://www.circl.lu/services/', 'https://www.circl.lu/pub/', 'https://www.circl.lu/projects'], 'apiKeyInstructions': ['Visit https://www.circl.lu/contact/', 'Contact with email or phone to request access for Passive DNS and Passive SSL API services', 'The API access will be provided once approved'], 'favIcon': 'https://www.google.com/s2/favicons?domain=https://www.circl.lu/', 'logo': 'https://www.circl.lu/assets/images/circl-logo.png', 'description': 'The Computer Incident Response Center Luxembourg (CIRCL) is a government-driven initiative designed to gather, review, report and respond to computer security threats and incidents.\nCIRCL provides a reliable and trusted point of contact for any users, companies and organizations based in Luxembourg, for the handling of attacks and incidents. Its team of experts acts like a fire brigade, with the ability to react promptly and efficiently whenever threats are suspected, detected or incidents occur.'}}
    opts = {'api_key_login': '', 'api_key_password': '', 'age_limit_days': 0, 'verify': True, 'cohostsamedomain': False, 'maxcohost': 100}
    optdescs = {'api_key_login': 'CIRCL.LU login.', 'api_key_password': 'CIRCL.LU password.', 'age_limit_days': 'Ignore any Passive DNS records older than this many days. 0 = unlimited.', 'verify': 'Verify co-hosts are valid by checking if they still resolve to the shared IP.', 'cohostsamedomain': 'Treat co-hosted sites on the same target domain as co-hosting?', 'maxcohost': 'Stop reporting co-hosted sites after this many are found, as it would likely indicate web hosting.'}
    results = None
    errorState = False
    cohostcount = 0

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.cohostcount = 0
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['INTERNET_NAME', 'NETBLOCK_OWNER', 'IP_ADDRESS', 'DOMAIN_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['IP_ADDRESS', 'SSL_CERTIFICATE_ISSUED', 'CO_HOSTED_SITE']

    def query(self, qry, qtype):
        if False:
            for i in range(10):
                print('nop')
        if self.errorState:
            return None
        if qtype == 'PDNS':
            url = 'https://www.circl.lu/pdns/query/' + qry
        else:
            url = 'https://www.circl.lu/v2pssl/query/' + qry
        secret = self.opts['api_key_login'] + ':' + self.opts['api_key_password']
        b64_val = base64.b64encode(secret.encode('utf-8'))
        headers = {'Authorization': f"Basic {b64_val.decode('utf-8')}"}
        res = self.sf.fetchUrl(url, timeout=30, useragent='SpiderFoot', headers=headers)
        if res['code'] not in ['200', '201']:
            self.error('CIRCL.LU access seems to have been rejected or you have exceeded usage limits.')
            self.errorState = True
            return None
        if res['content'] is None:
            self.info('No CIRCL.LU info found for ' + qry)
            return None
        return res['content']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        ret = None
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_circllu':
            self.debug('Ignoring ' + eventName + ', from self.')
            return
        if self.opts['api_key_login'] == '' or self.opts['api_key_password'] == '':
            self.error('You enabled sfp_circllu but did not set an credentials!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName in ['IP_ADDRESS', 'NETBLOCK_OWNER']:
            if '/' in eventData:
                (addr, mask) = eventData.split('/')
                if int(mask) < 23:
                    self.debug('Network size bigger than permitted by CIRCL.LU.')
                else:
                    ret = self.query(eventData, 'PSSL')
                    if not ret:
                        self.info('No CIRCL.LU passive SSL data found for ' + eventData)
            else:
                ret = self.query(eventData, 'PSSL')
                if not ret:
                    self.info('No CIRCL.LU passive SSL data found for ' + eventData)
            if ret:
                try:
                    j = json.loads(ret)
                    for ip in j:
                        ipe = event
                        if ip != eventData:
                            ipe = SpiderFootEvent('IP_ADDRESS', ip, self.__name__, event)
                            self.notifyListeners(ipe)
                        for crt in j[ip]['subjects']:
                            r = re.findall('.*[\\"\\\'](.+CN=([a-zA-Z0-9\\-\\*\\.])+)[\\"\\\'].*', str(j[ip]['subjects'][crt]), re.IGNORECASE)
                            if r:
                                e = SpiderFootEvent('SSL_CERTIFICATE_ISSUED', r[0][0], self.__name__, ipe)
                                self.notifyListeners(e)
                except Exception as e:
                    self.error('Invalid response returned from CIRCL.LU: ' + str(e))
        if eventName in ['IP_ADDRESS', 'INTERNET_NAME', 'DOMAIN_NAME']:
            ret = self.query(eventData, 'PDNS')
            if not ret:
                self.info('No CIRCL.LU passive DNS data found for ' + eventData)
                return
            for line in ret.split('\n'):
                if len(line) < 2:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    self.error('Invalid response returned from CIRCL.LU: ' + str(e))
                    continue
                age_limit_ts = int(time.time()) - 86400 * self.opts['age_limit_days']
                if self.opts['age_limit_days'] > 0 and rec['time_last'] < age_limit_ts:
                    self.debug('Record found but too old, skipping.')
                    continue
                cohosts = list()
                if eventName == 'IP_ADDRESS':
                    if rec['rrtype'] == 'A' and rec['rdata'] == eventData:
                        if not self.getTarget().matches(rec['rrname']):
                            cohosts.append(rec['rrname'])
                if eventName in ['INTERNET_NAME', 'DOMAIN_NAME']:
                    if rec['rdata'] == eventData:
                        if not self.getTarget().matches(rec['rrname']):
                            cohosts.append(rec['rrname'])
                for co in cohosts:
                    if eventName == 'IP_ADDRESS' and (self.opts['verify'] and (not self.sf.validateIP(co, eventData))):
                        self.debug('Host no longer resolves to our IP.')
                        continue
                    if not self.opts['cohostsamedomain']:
                        if self.getTarget().matches(co, includeParents=True):
                            self.debug('Skipping ' + co + ' because it is on the same domain.')
                            continue
                    if self.cohostcount < self.opts['maxcohost']:
                        e = SpiderFootEvent('CO_HOSTED_SITE', co, self.__name__, event)
                        self.notifyListeners(e)
                        self.cohostcount += 1