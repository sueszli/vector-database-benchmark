import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_googlesafebrowsing(SpiderFootPlugin):
    meta = {'name': 'Google SafeBrowsing', 'summary': 'Check if the URL is included on any of the Safe Browsing lists.', 'flags': ['slow', 'apikey'], 'useCases': ['Passive', 'Investigate'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://developers.google.com/safe-browsing/v4/lookup-api', 'model': 'FREE_AUTH_UNLIMITED', 'references': ['https://developers.google.com/safe-browsing/v4/reference/rest'], 'apiKeyInstructions': ['Visit https://console.developers.google.com/', 'Register a free Google account or sign in', 'Create or select existing Google Developer Console project', 'Go to the Cloud Console API Library https://console.cloud.google.com/apis/library', 'From the projects list, select the project you want to use', "In the API Library select 'Safe Browsing APIs'", 'On the API page, click ENABLE', 'Navigate to the APIs & Services→Credentials panel in Cloud Console', 'Select Create credentials, then select API key from the dropdown menu', 'The API key created dialog box displays your newly created key'], 'favIcon': 'https://www.gstatic.com/devrel-devsite/prod/v1241c04ebcb2127897d6c18221acbd64e7ed5c46e5217fd83dd808e592c47bf6/developers/images/favicon.png', 'logo': 'https://developers.google.com/safe-browsing/images/SafeBrowsing_Icon.png', 'description': "The Safe Browsing APIs (v4) let your client applications check URLs against Google's constantly updated lists of unsafe web resources. Any URL found on a Safe Browsing list is considered unsafe."}}
    opts = {'api_key': ''}
    optdescs = {'api_key': 'Google Safe Browsing API key.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['INTERNET_NAME', 'IP_ADDRESS', 'AFFILIATE_INTERNET_NAME', 'AFFILIATE_IPADDR', 'CO_HOSTED_SITE']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['MALICIOUS_IPADDR', 'MALICIOUS_INTERNET_NAME', 'MALICIOUS_AFFILIATE_IPADDR', 'MALICIOUS_AFFILIATE_INTERNET_NAME', 'MALICIOUS_COHOST', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            i = 10
            return i + 15
        headers = {'Content-Type': 'application/json'}
        url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={self.opts['api_key']}"
        payload = {'client': {'clientId': 'SpiderFoot', 'clientVersion': '3.2'}, 'threatInfo': {'threatTypes': ['THREAT_TYPE_UNSPECIFIED', 'MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION'], 'platformTypes': ['PLATFORM_TYPE_UNSPECIFIED', 'ANY_PLATFORM'], 'threatEntryTypes': ['THREAT_ENTRY_TYPE_UNSPECIFIED', 'URL', 'EXECUTABLE'], 'threatEntries': [{'url': qry.encode('raw_unicode_escape').decode('ascii', errors='replace')}]}}
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], headers=headers, postData=json.dumps(payload))
        if res['code'] == '400':
            self.error('Invalid request payload on Google Safe Browsing API')
            self.errorState = True
            return None
        if res['code'] == '429':
            self.error('Reaching rate limit on Google Safe Browsing API')
            self.errorState = True
            return None
        if res['code'] == '403':
            self.error('Permission denied, invalid API key on Google Safe Browsing API')
            self.errorState = True
            return None
        if res['code'] in ['500', '503', '504']:
            self.error('Google Safe Browsing API is having some troubles or unavailable.')
            self.errorState = True
            return None
        try:
            info = json.loads(res['content'])
            if info == {}:
                self.info('No Google Safe Browsing matches found for ' + qry)
                return None
        except Exception as e:
            self.error(f'Error processing JSON response from SHODAN: {e}')
            return None
        return info

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_googlesafebrowsing but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        evtType = ''
        if eventName in ['IP_ADDRESS', 'AFFILIATE_IPADDR']:
            if eventName == 'IP_ADDRESS':
                evtType = 'MALICIOUS_IPADDR'
            else:
                evtType = 'MALICIOUS_AFFILIATE_IPADDR'
        if eventName in ['INTERNET_NAME', 'CO_HOSTED_SITE', 'AFFILIATE_INTERNET_NAME']:
            if eventName == 'INTERNET_NAME':
                evtType = 'MALICIOUS_INTERNET_NAME'
            if eventName == 'AFFILIATE_INTERNET_NAME':
                evtType = 'MALICIOUS_AFFILIATE_INTERNET_NAME'
            if eventName == 'CO_HOSTED_SITE':
                evtType = 'MALICIOUS_COHOST'
        rec = self.query(eventData)
        if rec is None:
            return
        evt = SpiderFootEvent('RAW_RIR_DATA', str(rec), self.__name__, event)
        self.notifyListeners(evt)
        evt = SpiderFootEvent(evtType, 'Google SafeBrowsing [' + eventData + ']', self.__name__, event)
        self.notifyListeners(evt)