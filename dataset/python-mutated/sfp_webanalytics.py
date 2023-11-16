import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_webanalytics(SpiderFootPlugin):
    meta = {'name': 'Web Analytics Extractor', 'summary': 'Identify web analytics IDs in scraped webpages and DNS TXT records.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

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
        return ['TARGET_WEB_CONTENT', 'DNS_TEXT']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['WEB_ANALYTICS_ID']

    def handleEvent(self, event):
        if False:
            i = 10
            return i + 15
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        sourceData = self.sf.hashstring(eventData)
        if sourceData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[sourceData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if event.moduleDataSource:
            datasource = event.moduleDataSource
        else:
            datasource = 'Unknown'
        if eventName == 'TARGET_WEB_CONTENT':
            matches = re.findall('\\bua\\-\\d{4,10}\\-\\d{1,4}\\b', eventData, re.IGNORECASE)
            for m in matches:
                if m.lower().startswith('ua-000000-'):
                    continue
                if m.lower().startswith('ua-123456-'):
                    continue
                if m.lower().startswith('ua-12345678'):
                    continue
                self.debug('Google Analytics match: ' + m)
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Google Analytics: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('\\b(pub-\\d{10,20})\\b', eventData, re.IGNORECASE)
            for m in matches:
                if m.lower().startswith('pub-12345678'):
                    continue
                self.debug('Google AdSense match: ' + m)
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Google AdSense: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('\\b(GTM-[0-9a-zA-Z]{6,10})\\b', eventData)
            for m in set(matches):
                if m.lower().startswith('GTM-XXXXXX'):
                    continue
                self.debug(f'Google Tag Manager match: {m}')
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', f'Google Tag Manager: {m}', self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('<meta name="google-site-verification" content="([a-z0-9\\-\\+_=]{43,44})"', eventData, re.IGNORECASE)
            for m in matches:
                self.debug('Google Site Verification match: ' + m)
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Google Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('<meta name="verify-v1" content="([a-z0-9\\-\\+_=]{43,44})"', eventData, re.IGNORECASE)
            for m in matches:
                self.debug('Google Site Verification match: ' + m)
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Google Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            if '_qevents.push' in eventData:
                matches = re.findall('\\bqacct:\\"(p-[a-z0-9]+)\\"', eventData, re.IGNORECASE)
                for m in matches:
                    self.debug('Quantcast match: ' + m)
                    evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Quantcast: ' + m, self.__name__, event)
                    evt.moduleDataSource = datasource
                    self.notifyListeners(evt)
            matches = re.findall('<meta name="ahrefs-site-verification" content="([a-f0-9]{64})"', eventData, re.IGNORECASE)
            for m in matches:
                self.debug('Ahrefs Site Verification match: ' + m)
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Ahrefs Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
        if eventName == 'DNS_TEXT':
            matches = re.findall('google-site-verification=([a-z0-9\\-\\+_=]{43,44})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Google Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('logmein-domain-confirmation ([A-Z0-9]{24})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'LogMeIn Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('logmein-verification-code=([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'LogMeIn Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('docusign=([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'DocuSign Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('globalsign-domain-verification=([a-z0-9\\-\\+_=]{42,44})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'GlobalSign Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('atlassian-domain-verification=([a-z0-9\\-\\+\\/_=]{64})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Atlassian Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('adobe-idp-site-verification=([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Adobe IDP Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('adobe-idp-site-verification=([a-f0-9]{64})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Adobe IDP Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('adobe-sign-verification=([a-f0-9]{32})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Adobe Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('stripe-verification=([a-f0-9]{64})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Stripe Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('teamviewer-sso-verification=([a-f0-9]{32})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'TeamViewer SSO Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('aliyun-site-verification=([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Aliyun Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('facebook-domain-verification=([a-z0-9]{30})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Facebook Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('citrix-verification-code=([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Citrix Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('dropbox-domain-verification=([a-z0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Dropbox Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('detectify-verification=([a-f0-9]{32})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Detectify Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('drift-verification=([a-f0-9]{64})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Drift Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('ahrefs-site-verification_([a-f0-9]{64})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Ahrefs Site Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('status-page-domain-verification=([a-z0-9]{12})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Statuspage Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('ZOOM_verify_([a-z0-9\\-\\+\\/_=]{22})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Zoom.us Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('mailru-verification: ([a-z0-9]{16})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Mail.ru Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('yandex-verification: ([a-z0-9]{16})$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Yandex Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('brave-ledger-verification=([a-z0-9]+)$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Brave Ledger Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('have-i-been-pwned-verification=([a-f0-9]+)$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'have-i-been-pwned Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)
            matches = re.findall('cisco-ci-domain-verification=([a-f0-9]+)$', eventData.strip(), re.IGNORECASE)
            for m in matches:
                evt = SpiderFootEvent('WEB_ANALYTICS_ID', 'Cisco Live Domain Verification: ' + m, self.__name__, event)
                evt.moduleDataSource = datasource
                self.notifyListeners(evt)