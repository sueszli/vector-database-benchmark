from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_email(SpiderFootPlugin):
    meta = {'name': 'E-Mail Address Extractor', 'summary': 'Identify e-mail addresses in any obtained data.', 'useCases': ['Passive', 'Investigate', 'Footprint'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['TARGET_WEB_CONTENT', 'BASE64_DATA', 'AFFILIATE_DOMAIN_WHOIS', 'CO_HOSTED_SITE_DOMAIN_WHOIS', 'DOMAIN_WHOIS', 'NETBLOCK_WHOIS', 'LEAKSITE_CONTENT', 'RAW_DNS_RECORDS', 'RAW_FILE_META_DATA', 'RAW_RIR_DATA', 'SIMILARDOMAIN_WHOIS', 'SSL_CERTIFICATE_RAW', 'SSL_CERTIFICATE_ISSUED', 'TCP_PORT_OPEN_BANNER', 'WEBSERVER_BANNER', 'WEBSERVER_HTTPHEADERS']

    def producedEvents(self):
        if False:
            return 10
        return ['EMAILADDR', 'EMAILADDR_GENERIC', 'AFFILIATE_EMAILADDR']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        emails = SpiderFootHelpers.extractEmailsFromText(eventData)
        for email in set(emails):
            evttype = 'EMAILADDR'
            email = email.lower()
            mailDom = email.split('@')[1].strip('.')
            if not self.sf.validHost(mailDom, self.opts['_internettlds']):
                self.debug(f'Skipping {email} as not a valid e-mail.')
                continue
            if not self.getTarget().matches(mailDom, includeChildren=True, includeParents=True) and (not self.getTarget().matches(email)):
                self.debug('External domain, so possible affiliate e-mail')
                evttype = 'AFFILIATE_EMAILADDR'
            if eventName.startswith('AFFILIATE_'):
                evttype = 'AFFILIATE_EMAILADDR'
            if not evttype.startswith('AFFILIATE_') and email.split('@')[0] in self.opts['_genericusers'].split(','):
                evttype = 'EMAILADDR_GENERIC'
            self.info(f'Found e-mail address: {email}')
            mail = email.strip('.')
            evt = SpiderFootEvent(evttype, mail, self.__name__, event)
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            self.notifyListeners(evt)