import re
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_crossref(SpiderFootPlugin):
    meta = {'name': 'Cross-Referencer', 'summary': "Identify whether other domains are associated ('Affiliates') of the target by looking for links back to the target site(s).", 'flags': [], 'useCases': ['Footprint'], 'categories': ['Crawling and Scanning']}
    opts = {'checkbase': True}
    optdescs = {'checkbase': 'Check the base URL of the potential affiliate if no direct affiliation found?'}
    fetched = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.fetched = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['LINKED_URL_EXTERNAL', 'SIMILARDOMAIN', 'CO_HOSTED_SITE', 'DARKNET_MENTION_URL']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['AFFILIATE_INTERNET_NAME', 'AFFILIATE_WEB_CONTENT']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName in ['SIMILARDOMAIN', 'CO_HOSTED_SITE']:
            url = 'http://' + eventData.lower()
        elif 'URL' in eventName:
            url = eventData
        else:
            return
        fqdn = self.sf.urlFQDN(url)
        if self.getTarget().matches(fqdn):
            self.debug(f'Ignoring {url} as not external')
            return
        if eventData in self.fetched:
            self.debug(f'Ignoring {url} as already tested')
            return
        if not self.sf.resolveHost(fqdn) and (not self.sf.resolveHost6(fqdn)):
            self.debug(f'Ignoring {url} as {fqdn} does not resolve')
            return
        self.fetched[url] = True
        self.debug(f'Testing URL for affiliation: {url}')
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], sizeLimit=10000000, verify=False)
        if res['content'] is None:
            self.debug(f'Ignoring {url} as no data returned')
            return
        matched = False
        for name in self.getTarget().getNames():
            pat = re.compile('([\\.\\\'\\/\\"\\ ]' + re.escape(name) + '[\\.\\\'\\/\\"\\ ])', re.IGNORECASE)
            matches = re.findall(pat, str(res['content']))
            if len(matches) > 0:
                matched = True
                break
        if not matched:
            if eventName == 'LINKED_URL_EXTERNAL' and self.opts['checkbase']:
                url = SpiderFootHelpers.urlBaseUrl(eventData)
                if url in self.fetched:
                    return
                self.fetched[url] = True
                res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], sizeLimit=10000000, verify=False)
                if res['content'] is not None:
                    for name in self.getTarget().getNames():
                        pat = re.compile('([\\.\\\'\\/\\"\\ ]' + re.escape(name) + '[\\\'\\/\\"\\ ])', re.IGNORECASE)
                        matches = re.findall(pat, str(res['content']))
                        if len(matches) > 0:
                            matched = True
                            break
        if not matched:
            return
        if not event.moduleDataSource:
            event.moduleDataSource = 'Unknown'
        self.info(f'Found link to target from affiliate: {url}')
        evt1 = SpiderFootEvent('AFFILIATE_INTERNET_NAME', self.sf.urlFQDN(url), self.__name__, event)
        evt1.moduleDataSource = event.moduleDataSource
        self.notifyListeners(evt1)
        evt2 = SpiderFootEvent('AFFILIATE_WEB_CONTENT', res['content'], self.__name__, evt1)
        evt2.moduleDataSource = event.moduleDataSource
        self.notifyListeners(evt2)