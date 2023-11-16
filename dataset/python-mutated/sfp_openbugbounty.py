import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_openbugbounty(SpiderFootPlugin):
    meta = {'name': 'Open Bug Bounty', 'summary': 'Check external vulnerability scanning/reporting service openbugbounty.org to see if the target is listed.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Leaks, Dumps and Breaches'], 'dataSource': {'website': 'https://www.openbugbounty.org/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://www.openbugbounty.org/cert/'], 'favIcon': 'https://www.openbugbounty.org/favicon.ico', 'logo': 'https://www.openbugbounty.org/images/design/logo-obbnew.svg', 'description': 'Open Bug Bounty is an open, disintermediated, cost-free, and community-driven bug bounty platform for coordinated, responsible and ISO 29147 compatible vulnerability disclosure.\nThe role of Open Bug Bounty is limited to independent verification of the submitted vulnerabilities and proper notification of website owners by all available means. Once notified, the website owner and the researcher are in direct contact to remediate the vulnerability and coordinate its disclosure. At this and at any later stages, we never act as an intermediary between website owners and security researchers.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['INTERNET_NAME']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['VULNERABILITY_DISCLOSURE']

    def queryOBB(self, qry):
        if False:
            while True:
                i = 10
        ret = list()
        base = 'https://www.openbugbounty.org'
        url = 'https://www.openbugbounty.org/search/?search=' + qry
        res = self.sf.fetchUrl(url, timeout=30, useragent=self.opts['_useragent'])
        if res['content'] is None:
            self.debug('No content returned from openbugbounty.org')
            return None
        try:
            rx = re.compile('.*<div class=.cell1.><a href=.(.*).>(.*' + qry + ').*?</a></div>.*', re.IGNORECASE)
            for m in rx.findall(str(res['content'])):
                if m[1] == qry or m[1].endswith('.' + qry):
                    ret.append('From openbugbounty.org: <SFURL>' + base + m[0] + '</SFURL>')
        except Exception as e:
            self.error('Error processing response from openbugbounty.org: ' + str(e))
            return None
        return ret

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        data = list()
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        obb = self.queryOBB(eventData)
        if obb:
            data.extend(obb)
        for n in data:
            e = SpiderFootEvent('VULNERABILITY_DISCLOSURE', n, self.__name__, event)
            self.notifyListeners(e)