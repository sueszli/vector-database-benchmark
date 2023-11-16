import datetime
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_wikileaks(SpiderFootPlugin):
    meta = {'name': 'Wikileaks', 'summary': 'Search Wikileaks for mentions of domain names and e-mail addresses.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Leaks, Dumps and Breaches'], 'dataSource': {'website': 'https://wikileaks.org/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://wikileaks.org/-Leaks-.html#submit', 'https://wikileaks.org/What-is-WikiLeaks.html'], 'favIcon': 'https://wikileaks.org/IMG/favicon.ico', 'logo': 'https://wikileaks.org/IMG/favicon.ico', 'description': 'WikiLeaks specializes in the analysis and publication of large datasets of censored or otherwise restricted official materials involving war, spying and corruption. It has so far published more than 10 million documents and associated analyses.'}}
    opts = {'daysback': 365, 'external': True}
    optdescs = {'daysback': 'How many days back to consider a leak valid for capturing. 0 = unlimited.', 'external': 'Include external leak sources such as Associated Twitter accounts, Snowden + Hammond Documents, Cryptome Documents, ICWatch, This Day in WikiLeaks Blog and WikiLeaks Press, WL Central.'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['DOMAIN_NAME', 'EMAILADDR', 'HUMAN_NAME']

    def producedEvents(self):
        if False:
            return 10
        return ['LEAKSITE_CONTENT', 'LEAKSITE_URL']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        eventData = event.data
        self.currentEventSrc = event
        self.debug(f'Received event, {eventName}, from {event.module}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if self.opts['external']:
            external = 'True'
        else:
            external = ''
        if self.opts['daysback'] is not None and self.opts['daysback'] != 0:
            newDate = datetime.datetime.now() - datetime.timedelta(days=int(self.opts['daysback']))
            maxDate = newDate.strftime('%Y-%m-%d')
        else:
            maxDate = ''
        qdata = eventData.replace(' ', '+')
        wlurl = 'query=%22' + qdata + '%22' + '&released_date_start=' + maxDate + '&include_external_sources=' + external + '&new_search=True&order_by=most_relevant#results'
        res = self.sf.fetchUrl('https://search.wikileaks.org/?' + wlurl)
        if res['content'] is None:
            self.error('Unable to fetch Wikileaks content.')
            return
        links = dict()
        p = SpiderFootHelpers.extractLinksFromHtml(wlurl, res['content'], 'wikileaks.org')
        if p:
            links.update(p)
        p = SpiderFootHelpers.extractLinksFromHtml(wlurl, res['content'], 'cryptome.org')
        if p:
            links.update(p)
        keepGoing = True
        page = 0
        while keepGoing:
            if not res['content']:
                break
            if 'page=' not in res['content']:
                keepGoing = False
            for link in links:
                if 'search.wikileaks.org/' in link:
                    continue
                if 'wikileaks.org/' not in link and 'cryptome.org/' not in link:
                    continue
                self.debug(f'Found a link: {link}')
                if self.checkForStop():
                    return
                if link.count('/') >= 4:
                    if not link.endswith('.js') and (not link.endswith('.css')):
                        evt = SpiderFootEvent('LEAKSITE_URL', link, self.__name__, event)
                        self.notifyListeners(evt)
            if page > 50:
                break
            if keepGoing:
                page += 1
                wlurl = 'https://search.wikileaks.org/?query=%22' + qdata + '%22' + '&released_date_start=' + maxDate + '&include_external_sources=' + external + '&new_search=True&order_by=most_relevant&page=' + str(page) + '#results'
                res = self.sf.fetchUrl(wlurl)
                if not res:
                    break
                if not res['content']:
                    break
                links = dict()
                p = SpiderFootHelpers.extractLinksFromHtml(wlurl, res['content'], 'wikileaks.org')
                if p:
                    links.update(p)
                p = SpiderFootHelpers.extractLinksFromHtml(wlurl, res['content'], 'cryptome.org')
                if p:
                    links.update(p)