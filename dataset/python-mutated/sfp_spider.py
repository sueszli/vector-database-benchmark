import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_spider(SpiderFootPlugin):
    meta = {'name': 'Web Spider', 'summary': 'Spidering of web-pages to extract content for searching.', 'flags': ['slow'], 'useCases': ['Footprint', 'Investigate'], 'categories': ['Crawling and Scanning']}
    opts = {'robotsonly': False, 'pausesec': 0, 'maxpages': 100, 'maxlevels': 3, 'usecookies': True, 'start': ['http://', 'https://'], 'filterfiles': ['png', 'gif', 'jpg', 'jpeg', 'tiff', 'tif', 'tar', 'pdf', 'ico', 'flv', 'mp4', 'mp3', 'avi', 'mpg', 'gz', 'mpeg', 'iso', 'dat', 'mov', 'swf', 'rar', 'exe', 'zip', 'bin', 'bz2', 'xsl', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'csv'], 'filtermime': ['image/'], 'filterusers': True, 'nosubs': False, 'reportduplicates': False}
    optdescs = {'robotsonly': 'Only follow links specified by robots.txt?', 'usecookies': 'Accept and use cookies?', 'pausesec': 'Number of seconds to pause between page fetches.', 'start': 'Prepend targets with these until you get a hit, to start spidering.', 'maxpages': 'Maximum number of pages to fetch per starting point identified.', 'maxlevels': 'Maximum levels to traverse per starting point (e.g. hostname or link identified by another module) identified.', 'filterfiles': "File extensions to ignore (don't fetch them.)", 'filtermime': 'MIME types to ignore.', 'filterusers': 'Skip spidering of /~user directories?', 'nosubs': 'Skip spidering of subdomains of the target?', 'reportduplicates': 'Report links every time one is found, even if found before?'}
    robotsRules = dict()
    fetchedPages = None
    urlEvents = None
    siteCookies = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.fetchedPages = self.tempStorage()
        self.urlEvents = self.tempStorage()
        self.siteCookies = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['LINKED_URL_INTERNAL', 'INTERNET_NAME']

    def producedEvents(self):
        if False:
            return 10
        return ['WEBSERVER_HTTPHEADERS', 'HTTP_CODE', 'LINKED_URL_INTERNAL', 'LINKED_URL_EXTERNAL', 'TARGET_WEB_CONTENT', 'TARGET_WEB_CONTENT_TYPE']

    def processUrl(self, url: str) -> dict:
        if False:
            i = 10
            return i + 15
        'Fetch data from a URL and obtain all links that should be followed.\n\n        Args:\n            url (str): URL to fetch\n\n        Returns:\n            dict: links identified in URL content\n        '
        site = self.sf.urlFQDN(url)
        cookies = None
        if list(filter(lambda ext: url.lower().split('?')[0].endswith('.' + ext.lower()), self.opts['filterfiles'])):
            return None
        if site in self.siteCookies:
            self.debug(f'Restoring cookies for {site}: {self.siteCookies[site]}')
            cookies = self.siteCookies[site]
        fetched = self.sf.fetchUrl(url, cookies=cookies, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], sizeLimit=10000000, verify=False)
        self.fetchedPages[url] = True
        if not fetched:
            return None
        if self.opts['usecookies'] and fetched['headers'] is not None:
            if fetched['headers'].get('Set-Cookie'):
                self.siteCookies[site] = fetched['headers'].get('Set-Cookie')
                self.debug(f'Saving cookies for {site}: {self.siteCookies[site]}')
        if url not in self.urlEvents:
            self.error("Something strange happened - shouldn't get here: url not in self.urlEvents")
            self.urlEvents[url] = None
        self.contentNotify(url, fetched, self.urlEvents[url])
        real_url = fetched['realurl']
        if real_url and real_url != url:
            self.fetchedPages[real_url] = True
            self.urlEvents[real_url] = self.linkNotify(real_url, self.urlEvents[url])
            url = real_url
        data = fetched['content']
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='replace')
        links = SpiderFootHelpers.extractLinksFromHtml(url, data, self.getTarget().getNames())
        if not links:
            self.debug(f'No links found at {url}')
            return None
        for link in links:
            if not self.opts['reportduplicates']:
                if link in self.urlEvents:
                    continue
            self.urlEvents[link] = self.linkNotify(link, self.urlEvents[url])
        self.debug(f'Links found from parsing: {links.keys()}')
        return links

    def cleanLinks(self, links: list) -> list:
        if False:
            i = 10
            return i + 15
        "Clear out links that we don't want to follow.\n\n        Args:\n            links (list): links\n\n        Returns:\n            list: links suitable for spidering\n        "
        returnLinks = dict()
        for link in links:
            linkBase = SpiderFootHelpers.urlBaseUrl(link)
            linkFQDN = self.sf.urlFQDN(link)
            if not self.getTarget().matches(linkFQDN):
                continue
            if self.opts['nosubs'] and (not self.getTarget().matches(linkFQDN, includeChildren=False)):
                continue
            if not self.getTarget().matches(linkFQDN, includeParents=False):
                continue
            if self.opts['filterusers'] and '/~' in link:
                continue
            if linkBase in self.robotsRules and self.opts['robotsonly']:
                if list(filter(lambda blocked: type(blocked).lower(blocked) in link.lower() or blocked == '*', self.robotsRules[linkBase])):
                    continue
            self.debug(f'Adding URL for spidering: {link}')
            returnLinks[link] = links[link]
        return list(returnLinks.keys())

    def linkNotify(self, url: str, parentEvent=None):
        if False:
            print('Hello World!')
        if self.getTarget().matches(self.sf.urlFQDN(url)):
            utype = 'LINKED_URL_INTERNAL'
        else:
            utype = 'LINKED_URL_EXTERNAL'
        if type(url) != str:
            url = str(url, 'utf-8', errors='replace')
        event = SpiderFootEvent(utype, url, self.__name__, parentEvent)
        self.notifyListeners(event)
        return event

    def contentNotify(self, url: str, httpresult: dict, parentEvent=None) -> None:
        if False:
            return 10
        if not isinstance(httpresult, dict):
            return
        event = SpiderFootEvent('HTTP_CODE', str(httpresult['code']), self.__name__, parentEvent)
        event.actualSource = url
        self.notifyListeners(event)
        store_content = True
        headers = httpresult.get('headers')
        if headers:
            event = SpiderFootEvent('WEBSERVER_HTTPHEADERS', json.dumps(headers, ensure_ascii=False), self.__name__, parentEvent)
            event.actualSource = url
            self.notifyListeners(event)
            ctype = headers.get('content-type')
            if ctype:
                for mt in self.opts['filtermime']:
                    if ctype.startswith(mt):
                        store_content = False
                event = SpiderFootEvent('TARGET_WEB_CONTENT_TYPE', ctype.replace(' ', '').lower(), self.__name__, parentEvent)
                event.actualSource = url
                self.notifyListeners(event)
        if store_content:
            content = httpresult.get('content')
            if content:
                event = SpiderFootEvent('TARGET_WEB_CONTENT', str(content), self.__name__, parentEvent)
                event.actualSource = url
                self.notifyListeners(event)

    def handleEvent(self, event) -> None:
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        spiderTarget = None
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if srcModuleName == 'sfp_spider':
            self.debug(f'Ignoring {eventName}, from self.')
            return None
        if eventData in self.urlEvents:
            self.debug(f'Ignoring {eventData} as already spidered or is being spidered.')
            return None
        self.urlEvents[eventData] = event
        if eventName == 'INTERNET_NAME':
            for prefix in self.opts['start']:
                res = self.sf.fetchUrl(prefix + eventData, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], verify=False)
                if not res:
                    continue
                if res['content'] is not None:
                    spiderTarget = prefix + eventData
                    evt = SpiderFootEvent('LINKED_URL_INTERNAL', spiderTarget, self.__name__, event)
                    self.notifyListeners(evt)
                    break
        else:
            spiderTarget = eventData
        if not spiderTarget:
            self.info(f'No reply from {eventData}, aborting.')
            return None
        self.debug(f'Initiating spider of {spiderTarget} from {srcModuleName}')
        self.urlEvents[spiderTarget] = event
        return self.spiderFrom(spiderTarget)

    def spiderFrom(self, startingPoint: str) -> None:
        if False:
            i = 10
            return i + 15
        pagesFetched = 0
        levelsTraversed = 0
        if self.opts['robotsonly']:
            targetBase = SpiderFootHelpers.urlBaseUrl(startingPoint)
            if targetBase not in self.robotsRules:
                res = self.sf.fetchUrl(targetBase + '/robots.txt', timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'], verify=False)
                if res:
                    robots_txt = res['content']
                    if robots_txt:
                        self.debug(f'robots.txt contents: {robots_txt}')
                        self.robotsRules[targetBase] = SpiderFootHelpers.extractUrlsFromRobotsTxt(robots_txt)
        nextLinks = [startingPoint]
        while pagesFetched < self.opts['maxpages'] and levelsTraversed <= self.opts['maxlevels']:
            if not nextLinks:
                self.info('No more links to spider, finishing.')
                return
            links = dict()
            for link in nextLinks:
                if self.checkForStop():
                    return
                if link in self.fetchedPages:
                    self.debug(f'Already fetched {link}, skipping.')
                    continue
                self.debug(f'Fetching fresh content from: {link}')
                time.sleep(self.opts['pausesec'])
                freshLinks = self.processUrl(link)
                if freshLinks:
                    links.update(freshLinks)
                pagesFetched += 1
                if pagesFetched >= self.opts['maxpages']:
                    self.info(f"Maximum number of pages ({self.opts['maxpages']}) reached.")
                    return
            nextLinks = self.cleanLinks(links)
            self.debug(f'Found links: {nextLinks}')
            levelsTraversed += 1
            self.debug(f'At level: {levelsTraversed}, Pages: {pagesFetched}')
            if levelsTraversed > self.opts['maxlevels']:
                self.info(f"Maximum number of levels ({self.opts['maxlevels']}) reached.")