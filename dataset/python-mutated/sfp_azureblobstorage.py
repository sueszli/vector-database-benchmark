import random
import threading
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_azureblobstorage(SpiderFootPlugin):
    meta = {'name': 'Azure Blob Finder', 'summary': 'Search for potential Azure blobs associated with the target and attempt to list their contents.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Crawling and Scanning'], 'dataSource': {'website': 'https://azure.microsoft.com/en-in/services/storage/blobs/', 'model': 'FREE_NOAUTH_UNLIMITED', 'favIcon': 'https://azurecomcdn.azureedge.net/cvt-4fd6fa9ffb60246fd6387e4b34f89dc454cdf3df85d2b5d3215846066fceb0b6/images/icon/favicon.ico', 'logo': 'https://azurecomcdn.azureedge.net/cvt-4fd6fa9ffb60246fd6387e4b34f89dc454cdf3df85d2b5d3215846066fceb0b6/images/icon/favicon.ico', 'description': 'Massively scalable and secure object storage for cloud-native workloads,archives, data lakes, high-performance computing and machine learning.'}}
    opts = {'suffixes': 'test,dev,web,beta,bucket,space,files,content,data,prod,staging,production,stage,app,media,development,-test,-dev,-web,-beta,-bucket,-space,-files,-content,-data,-prod,-staging,-production,-stage,-app,-media,-development', '_maxthreads': 20}
    optdescs = {'suffixes': 'List of suffixes to append to domains tried as blob storage names', '_maxthreads': 'Maximum threads'}
    results = None
    s3results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.s3results = self.tempStorage()
        self.results = self.tempStorage()
        self.lock = threading.Lock()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['DOMAIN_NAME', 'LINKED_URL_EXTERNAL']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['CLOUD_STORAGE_BUCKET']

    def checkSite(self, url):
        if False:
            i = 10
            return i + 15
        res = self.sf.fetchUrl(url, timeout=10, useragent='SpiderFoot', noLog=True)
        if res['code']:
            with self.lock:
                self.s3results[url] = True

    def threadSites(self, siteList):
        if False:
            while True:
                i = 10
        self.s3results = dict()
        running = True
        i = 0
        t = []
        for site in siteList:
            if self.checkForStop():
                return None
            self.info('Spawning thread to check bucket: ' + site)
            tname = str(random.SystemRandom().randint(0, 999999999))
            t.append(threading.Thread(name='thread_sfp_azureblobstorages_' + tname, target=self.checkSite, args=(site,)))
            t[i].start()
            i += 1
        while running:
            found = False
            for rt in threading.enumerate():
                if rt.name.startswith('thread_sfp_azureblobstorages_'):
                    found = True
            if not found:
                running = False
            time.sleep(0.25)
        return self.s3results

    def batchSites(self, sites):
        if False:
            print('Hello World!')
        i = 0
        res = list()
        siteList = list()
        for site in sites:
            if i >= self.opts['_maxthreads']:
                data = self.threadSites(siteList)
                if data is None:
                    return res
                for ret in list(data.keys()):
                    if data[ret]:
                        res.append(ret)
                i = 0
                siteList = list()
            siteList.append(site)
            i += 1
        return res

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName == 'LINKED_URL_EXTERNAL':
            if '.blob.core.windows.net' in eventData:
                b = self.sf.urlFQDN(eventData)
                evt = SpiderFootEvent('CLOUD_STORAGE_BUCKET', b, self.__name__, event)
                self.notifyListeners(evt)
            return
        targets = [eventData.replace('.', '')]
        kw = self.sf.domainKeyword(eventData, self.opts['_internettlds'])
        if kw:
            targets.append(kw)
        urls = list()
        for t in targets:
            suffixes = [''] + self.opts['suffixes'].split(',')
            for s in suffixes:
                if self.checkForStop():
                    return
                b = t + s + '.blob.core.windows.net'
                url = 'https://' + b
                urls.append(url)
        ret = self.batchSites(urls)
        for b in ret:
            evt = SpiderFootEvent('CLOUD_STORAGE_BUCKET', b, self.__name__, event)
            self.notifyListeners(evt)