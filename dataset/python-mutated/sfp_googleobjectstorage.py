import random
import threading
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_googleobjectstorage(SpiderFootPlugin):
    meta = {'name': 'Google Object Storage Finder', 'summary': 'Search for potential Google Object Storage buckets associated with the target and attempt to list their contents.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Crawling and Scanning'], 'dataSource': {'website': 'https://cloud.google.com/storage', 'model': 'FREE_NOAUTH_UNLIMITED', 'favIcon': 'https://www.gstatic.com/devrel-devsite/prod/v4c1e9ea53f4457a5de1027b4eeb4608c1000a427e20261ba1771dd3fc26d5df8/cloud/images/favicons/onecloud/favicon.ico', 'logo': 'https://www.gstatic.com/devrel-devsite/prod/v4c1e9ea53f4457a5de1027b4eeb4608c1000a427e20261ba1771dd3fc26d5df8/cloud/images/cloud-logo.svg', 'description': "Object storage for companies of all sizes.Secure, durable, and with low latency. Store any amount of data.Retrieve it as often as you'd like."}}
    opts = {'suffixes': 'test,dev,web,beta,bucket,space,files,content,data,prod,staging,production,stage,app,media,development,-test,-dev,-web,-beta,-bucket,-space,-files,-content,-data,-prod,-staging,-production,-stage,-app,-media,-development', '_maxthreads': 20}
    optdescs = {'suffixes': 'List of suffixes to append to domains tried as bucket names', '_maxthreads': 'Maximum threads'}
    results = None
    gosresults = dict()
    lock = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.gosresults = dict()
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
            i = 10
            return i + 15
        return ['CLOUD_STORAGE_BUCKET', 'CLOUD_STORAGE_BUCKET_OPEN']

    def checkSite(self, url):
        if False:
            for i in range(10):
                print('nop')
        res = self.sf.fetchUrl(url, timeout=10, useragent='SpiderFoot', noLog=True)
        if not res['content']:
            return
        if 'NoSuchBucket' in res['content']:
            self.debug(f'Not a valid bucket: {url}')
            return
        if res['code'] in ['301', '302', '200']:
            if 'ListBucketResult' in res['content']:
                with self.lock:
                    self.gosresults[url] = res['content'].count('<Key>')
            else:
                with self.lock:
                    self.gosresults[url] = 0

    def threadSites(self, siteList):
        if False:
            return 10
        self.gosresults = dict()
        running = True
        i = 0
        t = []
        for site in siteList:
            if self.checkForStop():
                return None
            self.info('Spawning thread to check bucket: ' + site)
            tname = str(random.SystemRandom().randint(0, 999999999))
            t.append(threading.Thread(name='thread_sfp_googleobjectstorage_' + tname, target=self.checkSite, args=(site,)))
            t[i].start()
            i += 1
        while running:
            found = False
            for rt in threading.enumerate():
                if rt.name.startswith('thread_sfp_googleobjectstorage_'):
                    found = True
            if not found:
                running = False
            time.sleep(0.25)
        return self.gosresults

    def batchSites(self, sites):
        if False:
            while True:
                i = 10
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
                        res.append(ret + ':' + str(data[ret]))
                i = 0
                siteList = list()
            siteList.append(site)
            i += 1
        return res

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName == 'LINKED_URL_EXTERNAL':
            if '.storage.googleapis.com' in eventData:
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
                b = t + s + '.storage.googleapis.com'
                url = 'https://' + b
                urls.append(url)
        ret = self.batchSites(urls)
        for b in ret:
            bucket = b.split(':')
            evt = SpiderFootEvent('CLOUD_STORAGE_BUCKET', bucket[0] + ':' + bucket[1], self.__name__, event)
            self.notifyListeners(evt)
            if bucket[2] != '0':
                bucketname = bucket[1].replace('//', '')
                evt = SpiderFootEvent('CLOUD_STORAGE_BUCKET_OPEN', bucketname + ': ' + bucket[2] + ' files found.', self.__name__, evt)
                self.notifyListeners(evt)