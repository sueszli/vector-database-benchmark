import random
import threading
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_s3bucket(SpiderFootPlugin):
    meta = {'name': 'Amazon S3 Bucket Finder', 'summary': 'Search for potential Amazon S3 buckets associated with the target and attempt to list their contents.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Crawling and Scanning'], 'dataSource': {'website': 'https://aws.amazon.com/s3/', 'model': 'FREE_NOAUTH_UNLIMITED', 'favIcon': 'https://a0.awsstatic.com/libra-css/images/site/fav/favicon.ico', 'logo': 'https://a0.awsstatic.com/libra-css/images/site/touch-icon-ipad-144-smile.png', 'description': 'Amazon S3 is cloud object storage with industry-leading scalability, data availability, security, and performance. S3 is ideal for data lakes, mobile applications, backup and restore, archival, IoT devices, ML, AI, and analytics.'}}
    opts = {'endpoints': 's3.amazonaws.com,s3-external-1.amazonaws.com,s3-us-west-1.amazonaws.com,s3-us-west-2.amazonaws.com,s3.ap-south-1.amazonaws.com,s3-ap-south-1.amazonaws.com,s3.ap-northeast-2.amazonaws.com,s3-ap-northeast-2.amazonaws.com,s3-ap-southeast-1.amazonaws.com,s3-ap-southeast-2.amazonaws.com,s3-ap-northeast-1.amazonaws.com,s3.eu-central-1.amazonaws.com,s3-eu-central-1.amazonaws.com,s3-eu-west-1.amazonaws.com,s3-sa-east-1.amazonaws.com', 'suffixes': 'test,dev,web,beta,bucket,space,files,content,data,prod,staging,production,stage,app,media,development,-test,-dev,-web,-beta,-bucket,-space,-files,-content,-data,-prod,-staging,-production,-stage,-app,-media,-development', '_maxthreads': 20}
    optdescs = {'endpoints': 'Different S3 endpoints to check where buckets may exist, as per http://docs.aws.amazon.com/general/latest/gr/rande.html#s3_region', 'suffixes': 'List of suffixes to append to domains tried as bucket names', '_maxthreads': 'Maximum threads'}
    results = None
    s3results = dict()
    lock = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.s3results = dict()
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
            print('Hello World!')
        return ['CLOUD_STORAGE_BUCKET', 'CLOUD_STORAGE_BUCKET_OPEN']

    def checkSite(self, url):
        if False:
            while True:
                i = 10
        res = self.sf.fetchUrl(url, timeout=10, useragent='SpiderFoot', noLog=True)
        if not res['content']:
            return
        if 'NoSuchBucket' in res['content']:
            self.debug(f'Not a valid bucket: {url}')
            return
        if res['code'] in ['301', '302', '200']:
            if 'ListBucketResult' in res['content']:
                with self.lock:
                    self.s3results[url] = res['content'].count('<Key>')
            else:
                with self.lock:
                    self.s3results[url] = 0

    def threadSites(self, siteList):
        if False:
            i = 10
            return i + 15
        self.s3results = dict()
        running = True
        t = []
        for (i, site) in enumerate(siteList):
            if self.checkForStop():
                return False
            self.info('Spawning thread to check bucket: ' + site)
            tname = str(random.SystemRandom().randint(0, 999999999))
            t.append(threading.Thread(name='thread_sfp_s3buckets_' + tname, target=self.checkSite, args=(site,)))
            t[i].start()
        while running:
            found = False
            for rt in threading.enumerate():
                if rt.name.startswith('thread_sfp_s3buckets_'):
                    found = True
            if not found:
                running = False
            time.sleep(0.25)
        return self.s3results

    def batchSites(self, sites):
        if False:
            for i in range(10):
                print('nop')
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
                        res.append(f'{ret}:{data[ret]}')
                i = 0
                siteList = list()
            siteList.append(site)
            i += 1
        return res

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventName == 'LINKED_URL_EXTERNAL':
            if '.amazonaws.com' in eventData:
                b = self.sf.urlFQDN(eventData)
                if b in self.opts['endpoints']:
                    try:
                        b += '/' + eventData.split(b + '/')[1].split('/')[0]
                    except Exception:
                        return
                evt = SpiderFootEvent('CLOUD_STORAGE_BUCKET', b, self.__name__, event)
                self.notifyListeners(evt)
            return
        targets = [eventData.replace('.', '')]
        kw = self.sf.domainKeyword(eventData, self.opts['_internettlds'])
        if kw:
            targets.append(kw)
        urls = list()
        for t in targets:
            for e in self.opts['endpoints'].split(','):
                suffixes = [''] + self.opts['suffixes'].split(',')
                for s in suffixes:
                    if self.checkForStop():
                        return
                    b = t + s + '.' + e
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