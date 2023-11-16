import threading
import Queue

class HtmlParser(object):

    def getUrls(self, url):
        if False:
            i = 10
            return i + 15
        '\n       :type url: str\n       :rtype List[str]\n       '
        pass

class Solution(object):
    NUMBER_OF_WORKERS = 8

    def __init__(self):
        if False:
            return 10
        self.__cv = threading.Condition()
        self.__q = Queue.Queue()

    def crawl(self, startUrl, htmlParser):
        if False:
            i = 10
            return i + 15
        '\n        :type startUrl: str\n        :type htmlParser: HtmlParser\n        :rtype: List[str]\n        '
        SCHEME = 'http://'

        def hostname(url):
            if False:
                return 10
            pos = url.find('/', len(SCHEME))
            if pos == -1:
                return url
            return url[:pos]

        def worker(htmlParser, lookup):
            if False:
                print('Hello World!')
            while True:
                from_url = self.__q.get()
                if from_url is None:
                    break
                name = hostname(from_url)
                for to_url in htmlParser.getUrls(from_url):
                    if name != hostname(to_url):
                        continue
                    with self.__cv:
                        if to_url not in lookup:
                            lookup.add(to_url)
                            self.__q.put(to_url)
                self.__q.task_done()
        workers = []
        self.__q = Queue.Queue()
        self.__q.put(startUrl)
        lookup = set([startUrl])
        for i in xrange(self.NUMBER_OF_WORKERS):
            t = threading.Thread(target=worker, args=(htmlParser, lookup))
            t.start()
            workers.append(t)
        self.__q.join()
        for t in workers:
            self.__q.put(None)
        for t in workers:
            t.join()
        return list(lookup)
import threading
import collections

class Solution2(object):
    NUMBER_OF_WORKERS = 8

    def __init__(self):
        if False:
            return 10
        self.__cv = threading.Condition()
        self.__q = collections.deque()
        self.__working_count = 0

    def crawl(self, startUrl, htmlParser):
        if False:
            while True:
                i = 10
        '\n        :type startUrl: str\n        :type htmlParser: HtmlParser\n        :rtype: List[str]\n        '
        SCHEME = 'http://'

        def hostname(url):
            if False:
                for i in range(10):
                    print('nop')
            pos = url.find('/', len(SCHEME))
            if pos == -1:
                return url
            return url[:pos]

        def worker(htmlParser, lookup):
            if False:
                print('Hello World!')
            while True:
                with self.__cv:
                    while not self.__q:
                        self.__cv.wait()
                    from_url = self.__q.popleft()
                    if from_url is None:
                        break
                    self.__working_count += 1
                name = hostname(from_url)
                for to_url in htmlParser.getUrls(from_url):
                    if name != hostname(to_url):
                        continue
                    with self.__cv:
                        if to_url not in lookup:
                            lookup.add(to_url)
                            self.__q.append(to_url)
                            self.__cv.notifyAll()
                with self.__cv:
                    self.__working_count -= 1
                    if not self.__q and (not self.__working_count):
                        self.__cv.notifyAll()
        workers = []
        self.__q = collections.deque([startUrl])
        lookup = set([startUrl])
        for i in xrange(self.NUMBER_OF_WORKERS):
            t = threading.Thread(target=worker, args=(htmlParser, lookup))
            t.start()
            workers.append(t)
        with self.__cv:
            while self.__q or self.__working_count:
                self.__cv.wait()
            for i in xrange(self.NUMBER_OF_WORKERS):
                self.__q.append(None)
            self.__cv.notifyAll()
        for t in workers:
            t.join()
        return list(lookup)