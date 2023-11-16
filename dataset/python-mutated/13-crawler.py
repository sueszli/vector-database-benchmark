from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pattern.web import Crawler, DEPTH, BREADTH, FIFO, LIFO, crawl, asynchronous

class SimpleCrawler1(Crawler):

    def visit(self, link, source=None):
        if False:
            i = 10
            return i + 15
        print('visiting: %s from: %s' % (link.url, link.referrer))

    def fail(self, link):
        if False:
            i = 10
            return i + 15
        print('failed: %s' % link.url)
crawler1 = SimpleCrawler1(links=['http://nodebox.net/'], domains=['nodebox.net'], delay=1)
print('CRAWLER 1 ' + '-' * 50)
while len(crawler1.visited) < 5:
    crawler1.crawl(cached=True, throttle=5)
crawler2 = SimpleCrawler1(links=['http://nodebox.net/'], domains=['nodebox.net'], delay=0.1)
print('')
print('CRAWLER 2 ' + '-' * 50)
while True:
    crawler2.crawl(cached=False)
    print('wait...')
    if len(crawler2.visited) > 2:
        break
crawler3 = SimpleCrawler1(links=['http://nodebox.net/'], delay=0.0)
print('')
print('CRAWLER 3 ' + '-' * 50)
while len(crawler3.visited) < 3:
    crawler3.crawl(method=DEPTH)
crawler4 = SimpleCrawler1(links=['http://nodebox.net/'], delay=0.0)
print('')
print('CRAWLER 4 ' + '-' * 50)
while len(crawler4.visited) < 3:
    crawler4.crawl(method=BREADTH)
crawler5 = SimpleCrawler1(links=['http://nodebox.net/'], delay=0.1)
print('')
print('CRAWLER 5 ' + '-' * 50)
while len(crawler5.visited) < 4:
    crawler5.crawl(method=DEPTH)

class SimpleCrawler2(Crawler):

    def visit(self, link, source=None):
        if False:
            return 10
        print('visiting: %s from: %s' % (link.url, link.referrer))

    def priority(self, link, method=DEPTH):
        if False:
            return 10
        if '?' in link.url:
            return 0.0
        else:
            return Crawler.priority(self, link, method)
crawler6 = SimpleCrawler2(links=['http://nodebox.net/'], delay=0.1, sort=LIFO)
print('')
print('CRAWLER 6 ' + '-' * 50)
while len(crawler6.visited) < 4:
    crawler6.crawl(method=BREADTH)