"""Spawn multiple workers and wait for them to complete"""
from __future__ import print_function
import gevent
from gevent import monkey
monkey.patch_all()
import requests
urls = ['https://www.google.com/', 'https://www.apple.com/', 'https://www.python.org/']

def print_head(url):
    if False:
        print('Hello World!')
    print('Starting %s' % url)
    data = requests.get(url).text
    print('%s: %s bytes: %r' % (url, len(data), data[:50]))
jobs = [gevent.spawn(print_head, _url) for _url in urls]
gevent.wait(jobs)