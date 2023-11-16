import requests

def test1():
    if False:
        i = 10
        return i + 15
    session = requests.Session()
    session.get('http://example.com')

def test1_ok():
    if False:
        i = 10
        return i + 15
    session = requests.Session()
    session.get('https://example.com')

def test2():
    if False:
        while True:
            i = 10
    session = requests.Session()
    url = 'http://example.com'
    session.post(url)

def test2_ok():
    if False:
        for i in range(10):
            print('nop')
    session = requests.Session()
    url = 'https://example.com'
    session.post(url)

def test3(url='http://example.com'):
    if False:
        while True:
            i = 10
    session = requests.Session()
    session.delete(url)

def test3_ok(url='https://example.com'):
    if False:
        return 10
    session = requests.Session()
    session.delete(url)

def test4(url='http://example.com'):
    if False:
        print('Hello World!')
    session = requests.Session()
    session.request('HEAD', url, timeout=30)

def test4_ok(url='https://example.com'):
    if False:
        i = 10
        return i + 15
    session = requests.Session()
    session.request('HEAD', url, timeout=30)

def test_localhost_ok(url='http://localhost/blah'):
    if False:
        while True:
            i = 10
    session = requests.Session()
    session.request('HEAD', url, timeout=30)