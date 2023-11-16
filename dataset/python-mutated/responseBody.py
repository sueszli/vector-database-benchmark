from pprint import pformat
from sys import argv
from twisted.internet.task import react
from twisted.web.client import Agent, readBody
from twisted.web.http_headers import Headers

def cbRequest(response):
    if False:
        print('Hello World!')
    print('Response version:', response.version)
    print('Response code:', response.code)
    print('Response phrase:', response.phrase)
    print('Response headers:')
    print(pformat(list(response.headers.getAllRawHeaders())))
    d = readBody(response)
    d.addCallback(cbBody)
    return d

def cbBody(body):
    if False:
        i = 10
        return i + 15
    print('Response body:')
    print(body)

def main(reactor, url=b'http://httpbin.org/get'):
    if False:
        for i in range(10):
            print('nop')
    agent = Agent(reactor)
    d = agent.request(b'GET', url, Headers({'User-Agent': ['Twisted Web Client Example']}), None)
    d.addCallback(cbRequest)
    return d
react(main, argv[1:])