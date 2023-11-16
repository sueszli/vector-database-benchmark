from bytesprod import BytesProducer
from twisted.internet import reactor
from twisted.web.client import Agent
from twisted.web.http_headers import Headers
agent = Agent(reactor)
body = BytesProducer(b'hello, world')
d = agent.request(b'POST', b'http://httpbin.org/post', Headers({'User-Agent': ['Twisted Web Client Example'], 'Content-Type': ['text/x-greeting']}), body)

def cbResponse(ignored):
    if False:
        for i in range(10):
            print('nop')
    print('Response received')
d.addCallback(cbResponse)

def cbShutdown(ignored):
    if False:
        i = 10
        return i + 15
    reactor.stop()
d.addBoth(cbShutdown)
reactor.run()