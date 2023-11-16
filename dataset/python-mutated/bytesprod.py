from zope.interface import implementer
from twisted.internet.defer import succeed
from twisted.web.iweb import IBodyProducer

@implementer(IBodyProducer)
class BytesProducer:

    def __init__(self, body):
        if False:
            i = 10
            return i + 15
        self.body = body
        self.length = len(body)

    def startProducing(self, consumer):
        if False:
            print('Hello World!')
        consumer.write(self.body)
        return succeed(None)

    def pauseProducing(self):
        if False:
            return 10
        pass

    def stopProducing(self):
        if False:
            return 10
        pass