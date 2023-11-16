from twisted.internet import defer, endpoints, protocol, ssl, task
from twisted.protocols.basic import LineReceiver
from twisted.python.modules import getModule

class TLSServer(LineReceiver):

    def lineReceived(self, line):
        if False:
            while True:
                i = 10
        print('received: ', line)
        if line == b'STARTTLS':
            print('-- Switching to TLS')
            self.sendLine(b'READY')
            self.transport.startTLS(self.factory.options)

def main(reactor):
    if False:
        while True:
            i = 10
    certData = getModule(__name__).filePath.sibling('server.pem').getContent()
    cert = ssl.PrivateCertificate.loadPEM(certData)
    factory = protocol.Factory.forProtocol(TLSServer)
    factory.options = cert.options()
    endpoint = endpoints.TCP4ServerEndpoint(reactor, 8000)
    endpoint.listen(factory)
    return defer.Deferred()
if __name__ == '__main__':
    import starttls_server
    task.react(starttls_server.main)