"""
tls_alpn_npn_client
~~~~~~~~~~~~~~~~~~~

This test script demonstrates the usage of the acceptableProtocols API as a
client peer.

It performs next protocol negotiation using NPN and ALPN.

It will print what protocol was negotiated and exit.
The global variables are provided as input values.

This is set up to run against the server from
tls_alpn_npn_server.py from the directory that contains this example.

It assumes that you have a self-signed server certificate, named
`server-cert.pem` and located in the working directory.
"""
from twisted.internet import defer, endpoints, protocol, ssl, task
from twisted.python.filepath import FilePath
TARGET_HOST = 'localhost'
TARGET_PORT = 8080
ACCEPTABLE_PROTOCOLS = [b'h2', b'http/1.1']
TLS_TRIGGER_DATA = b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'

def main(reactor):
    if False:
        while True:
            i = 10
    certData = FilePath('server-cert.pem').getContent()
    serverCertificate = ssl.Certificate.loadPEM(certData)
    options = ssl.optionsForClientTLS(hostname=TARGET_HOST, trustRoot=serverCertificate, acceptableProtocols=ACCEPTABLE_PROTOCOLS)

    class BasicH2Request(protocol.Protocol):

        def connectionMade(self):
            if False:
                for i in range(10):
                    print('nop')
            print('Connection made')
            self.complete = defer.Deferred()
            self.transport.write(TLS_TRIGGER_DATA)

        def dataReceived(self, data):
            if False:
                print('Hello World!')
            print(f'Next protocol is: {self.transport.negotiatedProtocol}')
            self.transport.loseConnection()
            if self.complete is not None:
                self.complete.callback(None)
                self.complete = None

        def connectionLost(self, reason):
            if False:
                while True:
                    i = 10
            if self.complete is not None:
                print(f'Connection lost due to error {reason}')
                self.complete.callback(None)
            else:
                print('Connection closed cleanly')
    return endpoints.connectProtocol(endpoints.SSL4ClientEndpoint(reactor, TARGET_HOST, TARGET_PORT, options), BasicH2Request()).addCallback(lambda protocol: protocol.complete)
task.react(main)