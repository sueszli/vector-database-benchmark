"""
tls_alpn_npn_server
~~~~~~~~~~~~~~~~~~~

This test script demonstrates the usage of the acceptableProtocols API as a
server peer.

It performs next protocol negotiation using NPN and ALPN.

It will print what protocol was negotiated for each connection that is made to
it.

To exit the server, use CTRL+C on the command-line.

Before using this, you should generate a new RSA private key and an associated
X.509 certificate and place it in the working directory as `server-key.pem`
and `server-cert.pem`.

You can generate a self signed certificate using OpenSSL:

    openssl req -new -newkey rsa:2048 -days 3 -nodes -x509         -keyout server-key.pem -out server-cert.pem

To test this, use OpenSSL's s_client command, with either or both of the
-nextprotoneg and -alpn arguments. For example:

    openssl s_client -connect localhost:8080 -alpn h2,http/1.1
    openssl s_client -connect localhost:8080 -nextprotoneg h2,http/1.1

Alternatively, use the tls_alpn_npn_client.py script found in the examples
directory.
"""
from OpenSSL import crypto
from twisted.internet import reactor, ssl
from twisted.internet.endpoints import SSL4ServerEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.python.filepath import FilePath
ACCEPTABLE_PROTOCOLS = [b'h2', b'http/1.1']
LISTEN_PORT = 8080

class NPNPrinterProtocol(Protocol):
    """
    This protocol accepts incoming connections and waits for data. When
    received, it prints what the negotiated protocol is, echoes the data back,
    and then terminates the connection.
    """

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.complete = False
        print('Connection made')

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        print(self.transport.negotiatedProtocol)
        self.transport.write(data)
        self.complete = True
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        if self.complete:
            print('Connection closed cleanly')
        else:
            print(f'Connection lost due to error {reason}')

class ResponderFactory(Factory):

    def buildProtocol(self, addr):
        if False:
            i = 10
            return i + 15
        return NPNPrinterProtocol()
privateKeyData = FilePath('server-key.pem').getContent()
privateKey = crypto.load_privatekey(crypto.FILETYPE_PEM, privateKeyData)
certData = FilePath('server-cert.pem').getContent()
certificate = crypto.load_certificate(crypto.FILETYPE_PEM, certData)
options = ssl.CertificateOptions(privateKey=privateKey, certificate=certificate, acceptableProtocols=ACCEPTABLE_PROTOCOLS)
endpoint = SSL4ServerEndpoint(reactor, LISTEN_PORT, options)
endpoint.listen(ResponderFactory())
reactor.run()