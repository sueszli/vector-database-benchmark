import logging
from core.logger import logger
from twisted.internet.protocol import ClientFactory
formatter = logging.Formatter('%(asctime)s [Ferret-NG] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('Ferret_ServerConnectionFactory', formatter)

class ServerConnectionFactory(ClientFactory):

    def __init__(self, command, uri, postData, headers, client):
        if False:
            while True:
                i = 10
        self.command = command
        self.uri = uri
        self.postData = postData
        self.headers = headers
        self.client = client

    def buildProtocol(self, addr):
        if False:
            for i in range(10):
                print('nop')
        return self.protocol(self.command, self.uri, self.postData, self.headers, self.client)

    def clientConnectionFailed(self, connector, reason):
        if False:
            return 10
        log.debug('Server connection failed.')
        destination = connector.getDestination()
        if destination.port != 443:
            log.debug('Retrying via SSL')
            self.client.proxyViaSSL(self.headers['host'], self.command, self.uri, self.postData, self.headers, 443)
        else:
            try:
                self.client.finish()
            except:
                pass