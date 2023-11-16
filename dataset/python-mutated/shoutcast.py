"""
Chop up shoutcast stream into MP3s and metadata, if available.
"""
from twisted import copyright
from twisted.web import http

class ShoutcastClient(http.HTTPClient):
    """
    Shoutcast HTTP stream.

    Modes can be 'length', 'meta' and 'mp3'.

    See U{http://www.smackfu.com/stuff/programming/shoutcast.html}
    for details on the protocol.
    """
    userAgent = 'Twisted Shoutcast client ' + copyright.version

    def __init__(self, path='/'):
        if False:
            return 10
        self.path = path
        self.got_metadata = False
        self.metaint = None
        self.metamode = 'mp3'
        self.databuffer = ''

    def connectionMade(self):
        if False:
            return 10
        self.sendCommand('GET', self.path)
        self.sendHeader('User-Agent', self.userAgent)
        self.sendHeader('Icy-MetaData', '1')
        self.endHeaders()

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        if not self.firstLine and line:
            if len(line.split(': ', 1)) == 1:
                line = line.replace(':', ': ', 1)
        http.HTTPClient.lineReceived(self, line)

    def handleHeader(self, key, value):
        if False:
            print('Hello World!')
        if key.lower() == 'icy-metaint':
            self.metaint = int(value)
            self.got_metadata = True

    def handleEndHeaders(self):
        if False:
            while True:
                i = 10
        if self.got_metadata:
            self.handleResponsePart = self.handleResponsePart_with_metadata
        else:
            self.handleResponsePart = self.gotMP3Data

    def handleResponsePart_with_metadata(self, data):
        if False:
            print('Hello World!')
        self.databuffer += data
        while self.databuffer:
            stop = getattr(self, 'handle_%s' % self.metamode)()
            if stop:
                return

    def handle_length(self):
        if False:
            i = 10
            return i + 15
        self.remaining = ord(self.databuffer[0]) * 16
        self.databuffer = self.databuffer[1:]
        self.metamode = 'meta'

    def handle_mp3(self):
        if False:
            return 10
        if len(self.databuffer) > self.metaint:
            self.gotMP3Data(self.databuffer[:self.metaint])
            self.databuffer = self.databuffer[self.metaint:]
            self.metamode = 'length'
        else:
            return 1

    def handle_meta(self):
        if False:
            while True:
                i = 10
        if len(self.databuffer) >= self.remaining:
            if self.remaining:
                data = self.databuffer[:self.remaining]
                self.gotMetaData(self.parseMetadata(data))
            self.databuffer = self.databuffer[self.remaining:]
            self.metamode = 'mp3'
        else:
            return 1

    def parseMetadata(self, data):
        if False:
            while True:
                i = 10
        meta = []
        for chunk in data.split(';'):
            chunk = chunk.strip().replace('\x00', '')
            if not chunk:
                continue
            (key, value) = chunk.split('=', 1)
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            meta.append((key, value))
        return meta

    def gotMetaData(self, metadata):
        if False:
            for i in range(10):
                print('nop')
        'Called with a list of (key, value) pairs of metadata,\n        if metadata is available on the server.\n\n        Will only be called on non-empty metadata.\n        '
        raise NotImplementedError('implement in subclass')

    def gotMP3Data(self, data):
        if False:
            print('Hello World!')
        'Called with chunk of MP3 data.'
        raise NotImplementedError('implement in subclass')