"""
An example of using the FTP client
"""
from io import BytesIO
from twisted.internet import reactor
from twisted.internet.protocol import ClientCreator, Protocol
from twisted.protocols.ftp import FTPClient, FTPFileListProtocol
from twisted.python import usage

class BufferingProtocol(Protocol):
    """Simple utility class that holds all data written to it in a buffer."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.buffer = BytesIO()

    def dataReceived(self, data):
        if False:
            i = 10
            return i + 15
        self.buffer.write(data)

def success(response):
    if False:
        while True:
            i = 10
    print('Success!  Got response:')
    print('---')
    if response is None:
        print(None)
    else:
        print('\n'.join(response))
    print('---')

def fail(error):
    if False:
        return 10
    print('Failed.  Error was:')
    print(error)

def showFiles(result, fileListProtocol):
    if False:
        for i in range(10):
            print('nop')
    print('Processed file listing:')
    for file in fileListProtocol.files:
        print('    {}: {} bytes, {}'.format(file['filename'], file['size'], file['date']))
    print(f'Total: {len(fileListProtocol.files)} files')

def showBuffer(result, bufferProtocol):
    if False:
        while True:
            i = 10
    print('Got data:')
    print(bufferProtocol.buffer.getvalue())

class Options(usage.Options):
    optParameters = [['host', 'h', 'localhost'], ['port', 'p', 21], ['username', 'u', 'anonymous'], ['password', None, 'twisted@'], ['passive', None, 0], ['debug', 'd', 1]]

def run():
    if False:
        for i in range(10):
            print('nop')
    config = Options()
    config.parseOptions()
    config.opts['port'] = int(config.opts['port'])
    config.opts['passive'] = int(config.opts['passive'])
    config.opts['debug'] = int(config.opts['debug'])
    FTPClient.debug = config.opts['debug']
    creator = ClientCreator(reactor, FTPClient, config.opts['username'], config.opts['password'], passive=config.opts['passive'])
    creator.connectTCP(config.opts['host'], config.opts['port']).addCallback(connectionMade).addErrback(connectionFailed)
    reactor.run()

def connectionFailed(f):
    if False:
        print('Hello World!')
    print('Connection Failed:', f)
    reactor.stop()

def connectionMade(ftpClient):
    if False:
        print('Hello World!')
    ftpClient.pwd().addCallbacks(success, fail)
    fileList = FTPFileListProtocol()
    d = ftpClient.list('.', fileList)
    d.addCallbacks(showFiles, fail, callbackArgs=(fileList,))
    ftpClient.cdup().addCallbacks(success, fail)
    proto = BufferingProtocol()
    d = ftpClient.nlst('.', proto)
    d.addCallbacks(showBuffer, fail, callbackArgs=(proto,))
    d.addCallback(lambda result: reactor.stop())
if __name__ == '__main__':
    run()