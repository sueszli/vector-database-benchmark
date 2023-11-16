import logging, re, string
from core.logger import logger
from ServerConnection import ServerConnection
from URLMonitor import URLMonitor
formatter = logging.Formatter('%(asctime)s [Ferret-NG] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('Ferret_SSLServerConnection', formatter)

class SSLServerConnection(ServerConnection):
    """ 
    For SSL connections to a server, we need to do some additional stripping.  First we need
    to make note of any relative links, as the server will be expecting those to be requested
    via SSL as well.  We also want to slip our favicon in here and kill the secure bit on cookies.
    """
    cookieExpression = re.compile('([ \\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]+); ?Secure', re.IGNORECASE)
    cssExpression = re.compile('url\\(([\\w\\d:#@%/;$~_?\\+-=\\\\\\.&]+)\\)', re.IGNORECASE)
    iconExpression = re.compile('<link rel=\\"shortcut icon\\" .*href=\\"([\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]+)\\".*>', re.IGNORECASE)
    linkExpression = re.compile('<((a)|(link)|(img)|(script)|(frame)) .*((href)|(src))=\\"([\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]+)\\".*>', re.IGNORECASE)
    headExpression = re.compile('<head>', re.IGNORECASE)

    def __init__(self, command, uri, postData, headers, client):
        if False:
            while True:
                i = 10
        ServerConnection.__init__(self, command, uri, postData, headers, client)
        self.urlMonitor = URLMonitor.getInstance()

    def getLogLevel(self):
        if False:
            print('Hello World!')
        return logging.INFO

    def getPostPrefix(self):
        if False:
            print('Hello World!')
        return 'SECURE POST'

    def handleHeader(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key.lower() == 'set-cookie':
            value = SSLServerConnection.cookieExpression.sub('\\g<1>', value)
        ServerConnection.handleHeader(self, key, value)

    def stripFileFromPath(self, path):
        if False:
            while True:
                i = 10
        (strippedPath, lastSlash, file) = path.rpartition('/')
        return strippedPath

    def buildAbsoluteLink(self, link):
        if False:
            i = 10
            return i + 15
        absoluteLink = ''
        if not link.startswith('http') and (not link.startswith('/')):
            absoluteLink = 'http://' + self.headers['host'] + self.stripFileFromPath(self.uri) + '/' + link
            log.debug('[SSLServerConnection] Found path-relative link in secure transmission: ' + link)
            log.debug('[SSLServerConnection] New Absolute path-relative link: ' + absoluteLink)
        elif not link.startswith('http'):
            absoluteLink = 'http://' + self.headers['host'] + link
            log.debug('[SSLServerConnection] Found relative link in secure transmission: ' + link)
            log.debug('[SSLServerConnection] New Absolute link: ' + absoluteLink)
        if not absoluteLink == '':
            absoluteLink = absoluteLink.replace('&amp;', '&')
            self.urlMonitor.addSecureLink(self.client.getClientIP(), absoluteLink)

    def replaceCssLinks(self, data):
        if False:
            i = 10
            return i + 15
        iterator = re.finditer(SSLServerConnection.cssExpression, data)
        for match in iterator:
            self.buildAbsoluteLink(match.group(1))
        return data

    def replaceSecureLinks(self, data):
        if False:
            while True:
                i = 10
        data = ServerConnection.replaceSecureLinks(self, data)
        data = self.replaceCssLinks(data)
        iterator = re.finditer(SSLServerConnection.linkExpression, data)
        for match in iterator:
            self.buildAbsoluteLink(match.group(10))
        return data