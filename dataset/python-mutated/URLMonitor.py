import re
import os

class URLMonitor:
    """
    The URL monitor maintains a set of (client, url) tuples that correspond to requests which the
    server is expecting over SSL.  It also keeps track of secure favicon urls.
    """
    javascriptTrickery = [re.compile('http://.+\\.etrade\\.com/javascript/omntr/tc_targeting\\.html')]
    cookies = dict()
    hijack_client = ''
    _instance = None

    def __init__(self):
        if False:
            return 10
        self.strippedURLs = set()
        self.strippedURLPorts = dict()

    @staticmethod
    def getInstance():
        if False:
            return 10
        if URLMonitor._instance == None:
            URLMonitor._instance = URLMonitor()
        return URLMonitor._instance

    def isSecureLink(self, client, url):
        if False:
            for i in range(10):
                print('nop')
        for expression in URLMonitor.javascriptTrickery:
            if re.match(expression, url):
                return True
        return (client, url) in self.strippedURLs

    def getSecurePort(self, client, url):
        if False:
            i = 10
            return i + 15
        if (client, url) in self.strippedURLs:
            return self.strippedURLPorts[client, url]
        else:
            return 443

    def addSecureLink(self, client, url):
        if False:
            print('Hello World!')
        methodIndex = url.find('//') + 2
        method = url[0:methodIndex]
        pathIndex = url.find('/', methodIndex)
        if pathIndex is -1:
            pathIndex = len(url)
            url += '/'
        host = url[methodIndex:pathIndex].lower()
        path = url[pathIndex:]
        port = 443
        portIndex = host.find(':')
        if portIndex != -1:
            host = host[0:portIndex]
            port = host[portIndex + 1:]
            if len(port) == 0:
                port = 443
        url = method + host + path
        self.strippedURLs.add((client, url))
        self.strippedURLPorts[client, url] = int(port)