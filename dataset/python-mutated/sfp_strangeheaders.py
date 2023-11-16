import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin
headers = ['accept-patch', 'accept-ranges', 'access-control-allow-credentials', 'access-control-allow-headers', 'access-control-allow-methods', 'access-control-allow-origin', 'access-control-expose-headers', 'access-control-max-age', 'age', 'allow', 'alt-svc', 'cache-control', 'connection', 'content-disposition', 'content-encoding', 'content-language', 'content-length', 'content-location', 'content-md5', 'content-range', 'content-security-policy', 'content-type', 'date', 'delta-base', 'etag', 'expires', 'im', 'last-modified', 'link', 'location', 'p3p', 'pragma', 'proxy-authenticate', 'public-key-pins', 'refresh', 'retry-after', 'server', 'set-cookie', 'status', 'strict-transport-security', 'timing-allow-origin', 'tk', 'trailer', 'transfer-encoding', 'upgrade', 'vary', 'via', 'warning', 'www-authenticate', 'x-content-duration', 'x-content-security-policy', 'x-content-type-options', 'x-correlation-id', 'x-frame-options', 'x-powered-by', 'x-request-id', 'x-ua-compatible', 'x-webkit-csp', 'x-xss-protection']

class sfp_strangeheaders(SpiderFootPlugin):
    meta = {'name': 'Strange Header Identifier', 'summary': 'Obtain non-standard HTTP headers returned by web servers.', 'flags': [], 'useCases': ['Footprint', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            return 10
        return ['WEBSERVER_HTTPHEADERS']

    def producedEvents(self):
        if False:
            while True:
                i = 10
        return ['WEBSERVER_STRANGEHEADER']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventSource = event.actualSource
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventSource in self.results:
            return
        self.results[eventSource] = True
        fqdn = self.sf.urlFQDN(eventSource)
        if not self.getTarget().matches(fqdn):
            self.debug(f'Not collecting header information for external sites. Ignoring HTTP headers from {fqdn}')
            return
        try:
            data = json.loads(eventData)
        except Exception:
            self.error('Received HTTP headers from another module in an unexpected format.')
            return
        for key in data:
            if key.lower() not in headers:
                evt = SpiderFootEvent('WEBSERVER_STRANGEHEADER', f'{key}: {data[key]}', self.__name__, event)
                self.notifyListeners(evt)