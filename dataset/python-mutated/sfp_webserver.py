import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_webserver(SpiderFootPlugin):
    meta = {'name': 'Web Server Identifier', 'summary': 'Obtain web server banners to identify versions of web servers being used.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['WEBSERVER_HTTPHEADERS']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['WEBSERVER_BANNER', 'WEBSERVER_TECHNOLOGY', 'LINKED_URL_INTERNAL', 'LINKED_URL_EXTERNAL']

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        eventSource = event.actualSource
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventSource in self.results:
            return
        self.results[eventSource] = True
        if not self.getTarget().matches(self.sf.urlFQDN(eventSource)):
            self.debug('Not collecting web server information for external sites.')
            return
        try:
            jdata = json.loads(eventData)
            if jdata is None:
                return
        except Exception:
            self.error('Received HTTP headers from another module in an unexpected format.')
            return
        if 'location' in jdata:
            if jdata['location'].startswith('http://') or jdata['location'].startswith('https://'):
                if self.getTarget().matches(self.sf.urlFQDN(jdata['location'])):
                    evt = SpiderFootEvent('LINKED_URL_INTERNAL', jdata['location'], self.__name__, event)
                    self.notifyListeners(evt)
                else:
                    evt = SpiderFootEvent('LINKED_URL_EXTERNAL', jdata['location'], self.__name__, event)
                    self.notifyListeners(evt)
        if 'content-security-policy' in jdata:
            for directive in jdata['content-security-policy'].split(';'):
                for string in directive.split(' '):
                    if string.startswith('http://') or string.startswith('https://'):
                        if self.getTarget().matches(self.sf.urlFQDN(string)):
                            evt = SpiderFootEvent('LINKED_URL_INTERNAL', string, self.__name__, event)
                            self.notifyListeners(evt)
                        else:
                            evt = SpiderFootEvent('LINKED_URL_EXTERNAL', string, self.__name__, event)
                            self.notifyListeners(evt)
        server = jdata.get('server')
        if server:
            self.info(f'Found web server: {server} ({eventSource})')
            evt = SpiderFootEvent('WEBSERVER_BANNER', server, self.__name__, event)
            self.notifyListeners(evt)
        cookies = jdata.get('set-cookie')
        tech = list()
        powered_by = jdata.get('x-powered-by')
        if powered_by:
            tech.append(powered_by)
        if 'x-aspnet-version' in jdata:
            tech.append('ASP.NET')
        if cookies and 'PHPSESS' in cookies:
            tech.append('PHP')
        if cookies and 'JSESSIONID' in cookies:
            tech.append('Java/JSP')
        if cookies and 'ASP.NET' in cookies:
            tech.append('ASP.NET')
        if '.asp' in eventSource:
            tech.append('ASP')
        if '.jsp' in eventSource:
            tech.append('Java/JSP')
        if '.php' in eventSource:
            tech.append('PHP')
        for t in set(tech):
            evt = SpiderFootEvent('WEBSERVER_TECHNOLOGY', t, self.__name__, event)
            self.notifyListeners(evt)