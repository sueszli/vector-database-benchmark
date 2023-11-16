import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_venmo(SpiderFootPlugin):
    meta = {'name': 'Venmo', 'summary': 'Gather user information from Venmo API.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Social Media'], 'dataSource': {'website': 'https://venmo.com/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': [], 'favIcon': 'https://d1v6x81qdeozhc.cloudfront.net/static/images/logo/apple-touch-icon-1a10ee4b947b728d54265ac8c5084f78.png', 'logo': 'https://d1v6x81qdeozhc.cloudfront.net/static/images/logo/apple-touch-icon-1a10ee4b947b728d54265ac8c5084f78.png', 'description': 'Venmo is a digital wallet that allows you to send money and make purchases at approved merchants.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            for i in range(10):
                print('nop')
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['USERNAME']

    def producedEvents(self):
        if False:
            return 10
        return ['RAW_RIR_DATA', 'HUMAN_NAME']

    def query(self, qry):
        if False:
            print('Hello World!')
        res = self.sf.fetchUrl('https://api.venmo.com/v1/users/' + qry, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        time.sleep(1)
        if res['content'] is None:
            self.debug('No response from api.venmo.com')
            return None
        try:
            data = json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        json_data = data.get('data')
        if not json_data:
            self.debug(qry + ' is not a valid Venmo username')
            return None
        return json_data

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if eventData in self.results:
            return
        self.results[eventData] = True
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        data = self.query(eventData)
        if not data:
            return
        display_name = data.get('display_name')
        if ' ' not in display_name:
            if not data.get('first_name') or not data.get('last_name'):
                return
            display_name = data['first_name'] + ' ' + data['last_name']
        if display_name:
            evt = SpiderFootEvent('HUMAN_NAME', display_name, self.__name__, event)
            self.notifyListeners(evt)
            evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
            self.notifyListeners(evt)