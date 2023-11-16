import base64
import re
import urllib.parse
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_base64(SpiderFootPlugin):
    meta = {'name': 'Base64 Decoder', 'summary': 'Identify Base64-encoded strings in URLs, often revealing interesting hidden information.', 'flags': [], 'useCases': ['Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {'minlength': 10}
    optdescs = {'minlength': 'The minimum length a string that looks like a base64-encoded string needs to be.'}

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.__dataSource__ = 'Target Website'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['LINKED_URL_INTERNAL']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['BASE64_DATA']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        decoded_data = urllib.parse.unquote(eventData)
        pat = re.compile('([A-Za-z0-9+\\/]+={1,2})')
        m = re.findall(pat, decoded_data)
        for match in m:
            if self.checkForStop():
                return
            minlen = int(self.opts['minlength'])
            if len(match) < minlen:
                continue
            caps = sum((1 for c in match if c.isupper()))
            if caps < minlen / 4:
                continue
            if isinstance(match, str):
                string = match
            else:
                string = str(match)
            self.info(f'Found Base64 string: {match}')
            try:
                string += f" ({base64.b64decode(match).decode('utf-8')})"
            except Exception as e:
                self.debug(f'Unable to base64-decode string: {e}')
                continue
            evt = SpiderFootEvent('BASE64_DATA', string, self.__name__, event)
            self.notifyListeners(evt)