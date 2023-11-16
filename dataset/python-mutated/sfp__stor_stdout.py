import json
from spiderfoot import SpiderFootPlugin

class sfp__stor_stdout(SpiderFootPlugin):
    meta = {'name': 'Command-line output', 'summary': 'Dumps output to standard out. Used for when a SpiderFoot scan is run via the command-line.'}
    _priority = 0
    firstEvent = True
    opts = {'_format': 'tab', '_requested': [], '_showonlyrequested': False, '_stripnewline': False, '_showsource': False, '_csvdelim': ',', '_maxlength': 0, '_eventtypes': dict()}
    optdescs = {}

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['*']

    def output(self, event):
        if False:
            while True:
                i = 10
        d = self.opts['_csvdelim']
        if type(event.data) in [list, dict]:
            data = str(event.data)
        else:
            data = event.data
        if type(data) != str:
            data = str(event.data)
        if type(event.sourceEvent.data) in [list, dict]:
            srcdata = str(event.sourceEvent.data)
        else:
            srcdata = event.sourceEvent.data
        if type(srcdata) != str:
            srcdata = str(event.sourceEvent.data)
        if self.opts['_stripnewline']:
            data = data.replace('\n', ' ').replace('\r', '')
            srcdata = srcdata.replace('\n', ' ').replace('\r', '')
        if '<SFURL>' in data:
            data = data.replace('<SFURL>', '').replace('</SFURL>', '')
        if '<SFURL>' in srcdata:
            srcdata = srcdata.replace('<SFURL>', '').replace('</SFURL>', '')
        if self.opts['_maxlength'] > 0:
            data = data[0:self.opts['_maxlength']]
            srcdata = srcdata[0:self.opts['_maxlength']]
        if self.opts['_format'] == 'tab':
            event_type = self.opts['_eventtypes'][event.eventType]
            if self.opts['_showsource']:
                print(f'{event.module.ljust(30)}\t{event_type.ljust(45)}\t{srcdata}\t{data}')
            else:
                print(f'{event.module.ljust(30)}\t{event_type.ljust(45)}\t{data}')
        if self.opts['_format'] == 'csv':
            print(event.module + d + self.opts['_eventtypes'][event.eventType] + d + srcdata + d + data)
        if self.opts['_format'] == 'json':
            d = event.asDict()
            d['type'] = self.opts['_eventtypes'][event.eventType]
            if self.firstEvent:
                self.firstEvent = False
            else:
                print(',')
            print(json.dumps(d), end='')

    def handleEvent(self, sfEvent):
        if False:
            i = 10
            return i + 15
        if sfEvent.eventType == 'ROOT':
            return
        if self.opts['_showonlyrequested']:
            if sfEvent.eventType in self.opts['_requested']:
                self.output(sfEvent)
        else:
            self.output(sfEvent)