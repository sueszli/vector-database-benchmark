from spiderfoot import SpiderFootPlugin

class sfp__stor_db(SpiderFootPlugin):
    meta = {'name': 'Storage', 'summary': 'Stores scan results into the back-end SpiderFoot database. You will need this.'}
    _priority = 0
    opts = {'maxstorage': 1024, '_store': True}
    optdescs = {'maxstorage': 'Maximum bytes to store for any piece of information retrieved (0 = unlimited.)'}

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['*']

    def handleEvent(self, sfEvent):
        if False:
            i = 10
            return i + 15
        if not self.opts['_store']:
            return
        if self.opts['maxstorage'] != 0 and len(sfEvent.data) > self.opts['maxstorage']:
            self.debug('Storing an event: ' + sfEvent.eventType)
            self.__sfdb__.scanEventStore(self.getScanId(), sfEvent, self.opts['maxstorage'])
            return
        self.debug('Storing an event: ' + sfEvent.eventType)
        self.__sfdb__.scanEventStore(self.getScanId(), sfEvent)