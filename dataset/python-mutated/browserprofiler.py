import json
from pprint import pformat
from plugins.plugin import Plugin
from plugins.inject import Inject

class BrowserProfiler(Inject, Plugin):
    name = 'BrowserProfiler'
    optname = 'browserprofiler'
    desc = 'Attempts to enumerate all browser plugins of connected clients'
    version = '0.3'

    def initialize(self, options):
        if False:
            return 10
        Inject.initialize(self, options)
        self.js_file = './core/javascript/plugindetect.js'
        self.output = {}

    def request(self, request):
        if False:
            print('Hello World!')
        if request.command == 'POST' and 'clientprfl' in request.uri:
            request.handle_post_output = True
            self.output = json.loads(request.postData)
            self.output['ip'] = request.client.getClientIP()
            pretty_output = pformat(self.output)
            self.clientlog.info('Got profile:\n{}'.format(pretty_output), extra=request.clientInfo)

    def options(self, options):
        if False:
            i = 10
            return i + 15
        pass