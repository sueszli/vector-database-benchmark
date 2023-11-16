import os
import re
import urllib.request
from vendor.munin import MuninPlugin

class MuninNginxPlugin(MuninPlugin):
    category = 'Nginx'
    status_re = re.compile('Active connections:\\s+(?P<active>\\d+)\\s+server accepts handled requests\\s+(?P<accepted>\\d+)\\s+(?P<handled>\\d+)\\s+(?P<requests>\\d+)\\s+Reading: (?P<reading>\\d+) Writing: (?P<writing>\\d+) Waiting: (?P<waiting>\\d+)')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(MuninNginxPlugin, self).__init__()
        self.url = os.environ.get('NX_STATUS_URL') or 'http://localhost/nginx_status'

    def autoconf(self):
        if False:
            while True:
                i = 10
        return bool(self.get_status())

    def get_status(self):
        if False:
            print('Hello World!')
        return self.status_re.search(urllib.request.urlopen(self.url).read()).groupdict()