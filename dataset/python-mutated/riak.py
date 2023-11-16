try:
    import json
except ImportError:
    import simplejson as json
import os
import sys
import urllib.request
from vendor.munin import MuninPlugin

class MuninRiakPlugin(MuninPlugin):
    category = 'Riak'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(MuninRiakPlugin, self).__init__()
        host = os.environ.get('RIAK_HOST') or 'localhost'
        if ':' in host:
            (host, port) = host.split(':')
            port = int(port)
        else:
            port = 8098
        self.host = '%s:%s' % (host, port)

    def get_status(self):
        if False:
            i = 10
            return i + 15
        res = urllib.request.urlopen('http://%s/stats' % self.host)
        return json.loads(res.read())