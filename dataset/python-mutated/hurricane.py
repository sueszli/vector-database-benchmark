"""
Hurricane Electric DNS Support
"""
import json
try:
    hedns_dependencies = True
    import HurricaneDNS as _hurricanedns
except ImportError:
    hedns_dependencies = False
from . import common

class _Response(object):
    """
    wrapper aliyun resp to the format sewer wanted.
    """

    def __init__(self, status_code=200, content=None, headers=None):
        if False:
            while True:
                i = 10
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content or {}
        self.content = json.dumps(content)
        super(_Response, self).__init__()

    def json(self):
        if False:
            print('Hello World!')
        return json.loads(self.content)

class HurricaneDns(common.BaseDns):

    def __init__(self, username, password):
        if False:
            while True:
                i = 10
        super(HurricaneDns, self).__init__()
        if not hedns_dependencies:
            raise ImportError('You need to install HurricaneDns dependencies. run: pip3 install sewer[hurricane]')
        self.clt = _hurricanedns.HurricaneDNS(username, password)

    @staticmethod
    def extract_zone(domain_name):
        if False:
            print('Hello World!')
        '\n        extract domain to root, sub, acme_txt\n        :param str domain_name: the value sewer client passed in, like *.menduo.example.com\n        :return tuple: root, zone, acme_txt\n        '
        domain_name = domain_name.lstrip('*.')
        if domain_name.count('.') > 1:
            (zone, middle, last) = str(domain_name).rsplit('.', 2)
            root = '.'.join([middle, last])
            acme_txt = '_acme-challenge.%s' % zone
        else:
            zone = ''
            root = domain_name
            acme_txt = '_acme-challenge'
        return (root, zone, acme_txt)

    def create_dns_record(self, domain_name, domain_dns_value):
        if False:
            return 10
        (root, _, acme_txt) = self.extract_zone(domain_name)
        self.clt.add_record(root, acme_txt, 'TXT', domain_dns_value, ttl=300)

    def delete_dns_record(self, domain_name, domain_dns_value):
        if False:
            while True:
                i = 10
        (root, _, acme_txt) = self.extract_zone(domain_name)
        host = '%s.%s' % (acme_txt, root)
        recored_list = self.clt.get_records(root, host, 'TXT')
        for i in recored_list:
            self.clt.del_record(root, i['id'])