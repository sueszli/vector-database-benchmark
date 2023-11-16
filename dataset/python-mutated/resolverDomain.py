from app import utils
import threading
import collections
from app.modules import DomainInfo
from .baseThread import BaseThread
logger = utils.get_logger()

class ResolverDomain(BaseThread):

    def __init__(self, domains, concurrency=6):
        if False:
            i = 10
            return i + 15
        super().__init__(domains, concurrency=concurrency)
        self.resolver_map = {}
    '\n    {\n        "api.baike.baidu.com":[\n            "180.97.93.62",\n            "180.97.93.61"\n        ],\n        "apollo.baidu.com":[\n            "123.125.115.15"\n        ],\n        "www.baidu.com":[\n            "180.101.49.12",\n            "180.101.49.11"\n        ]\n    }\n    '

    def work(self, domain):
        if False:
            while True:
                i = 10
        curr_domain = domain
        if isinstance(domain, dict):
            curr_domain = domain.get('domain')
        elif isinstance(domain, DomainInfo):
            curr_domain = domain.domain
        if not curr_domain:
            return
        if curr_domain in self.resolver_map:
            return
        self.resolver_map[curr_domain] = utils.get_ip(curr_domain)

    def run(self):
        if False:
            print('Hello World!')
        self._run()
        return self.resolver_map

def resolver_domain(domains, concurrency=15):
    if False:
        i = 10
        return i + 15
    r = ResolverDomain(domains, concurrency)
    return r.run()