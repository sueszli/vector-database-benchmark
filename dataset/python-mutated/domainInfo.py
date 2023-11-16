from .baseInfo import BaseInfo

class DomainInfo(BaseInfo):

    def __init__(self, domain, record, type, ips):
        if False:
            print('Hello World!')
        self.record_list = record
        self.domain = domain
        self.type = type
        self.ip_list = ips

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, DomainInfo):
            if self.domain == other.domain:
                return True

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.domain)

    def _dump_json(self):
        if False:
            for i in range(10):
                print('nop')
        item = {'domain': self.domain, 'record': self.record_list, 'type': self.type, 'ips': self.ip_list}
        return item