from __future__ import annotations

class FilterModule(object):

    def filters(self):
        if False:
            i = 10
            return i + 15
        return {'parse_ip': self.parse_ip}

    def parse_ip(self, ip):
        if False:
            return 10
        return ip