from .baseInfo import BaseInfo
from app import utils

class IPInfo(BaseInfo):

    def __init__(self, ip, port_info, os_info, domain, cdn_name):
        if False:
            for i in range(10):
                print('nop')
        self.ip = ip
        self.port_info_list = port_info
        self.os_info = os_info
        self.domain = domain
        self._geo_asn = None
        self._geo_city = None
        self._ip_type = None
        self.cdn_name = cdn_name

    @property
    def geo_asn(self):
        if False:
            return 10
        if self._geo_asn:
            return self._geo_asn
        elif self.ip_type == 'PUBLIC':
            self._geo_asn = utils.get_ip_asn(self.ip)
        else:
            self._geo_asn = {}
        return self._geo_asn

    @property
    def geo_city(self):
        if False:
            while True:
                i = 10
        if self._geo_city:
            return self._geo_city
        elif self.ip_type == 'PUBLIC':
            self._geo_city = utils.get_ip_city(self.ip)
        else:
            self._geo_city = {}
        return self._geo_city

    @property
    def ip_type(self):
        if False:
            for i in range(10):
                print('nop')
        if self._ip_type:
            return self._ip_type
        else:
            self._ip_type = utils.get_ip_type(self.ip)
        return self._ip_type

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, IPInfo):
            if self.ip == other.ip:
                return True

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.ip)

    def _dump_json(self):
        if False:
            return 10
        port_info = []
        for x in self.port_info_list:
            port_info.append(x.dump_json(flag=False))
        item = {'ip': self.ip, 'domain': self.domain, 'port_info': port_info, 'os_info': self.os_info, 'ip_type': self.ip_type, 'geo_asn': self.geo_asn, 'geo_city': self.geo_city, 'cdn_name': self.cdn_name}
        return item

class PortInfo(BaseInfo):

    def __init__(self, port_id, service_name='', version='', protocol='tcp', product=''):
        if False:
            while True:
                i = 10
        self.port_id = port_id
        self.service_name = service_name
        self.version = version
        self.protocol = protocol
        self.product = product

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, PortInfo):
            if self.port_id == other.port_id:
                return True

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.port_id)

    def _dump_json(self):
        if False:
            print('Hello World!')
        item = {'port_id': self.port_id, 'service_name': self.service_name, 'version': self.version, 'protocol': self.protocol, 'product': self.product}
        return item