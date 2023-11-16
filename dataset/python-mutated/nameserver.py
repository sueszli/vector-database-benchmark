from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query

class Nameserver:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def kind(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def is_always_max_size(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def answer_nameserver(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def answer_port(self) -> int:
        if False:
            return 10
        raise NotImplementedError

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        raise NotImplementedError

class AddressAndPortNameserver(Nameserver):

    def __init__(self, address: str, port: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.address = address
        self.port = port

    def kind(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def is_always_max_size(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def __str__(self):
        if False:
            while True:
                i = 10
        ns_kind = self.kind()
        return f'{ns_kind}:{self.address}@{self.port}'

    def answer_nameserver(self) -> str:
        if False:
            return 10
        return self.address

    def answer_port(self) -> int:
        if False:
            return 10
        return self.port

class Do53Nameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=53):
        if False:
            while True:
                i = 10
        super().__init__(address, port)

    def kind(self):
        if False:
            i = 10
            return i + 15
        return 'Do53'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if False:
            i = 10
            return i + 15
        if max_size:
            response = dns.query.tcp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        else:
            response = dns.query.udp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, raise_on_truncation=True, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        return response

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if max_size:
            response = await dns.asyncquery.tcp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, backend=backend, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        else:
            response = await dns.asyncquery.udp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, raise_on_truncation=True, backend=backend, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        return response

class DoHNameserver(Nameserver):

    def __init__(self, url: str, bootstrap_address: Optional[str]=None):
        if False:
            return 10
        super().__init__()
        self.url = url
        self.bootstrap_address = bootstrap_address

    def kind(self):
        if False:
            print('Hello World!')
        return 'DoH'

    def is_always_max_size(self) -> bool:
        if False:
            return 10
        return True

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.url

    def answer_nameserver(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.url

    def answer_port(self) -> int:
        if False:
            print('Hello World!')
        port = urlparse(self.url).port
        if port is None:
            port = 443
        return port

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if False:
            while True:
                i = 10
        return dns.query.https(request, self.url, timeout=timeout, bootstrap_address=self.bootstrap_address, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.https(request, self.url, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)

class DoTNameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=853, hostname: Optional[str]=None):
        if False:
            while True:
                i = 10
        super().__init__(address, port)
        self.hostname = hostname

    def kind(self):
        if False:
            print('Hello World!')
        return 'DoT'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if False:
            while True:
                i = 10
        return dns.query.tls(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, server_hostname=self.hostname)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.tls(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, server_hostname=self.hostname)

class DoQNameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=853, verify: Union[bool, str]=True, server_hostname: Optional[str]=None):
        if False:
            return 10
        super().__init__(address, port)
        self.verify = verify
        self.server_hostname = server_hostname

    def kind(self):
        if False:
            i = 10
            return i + 15
        return 'DoQ'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if False:
            while True:
                i = 10
        return dns.query.quic(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, server_hostname=self.server_hostname)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.quic(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, server_hostname=self.server_hostname)