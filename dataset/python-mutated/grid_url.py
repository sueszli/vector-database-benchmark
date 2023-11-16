from __future__ import annotations
import copy
import os
import re
from typing import Optional
from typing import Union
from urllib.parse import urlparse
import requests
from typing_extensions import Self
from ..serde.serializable import serializable
from ..util.util import verify_tls

@serializable(attrs=['protocol', 'host_or_ip', 'port', 'path', 'query'])
class GridURL:

    @classmethod
    def from_url(cls, url: Union[str, GridURL]) -> Self:
        if False:
            while True:
                i = 10
        if isinstance(url, GridURL):
            return url
        try:
            if '://' not in url:
                url = 'http://' + url
            parts = urlparse(url)
            host_or_ip_parts = parts.netloc.split(':')
            port = 80
            if len(host_or_ip_parts) > 1:
                port = int(host_or_ip_parts[1])
            host_or_ip = host_or_ip_parts[0]
            if parts.scheme == 'https':
                port = 443
            return cls(host_or_ip=host_or_ip, path=parts.path, port=port, protocol=parts.scheme, query=getattr(parts, 'query', ''))
        except Exception as e:
            print(f'Failed to convert url: {url} to GridURL. {e}')
            raise e

    def __init__(self, protocol: str='http', host_or_ip: str='localhost', port: Optional[int]=80, path: str='', query: str='') -> None:
        if False:
            print('Hello World!')
        match_port = re.search(':[0-9]{1,5}', host_or_ip)
        if match_port:
            sub_grid_url: GridURL = GridURL.from_url(host_or_ip)
            host_or_ip = str(sub_grid_url.host_or_ip)
            port = int(sub_grid_url.port)
            protocol = str(sub_grid_url.protocol)
            path = str(sub_grid_url.path)
        prtcl_pattrn = '://'
        if prtcl_pattrn in host_or_ip:
            protocol = host_or_ip[:host_or_ip.find(prtcl_pattrn)]
            start_index = host_or_ip.find(prtcl_pattrn) + len(prtcl_pattrn)
            host_or_ip = host_or_ip[start_index:]
        self.host_or_ip = host_or_ip
        self.path: str = path
        self.port = port
        self.protocol = protocol
        self.query = query

    def with_path(self, path: str) -> Self:
        if False:
            i = 10
            return i + 15
        dupe = copy.copy(self)
        dupe.path = path
        return dupe

    def as_container_host(self, container_host: Optional[str]=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if self.host_or_ip not in ['localhost', 'host.docker.internal', 'host.k3d.internal']:
            return self
        if container_host is None:
            container_host = os.getenv('CONTAINER_HOST', None)
        if container_host:
            if container_host == 'docker':
                hostname = 'host.docker.internal'
            elif container_host == 'podman':
                hostname = 'host.containers.internal'
            else:
                hostname = 'host.k3d.internal'
        else:
            hostname = 'localhost'
        return self.__class__(protocol=self.protocol, host_or_ip=hostname, port=self.port, path=self.path)

    @property
    def query_string(self) -> str:
        if False:
            while True:
                i = 10
        query_string = ''
        if len(self.query) > 0:
            query_string = f'?{self.query}'
        return query_string

    @property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.base_url}{self.path}{self.query_string}'

    @property
    def url_no_port(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.base_url_no_port}{self.path}{self.query_string}'

    @property
    def base_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.protocol}://{self.host_or_ip}:{self.port}'

    @property
    def base_url_no_port(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.protocol}://{self.host_or_ip}'

    @property
    def url_path(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.path}{self.query_string}'

    def to_tls(self) -> Self:
        if False:
            print('Hello World!')
        if self.protocol == 'https':
            return self
        r = requests.get(self.base_url, verify=verify_tls())
        new_base_url = r.url
        if new_base_url.endswith('/'):
            new_base_url = new_base_url[0:-1]
        return self.__class__.from_url(url=f'{new_base_url}{self.path}{self.query_string}')

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<{type(self).__name__} {self.url}>'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.url

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(self.__str__())

    def __copy__(self) -> Self:
        if False:
            print('Hello World!')
        return self.__class__.from_url(self.url)

    def set_port(self, port: int) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.port = port
        return self