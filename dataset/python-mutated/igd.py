import functools
from itertools import chain
from typing import Optional, Tuple
import miniupnpc
from golem.network.upnp.mapper import IPortMapper

class IGDPortMapper(IPortMapper):
    name = 'IGD'

    def __init__(self, discovery_delay: int=200):
        if False:
            while True:
                i = 10
        '\n        :param discovery_delay: IGD discovery delay in ms\n        '
        self.upnp = miniupnpc.UPnP()
        self.upnp.discoverdelay = discovery_delay
        self._available = None

    @property
    def available(self) -> bool:
        if False:
            print('Hello World!')
        return self._available is True

    @property
    def network(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'local_ip_address': self.upnp.lanaddr, 'external_ip_address': self.upnp.externalipaddress(), 'connection_type': self.upnp.connectiontype(), 'status_info': self.upnp.statusinfo()}

    def discover(self) -> str:
        if False:
            return 10
        num_devices = self.upnp.discover()
        if not num_devices:
            raise RuntimeError('no devices discovered')
        igd = self.upnp.selectigd()
        self._available = True
        return igd

    def get_mapping(self, external_port: int, protocol: str='TCP') -> Optional[Tuple[str, int, bool]]:
        if False:
            for i in range(10):
                print('nop')
        mapping = self.upnp.getspecificportmapping(external_port, protocol)
        if mapping:
            (ip, port, _description, enabled) = mapping[:4]
            return (ip, port, enabled)
        return None

    def create_mapping(self, local_port: int, external_port: int=None, protocol: str='TCP', lease_duration: int=None) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        local_ip = self.network['local_ip_address']
        external_port = external_port or local_port
        if self._mapping_exists(local_port, external_port, protocol):
            return external_port
        create_mapping = functools.partial(self._create_mapping, protocol, local_ip, local_port, external_port)
        try:
            return create_mapping(auto=True)
        except Exception:
            return create_mapping()

    def remove_mapping(self, _port, external_port: int, protocol: str='TCP'):
        if False:
            i = 10
            return i + 15
        return self.upnp.deleteportmapping(external_port, protocol)

    def _mapping_exists(self, local_port: int, external_port: int, protocol: str='TCP'):
        if False:
            for i in range(10):
                print('nop')
        local_ip = self.network['local_ip_address']
        try:
            existing_mapping = self.get_mapping(external_port, protocol)
            (ip, port, enabled) = existing_mapping
            return enabled and local_ip == ip and (local_port == port)
        except Exception:
            return False

    def _create_mapping(self, protocol: str, local_ip: str, local_port: int, external_port: int, auto: bool=False) -> int:
        if False:
            print('Hello World!')
        description = 'Golem[{}]'.format(local_port)
        remote_host = ''
        if auto:
            method = self.upnp.addanyportmapping
        else:
            method = self.upnp.addportmapping
            external_port = self._find_free_port(external_port, protocol)
        port = method(external_port, protocol, local_ip, local_port, description, remote_host)
        return port if auto else external_port

    def _find_free_port(self, preferred_port: int, protocol: str):
        if False:
            while True:
                i = 10
        range_1 = range(preferred_port, 65536)
        range_2 = range(1024, preferred_port)
        for port in chain(range_1, range_2):
            if not self.get_mapping(port, protocol):
                return port
        raise RuntimeError('no free external ports are available')