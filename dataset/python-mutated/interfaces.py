"""
Interfaces management
"""
import itertools
import uuid
from collections import defaultdict
from scapy.config import conf
from scapy.consts import WINDOWS, LINUX
from scapy.utils import pretty_list
from scapy.utils6 import in6_isvalid
import scapy
from scapy.compat import UserDict
from typing import cast, Any, DefaultDict, Dict, List, NoReturn, Optional, Tuple, Type, Union

class InterfaceProvider(object):
    name = 'Unknown'
    headers: Tuple[str, ...] = ('Index', 'Name', 'MAC', 'IPv4', 'IPv6')
    header_sort = 1
    libpcap = False

    def load(self):
        if False:
            print('Hello World!')
        'Returns a dictionary of the loaded interfaces, by their\n        name.'
        raise NotImplementedError

    def reload(self):
        if False:
            return 10
        'Same than load() but for reloads. By default calls load'
        return self.load()

    def _l2socket(self, dev):
        if False:
            print('Hello World!')
        'Return L2 socket used by interfaces of this provider'
        return conf.L2socket

    def _l2listen(self, dev):
        if False:
            while True:
                i = 10
        'Return L2listen socket used by interfaces of this provider'
        return conf.L2listen

    def _l3socket(self, dev, ipv6):
        if False:
            while True:
                i = 10
        'Return L3 socket used by interfaces of this provider'
        if LINUX and (not self.libpcap) and (dev.name == conf.loopback_name):
            if ipv6:
                from scapy.supersocket import L3RawSocket6
                return cast(Type['scapy.supersocket.SuperSocket'], L3RawSocket6)
            else:
                from scapy.supersocket import L3RawSocket
                return L3RawSocket
        return conf.L3socket

    def _is_valid(self, dev):
        if False:
            print('Hello World!')
        'Returns whether an interface is valid or not'
        return bool((dev.ips[4] or dev.ips[6]) and dev.mac)

    def _format(self, dev, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns the elements used by show()\n\n        If a tuple is returned, this consist of the strings that will be\n        inlined along with the interface.\n        If a list of tuples is returned, they will be appended one above the\n        other and should all be part of a single interface.\n        '
        mac = dev.mac
        resolve_mac = kwargs.get('resolve_mac', True)
        if resolve_mac and conf.manufdb and mac:
            mac = conf.manufdb._resolve_MAC(mac)
        index = str(dev.index)
        return (index, dev.description, mac or '', dev.ips[4], dev.ips[6])

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        repr\n        '
        return '<InterfaceProvider: %s>' % self.name

class NetworkInterface(object):

    def __init__(self, provider, data=None):
        if False:
            print('Hello World!')
        self.provider = provider
        self.name = ''
        self.description = ''
        self.network_name = ''
        self.index = -1
        self.ip = None
        self.ips = defaultdict(list)
        self.mac = None
        self.dummy = False
        if data is not None:
            self.update(data)

    def update(self, data):
        if False:
            while True:
                i = 10
        'Update info about a network interface according\n        to a given dictionary. Such data is provided by providers\n        '
        self.name = data.get('name', '')
        self.description = data.get('description', '')
        self.network_name = data.get('network_name', '')
        self.index = data.get('index', 0)
        self.ip = data.get('ip', '')
        self.mac = data.get('mac', '')
        self.flags = data.get('flags', 0)
        self.dummy = data.get('dummy', False)
        for ip in data.get('ips', []):
            if in6_isvalid(ip):
                self.ips[6].append(ip)
            else:
                self.ips[4].append(ip)
        if self.ips[4] and (not self.ip):
            self.ip = self.ips[4][0]

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, str):
            return other in [self.name, self.network_name, self.description]
        if isinstance(other, NetworkInterface):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.network_name)

    def is_valid(self):
        if False:
            return 10
        if self.dummy:
            return False
        return self.provider._is_valid(self)

    def l2socket(self):
        if False:
            i = 10
            return i + 15
        return self.provider._l2socket(self)

    def l2listen(self):
        if False:
            for i in range(10):
                print('nop')
        return self.provider._l2listen(self)

    def l3socket(self, ipv6=False):
        if False:
            while True:
                i = 10
        return self.provider._l3socket(self, ipv6)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s %s [%s]>' % (self.__class__.__name__, self.description, self.dummy and 'dummy' or (self.flags or ''))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.network_name

    def __add__(self, other):
        if False:
            return 10
        return self.network_name + other

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return other + self.network_name
_GlobInterfaceType = Union[NetworkInterface, str]

class NetworkInterfaceDict(UserDict[str, NetworkInterface]):
    """Store information about network interfaces and convert between names"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.providers = {}
        super(NetworkInterfaceDict, self).__init__()

    def _load(self, dat, prov):
        if False:
            while True:
                i = 10
        for (ifname, iface) in dat.items():
            if ifname in self.data:
                if prov.libpcap:
                    self.data[ifname] = iface
            else:
                self.data[ifname] = iface

    def register_provider(self, provider):
        if False:
            return 10
        prov = provider()
        self.providers[provider] = prov
        if self.data:
            self._load(prov.reload(), prov)

    def load_confiface(self):
        if False:
            while True:
                i = 10
        '\n        Reload conf.iface\n        '
        if not conf.route:
            raise ValueError("Error: conf.route isn't populated !")
        conf.iface = get_working_if()

    def _reload_provs(self):
        if False:
            while True:
                i = 10
        self.clear()
        for prov in self.providers.values():
            self._load(prov.reload(), prov)

    def reload(self):
        if False:
            print('Hello World!')
        self._reload_provs()
        if not conf.route:
            return
        self.load_confiface()

    def dev_from_name(self, name):
        if False:
            return 10
        'Return the first network device name for a given\n        device name.\n        '
        try:
            return next((iface for iface in self.values() if iface.name == name or iface.description == name))
        except (StopIteration, RuntimeError):
            raise ValueError('Unknown network interface %r' % name)

    def dev_from_networkname(self, network_name):
        if False:
            i = 10
            return i + 15
        'Return interface for a given network device name.'
        try:
            return next((iface for iface in self.values() if iface.network_name == network_name))
        except (StopIteration, RuntimeError):
            raise ValueError('Unknown network interface %r' % network_name)

    def dev_from_index(self, if_index):
        if False:
            i = 10
            return i + 15
        'Return interface name from interface index'
        try:
            if_index = int(if_index)
            return next((iface for iface in self.values() if iface.index == if_index))
        except (StopIteration, RuntimeError):
            if str(if_index) == '1':
                return self.dev_from_networkname(conf.loopback_name)
            raise ValueError('Unknown network interface index %r' % if_index)

    def _add_fake_iface(self, ifname):
        if False:
            for i in range(10):
                print('nop')
        'Internal function used for a testing purpose'
        data = {'name': ifname, 'description': ifname, 'network_name': ifname, 'index': -1000, 'dummy': True, 'mac': '00:00:00:00:00:00', 'flags': 0, 'ips': ['127.0.0.1', '::'], 'guid': '{%s}' % uuid.uuid1(), 'ipv4_metric': 0, 'ipv6_metric': 0, 'nameservers': []}
        if WINDOWS:
            from scapy.arch.windows import NetworkInterface_Win, WindowsInterfacesProvider

            class FakeProv(WindowsInterfacesProvider):
                name = 'fake'
            self.data[ifname] = NetworkInterface_Win(FakeProv(), data)
        else:
            self.data[ifname] = NetworkInterface(InterfaceProvider(), data)

    def show(self, print_result=True, hidden=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print list of available network interfaces in human readable form\n\n        :param print_result: print the results if True, else return it\n        :param hidden: if True, also displays invalid interfaces\n        '
        res = defaultdict(list)
        for iface_name in sorted(self.data):
            dev = self.data[iface_name]
            if not hidden and (not dev.is_valid()):
                continue
            prov = dev.provider
            res[prov.headers, prov.header_sort].append((prov.name,) + prov._format(dev, **kwargs))
        output = ''
        for key in res:
            (hdrs, sortBy) = key
            output += pretty_list(res[key], [('Source',) + hdrs], sortBy=sortBy) + '\n'
        output = output[:-1]
        if print_result:
            print(output)
            return None
        else:
            return output

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.show(print_result=False)
conf.ifaces = IFACES = ifaces = NetworkInterfaceDict()

def get_if_list():
    if False:
        print('Hello World!')
    'Return a list of interface names'
    return list(conf.ifaces.keys())

def get_working_if():
    if False:
        i = 10
        return i + 15
    'Return an interface that works'
    routes = conf.route.routes[:]
    routes.sort(key=lambda x: x[1])
    ifaces = (x[3] for x in routes)
    for ifname in itertools.chain(ifaces, conf.ifaces.values()):
        iface = resolve_iface(ifname)
        if iface.is_valid():
            return iface
    return resolve_iface(conf.loopback_name)

def get_working_ifaces():
    if False:
        print('Hello World!')
    'Return all interfaces that work'
    return [iface for iface in conf.ifaces.values() if iface.is_valid()]

def dev_from_networkname(network_name):
    if False:
        while True:
            i = 10
    'Return Scapy device name for given network device name'
    return conf.ifaces.dev_from_networkname(network_name)

def dev_from_index(if_index):
    if False:
        return 10
    'Return interface for a given interface index'
    return conf.ifaces.dev_from_index(if_index)

def resolve_iface(dev):
    if False:
        for i in range(10):
            print('nop')
    '\n    Resolve an interface name into the interface\n    '
    if isinstance(dev, NetworkInterface):
        return dev
    try:
        return conf.ifaces.dev_from_name(dev)
    except ValueError:
        try:
            return dev_from_networkname(dev)
        except ValueError:
            pass
    return NetworkInterface(InterfaceProvider(), data={'name': dev, 'description': dev, 'network_name': dev, 'dummy': True})

def network_name(dev):
    if False:
        print('Hello World!')
    '\n    Resolves the device network name of a device or Scapy NetworkInterface\n    '
    return resolve_iface(dev).network_name

def show_interfaces(resolve_mac=True):
    if False:
        print('Hello World!')
    'Print list of available network interfaces'
    return conf.ifaces.show(resolve_mac)