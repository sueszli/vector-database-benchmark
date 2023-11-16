import ipaddress
import logging
import os
import socket
from collections import Iterable
from typing import Union, List
import netifaces
from golem.network.stun import pystun as stun
from .variables import DEFAULT_CONNECT_TO, DEFAULT_CONNECT_TO_PORT
logger = logging.getLogger(__name__)

def ip_addresses(use_ipv6: bool=False) -> List[str]:
    if False:
        return 10
    ' Return list of internet addresses that this host have\n    :param bool use_ipv6: *Default: False* if True it returns this host IPv6\n    addresses, otherwise IPv4 addresses are returned\n    :return list: list of host addresses\n    '
    if use_ipv6:
        addr_family = netifaces.AF_INET6
    else:
        addr_family = netifaces.AF_INET
    addresses = []
    for inter in netifaces.interfaces():
        ip = netifaces.ifaddresses(inter).get(addr_family)
        if not isinstance(ip, Iterable):
            continue
        for addrInfo in ip:
            addr = addrInfo.get('addr')
            try:
                ip_addr = ipaddress.ip_address(addr)
            except Exception as exc:
                logger.error('Error parsing ip address %r: %r', addr, exc)
                continue
            if not is_ip_address_allowed(ip_addr):
                continue
            addresses.append(str(ip_addr))
    return addresses

def ipv4_networks():
    if False:
        return 10
    addr_family = netifaces.AF_INET
    addresses = []
    for inter in netifaces.interfaces():
        ip = netifaces.ifaddresses(inter).get(addr_family)
        if not isinstance(ip, Iterable):
            continue
        for addrInfo in ip:
            addr = addrInfo.get('addr')
            mask = addrInfo.get('netmask', '255.255.255.0')
            try:
                ip_net = ipaddress.ip_network((addr, mask), strict=False)
            except Exception as exc:
                logger.error('Error parsing ip address %r: %r', addr, exc)
                continue
            if not is_ip_network_allowed(ip_net):
                continue
            split = str(ip_net).split('/')
            addresses.append((split[0], split[1]))
    return addresses
get_host_addresses = ip_addresses

def is_ip_address_allowed(ip_addr: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    if False:
        print('Hello World!')
    return not (ip_addr.is_loopback or ip_addr.is_link_local or ip_addr.is_multicast or ip_addr.is_unspecified or ip_addr.is_reserved)

def is_ip_network_allowed(ip_net: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]) -> bool:
    if False:
        while True:
            i = 10
    return not (ip_net.is_loopback or ip_net.is_link_local or ip_net.is_multicast or ip_net.is_unspecified or ip_net.is_reserved)

def ip_address_private(address):
    if False:
        return 10
    if address.find(':') != -1:
        try:
            return ipaddress.IPv6Address(str(address)).is_private
        except Exception as exc:
            logger.error('Cannot parse IPv6 address {}: {}'.format(address, exc))
            return False
    try:
        return ipaddress.IPv4Address(str(address)).is_private
    except Exception as exc:
        logger.error('Cannot parse IPv4 address {}: {}'.format(address, exc))
        return False

def ip_network_contains(network, mask, address):
    if False:
        for i in range(10):
            print('nop')
    return ipaddress.ip_network((network, mask), strict=False) == ipaddress.ip_network((str(address), mask), strict=False)

def get_host_address_from_connection(connect_to=DEFAULT_CONNECT_TO, connect_to_port=DEFAULT_CONNECT_TO_PORT, use_ipv6=False):
    if False:
        return 10
    'Get host address by connecting with given address and checking which one of host addresses was used\n    :param str connect_to: address that host should connect to\n    :param int connect_to_port: port that host should connect to\n    :param bool use_ipv6: *Default: False* should IPv6 be use to connect?\n    :return str: host address used to connect\n    '
    if use_ipv6:
        addr_family = socket.AF_INET6
    else:
        addr_family = socket.AF_INET
    return [(s.connect((connect_to, connect_to_port)), s.getsockname()[0], s.close()) for s in [socket.socket(addr_family, socket.SOCK_DGRAM)]][0][1]

def get_external_address(source_port=0):
    if False:
        i = 10
        return i + 15
    'This method tries to get host public address with STUN protocol\n    :param int source_port: port that should be used for connection.\n    If 0, a free port will be picked by OS.\n    :return (str, int, str): tuple with host public address, public port that is\n    mapped to local <source_port> and this host nat type\n    '
    (external_ip, external_port) = stun.get_ip_info(source_port=source_port)
    logger.debug('external_ip [%r] external_port %r', external_ip, external_port)
    return (external_ip, external_port)

def get_host_address(seed_addr=None, use_ipv6=False):
    if False:
        while True:
            i = 10
    '\n    Return this host most useful internet address. Host will try to connect with outer service to determine the address.\n    If connection fail, one of the private address will be used - the one with longest common prefix with given address\n    or the first one if seed address is None\n    :param None|str seed_addr: seed address that may be used to compare addresses\n    :param bool use_ipv6: if True then IPv6 address will be determine, otherwise IPv4 address\n    :return str: host address that is most probably the useful one\n    '
    try:
        ip = get_host_address_from_connection(use_ipv6=use_ipv6)
        if ip is not None:
            return ip
    except Exception as err:
        logger.error("Can't connect to outer service: {}".format(err))
    try:
        ips = ip_addresses(use_ipv6)
        if seed_addr is not None:
            len_pref = [len(os.path.commonprefix([addr, seed_addr])) for addr in ips]
            return ips[len_pref.index(max(len_pref))]
        else:
            if len(ips) < 1:
                raise Exception('Netifaces return empty list of addresses')
            return ips[0]
    except Exception as err:
        logger.error('get_host_address error {}'.format(str(err)))
        return socket.gethostbyname(socket.gethostname())