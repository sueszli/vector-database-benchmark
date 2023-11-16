"""
Module for getting information about network addresses.

.. versionadded:: 2016.3.0

:depends: netaddr
"""
__virtualname__ = 'netaddress'
try:
    import netaddr
    HAS_NETADDR = True
except ImportError as e:
    HAS_NETADDR = False

def __virtual__():
    if False:
        return 10
    '\n    Only load if netaddr library exist.\n    '
    if not HAS_NETADDR:
        return (False, 'The netaddress execution module cannot be loaded: netaddr python library is not installed.')
    return __virtualname__

def list_cidr_ips(cidr):
    if False:
        while True:
            i = 10
    '\n    Get a list of IP addresses from a CIDR.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netaddress.list_cidr_ips 192.168.0.0/20\n    '
    ips = netaddr.IPNetwork(cidr)
    return [str(ip) for ip in list(ips)]

def list_cidr_ips_ipv6(cidr):
    if False:
        print('Hello World!')
    '\n    Get a list of IPv6 addresses from a CIDR.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netaddress.list_cidr_ips_ipv6 192.168.0.0/20\n    '
    ips = netaddr.IPNetwork(cidr)
    return [str(ip.ipv6()) for ip in list(ips)]

def cidr_netmask(cidr):
    if False:
        print('Hello World!')
    '\n    Get the netmask address associated with a CIDR address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netaddress.cidr_netmask 192.168.0.0/20\n    '
    ips = netaddr.IPNetwork(cidr)
    return str(ips.netmask)

def cidr_broadcast(cidr):
    if False:
        print('Hello World!')
    '\n    Get the broadcast address associated with a CIDR address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netaddress.cidr_netmask 192.168.0.0/20\n    '
    ips = netaddr.IPNetwork(cidr)
    return str(ips.broadcast)