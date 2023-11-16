"""
============
DNS Browsing
============

Explore the space around known hosts & ips for extra catches.
"""
import asyncio
import re
import sys
from ipaddress import IPv4Network
from typing import Callable, List, Optional
from aiodns import DNSResolver
from theHarvester.lib import hostchecker
from theHarvester.lib.core import DATA_DIR
DNS_NAMES = DATA_DIR / 'wordlists' / 'dns-names.txt'

class DnsForce:

    def __init__(self, domain, dnsserver, verbose: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.domain = domain
        self.subdo = False
        self.verbose = verbose
        self.dnsserver = dnsserver
        with DNS_NAMES.open('r') as file:
            self.list = file.readlines()
        self.domain = domain.replace('www.', '')
        self.list = [f'{word.strip()}.{self.domain}' for word in self.list]

    async def run(self):
        print(f'Starting DNS brute forcing with {len(self.list)} words')
        checker = hostchecker.Checker(self.list, nameserver=self.dnsserver)
        (resolved_pair, hosts, ips) = await checker.check()
        return (resolved_pair, hosts, ips)
IP_REGEX = '\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}'
PORT_REGEX = '\\d{1,5}'
NETMASK_REGEX: str = '\\d{1,2}|' + IP_REGEX
NETWORK_REGEX: str = '\\b({})(?:\\:({}))?(?:\\/({}))?\\b'.format(IP_REGEX, PORT_REGEX, NETMASK_REGEX)

def serialize_ip_range(ip: str, netmask: str='24') -> str:
    if False:
        return 10
    "\n    Serialize a network range in a constant format, 'x.x.x.x/y'.\n\n    Parameters\n    ----------\n    ip: str.\n        A serialized ip in the format 'x.x.x.x'.\n        Extra information like port (':z') or subnet ('/n')\n        will be ignored.\n    netmask: str.\n        The subnet subdivision, represented by a 2 digit netmask.\n\n    Returns\n    -------\n    out: str.\n        The network OSI address, like '192.168.0.0/24'.\n    "
    __ip_matches = re.search(NETWORK_REGEX, ip, re.IGNORECASE)
    if __ip_matches and __ip_matches.groups():
        __ip = __ip_matches.group(1)
        __netmask = netmask if netmask else __ip_matches.group(3)
        if __ip and __netmask:
            return str(IPv4Network('{}/{}'.format(__ip, __netmask), strict=False))
        elif __ip:
            return str(IPv4Network('{}/{}'.format(__ip, '24'), strict=False))
    return ''

def list_ips_in_network_range(iprange: str) -> List[str]:
    if False:
        print('Hello World!')
    "\n    List all the IPs in the range.\n\n    Parameters\n    ----------\n    iprange: str.\n        A serialized ip range, like '1.2.3.0/24'.\n        The last digit can be set to anything, it will be ignored.\n\n    Returns\n    -------\n    out: list.\n        The list of IPs in the range.\n    "
    try:
        __network = IPv4Network(iprange, strict=False)
        return [__address.exploded for __address in __network.hosts()]
    except Exception:
        return []

async def reverse_single_ip(ip: str, resolver: DNSResolver) -> str:
    """
    Reverse a single IP and output the linked CNAME, if it exists.
        Parameters
        ----------
        :param ip:  IP address to reverse
        :param resolver: DNS server to use

        Returns
        -------
        :return str: with the corresponding CNAME or None
    """
    try:
        __host = await resolver.gethostbyaddr(ip)
        return __host.name if __host else ''
    except Exception:
        return ''

async def reverse_all_ips_in_range(iprange: str, callback: Callable, nameservers: Optional[List[str]]=None) -> None:
    """
    Reverse all the IPs stored in a network range.
    All the queries are made concurrently.

    Parameters
    ----------
    iprange: str.
        An IPv4 range formatted as 'x.x.x.x/y'.
        The last 2 digits of the ip can be set to anything,
        they will be ignored.
    callback: Callable.
        Arbitrary postprocessing function.
    nameservers: List[str].
        Optional list of DNS servers.

    Returns
    -------
    out: None.
    """
    loop = asyncio.get_event_loop()
    __resolver = DNSResolver(loop=loop, timeout=8, nameservers=nameservers)
    for __ip in list_ips_in_network_range(iprange):
        log_query(__ip)
        __host = await reverse_single_ip(ip=__ip, resolver=__resolver)
        callback(__host)
        log_result(__host)

def log_query(ip: str) -> None:
    if False:
        return 10
    '\n    Display the current query in the console.\n\n    Parameters\n    ----------\n    ip: str.\n        Queried ip.\n\n    Results\n    -------\n    out: None.\n    '
    sys.stdout.write(chr(27) + '[2K' + chr(27) + '[G')
    sys.stdout.write('\r' + ip + ' - ')
    sys.stdout.flush()

def log_result(host: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Display the query result in the console.\n\n    Parameters\n    ----------\n    host: str.\n        Host name returned by the DNS query.\n\n    Results\n    -------\n    out: None.\n    '
    if host:
        print(host)

def generate_postprocessing_callback(target: str, **allhosts: List[str]) -> Callable:
    if False:
        i = 10
        return i + 15
    '\n    Postprocess the query results asynchronously too, instead of waiting for\n    the querying stage to be completely finished.\n\n    Parameters\n    ----------\n    target: str.\n        The domain wanted as TLD.\n    allhosts: List.\n        A collection of all the subdomains -of target- found so far.\n\n    Returns\n    -------\n    out: Callable.\n        A function that will update the collection of target subdomains\n        when the query result is satisfying.\n    '

    def append_matching_hosts(host: str) -> None:
        if False:
            while True:
                i = 10
        if host and target in host:
            for (__name, __hosts) in allhosts.items():
                if host not in __hosts:
                    __hosts.append(host)
    return append_matching_hosts