from __future__ import absolute_import
import re
import ipaddress
from st2common.log import logging
LOG = logging.getLogger(__name__)
__all__ = ['is_ipv4', 'is_ipv6', 'split_host_port']
BRACKET_PATTERN = '^\\[.*\\]'
COMPILED_BRACKET_PATTERN = re.compile(BRACKET_PATTERN)
HOST_ONLY_IN_BRACKET = '^\\[.*\\]$'
COMPILED_HOST_ONLY_IN_BRACKET_PATTERN = re.compile(HOST_ONLY_IN_BRACKET)

def is_ipv6(ip_str):
    if False:
        return 10
    '\n    Validate whether given string is IPv6.\n\n    :param ip_str: String to validate.\n    :type ip_str: ``str``\n\n    :rtype: ``bool``\n    '
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr.version == 6
    except:
        return False

def is_ipv4(ip_str):
    if False:
        i = 10
        return i + 15
    '\n    Validate whether given string is IPv4.\n\n    :param ip_str: String to validate.\n    :type ip_str: ``str``\n\n    :rtype: ``bool``\n    '
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr.version == 4
    except:
        return False

def split_host_port(host_str):
    if False:
        print('Hello World!')
    '\n    Split host_str into host and port.\n    Can handle IPv4, IPv6, hostname inside or outside brackets.\n\n    Note: If you want to specify a port with IPv6, you definitely\n    should enclose IP address within [].\n\n    :param host_str: Host port string.\n    :type host_str: ``str``\n\n    :return: Hostname (string), port (int) tuple. Raises exception on invalid port.\n    :rtype: ``tuple`` of ``str`` and ``int``\n    '
    hostname = host_str
    port = None
    if is_ipv6(host_str) or is_ipv4(host_str):
        return (hostname, port)
    match = COMPILED_BRACKET_PATTERN.match(host_str)
    if match:
        LOG.debug('Square bracket style.')
        match = COMPILED_HOST_ONLY_IN_BRACKET_PATTERN.match(host_str)
        if match:
            hostname = match.group().strip('[]')
            return (hostname, port)
        (hostname, separator, port) = hostname.rpartition(':')
        try:
            LOG.debug('host_str: %s, hostname: %s port: %s' % (host_str, hostname, port))
            port = int(port)
            hostname = hostname.strip('[]')
            return (hostname, port)
        except:
            raise Exception('Invalid port %s specified.' % port)
    else:
        LOG.debug('Non-bracket address. host_str: %s' % host_str)
        if ':' in host_str:
            LOG.debug('Non-bracket with port.')
            (hostname, separator, port) = hostname.rpartition(':')
            try:
                port = int(port)
                return (hostname, port)
            except:
                raise Exception('Invalid port %s specified.' % port)
    return (hostname, port)