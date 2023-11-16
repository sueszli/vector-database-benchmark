"""
The networking module for Debian-based distros

References:

* http://www.debian.org/doc/manuals/debian-reference/ch05.en.html
"""
import functools
import io
import logging
import os
import os.path
import re
import time
import jinja2
import jinja2.exceptions
import salt.utils.dns
import salt.utils.files
import salt.utils.odict
import salt.utils.stringutils
import salt.utils.templates
import salt.utils.validate.net
log = logging.getLogger(__name__)
JINJA = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(salt.utils.templates.TEMPLATE_DIRNAME, 'debian_ip')))
__virtualname__ = 'ip'

def __virtual__():
    if False:
        return 10
    '\n    Confine this module to Debian-based distros\n    '
    if __grains__['os_family'] == 'Debian':
        return __virtualname__
    return (False, 'The debian_ip module could not be loaded: unsupported OS family')
_ETHTOOL_CONFIG_OPTS = {'speed': 'link-speed', 'duplex': 'link-duplex', 'autoneg': 'ethernet-autoneg', 'ethernet-port': 'ethernet-port', 'wol': 'ethernet-wol', 'driver-message-level': 'driver-message-level', 'ethernet-pause-rx': 'ethernet-pause-rx', 'ethernet-pause-tx': 'ethernet-pause-tx', 'ethernet-pause-autoneg': 'ethernet-pause-autoneg', 'rx': 'offload-rx', 'tx': 'offload-tx', 'sg': 'offload-sg', 'tso': 'offload-tso', 'ufo': 'offload-ufo', 'gso': 'offload-gso', 'gro': 'offload-gro', 'lro': 'offload-lro', 'hardware-irq-coalesce-adaptive-rx': 'hardware-irq-coalesce-adaptive-rx', 'hardware-irq-coalesce-adaptive-tx': 'hardware-irq-coalesce-adaptive-tx', 'hardware-irq-coalesce-rx-usecs': 'hardware-irq-coalesce-rx-usecs', 'hardware-irq-coalesce-rx-frames': 'hardware-irq-coalesce-rx-frames', 'hardware-dma-ring-rx': 'hardware-dma-ring-rx', 'hardware-dma-ring-rx-mini': 'hardware-dma-ring-rx-mini', 'hardware-dma-ring-rx-jumbo': 'hardware-dma-ring-rx-jumbo', 'hardware-dma-ring-tx': 'hardware-dma-ring-tx'}
_REV_ETHTOOL_CONFIG_OPTS = {'link-speed': 'speed', 'link-duplex': 'duplex', 'ethernet-autoneg': 'autoneg', 'ethernet-port': 'ethernet-port', 'ethernet-wol': 'wol', 'driver-message-level': 'driver-message-level', 'ethernet-pause-rx': 'ethernet-pause-rx', 'ethernet-pause-tx': 'ethernet-pause-tx', 'ethernet-pause-autoneg': 'ethernet-pause-autoneg', 'offload-rx': 'rx', 'offload-tx': 'tx', 'offload-sg': 'sg', 'offload-tso': 'tso', 'offload-ufo': 'ufo', 'offload-gso': 'gso', 'offload-lro': 'lro', 'offload-gro': 'gro', 'hardware-irq-coalesce-adaptive-rx': 'hardware-irq-coalesce-adaptive-rx', 'hardware-irq-coalesce-adaptive-tx': 'hardware-irq-coalesce-adaptive-tx', 'hardware-irq-coalesce-rx-usecs': 'hardware-irq-coalesce-rx-usecs', 'hardware-irq-coalesce-rx-frames': 'hardware-irq-coalesce-rx-frames', 'hardware-dma-ring-rx': 'hardware-dma-ring-rx', 'hardware-dma-ring-rx-mini': 'hardware-dma-ring-rx-mini', 'hardware-dma-ring-rx-jumbo': 'hardware-dma-ring-rx-jumbo', 'hardware-dma-ring-tx': 'hardware-dma-ring-tx'}
_DEB_CONFIG_PPPOE_OPTS = {'user': 'user', 'password': 'password', 'provider': 'provider', 'pppoe_iface': 'pppoe_iface', 'noipdefault': 'noipdefault', 'usepeerdns': 'usepeerdns', 'defaultroute': 'defaultroute', 'holdoff': 'holdoff', 'maxfail': 'maxfail', 'hide-password': 'hide-password', 'lcp-echo-interval': 'lcp-echo-interval', 'lcp-echo-failure': 'lcp-echo-failure', 'connect': 'connect', 'noauth': 'noauth', 'persist': 'persist', 'mtu': 'mtu', 'noaccomp': 'noaccomp', 'linkname': 'linkname'}
_DEB_ROUTES_FILE = '/etc/network/routes'
_DEB_NETWORK_FILE = '/etc/network/interfaces'
_DEB_NETWORK_DIR = '/etc/network/interfaces.d/'
_DEB_NETWORK_UP_DIR = '/etc/network/if-up.d/'
_DEB_NETWORK_DOWN_DIR = '/etc/network/if-down.d/'
_DEB_NETWORK_CONF_FILES = '/etc/modprobe.d/'
_DEB_NETWORKING_FILE = '/etc/default/networking'
_DEB_HOSTNAME_FILE = '/etc/hostname'
_DEB_RESOLV_FILE = '/etc/resolv.conf'
_DEB_PPP_DIR = '/etc/ppp/peers/'
_CONFIG_TRUE = ['yes', 'on', 'true', '1', True]
_CONFIG_FALSE = ['no', 'off', 'false', '0', False]
_IFACE_TYPES = ['eth', 'bond', 'alias', 'clone', 'ipsec', 'dialup', 'bridge', 'slave', 'vlan', 'pppoe', 'source']

def _error_msg_iface(iface, option, expected):
    if False:
        i = 10
        return i + 15
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    msg = 'Invalid option -- Interface: {0}, Option: {1}, Expected: [{2}]'
    return msg.format(iface, option, '|'.join((str(e) for e in expected)))

def _error_msg_routes(iface, option, expected):
    if False:
        print('Hello World!')
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    msg = 'Invalid option -- Route interface: {0}, Option: {1}, Expected: [{2}]'
    return msg.format(iface, option, expected)

def _log_default_iface(iface, opt, value):
    if False:
        for i in range(10):
            print('nop')
    log.info('Using default option -- Interface: %s Option: %s Value: %s', iface, opt, value)

def _error_msg_network(option, expected):
    if False:
        return 10
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    msg = 'Invalid network setting -- Setting: {0}, Expected: [{1}]'
    return msg.format(option, '|'.join((str(e) for e in expected)))

def _log_default_network(opt, value):
    if False:
        i = 10
        return i + 15
    log.info('Using existing setting -- Setting: %s Value: %s', opt, value)

def _raise_error_iface(iface, option, expected):
    if False:
        while True:
            i = 10
    '\n    Log and raise an error with a logical formatted message.\n    '
    msg = _error_msg_iface(iface, option, expected)
    log.error(msg)
    raise AttributeError(msg)

def _raise_error_network(option, expected):
    if False:
        while True:
            i = 10
    '\n    Log and raise an error with a logical formatted message.\n    '
    msg = _error_msg_network(option, expected)
    log.error(msg)
    raise AttributeError(msg)

def _raise_error_routes(iface, option, expected):
    if False:
        print('Hello World!')
    '\n    Log and raise an error with a logical formatted message.\n    '
    msg = _error_msg_routes(iface, option, expected)
    log.error(msg)
    raise AttributeError(msg)

def _read_file(path):
    if False:
        while True:
            i = 10
    '\n    Reads and returns the contents of a text file\n    '
    try:
        with salt.utils.files.flopen(path, 'rb') as contents:
            return [salt.utils.stringutils.to_str(line) for line in contents.readlines()]
    except OSError:
        return ''

def _parse_resolve():
    if False:
        return 10
    '\n    Parse /etc/resolv.conf\n    '
    return salt.utils.dns.parse_resolv(_DEB_RESOLV_FILE)

def _parse_domainname():
    if False:
        return 10
    '\n    Parse /etc/resolv.conf and return domainname\n    '
    return _parse_resolve().get('domain', '')

def _parse_searchdomain():
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse /etc/resolv.conf and return searchdomain\n    '
    return _parse_resolve().get('search', '')

def _parse_hostname():
    if False:
        while True:
            i = 10
    '\n    Parse /etc/hostname and return hostname\n    '
    contents = _read_file(_DEB_HOSTNAME_FILE)
    if contents:
        return contents[0].split('\n')[0]
    else:
        return ''

def _parse_current_network_settings():
    if False:
        while True:
            i = 10
    '\n    Parse /etc/default/networking and return current configuration\n    '
    opts = salt.utils.odict.OrderedDict()
    opts['networking'] = ''
    if os.path.isfile(_DEB_NETWORKING_FILE):
        with salt.utils.files.fopen(_DEB_NETWORKING_FILE) as contents:
            for line in contents:
                salt.utils.stringutils.to_unicode(line)
                if line.startswith('#'):
                    continue
                elif line.startswith('CONFIGURE_INTERFACES'):
                    opts['networking'] = line.split('=', 1)[1].strip()
    hostname = _parse_hostname()
    domainname = _parse_domainname()
    searchdomain = _parse_searchdomain()
    opts['hostname'] = hostname
    opts['domainname'] = domainname
    opts['searchdomain'] = searchdomain
    return opts

def __ipv4_quad(value):
    if False:
        return 10
    'validate an IPv4 address'
    return (salt.utils.validate.net.ipv4_addr(value), value, 'dotted IPv4 address')

def __ipv6(value):
    if False:
        while True:
            i = 10
    'validate an IPv6 address'
    return (salt.utils.validate.net.ipv6_addr(value), value, 'IPv6 address')

def __mac(value):
    if False:
        return 10
    'validate a mac address'
    return (salt.utils.validate.net.mac(value), value, 'MAC address')

def __anything(value):
    if False:
        for i in range(10):
            print('nop')
    return (True, value, None)

def __int(value):
    if False:
        for i in range(10):
            print('nop')
    'validate an integer'
    (valid, _value) = (False, value)
    try:
        _value = int(value)
        valid = True
    except ValueError:
        pass
    return (valid, _value, 'integer')

def __float(value):
    if False:
        for i in range(10):
            print('nop')
    'validate a float'
    (valid, _value) = (False, value)
    try:
        _value = float(value)
        valid = True
    except ValueError:
        pass
    return (valid, _value, 'float')

def __ipv4_netmask(value):
    if False:
        return 10
    'validate an IPv4 dotted quad or integer CIDR netmask'
    (valid, errmsg) = (False, 'dotted quad or integer CIDR (0->32)')
    (valid, value, _) = __int(value)
    if not (valid and 0 <= value <= 32):
        valid = salt.utils.validate.net.netmask(value)
    return (valid, value, errmsg)

def __ipv6_netmask(value):
    if False:
        return 10
    'validate an IPv6 integer netmask'
    (valid, errmsg) = (False, 'IPv6 netmask (0->128)')
    (valid, value, _) = __int(value)
    valid = valid and 0 <= value <= 128
    return (valid, value, errmsg)

def __within2(value, within=None, errmsg=None, dtype=None):
    if False:
        return 10
    'validate that a value is in ``within`` and optionally a ``dtype``'
    (valid, _value) = (False, value)
    if dtype:
        try:
            _value = dtype(value)
            valid = _value in within
        except ValueError:
            pass
    else:
        valid = _value in within
    if errmsg is None:
        if dtype:
            typename = getattr(dtype, '__name__', hasattr(dtype, '__class__') and getattr(dtype.__class__, 'name', dtype))
            errmsg = "{} within '{}'".format(typename, within)
        else:
            errmsg = "within '{}'".format(within)
    return (valid, _value, errmsg)

def __within(within=None, errmsg=None, dtype=None):
    if False:
        for i in range(10):
            print('nop')
    return functools.partial(__within2, within=within, errmsg=errmsg, dtype=dtype)

def __space_delimited_list(value):
    if False:
        return 10
    'validate that a value contains one or more space-delimited values'
    if isinstance(value, str):
        value = value.strip().split()
    if hasattr(value, '__iter__') and value != []:
        return (True, value, 'space-delimited string')
    else:
        return (False, value, '{} is not a valid space-delimited value.\n'.format(value))
SALT_ATTR_TO_DEBIAN_ATTR_MAP = {'dns': 'dns-nameservers', 'search': 'dns-search', 'hwaddr': 'hwaddress', 'ipaddr': 'address', 'ipaddrs': 'addresses'}
DEBIAN_ATTR_TO_SALT_ATTR_MAP = {v: k for (k, v) in SALT_ATTR_TO_DEBIAN_ATTR_MAP.items()}
DEBIAN_ATTR_TO_SALT_ATTR_MAP['address'] = 'address'
DEBIAN_ATTR_TO_SALT_ATTR_MAP['hwaddress'] = 'hwaddress'
IPV4_VALID_PROTO = ['bootp', 'dhcp', 'static', 'manual', 'loopback', 'ppp']
IPV4_ATTR_MAP = {'proto': __within(IPV4_VALID_PROTO, dtype=str), 'address': __ipv4_quad, 'addresses': __anything, 'netmask': __ipv4_netmask, 'broadcast': __ipv4_quad, 'metric': __int, 'gateway': __ipv4_quad, 'pointopoint': __ipv4_quad, 'hwaddress': __mac, 'mtu': __int, 'scope': __within(['global', 'link', 'host'], dtype=str), 'hostname': __anything, 'leasehours': __int, 'leasetime': __int, 'vendor': __anything, 'client': __anything, 'bootfile': __anything, 'server': __ipv4_quad, 'hwaddr': __mac, 'mode': __within(['gre', 'GRE', 'ipip', 'IPIP', '802.3ad'], dtype=str), 'endpoint': __ipv4_quad, 'dstaddr': __ipv4_quad, 'local': __ipv4_quad, 'ttl': __int, 'slaves': __anything, 'provider': __anything, 'unit': __int, 'options': __anything, 'dns-nameservers': __space_delimited_list, 'dns-search': __space_delimited_list, 'vlan-raw-device': __anything, 'network': __anything, 'test': __anything, 'enable_ipv4': __anything, 'enable_ipv6': __anything}
IPV6_VALID_PROTO = ['auto', 'loopback', 'static', 'manual', 'dhcp', 'v4tunnel', '6to4']
IPV6_ATTR_MAP = {'proto': __within(IPV6_VALID_PROTO), 'address': __ipv6, 'addresses': __anything, 'netmask': __ipv6_netmask, 'broadcast': __ipv6, 'gateway': __ipv6, 'hwaddress': __mac, 'mtu': __int, 'scope': __within(['global', 'site', 'link', 'host'], dtype=str), 'privext': __within([0, 1, 2], dtype=int), 'dhcp': __within([0, 1], dtype=int), 'media': __anything, 'accept_ra': __within([0, 1, 2], dtype=int), 'autoconf': __within([0, 1], dtype=int), 'preferred-lifetime': __int, 'dad-attempts': __int, 'dad-interval': __float, 'slaves': __anything, 'mode': __within(['gre', 'GRE', 'ipip', 'IPIP', '802.3ad'], dtype=str), 'endpoint': __ipv4_quad, 'local': __ipv4_quad, 'ttl': __int, 'dns-nameservers': __space_delimited_list, 'dns-search': __space_delimited_list, 'vlan-raw-device': __anything, 'test': __anything, 'enable_ipv4': __anything, 'enable_ipv6': __anything}
WIRELESS_ATTR_MAP = {'wireless-essid': __anything, 'wireless-mode': __anything, 'wpa-ap-scan': __within([0, 1, 2], dtype=int), 'wpa-conf': __anything, 'wpa-driver': __anything, 'wpa-group': __anything, 'wpa-key-mgmt': __anything, 'wpa-pairwise': __anything, 'wpa-psk': __anything, 'wpa-proto': __anything, 'wpa-roam': __anything, 'wpa-ssid': __anything}
ATTRMAPS = {'inet': [IPV4_ATTR_MAP, WIRELESS_ATTR_MAP], 'inet6': [IPV6_ATTR_MAP, WIRELESS_ATTR_MAP]}

def _validate_interface_option(attr, value, addrfam='inet'):
    if False:
        for i in range(10):
            print('nop')
    'lookup the validation function for a [addrfam][attr] and\n    return the results\n\n    :param attr: attribute name\n    :param value: raw setting value\n    :param addrfam: address family (inet, inet6,\n    '
    (valid, _value, errmsg) = (False, value, 'Unknown validator')
    attrmaps = ATTRMAPS.get(addrfam, [])
    for attrmap in attrmaps:
        if attr in attrmap:
            validate_func = attrmap[attr]
            (valid, _value, errmsg) = validate_func(value)
            break
    return (valid, _value, errmsg)

def _attrmaps_contain_attr(attr):
    if False:
        while True:
            i = 10
    return attr in WIRELESS_ATTR_MAP or attr in IPV4_ATTR_MAP or attr in IPV6_ATTR_MAP

def _parse_interfaces(interface_files=None):
    if False:
        return 10
    '\n    Parse /etc/network/interfaces and return current configured interfaces\n    '
    if interface_files is None:
        interface_files = []
        if os.path.exists(_DEB_NETWORK_DIR):
            interface_files += ['{}/{}'.format(_DEB_NETWORK_DIR, dir) for dir in os.listdir(_DEB_NETWORK_DIR)]
        if os.path.isfile(_DEB_NETWORK_FILE):
            interface_files.insert(0, _DEB_NETWORK_FILE)
    adapters = salt.utils.odict.OrderedDict()
    method = -1
    for interface_file in interface_files:
        with salt.utils.files.fopen(interface_file) as interfaces:
            iface_dict = {}
            for line in interfaces:
                line = salt.utils.stringutils.to_unicode(line)
                if line.lstrip().startswith('#') or line.isspace():
                    continue
                if line.startswith('iface'):
                    sline = line.split()
                    if len(sline) != 4:
                        msg = 'Interface file malformed: {0}.'
                        msg = msg.format(sline)
                        log.error(msg)
                        raise AttributeError(msg)
                    iface_name = sline[1]
                    addrfam = sline[2]
                    method = sline[3]
                    if iface_name not in adapters:
                        adapters[iface_name] = salt.utils.odict.OrderedDict()
                    if 'data' not in adapters[iface_name]:
                        adapters[iface_name]['data'] = salt.utils.odict.OrderedDict()
                    if addrfam not in adapters[iface_name]['data']:
                        adapters[iface_name]['data'][addrfam] = salt.utils.odict.OrderedDict()
                    iface_dict = adapters[iface_name]['data'][addrfam]
                    iface_dict['addrfam'] = addrfam
                    iface_dict['proto'] = method
                    iface_dict['filename'] = interface_file
                elif line[0].isspace():
                    sline = line.split()
                    (attr, valuestr) = line.rstrip().split(None, 1)
                    if _attrmaps_contain_attr(attr):
                        if '-' in attr:
                            attrname = attr.replace('-', '_')
                        else:
                            attrname = attr
                        (valid, value, errmsg) = _validate_interface_option(attr, valuestr, addrfam)
                        if attrname == 'address' and 'address' in iface_dict:
                            if 'addresses' not in iface_dict:
                                iface_dict['addresses'] = []
                            iface_dict['addresses'].append(value)
                        else:
                            iface_dict[attrname] = value
                    elif attr in _REV_ETHTOOL_CONFIG_OPTS:
                        if 'ethtool' not in iface_dict:
                            iface_dict['ethtool'] = salt.utils.odict.OrderedDict()
                        iface_dict['ethtool'][attr] = valuestr
                    elif attr.startswith('bond'):
                        opt = re.split('[_-]', attr, maxsplit=1)[1]
                        if 'bonding' not in iface_dict:
                            iface_dict['bonding'] = salt.utils.odict.OrderedDict()
                        iface_dict['bonding'][opt] = valuestr
                    elif attr.startswith('bridge'):
                        opt = re.split('[_-]', attr, maxsplit=1)[1]
                        if 'bridging' not in iface_dict:
                            iface_dict['bridging'] = salt.utils.odict.OrderedDict()
                        iface_dict['bridging'][opt] = valuestr
                    elif attr in ['up', 'pre-up', 'post-up', 'down', 'pre-down', 'post-down']:
                        cmd = valuestr
                        cmd_key = '{}_cmds'.format(re.sub('-', '_', attr))
                        if cmd_key not in iface_dict:
                            iface_dict[cmd_key] = []
                        iface_dict[cmd_key].append(cmd)
                elif line.startswith('auto'):
                    for word in line.split()[1:]:
                        if word not in adapters:
                            adapters[word] = salt.utils.odict.OrderedDict()
                        adapters[word]['enabled'] = True
                elif line.startswith('allow-hotplug'):
                    for word in line.split()[1:]:
                        if word not in adapters:
                            adapters[word] = salt.utils.odict.OrderedDict()
                        adapters[word]['hotplug'] = True
                elif line.startswith('source'):
                    if 'source' not in adapters:
                        adapters['source'] = salt.utils.odict.OrderedDict()
                    if 'data' not in adapters['source']:
                        adapters['source']['data'] = salt.utils.odict.OrderedDict()
                        adapters['source']['data']['sources'] = []
                    adapters['source']['data']['sources'].append(line.split()[1])
    adapters = _filter_malformed_interfaces(adapters=adapters)
    return adapters

def _filter_malformed_interfaces(*, adapters):
    if False:
        for i in range(10):
            print('nop')
    for iface_name in list(adapters):
        if iface_name == 'source':
            continue
        if 'data' not in adapters[iface_name]:
            msg = 'Interface file malformed for interface: {}.'.format(iface_name)
            log.error(msg)
            adapters.pop(iface_name)
            continue
        for opt in ['ethtool', 'bonding', 'bridging']:
            for inet in ['inet', 'inet6']:
                if inet in adapters[iface_name]['data']:
                    if opt in adapters[iface_name]['data'][inet]:
                        opt_keys = sorted(adapters[iface_name]['data'][inet][opt].keys())
                        adapters[iface_name]['data'][inet][opt + '_keys'] = opt_keys
    return adapters

def _parse_ethtool_opts(opts, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for ETHTOOLS_OPTS\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    config = {}
    if 'autoneg' in opts:
        if opts['autoneg'] in _CONFIG_TRUE:
            config.update({'autoneg': 'on'})
        elif opts['autoneg'] in _CONFIG_FALSE:
            config.update({'autoneg': 'off'})
        else:
            _raise_error_iface(iface, 'autoneg', _CONFIG_TRUE + _CONFIG_FALSE)
    if 'duplex' in opts:
        valid = ['full', 'half']
        if opts['duplex'] in valid:
            config.update({'duplex': opts['duplex']})
        else:
            _raise_error_iface(iface, 'duplex', valid)
    if 'speed' in opts:
        valid = ['10', '100', '1000', '10000']
        if str(opts['speed']) in valid:
            config.update({'speed': opts['speed']})
        else:
            _raise_error_iface(iface, opts['speed'], valid)
    valid = _CONFIG_TRUE + _CONFIG_FALSE
    for option in ('rx', 'tx', 'sg', 'tso', 'ufo', 'gso', 'gro', 'lro'):
        if option in opts:
            if opts[option] in _CONFIG_TRUE:
                config.update({option: 'on'})
            elif opts[option] in _CONFIG_FALSE:
                config.update({option: 'off'})
            else:
                _raise_error_iface(iface, option, valid)
    return config

def _parse_ethtool_pppoe_opts(opts, iface):
    if False:
        return 10
    '\n    Filters given options and outputs valid settings for ETHTOOLS_PPPOE_OPTS\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    config = {}
    for opt in _DEB_CONFIG_PPPOE_OPTS:
        if opt in opts:
            config[opt] = opts[opt]
    if 'provider' in opts and (not opts['provider']):
        _raise_error_iface(iface, 'provider', _CONFIG_TRUE + _CONFIG_FALSE)
    valid = _CONFIG_TRUE + _CONFIG_FALSE
    for option in ('noipdefault', 'usepeerdns', 'defaultroute', 'hide-password', 'noauth', 'persist', 'noaccomp'):
        if option in opts:
            if opts[option] in _CONFIG_TRUE:
                config.update({option: 'True'})
            elif opts[option] in _CONFIG_FALSE:
                config.update({option: 'False'})
            else:
                _raise_error_iface(iface, option, valid)
    return config

def _parse_settings_bond(opts, iface):
    if False:
        return 10
    '\n    Filters given options and outputs valid settings for requested\n    operation. If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond_def = {'ad_select': '0', 'tx_queues': '16', 'miimon': '100', 'arp_interval': '250', 'downdelay': '200', 'lacp_rate': '0', 'max_bonds': '1', 'updelay': '0', 'use_carrier': 'on', 'xmit_hash_policy': 'layer2'}
    if opts['mode'] in ['balance-rr', '0']:
        log.info('Device: %s Bonding Mode: load balancing (round-robin)', iface)
        return _parse_settings_bond_0(opts, iface, bond_def)
    elif opts['mode'] in ['active-backup', '1']:
        log.info('Device: %s Bonding Mode: fault-tolerance (active-backup)', iface)
        return _parse_settings_bond_1(opts, iface, bond_def)
    elif opts['mode'] in ['balance-xor', '2']:
        log.info('Device: %s Bonding Mode: load balancing (xor)', iface)
        return _parse_settings_bond_2(opts, iface, bond_def)
    elif opts['mode'] in ['broadcast', '3']:
        log.info('Device: %s Bonding Mode: fault-tolerance (broadcast)', iface)
        return _parse_settings_bond_3(opts, iface, bond_def)
    elif opts['mode'] in ['802.3ad', '4']:
        log.info('Device: %s Bonding Mode: IEEE 802.3ad Dynamic link aggregation', iface)
        return _parse_settings_bond_4(opts, iface, bond_def)
    elif opts['mode'] in ['balance-tlb', '5']:
        log.info('Device: %s Bonding Mode: transmit load balancing', iface)
        return _parse_settings_bond_5(opts, iface, bond_def)
    elif opts['mode'] in ['balance-alb', '6']:
        log.info('Device: %s Bonding Mode: adaptive load balancing', iface)
        return _parse_settings_bond_6(opts, iface, bond_def)
    else:
        valid = ['0', '1', '2', '3', '4', '5', '6', 'balance-rr', 'active-backup', 'balance-xor', 'broadcast', '802.3ad', 'balance-tlb', 'balance-alb']
        _raise_error_iface(iface, 'mode', valid)

def _parse_settings_bond_0(opts, iface, bond_def):
    if False:
        return 10
    '\n    Filters given options and outputs valid settings for bond0.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '0'}
    valid = ['list of ips (up to 16)']
    if 'arp_ip_target' in opts:
        if isinstance(opts['arp_ip_target'], list):
            if 1 <= len(opts['arp_ip_target']) <= 16:
                bond.update({'arp_ip_target': ''})
                for ip in opts['arp_ip_target']:
                    if len(bond['arp_ip_target']) > 0:
                        bond['arp_ip_target'] = bond['arp_ip_target'] + ',' + ip
                    else:
                        bond['arp_ip_target'] = ip
            else:
                _raise_error_iface(iface, 'arp_ip_target', valid)
        else:
            _raise_error_iface(iface, 'arp_ip_target', valid)
    else:
        _raise_error_iface(iface, 'arp_ip_target', valid)
    if 'arp_interval' in opts:
        try:
            int(opts['arp_interval'])
            bond.update({'arp_interval': opts['arp_interval']})
        except ValueError:
            _raise_error_iface(iface, 'arp_interval', ['integer'])
    else:
        _log_default_iface(iface, 'arp_interval', bond_def['arp_interval'])
        bond.update({'arp_interval': bond_def['arp_interval']})
    return bond

def _parse_settings_bond_1(opts, iface, bond_def):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for bond1.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '1'}
    for binding in ['miimon', 'downdelay', 'updelay']:
        if binding in opts:
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, ['integer'])
        else:
            _log_default_iface(iface, binding, bond_def[binding])
            bond.update({binding: bond_def[binding]})
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    if not (__grains__['os'] == 'Ubuntu' and __grains__['osrelease_info'][0] >= 16):
        if 'use_carrier' in opts:
            if opts['use_carrier'] in _CONFIG_TRUE:
                bond.update({'use_carrier': '1'})
            elif opts['use_carrier'] in _CONFIG_FALSE:
                bond.update({'use_carrier': '0'})
            else:
                valid = _CONFIG_TRUE + _CONFIG_FALSE
                _raise_error_iface(iface, 'use_carrier', valid)
        else:
            _log_default_iface(iface, 'use_carrier', bond_def['use_carrier'])
            bond.update({'use_carrier': bond_def['use_carrier']})
    return bond

def _parse_settings_bond_2(opts, iface, bond_def):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for bond2.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '2'}
    valid = ['list of ips (up to 16)']
    if 'arp_ip_target' in opts:
        if isinstance(opts['arp_ip_target'], list):
            if 1 <= len(opts['arp_ip_target']) <= 16:
                bond.update({'arp_ip_target': ''})
                for ip in opts['arp_ip_target']:
                    if len(bond['arp_ip_target']) > 0:
                        bond['arp_ip_target'] = bond['arp_ip_target'] + ',' + ip
                    else:
                        bond['arp_ip_target'] = ip
            else:
                _raise_error_iface(iface, 'arp_ip_target', valid)
        else:
            _raise_error_iface(iface, 'arp_ip_target', valid)
    else:
        _raise_error_iface(iface, 'arp_ip_target', valid)
    if 'arp_interval' in opts:
        try:
            int(opts['arp_interval'])
            bond.update({'arp_interval': opts['arp_interval']})
        except ValueError:
            _raise_error_iface(iface, 'arp_interval', ['integer'])
    else:
        _log_default_iface(iface, 'arp_interval', bond_def['arp_interval'])
        bond.update({'arp_interval': bond_def['arp_interval']})
    if 'hashing-algorithm' in opts:
        valid = ['layer2', 'layer2+3', 'layer3+4']
        if opts['hashing-algorithm'] in valid:
            bond.update({'xmit_hash_policy': opts['hashing-algorithm']})
        else:
            _raise_error_iface(iface, 'hashing-algorithm', valid)
    return bond

def _parse_settings_bond_3(opts, iface, bond_def):
    if False:
        print('Hello World!')
    '\n    Filters given options and outputs valid settings for bond3.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '3'}
    for binding in ['miimon', 'downdelay', 'updelay']:
        if binding in opts:
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, ['integer'])
        else:
            _log_default_iface(iface, binding, bond_def[binding])
            bond.update({binding: bond_def[binding]})
    if 'use_carrier' in opts:
        if opts['use_carrier'] in _CONFIG_TRUE:
            bond.update({'use_carrier': '1'})
        elif opts['use_carrier'] in _CONFIG_FALSE:
            bond.update({'use_carrier': '0'})
        else:
            valid = _CONFIG_TRUE + _CONFIG_FALSE
            _raise_error_iface(iface, 'use_carrier', valid)
    else:
        _log_default_iface(iface, 'use_carrier', bond_def['use_carrier'])
        bond.update({'use_carrier': bond_def['use_carrier']})
    return bond

def _parse_settings_bond_4(opts, iface, bond_def):
    if False:
        i = 10
        return i + 15
    '\n    Filters given options and outputs valid settings for bond4.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '4'}
    for binding in ['miimon', 'downdelay', 'updelay', 'lacp_rate', 'ad_select']:
        if binding in opts:
            if binding == 'lacp_rate':
                if opts[binding] == 'fast':
                    opts.update({binding: '1'})
                if opts[binding] == 'slow':
                    opts.update({binding: '0'})
                valid = ['fast', '1', 'slow', '0']
            else:
                valid = ['integer']
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, valid)
        else:
            _log_default_iface(iface, binding, bond_def[binding])
            bond.update({binding: bond_def[binding]})
    if 'use_carrier' in opts:
        if opts['use_carrier'] in _CONFIG_TRUE:
            bond.update({'use_carrier': '1'})
        elif opts['use_carrier'] in _CONFIG_FALSE:
            bond.update({'use_carrier': '0'})
        else:
            valid = _CONFIG_TRUE + _CONFIG_FALSE
            _raise_error_iface(iface, 'use_carrier', valid)
    else:
        _log_default_iface(iface, 'use_carrier', bond_def['use_carrier'])
        bond.update({'use_carrier': bond_def['use_carrier']})
    if 'hashing-algorithm' in opts:
        valid = ['layer2', 'layer2+3', 'layer3+4']
        if opts['hashing-algorithm'] in valid:
            bond.update({'xmit_hash_policy': opts['hashing-algorithm']})
        else:
            _raise_error_iface(iface, 'hashing-algorithm', valid)
    return bond

def _parse_settings_bond_5(opts, iface, bond_def):
    if False:
        while True:
            i = 10
    '\n    Filters given options and outputs valid settings for bond5.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '5'}
    for binding in ['miimon', 'downdelay', 'updelay']:
        if binding in opts:
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, ['integer'])
        else:
            _log_default_iface(iface, binding, bond_def[binding])
            bond.update({binding: bond_def[binding]})
    if 'use_carrier' in opts:
        if opts['use_carrier'] in _CONFIG_TRUE:
            bond.update({'use_carrier': '1'})
        elif opts['use_carrier'] in _CONFIG_FALSE:
            bond.update({'use_carrier': '0'})
        else:
            valid = _CONFIG_TRUE + _CONFIG_FALSE
            _raise_error_iface(iface, 'use_carrier', valid)
    else:
        _log_default_iface(iface, 'use_carrier', bond_def['use_carrier'])
        bond.update({'use_carrier': bond_def['use_carrier']})
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    return bond

def _parse_settings_bond_6(opts, iface, bond_def):
    if False:
        while True:
            i = 10
    '\n    Filters given options and outputs valid settings for bond6.\n    If an option has a value that is not expected, this\n    function will log what the Interface, Setting and what it was\n    expecting.\n    '
    bond = {'mode': '6'}
    for binding in ['miimon', 'downdelay', 'updelay']:
        if binding in opts:
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, ['integer'])
        else:
            _log_default_iface(iface, binding, bond_def[binding])
            bond.update({binding: bond_def[binding]})
    if 'use_carrier' in opts:
        if opts['use_carrier'] in _CONFIG_TRUE:
            bond.update({'use_carrier': '1'})
        elif opts['use_carrier'] in _CONFIG_FALSE:
            bond.update({'use_carrier': '0'})
        else:
            valid = _CONFIG_TRUE + _CONFIG_FALSE
            _raise_error_iface(iface, 'use_carrier', valid)
    else:
        _log_default_iface(iface, 'use_carrier', bond_def['use_carrier'])
        bond.update({'use_carrier': bond_def['use_carrier']})
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    return bond

def _parse_bridge_opts(opts, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for BRIDGING_OPTS\n    If an option has a value that is not expected, this\n    function will log the Interface, Setting and what was expected.\n    '
    config = {}
    if 'ports' in opts:
        if isinstance(opts['ports'], list):
            opts['ports'] = ' '.join(opts['ports'])
        config.update({'ports': opts['ports']})
    for opt in ['ageing', 'fd', 'gcint', 'hello', 'maxage']:
        if opt in opts:
            try:
                float(opts[opt])
                config.update({opt: opts[opt]})
            except ValueError:
                _raise_error_iface(iface, opt, ['float'])
    for opt in ['bridgeprio', 'maxwait']:
        if opt in opts:
            if isinstance(opts[opt], int):
                config.update({opt: opts[opt]})
            else:
                _raise_error_iface(iface, opt, ['integer'])
    if 'hw' in opts:
        if re.match('[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$', opts['hw'].lower()):
            config.update({'hw': opts['hw']})
        else:
            _raise_error_iface(iface, 'hw', ['valid MAC address'])
    for opt in ['pathcost', 'portprio']:
        if opt in opts:
            try:
                (port, cost_or_prio) = opts[opt].split()
                int(cost_or_prio)
                config.update({opt: '{} {}'.format(port, cost_or_prio)})
            except ValueError:
                _raise_error_iface(iface, opt, ['interface integer'])
    if 'stp' in opts:
        if opts['stp'] in _CONFIG_TRUE:
            config.update({'stp': 'on'})
        elif opts['stp'] in _CONFIG_FALSE:
            config.update({'stp': 'off'})
        else:
            _raise_error_iface(iface, 'stp', _CONFIG_TRUE + _CONFIG_FALSE)
    if 'waitport' in opts:
        if isinstance(opts['waitport'], int):
            config.update({'waitport': opts['waitport']})
        else:
            values = opts['waitport'].split()
            waitport_time = values.pop(0)
            if waitport_time.isdigit() and values:
                config.update({'waitport': '{} {}'.format(waitport_time, ' '.join(values))})
            else:
                _raise_error_iface(iface, opt, ['integer [interfaces]'])
    return config

def _parse_settings_eth(opts, iface_type, enabled, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for a\n    network interface.\n    '
    adapters = salt.utils.odict.OrderedDict()
    adapters[iface] = salt.utils.odict.OrderedDict()
    adapters[iface]['type'] = iface_type
    adapters[iface]['data'] = salt.utils.odict.OrderedDict()
    iface_data = adapters[iface]['data']
    iface_data['inet'] = salt.utils.odict.OrderedDict()
    iface_data['inet6'] = salt.utils.odict.OrderedDict()
    if enabled:
        adapters[iface]['enabled'] = True
    if opts.get('hotplug', False):
        adapters[iface]['hotplug'] = True
    if opts.get('enable_ipv6', None) and opts.get('iface_type', '') == 'vlan':
        iface_data['inet6']['vlan_raw_device'] = re.sub('\\.\\d*', '', iface)
    for addrfam in ['inet', 'inet6']:
        if iface_type not in ['bridge']:
            tmp_ethtool = _parse_ethtool_opts(opts, iface)
            if tmp_ethtool:
                ethtool = {}
                for item in tmp_ethtool:
                    ethtool[_ETHTOOL_CONFIG_OPTS[item]] = tmp_ethtool[item]
                iface_data[addrfam]['ethtool'] = ethtool
                iface_data[addrfam]['ethtool_keys'] = sorted(ethtool)
        if iface_type == 'bridge':
            bridging = _parse_bridge_opts(opts, iface)
            if bridging:
                iface_data[addrfam]['bridging'] = bridging
                iface_data[addrfam]['bridging_keys'] = sorted(bridging)
                iface_data[addrfam]['addrfam'] = addrfam
        elif iface_type == 'bond':
            bonding = _parse_settings_bond(opts, iface)
            if bonding:
                iface_data[addrfam]['bonding'] = bonding
                iface_data[addrfam]['bonding']['slaves'] = opts['slaves']
                iface_data[addrfam]['bonding_keys'] = sorted(bonding)
                iface_data[addrfam]['addrfam'] = addrfam
        elif iface_type == 'slave':
            adapters[iface]['master'] = opts['master']
            opts['proto'] = 'manual'
            iface_data[addrfam]['master'] = adapters[iface]['master']
            iface_data[addrfam]['addrfam'] = addrfam
        elif iface_type == 'vlan':
            iface_data[addrfam]['vlan_raw_device'] = re.sub('\\.\\d*', '', iface)
            iface_data[addrfam]['addrfam'] = addrfam
        elif iface_type == 'pppoe':
            tmp_ethtool = _parse_ethtool_pppoe_opts(opts, iface)
            if tmp_ethtool:
                for item in tmp_ethtool:
                    adapters[iface]['data'][addrfam][_DEB_CONFIG_PPPOE_OPTS[item]] = tmp_ethtool[item]
            iface_data[addrfam]['addrfam'] = addrfam
    opts.pop('mode', None)
    for (opt, val) in opts.items():
        inet = None
        if opt.startswith('ipv4'):
            opt = opt[4:]
            inet = 'inet'
            iface_data['inet']['addrfam'] = 'inet'
        elif opt.startswith('ipv6'):
            iface_data['inet6']['addrfam'] = 'inet6'
            opt = opt[4:]
            inet = 'inet6'
        elif opt in ['ipaddr', 'address', 'ipaddresses', 'addresses', 'gateway', 'proto']:
            iface_data['inet']['addrfam'] = 'inet'
            inet = 'inet'
        _opt = SALT_ATTR_TO_DEBIAN_ATTR_MAP.get(opt, opt)
        _debopt = _opt.replace('-', '_')
        for addrfam in ['inet', 'inet6']:
            (valid, value, errmsg) = _validate_interface_option(_opt, val, addrfam=addrfam)
            if not valid:
                continue
            if inet is None and _debopt not in iface_data[addrfam]:
                iface_data[addrfam][_debopt] = value
            elif inet == addrfam:
                iface_data[addrfam][_debopt] = value
    for opt in ['up_cmds', 'pre_up_cmds', 'post_up_cmds', 'down_cmds', 'pre_down_cmds', 'post_down_cmds']:
        if opt in opts:
            iface_data['inet'][opt] = opts[opt]
            iface_data['inet6'][opt] = opts[opt]
    for (addrfam, opt) in [('inet', 'enable_ipv4'), ('inet6', 'enable_ipv6')]:
        if opts.get(opt, None) is False:
            iface_data.pop(addrfam)
        elif iface_data[addrfam].get('addrfam', '') != addrfam:
            iface_data.pop(addrfam)
    return adapters

def _parse_settings_source(opts, iface_type, enabled, iface):
    if False:
        return 10
    '\n    Filters given options and outputs valid settings for a\n    network interface.\n    '
    adapters = salt.utils.odict.OrderedDict()
    adapters[iface] = salt.utils.odict.OrderedDict()
    adapters[iface]['type'] = iface_type
    adapters[iface]['data'] = salt.utils.odict.OrderedDict()
    iface_data = adapters[iface]['data']
    iface_data['sources'] = [opts['source']]
    return adapters

def _parse_network_settings(opts, current):
    if False:
        i = 10
        return i + 15
    '\n    Filters given options and outputs valid settings for\n    the global network settings file.\n    '
    opts = {k.lower(): v for (k, v) in opts.items()}
    current = {k.lower(): v for (k, v) in current.items()}
    result = {}
    valid = _CONFIG_TRUE + _CONFIG_FALSE
    if 'enabled' not in opts:
        try:
            opts['networking'] = current['networking']
            _log_default_network('networking', current['networking'])
        except ValueError:
            _raise_error_network('networking', valid)
    else:
        opts['networking'] = opts['enabled']
    if opts['networking'] in valid:
        if opts['networking'] in _CONFIG_TRUE:
            result['networking'] = 'yes'
        elif opts['networking'] in _CONFIG_FALSE:
            result['networking'] = 'no'
    else:
        _raise_error_network('networking', valid)
    if 'hostname' not in opts:
        try:
            opts['hostname'] = current['hostname']
            _log_default_network('hostname', current['hostname'])
        except ValueError:
            _raise_error_network('hostname', ['server1.example.com'])
    if opts['hostname']:
        result['hostname'] = opts['hostname']
    else:
        _raise_error_network('hostname', ['server1.example.com'])
    if 'search' in opts:
        result['search'] = opts['search']
    return result

def _parse_routes(iface, opts):
    if False:
        i = 10
        return i + 15
    '\n    Filters given options and outputs valid settings for\n    the route settings file.\n    '
    opts = {k.lower(): v for (k, v) in opts.items()}
    result = {}
    if 'routes' not in opts:
        _raise_error_routes(iface, 'routes', 'List of routes')
    for opt in opts:
        result[opt] = opts[opt]
    return result

def _write_file(iface, data, folder, pattern):
    if False:
        print('Hello World!')
    '\n    Writes a file to disk\n    '
    filename = os.path.join(folder, pattern.format(iface))
    if not os.path.exists(folder):
        msg = '{0} cannot be written. {1} does not exist'
        msg = msg.format(filename, folder)
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.flopen(filename, 'w') as fout:
        fout.write(salt.utils.stringutils.to_str(data))
    return filename

def _write_file_routes(iface, data, folder, pattern):
    if False:
        return 10
    '\n    Writes a file to disk\n    '
    iface = iface.replace('.', '_')
    filename = os.path.join(folder, pattern.format(iface))
    if not os.path.exists(folder):
        msg = '{0} cannot be written. {1} does not exist'
        msg = msg.format(filename, folder)
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.flopen(filename, 'w') as fout:
        fout.write(salt.utils.stringutils.to_str(data))
    __salt__['file.set_mode'](filename, '0755')
    return filename

def _write_file_network(data, filename, create=False):
    if False:
        return 10
    '\n    Writes a file to disk\n    If file does not exist, only create if create\n    argument is True\n    '
    if not os.path.exists(filename) and (not create):
        msg = '{0} cannot be written. {0} does not exist and create is setto False'.format(filename)
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.flopen(filename, 'w') as fout:
        fout.write(salt.utils.stringutils.to_str(data))

def _read_temp(data):
    if False:
        print('Hello World!')
    '\n    Return what would be written to disk\n    '
    tout = io.StringIO()
    tout.write(data)
    tout.seek(0)
    output = tout.readlines()
    tout.close()
    return output

def _read_temp_ifaces(iface, data):
    if False:
        i = 10
        return i + 15
    '\n    Return what would be written to disk for interfaces\n    '
    try:
        template = JINJA.get_template('debian_eth.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template debian_eth.jinja')
        return ''
    ifcfg = template.render({'name': iface, 'data': data})
    return [item + '\n' for item in ifcfg.split('\n')]

def _write_file_ifaces(iface, data, **settings):
    if False:
        while True:
            i = 10
    '\n    Writes a file to disk\n    '
    try:
        eth_template = JINJA.get_template('debian_eth.jinja')
        source_template = JINJA.get_template('debian_source.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template debian_eth.jinja')
        return ''
    adapters = _parse_interfaces()
    adapters[iface] = data
    ifcfg = ''
    for adapter in adapters:
        if 'type' in adapters[adapter] and adapters[adapter]['type'] == 'source':
            tmp = source_template.render({'name': adapter, 'data': adapters[adapter]})
        else:
            tmp = eth_template.render({'name': adapter, 'data': adapters[adapter]})
        ifcfg = ifcfg + tmp
        if adapter == iface:
            saved_ifcfg = tmp
    _SEPARATE_FILE = False
    if 'filename' in settings:
        if not settings['filename'].startswith('/'):
            filename = '{}/{}'.format(_DEB_NETWORK_DIR, settings['filename'])
        else:
            filename = settings['filename']
        _SEPARATE_FILE = True
    elif 'filename' in adapters[adapter]['data']:
        filename = adapters[adapter]['data']
    else:
        filename = _DEB_NETWORK_FILE
    if not os.path.exists(os.path.dirname(filename)):
        msg = '{0} cannot be written.'
        msg = msg.format(os.path.dirname(filename))
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.flopen(filename, 'w') as fout:
        if _SEPARATE_FILE:
            fout.write(salt.utils.stringutils.to_str(saved_ifcfg))
        else:
            fout.write(salt.utils.stringutils.to_str(ifcfg))
    return saved_ifcfg.split('\n')

def _write_file_ppp_ifaces(iface, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Writes a file to disk\n    '
    try:
        template = JINJA.get_template('debian_ppp_eth.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template debian_ppp_eth.jinja')
        return ''
    adapters = _parse_interfaces()
    adapters[iface] = data
    ifcfg = ''
    tmp = template.render({'data': adapters[iface]})
    ifcfg = tmp + ifcfg
    filename = _DEB_PPP_DIR + '/' + adapters[iface]['data']['inet']['provider']
    if not os.path.exists(os.path.dirname(filename)):
        msg = '{0} cannot be written.'
        msg = msg.format(os.path.dirname(filename))
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.fopen(filename, 'w') as fout:
        fout.write(salt.utils.stringutils.to_str(ifcfg))
    return filename

def build_bond(iface, **settings):
    if False:
        return 10
    "\n    Create a bond script in /etc/modprobe.d with the passed settings\n    and load the bonding kernel module.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_bond bond0 mode=balance-alb\n    "
    deb_major = __grains__['osrelease'][:1]
    opts = _parse_settings_bond(settings, iface)
    try:
        template = JINJA.get_template('conf.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template conf.jinja')
        return ''
    data = template.render({'name': iface, 'bonding': opts})
    if 'test' in settings and settings['test']:
        return _read_temp(data)
    _write_file(iface, data, _DEB_NETWORK_CONF_FILES, '{}.conf'.format(iface))
    path = os.path.join(_DEB_NETWORK_CONF_FILES, '{}.conf'.format(iface))
    if deb_major == '5':
        for line_type in ('alias', 'options'):
            cmd = ['sed', '-i', '-e', '/^{}\\s{}.*/d'.format(line_type, iface), '/etc/modprobe.conf']
            __salt__['cmd.run'](cmd, python_shell=False)
        __salt__['file.append']('/etc/modprobe.conf', path)
    __salt__['kmod.load']('bonding')
    __salt__['pkg.install']('ifenslave')
    return _read_file(path)

def build_interface(iface, iface_type, enabled, **settings):
    if False:
        i = 10
        return i + 15
    "\n    Build an interface script for a network interface.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_interface eth0 eth <settings>\n    "
    iface_type = iface_type.lower()
    if iface_type not in _IFACE_TYPES:
        _raise_error_iface(iface, iface_type, _IFACE_TYPES)
    if iface_type == 'slave':
        settings['slave'] = 'yes'
        if 'master' not in settings:
            msg = 'master is a required setting for slave interfaces'
            log.error(msg)
            raise AttributeError(msg)
    elif iface_type == 'vlan':
        settings['vlan'] = 'yes'
        __salt__['pkg.install']('vlan')
    elif iface_type == 'pppoe':
        settings['pppoe'] = 'yes'
        if not __salt__['pkg.version']('ppp'):
            inst = __salt__['pkg.install']('ppp')
    elif iface_type == 'bond':
        if 'slaves' not in settings:
            msg = 'slaves is a required setting for bond interfaces'
            log.error(msg)
            raise AttributeError(msg)
    elif iface_type == 'bridge':
        if 'ports' not in settings:
            msg = 'ports is a required setting for bridge interfaces on Debian or Ubuntu based systems'
            log.error(msg)
            raise AttributeError(msg)
        __salt__['pkg.install']('bridge-utils')
    if iface_type in ['eth', 'bond', 'bridge', 'slave', 'vlan', 'pppoe']:
        opts = _parse_settings_eth(settings, iface_type, enabled, iface)
    if iface_type in ['source']:
        opts = _parse_settings_source(settings, iface_type, enabled, iface)
    if 'test' in settings and settings['test']:
        return _read_temp_ifaces(iface, opts[iface])
    ifcfg = _write_file_ifaces(iface, opts[iface], **settings)
    if iface_type == 'pppoe':
        _write_file_ppp_ifaces(iface, opts[iface])
    return [item + '\n' for item in ifcfg]

def build_routes(iface, **settings):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add route scripts for a network interface using up commands.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_routes eth0 <settings>\n    "
    opts = _parse_routes(iface, settings)
    try:
        template = JINJA.get_template('route_eth.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template route_eth.jinja')
        return ''
    add_routecfg = template.render(route_type='add', routes=opts['routes'], iface=iface)
    del_routecfg = template.render(route_type='del', routes=opts['routes'], iface=iface)
    if 'test' in settings and settings['test']:
        return _read_temp(add_routecfg + del_routecfg)
    filename = _write_file_routes(iface, add_routecfg, _DEB_NETWORK_UP_DIR, 'route-{0}')
    results = _read_file(filename)
    filename = _write_file_routes(iface, del_routecfg, _DEB_NETWORK_DOWN_DIR, 'route-{0}')
    results += _read_file(filename)
    return results

def down(iface, iface_type):
    if False:
        print('Hello World!')
    "\n    Shutdown a network interface\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.down eth0 eth\n    "
    if iface_type not in ['slave', 'source']:
        return __salt__['cmd.run'](['ifdown', iface])
    return None

def get_bond(iface):
    if False:
        i = 10
        return i + 15
    "\n    Return the content of a bond script\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_bond bond0\n    "
    path = os.path.join(_DEB_NETWORK_CONF_FILES, '{}.conf'.format(iface))
    return _read_file(path)

def get_interface(iface):
    if False:
        while True:
            i = 10
    "\n    Return the contents of an interface script\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_interface eth0\n    "
    adapters = _parse_interfaces()
    if iface in adapters:
        try:
            if iface == 'source':
                template = JINJA.get_template('debian_source.jinja')
            else:
                template = JINJA.get_template('debian_eth.jinja')
        except jinja2.exceptions.TemplateNotFound:
            log.error('Could not load template debian_eth.jinja')
            return ''
        ifcfg = template.render({'name': iface, 'data': adapters[iface]})
        return [item + '\n' for item in ifcfg.split('\n')]
    else:
        return []

def up(iface, iface_type):
    if False:
        print('Hello World!')
    "\n    Start up a network interface\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.up eth0 eth\n    "
    if iface_type not in ('slave', 'source'):
        return __salt__['cmd.run'](['ifup', iface])
    return None

def get_network_settings():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the contents of the global network script.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_network_settings\n    "
    skip_etc_default_networking = __grains__['osfullname'] == 'Ubuntu' and int(__grains__['osrelease'].split('.')[0]) >= 12
    if skip_etc_default_networking:
        settings = {}
        if __salt__['service.available']('networking'):
            if __salt__['service.status']('networking'):
                settings['networking'] = 'yes'
            else:
                settings['networking'] = 'no'
        else:
            settings['networking'] = 'no'
        hostname = _parse_hostname()
        domainname = _parse_domainname()
        settings['hostname'] = hostname
        settings['domainname'] = domainname
    else:
        settings = _parse_current_network_settings()
    try:
        template = JINJA.get_template('display-network.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template display-network.jinja')
        return ''
    network = template.render(settings)
    return _read_temp(network)

def get_routes(iface):
    if False:
        print('Hello World!')
    "\n    Return the routes for the interface\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_routes eth0\n    "
    filename = os.path.join(_DEB_NETWORK_UP_DIR, 'route-{}'.format(iface))
    results = _read_file(filename)
    filename = os.path.join(_DEB_NETWORK_DOWN_DIR, 'route-{}'.format(iface))
    results += _read_file(filename)
    return results

def apply_network_settings(**settings):
    if False:
        print('Hello World!')
    "\n    Apply global network configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.apply_network_settings\n    "
    if 'require_reboot' not in settings:
        settings['require_reboot'] = False
    if 'apply_hostname' not in settings:
        settings['apply_hostname'] = False
    hostname_res = True
    if settings['apply_hostname'] in _CONFIG_TRUE:
        if 'hostname' in settings:
            hostname_res = __salt__['network.mod_hostname'](settings['hostname'])
        else:
            log.warning('The network state sls is trying to apply hostname changes but no hostname is defined.')
            hostname_res = False
    res = True
    if settings['require_reboot'] in _CONFIG_TRUE:
        log.warning('The network state sls is requiring a reboot of the system to properly apply network configuration.')
        res = True
    else:
        stop = __salt__['service.stop']('networking')
        time.sleep(2)
        res = stop and __salt__['service.start']('networking')
    return hostname_res and res

def build_network_settings(**settings):
    if False:
        i = 10
        return i + 15
    "\n    Build the global network script.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_network_settings <settings>\n    "
    changes = []
    current_network_settings = _parse_current_network_settings()
    opts = _parse_network_settings(settings, current_network_settings)
    skip_etc_default_networking = __grains__['osfullname'] == 'Ubuntu' and int(__grains__['osrelease'].split('.')[0]) >= 12
    if skip_etc_default_networking:
        if opts['networking'] == 'yes':
            service_cmd = 'service.enable'
        else:
            service_cmd = 'service.disable'
        if __salt__['service.available']('NetworkManager'):
            __salt__[service_cmd]('NetworkManager')
        if __salt__['service.available']('networking'):
            __salt__[service_cmd]('networking')
    else:
        try:
            template = JINJA.get_template('network.jinja')
        except jinja2.exceptions.TemplateNotFound:
            log.error('Could not load template network.jinja')
            return ''
        network = template.render(opts)
        if 'test' in settings and settings['test']:
            return _read_temp(network)
        _write_file_network(network, _DEB_NETWORKING_FILE, True)
    sline = opts['hostname'].split('.', 1)
    opts['hostname'] = sline[0]
    current_domainname = current_network_settings['domainname']
    current_searchdomain = current_network_settings['searchdomain']
    new_domain = False
    if len(sline) > 1:
        new_domainname = sline[1]
        if new_domainname != current_domainname:
            domainname = new_domainname
            opts['domainname'] = new_domainname
            new_domain = True
        else:
            domainname = current_domainname
            opts['domainname'] = domainname
    else:
        domainname = current_domainname
        opts['domainname'] = domainname
    new_search = False
    if 'search' in opts:
        new_searchdomain = opts['search']
        if new_searchdomain != current_searchdomain:
            searchdomain = new_searchdomain
            opts['searchdomain'] = new_searchdomain
            new_search = True
        else:
            searchdomain = current_searchdomain
            opts['searchdomain'] = searchdomain
    else:
        searchdomain = current_searchdomain
        opts['searchdomain'] = searchdomain
    if new_domain or new_search:
        resolve = _parse_resolve()
        domain_prog = re.compile('domain\\s+')
        search_prog = re.compile('search\\s+')
        new_contents = []
        for item in _read_file(_DEB_RESOLV_FILE):
            if domain_prog.match(item):
                item = 'domain {}'.format(domainname)
            elif search_prog.match(item):
                item = 'search {}'.format(searchdomain)
            new_contents.append(item)
        if 'domain' not in resolve:
            new_contents.insert(0, 'domain {}'.format(domainname))
        if 'search' not in resolve:
            new_contents.insert('domain' in resolve, 'search {}'.format(searchdomain))
        new_resolv = '\n'.join(new_contents)
        if not ('test' in settings and settings['test']):
            _write_file_network(new_resolv, _DEB_RESOLV_FILE)
    try:
        template = JINJA.get_template('display-network.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template display-network.jinja')
        return ''
    network = template.render(opts)
    changes.extend(_read_temp(network))
    return changes