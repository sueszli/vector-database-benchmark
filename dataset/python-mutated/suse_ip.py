"""
The networking module for SUSE based distros

.. versionadded:: 3005

"""
import logging
import os
import jinja2
import jinja2.exceptions
import salt.utils.files
import salt.utils.stringutils
import salt.utils.templates
import salt.utils.validate.net
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
JINJA = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(salt.utils.templates.TEMPLATE_DIRNAME, 'suse_ip')))
__virtualname__ = 'ip'
_BOND_DEFAULTS = {'ad_select': '0', 'tx_queues': '16', 'lacp_rate': '0', 'max_bonds': '1', 'use_carrier': '0', 'xmit_hash_policy': 'layer2'}
_SUSE_NETWORK_SCRIPT_DIR = '/etc/sysconfig/network'
_SUSE_NETWORK_FILE = '/etc/sysconfig/network/config'
_SUSE_NETWORK_ROUTES_FILE = '/etc/sysconfig/network/routes'
_CONFIG_TRUE = ('yes', 'on', 'true', '1', True)
_CONFIG_FALSE = ('no', 'off', 'false', '0', False)
_IFACE_TYPES = ('eth', 'bond', 'alias', 'clone', 'ipsec', 'dialup', 'bridge', 'slave', 'vlan', 'ipip', 'ib')

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Confine this module to SUSE based distros\n    '
    if __grains__['os_family'] == 'Suse':
        return __virtualname__
    return (False, 'The suse_ip execution module cannot be loaded: this module is only available on SUSE based distributions.')

def _error_msg_iface(iface, option, expected):
    if False:
        i = 10
        return i + 15
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    if isinstance(expected, str):
        expected = (expected,)
    msg = 'Invalid option -- Interface: {}, Option: {}, Expected: [{}]'
    return msg.format(iface, option, '|'.join((str(e) for e in expected)))

def _error_msg_routes(iface, option, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    msg = 'Invalid option -- Route interface: {}, Option: {}, Expected: [{}]'
    return msg.format(iface, option, expected)

def _log_default_iface(iface, opt, value):
    if False:
        i = 10
        return i + 15
    log.info('Using default option -- Interface: %s Option: %s Value: %s', iface, opt, value)

def _error_msg_network(option, expected):
    if False:
        return 10
    '\n    Build an appropriate error message from a given option and\n    a list of expected values.\n    '
    if isinstance(expected, str):
        expected = (expected,)
    msg = 'Invalid network setting -- Setting: {}, Expected: [{}]'
    return msg.format(option, '|'.join((str(e) for e in expected)))

def _log_default_network(opt, value):
    if False:
        print('Hello World!')
    log.info('Using existing setting -- Setting: %s Value: %s', opt, value)

def _parse_suse_config(path):
    if False:
        for i in range(10):
            print('nop')
    suse_config = _read_file(path)
    cv_suse_config = {}
    if suse_config:
        for line in suse_config:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            pair = [p.rstrip() for p in line.split('=', 1)]
            if len(pair) != 2:
                continue
            (name, value) = pair
            cv_suse_config[name.upper()] = salt.utils.stringutils.dequote(value)
    return cv_suse_config

def _parse_ethtool_opts(opts, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses valid options for ETHTOOL_OPTIONS of the interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
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
    if 'advertise' in opts:
        valid = ['0x001', '0x002', '0x004', '0x008', '0x010', '0x020', '0x20000', '0x8000', '0x1000', '0x40000', '0x80000', '0x200000', '0x400000', '0x800000', '0x1000000', '0x2000000', '0x4000000']
        if str(opts['advertise']) in valid:
            config.update({'advertise': opts['advertise']})
        else:
            _raise_error_iface(iface, 'advertise', valid)
    if 'channels' in opts:
        channels_cmd = '-L {}'.format(iface.strip())
        channels_params = []
        for option in ('rx', 'tx', 'other', 'combined'):
            if option in opts['channels']:
                valid = range(1, __grains__['num_cpus'] + 1)
                if opts['channels'][option] in valid:
                    channels_params.append('{} {}'.format(option, opts['channels'][option]))
                else:
                    _raise_error_iface(iface, opts['channels'][option], valid)
        if channels_params:
            config.update({channels_cmd: ' '.join(channels_params)})
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

def _parse_settings_bond(opts, iface):
    if False:
        while True:
            i = 10
    '\n    Parses valid options for bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    if opts['mode'] in ('balance-rr', '0'):
        log.info('Device: %s Bonding Mode: load balancing (round-robin)', iface)
        return _parse_settings_bond_0(opts, iface)
    elif opts['mode'] in ('active-backup', '1'):
        log.info('Device: %s Bonding Mode: fault-tolerance (active-backup)', iface)
        return _parse_settings_bond_1(opts, iface)
    elif opts['mode'] in ('balance-xor', '2'):
        log.info('Device: %s Bonding Mode: load balancing (xor)', iface)
        return _parse_settings_bond_2(opts, iface)
    elif opts['mode'] in ('broadcast', '3'):
        log.info('Device: %s Bonding Mode: fault-tolerance (broadcast)', iface)
        return _parse_settings_bond_3(opts, iface)
    elif opts['mode'] in ('802.3ad', '4'):
        log.info('Device: %s Bonding Mode: IEEE 802.3ad Dynamic link aggregation', iface)
        return _parse_settings_bond_4(opts, iface)
    elif opts['mode'] in ('balance-tlb', '5'):
        log.info('Device: %s Bonding Mode: transmit load balancing', iface)
        return _parse_settings_bond_5(opts, iface)
    elif opts['mode'] in ('balance-alb', '6'):
        log.info('Device: %s Bonding Mode: adaptive load balancing', iface)
        return _parse_settings_bond_6(opts, iface)
    else:
        valid = ('0', '1', '2', '3', '4', '5', '6', 'balance-rr', 'active-backup', 'balance-xor', 'broadcast', '802.3ad', 'balance-tlb', 'balance-alb')
        _raise_error_iface(iface, 'mode', valid)

def _parse_settings_miimon(opts, iface):
    if False:
        print('Hello World!')
    '\n    Add shared settings for miimon support used by balance-rr, balance-xor\n    bonding types.\n    '
    ret = {}
    for binding in ('miimon', 'downdelay', 'updelay'):
        if binding in opts:
            try:
                int(opts[binding])
                ret.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, 'integer')
    if 'miimon' in opts and 'downdelay' not in opts:
        ret['downdelay'] = ret['miimon'] * 2
    if 'miimon' in opts:
        if not opts['miimon']:
            _raise_error_iface(iface, 'miimon', 'nonzero integer')
        for binding in ('downdelay', 'updelay'):
            if binding in ret:
                if ret[binding] % ret['miimon']:
                    _raise_error_iface(iface, binding, '0 or a multiple of miimon ({})'.format(ret['miimon']))
        if 'use_carrier' in opts:
            if opts['use_carrier'] in _CONFIG_TRUE:
                ret.update({'use_carrier': '1'})
            elif opts['use_carrier'] in _CONFIG_FALSE:
                ret.update({'use_carrier': '0'})
            else:
                valid = _CONFIG_TRUE + _CONFIG_FALSE
                _raise_error_iface(iface, 'use_carrier', valid)
        else:
            _log_default_iface(iface, 'use_carrier', _BOND_DEFAULTS['use_carrier'])
            ret.update({'use_carrier': _BOND_DEFAULTS['use_carrier']})
    return ret

def _parse_settings_arp(opts, iface):
    if False:
        while True:
            i = 10
    '\n    Add shared settings for arp used by balance-rr, balance-xor bonding types.\n    '
    ret = {}
    if 'arp_interval' in opts:
        try:
            int(opts['arp_interval'])
            ret.update({'arp_interval': opts['arp_interval']})
        except ValueError:
            _raise_error_iface(iface, 'arp_interval', 'integer')
        valid = 'list of ips (up to 16)'
        if 'arp_ip_target' in opts:
            if isinstance(opts['arp_ip_target'], list):
                if 1 <= len(opts['arp_ip_target']) <= 16:
                    ret.update({'arp_ip_target': ','.join(opts['arp_ip_target'])})
                else:
                    _raise_error_iface(iface, 'arp_ip_target', valid)
            else:
                _raise_error_iface(iface, 'arp_ip_target', valid)
        else:
            _raise_error_iface(iface, 'arp_ip_target', valid)
    return ret

def _parse_settings_bond_0(opts, iface):
    if False:
        print('Hello World!')
    '\n    Parses valid options for balance-rr (type 0) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '0'}
    bond.update(_parse_settings_miimon(opts, iface))
    bond.update(_parse_settings_arp(opts, iface))
    if 'miimon' not in opts and 'arp_interval' not in opts:
        _raise_error_iface(iface, 'miimon or arp_interval', 'at least one of these is required')
    return bond

def _parse_settings_bond_1(opts, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses valid options for active-backup (type 1) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '1'}
    bond.update(_parse_settings_miimon(opts, iface))
    if 'miimon' not in opts:
        _raise_error_iface(iface, 'miimon', 'integer')
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    return bond

def _parse_settings_bond_2(opts, iface):
    if False:
        while True:
            i = 10
    '\n    Parses valid options for balance-xor (type 2) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '2'}
    bond.update(_parse_settings_miimon(opts, iface))
    bond.update(_parse_settings_arp(opts, iface))
    if 'miimon' not in opts and 'arp_interval' not in opts:
        _raise_error_iface(iface, 'miimon or arp_interval', 'at least one of these is required')
    if 'hashing-algorithm' in opts:
        valid = ('layer2', 'layer2+3', 'layer3+4')
        if opts['hashing-algorithm'] in valid:
            bond.update({'xmit_hash_policy': opts['hashing-algorithm']})
        else:
            _raise_error_iface(iface, 'hashing-algorithm', valid)
    return bond

def _parse_settings_bond_3(opts, iface):
    if False:
        while True:
            i = 10
    '\n    Parses valid options for broadcast (type 3) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '3'}
    bond.update(_parse_settings_miimon(opts, iface))
    if 'miimon' not in opts:
        _raise_error_iface(iface, 'miimon', 'integer')
    return bond

def _parse_settings_bond_4(opts, iface):
    if False:
        print('Hello World!')
    '\n    Parses valid options for 802.3ad (type 4) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '4'}
    bond.update(_parse_settings_miimon(opts, iface))
    if 'miimon' not in opts:
        _raise_error_iface(iface, 'miimon', 'integer')
    for binding in ('lacp_rate', 'ad_select'):
        if binding in opts:
            if binding == 'lacp_rate':
                valid = ('fast', '1', 'slow', '0')
                if opts[binding] not in valid:
                    _raise_error_iface(iface, binding, valid)
                if opts[binding] == 'fast':
                    opts.update({binding: '1'})
                if opts[binding] == 'slow':
                    opts.update({binding: '0'})
            else:
                valid = 'integer'
            try:
                int(opts[binding])
                bond.update({binding: opts[binding]})
            except ValueError:
                _raise_error_iface(iface, binding, valid)
        else:
            _log_default_iface(iface, binding, _BOND_DEFAULTS[binding])
            bond.update({binding: _BOND_DEFAULTS[binding]})
    if 'hashing-algorithm' in opts:
        valid = ('layer2', 'layer2+3', 'layer3+4')
        if opts['hashing-algorithm'] in valid:
            bond.update({'xmit_hash_policy': opts['hashing-algorithm']})
        else:
            _raise_error_iface(iface, 'hashing-algorithm', valid)
    return bond

def _parse_settings_bond_5(opts, iface):
    if False:
        print('Hello World!')
    '\n    Parses valid options for balance-tlb (type 5) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '5'}
    bond.update(_parse_settings_miimon(opts, iface))
    if 'miimon' not in opts:
        _raise_error_iface(iface, 'miimon', 'integer')
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    return bond

def _parse_settings_bond_6(opts, iface):
    if False:
        return 10
    '\n    Parses valid options for balance-alb (type 6) bonding interface\n    Logs the error and raises AttributeError in case of getting invalid options\n    '
    bond = {'mode': '6'}
    bond.update(_parse_settings_miimon(opts, iface))
    if 'miimon' not in opts:
        _raise_error_iface(iface, 'miimon', 'integer')
    if 'primary' in opts:
        bond.update({'primary': opts['primary']})
    return bond

def _parse_settings_vlan(opts, iface):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters given options and outputs valid settings for a vlan\n    '
    vlan = {}
    if 'reorder_hdr' in opts:
        if opts['reorder_hdr'] in _CONFIG_TRUE + _CONFIG_FALSE:
            vlan.update({'reorder_hdr': opts['reorder_hdr']})
        else:
            valid = _CONFIG_TRUE + _CONFIG_FALSE
            _raise_error_iface(iface, 'reorder_hdr', valid)
    if 'vlan_id' in opts:
        if opts['vlan_id'] > 0:
            vlan.update({'vlan_id': opts['vlan_id']})
        else:
            _raise_error_iface(iface, 'vlan_id', 'Positive integer')
    if 'phys_dev' in opts:
        if len(opts['phys_dev']) > 0:
            vlan.update({'phys_dev': opts['phys_dev']})
        else:
            _raise_error_iface(iface, 'phys_dev', 'Non-empty string')
    return vlan

def _parse_settings_eth(opts, iface_type, enabled, iface):
    if False:
        i = 10
        return i + 15
    '\n    Filters given options and outputs valid settings for a\n    network interface.\n    '
    result = {'name': iface}
    if 'proto' in opts:
        valid = ['static', 'dhcp', 'dhcp4', 'dhcp6', 'autoip', 'dhcp+autoip', 'auto6', '6to4', 'none']
        if opts['proto'] in valid:
            result['proto'] = opts['proto']
        else:
            _raise_error_iface(iface, opts['proto'], valid)
    if 'mtu' in opts:
        try:
            result['mtu'] = int(opts['mtu'])
        except ValueError:
            _raise_error_iface(iface, 'mtu', ['integer'])
    if 'hwaddr' in opts and 'macaddr' in opts:
        msg = 'Cannot pass both hwaddr and macaddr. Must use either hwaddr or macaddr'
        log.error(msg)
        raise AttributeError(msg)
    if iface_type not in ('bridge',):
        ethtool = _parse_ethtool_opts(opts, iface)
        if ethtool:
            result['ethtool'] = ' '.join(['{} {}'.format(x, y) for (x, y) in ethtool.items()])
    if iface_type == 'slave':
        result['proto'] = 'none'
    if iface_type == 'bond':
        if 'mode' not in opts:
            msg = "Missing required option 'mode'"
            log.error("%s for bond interface '%s'", msg, iface)
            raise AttributeError(msg)
        bonding = _parse_settings_bond(opts, iface)
        if bonding:
            result['bonding'] = ' '.join(['{}={}'.format(x, y) for (x, y) in bonding.items()])
            result['devtype'] = 'Bond'
            if 'slaves' in opts:
                if isinstance(opts['slaves'], list):
                    result['slaves'] = opts['slaves']
                else:
                    result['slaves'] = opts['slaves'].split()
    if iface_type == 'vlan':
        vlan = _parse_settings_vlan(opts, iface)
        if vlan:
            result['devtype'] = 'Vlan'
            for opt in vlan:
                result[opt] = opts[opt]
    if iface_type == 'eth':
        result['devtype'] = 'Ethernet'
    if iface_type == 'bridge':
        result['devtype'] = 'Bridge'
        bypassfirewall = True
        valid = _CONFIG_TRUE + _CONFIG_FALSE
        for opt in ('bypassfirewall',):
            if opt in opts:
                if opts[opt] in _CONFIG_TRUE:
                    bypassfirewall = True
                elif opts[opt] in _CONFIG_FALSE:
                    bypassfirewall = False
                else:
                    _raise_error_iface(iface, opts[opt], valid)
        bridgectls = ['net.bridge.bridge-nf-call-ip6tables', 'net.bridge.bridge-nf-call-iptables', 'net.bridge.bridge-nf-call-arptables']
        if bypassfirewall:
            sysctl_value = 0
        else:
            sysctl_value = 1
        for sysctl in bridgectls:
            try:
                __salt__['sysctl.persist'](sysctl, sysctl_value)
            except CommandExecutionError:
                log.warning('Failed to set sysctl: %s', sysctl)
    elif 'bridge' in opts:
        result['bridge'] = opts['bridge']
    if iface_type == 'ipip':
        result['devtype'] = 'IPIP'
        for opt in ('my_inner_ipaddr', 'my_outer_ipaddr'):
            if opt not in opts:
                _raise_error_iface(iface, opt, '1.2.3.4')
            else:
                result[opt] = opts[opt]
    if iface_type == 'ib':
        result['devtype'] = 'InfiniBand'
    if 'prefix' in opts:
        if 'netmask' in opts:
            msg = 'Cannot use prefix and netmask together'
            log.error(msg)
            raise AttributeError(msg)
        result['prefix'] = opts['prefix']
    elif 'netmask' in opts:
        result['netmask'] = opts['netmask']
    for opt in ('ipaddr', 'master', 'srcaddr', 'delay', 'domain', 'gateway', 'uuid', 'nickname', 'zone'):
        if opt in opts:
            result[opt] = opts[opt]
    if 'ipaddrs' in opts or 'ipv6addr' in opts or 'ipv6addrs' in opts:
        result['ipaddrs'] = []
    if 'ipaddrs' in opts:
        for opt in opts['ipaddrs']:
            if salt.utils.validate.net.ipv4_addr(opt) or salt.utils.validate.net.ipv6_addr(opt):
                result['ipaddrs'].append(opt)
            else:
                msg = '{} is invalid ipv4 or ipv6 CIDR'.format(opt)
                log.error(msg)
                raise AttributeError(msg)
    if 'ipv6addr' in opts:
        if salt.utils.validate.net.ipv6_addr(opts['ipv6addr']):
            result['ipaddrs'].append(opts['ipv6addr'])
        else:
            msg = '{} is invalid ipv6 CIDR'.format(opt)
            log.error(msg)
            raise AttributeError(msg)
    if 'ipv6addrs' in opts:
        for opt in opts['ipv6addrs']:
            if salt.utils.validate.net.ipv6_addr(opt):
                result['ipaddrs'].append(opt)
            else:
                msg = '{} is invalid ipv6 CIDR'.format(opt)
                log.error(msg)
                raise AttributeError(msg)
    if 'enable_ipv6' in opts:
        result['enable_ipv6'] = opts['enable_ipv6']
    valid = _CONFIG_TRUE + _CONFIG_FALSE
    for opt in ('onparent', 'peerdns', 'peerroutes', 'slave', 'vlan', 'defroute', 'stp', 'ipv6_peerdns', 'ipv6_defroute', 'ipv6_peerroutes', 'ipv6_autoconf', 'ipv4_failure_fatal', 'dhcpv6c'):
        if opt in opts:
            if opts[opt] in _CONFIG_TRUE:
                result[opt] = 'yes'
            elif opts[opt] in _CONFIG_FALSE:
                result[opt] = 'no'
            else:
                _raise_error_iface(iface, opts[opt], valid)
    if 'onboot' in opts:
        log.warning("The 'onboot' option is controlled by the 'enabled' option. Interface: %s Enabled: %s", iface, enabled)
    if 'startmode' in opts:
        valid = ('manual', 'auto', 'nfsroot', 'hotplug', 'off')
        if opts['startmode'] in valid:
            result['startmode'] = opts['startmode']
        else:
            _raise_error_iface(iface, opts['startmode'], valid)
    elif enabled:
        result['startmode'] = 'auto'
    else:
        result['startmode'] = 'off'
    if 'vlan' in opts:
        if opts['vlan'] in _CONFIG_TRUE:
            result['vlan'] = 'yes'
        elif opts['vlan'] in _CONFIG_FALSE:
            result['vlan'] = 'no'
        else:
            _raise_error_iface(iface, opts['vlan'], valid)
    if 'arpcheck' in opts:
        if opts['arpcheck'] in _CONFIG_FALSE:
            result['arpcheck'] = 'no'
    if 'ipaddr_start' in opts:
        result['ipaddr_start'] = opts['ipaddr_start']
    if 'ipaddr_end' in opts:
        result['ipaddr_end'] = opts['ipaddr_end']
    if 'clonenum_start' in opts:
        result['clonenum_start'] = opts['clonenum_start']
    if 'hwaddr' in opts:
        result['hwaddr'] = opts['hwaddr']
    if 'macaddr' in opts:
        result['macaddr'] = opts['macaddr']
    if 'nm_controlled' in opts:
        if opts['nm_controlled'] in _CONFIG_TRUE:
            result['nm_controlled'] = 'yes'
        elif opts['nm_controlled'] in _CONFIG_FALSE:
            result['nm_controlled'] = 'no'
        else:
            _raise_error_iface(iface, opts['nm_controlled'], valid)
    else:
        result['nm_controlled'] = 'no'
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

def _parse_network_settings(opts, current):
    if False:
        while True:
            i = 10
    '\n    Filters given options and outputs valid settings for\n    the global network settings file.\n    '
    opts = {k.lower(): v for (k, v) in opts.items()}
    current = {k.lower(): v for (k, v) in current.items()}
    retain_settings = opts.get('retain_settings', False)
    result = {}
    if retain_settings:
        for opt in current:
            nopt = opt
            if opt == 'netconfig_dns_static_servers':
                nopt = 'dns'
                result[nopt] = current[opt].split()
            elif opt == 'netconfig_dns_static_searchlist':
                nopt = 'dns_search'
                result[nopt] = current[opt].split()
            elif opt.startswith('netconfig_') and opt not in ('netconfig_modules_order', 'netconfig_verbose', 'netconfig_force_replace'):
                nopt = opt[10:]
                result[nopt] = current[opt]
            else:
                result[nopt] = current[opt]
            _log_default_network(nopt, current[opt])
    for opt in opts:
        if opt in ('dns', 'dns_search') and (not isinstance(opts[opt], list)):
            result[opt] = opts[opt].split()
        else:
            result[opt] = opts[opt]
    return result

def _raise_error_iface(iface, option, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Log and raise an error with a logical formatted message.\n    '
    msg = _error_msg_iface(iface, option, expected)
    log.error(msg)
    raise AttributeError(msg)

def _raise_error_network(option, expected):
    if False:
        for i in range(10):
            print('nop')
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
        i = 10
        return i + 15
    '\n    Reads and returns the contents of a file\n    '
    try:
        with salt.utils.files.fopen(path, 'rb') as rfh:
            return _get_non_blank_lines(salt.utils.stringutils.to_unicode(rfh.read()))
    except OSError:
        return []

def _write_file_iface(iface, data, folder, pattern):
    if False:
        for i in range(10):
            print('nop')
    '\n    Writes a file to disk\n    '
    filename = os.path.join(folder, pattern.format(iface))
    if not os.path.exists(folder):
        msg = '{} cannot be written. {} does not exist'.format(filename, folder)
        log.error(msg)
        raise AttributeError(msg)
    with salt.utils.files.fopen(filename, 'w') as fp_:
        fp_.write(salt.utils.stringutils.to_str(data))

def _write_file_network(data, filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Writes a file to disk\n    '
    with salt.utils.files.fopen(filename, 'w') as fp_:
        fp_.write(salt.utils.stringutils.to_str(data))

def _get_non_blank_lines(data):
    if False:
        while True:
            i = 10
    lines = data.splitlines()
    try:
        lines.remove('')
    except ValueError:
        pass
    return lines

def build_interface(iface, iface_type, enabled, **settings):
    if False:
        while True:
            i = 10
    "\n    Build an interface script for a network interface.\n\n    Args:\n        :param iface:\n            The name of the interface to build the configuration for\n\n        :param iface_type:\n            The type of the interface. The following types are possible:\n              - eth\n              - bond\n              - alias\n              - clone\n              - ipsec\n              - dialup\n              - bridge\n              - slave\n              - vlan\n              - ipip\n              - ib\n\n        :param enabled:\n            Build the interface enabled or disabled\n\n        :param settings:\n            The settings for the interface\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_interface eth0 eth <settings>\n    "
    iface_type = iface_type.lower()
    if iface_type not in _IFACE_TYPES:
        _raise_error_iface(iface, iface_type, _IFACE_TYPES)
    if iface_type == 'slave':
        settings['slave'] = 'yes'
        if 'master' not in settings:
            msg = 'master is a required setting for slave interfaces'
            log.error(msg)
            raise AttributeError(msg)
    if iface_type == 'bond':
        if 'mode' not in settings:
            msg = 'mode is required for bond interfaces'
            log.error(msg)
            raise AttributeError(msg)
        settings['mode'] = str(settings['mode'])
    if iface_type == 'vlan':
        settings['vlan'] = 'yes'
    if iface_type == 'bridge' and (not __salt__['pkg.version']('bridge-utils')):
        __salt__['pkg.install']('bridge-utils')
    if iface_type in ('eth', 'bond', 'bridge', 'slave', 'vlan', 'ipip', 'ib', 'alias'):
        opts = _parse_settings_eth(settings, iface_type, enabled, iface)
        try:
            template = JINJA.get_template('ifcfg.jinja')
        except jinja2.exceptions.TemplateNotFound:
            log.error('Could not load template ifcfg.jinja')
            return ''
        log.debug('Interface opts:\n%s', opts)
        ifcfg = template.render(opts)
    if settings.get('test'):
        return _get_non_blank_lines(ifcfg)
    _write_file_iface(iface, ifcfg, _SUSE_NETWORK_SCRIPT_DIR, 'ifcfg-{}')
    path = os.path.join(_SUSE_NETWORK_SCRIPT_DIR, 'ifcfg-{}'.format(iface))
    return _read_file(path)

def build_routes(iface, **settings):
    if False:
        for i in range(10):
            print('nop')
    "\n    Build a route script for a network interface.\n\n    Args:\n        :param iface:\n            The name of the interface to build the routes for\n\n        :param settings:\n            The settings for the routes\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_routes eth0 <settings>\n    "
    template = 'ifroute.jinja'
    log.debug('Template name: %s', template)
    opts = _parse_routes(iface, settings)
    log.debug('Opts:\n%s', opts)
    try:
        template = JINJA.get_template(template)
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template %s', template)
        return ''
    log.debug('IP routes:\n%s', opts['routes'])
    if iface == 'routes':
        routecfg = template.render(routes=opts['routes'])
    else:
        routecfg = template.render(routes=opts['routes'], iface=iface)
    if settings['test']:
        return _get_non_blank_lines(routecfg)
    if iface == 'routes':
        path = _SUSE_NETWORK_ROUTES_FILE
    else:
        path = os.path.join(_SUSE_NETWORK_SCRIPT_DIR, 'ifroute-{}'.format(iface))
    _write_file_network(routecfg, path)
    return _read_file(path)

def down(iface, iface_type=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Shutdown a network interface\n\n    Args:\n        :param iface:\n            The name of the interface to shutdown\n\n        :param iface_type:\n            The type of the interface\n            If ``slave`` is specified, no any action is performing\n            Default is ``None``\n\n    Returns:\n        str: The result of ``ifdown`` command or ``None`` if ``slave``\n        iface_type was specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.down eth0\n    "
    if not iface_type or iface_type.lower() != 'slave':
        return __salt__['cmd.run']('ifdown {}'.format(iface))
    return None

def get_interface(iface):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the contents of an interface script\n\n    Args:\n        :param iface:\n            The name of the interface to get settings for\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_interface eth0\n    "
    path = os.path.join(_SUSE_NETWORK_SCRIPT_DIR, 'ifcfg-{}'.format(iface))
    return _read_file(path)

def up(iface, iface_type=None):
    if False:
        return 10
    "\n    Start up a network interface\n\n    Args:\n        :param iface:\n            The name of the interface to start up\n\n        :param iface_type:\n            The type of the interface\n            If ``slave`` is specified, no any action is performing\n            Default is ``None``\n\n    Returns:\n        str: The result of ``ifup`` command or ``None`` if ``slave``\n        iface_type was specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.up eth0\n    "
    if not iface_type or iface_type.lower() != 'slave':
        return __salt__['cmd.run']('ifup {}'.format(iface))
    return None

def get_routes(iface):
    if False:
        i = 10
        return i + 15
    "\n    Return the contents of the interface routes script.\n\n    Args:\n        :param iface:\n            The name of the interface to get the routes for\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_routes eth0\n    "
    if iface == 'routes':
        path = _SUSE_NETWORK_ROUTES_FILE
    else:
        path = os.path.join(_SUSE_NETWORK_SCRIPT_DIR, 'ifroute-{}'.format(iface))
    return _read_file(path)

def get_network_settings():
    if False:
        return 10
    "\n    Return the contents of the global network script.\n\n    Args:\n        :param iface:\n            The name of the interface to start up\n\n        :param iface_type:\n            The type of the interface\n            If ``slave`` is specified, no any action is performing\n            Default is ``None``\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.get_network_settings\n    "
    return _read_file(_SUSE_NETWORK_FILE)

def apply_network_settings(**settings):
    if False:
        while True:
            i = 10
    "\n    Apply global network configuration.\n\n    Args:\n        :param settings:\n            The network settings to apply\n\n    Returns:\n        The result of ``service.reload`` for ``network`` service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.apply_network_settings\n    "
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
        res = __salt__['service.reload']('network')
    return hostname_res and res

def build_network_settings(**settings):
    if False:
        i = 10
        return i + 15
    "\n    Build the global network script.\n\n    Args:\n        :param settings:\n            The network settings\n\n    Returns:\n        dict: A dictionary of file/content\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ip.build_network_settings <settings>\n    "
    current_network_settings = _parse_suse_config(_SUSE_NETWORK_FILE)
    opts = _parse_network_settings(settings, current_network_settings)
    try:
        template = JINJA.get_template('network.jinja')
    except jinja2.exceptions.TemplateNotFound:
        log.error('Could not load template network.jinja')
        return ''
    network = template.render(opts)
    if settings['test']:
        return _get_non_blank_lines(network)
    _write_file_network(network, _SUSE_NETWORK_FILE)
    __salt__['cmd.run']('netconfig update -f')
    return _read_file(_SUSE_NETWORK_FILE)