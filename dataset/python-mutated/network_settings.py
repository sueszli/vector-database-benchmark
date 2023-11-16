"""
Beacon to monitor network adapter setting changes on Linux

.. versionadded:: 2016.3.0

"""
import ast
import logging
import re
import salt.loader
import salt.utils.beacons
try:
    from pyroute2 import NDB
    from pyroute2.ndb.compat import ipdb_interfaces_view
    IP = NDB()
    HAS_PYROUTE2 = True
    HAS_NDB = True
except ImportError:
    IP = None
    HAS_NDB = False
    HAS_PYROUTE2 = False
try:
    from pyroute2 import IPDB
    if IP is None:
        IP = IPDB()
        HAS_PYROUTE2 = True
except ImportError:
    IP = None
log = logging.getLogger(__name__)
__virtualname__ = 'network_settings'
ATTRS = ['family', 'txqlen', 'ipdb_scope', 'index', 'operstate', 'group', 'carrier_changes', 'ipaddr', 'neighbours', 'ifname', 'promiscuity', 'linkmode', 'broadcast', 'address', 'num_tx_queues', 'ipdb_priority', 'kind', 'qdisc', 'mtu', 'num_rx_queues', 'carrier', 'flags', 'ifi_type', 'ports']
LAST_STATS = {}

class Hashabledict(dict):
    """
    Helper class that implements a hash function for a dictionary
    """

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(tuple(sorted(self.items())))

def __virtual__():
    if False:
        i = 10
        return i + 15
    if HAS_PYROUTE2:
        return __virtualname__
    err_msg = 'pyroute2 library is missing'
    log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
    return (False, err_msg)

def validate(config):
    if False:
        while True:
            i = 10
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for network_settings beacon must be a list.')
    else:
        config = salt.utils.beacons.list_to_dict(config)
        interfaces = config.get('interfaces', {})
        if isinstance(interfaces, list):
            return (False, 'interfaces section for network_settings beacon must be a dictionary.')
        for item in interfaces:
            if not isinstance(config['interfaces'][item], dict):
                return (False, 'Interface attributes for network_settings beacon must be a dictionary.')
            if not all((j in ATTRS for j in config['interfaces'][item])):
                return (False, 'Invalid attributes in beacon configuration.')
    return (True, 'Valid beacon configuration')

def _copy_interfaces_info(interfaces):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a dictionary with a copy of each interface attributes in ATTRS\n    '
    ret = {}
    for interface in interfaces:
        _interface_attrs_cpy = set()
        for attr in ATTRS:
            if attr in interfaces[interface]:
                attr_dict = Hashabledict()
                attr_dict[attr] = repr(interfaces[interface][attr])
                _interface_attrs_cpy.add(attr_dict)
        ret[interface] = _interface_attrs_cpy
    return ret

def beacon(config):
    if False:
        i = 10
        return i + 15
    '\n    Watch for changes on network settings\n\n    By default, the beacon will emit when there is a value change on one of the\n    settings on watch. The config also support the onvalue parameter for each\n    setting, which instruct the beacon to only emit if the setting changed to\n    the value defined.\n\n    Example Config\n\n    .. code-block:: yaml\n\n        beacons:\n          network_settings:\n            - interfaces:\n                eth0:\n                  ipaddr:\n                  promiscuity:\n                    onvalue: 1\n                eth1:\n                  linkmode:\n\n    The config above will check for value changes on eth0 ipaddr and eth1 linkmode. It will also\n    emit if the promiscuity value changes to 1.\n\n    Beacon items can use the * wildcard to make a definition apply to several interfaces. For\n    example an eth* would apply to all ethernet interfaces.\n\n    Setting the argument coalesce = True will combine all the beacon results on a single event.\n    The example below shows how to trigger coalesced results:\n\n    .. code-block:: yaml\n\n        beacons:\n          network_settings:\n            - coalesce: True\n            - interfaces:\n                eth0:\n                  ipaddr:\n                  promiscuity:\n\n    '
    _config = salt.utils.beacons.list_to_dict(config)
    ret = []
    interfaces = []
    expanded_config = {'interfaces': {}}
    global LAST_STATS
    coalesce = False
    _stats = _copy_interfaces_info(ipdb_interfaces_view(IP) if HAS_NDB else IP.by_name)
    if not LAST_STATS:
        LAST_STATS = _stats
    if 'coalesce' in _config and _config['coalesce']:
        coalesce = True
        changes = {}
    log.debug('_stats %s', _stats)
    for interface_config in _config.get('interfaces', {}):
        if interface_config in _stats:
            interfaces.append(interface_config)
        else:
            for interface_stat in _stats:
                match = re.search(interface_config, interface_stat)
                if match:
                    interfaces.append(interface_stat)
                    expanded_config['interfaces'][interface_stat] = _config['interfaces'][interface_config]
    if expanded_config:
        _config['interfaces'].update(expanded_config['interfaces'])
        _config = salt.utils.beacons.list_to_dict(config)
    log.debug('interfaces %s', interfaces)
    for interface in interfaces:
        _send_event = False
        _diff_stats = _stats[interface] - LAST_STATS[interface]
        _ret_diff = {}
        interface_config = _config['interfaces'][interface]
        log.debug('_diff_stats %s', _diff_stats)
        if _diff_stats:
            _diff_stats_dict = {}
            LAST_STATS[interface] = _stats[interface]
            for item in _diff_stats:
                _diff_stats_dict.update(item)
            for attr in interface_config:
                if attr in _diff_stats_dict:
                    config_value = None
                    if interface_config[attr] and 'onvalue' in interface_config[attr]:
                        config_value = interface_config[attr]['onvalue']
                    new_value = ast.literal_eval(_diff_stats_dict[attr])
                    if not config_value or config_value == new_value:
                        _send_event = True
                        _ret_diff[attr] = new_value
            if _send_event:
                if coalesce:
                    changes[interface] = _ret_diff
                else:
                    ret.append({'tag': interface, 'interface': interface, 'change': _ret_diff})
    if coalesce and changes:
        grains_info = salt.loader.grains(__opts__, True)
        __grains__.update(grains_info)
        ret.append({'tag': 'result', 'changes': changes})
    return ret