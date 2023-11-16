"""
Beacon to monitor statistics from ethernet adapters

.. versionadded:: 2015.5.0
"""
import logging
import salt.utils.beacons
try:
    import salt.utils.psutil_compat as psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
log = logging.getLogger(__name__)
__virtualname__ = 'network_info'
__attrs = ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv', 'errin', 'errout', 'dropin', 'dropout']

def _to_list(obj):
    if False:
        return 10
    '\n    Convert snetinfo object to list\n    '
    ret = {}
    for attr in __attrs:
        if hasattr(obj, attr):
            ret[attr] = getattr(obj, attr)
    return ret

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    if not HAS_PSUTIL:
        err_msg = 'psutil not available'
        log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
        return (False, err_msg)
    return __virtualname__

def validate(config):
    if False:
        return 10
    '\n    Validate the beacon configuration\n    '
    VALID_ITEMS = ['type', 'bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv', 'errin', 'errout', 'dropin', 'dropout']
    if not isinstance(config, list):
        return (False, 'Configuration for network_info beacon must be a list.')
    else:
        config = salt.utils.beacons.list_to_dict(config)
        for item in config.get('interfaces', {}):
            if not isinstance(config['interfaces'][item], dict):
                return (False, 'Configuration for network_info beacon must be a list of dictionaries.')
            elif not any((j in VALID_ITEMS for j in config['interfaces'][item])):
                return (False, 'Invalid configuration item in Beacon configuration.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        while True:
            i = 10
    '\n    Emit the network statistics of this host.\n\n    Specify thresholds for each network stat\n    and only emit a beacon if any of them are\n    exceeded.\n\n    Emit beacon when any values are equal to\n    configured values.\n\n    .. code-block:: yaml\n\n        beacons:\n          network_info:\n            - interfaces:\n                eth0:\n                  type: equal\n                  bytes_sent: 100000\n                  bytes_recv: 100000\n                  packets_sent: 100000\n                  packets_recv: 100000\n                  errin: 100\n                  errout: 100\n                  dropin: 100\n                  dropout: 100\n\n    Emit beacon when any values are greater\n    than configured values.\n\n    .. code-block:: yaml\n\n        beacons:\n          network_info:\n            - interfaces:\n                eth0:\n                  type: greater\n                  bytes_sent: 100000\n                  bytes_recv: 100000\n                  packets_sent: 100000\n                  packets_recv: 100000\n                  errin: 100\n                  errout: 100\n                  dropin: 100\n                  dropout: 100\n\n\n    '
    ret = []
    config = salt.utils.beacons.list_to_dict(config)
    log.debug('psutil.net_io_counters %s', psutil.net_io_counters)
    _stats = psutil.net_io_counters(pernic=True)
    log.debug('_stats %s', _stats)
    for interface in config.get('interfaces', {}):
        if interface in _stats:
            interface_config = config['interfaces'][interface]
            _if_stats = _stats[interface]
            _diff = False
            for attr in __attrs:
                if attr in interface_config:
                    if 'type' in interface_config and interface_config['type'] == 'equal':
                        if getattr(_if_stats, attr, None) == int(interface_config[attr]):
                            _diff = True
                    elif 'type' in interface_config and interface_config['type'] == 'greater':
                        if getattr(_if_stats, attr, None) > int(interface_config[attr]):
                            _diff = True
                        else:
                            log.debug('attr %s', getattr(_if_stats, attr, None))
                    elif getattr(_if_stats, attr, None) == int(interface_config[attr]):
                        _diff = True
            if _diff:
                ret.append({'interface': interface, 'network_info': _to_list(_if_stats)})
    return ret