"""
A simple beacon to watch journald for specific entries
"""
import logging
import salt.utils.beacons
import salt.utils.data
try:
    import systemd.journal
    HAS_SYSTEMD = True
except ImportError:
    HAS_SYSTEMD = False
log = logging.getLogger(__name__)
__virtualname__ = 'journald'

def __virtual__():
    if False:
        print('Hello World!')
    if HAS_SYSTEMD:
        return __virtualname__
    err_msg = 'systemd library is missing.'
    log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
    return (False, err_msg)

def _get_journal():
    if False:
        print('Hello World!')
    '\n    Return the active running journal object\n    '
    if 'systemd.journald' in __context__:
        return __context__['systemd.journald']
    __context__['systemd.journald'] = systemd.journal.Reader()
    __context__['systemd.journald'].seek_tail()
    __context__['systemd.journald'].get_previous()
    return __context__['systemd.journald']

def validate(config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for journald beacon must be a list.')
    else:
        config = salt.utils.beacons.list_to_dict(config)
        for name in config.get('services', {}):
            if not isinstance(config['services'][name], dict):
                return (False, 'Services configuration for journald beacon must be a list of dictionaries.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        while True:
            i = 10
    '\n    The journald beacon allows for the systemd journal to be parsed and linked\n    objects to be turned into events.\n\n    This beacons config will return all sshd jornal entries\n\n    .. code-block:: yaml\n\n        beacons:\n          journald:\n            - services:\n                sshd:\n                  SYSLOG_IDENTIFIER: sshd\n                  PRIORITY: 6\n    '
    ret = []
    journal = _get_journal()
    config = salt.utils.beacons.list_to_dict(config)
    while True:
        cur = journal.get_next()
        if not cur:
            break
        for name in config.get('services', {}):
            n_flag = 0
            for key in config['services'][name]:
                if isinstance(key, str):
                    key = salt.utils.data.decode(key)
                if key in cur:
                    if config['services'][name][key] == cur[key]:
                        n_flag += 1
            if n_flag == len(config['services'][name]):
                sub = salt.utils.data.simple_types_filter(cur)
                sub.update({'tag': name})
                ret.append(sub)
    return ret