"""
Beacon to announce via avahi (zeroconf)

.. versionadded:: 2016.11.0

Dependencies
============

- python-avahi
- dbus-python

"""
import logging
import time
import salt.utils.beacons
import salt.utils.stringutils
try:
    import avahi
    HAS_PYAVAHI = True
except ImportError:
    HAS_PYAVAHI = False
try:
    import dbus
    from dbus import DBusException
    BUS = dbus.SystemBus()
    SERVER = dbus.Interface(BUS.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
    GROUP = dbus.Interface(BUS.get_object(avahi.DBUS_NAME, SERVER.EntryGroupNew()), avahi.DBUS_INTERFACE_ENTRY_GROUP)
    HAS_DBUS = True
except (ImportError, NameError):
    HAS_DBUS = False
except DBusException:
    HAS_DBUS = False
log = logging.getLogger(__name__)
__virtualname__ = 'avahi_announce'
LAST_GRAINS = {}

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    if HAS_PYAVAHI:
        if HAS_DBUS:
            return __virtualname__
        err_msg = "The 'python-dbus' dependency is missing."
        log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
        return (False, err_msg)
    err_msg = "The 'python-avahi' dependency is missing."
    log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
    return (False, err_msg)

def validate(config):
    if False:
        i = 10
        return i + 15
    '\n    Validate the beacon configuration\n    '
    _config = salt.utils.beacons.list_to_dict(config)
    if not isinstance(config, list):
        return (False, 'Configuration for avahi_announce beacon must be a list.')
    elif not all((x in _config for x in ('servicetype', 'port', 'txt'))):
        return (False, 'Configuration for avahi_announce beacon must contain servicetype, port and txt items.')
    return (True, 'Valid beacon configuration.')

def _enforce_txt_record_maxlen(key, value):
    if False:
        return 10
    "\n    Enforces the TXT record maximum length of 255 characters.\n    TXT record length includes key, value, and '='.\n\n    :param str key: Key of the TXT record\n    :param str value: Value of the TXT record\n\n    :rtype: str\n    :return: The value of the TXT record. It may be truncated if it exceeds\n             the maximum permitted length. In case of truncation, '...' is\n             appended to indicate that the entire value is not present.\n    "
    if len(key) + len(value) + 1 > 255:
        return value[:251 - len(key)] + '...'
    return value

def beacon(config):
    if False:
        print('Hello World!')
    "\n    Broadcast values via zeroconf\n\n    If the announced values are static, it is advised to set run_once: True\n    (do not poll) on the beacon configuration.\n\n    The following are required configuration settings:\n\n    - ``servicetype`` - The service type to announce\n    - ``port`` - The port of the service to announce\n    - ``txt`` - The TXT record of the service being announced as a dict. Grains\n      can be used to define TXT values using one of following two formats:\n\n      - ``grains.<grain_name>``\n      - ``grains.<grain_name>[i]`` where i is an integer representing the\n        index of the grain to use. If the grain is not a list, the index is\n        ignored.\n\n    The following are optional configuration settings:\n\n    - ``servicename`` - Set the name of the service. Will use the hostname from\n      the minion's ``host`` grain if this value is not set.\n    - ``reset_on_change`` - If ``True`` and there is a change in TXT records\n      detected, it will stop announcing the service and then restart announcing\n      the service. This interruption in service announcement may be desirable\n      if the client relies on changes in the browse records to update its cache\n      of TXT records. Defaults to ``False``.\n    - ``reset_wait`` - The number of seconds to wait after announcement stops\n      announcing and before it restarts announcing in the case where there is a\n      change in TXT records detected and ``reset_on_change`` is ``True``.\n      Defaults to ``0``.\n    - ``copy_grains`` - If ``True``, Salt will copy the grains passed into the\n      beacon when it backs them up to check for changes on the next iteration.\n      Normally, instead of copy, it would use straight value assignment. This\n      will allow detection of changes to grains where the grains are modified\n      in-place instead of completely replaced.  In-place grains changes are not\n      currently done in the main Salt code but may be done due to a custom\n      plug-in. Defaults to ``False``.\n\n    Example Config\n\n    .. code-block:: yaml\n\n       beacons:\n         avahi_announce:\n           - run_once: True\n           - servicetype: _demo._tcp\n           - port: 1234\n           - txt:\n               ProdName: grains.productname\n               SerialNo: grains.serialnumber\n               Comments: 'this is a test'\n    "
    ret = []
    changes = {}
    txt = {}
    global LAST_GRAINS
    config = salt.utils.beacons.list_to_dict(config)
    if 'servicename' in config:
        servicename = config['servicename']
    else:
        servicename = __grains__['host']
        if LAST_GRAINS and LAST_GRAINS['host'] != servicename:
            changes['servicename'] = servicename
    if LAST_GRAINS and config.get('reset_on_change', False):
        if LAST_GRAINS.get('ipv4', []) != __grains__.get('ipv4', []):
            changes['ipv4'] = __grains__.get('ipv4', [])
        if LAST_GRAINS.get('ipv6', []) != __grains__.get('ipv6', []):
            changes['ipv6'] = __grains__.get('ipv6', [])
    for item in config['txt']:
        changes_key = 'txt.' + salt.utils.stringutils.to_unicode(item)
        if config['txt'][item].startswith('grains.'):
            grain = config['txt'][item][7:]
            grain_index = None
            square_bracket = grain.find('[')
            if square_bracket != -1 and grain[-1] == ']':
                grain_index = int(grain[square_bracket + 1:-1])
                grain = grain[:square_bracket]
            grain_value = __grains__.get(grain, '')
            if isinstance(grain_value, list):
                if grain_index is not None:
                    grain_value = grain_value[grain_index]
                else:
                    grain_value = ','.join(grain_value)
            txt[item] = _enforce_txt_record_maxlen(item, grain_value)
            if LAST_GRAINS and LAST_GRAINS.get(grain, '') != __grains__.get(grain, ''):
                changes[changes_key] = txt[item]
        else:
            txt[item] = _enforce_txt_record_maxlen(item, config['txt'][item])
        if not LAST_GRAINS:
            changes[changes_key] = txt[item]
    if changes:
        if not LAST_GRAINS:
            changes['servicename'] = servicename
            changes['servicetype'] = config['servicetype']
            changes['port'] = config['port']
            changes['ipv4'] = __grains__.get('ipv4', [])
            changes['ipv6'] = __grains__.get('ipv6', [])
            GROUP.AddService(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, dbus.UInt32(0), servicename, config['servicetype'], '', '', dbus.UInt16(config['port']), avahi.dict_to_txt_array(txt))
            GROUP.Commit()
        elif config.get('reset_on_change', False) or 'servicename' in changes:
            GROUP.Reset()
            reset_wait = config.get('reset_wait', 0)
            if reset_wait > 0:
                time.sleep(reset_wait)
            GROUP.AddService(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, dbus.UInt32(0), servicename, config['servicetype'], '', '', dbus.UInt16(config['port']), avahi.dict_to_txt_array(txt))
            GROUP.Commit()
        else:
            GROUP.UpdateServiceTxt(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, dbus.UInt32(0), servicename, config['servicetype'], '', avahi.dict_to_txt_array(txt))
        ret.append({'tag': 'result', 'changes': changes})
    if config.get('copy_grains', False):
        LAST_GRAINS = __grains__.copy()
    else:
        LAST_GRAINS = __grains__
    return ret