"""
Network SNMP
============

Manage the SNMP configuration on network devices.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------

- :mod:`napalm snmp management module (salt.modules.napalm_snmp) <salt.modules.napalm_snmp>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.json
import salt.utils.napalm
log = logging.getLogger(__name__)
__virtualname__ = 'netsnmp'
_COMMUNITY_MODE_MAP = {'read-only': 'ro', 'readonly': 'ro', 'read-write': 'rw', 'write': 'rw'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _ordered_dict_to_dict(config):
    if False:
        i = 10
        return i + 15
    '\n    Forced the datatype to dict, in case OrderedDict is used.\n    '
    return salt.utils.json.loads(salt.utils.json.dumps(config))

def _expand_config(config, defaults):
    if False:
        while True:
            i = 10
    '\n    Completed the values of the expected config for the edge cases with the default values.\n    '
    defaults.update(config)
    return defaults

def _valid_dict(dic):
    if False:
        for i in range(10):
            print('nop')
    '\n    Valid dictionary?\n    '
    return isinstance(dic, dict) and len(dic) > 0

def _valid_str(value):
    if False:
        while True:
            i = 10
    '\n    Valid str?\n    '
    return isinstance(value, str) and len(value) > 0

def _community_defaults():
    if False:
        return 10
    '\n    Returns the default values of a community.\n    '
    return {'mode': 'ro'}

def _clear_community_details(community_details):
    if False:
        return 10
    '\n    Clears community details.\n    '
    for key in ['acl', 'mode']:
        _str_elem(community_details, key)
    _mode = community_details.get['mode'] = community_details.get('mode').lower()
    if _mode in _COMMUNITY_MODE_MAP.keys():
        community_details['mode'] = _COMMUNITY_MODE_MAP.get(_mode)
    if community_details['mode'] not in ['ro', 'rw']:
        community_details['mode'] = 'ro'
    return community_details

def _str_elem(config, key):
    if False:
        print('Hello World!')
    '\n    Re-adds the value of a specific key in the dict, only in case of valid str value.\n    '
    _value = config.pop(key, '')
    if _valid_str(_value):
        config[key] = _value

def _check_config(config):
    if False:
        while True:
            i = 10
    '\n    Checks the desired config and clears interesting details.\n    '
    if not _valid_dict(config):
        return (True, '')
    _community = config.get('community')
    _community_tmp = {}
    if not _community:
        return (False, 'Must specify at least a community.')
    if _valid_str(_community):
        _community_tmp[_community] = _community_defaults()
    elif isinstance(_community, list):
        for _comm in _community:
            if _valid_str(_comm):
                _community_tmp[_comm] = _community_defaults()
            if _valid_dict(_comm):
                for (_comm_name, _comm_details) in _comm.items():
                    if _valid_str(_comm_name):
                        _community_tmp[_comm_name] = _clear_community_details(_comm_details)
    elif _valid_dict(_community):
        for (_comm_name, _comm_details) in _community.items():
            if _valid_str(_comm_name):
                _community_tmp[_comm_name] = _clear_community_details(_comm_details)
    else:
        return (False, 'Please specify a community or a list of communities.')
    if not _valid_dict(_community_tmp):
        return (False, 'Please specify at least a valid community!')
    config['community'] = _community_tmp
    for key in ['location', 'contact', 'chassis_id']:
        _str_elem(config, key)
    return (True, '')

def _retrieve_device_config():
    if False:
        i = 10
        return i + 15
    '\n    Retrieves the SNMP config from the device.\n    '
    return __salt__['snmp.config']()

def _create_diff_action(diff, diff_key, key, value):
    if False:
        while True:
            i = 10
    '\n    DRY to build diff parts (added, removed, updated).\n    '
    if diff_key not in diff.keys():
        diff[diff_key] = {}
    diff[diff_key][key] = value

def _create_diff(diff, fun, key, prev, curr):
    if False:
        print('Hello World!')
    '\n    Builds the diff dictionary.\n    '
    if not fun(prev):
        _create_diff_action(diff, 'added', key, curr)
    elif fun(prev) and (not fun(curr)):
        _create_diff_action(diff, 'removed', key, prev)
    elif not fun(curr):
        _create_diff_action(diff, 'updated', key, curr)

def _compute_diff(existing, expected):
    if False:
        i = 10
        return i + 15
    '\n    Computes the differences between the existing and the expected SNMP config.\n    '
    diff = {}
    for key in ['location', 'contact', 'chassis_id']:
        if existing.get(key) != expected.get(key):
            _create_diff(diff, _valid_str, key, existing.get(key), expected.get(key))
    for key in ['community']:
        if existing.get(key) != expected.get(key):
            _create_diff(diff, _valid_dict, key, existing.get(key), expected.get(key))
    return diff

def _configure(changes):
    if False:
        print('Hello World!')
    '\n    Calls the configuration template to apply the configuration changes on the device.\n    '
    cfgred = True
    reasons = []
    fun = 'update_config'
    for key in ['added', 'updated', 'removed']:
        _updated_changes = changes.get(key, {})
        if not _updated_changes:
            continue
        _location = _updated_changes.get('location', '')
        _contact = _updated_changes.get('contact', '')
        _community = _updated_changes.get('community', {})
        _chassis_id = _updated_changes.get('chassis_id', '')
        if key == 'removed':
            fun = 'remove_config'
        _ret = __salt__['snmp.{fun}'.format(fun=fun)](location=_location, contact=_contact, community=_community, chassis_id=_chassis_id, commit=False)
        cfgred = cfgred and _ret.get('result')
        if not _ret.get('result') and _ret.get('comment'):
            reasons.append(_ret.get('comment'))
    return {'result': cfgred, 'comment': '\n'.join(reasons) if reasons else ''}

def managed(name, config=None, defaults=None):
    if False:
        while True:
            i = 10
    '\n    Configures the SNMP on the device as specified in the SLS file.\n\n    SLS Example:\n\n    .. code-block:: yaml\n\n        snmp_example:\n            netsnmp.managed:\n                 - config:\n                    location: Honolulu, HI, US\n                 - defaults:\n                    contact: noc@cloudflare.com\n\n    Output example (for the SLS above, e.g. called snmp.sls under /router/):\n\n    .. code-block:: bash\n\n        $ sudo salt edge01.hnl01 state.sls router.snmp test=True\n        edge01.hnl01:\n        ----------\n                  ID: snmp_example\n            Function: snmp.managed\n              Result: None\n             Comment: Testing mode: configuration was not changed!\n             Started: 13:29:06.872363\n            Duration: 920.466 ms\n             Changes:\n                      ----------\n                      added:\n                          ----------\n                          chassis_id:\n                              None\n                          contact:\n                              noc@cloudflare.com\n                          location:\n                              Honolulu, HI, US\n\n        Summary for edge01.hnl01\n        ------------\n        Succeeded: 1 (unchanged=1, changed=1)\n        Failed:    0\n        ------------\n        Total states run:     1\n        Total run time: 920.466 ms\n    '
    result = False
    comment = ''
    changes = {}
    ret = {'name': name, 'changes': changes, 'result': result, 'comment': comment}
    config = _ordered_dict_to_dict(config)
    defaults = _ordered_dict_to_dict(defaults)
    expected_config = _expand_config(config, defaults)
    if not isinstance(expected_config, dict):
        ret['comment'] = 'User provided an empty SNMP config!'
        return ret
    (valid, message) = _check_config(expected_config)
    if not valid:
        ret['comment'] = 'Please provide a valid configuration: {error}'.format(error=message)
        return ret
    _device_config = _retrieve_device_config()
    if not _device_config.get('result'):
        ret['comment'] = 'Cannot retrieve SNMP config from the device: {reason}'.format(reason=_device_config.get('comment'))
        return ret
    device_config = _device_config.get('out', {})
    if device_config == expected_config:
        ret.update({'comment': 'SNMP already configured as needed.', 'result': True})
        return ret
    diff = _compute_diff(device_config, expected_config)
    changes.update(diff)
    ret.update({'changes': changes})
    if __opts__['test'] is True:
        ret.update({'result': None, 'comment': 'Testing mode: configuration was not changed!'})
        return ret
    expected_config_change = False
    result = True
    if diff:
        _configured = _configure(diff)
        if _configured.get('result'):
            expected_config_change = True
        else:
            result = False
            comment = 'Cannot push new SNMP config: \n{reason}'.format(reason=_configured.get('comment')) + comment
    if expected_config_change:
        (result, comment) = __salt__['net.config_control']()
    ret.update({'result': result, 'comment': comment})
    return ret