"""
Network Probes
===============

Configure RPM (JunOS)/SLA (Cisco) probes on the device via NAPALM proxy.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------

- :mod:`napalm probes management module <salt.modules.napalm_probes>`

.. versionadded:: 2016.11.0
"""
import copy
import logging
import salt.utils.json
import salt.utils.napalm
log = logging.getLogger(__name__)
__virtualname__ = 'probes'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _default_ret(name):
    if False:
        while True:
            i = 10
    '\n    Returns a default structure of the dictionary to be returned as output of the state functions.\n    '
    return {'name': name, 'result': False, 'changes': {}, 'comment': ''}

def _retrieve_rpm_probes():
    if False:
        print('Hello World!')
    '\n    Will retrieve the probes from the network device using salt module "probes" throught NAPALM proxy.\n    '
    return __salt__['probes.config']()

def _expand_probes(probes, defaults):
    if False:
        print('Hello World!')
    '\n    Updates the probes dictionary with different levels of default values.\n    '
    expected_probes = {}
    for (probe_name, probe_test) in probes.items():
        if probe_name not in expected_probes.keys():
            expected_probes[probe_name] = {}
        probe_defaults = probe_test.pop('defaults', {})
        for (test_name, test_details) in probe_test.items():
            test_defaults = test_details.pop('defaults', {})
            expected_test_details = copy.deepcopy(defaults)
            expected_test_details.update(probe_defaults)
            expected_test_details.update(test_defaults)
            expected_test_details.update(test_details)
            if test_name not in expected_probes[probe_name].keys():
                expected_probes[probe_name][test_name] = expected_test_details
    return expected_probes

def _clean_probes(probes):
    if False:
        return 10
    '\n    Will remove empty and useless values from the probes dictionary.\n    '
    probes = _ordered_dict_to_dict(probes)
    probes_copy = copy.deepcopy(probes)
    for (probe_name, probe_tests) in probes_copy.items():
        if not probe_tests:
            probes.pop(probe_name)
            continue
        for (test_name, test_params) in probe_tests.items():
            if not test_params:
                probes[probe_name].pop(test_name)
            if not probes.get(probe_name):
                probes.pop(probe_name)
    return True

def _compare_probes(configured_probes, expected_probes):
    if False:
        return 10
    '\n    Compares configured probes on the device with the expected configuration and returns the differences.\n    '
    new_probes = {}
    update_probes = {}
    remove_probes = {}
    if not configured_probes:
        return {'add': expected_probes}
    if not expected_probes:
        return {'remove': configured_probes}
    configured_probes_keys_set = set(configured_probes.keys())
    expected_probes_keys_set = set(expected_probes.keys())
    new_probes_keys_set = expected_probes_keys_set - configured_probes_keys_set
    remove_probes_keys_set = configured_probes_keys_set - expected_probes_keys_set
    for probe_name in new_probes_keys_set:
        new_probes[probe_name] = expected_probes.pop(probe_name)
    for probe_name in remove_probes_keys_set:
        remove_probes[probe_name] = configured_probes.pop(probe_name)
    for (probe_name, probe_tests) in expected_probes.items():
        configured_probe_tests = configured_probes.get(probe_name, {})
        configured_tests_keys_set = set(configured_probe_tests.keys())
        expected_tests_keys_set = set(probe_tests.keys())
        new_tests_keys_set = expected_tests_keys_set - configured_tests_keys_set
        remove_tests_keys_set = configured_tests_keys_set - expected_tests_keys_set
        for test_name in new_tests_keys_set:
            if probe_name not in new_probes.keys():
                new_probes[probe_name] = {}
            new_probes[probe_name].update({test_name: probe_tests.pop(test_name)})
        for test_name in remove_tests_keys_set:
            if probe_name not in remove_probes.keys():
                remove_probes[probe_name] = {}
            remove_probes[probe_name].update({test_name: configured_probe_tests.pop(test_name)})
        for (test_name, test_params) in probe_tests.items():
            configured_test_params = configured_probe_tests.get(test_name, {})
            if test_params != configured_test_params:
                if probe_name not in update_probes.keys():
                    update_probes[probe_name] = {}
                update_probes[probe_name].update({test_name: test_params})
    return {'add': new_probes, 'update': update_probes, 'remove': remove_probes}

def _ordered_dict_to_dict(probes):
    if False:
        return 10
    'Mandatory to be dict type in order to be used in the NAPALM Jinja template.'
    return salt.utils.json.loads(salt.utils.json.dumps(probes))

def _set_rpm_probes(probes):
    if False:
        return 10
    '\n    Calls the Salt module "probes" to configure the probes on the device.\n    '
    return __salt__['probes.set_probes'](_ordered_dict_to_dict(probes), commit=False)

def _schedule_probes(probes):
    if False:
        i = 10
        return i + 15
    '\n    Calls the Salt module "probes" to schedule the configured probes on the device.\n    '
    return __salt__['probes.schedule_probes'](_ordered_dict_to_dict(probes), commit=False)

def _delete_rpm_probes(probes):
    if False:
        i = 10
        return i + 15
    '\n    Calls the Salt module "probes" to delete probes from the device.\n    '
    return __salt__['probes.delete_probes'](_ordered_dict_to_dict(probes), commit=False)

def managed(name, probes, defaults=None):
    if False:
        return 10
    '\n    Ensure the networks device is configured as specified in the state SLS file.\n    Probes not specified will be removed, while probes not confiured as expected will trigger config updates.\n\n    :param probes: Defines the probes as expected to be configured on the\n        device.  In order to ease the configuration and avoid repeating the\n        same parameters for each probe, the next parameter (defaults) can be\n        used, providing common characteristics.\n\n    :param defaults: Specifies common parameters for the probes.\n\n    SLS Example:\n\n    .. code-block:: yaml\n\n        rpmprobes:\n            probes.managed:\n                - probes:\n                    probe_name1:\n                        probe1_test1:\n                            source: 192.168.0.2\n                            target: 192.168.0.1\n                        probe1_test2:\n                            target: 172.17.17.1\n                        probe1_test3:\n                            target: 8.8.8.8\n                            probe_type: http-ping\n                    probe_name2:\n                        probe2_test1:\n                            test_interval: 100\n                - defaults:\n                    target: 10.10.10.10\n                    probe_count: 15\n                    test_interval: 3\n                    probe_type: icmp-ping\n\n    In the probes configuration, the only mandatory attribute is *target*\n    (specified either in probes configuration, either in the defaults\n    dictionary).  All the other parameters will use the operating system\n    defaults, if not provided:\n\n    - ``source`` - Specifies the source IP Address to be used during the tests.  If\n      not specified will use the IP Address of the logical interface loopback0.\n\n    - ``target`` - Destination IP Address.\n    - ``probe_count`` - Total number of probes per test (1..15). System\n      defaults: 1 on both JunOS & Cisco.\n    - ``probe_interval`` - Delay between tests (0..86400 seconds). System\n      defaults: 3 on JunOS, 5 on Cisco.\n    - ``probe_type`` - Probe request type. Available options:\n\n      - icmp-ping\n      - tcp-ping\n      - udp-ping\n\n    Using the example configuration above, after running the state, on the device will be configured 4 probes,\n    with the following properties:\n\n    .. code-block:: yaml\n\n        probe_name1:\n            probe1_test1:\n                source: 192.168.0.2\n                target: 192.168.0.1\n                probe_count: 15\n                test_interval: 3\n                probe_type: icmp-ping\n            probe1_test2:\n                target: 172.17.17.1\n                probe_count: 15\n                test_interval: 3\n                probe_type: icmp-ping\n            probe1_test3:\n                target: 8.8.8.8\n                probe_count: 15\n                test_interval: 3\n                probe_type: http-ping\n        probe_name2:\n            probe2_test1:\n                target: 10.10.10.10\n                probe_count: 15\n                test_interval: 3\n                probe_type: icmp-ping\n    '
    ret = _default_ret(name)
    result = True
    comment = ''
    rpm_probes_config = _retrieve_rpm_probes()
    if not rpm_probes_config.get('result'):
        ret.update({'result': False, 'comment': 'Cannot retrieve configurtion of the probes from the device: {reason}'.format(reason=rpm_probes_config.get('comment'))})
        return ret
    configured_probes = rpm_probes_config.get('out', {})
    if not isinstance(defaults, dict):
        defaults = {}
    expected_probes = _expand_probes(probes, defaults)
    _clean_probes(configured_probes)
    _clean_probes(expected_probes)
    diff = _compare_probes(configured_probes, expected_probes)
    add_probes = diff.get('add')
    update_probes = diff.get('update')
    remove_probes = diff.get('remove')
    changes = {'added': _ordered_dict_to_dict(add_probes), 'updated': _ordered_dict_to_dict(update_probes), 'removed': _ordered_dict_to_dict(remove_probes)}
    ret.update({'changes': changes})
    if __opts__['test'] is True:
        ret.update({'comment': 'Testing mode: configuration was not changed!', 'result': None})
        return ret
    config_change_expected = False
    if add_probes:
        added = _set_rpm_probes(add_probes)
        if added.get('result'):
            config_change_expected = True
        else:
            result = False
            comment += 'Cannot define new probes: {reason}\n'.format(reason=added.get('comment'))
    if update_probes:
        updated = _set_rpm_probes(update_probes)
        if updated.get('result'):
            config_change_expected = True
        else:
            result = False
            comment += 'Cannot update probes: {reason}\n'.format(reason=updated.get('comment'))
    if remove_probes:
        removed = _delete_rpm_probes(remove_probes)
        if removed.get('result'):
            config_change_expected = True
        else:
            result = False
            comment += 'Cannot remove probes! {reason}\n'.format(reason=removed.get('comment'))
    if config_change_expected:
        (result, comment) = __salt__['net.config_control']()
    add_scheduled = _schedule_probes(add_probes)
    if add_scheduled.get('result'):
        (result, comment) = __salt__['net.config_control']()
    if config_change_expected:
        if result and comment == '':
            comment = 'Probes updated successfully!'
    ret.update({'result': result, 'comment': comment})
    return ret