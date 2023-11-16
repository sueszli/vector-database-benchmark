"""
NAPALM YANG state
=================

Manage the configuration of network devices according to
the YANG models (OpenConfig/IETF).

.. versionadded:: 2017.7.0

Dependencies
------------

- napalm-yang
- pyangbing > 0.5.11

To be able to load configuration on network devices,
it requires NAPALM_ library to be installed:  ``pip install napalm``.
Please check Installation_ for complete details.

.. _NAPALM: https://napalm.readthedocs.io
.. _Installation: https://napalm.readthedocs.io/en/latest/installation/index.html
"""
import logging
import salt.utils.files
import salt.utils.json
import salt.utils.napalm
import salt.utils.stringutils
import salt.utils.yaml
log = logging.getLogger(__file__)
try:
    import napalm_yang
    HAS_NAPALM_YANG = True
except ImportError:
    HAS_NAPALM_YANG = False
__virtualname__ = 'napalm_yang'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    This module in particular requires also napalm-yang.\n    '
    if not HAS_NAPALM_YANG:
        return (False, 'Unable to load napalm_yang execution module: please install napalm-yang!')
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def managed(name, data, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Manage the device configuration given the input data structured\n    according to the YANG models.\n\n    data\n        YANG structured data.\n\n    models\n         A list of models to be used when generating the config.\n\n    profiles: ``None``\n        Use certain profiles to generate the config.\n        If not specified, will use the platform default profile(s).\n\n    compliance_report: ``False``\n        Return the compliance report in the comment.\n\n        .. versionadded:: 2017.7.3\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard\n        and return the changes. Default: ``False`` and will commit\n        the changes on the device.\n\n    commit: ``True``\n        Commit? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key under the output dictionary,\n        as ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``False``\n        Should replace the config with the new generate one?\n\n    State SLS example:\n\n    .. code-block:: jinja\n\n        {%- set expected_config =  pillar.get(\'openconfig_interfaces_cfg\') -%}\n        interfaces_config:\n          napalm_yang.managed:\n            - data: {{ expected_config | json }}\n            - models:\n              - models.openconfig_interfaces\n            - debug: true\n\n    Pillar example:\n\n    .. code-block:: yaml\n\n        openconfig_interfaces_cfg:\n          _kwargs:\n            filter: true\n          interfaces:\n            interface:\n              Et1:\n                config:\n                  mtu: 9000\n              Et2:\n                config:\n                  description: "description example"\n    '
    models = kwargs.get('models', None)
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    ret = salt.utils.napalm.default_ret(name)
    test = kwargs.get('test', False) or __opts__.get('test', False)
    debug = kwargs.get('debug', False) or __opts__.get('debug', False)
    commit = kwargs.get('commit', True) or __opts__.get('commit', True)
    replace = kwargs.get('replace', False) or __opts__.get('replace', False)
    return_compliance_report = kwargs.get('compliance_report', False) or __opts__.get('compliance_report', False)
    profiles = kwargs.get('profiles', [])
    temp_file = __salt__['temp.file']()
    log.debug('Creating temp file: %s', temp_file)
    if 'to_dict' not in data:
        data = {'to_dict': data}
    data = [data]
    with salt.utils.files.fopen(temp_file, 'w') as file_handle:
        salt.utils.yaml.safe_dump(salt.utils.json.loads(salt.utils.json.dumps(data)), file_handle, encoding='utf-8')
    device_config = __salt__['napalm_yang.parse'](*models, config=True, profiles=profiles)
    log.debug('Parsed the config from the device:')
    log.debug(device_config)
    compliance_report = __salt__['napalm_yang.compliance_report'](device_config, *models, filepath=temp_file)
    log.debug('Compliance report:')
    log.debug(compliance_report)
    complies = compliance_report.get('complies', False)
    if complies:
        ret.update({'result': True, 'comment': 'Already configured as required.'})
        log.debug('All good here.')
        return ret
    log.debug('Does not comply, trying to generate and load config')
    data = data[0]['to_dict']
    if '_kwargs' in data:
        data.pop('_kwargs')
    loaded_changes = __salt__['napalm_yang.load_config'](data, *models, profiles=profiles, test=test, debug=debug, commit=commit, replace=replace)
    log.debug('Loaded config result:')
    log.debug(loaded_changes)
    __salt__['file.remove'](temp_file)
    loaded_changes['compliance_report'] = compliance_report
    return salt.utils.napalm.loaded_ret(ret, loaded_changes, test, debug, opts=__opts__, compliance_report=return_compliance_report)

def configured(name, data, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Configure the network device, given the input data strucuted\n    according to the YANG models.\n\n    .. note::\n        The main difference between this function and ``managed``\n        is that the later generates and loads the configuration\n        only when there are differences between the existing\n        configuration on the device and the expected\n        configuration. Depending on the platform and hardware\n        capabilities, one could be more optimal than the other.\n        Additionally, the output of the ``managed`` is different,\n        in such a way that the ``pchange`` field in the output\n        contains structured data, rather than text.\n\n    data\n        YANG structured data.\n\n    models\n         A list of models to be used when generating the config.\n\n    profiles: ``None``\n        Use certain profiles to generate the config.\n        If not specified, will use the platform default profile(s).\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard\n        and return the changes. Default: ``False`` and will commit\n        the changes on the device.\n\n    commit: ``True``\n        Commit? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key under the output dictionary,\n        as ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``False``\n        Should replace the config with the new generate one?\n\n    State SLS example:\n\n    .. code-block:: jinja\n\n        {%- set expected_config =  pillar.get(\'openconfig_interfaces_cfg\') -%}\n        interfaces_config:\n          napalm_yang.configured:\n            - data: {{ expected_config | json }}\n            - models:\n              - models.openconfig_interfaces\n            - debug: true\n\n    Pillar example:\n\n    .. code-block:: yaml\n\n        openconfig_interfaces_cfg:\n          _kwargs:\n            filter: true\n          interfaces:\n            interface:\n              Et1:\n                config:\n                  mtu: 9000\n              Et2:\n                config:\n                  description: "description example"\n    '
    models = kwargs.get('models', None)
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    ret = salt.utils.napalm.default_ret(name)
    test = kwargs.get('test', False) or __opts__.get('test', False)
    debug = kwargs.get('debug', False) or __opts__.get('debug', False)
    commit = kwargs.get('commit', True) or __opts__.get('commit', True)
    replace = kwargs.get('replace', False) or __opts__.get('replace', False)
    profiles = kwargs.get('profiles', [])
    if '_kwargs' in data:
        data.pop('_kwargs')
    loaded_changes = __salt__['napalm_yang.load_config'](data, *models, profiles=profiles, test=test, debug=debug, commit=commit, replace=replace)
    return salt.utils.napalm.loaded_ret(ret, loaded_changes, test, debug)