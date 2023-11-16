"""
NAPALM YANG
===========

NAPALM YANG basic operations.

.. versionadded:: 2017.7.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
try:
    import napalm_yang
    HAS_NAPALM_YANG = True
except ImportError:
    HAS_NAPALM_YANG = False
__virtualname__ = 'napalm_yang'
__proxyenabled__ = ['*']
log = logging.getLogger(__file__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    This module in particular requires also napalm-yang.\n    '
    if not HAS_NAPALM_YANG:
        return (False, 'Unable to load napalm_yang execution module: please install napalm-yang!')
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _get_root_object(models):
    if False:
        return 10
    '\n    Read list of models and returns a Root object with the proper models added.\n    '
    root = napalm_yang.base.Root()
    for model in models:
        current = napalm_yang
        for part in model.split('.'):
            current = getattr(current, part)
        root.add_model(current)
    return root

def diff(candidate, running, *models):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the difference between two configuration entities structured\n    according to the YANG model.\n\n    .. note::\n        This function is recommended to be used mostly as a state helper.\n\n    candidate\n        First model to compare.\n\n    running\n        Second model to compare.\n\n    models\n        A list of models to be used when comparing.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_yang.diff {} {} models.openconfig_interfaces\n\n    Output Example:\n\n    .. code-block:: python\n\n        {\n            "interfaces": {\n                "interface": {\n                    "both": {\n                        "Port-Channel1": {\n                            "config": {\n                                "mtu": {\n                                    "first": "0",\n                                    "second": "9000"\n                                }\n                            }\n                        }\n                    },\n                    "first_only": [\n                        "Loopback0"\n                    ],\n                    "second_only": [\n                        "Loopback1"\n                    ]\n                }\n            }\n        }\n    '
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    first = _get_root_object(models)
    first.load_dict(candidate)
    second = _get_root_object(models)
    second.load_dict(running)
    return napalm_yang.utils.diff(first, second)

@proxy_napalm_wrap
def parse(*models, **kwargs):
    if False:
        return 10
    '\n    Parse configuration from the device.\n\n    models\n        A list of models to be used when parsing.\n\n    config: ``False``\n        Parse config.\n\n    state: ``False``\n        Parse state.\n\n    profiles: ``None``\n        Use certain profiles to parse. If not specified, will use the device\n        default profile(s).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_yang.parse models.openconfig_interfaces\n\n    Output Example:\n\n    .. code-block:: python\n\n        {\n            "interfaces": {\n                "interface": {\n                    ".local.": {\n                        "name": ".local.",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "out-errors": 0\n                            },\n                            "enabled": True,\n                            "ifindex": 0,\n                            "last-change": 0,\n                            "oper-status": "UP",\n                            "type": "softwareLoopback"\n                        },\n                        "subinterfaces": {\n                            "subinterface": {\n                                ".local..0": {\n                                    "index": ".local..0",\n                                    "state": {\n                                        "ifindex": 0,\n                                        "name": ".local..0"\n                                    }\n                                }\n                            }\n                        }\n                    },\n                    "ae0": {\n                        "name": "ae0",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "out-errors": 0\n                            },\n                            "enabled": True,\n                            "ifindex": 531,\n                            "last-change": 255203,\n                            "mtu": 1518,\n                            "oper-status": "DOWN"\n                        },\n                        "subinterfaces": {\n                            "subinterface": {\n                                "ae0.0": {\n                                    "index": "ae0.0",\n                                    "state": {\n                                        "description": "ASDASDASD",\n                                        "ifindex": 532,\n                                        "name": "ae0.0"\n                                    }\n                                }\n                                "ae0.32767": {\n                                    "index": "ae0.32767",\n                                    "state": {\n                                        "ifindex": 535,\n                                        "name": "ae0.32767"\n                                    }\n                                }\n                            }\n                        }\n                    },\n                    "dsc": {\n                        "name": "dsc",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "out-errors": 0\n                            },\n                            "enabled": True,\n                            "ifindex": 5,\n                            "last-change": 0,\n                            "oper-status": "UP"\n                        }\n                    },\n                    "ge-0/0/0": {\n                        "name": "ge-0/0/0",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-broadcast-pkts": 0,\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "in-multicast-pkts": 0,\n                                "in-unicast-pkts": 16877,\n                                "out-broadcast-pkts": 0,\n                                "out-errors": 0,\n                                "out-multicast-pkts": 0,\n                                "out-unicast-pkts": 15742\n                            },\n                            "description": "management interface",\n                            "enabled": True,\n                            "ifindex": 507,\n                            "last-change": 258467,\n                            "mtu": 1400,\n                            "oper-status": "UP"\n                        },\n                        "subinterfaces": {\n                            "subinterface": {\n                                "ge-0/0/0.0": {\n                                    "index": "ge-0/0/0.0",\n                                    "state": {\n                                        "description": "ge-0/0/0.0",\n                                        "ifindex": 521,\n                                        "name": "ge-0/0/0.0"\n                                    }\n                                }\n                            }\n                        }\n                    }\n                    "irb": {\n                        "name": "irb",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "out-errors": 0\n                            },\n                            "enabled": True,\n                            "ifindex": 502,\n                            "last-change": 0,\n                            "mtu": 1514,\n                            "oper-status": "UP",\n                            "type": "ethernetCsmacd"\n                        }\n                    },\n                    "lo0": {\n                        "name": "lo0",\n                        "state": {\n                            "admin-status": "UP",\n                            "counters": {\n                                "in-discards": 0,\n                                "in-errors": 0,\n                                "out-errors": 0\n                            },\n                            "description": "lo0",\n                            "enabled": True,\n                            "ifindex": 6,\n                            "last-change": 0,\n                            "oper-status": "UP",\n                            "type": "softwareLoopback"\n                        },\n                        "subinterfaces": {\n                            "subinterface": {\n                                "lo0.0": {\n                                    "index": "lo0.0",\n                                    "state": {\n                                        "description": "lo0.0",\n                                        "ifindex": 16,\n                                        "name": "lo0.0"\n                                    }\n                                },\n                                "lo0.16384": {\n                                    "index": "lo0.16384",\n                                    "state": {\n                                        "ifindex": 21,\n                                        "name": "lo0.16384"\n                                    }\n                                },\n                                "lo0.16385": {\n                                    "index": "lo0.16385",\n                                    "state": {\n                                        "ifindex": 22,\n                                        "name": "lo0.16385"\n                                    }\n                                },\n                                "lo0.32768": {\n                                    "index": "lo0.32768",\n                                    "state": {\n                                        "ifindex": 248,\n                                        "name": "lo0.32768"\n                                    }\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    config = kwargs.pop('config', False)
    state = kwargs.pop('state', False)
    profiles = kwargs.pop('profiles', [])
    if not profiles and hasattr(napalm_device, 'profile'):
        profiles = napalm_device.profile
    if not profiles:
        profiles = [__grains__.get('os')]
    root = _get_root_object(models)
    parser_kwargs = {'device': napalm_device.get('DRIVER'), 'profile': profiles}
    if config:
        root.parse_config(**parser_kwargs)
    if state:
        root.parse_state(**parser_kwargs)
    return root.to_dict(filter=True)

@proxy_napalm_wrap
def get_config(data, *models, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the native config.\n\n    data\n        Dictionary structured with respect to the models referenced.\n\n    models\n        A list of models to be used when generating the config.\n\n    profiles: ``None``\n        Use certain profiles to generate the config.\n        If not specified, will use the platform default profile(s).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm_yang.get_config {} models.openconfig_interfaces\n\n    Output Example:\n\n    .. code-block:: text\n\n        interface et1\n            ip address 192.168.1.1/24\n            description Uplink1\n            mtu 9000\n        interface et2\n            ip address 192.168.2.1/24\n            description Uplink2\n            mtu 9000\n    "
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    profiles = kwargs.pop('profiles', [])
    if not profiles and hasattr(napalm_device, 'profile'):
        profiles = napalm_device.profile
    if not profiles:
        profiles = [__grains__.get('os')]
    parser_kwargs = {'profile': profiles}
    root = _get_root_object(models)
    root.load_dict(data)
    native_config = root.translate_config(**parser_kwargs)
    log.debug('Generated config')
    log.debug(native_config)
    return native_config

@proxy_napalm_wrap
def load_config(data, *models, **kwargs):
    if False:
        return 10
    '\n    Generate and load the config on the device using the OpenConfig or IETF\n    models and device profiles.\n\n    data\n        Dictionary structured with respect to the models referenced.\n\n    models\n        A list of models to be used when generating the config.\n\n    profiles: ``None``\n        Use certain profiles to generate the config.\n        If not specified, will use the platform default profile(s).\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard\n        and return the changes. Default: ``False`` and will commit\n        the changes on the device.\n\n    commit: ``True``\n        Commit? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key under the output dictionary,\n        as ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``False``\n        Should replace the config with the new generate one?\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_yang.load_config {} models.openconfig_interfaces test=True debug=True\n\n    Output Example:\n\n    .. code-block:: jinja\n\n        device1:\n            ----------\n            already_configured:\n                False\n            comment:\n            diff:\n                [edit interfaces ge-0/0/0]\n                -   mtu 1400;\n                [edit interfaces ge-0/0/0 unit 0 family inet]\n                -       dhcp;\n                [edit interfaces lo0]\n                -    unit 0 {\n                -        description lo0.0;\n                -    }\n                +    unit 1 {\n                +        description "new loopback";\n                +    }\n            loaded_config:\n                <configuration>\n                  <interfaces replace="replace">\n                    <interface>\n                      <name>ge-0/0/0</name>\n                      <unit>\n                        <name>0</name>\n                        <family>\n                          <inet/>\n                        </family>\n                        <description>ge-0/0/0.0</description>\n                      </unit>\n                      <description>management interface</description>\n                    </interface>\n                    <interface>\n                      <name>ge-0/0/1</name>\n                      <disable/>\n                      <description>ge-0/0/1</description>\n                    </interface>\n                    <interface>\n                      <name>ae0</name>\n                      <unit>\n                        <name>0</name>\n                        <vlan-id>100</vlan-id>\n                        <family>\n                          <inet>\n                            <address>\n                              <name>192.168.100.1/24</name>\n                            </address>\n                            <address>\n                              <name>172.20.100.1/24</name>\n                            </address>\n                          </inet>\n                        </family>\n                        <description>a description</description>\n                      </unit>\n                      <vlan-tagging/>\n                      <unit>\n                        <name>1</name>\n                        <vlan-id>1</vlan-id>\n                        <family>\n                          <inet>\n                            <address>\n                              <name>192.168.101.1/24</name>\n                            </address>\n                          </inet>\n                        </family>\n                        <disable/>\n                        <description>ae0.1</description>\n                      </unit>\n                      <vlan-tagging/>\n                      <unit>\n                        <name>2</name>\n                        <vlan-id>2</vlan-id>\n                        <family>\n                          <inet>\n                            <address>\n                              <name>192.168.102.1/24</name>\n                            </address>\n                          </inet>\n                        </family>\n                        <description>ae0.2</description>\n                      </unit>\n                      <vlan-tagging/>\n                    </interface>\n                    <interface>\n                      <name>lo0</name>\n                      <unit>\n                        <name>1</name>\n                        <description>new loopback</description>\n                      </unit>\n                      <description>lo0</description>\n                    </interface>\n                  </interfaces>\n                </configuration>\n            result:\n                True\n    '
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    config = get_config(data, *models, **kwargs)
    test = kwargs.pop('test', False)
    debug = kwargs.pop('debug', False)
    commit = kwargs.pop('commit', True)
    replace = kwargs.pop('replace', False)
    return __salt__['net.load_config'](text=config, test=test, debug=debug, commit=commit, replace=replace, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def compliance_report(data, *models, **kwargs):
    if False:
        print('Hello World!')
    '\n    Return the compliance report using YANG objects.\n\n    data\n        Dictionary structured with respect to the models referenced.\n\n    models\n        A list of models to be used when generating the config.\n\n    filepath\n        The absolute path to the validation file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_yang.compliance_report {} models.openconfig_interfaces filepath=~/validate.yml\n\n    Output Example:\n\n    .. code-block:: json\n\n        {\n          "skipped": [],\n          "complies": true,\n          "get_interfaces_ip": {\n            "missing": [],\n            "complies": true,\n            "present": {\n              "ge-0/0/0.0": {\n                "complies": true,\n                "nested": true\n              }\n            },\n            "extra": []\n          }\n        }\n    '
    if isinstance(models, tuple) and isinstance(models[0], list):
        models = models[0]
    filepath = kwargs.pop('filepath', '')
    root = _get_root_object(models)
    root.load_dict(data)
    return root.compliance_report(filepath)