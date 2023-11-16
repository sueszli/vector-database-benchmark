"""
NAPALM Probes
=============

Manages RPM/SLA probes on the network device.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------

- :mod:`napalm proxy minion <salt.proxy.napalm>`
- :mod:`NET basic features <salt.modules.napalm_network>`

.. seealso::
    :mod:`Probes configuration management state <salt.states.probes>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
log = logging.getLogger(__file__)
__virtualname__ = 'probes'
__proxyenabled__ = ['napalm']

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

@proxy_napalm_wrap
def config(**kwargs):
    if False:
        print('Hello World!')
    "\n    Returns the configuration of the RPM probes.\n\n    :return: A dictionary containing the configuration of the RPM/SLA probes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' probes.config\n\n    Output Example:\n\n    .. code-block:: python\n\n        {\n            'probe1':{\n                'test1': {\n                    'probe_type'   : 'icmp-ping',\n                    'target'       : '192.168.0.1',\n                    'source'       : '192.168.0.2',\n                    'probe_count'  : 13,\n                    'test_interval': 3\n                },\n                'test2': {\n                    'probe_type'   : 'http-ping',\n                    'target'       : '172.17.17.1',\n                    'source'       : '192.17.17.2',\n                    'probe_count'  : 5,\n                    'test_interval': 60\n                }\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_probes_config', **{})

@proxy_napalm_wrap
def results(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Provides the results of the measurements of the RPM/SLA probes.\n\n    :return a dictionary with the results of the probes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' probes.results\n\n\n    Output example:\n\n    .. code-block:: python\n\n        {\n            'probe1':  {\n                'test1': {\n                    'last_test_min_delay'   : 63.120,\n                    'global_test_min_delay' : 62.912,\n                    'current_test_avg_delay': 63.190,\n                    'global_test_max_delay' : 177.349,\n                    'current_test_max_delay': 63.302,\n                    'global_test_avg_delay' : 63.802,\n                    'last_test_avg_delay'   : 63.438,\n                    'last_test_max_delay'   : 65.356,\n                    'probe_type'            : 'icmp-ping',\n                    'rtt'                   : 63.138,\n                    'last_test_loss'        : 0,\n                    'round_trip_jitter'     : -59.0,\n                    'target'                : '192.168.0.1',\n                    'source'                : '192.168.0.2',\n                    'probe_count'           : 15,\n                    'current_test_min_delay': 63.138\n                },\n                'test2': {\n                    'last_test_min_delay'   : 176.384,\n                    'global_test_min_delay' : 169.226,\n                    'current_test_avg_delay': 177.098,\n                    'global_test_max_delay' : 292.628,\n                    'current_test_max_delay': 180.055,\n                    'global_test_avg_delay' : 177.959,\n                    'last_test_avg_delay'   : 177.178,\n                    'last_test_max_delay'   : 184.671,\n                    'probe_type'            : 'icmp-ping',\n                    'rtt'                   : 176.449,\n                    'last_test_loss'        : 0,\n                    'round_trip_jitter'     : -34.0,\n                    'target'                : '172.17.17.1',\n                    'source'                : '172.17.17.2',\n                    'probe_count'           : 15,\n                    'current_test_min_delay': 176.402\n                }\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_probes_results', **{})

@proxy_napalm_wrap
def set_probes(probes, test=False, commit=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Configures RPM/SLA probes on the device.\n    Calls the configuration template \'set_probes\' from the NAPALM library,\n    providing as input a rich formatted dictionary with the configuration details of the probes to be configured.\n\n    :param probes: Dictionary formatted as the output of the function config()\n\n    :param test: Dry run? If set as True, will apply the config, discard and return the changes. Default: False\n\n    :param commit: Commit? (default: True) Sometimes it is not needed to commit\n        the config immediately after loading the changes. E.g.: a state loads a\n        couple of parts (add / remove / update) and would not be optimal to\n        commit after each operation.  Also, from the CLI when the user needs to\n        apply the similar changes before committing, can specify commit=False\n        and will not discard the config.\n\n    :raise MergeConfigException: If there is an error on the configuration sent.\n    :return a dictionary having the following keys:\n\n        * result (bool): if the config was applied successfully. It is `False`\n          only in case of failure. In case there are no changes to be applied\n          and successfully performs all operations it is still `True` and so\n          will be the `already_configured` flag (example below)\n        * comment (str): a message for the user\n        * already_configured (bool): flag to check if there were no changes applied\n        * diff (str): returns the config changes applied\n\n    Input example - via state/script:\n\n    .. code-block:: python\n\n        probes = {\n            \'new_probe\':{\n                \'new_test1\': {\n                    \'probe_type\'   : \'icmp-ping\',\n                    \'target\'       : \'192.168.0.1\',\n                    \'source\'       : \'192.168.0.2\',\n                    \'probe_count\'  : 13,\n                    \'test_interval\': 3\n                },\n                \'new_test2\': {\n                    \'probe_type\'   : \'http-ping\',\n                    \'target\'       : \'172.17.17.1\',\n                    \'source\'       : \'192.17.17.2\',\n                    \'probe_count\'  : 5,\n                    \'test_interval\': 60\n                }\n            }\n        }\n        set_probes(probes)\n\n    CLI Example - to push changes on the fly (not recommended):\n\n    .. code-block:: bash\n\n        salt \'junos_minion\' probes.set_probes "{\'new_probe\':{\'new_test1\':{\'probe_type\':\'icmp-ping\',            \'target\':\'192.168.0.1\',\'source\':\'192.168.0.2\',\'probe_count\':13,\'test_interval\':3}}}" test=True\n\n    Output example - for the CLI example above:\n\n    .. code-block:: yaml\n\n        junos_minion:\n            ----------\n            already_configured:\n                False\n            comment:\n                Configuration discarded.\n            diff:\n                [edit services rpm]\n                     probe transit { ... }\n                +    probe new_probe {\n                +        test new_test1 {\n                +            probe-type icmp-ping;\n                +            target address 192.168.0.1;\n                +            probe-count 13;\n                +            test-interval 3;\n                +            source-address 192.168.0.2;\n                +        }\n                +    }\n            result:\n                True\n    '
    return __salt__['net.load_template']('set_probes', probes=probes, test=test, commit=commit, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def delete_probes(probes, test=False, commit=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Removes RPM/SLA probes from the network device.\n    Calls the configuration template 'delete_probes' from the NAPALM library,\n    providing as input a rich formatted dictionary with the configuration details of the probes to be removed\n    from the configuration of the device.\n\n    :param probes: Dictionary with a similar format as the output dictionary of\n        the function config(), where the details are not necessary.\n    :param test: Dry run? If set as True, will apply the config, discard and\n        return the changes. Default: False\n    :param commit: Commit? (default: True) Sometimes it is not needed to commit\n        the config immediately after loading the changes. E.g.: a state loads a\n        couple of parts (add / remove / update) and would not be optimal to\n        commit after each operation.  Also, from the CLI when the user needs to\n        apply the similar changes before committing, can specify commit=False\n        and will not discard the config.\n\n    :raise MergeConfigException: If there is an error on the configuration sent.\n    :return: A dictionary having the following keys:\n\n    - result (bool): if the config was applied successfully. It is `False` only\n      in case of failure. In case there are no changes to be applied and\n      successfully performs all operations it is still `True` and so will be\n      the `already_configured` flag (example below)\n    - comment (str): a message for the user\n    - already_configured (bool): flag to check if there were no changes applied\n    - diff (str): returns the config changes applied\n\n    Input example:\n\n    .. code-block:: python\n\n        probes = {\n            'existing_probe':{\n                'existing_test1': {},\n                'existing_test2': {}\n            }\n        }\n\n    "
    return __salt__['net.load_template']('delete_probes', probes=probes, test=test, commit=commit, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def schedule_probes(probes, test=False, commit=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Will schedule the probes. On Cisco devices, it is not enough to define the\n    probes, it is also necessary to schedule them.\n\n    This function calls the configuration template ``schedule_probes`` from the\n    NAPALM library, providing as input a rich formatted dictionary with the\n    names of the probes and the tests to be scheduled.\n\n    :param probes: Dictionary with a similar format as the output dictionary of\n        the function config(), where the details are not necessary.\n    :param test: Dry run? If set as True, will apply the config, discard and\n        return the changes. Default: False\n    :param commit: Commit? (default: True) Sometimes it is not needed to commit\n        the config immediately after loading the changes. E.g.: a state loads a\n        couple of parts (add / remove / update) and would not be optimal to\n        commit after each operation.  Also, from the CLI when the user needs to\n        apply the similar changes before committing, can specify commit=False\n        and will not discard the config.\n\n    :raise MergeConfigException: If there is an error on the configuration sent.\n    :return: a dictionary having the following keys:\n\n    - result (bool): if the config was applied successfully. It is `False` only\n      in case of failure. In case there are no changes to be applied and\n      successfully performs all operations it is still `True` and so will be\n      the `already_configured` flag (example below)\n    - comment (str): a message for the user\n    - already_configured (bool): flag to check if there were no changes applied\n    - diff (str): returns the config changes applied\n\n    Input example:\n\n    .. code-block:: python\n\n        probes = {\n            'new_probe':{\n                'new_test1': {},\n                'new_test2': {}\n            }\n        }\n    "
    return __salt__['net.load_template']('schedule_probes', probes=probes, test=test, commit=commit, inherit_napalm_device=napalm_device)