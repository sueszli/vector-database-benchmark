"""
NAPALM SNMP
===========

Manages SNMP on network devices.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------
- :mod:`NAPALM proxy minion <salt.proxy.napalm>`
- :mod:`NET basic features <salt.modules.napalm_network>`

.. seealso::
    :mod:`SNMP configuration management state <salt.states.netsnmp>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
log = logging.getLogger(__file__)
__virtualname__ = 'snmp'
__proxyenabled__ = ['napalm']
__virtual_aliases__ = ('napalm_snmp',)

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
    "\n    Returns the SNMP configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snmp.config\n    "
    return salt.utils.napalm.call(napalm_device, 'get_snmp_information', **{})

@proxy_napalm_wrap
def remove_config(chassis_id=None, community=None, contact=None, location=None, test=False, commit=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Removes a configuration element from the SNMP configuration.\n\n    :param chassis_id: (optional) Chassis ID\n\n    :param community: (optional) A dictionary having the following optional keys:\n\n    - acl (if any policy / ACL need to be set)\n    - mode: rw or ro. Default: ro\n\n    :param contact: Contact details\n\n    :param location: Location\n\n    :param test: Dry run? If set as True, will apply the config, discard and return the changes. Default: False\n\n    :param commit: Commit? (default: True) Sometimes it is not needed to commit\n        the config immediately after loading the changes. E.g.: a state loads a\n        couple of parts (add / remove / update) and would not be optimal to\n        commit after each operation.  Also, from the CLI when the user needs to\n        apply the similar changes before committing, can specify commit=False\n        and will not discard the config.\n\n    :raise MergeConfigException: If there is an error on the configuration sent.\n    :return: A dictionary having the following keys:\n\n    - result (bool): if the config was applied successfully. It is `False`\n      only in case of failure. In case there are no changes to be applied\n      and successfully performs all operations it is still `True` and so\n      will be the `already_configured` flag (example below)\n    - comment (str): a message for the user\n    - already_configured (bool): flag to check if there were no changes applied\n    - diff (str): returns the config changes applied\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snmp.remove_config community='abcd'\n    "
    dic = {'template_name': 'delete_snmp_config', 'test': test, 'commit': commit}
    if chassis_id:
        dic['chassis_id'] = chassis_id
    if community:
        dic['community'] = community
    if contact:
        dic['contact'] = contact
    if location:
        dic['location'] = location
    dic['inherit_napalm_device'] = napalm_device
    return __salt__['net.load_template'](**dic)

@proxy_napalm_wrap
def update_config(chassis_id=None, community=None, contact=None, location=None, test=False, commit=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Updates the SNMP configuration.\n\n    :param chassis_id: (optional) Chassis ID\n    :param community: (optional) A dictionary having the following optional keys:\n\n    - acl (if any policy / ACL need to be set)\n    - mode: rw or ro. Default: ro\n\n    :param contact: Contact details\n    :param location: Location\n    :param test: Dry run? If set as True, will apply the config, discard and return the changes. Default: False\n    :param commit: Commit? (default: True) Sometimes it is not needed to commit the config immediately\n        after loading the changes. E.g.: a state loads a couple of parts (add / remove / update)\n        and would not be optimal to commit after each operation.\n        Also, from the CLI when the user needs to apply the similar changes before committing,\n        can specify commit=False and will not discard the config.\n    :raise MergeConfigException: If there is an error on the configuration sent.\n    :return a dictionary having the following keys:\n\n    - result (bool): if the config was applied successfully. It is `False` only\n      in case of failure. In case there are no changes to be applied and\n      successfully performs all operations it is still `True` and so will be\n      the `already_configured` flag (example below)\n    - comment (str): a message for the user\n    - already_configured (bool): flag to check if there were no changes applied\n    - diff (str): returns the config changes applied\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'edge01.lon01\' snmp.update_config location="Greenwich, UK" test=True\n\n    Output example (for the CLI example above):\n\n    .. code-block:: yaml\n\n        edge01.lon01:\n            ----------\n            already_configured:\n                False\n            comment:\n                Configuration discarded.\n            diff:\n                [edit snmp]\n                -  location "London, UK";\n                +  location "Greenwich, UK";\n            result:\n                True\n    '
    dic = {'template_name': 'snmp_config', 'test': test, 'commit': commit}
    if chassis_id:
        dic['chassis_id'] = chassis_id
    if community:
        dic['community'] = community
    if contact:
        dic['contact'] = contact
    if location:
        dic['location'] = location
    dic['inherit_napalm_device'] = napalm_device
    return __salt__['net.load_template'](**dic)