"""
The module file for vyos_facts
"""
from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': [u'preview'], 'supported_by': 'network'}
DOCUMENTATION = "module: vyos_facts\nshort_description: Get facts about vyos devices.\ndescription:\n- Collects facts from network devices running the vyos operating system. This module\n  places the facts gathered in the fact tree keyed by the respective resource name.  The\n  facts module will always collect a base set of facts from the device and can enable\n  or disable collection of additional facts.\nauthor:\n- Nathaniel Case (@qalthos)\n- Nilashish Chakraborty (@Nilashishc)\n- Rohit Thakur (@rohitthakur2590)\nextends_documentation_fragment:\n- vyos.vyos.vyos\nnotes:\n- Tested against VyOS 1.1.8 (helium).\n- This module works with connection C(network_cli). See L(the VyOS OS Platform Options,../network/user_guide/platform_vyos.html).\noptions:\n  gather_subset:\n    description:\n    - When supplied, this argument will restrict the facts collected to a given subset.  Possible\n      values for this argument include all, default, config, and neighbors. Can specify\n      a list of values to include a larger subset. Values can also be used with an\n      initial C(M(!)) to specify that a specific subset should not be collected.\n    required: false\n    default: '!config'\n  gather_network_resources:\n    description:\n    - When supplied, this argument will restrict the facts collected to a given subset.\n      Possible values for this argument include all and the resources like interfaces.\n      Can specify a list of values to include a larger subset. Values can also be\n      used with an initial C(M(!)) to specify that a specific subset should not be\n      collected. Valid subsets are 'all', 'interfaces', 'l3_interfaces', 'lag_interfaces',\n      'lldp_global', 'lldp_interfaces', 'static_routes', 'firewall_rules'.\n    required: false\n"
EXAMPLES = '\n# Gather all facts\n- vyos_facts:\n    gather_subset: all\n    gather_network_resources: all\n\n# collect only the config and default facts\n- vyos_facts:\n    gather_subset: config\n\n# collect everything exception the config\n- vyos_facts:\n    gather_subset: "!config"\n\n# Collect only the interfaces facts\n- vyos_facts:\n    gather_subset:\n      - \'!all\'\n      - \'!min\'\n    gather_network_resources:\n      - interfaces\n\n# Do not collect interfaces facts\n- vyos_facts:\n    gather_network_resources:\n      - "!interfaces"\n\n# Collect interfaces and minimal default facts\n- vyos_facts:\n    gather_subset: min\n    gather_network_resources: interfaces\n'
RETURN = '\nansible_net_config:\n  description: The running-config from the device\n  returned: when config is configured\n  type: str\nansible_net_commits:\n  description: The set of available configuration revisions\n  returned: when present\n  type: list\nansible_net_hostname:\n  description: The configured system hostname\n  returned: always\n  type: str\nansible_net_model:\n  description: The device model string\n  returned: always\n  type: str\nansible_net_serialnum:\n  description: The serial number of the device\n  returned: always\n  type: str\nansible_net_version:\n  description: The version of the software running\n  returned: always\n  type: str\nansible_net_neighbors:\n  description: The set of LLDP neighbors\n  returned: when interface is configured\n  type: list\nansible_net_gather_subset:\n  description: The list of subsets gathered by the module\n  returned: always\n  type: list\nansible_net_api:\n  description: The name of the transport\n  returned: always\n  type: str\nansible_net_python_version:\n  description: The Python version Ansible controller is using\n  returned: always\n  type: str\nansible_net_gather_network_resources:\n  description: The list of fact resource subsets collected from the device\n  returned: always\n  type: list\n'
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.facts.facts import FactsArgs
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import vyos_argument_spec

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    Main entry point for module execution\n\n    :returns: ansible_facts\n    '
    argument_spec = FactsArgs.argument_spec
    argument_spec.update(vyos_argument_spec)
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    warnings = []
    if module.params['gather_subset'] == '!config':
        warnings.append('default value for `gather_subset` will be changed to `min` from `!config` v2.11 onwards')
    result = Facts(module).get_facts()
    (ansible_facts, additional_warnings) = result
    warnings.extend(additional_warnings)
    module.exit_json(ansible_facts=ansible_facts, warnings=warnings)
if __name__ == '__main__':
    main()