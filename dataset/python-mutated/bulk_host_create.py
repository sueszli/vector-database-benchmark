from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: bulk_host_create\nauthor: "Seth Foster (@fosterseth)"\nshort_description: Bulk host create in Automation Platform Controller\ndescription:\n    - Single-request bulk host creation in Automation Platform Controller.\n    - Provides a way to add many hosts at once to an inventory in Controller.\noptions:\n    hosts:\n      description:\n        - List of hosts to add to inventory.\n      required: True\n      type: list\n      elements: dict\n      suboptions:\n        name:\n          description:\n            - The name to use for the host.\n          type: str\n          required: True\n        description:\n          description:\n            - The description to use for the host.\n          type: str\n        enabled:\n          description:\n            - If the host should be enabled.\n          type: bool\n        variables:\n          description:\n            - Variables to use for the host.\n          type: dict\n        instance_id:\n          description:\n            - instance_id to use for the host.\n          type: str\n    inventory:\n      description:\n        - Inventory name, ID, or named URL the hosts should be made a member of.\n      required: True\n      type: str\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Bulk host create\n  bulk_host_create:\n    inventory: 1\n    hosts:\n      - name: foobar.org\n      - name: 127.0.0.1\n'
from ..module_utils.controller_api import ControllerAPIModule
import json

def main():
    if False:
        i = 10
        return i + 15
    argument_spec = dict(hosts=dict(required=True, type='list', elements='dict'), inventory=dict(required=True, type='str'))
    module = ControllerAPIModule(argument_spec=argument_spec)
    inv_name = module.params.get('inventory')
    hosts = module.params.get('hosts')
    for h in hosts:
        if 'variables' in h:
            h['variables'] = json.dumps(h['variables'])
    inv_id = module.resolve_name_to_id('inventories', inv_name)
    result = module.post_endpoint('bulk/host_create', data={'inventory': inv_id, 'hosts': hosts})
    if result['status_code'] != 201:
        module.fail_json(msg='Failed to create hosts, see response for details', response=result)
    module.json_output['changed'] = True
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()