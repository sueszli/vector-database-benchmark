from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: inventory_source_update\nauthor: "Bianca Henderson (@beeankha)"\nshort_description: Update inventory source(s).\ndescription:\n    - Update Automation Platform Controller inventory source(s). See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    name:\n      description:\n        - The name or id of the inventory source to update.\n      required: True\n      type: str\n      aliases:\n        - inventory_source\n    inventory:\n      description:\n        - Name or id of the inventory that contains the inventory source(s) to update.\n      required: True\n      type: str\n    organization:\n      description:\n        - Name, ID, or named URL of the inventory source\'s inventory\'s organization.\n      type: str\n    wait:\n      description:\n        - Wait for the job to complete.\n      default: False\n      type: bool\n    interval:\n      description:\n        - The interval to request an update from the controller.\n      required: False\n      default: 2\n      type: float\n    timeout:\n      description:\n        - If waiting for the job to complete this will abort after this\n          amount of seconds\n      type: int\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Update a single inventory source\n  inventory_source_update:\n    name: "Example Inventory Source"\n    inventory: "My Inventory"\n    organization: Default\n\n- name: Update all inventory sources\n  inventory_source_update:\n    name: "{{ item }}"\n    inventory: "My Other Inventory"\n  loop: "{{ query(\'awx.awx.controller_api\', \'inventory_sources\', query_params={ \'inventory\': 30 }, return_ids=True ) }}"\n'
RETURN = '\nid:\n    description: id of the inventory update\n    returned: success\n    type: int\n    sample: 86\nstatus:\n    description: status of the inventory update\n    returned: success\n    type: str\n    sample: pending\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        i = 10
        return i + 15
    argument_spec = dict(name=dict(required=True, aliases=['inventory_source']), inventory=dict(required=True), organization=dict(), wait=dict(default=False, type='bool'), interval=dict(default=2.0, type='float'), timeout=dict(type='int'))
    module = ControllerAPIModule(argument_spec=argument_spec)
    name = module.params.get('name')
    inventory = module.params.get('inventory')
    organization = module.params.get('organization')
    wait = module.params.get('wait')
    interval = module.params.get('interval')
    timeout = module.params.get('timeout')
    lookup_data = {}
    if organization:
        lookup_data['organization'] = module.resolve_name_to_id('organizations', organization)
    inventory_object = module.get_one('inventories', name_or_id=inventory, data=lookup_data)
    if not inventory_object:
        module.fail_json(msg='The specified inventory, {0}, was not found.'.format(lookup_data))
    inventory_source_object = module.get_one('inventory_sources', name_or_id=name, data={'inventory': inventory_object['id']})
    if not inventory_source_object:
        module.fail_json(msg='The specified inventory source was not found.')
    inventory_source_update_results = module.post_endpoint(inventory_source_object['related']['update'])
    if inventory_source_update_results['status_code'] != 202:
        module.fail_json(msg='Failed to update inventory source, see response for details', response=inventory_source_update_results)
    module.json_output['changed'] = True
    module.json_output['id'] = inventory_source_update_results['json']['id']
    module.json_output['status'] = inventory_source_update_results['json']['status']
    if not wait:
        module.exit_json(**module.json_output)
    module.wait_on_url(url=inventory_source_update_results['json']['url'], object_name=inventory_object, object_type='inventory_update', timeout=timeout, interval=interval)
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()