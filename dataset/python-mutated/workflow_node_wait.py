from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: workflow_node_wait\nauthor: "Sean Sullivan (@sean-m-sullivan)"\nshort_description: Wait for a workflow node to finish.\ndescription:\n    - Approve an approval node in a workflow job. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    workflow_job_id:\n      description:\n        - ID of the workflow job to monitor for node.\n      required: True\n      type: int\n    name:\n      description:\n        - Name of the workflow node to wait on.\n      required: True\n      type: str\n    interval:\n      description:\n        - The interval in sections, to request an update from the controller.\n      required: False\n      default: 1\n      type: float\n    timeout:\n      description:\n        - Maximum time in seconds to wait for a workflow job to reach approval node.\n      default: 10\n      type: int\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Launch a workflow with a timeout of 10 seconds\n  workflow_launch:\n    workflow_template: "Test Workflow"\n    wait: False\n  register: workflow\n\n- name: Wait for a workflow node to finish\n  workflow_node_wait:\n    workflow_job_id: "{{ workflow.id }}"\n    name: Approval Data Step\n    timeout: 120\n'
RETURN = '\n\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        for i in range(10):
            print('nop')
    argument_spec = dict(workflow_job_id=dict(type='int', required=True), name=dict(required=True), timeout=dict(type='int', default=10), interval=dict(type='float', default=1))
    module = ControllerAPIModule(argument_spec=argument_spec)
    workflow_job_id = module.params.get('workflow_job_id')
    name = module.params.get('name')
    timeout = module.params.get('timeout')
    interval = module.params.get('interval')
    module.wait_on_workflow_node_url(url='workflow_jobs/{0}/workflow_nodes/'.format(workflow_job_id), object_name=name, object_type='Workflow Node', timeout=timeout, interval=interval, **{'data': {'job__name': name}})
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()