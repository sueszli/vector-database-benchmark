from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: workflow_launch\nauthor: "John Westcott IV (@john-westcott-iv)"\nshort_description: Run a workflow in Automation Platform Controller\ndescription:\n    - Launch an Automation Platform Controller workflows. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    name:\n      description:\n        - The name of the workflow template to run.\n      required: True\n      type: str\n      aliases:\n        - workflow_template\n    organization:\n      description:\n        - Organization name, ID, or named URL the workflow job template exists in.\n        - Used to help lookup the object, cannot be modified using this module.\n        - If not provided, will lookup by name only, which does not work with duplicates.\n      type: str\n    inventory:\n      description:\n        - Inventory name, ID, or named URL to use for the job ran with this workflow, only used if prompt for inventory is set.\n      type: str\n    limit:\n      description:\n        - Limit to use for the I(job_template).\n      type: str\n    scm_branch:\n      description:\n        - A specific branch of the SCM project to run the template on.\n        - This is only applicable if your project allows for branch override.\n      type: str\n    extra_vars:\n      description:\n        - Any extra vars required to launch the job.\n      type: dict\n    wait:\n      description:\n        - Wait for the workflow to complete.\n      default: True\n      type: bool\n    interval:\n      description:\n        - The interval to request an update from the controller.\n      required: False\n      default: 2\n      type: float\n    timeout:\n      description:\n        - If waiting for the workflow to complete this will abort after this\n          amount of seconds\n      type: int\nextends_documentation_fragment: awx.awx.auth\n'
RETURN = '\njob_info:\n    description: dictionary containing information about the workflow executed\n    returned: If workflow launched\n    type: dict\n'
EXAMPLES = '\n- name: Launch a workflow with a timeout of 10 seconds\n  workflow_launch:\n    workflow_template: "Test Workflow"\n    timeout: 10\n\n- name: Launch a Workflow with extra_vars without waiting\n  workflow_launch:\n    workflow_template: "Test workflow"\n    extra_vars:\n      var1: My First Variable\n      var2: My Second Variable\n    wait: False\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        for i in range(10):
            print('nop')
    argument_spec = dict(name=dict(required=True, aliases=['workflow_template']), organization=dict(), inventory=dict(), limit=dict(), scm_branch=dict(), extra_vars=dict(type='dict'), wait=dict(required=False, default=True, type='bool'), interval=dict(required=False, default=2.0, type='float'), timeout=dict(required=False, type='int'))
    module = ControllerAPIModule(argument_spec=argument_spec)
    optional_args = {}
    name = module.params.get('name')
    organization = module.params.get('organization')
    inventory = module.params.get('inventory')
    wait = module.params.get('wait')
    interval = module.params.get('interval')
    timeout = module.params.get('timeout')
    for field_name in ('limit', 'extra_vars', 'scm_branch'):
        field_val = module.params.get(field_name)
        if field_val is not None:
            optional_args[field_name] = field_val
    post_data = {}
    for (arg_name, arg_value) in optional_args.items():
        if arg_value:
            post_data[arg_name] = arg_value
    if inventory:
        post_data['inventory'] = module.resolve_name_to_id('inventories', inventory)
    lookup_data = {}
    if organization:
        lookup_data['organization'] = module.resolve_name_to_id('organizations', organization)
    workflow_job_template = module.get_one('workflow_job_templates', name_or_id=name, data=lookup_data)
    if workflow_job_template is None:
        module.fail_json(msg='Unable to find workflow job template')
    check_vars_to_prompts = {'inventory': 'ask_inventory_on_launch', 'limit': 'ask_limit_on_launch', 'scm_branch': 'ask_scm_branch_on_launch'}
    param_errors = []
    for (variable_name, prompt) in check_vars_to_prompts.items():
        if variable_name in post_data and (not workflow_job_template[prompt]):
            param_errors.append('The field {0} was specified but the workflow job template does not allow for it to be overridden'.format(variable_name))
    if module.params.get('extra_vars') and (not (workflow_job_template['ask_variables_on_launch'] or workflow_job_template['survey_enabled'])):
        param_errors.append('The field extra_vars was specified but the workflow job template does not allow for it to be overridden')
    if len(param_errors) > 0:
        module.fail_json(msg='Parameters specified which can not be passed into workflow job template, see errors for details', errors=param_errors)
    result = module.post_endpoint(workflow_job_template['related']['launch'], data=post_data)
    if result['status_code'] != 201:
        module.fail_json(msg='Failed to launch workflow, see response for details', response=result)
    module.json_output['changed'] = True
    module.json_output['id'] = result['json']['id']
    module.json_output['status'] = result['json']['status']
    module.json_output['job_info'] = {'id': result['json']['id']}
    if not wait:
        module.exit_json(**module.json_output)
    module.wait_on_url(url=result['json']['url'], object_name=name, object_type='Workflow Job', timeout=timeout, interval=interval)
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()