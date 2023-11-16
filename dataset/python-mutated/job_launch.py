from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: job_launch\nauthor: "Wayne Witzel III (@wwitzel3)"\nshort_description: Launch an Ansible Job.\ndescription:\n    - Launch an Automation Platform Controller jobs. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    name:\n      description:\n        - Name of the job template to use.\n      required: True\n      type: str\n      aliases: [\'job_template\']\n    job_type:\n      description:\n        - Job_type to use for the job, only used if prompt for job_type is set.\n      choices: ["run", "check"]\n      type: str\n    inventory:\n      description:\n        - Inventory name, ID, or named URL to use for the job, only used if prompt for inventory is set.\n      type: str\n    organization:\n      description:\n        - Organization name, ID, or named URL the job template exists in.\n        - Used to help lookup the object, cannot be modified using this module.\n        - If not provided, will lookup by name only, which does not work with duplicates.\n      type: str\n    credentials:\n      description:\n        - Credential names, IDs, or named URLs to use for job, only used if prompt for credential is set.\n      type: list\n      aliases: [\'credential\']\n      elements: str\n    extra_vars:\n      description:\n        - extra_vars to use for the Job Template.\n        - ask_extra_vars needs to be set to True via job_template module\n          when creating the Job Template.\n      type: dict\n    limit:\n      description:\n        - Limit to use for the I(job_template).\n      type: str\n    tags:\n      description:\n        - Specific tags to use for from playbook.\n      type: list\n      elements: str\n    scm_branch:\n      description:\n        - A specific of the SCM project to run the template on.\n        - This is only applicable if your project allows for branch override.\n      type: str\n    skip_tags:\n      description:\n        - Specific tags to skip from the playbook.\n      type: list\n      elements: str\n    verbosity:\n      description:\n        - Verbosity level for this job run\n      type: int\n      choices: [ 0, 1, 2, 3, 4, 5 ]\n    diff_mode:\n      description:\n        - Show the changes made by Ansible tasks where supported\n      type: bool\n    credential_passwords:\n      description:\n        - Passwords for credentials which are set to prompt on launch\n      type: dict\n    execution_environment:\n      description:\n        - Execution environment name, ID, or named URL to use for the job, only used if prompt for execution environment is set.\n      type: str\n    forks:\n      description:\n        - Forks to use for the job, only used if prompt for forks is set.\n      type: int\n    instance_groups:\n      description:\n        - Instance group names, IDs, or named URLs to use for the job, only used if prompt for instance groups is set.\n      type: list\n      elements: str\n    job_slice_count:\n      description:\n        - Job slice count to use for the job, only used if prompt for job slice count is set.\n      type: int\n    labels:\n      description:\n        - Labels to use for the job, only used if prompt for labels is set.\n      type: list\n      elements: str\n    job_timeout:\n      description:\n        - Timeout to use for the job, only used if prompt for timeout is set.\n        - This parameter is sent through the API to the job.\n      type: int\n    wait:\n      description:\n        - Wait for the job to complete.\n      default: False\n      type: bool\n    interval:\n      description:\n        - The interval to request an update from the controller.\n      required: False\n      default: 2\n      type: float\n    timeout:\n      description:\n        - If waiting for the job to complete this will abort after this\n          amount of seconds. This happens on the module side.\n      type: int\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Launch a job\n  job_launch:\n    job_template: "My Job Template"\n  register: job\n\n- name: Launch a job template with extra_vars on remote controller instance\n  job_launch:\n    job_template: "My Job Template"\n    extra_vars:\n      var1: "My First Variable"\n      var2: "My Second Variable"\n      var3: "My Third Variable"\n    job_type: run\n\n- name: Launch a job with inventory and credential\n  job_launch:\n    job_template: "My Job Template"\n    inventory: "My Inventory"\n    credentials:\n      - "My Credential"\n      - "suplementary cred"\n  register: job\n- name: Wait for job max 120s\n  job_wait:\n    job_id: "{{ job.id }}"\n    timeout: 120\n'
RETURN = '\nid:\n    description: job id of the newly launched job\n    returned: success\n    type: int\n    sample: 86\nstatus:\n    description: status of newly launched job\n    returned: success\n    type: str\n    sample: pending\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        while True:
            i = 10
    argument_spec = dict(name=dict(required=True, aliases=['job_template']), job_type=dict(choices=['run', 'check']), inventory=dict(), organization=dict(), credentials=dict(type='list', aliases=['credential'], elements='str'), limit=dict(), tags=dict(type='list', elements='str'), extra_vars=dict(type='dict'), scm_branch=dict(), skip_tags=dict(type='list', elements='str'), verbosity=dict(type='int', choices=[0, 1, 2, 3, 4, 5]), diff_mode=dict(type='bool'), credential_passwords=dict(type='dict', no_log=False), execution_environment=dict(), forks=dict(type='int'), instance_groups=dict(type='list', elements='str'), job_slice_count=dict(type='int'), labels=dict(type='list', elements='str'), job_timeout=dict(type='int'), wait=dict(default=False, type='bool'), interval=dict(default=2.0, type='float'), timeout=dict(type='int'))
    module = ControllerAPIModule(argument_spec=argument_spec)
    optional_args = {}
    name = module.params.get('name')
    inventory = module.params.get('inventory')
    organization = module.params.get('organization')
    credentials = module.params.get('credentials')
    execution_environment = module.params.get('execution_environment')
    instance_groups = module.params.get('instance_groups')
    labels = module.params.get('labels')
    wait = module.params.get('wait')
    interval = module.params.get('interval')
    timeout = module.params.get('timeout')
    for field_name in ('job_type', 'limit', 'extra_vars', 'scm_branch', 'verbosity', 'diff_mode', 'credential_passwords', 'forks', 'job_slice_count', 'job_timeout'):
        field_val = module.params.get(field_name)
        if field_val is not None:
            optional_args[field_name] = field_val
        job_tags = module.params.get('tags')
        if job_tags is not None:
            optional_args['job_tags'] = ','.join(job_tags)
        skip_tags = module.params.get('skip_tags')
        if skip_tags is not None:
            optional_args['skip_tags'] = ','.join(skip_tags)
    job_timeout = module.params.get('job_timeout')
    if job_timeout is not None:
        optional_args['timeout'] = job_timeout
    post_data = {}
    for (arg_name, arg_value) in optional_args.items():
        if arg_value:
            post_data[arg_name] = arg_value
    if inventory:
        post_data['inventory'] = module.resolve_name_to_id('inventories', inventory)
    if execution_environment:
        post_data['execution_environment'] = module.resolve_name_to_id('execution_environments', execution_environment)
    if credentials:
        post_data['credentials'] = []
        for credential in credentials:
            post_data['credentials'].append(module.resolve_name_to_id('credentials', credential))
    if labels:
        post_data['labels'] = []
        for label in labels:
            post_data['labels'].append(module.resolve_name_to_id('labels', label))
    if instance_groups:
        post_data['instance_groups'] = []
        for instance_group in instance_groups:
            post_data['instance_groups'].append(module.resolve_name_to_id('instance_groups', instance_group))
    lookup_data = {}
    if organization:
        lookup_data['organization'] = module.resolve_name_to_id('organizations', organization)
    job_template = module.get_one('job_templates', name_or_id=name, data=lookup_data)
    if job_template is None:
        module.fail_json(msg='Unable to find job template by name {0}'.format(name))
    check_vars_to_prompts = {'scm_branch': 'ask_scm_branch_on_launch', 'diff_mode': 'ask_diff_mode_on_launch', 'limit': 'ask_limit_on_launch', 'tags': 'ask_tags_on_launch', 'skip_tags': 'ask_skip_tags_on_launch', 'job_type': 'ask_job_type_on_launch', 'verbosity': 'ask_verbosity_on_launch', 'inventory': 'ask_inventory_on_launch', 'credentials': 'ask_credential_on_launch'}
    param_errors = []
    for (variable_name, prompt) in check_vars_to_prompts.items():
        if module.params.get(variable_name) and (not job_template[prompt]):
            param_errors.append('The field {0} was specified but the job template does not allow for it to be overridden'.format(variable_name))
    if module.params.get('extra_vars') and (not (job_template['ask_variables_on_launch'] or job_template['survey_enabled'])):
        param_errors.append('The field extra_vars was specified but the job template does not allow for it to be overridden')
    if len(param_errors) > 0:
        module.fail_json(msg='Parameters specified which can not be passed into job template, see errors for details', **{'errors': param_errors})
    results = module.post_endpoint(job_template['related']['launch'], **{'data': post_data})
    if results['status_code'] != 201:
        module.fail_json(msg='Failed to launch job, see response for details', **{'response': results})
    if not wait:
        module.exit_json(**{'changed': True, 'id': results['json']['id'], 'status': results['json']['status']})
    results = module.wait_on_url(url=results['json']['url'], object_name=name, object_type='Job', timeout=timeout, interval=interval)
    module.exit_json(**{'changed': True, 'id': results['json']['id'], 'status': results['json']['status']})
if __name__ == '__main__':
    main()