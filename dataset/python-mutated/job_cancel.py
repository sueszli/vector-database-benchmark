from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: job_cancel\nauthor: "Wayne Witzel III (@wwitzel3)"\nshort_description: Cancel an Automation Platform Controller Job.\ndescription:\n    - Cancel Automation Platform Controller jobs. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    job_id:\n      description:\n        - ID of the job to cancel\n      required: True\n      type: int\n    fail_if_not_running:\n      description:\n        - Fail loudly if the I(job_id) can not be canceled\n      default: False\n      type: bool\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Cancel job\n  job_cancel:\n    job_id: job.id\n'
RETURN = '\nid:\n    description: job id requesting to cancel\n    returned: success\n    type: int\n    sample: 94\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        for i in range(10):
            print('nop')
    argument_spec = dict(job_id=dict(type='int', required=True), fail_if_not_running=dict(type='bool', default=False))
    module = ControllerAPIModule(argument_spec=argument_spec)
    job_id = module.params.get('job_id')
    fail_if_not_running = module.params.get('fail_if_not_running')
    job = module.get_one('jobs', **{'data': {'id': job_id}})
    if job is None:
        module.fail_json(msg='Unable to find job with id {0}'.format(job_id))
    cancel_page = module.get_endpoint(job['related']['cancel'])
    if 'json' not in cancel_page or 'can_cancel' not in cancel_page['json']:
        module.fail_json(msg='Failed to cancel job, got unexpected response from the controller', **{'response': cancel_page})
    if not cancel_page['json']['can_cancel']:
        if fail_if_not_running:
            module.fail_json(msg='Job is not running')
        else:
            module.exit_json(**{'changed': False})
    results = module.post_endpoint(job['related']['cancel'], **{'data': {}})
    if results['status_code'] != 202:
        module.fail_json(msg='Failed to cancel job, see response for details', **{'response': results})
    module.exit_json(**{'changed': True})
if __name__ == '__main__':
    main()