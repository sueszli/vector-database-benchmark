from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: job_wait\nauthor: "Wayne Witzel III (@wwitzel3)"\nshort_description: Wait for Automation Platform Controller job to finish.\ndescription:\n    - Wait for Automation Platform Controller job to finish and report success or failure. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    job_id:\n      description:\n        - ID of the job to monitor.\n      required: True\n      type: int\n    interval:\n      description:\n        - The interval in sections, to request an update from the controller.\n        - For backwards compatibility if unset this will be set to the average of min and max intervals\n      required: False\n      default: 2\n      type: float\n    timeout:\n      description:\n        - Maximum time in seconds to wait for a job to finish.\n      type: int\n    job_type:\n      description:\n        - Job type to wait for\n      choices: [\'project_updates\', \'jobs\', \'inventory_updates\', \'workflow_jobs\']\n      default: \'jobs\'\n      type: str\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Launch a job\n  job_launch:\n    job_template: "My Job Template"\n  register: job\n\n- name: Wait for job max 120s\n  job_wait:\n    job_id: "{{ job.id }}"\n    timeout: 120\n'
RETURN = '\nid:\n    description: job id that is being waited on\n    returned: success\n    type: int\n    sample: 99\nelapsed:\n    description: total time in seconds the job took to run\n    returned: success\n    type: float\n    sample: 10.879\nstarted:\n    description: timestamp of when the job started running\n    returned: success\n    type: str\n    sample: "2017-03-01T17:03:53.200234Z"\nfinished:\n    description: timestamp of when the job finished running\n    returned: success\n    type: str\n    sample: "2017-03-01T17:04:04.078782Z"\nstatus:\n    description: current status of job\n    returned: success\n    type: str\n    sample: successful\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        while True:
            i = 10
    argument_spec = dict(job_id=dict(type='int', required=True), job_type=dict(choices=['project_updates', 'jobs', 'inventory_updates', 'workflow_jobs'], default='jobs'), timeout=dict(type='int'), interval=dict(type='float', default=2))
    module = ControllerAPIModule(argument_spec=argument_spec)
    job_id = module.params.get('job_id')
    job_type = module.params.get('job_type')
    timeout = module.params.get('timeout')
    interval = module.params.get('interval')
    job = module.get_one(job_type, **{'data': {'id': job_id}})
    if job is None:
        module.fail_json(msg='Unable to wait on ' + job_type.rstrip('s') + ' {0}; that ID does not exist.'.format(job_id))
    module.wait_on_url(url=job['url'], object_name=job_id, object_type='legacy_job_wait', timeout=timeout, interval=interval)
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()