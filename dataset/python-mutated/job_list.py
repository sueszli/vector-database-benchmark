from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: job_list\nauthor: "Wayne Witzel III (@wwitzel3)"\nshort_description: List Automation Platform Controller jobs.\ndescription:\n    - List Automation Platform Controller jobs. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    status:\n      description:\n        - Only list jobs with this status.\n      choices: [\'pending\', \'waiting\', \'running\', \'error\', \'failed\', \'canceled\', \'successful\']\n      type: str\n    page:\n      description:\n        - Page number of the results to fetch.\n      type: int\n    all_pages:\n      description:\n        - Fetch all the pages and return a single result.\n      type: bool\n      default: \'no\'\n    query:\n      description:\n        - Query used to further filter the list of jobs. C({"foo":"bar"}) will be passed at C(?foo=bar)\n      type: dict\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: List running jobs for the testing.yml playbook\n  job_list:\n    status: running\n    query: {"playbook": "testing.yml"}\n    controller_config_file: "~/tower_cli.cfg"\n  register: testing_jobs\n'
RETURN = '\ncount:\n    description: Total count of objects return\n    returned: success\n    type: int\n    sample: 51\nnext:\n    description: next page available for the listing\n    returned: success\n    type: int\n    sample: 3\nprevious:\n    description: previous page available for the listing\n    returned: success\n    type: int\n    sample: 1\nresults:\n    description: a list of job objects represented as dictionaries\n    returned: success\n    type: list\n    sample: [{"allow_simultaneous": false, "artifacts": {}, "ask_credential_on_launch": false,\n              "ask_inventory_on_launch": false, "ask_job_type_on_launch": false, "failed": false,\n              "finished": "2017-02-22T15:09:05.633942Z", "force_handlers": false, "forks": 0, "id": 2,\n              "inventory": 1, "job_explanation": "", "job_tags": "", "job_template": 5, "job_type": "run"}, ...]\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        return 10
    argument_spec = dict(status=dict(choices=['pending', 'waiting', 'running', 'error', 'failed', 'canceled', 'successful']), page=dict(type='int'), all_pages=dict(type='bool', default=False), query=dict(type='dict'))
    module = ControllerAPIModule(argument_spec=argument_spec, mutually_exclusive=[('page', 'all_pages')])
    query = module.params.get('query')
    status = module.params.get('status')
    page = module.params.get('page')
    all_pages = module.params.get('all_pages')
    job_search_data = {}
    if page:
        job_search_data['page'] = page
    if status:
        job_search_data['status'] = status
    if query:
        job_search_data.update(query)
    if all_pages:
        job_list = module.get_all_endpoint('jobs', **{'data': job_search_data})
    else:
        job_list = module.get_endpoint('jobs', **{'data': job_search_data})
    module.exit_json(**job_list['json'])
if __name__ == '__main__':
    main()