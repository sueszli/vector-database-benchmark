from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: ad_hoc_command_cancel\nauthor: "John Westcott IV (@john-westcott-iv)"\nshort_description: Cancel an Ad Hoc Command.\ndescription:\n    - Cancel ad hoc command. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    command_id:\n      description:\n        - ID of the command to cancel\n      required: True\n      type: int\n    fail_if_not_running:\n      description:\n        - Fail loudly if the I(command_id) can not be canceled\n      default: False\n      type: bool\n    interval:\n      description:\n        - The interval in seconds, to request an update from .\n      required: False\n      default: 1\n      type: float\n    timeout:\n      description:\n        - Maximum time in seconds to wait for a job to finish.\n        - Not specifying means the task will wait until the controller cancels the command.\n      type: int\n      default: 0\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Cancel command\n  ad_hoc_command_cancel:\n    command_id: command.id\n'
RETURN = '\nid:\n    description: command id requesting to cancel\n    returned: success\n    type: int\n    sample: 94\n'
import time
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        i = 10
        return i + 15
    argument_spec = dict(command_id=dict(type='int', required=True), fail_if_not_running=dict(type='bool', default=False), interval=dict(type='float', default=1.0), timeout=dict(type='int', default=0))
    module = ControllerAPIModule(argument_spec=argument_spec)
    command_id = module.params.get('command_id')
    fail_if_not_running = module.params.get('fail_if_not_running')
    interval = module.params.get('interval')
    timeout = module.params.get('timeout')
    command = module.get_one('ad_hoc_commands', **{'data': {'id': command_id}})
    if command is None:
        module.fail_json(msg='Unable to find command with id {0}'.format(command_id))
    cancel_page = module.get_endpoint(command['related']['cancel'])
    if 'json' not in cancel_page or 'can_cancel' not in cancel_page['json']:
        module.fail_json(msg='Failed to cancel command, got unexpected response', **{'response': cancel_page})
    if not cancel_page['json']['can_cancel']:
        if fail_if_not_running:
            module.fail_json(msg='Ad Hoc Command is not running')
        else:
            module.exit_json(**{'changed': False})
    results = module.post_endpoint(command['related']['cancel'], **{'data': {}})
    if results['status_code'] != 202:
        module.fail_json(msg='Failed to cancel command, see response for details', **{'response': results})
    result = module.get_endpoint(command['related']['cancel'])
    start = time.time()
    while result['json']['can_cancel']:
        if timeout and timeout < time.time() - start:
            module.json_output['msg'] = 'Monitoring of ad hoc command aborted due to timeout'
            module.fail_json(**module.json_output)
        time.sleep(interval)
        result = module.get_endpoint(command['related']['cancel'])
    module.exit_json(**{'changed': True})
if __name__ == '__main__':
    main()