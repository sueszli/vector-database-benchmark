from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: ad_hoc_command_wait\nauthor: "John Westcott IV (@john-westcott-iv)"\nshort_description: Wait for Automation Platform Controller Ad Hoc Command to finish.\ndescription:\n    - Wait for Automation Platform Controller ad hoc command to finish and report success or failure. See\n      U(https://www.ansible.com/tower) for an overview.\noptions:\n    command_id:\n      description:\n        - ID of the ad hoc command to monitor.\n      required: True\n      type: int\n    interval:\n      description:\n        - The interval in sections, to request an update from the controller.\n      required: False\n      default: 2\n      type: float\n    timeout:\n      description:\n        - Maximum time in seconds to wait for a ad hoc command to finish.\n      type: int\nextends_documentation_fragment: awx.awx.auth\n'
EXAMPLES = '\n- name: Launch an ad hoc command\n  ad_hoc_command:\n    inventory: "Demo Inventory"\n    credential: "Demo Credential"\n    wait: false\n  register: command\n\n- name: Wait for ad joc command max 120s\n  ad_hoc_command_wait:\n    command_id: "{{ command.id }}"\n    timeout: 120\n'
RETURN = '\nid:\n    description: Ad hoc command id that is being waited on\n    returned: success\n    type: int\n    sample: 99\nelapsed:\n    description: total time in seconds the command took to run\n    returned: success\n    type: float\n    sample: 10.879\nstarted:\n    description: timestamp of when the command started running\n    returned: success\n    type: str\n    sample: "2017-03-01T17:03:53.200234Z"\nfinished:\n    description: timestamp of when the command finished running\n    returned: success\n    type: str\n    sample: "2017-03-01T17:04:04.078782Z"\nstatus:\n    description: current status of command\n    returned: success\n    type: str\n    sample: successful\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        for i in range(10):
            print('nop')
    argument_spec = dict(command_id=dict(type='int', required=True), timeout=dict(type='int'), interval=dict(type='float', default=2))
    module = ControllerAPIModule(argument_spec=argument_spec)
    command_id = module.params.get('command_id')
    timeout = module.params.get('timeout')
    interval = module.params.get('interval')
    command = module.get_one('ad_hoc_commands', **{'data': {'id': command_id}})
    if command is None:
        module.fail_json(msg='Unable to wait on ad hoc command {0}; that ID does not exist.'.format(command_id))
    module.wait_on_url(url=command['url'], object_name=command_id, object_type='ad hoc command', timeout=timeout, interval=interval)
    module.exit_json(**module.json_output)
if __name__ == '__main__':
    main()