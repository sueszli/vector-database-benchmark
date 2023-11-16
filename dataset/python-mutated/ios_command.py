from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'network'}
DOCUMENTATION = 'module: ios_command\nauthor: Peter Sprygada (@privateip)\nshort_description: Run commands on remote devices running Cisco IOS\ndescription:\n- Sends arbitrary commands to an ios node and returns the results read from the device.\n  This module includes an argument that will cause the module to wait for a specific\n  condition before returning or timing out if the condition is not met.\n- This module does not support running commands in configuration mode. Please use\n  M(ios_config) to configure IOS devices.\nextends_documentation_fragment:\n- cisco.ios.ios\nnotes:\n- Tested against IOS 15.6\noptions:\n  commands:\n    description:\n    - List of commands to send to the remote ios device over the configured provider.\n      The resulting output from the command is returned. If the I(wait_for) argument\n      is provided, the module is not returned until the condition is satisfied or\n      the number of retries has expired. If a command sent to the device requires\n      answering a prompt, it is possible to pass a dict containing I(command), I(answer)\n      and I(prompt). Common answers are \'y\' or "\r" (carriage return, must be double\n      quotes). See examples.\n    required: true\n  wait_for:\n    description:\n    - List of conditions to evaluate against the output of the command. The task will\n      wait for each condition to be true before moving forward. If the conditional\n      is not true within the configured number of retries, the task fails. See examples.\n    aliases:\n    - waitfor\n  match:\n    description:\n    - The I(match) argument is used in conjunction with the I(wait_for) argument to\n      specify the match policy.  Valid values are C(all) or C(any).  If the value\n      is set to C(all) then all conditionals in the wait_for must be satisfied.  If\n      the value is set to C(any) then only one of the values must be satisfied.\n    default: all\n    choices:\n    - any\n    - all\n  retries:\n    description:\n    - Specifies the number of retries a command should by tried before it is considered\n      failed. The command is run on the target device every retry and evaluated against\n      the I(wait_for) conditions.\n    default: 10\n  interval:\n    description:\n    - Configures the interval in seconds to wait between retries of the command. If\n      the command does not pass the specified conditions, the interval indicates how\n      long to wait before trying the command again.\n    default: 1\n'
EXAMPLES = '\ntasks:\n  - name: run show version on remote devices\n    ios_command:\n      commands: show version\n\n  - name: run show version and check to see if output contains IOS\n    ios_command:\n      commands: show version\n      wait_for: result[0] contains IOS\n\n  - name: run multiple commands on remote nodes\n    ios_command:\n      commands:\n        - show version\n        - show interfaces\n\n  - name: run multiple commands and evaluate the output\n    ios_command:\n      commands:\n        - show version\n        - show interfaces\n      wait_for:\n        - result[0] contains IOS\n        - result[1] contains Loopback0\n\n  - name: run commands that require answering a prompt\n    ios_command:\n      commands:\n        - command: \'clear counters GigabitEthernet0/1\'\n          prompt: \'Clear "show interface" counters on this interface \\[confirm\\]\'\n          answer: \'y\'\n        - command: \'clear counters GigabitEthernet0/2\'\n          prompt: \'[confirm]\'\n          answer: "\\r"\n'
RETURN = "\nstdout:\n  description: The set of responses from the commands\n  returned: always apart from low level errors (such as action plugin)\n  type: list\n  sample: ['...', '...']\nstdout_lines:\n  description: The value of stdout split into a list\n  returned: always apart from low level errors (such as action plugin)\n  type: list\n  sample: [['...', '...'], ['...'], ['...']]\nfailed_conditions:\n  description: The list of conditionals that have failed\n  returned: failed\n  type: list\n  sample: ['...', '...']\n"
import time
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Conditional
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import transform_commands, to_lines
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import run_commands
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import ios_argument_spec

def parse_commands(module, warnings):
    if False:
        i = 10
        return i + 15
    commands = transform_commands(module)
    if module.check_mode:
        for item in list(commands):
            if not item['command'].startswith('show'):
                warnings.append('Only show commands are supported when using check mode, not executing %s' % item['command'])
                commands.remove(item)
    return commands

def main():
    if False:
        while True:
            i = 10
    'main entry point for module execution\n    '
    argument_spec = dict(commands=dict(type='list', required=True), wait_for=dict(type='list', aliases=['waitfor']), match=dict(default='all', choices=['all', 'any']), retries=dict(default=10, type='int'), interval=dict(default=1, type='int'))
    argument_spec.update(ios_argument_spec)
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
    warnings = list()
    result = {'changed': False, 'warnings': warnings}
    commands = parse_commands(module, warnings)
    wait_for = module.params['wait_for'] or list()
    try:
        conditionals = [Conditional(c) for c in wait_for]
    except AttributeError as exc:
        module.fail_json(msg=to_text(exc))
    retries = module.params['retries']
    interval = module.params['interval']
    match = module.params['match']
    while retries > 0:
        responses = run_commands(module, commands)
        for item in list(conditionals):
            if item(responses):
                if match == 'any':
                    conditionals = list()
                    break
                conditionals.remove(item)
        if not conditionals:
            break
        time.sleep(interval)
        retries -= 1
    if conditionals:
        failed_conditions = [item.raw for item in conditionals]
        msg = 'One or more conditional statements have not been satisfied'
        module.fail_json(msg=msg, failed_conditions=failed_conditions)
    result.update({'stdout': responses, 'stdout_lines': list(to_lines(responses))})
    module.exit_json(**result)
if __name__ == '__main__':
    main()