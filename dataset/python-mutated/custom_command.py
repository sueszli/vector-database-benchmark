from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: custom_command\nshort_description: Adds or removes a user with custom commands by ssh\ndescription:\n    - You can add or edit users using ssh with custom commands.\n\noptions:\n  protocol:\n    default: ssh\n    choices: [ssh]\n    description:\n      - C(ssh) The remote asset is connected using ssh.\n    type: str\n  name:\n    description:\n      - The name of the user to add or remove.\n    required: true\n    aliases: [user]\n    type: str\n  password:\n    description:\n      - The password to use for the user.\n    type: str\n    aliases: [pass]\n  commands:\n    description:\n      - Custom change password commands.\n    type: list\n    required: true\n  first_conn_delay_time:\n    description:\n      - Delay for executing the command after SSH connection(unit: s)\n    type: float\n    required: false\n'
EXAMPLES = '\n- name: Create user with name \'jms\' and password \'123456\'.\n  custom_command:\n    login_host: "localhost"\n    login_port: 22\n    login_user: "admin"\n    login_password: "123456"\n    name: "jms"\n    password: "123456"\n    commands: [\'passwd {username}\', \'{password}\', \'{password}\']\n'
RETURN = '\nname:\n    description: The name of the user to add.\n    returned: success\n    type: str\n'
from ansible.module_utils.basic import AnsibleModule
from ops.ansible.modules_utils.custom_common import SSHClient, common_argument_spec

def get_commands(module):
    if False:
        for i in range(10):
            print('nop')
    username = module.params['name']
    password = module.params['password']
    commands = module.params['commands'] or []
    login_password = module.params['login_password']
    for (index, command) in enumerate(commands):
        commands[index] = command.format(username=username, password=password, login_password=login_password)
    return commands

def main():
    if False:
        while True:
            i = 10
    argument_spec = common_argument_spec()
    argument_spec.update(name=dict(required=True, aliases=['user']), password=dict(aliases=['pass'], no_log=True), commands=dict(type='list', required=False))
    module = AnsibleModule(argument_spec=argument_spec)
    ssh_client = SSHClient(module)
    commands = get_commands(module)
    if not commands:
        module.fail_json(msg='No command found, please go to the platform details to add')
    (output, err_msg) = ssh_client.execute(commands)
    if err_msg:
        module.fail_json(msg='There was a problem executing the command: %s' % err_msg)
    user = module.params['name']
    module.exit_json(changed=True, user=user)
if __name__ == '__main__':
    main()