from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: custom_ssh_ping\nshort_description: Use ssh to probe whether an asset is connectable\ndescription:\n    - Use ssh to probe whether an asset is connectable\n'
EXAMPLES = '\n- name: >\n    Ping asset server.\n  custom_ssh_ping:\n    login_host: 127.0.0.1\n    login_port: 22\n    login_user: jms\n    login_password: password\n'
RETURN = "\nis_available:\n  description: MongoDB server availability.\n  returned: always\n  type: bool\n  sample: true\nconn_err_msg:\n  description: Connection error message.\n  returned: always\n  type: str\n  sample: ''\n"
from ansible.module_utils.basic import AnsibleModule
from ops.ansible.modules_utils.custom_common import SSHClient, common_argument_spec

def main():
    if False:
        return 10
    options = common_argument_spec()
    module = AnsibleModule(argument_spec=options, supports_check_mode=True)
    result = {'changed': False, 'is_available': True}
    client = SSHClient(module)
    err = client.connect()
    if err:
        module.fail_json(msg='Unable to connect to asset: %s' % err)
        result['is_available'] = False
    return module.exit_json(**result)
if __name__ == '__main__':
    main()