from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = '\n---\nmodule: custom_rdp_ping\nshort_description: Use rdp to probe whether an asset is connectable\ndescription:\n    - Use rdp to probe whether an asset is connectable\n'
EXAMPLES = '\n- name: >\n    Ping asset server.\n  custom_rdp_ping:\n    login_host: 127.0.0.1\n    login_port: 3389\n    login_user: jms\n    login_password: password\n'
RETURN = "\nis_available:\n  description: Windows server availability.\n  returned: always\n  type: bool\n  sample: true\nconn_err_msg:\n  description: Connection error message.\n  returned: always\n  type: str\n  sample: ''\n"
import pyfreerdp
from typing import NamedTuple
from ansible.module_utils.basic import AnsibleModule

class Param(NamedTuple):
    hostname: str
    port: int
    username: str
    password: str

def common_argument_spec():
    if False:
        while True:
            i = 10
    options = dict(login_host=dict(type='str', required=False, default='localhost'), login_port=dict(type='int', required=False, default=22), login_user=dict(type='str', required=False, default='root'), login_password=dict(type='str', required=False, no_log=True), login_secret_type=dict(type='str', required=False, default='password'), login_private_key_path=dict(type='str', required=False, no_log=True))
    return options

def main():
    if False:
        print('Hello World!')
    options = common_argument_spec()
    module = AnsibleModule(argument_spec=options, supports_check_mode=True)
    result = {'changed': False, 'is_available': False}
    secret_type = module.params['login_secret_type']
    if secret_type != 'password':
        module.fail_json(msg=f'The current ansible does not support                 the verification method for {secret_type} types.')
        return module.exit_json(**result)
    params = Param(hostname=module.params['login_host'], port=module.params['login_port'], username=module.params['login_user'], password=module.params['login_password'])
    is_available = pyfreerdp.check_connectivity(*params, '', 0)
    result['is_available'] = is_available
    if not is_available:
        module.fail_json(msg='Unable to connect to asset.')
    return module.exit_json(**result)
if __name__ == '__main__':
    main()