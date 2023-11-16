from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: ping\nversion_added: historical\nshort_description: Try to connect to host, verify a usable python and return C(pong) on success\ndescription:\n  - A trivial test module, this module always returns C(pong) on successful\n    contact. It does not make sense in playbooks, but it is useful from\n    C(/usr/bin/ansible) to verify the ability to login and that a usable Python is configured.\n  - This is NOT ICMP ping, this is just a trivial test module that requires Python on the remote-node.\n  - For Windows targets, use the M(ansible.windows.win_ping) module instead.\n  - For Network targets, use the M(ansible.netcommon.net_ping) module instead.\noptions:\n  data:\n    description:\n      - Data to return for the C(ping) return value.\n      - If this parameter is set to C(crash), the module will cause an exception.\n    type: str\n    default: pong\nseealso:\n  - module: ansible.netcommon.net_ping\n  - module: ansible.windows.win_ping\nauthor:\n  - Ansible Core Team\n  - Michael DeHaan\nnotes:\n  - Supports C(check_mode).\n'
EXAMPLES = "\n# Test we can logon to 'webservers' and execute python with json lib.\n# ansible webservers -m ping\n\n- name: Example from an Ansible Playbook\n  ansible.builtin.ping:\n\n- name: Induce an exception to see what happens\n  ansible.builtin.ping:\n    data: crash\n"
RETURN = '\nping:\n    description: Value provided with the data parameter.\n    returned: success\n    type: str\n    sample: pong\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(data=dict(type='str', default='pong')), supports_check_mode=True)
    if module.params['data'] == 'crash':
        raise Exception('boom')
    result = dict(ping=module.params['data'])
    module.exit_json(**result)
if __name__ == '__main__':
    main()