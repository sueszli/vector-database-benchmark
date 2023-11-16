from __future__ import annotations
DOCUMENTATION = '\nmodule: module2\nshort_description: Hello test module\ndescription: Hello test module.\noptions: {}\nauthor:\n  - Ansible Core Team\n'
EXAMPLES = '\n- minimal:\n'
RETURN = ''
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec={})
    module.exit_json()
if __name__ == '__main__':
    main()