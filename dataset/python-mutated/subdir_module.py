from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: subdir_module\nshort_description: A module in multiple subdirectories\ndescription:\n    - A module in multiple subdirectories\nauthor:\n    - Ansible Core Team\nversion_added: 1.0.0\noptions: {}\n'
EXAMPLES = '\n'
RETURN = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict())
    module.exit_json()
if __name__ == '__main__':
    main()