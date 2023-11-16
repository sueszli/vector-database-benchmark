from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: test_docs_returns\nshort_description: Test module\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\n'
EXAMPLES = '\n'
RETURN = '\nz_last:\n    description: A last result.\n    type: str\n    returned: success\n\nm_middle:\n    description:\n        - This should be in the middle.\n        - Has some more data\n    type: dict\n    returned: success and 1st of month\n    contains:\n        suboption:\n            description: A suboption.\n            type: str\n            choices: [ARF, BARN, c_without_capital_first_letter]\n\na_first:\n    description: A first result.\n    type: str\n    returned: success\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict())
    module.exit_json()
if __name__ == '__main__':
    main()