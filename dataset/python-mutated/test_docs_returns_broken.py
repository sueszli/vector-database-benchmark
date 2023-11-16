from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: test_docs_returns_broken\nshort_description: Test module\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\n'
EXAMPLES = '\n'
RETURN = '\ntest:\n    description: A test return value.\n   type: str\n\nbroken_key: [\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict())
    module.exit_json()
if __name__ == '__main__':
    main()