from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: test_docs_no_metadata\nshort_description: Test module\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\n'
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