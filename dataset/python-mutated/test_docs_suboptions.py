from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: test_docs_suboptions\nshort_description: Test module\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\noptions:\n    with_suboptions:\n        description:\n            - An option with suboptions.\n            - Use with care.\n        type: dict\n        suboptions:\n            z_last:\n                description: The last suboption.\n                type: str\n            m_middle:\n                description:\n                    - The suboption in the middle.\n                    - Has its own suboptions.\n                suboptions:\n                    a_suboption:\n                        description: A sub-suboption.\n                        type: str\n            a_first:\n                description: The first suboption.\n                type: str\n'
EXAMPLES = '\n'
RETURN = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(test_docs_suboptions=dict(type='dict', options=dict(a_first=dict(type='str'), m_middle=dict(type='dict', options=dict(a_suboption=dict(type='str'))), z_last=dict(type='str')))))
    module.exit_json()
if __name__ == '__main__':
    main()