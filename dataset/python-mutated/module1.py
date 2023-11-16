from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['stableinterface'], 'supported_by': 'core'}
DOCUMENTATION = '\n---\nmodule: module1\nshort_description: module1 Test module\ndescription:\n    - module1 Test module\nauthor:\n    - Ansible Core Team\n'
EXAMPLES = '\n'
RETURN = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict(desc=dict(type='str')))
    results = dict(msg='you just ran me.mycoll2.module1', desc=module.params.get('desc'))
    module.exit_json(**results)
if __name__ == '__main__':
    main()