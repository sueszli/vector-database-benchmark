from __future__ import annotations
DOCUMENTATION = "\n---\nmodule: ios_facts\nshort_description: module to test module_defaults\ndescription: module to test module_defaults\nversion_added: '2.13'\n"
EXAMPLES = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(ios_facts=dict(type=bool)), supports_check_mode=True)
    module.exit_json(ios_facts=module.params['ios_facts'])
if __name__ == '__main__':
    main()