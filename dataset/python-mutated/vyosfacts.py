from __future__ import annotations
DOCUMENTATION = "\n---\nmodule: vyosfacts\nshort_description: module to test module_defaults\ndescription: module to test module_defaults\nversion_added: '2.13'\n"
EXAMPLES = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict(vyosfacts=dict(type=bool)), supports_check_mode=True)
    module.exit_json(vyosfacts=module.params['vyosfacts'])
if __name__ == '__main__':
    main()