from __future__ import annotations
DOCUMENTATION = "\n---\nmodule: eosfacts\nshort_description: module to test module_defaults\ndescription: module to test module_defaults\nversion_added: '2.13'\n"
EXAMPLES = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec=dict(eosfacts=dict(type=bool)), supports_check_mode=True)
    module.exit_json(eosfacts=module.params['eosfacts'])
if __name__ == '__main__':
    main()