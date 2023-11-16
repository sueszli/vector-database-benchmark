from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: convert_camelCase\nshort_description: test converting data to camelCase\ndescription: test converting data to camelCase\noptions:\n  data:\n    description: Data to modify\n    type: dict\n    required: True\n  capitalize_first:\n    description: Whether to capitalize the first character\n    default: False\n    type: bool\n'
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(data=dict(type='dict', required=True), capitalize_first=dict(type='bool', default=False)))
    result = snake_dict_to_camel_dict(module.params['data'], module.params['capitalize_first'])
    module.exit_json(data=result)
if __name__ == '__main__':
    main()