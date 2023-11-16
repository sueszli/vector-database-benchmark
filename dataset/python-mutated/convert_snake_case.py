from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: convert_snake_case\nshort_description: test converting data to snake_case\ndescription: test converting data to snake_case\noptions:\n  data:\n    description: Data to modify\n    type: dict\n    required: True\n  reversible:\n    description:\n      - Make the snake_case conversion in a way that can be converted back to the original value\n      - For example, convert IAMUser to i_a_m_user instead of iam_user\n    default: False\n  ignore_list:\n    description: list of top level keys that should not have their contents converted\n    type: list\n    default: []\n'
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec=dict(data=dict(type='dict', required=True), reversible=dict(type='bool', default=False), ignore_list=dict(type='list', default=[])))
    result = camel_dict_to_snake_dict(module.params['data'], module.params['reversible'], module.params['ignore_list'])
    module.exit_json(data=result)
if __name__ == '__main__':
    main()