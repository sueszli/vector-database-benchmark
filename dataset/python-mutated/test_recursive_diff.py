from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import recursive_diff

def main():
    if False:
        return 10
    module = AnsibleModule({'a': {'type': 'dict'}, 'b': {'type': 'dict'}})
    module.exit_json(the_diff=recursive_diff(module.params['a'], module.params['b']))
if __name__ == '__main__':
    main()