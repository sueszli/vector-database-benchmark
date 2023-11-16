from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
try:
    import ansible_missing_lib
    HAS_LIB = True
except ImportError as e:
    HAS_LIB = False

def main():
    if False:
        return 10
    module = AnsibleModule({'url': {'type': 'bool'}, 'reason': {'type': 'bool'}})
    kwargs = {}
    if module.params['url']:
        kwargs['url'] = 'https://github.com/ansible/ansible'
    if module.params['reason']:
        kwargs['reason'] = 'for fun'
    if not HAS_LIB:
        module.fail_json(msg=missing_required_lib('ansible_missing_lib', **kwargs))
if __name__ == '__main__':
    main()