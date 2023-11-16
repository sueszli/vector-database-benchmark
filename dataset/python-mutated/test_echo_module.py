from __future__ import annotations
import sys
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        while True:
            i = 10
    result = dict(changed=False)
    module = AnsibleModule(argument_spec=dict(facts=dict(type=dict, default={})))
    result['ansible_facts'] = module.params['facts']
    result['running_python_interpreter'] = sys.executable
    module.exit_json(**result)
if __name__ == '__main__':
    main()