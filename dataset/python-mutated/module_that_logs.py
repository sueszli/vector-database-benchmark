from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict(number=dict(type='int')))
    module.log('My number is: (%d)' % module.params['number'])
    module.exit_json()
if __name__ == '__main__':
    main()