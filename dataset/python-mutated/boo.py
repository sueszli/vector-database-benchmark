"""Say hello in Ukrainian."""
from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec={'name': {'default': 'світ'}})
    name = module.params['name']
    module.exit_json(msg='Greeting {name} completed.'.format(name=name.title()), greeting='Привіт, {name}!'.format(name=name))
if __name__ == '__main__':
    main()