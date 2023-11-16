from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['stableinterface'], 'supported_by': 'core'}
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict())
    module.exit_json()
if __name__ == '__main__':
    main()