from __future__ import annotations
DOCUMENTATION = "\n---\nmodule: legacy_module\nshort_description: A module to test a task's resolved action name.\ndescription: A module to test a task's resolved action name.\noptions: {}\nauthor: Ansible Core Team\nnotes:\n  - Supports C(check_mode).\n"
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(supports_check_mode=True, argument_spec={})
    module.exit_json(changed=False)
if __name__ == '__main__':
    main()