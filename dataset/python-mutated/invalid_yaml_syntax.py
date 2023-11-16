from __future__ import annotations
DOCUMENTATION = '\n- key: "value"wrong\n'
EXAMPLES = '\n- key: "value"wrong\n'
RETURN = '\n- key: "value"wrong\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        return 10
    AnsibleModule(argument_spec=dict())
if __name__ == '__main__':
    main()