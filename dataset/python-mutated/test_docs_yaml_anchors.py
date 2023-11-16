from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: test_docs_yaml_anchors\nshort_description: Test module with YAML anchors in docs\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\noptions:\n  at_the_top: &toplevel_anchor\n    description:\n        - Short desc\n    default: some string\n    type: str\n\n  last_one: *toplevel_anchor\n\n  egress:\n    description:\n        - Egress firewall rules\n    type: list\n    elements: dict\n    suboptions: &sub_anchor\n        port:\n            description:\n                - Rule port\n            type: int\n            required: true\n\n  ingress:\n    description:\n        - Ingress firewall rules\n    type: list\n    elements: dict\n    suboptions: *sub_anchor\n'
EXAMPLES = '\n'
RETURN = '\n'
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec=dict(at_the_top=dict(type='str', default='some string'), last_one=dict(type='str', default='some string'), egress=dict(type='list', elements='dict', options=dict(port=dict(type='int', required=True))), ingress=dict(type='list', elements='dict', options=dict(port=dict(type='int', required=True)))))
    module.exit_json()
if __name__ == '__main__':
    main()