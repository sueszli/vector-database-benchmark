from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule, env_fallback

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict(explicit_pass=dict(type='str', no_log=True), fallback_pass=dict(type='str', no_log=True, fallback=(env_fallback, ['SECRET_ENV'])), default_pass=dict(type='str', no_log=True, default='zyx'), normal=dict(type='str', default='plaintext'), suboption=dict(type='dict', options=dict(explicit_sub_pass=dict(type='str', no_log=True), fallback_sub_pass=dict(type='str', no_log=True, fallback=(env_fallback, ['SECRET_SUB_ENV'])), default_sub_pass=dict(type='str', no_log=True, default='xvu'), normal=dict(type='str', default='plaintext')))))
    module.exit_json(changed=False)
if __name__ == '__main__':
    main()