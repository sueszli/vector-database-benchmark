from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: dpkg_selections\nshort_description: Dpkg package selection selections\ndescription:\n    - Change dpkg package selection state via --get-selections and --set-selections.\nversion_added: "2.0"\nauthor:\n- Brian Brazil (@brian-brazil)  <brian.brazil@boxever.com>\noptions:\n    name:\n        description:\n            - Name of the package.\n        required: true\n        type: str\n    selection:\n        description:\n            - The selection state to set the package to.\n        choices: [ \'install\', \'hold\', \'deinstall\', \'purge\' ]\n        required: true\n        type: str\nextends_documentation_fragment:\n- action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        support: full\n        platforms: debian\nnotes:\n    - This module will not cause any packages to be installed/removed/purged, use the M(ansible.builtin.apt) module for that.\n'
EXAMPLES = '\n- name: Prevent python from being upgraded\n  ansible.builtin.dpkg_selections:\n    name: python\n    selection: hold\n\n- name: Allow python to be upgraded\n  ansible.builtin.dpkg_selections:\n    name: python\n    selection: install\n'
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict(name=dict(required=True), selection=dict(choices=['install', 'hold', 'deinstall', 'purge'], required=True)), supports_check_mode=True)
    dpkg = module.get_bin_path('dpkg', True)
    locale = get_best_parsable_locale(module)
    DPKG_ENV = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LC_CTYPE=locale)
    module.run_command_environ_update = DPKG_ENV
    name = module.params['name']
    selection = module.params['selection']
    (rc, out, err) = module.run_command([dpkg, '--get-selections', name], check_rc=True)
    if 'no packages found matching' in err:
        module.fail_json(msg="Failed to find package '%s' to perform selection '%s'." % (name, selection))
    elif not out:
        current = 'not present'
    else:
        current = out.split()[1]
    changed = current != selection
    if module.check_mode or not changed:
        module.exit_json(changed=changed, before=current, after=selection)
    module.run_command([dpkg, '--set-selections'], data='%s %s' % (name, selection), check_rc=True)
    module.exit_json(changed=changed, before=current, after=selection)
if __name__ == '__main__':
    main()