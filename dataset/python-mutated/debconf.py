from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: debconf\nshort_description: Configure a .deb package\ndescription:\n     - Configure a .deb package using debconf-set-selections.\n     - Or just query existing selections.\nversion_added: "1.6"\nextends_documentation_fragment:\n- action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        support: full\n        platforms: debian\nnotes:\n    - This module requires the command line debconf tools.\n    - Several questions have to be answered (depending on the package).\n      Use \'debconf-show <package>\' on any Debian or derivative with the package\n      installed to see questions/settings available.\n    - Some distros will always record tasks involving the setting of passwords as changed. This is due to debconf-get-selections masking passwords.\n    - It is highly recommended to add C(no_log=True) to the task while handling sensitive information using this module.\n    - The debconf module does not reconfigure packages, it just updates the debconf database.\n      An additional step is needed (typically with C(notify) if debconf makes a change)\n      to reconfigure the package and apply the changes.\n      debconf is extensively used for pre-seeding configuration prior to installation\n      rather than modifying configurations.\n      So, while dpkg-reconfigure does use debconf data, it is not always authoritative\n      and you may need to check how your package is handled.\n    - Also note dpkg-reconfigure is a 3-phase process. It invokes the\n      control scripts from the C(/var/lib/dpkg/info) directory with the\n      C(<package>.prerm  reconfigure <version>),\n      C(<package>.config reconfigure <version>) and C(<package>.postinst control <version>) arguments.\n    - The main issue is that the C(<package>.config reconfigure) step for many packages\n      will first reset the debconf database (overriding changes made by this module) by\n      checking the on-disk configuration. If this is the case for your package then\n      dpkg-reconfigure will effectively ignore changes made by debconf.\n    - However as dpkg-reconfigure only executes the C(<package>.config) step if the file\n      exists, it is possible to rename it to C(/var/lib/dpkg/info/<package>.config.ignore)\n      before executing C(dpkg-reconfigure -f noninteractive <package>) and then restore it.\n      This seems to be compliant with Debian policy for the .config file.\nrequirements:\n- debconf\n- debconf-utils\noptions:\n  name:\n    description:\n      - Name of package to configure.\n    type: str\n    required: true\n    aliases: [ pkg ]\n  question:\n    description:\n      - A debconf configuration setting.\n    type: str\n    aliases: [ selection, setting ]\n  vtype:\n    description:\n      - The type of the value supplied.\n      - It is highly recommended to add C(no_log=True) to task while specifying O(vtype=password).\n      - V(seen) was added in Ansible 2.2.\n    type: str\n    choices: [ boolean, error, multiselect, note, password, seen, select, string, text, title ]\n  value:\n    description:\n      -  Value to set the configuration to.\n    type: str\n    aliases: [ answer ]\n  unseen:\n    description:\n      - Do not set \'seen\' flag when pre-seeding.\n    type: bool\n    default: false\nauthor:\n- Brian Coca (@bcoca)\n'
EXAMPLES = '\n- name: Set default locale to fr_FR.UTF-8\n  ansible.builtin.debconf:\n    name: locales\n    question: locales/default_environment_locale\n    value: fr_FR.UTF-8\n    vtype: select\n\n- name: Set to generate locales\n  ansible.builtin.debconf:\n    name: locales\n    question: locales/locales_to_be_generated\n    value: en_US.UTF-8 UTF-8, fr_FR.UTF-8 UTF-8\n    vtype: multiselect\n\n- name: Accept oracle license\n  ansible.builtin.debconf:\n    name: oracle-java7-installer\n    question: shared/accepted-oracle-license-v1-1\n    value: \'true\'\n    vtype: select\n\n- name: Specifying package you can register/return the list of questions and current values\n  ansible.builtin.debconf:\n    name: tzdata\n\n- name: Pre-configure tripwire site passphrase\n  ansible.builtin.debconf:\n    name: tripwire\n    question: tripwire/site-passphrase\n    value: "{{ site_passphrase }}"\n    vtype: password\n  no_log: True\n'
RETURN = '#'
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule

def get_password_value(module, pkg, question, vtype):
    if False:
        for i in range(10):
            print('nop')
    getsel = module.get_bin_path('debconf-get-selections', True)
    cmd = [getsel]
    (rc, out, err) = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg="Failed to get the value '%s' from '%s'" % (question, pkg))
    desired_line = None
    for line in out.split('\n'):
        if line.startswith(pkg):
            desired_line = line
            break
    if not desired_line:
        module.fail_json(msg="Failed to find the value '%s' from '%s'" % (question, pkg))
    (dpkg, dquestion, dvtype, dvalue) = desired_line.split()
    if dquestion == question and dvtype == vtype:
        return dvalue
    return ''

def get_selections(module, pkg):
    if False:
        for i in range(10):
            print('nop')
    cmd = [module.get_bin_path('debconf-show', True), pkg]
    (rc, out, err) = module.run_command(' '.join(cmd))
    if rc != 0:
        module.fail_json(msg=err)
    selections = {}
    for line in out.splitlines():
        (key, value) = line.split(':', 1)
        selections[key.strip('*').strip()] = value.strip()
    return selections

def set_selection(module, pkg, question, vtype, value, unseen):
    if False:
        return 10
    setsel = module.get_bin_path('debconf-set-selections', True)
    cmd = [setsel]
    if unseen:
        cmd.append('-u')
    if vtype == 'boolean':
        value = value.lower()
    data = ' '.join([pkg, question, vtype, value])
    return module.run_command(cmd, data=data)

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(name=dict(type='str', required=True, aliases=['pkg']), question=dict(type='str', aliases=['selection', 'setting']), vtype=dict(type='str', choices=['boolean', 'error', 'multiselect', 'note', 'password', 'seen', 'select', 'string', 'text', 'title']), value=dict(type='str', aliases=['answer']), unseen=dict(type='bool', default=False)), required_together=(['question', 'vtype', 'value'],), supports_check_mode=True)
    pkg = module.params['name']
    question = module.params['question']
    vtype = module.params['vtype']
    value = module.params['value']
    unseen = module.params['unseen']
    prev = get_selections(module, pkg)
    changed = False
    msg = ''
    if question is not None:
        if vtype is None or value is None:
            module.fail_json(msg='when supplying a question you must supply a valid vtype and value')
        if question not in prev:
            changed = True
        else:
            existing = prev[question]
            if vtype == 'boolean':
                value = to_text(value).lower()
                existing = to_text(prev[question]).lower()
            if vtype == 'password':
                existing = get_password_value(module, pkg, question, vtype)
            if value != existing:
                changed = True
    if changed:
        if not module.check_mode:
            (rc, msg, e) = set_selection(module, pkg, question, vtype, value, unseen)
            if rc:
                module.fail_json(msg=e)
        curr = {question: value}
        if question in prev:
            prev = {question: prev[question]}
        else:
            prev[question] = ''
        diff_dict = {}
        if module._diff:
            after = prev.copy()
            after.update(curr)
            diff_dict = {'before': prev, 'after': after}
        module.exit_json(changed=changed, msg=msg, current=curr, previous=prev, diff=diff_dict)
    module.exit_json(changed=changed, msg=msg, current=prev)
if __name__ == '__main__':
    main()