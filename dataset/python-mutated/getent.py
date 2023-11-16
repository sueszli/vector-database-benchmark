from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: getent\nshort_description: A wrapper to the unix getent utility\ndescription:\n     - Runs getent against one of its various databases and returns information into\n       the host\'s facts, in a C(getent_<database>) prefixed variable.\nversion_added: "1.8"\noptions:\n    database:\n        description:\n            - The name of a getent database supported by the target system (passwd, group,\n              hosts, etc).\n        type: str\n        required: True\n    key:\n        description:\n            - Key from which to return values from the specified database, otherwise the\n              full contents are returned.\n        type: str\n    service:\n        description:\n            - Override all databases with the specified service\n            - The underlying system must support the service flag which is not always available.\n        type: str\n        version_added: "2.9"\n    split:\n        description:\n            - Character used to split the database values into lists/arrays such as V(:) or V(\\\\t),\n              otherwise it will try to pick one depending on the database.\n        type: str\n    fail_key:\n        description:\n            - If a supplied key is missing this will make the task fail if V(true).\n        type: bool\n        default: \'yes\'\nextends_documentation_fragment:\n  - action_common_attributes\n  - action_common_attributes.facts\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    facts:\n        support: full\n    platform:\n        platforms: posix\nnotes:\n   - Not all databases support enumeration, check system documentation for details.\nauthor:\n- Brian Coca (@bcoca)\n'
EXAMPLES = "\n- name: Get root user info\n  ansible.builtin.getent:\n    database: passwd\n    key: root\n- ansible.builtin.debug:\n    var: ansible_facts.getent_passwd\n\n- name: Get all groups\n  ansible.builtin.getent:\n    database: group\n    split: ':'\n- ansible.builtin.debug:\n    var: ansible_facts.getent_group\n\n- name: Get all hosts, split by tab\n  ansible.builtin.getent:\n    database: hosts\n- ansible.builtin.debug:\n    var: ansible_facts.getent_hosts\n\n- name: Get http service info, no error if missing\n  ansible.builtin.getent:\n    database: services\n    key: http\n    fail_key: False\n- ansible.builtin.debug:\n    var: ansible_facts.getent_services\n\n- name: Get user password hash (requires sudo/root)\n  ansible.builtin.getent:\n    database: shadow\n    key: www-data\n    split: ':'\n- ansible.builtin.debug:\n    var: ansible_facts.getent_shadow\n\n"
RETURN = '\nansible_facts:\n  description: Facts to add to ansible_facts.\n  returned: always\n  type: dict\n  contains:\n    getent_<database>:\n      description:\n        - A list of results or a single result as a list of the fields the db provides\n        - The list elements depend on the database queried, see getent man page for the structure\n        - Starting at 2.11 it now returns multiple duplicate entries, previously it only returned the last one\n      returned: always\n      type: list\n'
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(database=dict(type='str', required=True), key=dict(type='str', no_log=False), service=dict(type='str'), split=dict(type='str'), fail_key=dict(type='bool', default=True)), supports_check_mode=True)
    colon = ['passwd', 'shadow', 'group', 'gshadow']
    database = module.params['database']
    key = module.params.get('key')
    split = module.params.get('split')
    service = module.params.get('service')
    fail_key = module.params.get('fail_key')
    getent_bin = module.get_bin_path('getent', True)
    if key is not None:
        cmd = [getent_bin, database, key]
    else:
        cmd = [getent_bin, database]
    if service is not None:
        cmd.extend(['-s', service])
    if split is None and database in colon:
        split = ':'
    try:
        (rc, out, err) = module.run_command(cmd)
    except Exception as e:
        module.fail_json(msg=to_native(e), exception=traceback.format_exc())
    msg = 'Unexpected failure!'
    dbtree = 'getent_%s' % database
    results = {dbtree: {}}
    if rc == 0:
        seen = {}
        for line in out.splitlines():
            record = line.split(split)
            if record[0] in seen:
                if seen[record[0]] == 1:
                    results[dbtree][record[0]] = [results[dbtree][record[0]]]
                results[dbtree][record[0]].append(record[1:])
                seen[record[0]] += 1
            else:
                results[dbtree][record[0]] = record[1:]
                seen[record[0]] = 1
        module.exit_json(ansible_facts=results)
    elif rc == 1:
        msg = 'Missing arguments, or database unknown.'
    elif rc == 2:
        msg = 'One or more supplied key could not be found in the database.'
        if not fail_key:
            results[dbtree][key] = None
            module.exit_json(ansible_facts=results, msg=msg)
    elif rc == 3:
        msg = 'Enumeration not supported on this database.'
    module.fail_json(msg=msg)
if __name__ == '__main__':
    main()