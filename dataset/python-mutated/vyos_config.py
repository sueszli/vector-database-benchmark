from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'network'}
DOCUMENTATION = "module: vyos_config\nauthor: Nathaniel Case (@Qalthos)\nshort_description: Manage VyOS configuration on remote device\ndescription:\n- This module provides configuration file management of VyOS devices. It provides\n  arguments for managing both the configuration file and state of the active configuration.\n  All configuration statements are based on `set` and `delete` commands in the device\n  configuration.\nextends_documentation_fragment:\n- vyos.vyos.vyos\nnotes:\n- Tested against VyOS 1.1.8 (helium).\n- This module works with connection C(network_cli). See L(the VyOS OS Platform Options,../network/user_guide/platform_vyos.html).\noptions:\n  lines:\n    description:\n    - The ordered set of configuration lines to be managed and compared with the existing\n      configuration on the remote device.\n  src:\n    description:\n    - The C(src) argument specifies the path to the source config file to load.  The\n      source config file can either be in bracket format or set format.  The source\n      file can include Jinja2 template variables.\n  match:\n    description:\n    - The C(match) argument controls the method used to match against the current\n      active configuration.  By default, the desired config is matched against the\n      active config and the deltas are loaded.  If the C(match) argument is set to\n      C(none) the active configuration is ignored and the configuration is always\n      loaded.\n    default: line\n    choices:\n    - line\n    - none\n  backup:\n    description:\n    - The C(backup) argument will backup the current devices active configuration\n      to the Ansible control host prior to making any changes. If the C(backup_options)\n      value is not given, the backup file will be located in the backup folder in\n      the playbook root directory or role root directory, if playbook is part of an\n      ansible role. If the directory does not exist, it is created.\n    type: bool\n    default: 'no'\n  comment:\n    description:\n    - Allows a commit description to be specified to be included when the configuration\n      is committed.  If the configuration is not changed or committed, this argument\n      is ignored.\n    default: configured by vyos_config\n  config:\n    description:\n    - The C(config) argument specifies the base configuration to use to compare against\n      the desired configuration.  If this value is not specified, the module will\n      automatically retrieve the current active configuration from the remote device.\n  save:\n    description:\n    - The C(save) argument controls whether or not changes made to the active configuration\n      are saved to disk.  This is independent of committing the config.  When set\n      to True, the active configuration is saved.\n    type: bool\n    default: 'no'\n  backup_options:\n    description:\n    - This is a dict object containing configurable options related to backup file\n      path. The value of this option is read only when C(backup) is set to I(yes),\n      if C(backup) is set to I(no) this option will be silently ignored.\n    suboptions:\n      filename:\n        description:\n        - The filename to be used to store the backup configuration. If the filename\n          is not given it will be generated based on the hostname, current time and\n          date in format defined by <hostname>_config.<current-date>@<current-time>\n      dir_path:\n        description:\n        - This option provides the path ending with directory name in which the backup\n          configuration file will be stored. If the directory does not exist it will\n          be first created and the filename is either the value of C(filename) or\n          default filename as described in C(filename) options description. If the\n          path value is not given in that case a I(backup) directory will be created\n          in the current working directory and backup configuration will be copied\n          in C(filename) within I(backup) directory.\n        type: path\n    type: dict\n"
EXAMPLES = "\n- name: configure the remote device\n  vyos_config:\n    lines:\n      - set system host-name {{ inventory_hostname }}\n      - set service lldp\n      - delete service dhcp-server\n\n- name: backup and load from file\n  vyos_config:\n    src: vyos.cfg\n    backup: yes\n\n- name: render a Jinja2 template onto the VyOS router\n  vyos_config:\n    src: vyos_template.j2\n\n- name: for idempotency, use full-form commands\n  vyos_config:\n    lines:\n      # - set int eth eth2 description 'OUTSIDE'\n      - set interface ethernet eth2 description 'OUTSIDE'\n\n- name: configurable backup path\n  vyos_config:\n    backup: yes\n    backup_options:\n      filename: backup.cfg\n      dir_path: /home/user\n"
RETURN = '\ncommands:\n  description: The list of configuration commands sent to the device\n  returned: always\n  type: list\n  sample: [\'...\', \'...\']\nfiltered:\n  description: The list of configuration commands removed to avoid a load failure\n  returned: always\n  type: list\n  sample: [\'...\', \'...\']\nbackup_path:\n  description: The full path to the backup file\n  returned: when backup is yes\n  type: str\n  sample: /playbooks/ansible/backup/vyos_config.2016-07-16@22:28:34\nfilename:\n  description: The name of the backup file\n  returned: when backup is yes and filename is not specified in backup options\n  type: str\n  sample: vyos_config.2016-07-16@22:28:34\nshortname:\n  description: The full path to the backup file excluding the timestamp\n  returned: when backup is yes and filename is not specified in backup options\n  type: str\n  sample: /playbooks/ansible/backup/vyos_config\ndate:\n  description: The date extracted from the backup file name\n  returned: when backup is yes\n  type: str\n  sample: "2016-07-16"\ntime:\n  description: The time extracted from the backup file name\n  returned: when backup is yes\n  type: str\n  sample: "22:28:34"\n'
import re
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import load_config, get_config, run_commands
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import vyos_argument_spec, get_connection
DEFAULT_COMMENT = 'configured by vyos_config'
CONFIG_FILTERS = [re.compile('set system login user \\S+ authentication encrypted-password')]

def get_candidate(module):
    if False:
        i = 10
        return i + 15
    contents = module.params['src'] or module.params['lines']
    if module.params['src']:
        contents = format_commands(contents.splitlines())
    contents = '\n'.join(contents)
    return contents

def format_commands(commands):
    if False:
        print('Hello World!')
    "\n    This function format the input commands and removes the prepend white spaces\n    for command lines having 'set' or 'delete' and it skips empty lines.\n    :param commands:\n    :return: list of commands\n    "
    return [line.strip() if line.split()[0] in ('set', 'delete') else line for line in commands if len(line.strip()) > 0]

def diff_config(commands, config):
    if False:
        for i in range(10):
            print('nop')
    config = [str(c).replace("'", '') for c in config.splitlines()]
    updates = list()
    visited = set()
    for line in commands:
        item = str(line).replace("'", '')
        if not item.startswith('set') and (not item.startswith('delete')):
            raise ValueError('line must start with either `set` or `delete`')
        elif item.startswith('set') and item not in config:
            updates.append(line)
        elif item.startswith('delete'):
            if not config:
                updates.append(line)
            else:
                item = re.sub('delete', 'set', item)
                for entry in config:
                    if entry.startswith(item) and line not in visited:
                        updates.append(line)
                        visited.add(line)
    return list(updates)

def sanitize_config(config, result):
    if False:
        while True:
            i = 10
    result['filtered'] = list()
    index_to_filter = list()
    for regex in CONFIG_FILTERS:
        for (index, line) in enumerate(list(config)):
            if regex.search(line):
                result['filtered'].append(line)
                index_to_filter.append(index)
    for filter_index in sorted(index_to_filter, reverse=True):
        del config[filter_index]

def run(module, result):
    if False:
        i = 10
        return i + 15
    config = module.params['config'] or get_config(module)
    candidate = get_candidate(module)
    connection = get_connection(module)
    try:
        response = connection.get_diff(candidate=candidate, running=config, diff_match=module.params['match'])
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    commands = response.get('config_diff')
    sanitize_config(commands, result)
    result['commands'] = commands
    commit = not module.check_mode
    comment = module.params['comment']
    diff = None
    if commands:
        diff = load_config(module, commands, commit=commit, comment=comment)
        if result.get('filtered'):
            result['warnings'].append('Some configuration commands were removed, please see the filtered key')
        result['changed'] = True
    if module._diff:
        result['diff'] = {'prepared': diff}

def main():
    if False:
        print('Hello World!')
    backup_spec = dict(filename=dict(), dir_path=dict(type='path'))
    argument_spec = dict(src=dict(type='path'), lines=dict(type='list'), match=dict(default='line', choices=['line', 'none']), comment=dict(default=DEFAULT_COMMENT), config=dict(), backup=dict(type='bool', default=False), backup_options=dict(type='dict', options=backup_spec), save=dict(type='bool', default=False))
    argument_spec.update(vyos_argument_spec)
    mutually_exclusive = [('lines', 'src')]
    module = AnsibleModule(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True)
    warnings = list()
    result = dict(changed=False, warnings=warnings)
    if module.params['backup']:
        result['__backup__'] = get_config(module=module)
    if any((module.params['src'], module.params['lines'])):
        run(module, result)
    if module.params['save']:
        diff = run_commands(module, commands=['configure', 'compare saved'])[1]
        if diff != '[edit]':
            run_commands(module, commands=['save'])
            result['changed'] = True
        run_commands(module, commands=['exit'])
    module.exit_json(**result)
if __name__ == '__main__':
    main()