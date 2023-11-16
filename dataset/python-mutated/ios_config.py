from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'network'}
DOCUMENTATION = "module: ios_config\nauthor: Peter Sprygada (@privateip)\nshort_description: Manage Cisco IOS configuration sections\ndescription:\n- Cisco IOS configurations use a simple block indent file syntax for segmenting configuration\n  into sections.  This module provides an implementation for working with IOS configuration\n  sections in a deterministic way.\nextends_documentation_fragment:\n- cisco.ios.ios\nnotes:\n- Tested against IOS 15.6\n- Abbreviated commands are NOT idempotent,\n  see L(Network FAQ,../network/user_guide/faq.html#why-do-the-config-modules-always-return-changed-true-with-abbreviated-commands).\noptions:\n  lines:\n    description:\n    - The ordered set of commands that should be configured in the section.  The commands\n      must be the exact same commands as found in the device running-config.  Be sure\n      to note the configuration command syntax as some commands are automatically\n      modified by the device config parser.\n    aliases:\n    - commands\n  parents:\n    description:\n    - The ordered set of parents that uniquely identify the section or hierarchy the\n      commands should be checked against.  If the parents argument is omitted, the\n      commands are checked against the set of top level or global commands.\n  src:\n    description:\n    - Specifies the source path to the file that contains the configuration or configuration\n      template to load.  The path to the source file can either be the full path on\n      the Ansible control host or a relative path from the playbook or role root directory.  This\n      argument is mutually exclusive with I(lines), I(parents).\n  before:\n    description:\n    - The ordered set of commands to push on to the command stack if a change needs\n      to be made.  This allows the playbook designer the opportunity to perform configuration\n      commands prior to pushing any changes without affecting how the set of commands\n      are matched against the system.\n  after:\n    description:\n    - The ordered set of commands to append to the end of the command stack if a change\n      needs to be made.  Just like with I(before) this allows the playbook designer\n      to append a set of commands to be executed after the command set.\n  match:\n    description:\n    - Instructs the module on the way to perform the matching of the set of commands\n      against the current device config.  If match is set to I(line), commands are\n      matched line by line.  If match is set to I(strict), command lines are matched\n      with respect to position.  If match is set to I(exact), command lines must be\n      an equal match.  Finally, if match is set to I(none), the module will not attempt\n      to compare the source configuration with the running configuration on the remote\n      device.\n    choices:\n    - line\n    - strict\n    - exact\n    - none\n    default: line\n  replace:\n    description:\n    - Instructs the module on the way to perform the configuration on the device.\n      If the replace argument is set to I(line) then the modified lines are pushed\n      to the device in configuration mode.  If the replace argument is set to I(block)\n      then the entire command block is pushed to the device in configuration mode\n      if any line is not correct.\n    default: line\n    choices:\n    - line\n    - block\n  multiline_delimiter:\n    description:\n    - This argument is used when pushing a multiline configuration element to the\n      IOS device.  It specifies the character to use as the delimiting character.  This\n      only applies to the configuration action.\n    default: '@'\n  backup:\n    description:\n    - This argument will cause the module to create a full backup of the current C(running-config)\n      from the remote device before any changes are made. If the C(backup_options)\n      value is not given, the backup file is written to the C(backup) folder in the\n      playbook root directory or role root directory, if playbook is part of an ansible\n      role. If the directory does not exist, it is created.\n    type: bool\n    default: 'no'\n  running_config:\n    description:\n    - The module, by default, will connect to the remote device and retrieve the current\n      running-config to use as a base for comparing against the contents of source.\n      There are times when it is not desirable to have the task get the current running-config\n      for every task in a playbook.  The I(running_config) argument allows the implementer\n      to pass in the configuration to use as the base config for comparison.\n    aliases:\n    - config\n  defaults:\n    description:\n    - This argument specifies whether or not to collect all defaults when getting\n      the remote device running config.  When enabled, the module will get the current\n      config by issuing the command C(show running-config all).\n    type: bool\n    default: 'no'\n  save_when:\n    description:\n    - When changes are made to the device running-configuration, the changes are not\n      copied to non-volatile storage by default.  Using this argument will change\n      that before.  If the argument is set to I(always), then the running-config will\n      always be copied to the startup-config and the I(modified) flag will always\n      be set to True.  If the argument is set to I(modified), then the running-config\n      will only be copied to the startup-config if it has changed since the last save\n      to startup-config.  If the argument is set to I(never), the running-config will\n      never be copied to the startup-config.  If the argument is set to I(changed),\n      then the running-config will only be copied to the startup-config if the task\n      has made a change. I(changed) was added in Ansible 2.5.\n    default: never\n    choices:\n    - always\n    - never\n    - modified\n    - changed\n  diff_against:\n    description:\n    - When using the C(ansible-playbook --diff) command line argument the module can\n      generate diffs against different sources.\n    - When this option is configure as I(startup), the module will return the diff\n      of the running-config against the startup-config.\n    - When this option is configured as I(intended), the module will return the diff\n      of the running-config against the configuration provided in the C(intended_config)\n      argument.\n    - When this option is configured as I(running), the module will return the before\n      and after diff of the running-config with respect to any changes made to the\n      device configuration.\n    choices:\n    - running\n    - startup\n    - intended\n  diff_ignore_lines:\n    description:\n    - Use this argument to specify one or more lines that should be ignored during\n      the diff.  This is used for lines in the configuration that are automatically\n      updated by the system.  This argument takes a list of regular expressions or\n      exact line matches.\n  intended_config:\n    description:\n    - The C(intended_config) provides the master configuration that the node should\n      conform to and is used to check the final running-config against. This argument\n      will not modify any settings on the remote device and is strictly used to check\n      the compliance of the current device's configuration against.  When specifying\n      this argument, the task should also modify the C(diff_against) value and set\n      it to I(intended).\n  backup_options:\n    description:\n    - This is a dict object containing configurable options related to backup file\n      path. The value of this option is read only when C(backup) is set to I(yes),\n      if C(backup) is set to I(no) this option will be silently ignored.\n    suboptions:\n      filename:\n        description:\n        - The filename to be used to store the backup configuration. If the filename\n          is not given it will be generated based on the hostname, current time and\n          date in format defined by <hostname>_config.<current-date>@<current-time>\n      dir_path:\n        description:\n        - This option provides the path ending with directory name in which the backup\n          configuration file will be stored. If the directory does not exist it will\n          be first created and the filename is either the value of C(filename) or\n          default filename as described in C(filename) options description. If the\n          path value is not given in that case a I(backup) directory will be created\n          in the current working directory and backup configuration will be copied\n          in C(filename) within I(backup) directory.\n        type: path\n    type: dict\n"
EXAMPLES = '\n- name: configure top level configuration\n  ios_config:\n    lines: hostname {{ inventory_hostname }}\n\n- name: configure interface settings\n  ios_config:\n    lines:\n      - description test interface\n      - ip address 172.31.1.1 255.255.255.0\n    parents: interface Ethernet1\n\n- name: configure ip helpers on multiple interfaces\n  ios_config:\n    lines:\n      - ip helper-address 172.26.1.10\n      - ip helper-address 172.26.3.8\n    parents: "{{ item }}"\n  with_items:\n    - interface Ethernet1\n    - interface Ethernet2\n    - interface GigabitEthernet1\n\n- name: configure policer in Scavenger class\n  ios_config:\n    lines:\n      - conform-action transmit\n      - exceed-action drop\n    parents:\n      - policy-map Foo\n      - class Scavenger\n      - police cir 64000\n\n- name: load new acl into device\n  ios_config:\n    lines:\n      - 10 permit ip host 192.0.2.1 any log\n      - 20 permit ip host 192.0.2.2 any log\n      - 30 permit ip host 192.0.2.3 any log\n      - 40 permit ip host 192.0.2.4 any log\n      - 50 permit ip host 192.0.2.5 any log\n    parents: ip access-list extended test\n    before: no ip access-list extended test\n    match: exact\n\n- name: check the running-config against master config\n  ios_config:\n    diff_against: intended\n    intended_config: "{{ lookup(\'file\', \'master.cfg\') }}"\n\n- name: check the startup-config against the running-config\n  ios_config:\n    diff_against: startup\n    diff_ignore_lines:\n      - ntp clock .*\n\n- name: save running to startup when modified\n  ios_config:\n    save_when: modified\n\n- name: for idempotency, use full-form commands\n  ios_config:\n    lines:\n      # - shut\n      - shutdown\n    # parents: int gig1/0/11\n    parents: interface GigabitEthernet1/0/11\n\n# Set boot image based on comparison to a group_var (version) and the version\n# that is returned from the `ios_facts` module\n- name: SETTING BOOT IMAGE\n  ios_config:\n    lines:\n      - no boot system\n      - boot system flash bootflash:{{new_image}}\n    host: "{{ inventory_hostname }}"\n  when: ansible_net_version != version\n\n- name: render a Jinja2 template onto an IOS device\n  ios_config:\n    backup: yes\n    src: ios_template.j2\n\n- name: configurable backup path\n  ios_config:\n    src: ios_template.j2\n    backup: yes\n    backup_options:\n      filename: backup.cfg\n      dir_path: /home/user\n'
RETURN = '\nupdates:\n  description: The set of commands that will be pushed to the remote device\n  returned: always\n  type: list\n  sample: [\'hostname foo\', \'router ospf 1\', \'router-id 192.0.2.1\']\ncommands:\n  description: The set of commands that will be pushed to the remote device\n  returned: always\n  type: list\n  sample: [\'hostname foo\', \'router ospf 1\', \'router-id 192.0.2.1\']\nbackup_path:\n  description: The full path to the backup file\n  returned: when backup is yes\n  type: str\n  sample: /playbooks/ansible/backup/ios_config.2016-07-16@22:28:34\nfilename:\n  description: The name of the backup file\n  returned: when backup is yes and filename is not specified in backup options\n  type: str\n  sample: ios_config.2016-07-16@22:28:34\nshortname:\n  description: The full path to the backup file excluding the timestamp\n  returned: when backup is yes and filename is not specified in backup options\n  type: str\n  sample: /playbooks/ansible/backup/ios_config\ndate:\n  description: The date extracted from the backup file name\n  returned: when backup is yes\n  type: str\n  sample: "2016-07-16"\ntime:\n  description: The time extracted from the backup file name\n  returned: when backup is yes\n  type: str\n  sample: "22:28:34"\n'
import json
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import ConnectionError
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import run_commands, get_config
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import get_defaults_flag, get_connection
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import ios_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps

def check_args(module, warnings):
    if False:
        i = 10
        return i + 15
    if module.params['multiline_delimiter']:
        if len(module.params['multiline_delimiter']) != 1:
            module.fail_json(msg='multiline_delimiter value can only be a single character')

def edit_config_or_macro(connection, commands):
    if False:
        while True:
            i = 10
    if commands[0].startswith('macro name'):
        connection.edit_macro(candidate=commands)
    else:
        connection.edit_config(candidate=commands)

def get_candidate_config(module):
    if False:
        for i in range(10):
            print('nop')
    candidate = ''
    if module.params['src']:
        candidate = module.params['src']
    elif module.params['lines']:
        candidate_obj = NetworkConfig(indent=1)
        parents = module.params['parents'] or list()
        candidate_obj.add(module.params['lines'], parents=parents)
        candidate = dumps(candidate_obj, 'raw')
    return candidate

def get_running_config(module, current_config=None, flags=None):
    if False:
        i = 10
        return i + 15
    running = module.params['running_config']
    if not running:
        if not module.params['defaults'] and current_config:
            running = current_config
        else:
            running = get_config(module, flags=flags)
    return running

def save_config(module, result):
    if False:
        while True:
            i = 10
    result['changed'] = True
    if not module.check_mode:
        run_commands(module, 'copy running-config startup-config\r')
    else:
        module.warn('Skipping command `copy running-config startup-config` due to check_mode.  Configuration not copied to non-volatile storage')

def main():
    if False:
        return 10
    ' main entry point for module execution\n    '
    backup_spec = dict(filename=dict(), dir_path=dict(type='path'))
    argument_spec = dict(src=dict(type='path'), lines=dict(aliases=['commands'], type='list'), parents=dict(type='list'), before=dict(type='list'), after=dict(type='list'), match=dict(default='line', choices=['line', 'strict', 'exact', 'none']), replace=dict(default='line', choices=['line', 'block']), multiline_delimiter=dict(default='@'), running_config=dict(aliases=['config']), intended_config=dict(), defaults=dict(type='bool', default=False), backup=dict(type='bool', default=False), backup_options=dict(type='dict', options=backup_spec), save_when=dict(choices=['always', 'never', 'modified', 'changed'], default='never'), diff_against=dict(choices=['startup', 'intended', 'running']), diff_ignore_lines=dict(type='list'))
    argument_spec.update(ios_argument_spec)
    mutually_exclusive = [('lines', 'src'), ('parents', 'src')]
    required_if = [('match', 'strict', ['lines']), ('match', 'exact', ['lines']), ('replace', 'block', ['lines']), ('diff_against', 'intended', ['intended_config'])]
    module = AnsibleModule(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, required_if=required_if, supports_check_mode=True)
    result = {'changed': False}
    warnings = list()
    check_args(module, warnings)
    result['warnings'] = warnings
    diff_ignore_lines = module.params['diff_ignore_lines']
    config = None
    contents = None
    flags = get_defaults_flag(module) if module.params['defaults'] else []
    connection = get_connection(module)
    if module.params['backup'] or (module._diff and module.params['diff_against'] == 'running'):
        contents = get_config(module, flags=flags)
        config = NetworkConfig(indent=1, contents=contents)
        if module.params['backup']:
            result['__backup__'] = contents
    if any((module.params['lines'], module.params['src'])):
        match = module.params['match']
        replace = module.params['replace']
        path = module.params['parents']
        candidate = get_candidate_config(module)
        running = get_running_config(module, contents, flags=flags)
        try:
            response = connection.get_diff(candidate=candidate, running=running, diff_match=match, diff_ignore_lines=diff_ignore_lines, path=path, diff_replace=replace)
        except ConnectionError as exc:
            module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
        config_diff = response['config_diff']
        banner_diff = response['banner_diff']
        if config_diff or banner_diff:
            commands = config_diff.split('\n')
            if module.params['before']:
                commands[:0] = module.params['before']
            if module.params['after']:
                commands.extend(module.params['after'])
            result['commands'] = commands
            result['updates'] = commands
            result['banners'] = banner_diff
            if not module.check_mode:
                if commands:
                    edit_config_or_macro(connection, commands)
                if banner_diff:
                    connection.edit_banner(candidate=json.dumps(banner_diff), multiline_delimiter=module.params['multiline_delimiter'])
            result['changed'] = True
    running_config = module.params['running_config']
    startup_config = None
    if module.params['save_when'] == 'always':
        save_config(module, result)
    elif module.params['save_when'] == 'modified':
        output = run_commands(module, ['show running-config', 'show startup-config'])
        running_config = NetworkConfig(indent=1, contents=output[0], ignore_lines=diff_ignore_lines)
        startup_config = NetworkConfig(indent=1, contents=output[1], ignore_lines=diff_ignore_lines)
        if running_config.sha1 != startup_config.sha1:
            save_config(module, result)
    elif module.params['save_when'] == 'changed' and result['changed']:
        save_config(module, result)
    if module._diff:
        if not running_config:
            output = run_commands(module, 'show running-config')
            contents = output[0]
        else:
            contents = running_config
        running_config = NetworkConfig(indent=1, contents=contents, ignore_lines=diff_ignore_lines)
        if module.params['diff_against'] == 'running':
            if module.check_mode:
                module.warn('unable to perform diff against running-config due to check mode')
                contents = None
            else:
                contents = config.config_text
        elif module.params['diff_against'] == 'startup':
            if not startup_config:
                output = run_commands(module, 'show startup-config')
                contents = output[0]
            else:
                contents = startup_config.config_text
        elif module.params['diff_against'] == 'intended':
            contents = module.params['intended_config']
        if contents is not None:
            base_config = NetworkConfig(indent=1, contents=contents, ignore_lines=diff_ignore_lines)
            if running_config.sha1 != base_config.sha1:
                (before, after) = ('', '')
                if module.params['diff_against'] == 'intended':
                    before = running_config
                    after = base_config
                elif module.params['diff_against'] in ('startup', 'running'):
                    before = base_config
                    after = running_config
                result.update({'changed': True, 'diff': {'before': str(before), 'after': str(after)}})
    module.exit_json(**result)
if __name__ == '__main__':
    main()