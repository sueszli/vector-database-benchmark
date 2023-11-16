from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: blockinfile\nshort_description: Insert/update/remove a text block surrounded by marker lines\nversion_added: \'2.0\'\ndescription:\n- This module will insert/update/remove a block of multi-line text surrounded by customizable marker lines.\nauthor:\n- Yaegashi Takeshi (@yaegashi)\noptions:\n  path:\n    description:\n    - The file to modify.\n    - Before Ansible 2.3 this option was only usable as O(dest), O(destfile) and O(name).\n    type: path\n    required: yes\n    aliases: [ dest, destfile, name ]\n  state:\n    description:\n    - Whether the block should be there or not.\n    type: str\n    choices: [ absent, present ]\n    default: present\n  marker:\n    description:\n    - The marker line template.\n    - C({mark}) will be replaced with the values in O(marker_begin) (default="BEGIN") and O(marker_end) (default="END").\n    - Using a custom marker without the C({mark}) variable may result in the block being repeatedly inserted on subsequent playbook runs.\n    - Multi-line markers are not supported and will result in the block being repeatedly inserted on subsequent playbook runs.\n    - A newline is automatically appended by the module to O(marker_begin) and O(marker_end).\n    type: str\n    default: \'# {mark} ANSIBLE MANAGED BLOCK\'\n  block:\n    description:\n    - The text to insert inside the marker lines.\n    - If it is missing or an empty string, the block will be removed as if O(state) were specified to V(absent).\n    type: str\n    default: \'\'\n    aliases: [ content ]\n  insertafter:\n    description:\n    - If specified and no begin/ending O(marker) lines are found, the block will be inserted after the last match of specified regular expression.\n    - A special value is available; V(EOF) for inserting the block at the end of the file.\n    - If specified regular expression has no matches, V(EOF) will be used instead.\n    - The presence of the multiline flag (?m) in the regular expression controls whether the match is done line by line or with multiple lines.\n      This behaviour was added in ansible-core 2.14.\n    type: str\n    choices: [ EOF, \'*regex*\' ]\n    default: EOF\n  insertbefore:\n    description:\n    - If specified and no begin/ending O(marker) lines are found, the block will be inserted before the last match of specified regular expression.\n    - A special value is available; V(BOF) for inserting the block at the beginning of the file.\n    - If specified regular expression has no matches, the block will be inserted at the end of the file.\n    - The presence of the multiline flag (?m) in the regular expression controls whether the match is done line by line or with multiple lines.\n      This behaviour was added in ansible-core 2.14.\n    type: str\n    choices: [ BOF, \'*regex*\' ]\n  create:\n    description:\n    - Create a new file if it does not exist.\n    type: bool\n    default: no\n  backup:\n    description:\n    - Create a backup file including the timestamp information so you can\n      get the original file back if you somehow clobbered it incorrectly.\n    type: bool\n    default: no\n  marker_begin:\n    description:\n    - This will be inserted at C({mark}) in the opening ansible block O(marker).\n    type: str\n    default: BEGIN\n    version_added: \'2.5\'\n  marker_end:\n    required: false\n    description:\n    - This will be inserted at C({mark}) in the closing ansible block O(marker).\n    type: str\n    default: END\n    version_added: \'2.5\'\n  append_newline:\n    required: false\n    description:\n    - Append a blank line to the inserted block, if this does not appear at the end of the file.\n    - Note that this attribute is not considered when C(state) is set to C(absent)\n    type: bool\n    default: no\n    version_added: \'2.16\'\n  prepend_newline:\n    required: false\n    description:\n    - Prepend a blank line to the inserted block, if this does not appear at the beginning of the file.\n    - Note that this attribute is not considered when C(state) is set to C(absent)\n    type: bool\n    default: no\n    version_added: \'2.16\'\nnotes:\n  - When using \'with_*\' loops be aware that if you do not set a unique mark the block will be overwritten on each iteration.\n  - As of Ansible 2.3, the O(dest) option has been changed to O(path) as default, but O(dest) still works as well.\n  - Option O(ignore:follow) has been removed in Ansible 2.5, because this module modifies the contents of the file\n    so O(ignore:follow=no) does not make sense.\n  - When more than one block should be handled in one file you must change the O(marker) per task.\nextends_documentation_fragment:\n    - action_common_attributes\n    - action_common_attributes.files\n    - files\n    - validate\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    safe_file_operations:\n      support: full\n    platform:\n      support: full\n      platforms: posix\n    vault:\n      support: none\n'
EXAMPLES = '\n# Before Ansible 2.3, option \'dest\' or \'name\' was used instead of \'path\'\n- name: Insert/Update "Match User" configuration block in /etc/ssh/sshd_config prepending and appending a new line\n  ansible.builtin.blockinfile:\n    path: /etc/ssh/sshd_config\n    append_newline: true\n    prepend_newline: true\n    block: |\n      Match User ansible-agent\n      PasswordAuthentication no\n\n- name: Insert/Update eth0 configuration stanza in /etc/network/interfaces\n        (it might be better to copy files into /etc/network/interfaces.d/)\n  ansible.builtin.blockinfile:\n    path: /etc/network/interfaces\n    block: |\n      iface eth0 inet static\n          address 192.0.2.23\n          netmask 255.255.255.0\n\n- name: Insert/Update configuration using a local file and validate it\n  ansible.builtin.blockinfile:\n    block: "{{ lookup(\'ansible.builtin.file\', \'./local/sshd_config\') }}"\n    path: /etc/ssh/sshd_config\n    backup: yes\n    validate: /usr/sbin/sshd -T -f %s\n\n- name: Insert/Update HTML surrounded by custom markers after <body> line\n  ansible.builtin.blockinfile:\n    path: /var/www/html/index.html\n    marker: "<!-- {mark} ANSIBLE MANAGED BLOCK -->"\n    insertafter: "<body>"\n    block: |\n      <h1>Welcome to {{ ansible_hostname }}</h1>\n      <p>Last updated on {{ ansible_date_time.iso8601 }}</p>\n\n- name: Remove HTML as well as surrounding markers\n  ansible.builtin.blockinfile:\n    path: /var/www/html/index.html\n    marker: "<!-- {mark} ANSIBLE MANAGED BLOCK -->"\n    block: ""\n\n- name: Add mappings to /etc/hosts\n  ansible.builtin.blockinfile:\n    path: /etc/hosts\n    block: |\n      {{ item.ip }} {{ item.name }}\n    marker: "# {mark} ANSIBLE MANAGED BLOCK {{ item.name }}"\n  loop:\n    - { name: host1, ip: 10.10.1.10 }\n    - { name: host2, ip: 10.10.1.11 }\n    - { name: host3, ip: 10.10.1.12 }\n\n- name: Search with a multiline search flags regex and if found insert after\n  blockinfile:\n    path: listener.ora\n    block: "{{ listener_line | indent(width=8, first=True) }}"\n    insertafter: \'(?m)SID_LIST_LISTENER_DG =\\n.*\\(SID_LIST =\'\n    marker: "    <!-- {mark} ANSIBLE MANAGED BLOCK -->"\n\n'
import re
import os
import tempfile
from ansible.module_utils.six import b
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native

def write_changes(module, contents, path):
    if False:
        i = 10
        return i + 15
    (tmpfd, tmpfile) = tempfile.mkstemp(dir=module.tmpdir)
    f = os.fdopen(tmpfd, 'wb')
    f.write(contents)
    f.close()
    validate = module.params.get('validate', None)
    valid = not validate
    if validate:
        if '%s' not in validate:
            module.fail_json(msg='validate must contain %%s: %s' % validate)
        (rc, out, err) = module.run_command(validate % tmpfile)
        valid = rc == 0
        if rc != 0:
            module.fail_json(msg='failed to validate: rc:%s error:%s' % (rc, err))
    if valid:
        module.atomic_move(tmpfile, path, unsafe_writes=module.params['unsafe_writes'])

def check_file_attrs(module, changed, message, diff):
    if False:
        print('Hello World!')
    file_args = module.load_file_common_arguments(module.params)
    if module.set_file_attributes_if_different(file_args, False, diff=diff):
        if changed:
            message += ' and '
        changed = True
        message += 'ownership, perms or SE linux context changed'
    return (message, changed)

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict(path=dict(type='path', required=True, aliases=['dest', 'destfile', 'name']), state=dict(type='str', default='present', choices=['absent', 'present']), marker=dict(type='str', default='# {mark} ANSIBLE MANAGED BLOCK'), block=dict(type='str', default='', aliases=['content']), insertafter=dict(type='str'), insertbefore=dict(type='str'), create=dict(type='bool', default=False), backup=dict(type='bool', default=False), validate=dict(type='str'), marker_begin=dict(type='str', default='BEGIN'), marker_end=dict(type='str', default='END'), append_newline=dict(type='bool', default=False), prepend_newline=dict(type='bool', default=False)), mutually_exclusive=[['insertbefore', 'insertafter']], add_file_common_args=True, supports_check_mode=True)
    params = module.params
    path = params['path']
    if os.path.isdir(path):
        module.fail_json(rc=256, msg='Path %s is a directory !' % path)
    path_exists = os.path.exists(path)
    if not path_exists:
        if not module.boolean(params['create']):
            module.fail_json(rc=257, msg='Path %s does not exist !' % path)
        destpath = os.path.dirname(path)
        if not os.path.exists(destpath) and (not module.check_mode):
            try:
                os.makedirs(destpath)
            except OSError as e:
                module.fail_json(msg='Error creating %s Error code: %s Error description: %s' % (destpath, e.errno, e.strerror))
            except Exception as e:
                module.fail_json(msg='Error creating %s Error: %s' % (destpath, to_native(e)))
        original = None
        lines = []
    else:
        with open(path, 'rb') as f:
            original = f.read()
        lines = original.splitlines(True)
    diff = {'before': '', 'after': '', 'before_header': '%s (content)' % path, 'after_header': '%s (content)' % path}
    if module._diff and original:
        diff['before'] = original
    insertbefore = params['insertbefore']
    insertafter = params['insertafter']
    block = to_bytes(params['block'])
    marker = to_bytes(params['marker'])
    present = params['state'] == 'present'
    blank_line = [b(os.linesep)]
    if not present and (not path_exists):
        module.exit_json(changed=False, msg='File %s not present' % path)
    if insertbefore is None and insertafter is None:
        insertafter = 'EOF'
    if insertafter not in (None, 'EOF'):
        insertre = re.compile(to_bytes(insertafter, errors='surrogate_or_strict'))
    elif insertbefore not in (None, 'BOF'):
        insertre = re.compile(to_bytes(insertbefore, errors='surrogate_or_strict'))
    else:
        insertre = None
    marker0 = re.sub(b('{mark}'), b(params['marker_begin']), marker) + b(os.linesep)
    marker1 = re.sub(b('{mark}'), b(params['marker_end']), marker) + b(os.linesep)
    if present and block:
        if not block.endswith(b(os.linesep)):
            block += b(os.linesep)
        blocklines = [marker0] + block.splitlines(True) + [marker1]
    else:
        blocklines = []
    n0 = n1 = None
    for (i, line) in enumerate(lines):
        if line == marker0:
            n0 = i
        if line == marker1:
            n1 = i
    if None in (n0, n1):
        n0 = None
        if insertre is not None:
            if insertre.flags & re.MULTILINE:
                match = insertre.search(original)
                if match:
                    if insertafter:
                        n0 = to_native(original).count('\n', 0, match.end())
                    elif insertbefore:
                        n0 = to_native(original).count('\n', 0, match.start())
            else:
                for (i, line) in enumerate(lines):
                    if insertre.search(line):
                        n0 = i
            if n0 is None:
                n0 = len(lines)
            elif insertafter is not None:
                n0 += 1
        elif insertbefore is not None:
            n0 = 0
        else:
            n0 = len(lines)
    elif n0 < n1:
        lines[n0:n1 + 1] = []
    else:
        lines[n1:n0 + 1] = []
        n0 = n1
    if n0 > 0:
        if not lines[n0 - 1].endswith(b(os.linesep)):
            lines[n0 - 1] += b(os.linesep)
    if params['prepend_newline'] and present:
        if n0 != 0 and lines[n0 - 1] != b(os.linesep):
            lines[n0:n0] = blank_line
            n0 += 1
    lines[n0:n0] = blocklines
    if params['append_newline'] and present:
        line_after_block = n0 + len(blocklines)
        if line_after_block < len(lines) and lines[line_after_block] != b(os.linesep):
            lines[line_after_block:line_after_block] = blank_line
    if lines:
        result = b''.join(lines)
    else:
        result = b''
    if module._diff:
        diff['after'] = result
    if original == result:
        msg = ''
        changed = False
    elif original is None:
        msg = 'File created'
        changed = True
    elif not blocklines:
        msg = 'Block removed'
        changed = True
    else:
        msg = 'Block inserted'
        changed = True
    backup_file = None
    if changed and (not module.check_mode):
        if module.boolean(params['backup']) and path_exists:
            backup_file = module.backup_local(path)
        real_path = os.path.realpath(params['path'])
        write_changes(module, result, real_path)
    if module.check_mode and (not path_exists):
        module.exit_json(changed=changed, msg=msg, diff=diff)
    attr_diff = {}
    (msg, changed) = check_file_attrs(module, changed, msg, attr_diff)
    attr_diff['before_header'] = '%s (file attributes)' % path
    attr_diff['after_header'] = '%s (file attributes)' % path
    difflist = [diff, attr_diff]
    if backup_file is None:
        module.exit_json(changed=changed, msg=msg, diff=difflist)
    else:
        module.exit_json(changed=changed, msg=msg, diff=difflist, backup_file=backup_file)
if __name__ == '__main__':
    main()