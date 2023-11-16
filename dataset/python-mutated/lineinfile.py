from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: lineinfile\nshort_description: Manage lines in text files\ndescription:\n  - This module ensures a particular line is in a file, or replace an\n    existing line using a back-referenced regular expression.\n  - This is primarily useful when you want to change a single line in a file only.\n  - See the M(ansible.builtin.replace) module if you want to change multiple, similar lines\n    or check M(ansible.builtin.blockinfile) if you want to insert/update/remove a block of lines in a file.\n    For other cases, see the M(ansible.builtin.copy) or M(ansible.builtin.template) modules.\nversion_added: "0.7"\noptions:\n  path:\n    description:\n      - The file to modify.\n      - Before Ansible 2.3 this option was only usable as O(dest), O(destfile) and O(name).\n    type: path\n    required: true\n    aliases: [ dest, destfile, name ]\n  regexp:\n    description:\n      - The regular expression to look for in every line of the file.\n      - For O(state=present), the pattern to replace if found. Only the last line found will be replaced.\n      - For O(state=absent), the pattern of the line(s) to remove.\n      - If the regular expression is not matched, the line will be\n        added to the file in keeping with O(insertbefore) or O(insertafter)\n        settings.\n      - When modifying a line the regexp should typically match both the initial state of\n        the line as well as its state after replacement by O(line) to ensure idempotence.\n      - Uses Python regular expressions. See U(https://docs.python.org/3/library/re.html).\n    type: str\n    aliases: [ regex ]\n    version_added: \'1.7\'\n  search_string:\n    description:\n      - The literal string to look for in every line of the file. This does not have to match the entire line.\n      - For O(state=present), the line to replace if the string is found in the file. Only the last line found will be replaced.\n      - For O(state=absent), the line(s) to remove if the string is in the line.\n      - If the literal expression is not matched, the line will be\n        added to the file in keeping with O(insertbefore) or O(insertafter)\n        settings.\n      - Mutually exclusive with O(backrefs) and O(regexp).\n    type: str\n    version_added: \'2.11\'\n  state:\n    description:\n      - Whether the line should be there or not.\n    type: str\n    choices: [ absent, present ]\n    default: present\n  line:\n    description:\n      - The line to insert/replace into the file.\n      - Required for O(state=present).\n      - If O(backrefs) is set, may contain backreferences that will get\n        expanded with the O(regexp) capture groups if the regexp matches.\n    type: str\n    aliases: [ value ]\n  backrefs:\n    description:\n      - Used with O(state=present).\n      - If set, O(line) can contain backreferences (both positional and named)\n        that will get populated if the O(regexp) matches.\n      - This parameter changes the operation of the module slightly;\n        O(insertbefore) and O(insertafter) will be ignored, and if the O(regexp)\n        does not match anywhere in the file, the file will be left unchanged.\n      - If the O(regexp) does match, the last matching line will be replaced by\n        the expanded line parameter.\n      - Mutually exclusive with O(search_string).\n    type: bool\n    default: no\n    version_added: "1.1"\n  insertafter:\n    description:\n      - Used with O(state=present).\n      - If specified, the line will be inserted after the last match of specified regular expression.\n      - If the first match is required, use(firstmatch=yes).\n      - A special value is available; V(EOF) for inserting the line at the end of the file.\n      - If specified regular expression has no matches, EOF will be used instead.\n      - If O(insertbefore) is set, default value V(EOF) will be ignored.\n      - If regular expressions are passed to both O(regexp) and O(insertafter), O(insertafter) is only honored if no match for O(regexp) is found.\n      - May not be used with O(backrefs) or O(insertbefore).\n    type: str\n    choices: [ EOF, \'*regex*\' ]\n    default: EOF\n  insertbefore:\n    description:\n      - Used with O(state=present).\n      - If specified, the line will be inserted before the last match of specified regular expression.\n      - If the first match is required, use O(firstmatch=yes).\n      - A value is available; V(BOF) for inserting the line at the beginning of the file.\n      - If specified regular expression has no matches, the line will be inserted at the end of the file.\n      - If regular expressions are passed to both O(regexp) and O(insertbefore), O(insertbefore) is only honored if no match for O(regexp) is found.\n      - May not be used with O(backrefs) or O(insertafter).\n    type: str\n    choices: [ BOF, \'*regex*\' ]\n    version_added: "1.1"\n  create:\n    description:\n      - Used with O(state=present).\n      - If specified, the file will be created if it does not already exist.\n      - By default it will fail if the file is missing.\n    type: bool\n    default: no\n  backup:\n    description:\n      - Create a backup file including the timestamp information so you can\n        get the original file back if you somehow clobbered it incorrectly.\n    type: bool\n    default: no\n  firstmatch:\n    description:\n      - Used with O(insertafter) or O(insertbefore).\n      - If set, O(insertafter) and O(insertbefore) will work with the first line that matches the given regular expression.\n    type: bool\n    default: no\n    version_added: "2.5"\n  others:\n    description:\n      - All arguments accepted by the M(ansible.builtin.file) module also work here.\n    type: str\nextends_documentation_fragment:\n    - action_common_attributes\n    - action_common_attributes.files\n    - files\n    - validate\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        platforms: posix\n    safe_file_operations:\n        support: full\n    vault:\n        support: none\nnotes:\n  - As of Ansible 2.3, the O(dest) option has been changed to O(path) as default, but O(dest) still works as well.\nseealso:\n- module: ansible.builtin.blockinfile\n- module: ansible.builtin.copy\n- module: ansible.builtin.file\n- module: ansible.builtin.replace\n- module: ansible.builtin.template\n- module: community.windows.win_lineinfile\nauthor:\n    - Daniel Hokka Zakrissoni (@dhozac)\n    - Ahti Kitsik (@ahtik)\n    - Jose Angel Munoz (@imjoseangel)\n'
EXAMPLES = '\n# NOTE: Before 2.3, option \'dest\', \'destfile\' or \'name\' was used instead of \'path\'\n- name: Ensure SELinux is set to enforcing mode\n  ansible.builtin.lineinfile:\n    path: /etc/selinux/config\n    regexp: \'^SELINUX=\'\n    line: SELINUX=enforcing\n\n- name: Make sure group wheel is not in the sudoers configuration\n  ansible.builtin.lineinfile:\n    path: /etc/sudoers\n    state: absent\n    regexp: \'^%wheel\'\n\n- name: Replace a localhost entry with our own\n  ansible.builtin.lineinfile:\n    path: /etc/hosts\n    regexp: \'^127\\.0\\.0\\.1\'\n    line: 127.0.0.1 localhost\n    owner: root\n    group: root\n    mode: \'0644\'\n\n- name: Replace a localhost entry searching for a literal string to avoid escaping\n  ansible.builtin.lineinfile:\n    path: /etc/hosts\n    search_string: \'127.0.0.1\'\n    line: 127.0.0.1 localhost\n    owner: root\n    group: root\n    mode: \'0644\'\n\n- name: Ensure the default Apache port is 8080\n  ansible.builtin.lineinfile:\n    path: /etc/httpd/conf/httpd.conf\n    regexp: \'^Listen \'\n    insertafter: \'^#Listen \'\n    line: Listen 8080\n\n- name: Ensure php extension matches new pattern\n  ansible.builtin.lineinfile:\n    path: /etc/httpd/conf/httpd.conf\n    search_string: \'<FilesMatch ".php[45]?$">\'\n    insertafter: \'^\\t<Location \\/>\\n\'\n    line: \'        <FilesMatch ".php[34]?$">\'\n\n- name: Ensure we have our own comment added to /etc/services\n  ansible.builtin.lineinfile:\n    path: /etc/services\n    regexp: \'^# port for http\'\n    insertbefore: \'^www.*80/tcp\'\n    line: \'# port for http by default\'\n\n- name: Add a line to a file if the file does not exist, without passing regexp\n  ansible.builtin.lineinfile:\n    path: /tmp/testfile\n    line: 192.168.1.99 foo.lab.net foo\n    create: yes\n\n# NOTE: Yaml requires escaping backslashes in double quotes but not in single quotes\n- name: Ensure the JBoss memory settings are exactly as needed\n  ansible.builtin.lineinfile:\n    path: /opt/jboss-as/bin/standalone.conf\n    regexp: \'^(.*)Xms(\\d+)m(.*)$\'\n    line: \'\\1Xms${xms}m\\3\'\n    backrefs: yes\n\n# NOTE: Fully quoted because of the \': \' on the line. See the Gotchas in the YAML docs.\n- name: Validate the sudoers file before saving\n  ansible.builtin.lineinfile:\n    path: /etc/sudoers\n    state: present\n    regexp: \'^%ADMIN ALL=\'\n    line: \'%ADMIN ALL=(ALL) NOPASSWD: ALL\'\n    validate: /usr/sbin/visudo -cf %s\n\n# See https://docs.python.org/3/library/re.html for further details on syntax\n- name: Use backrefs with alternative group syntax to avoid conflicts with variable values\n  ansible.builtin.lineinfile:\n    path: /tmp/config\n    regexp: ^(host=).*\n    line: \\g<1>{{ hostname }}\n    backrefs: yes\n'
RETURN = '#'
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text

def write_changes(module, b_lines, dest):
    if False:
        return 10
    (tmpfd, tmpfile) = tempfile.mkstemp(dir=module.tmpdir)
    with os.fdopen(tmpfd, 'wb') as f:
        f.writelines(b_lines)
    validate = module.params.get('validate', None)
    valid = not validate
    if validate:
        if '%s' not in validate:
            module.fail_json(msg='validate must contain %%s: %s' % validate)
        (rc, out, err) = module.run_command(to_bytes(validate % tmpfile, errors='surrogate_or_strict'))
        valid = rc == 0
        if rc != 0:
            module.fail_json(msg='failed to validate: rc:%s error:%s' % (rc, err))
    if valid:
        module.atomic_move(tmpfile, to_native(os.path.realpath(to_bytes(dest, errors='surrogate_or_strict')), errors='surrogate_or_strict'), unsafe_writes=module.params['unsafe_writes'])

def check_file_attrs(module, changed, message, diff):
    if False:
        return 10
    file_args = module.load_file_common_arguments(module.params)
    if module.set_fs_attributes_if_different(file_args, False, diff=diff):
        if changed:
            message += ' and '
        changed = True
        message += 'ownership, perms or SE linux context changed'
    return (message, changed)

def present(module, dest, regexp, search_string, line, insertafter, insertbefore, create, backup, backrefs, firstmatch):
    if False:
        i = 10
        return i + 15
    diff = {'before': '', 'after': '', 'before_header': '%s (content)' % dest, 'after_header': '%s (content)' % dest}
    b_dest = to_bytes(dest, errors='surrogate_or_strict')
    if not os.path.exists(b_dest):
        if not create:
            module.fail_json(rc=257, msg='Destination %s does not exist !' % dest)
        b_destpath = os.path.dirname(b_dest)
        if b_destpath and (not os.path.exists(b_destpath)) and (not module.check_mode):
            try:
                os.makedirs(b_destpath)
            except Exception as e:
                module.fail_json(msg='Error creating %s (%s)' % (to_text(b_destpath), to_text(e)))
        b_lines = []
    else:
        with open(b_dest, 'rb') as f:
            b_lines = f.readlines()
    if module._diff:
        diff['before'] = to_native(b''.join(b_lines))
    if regexp is not None:
        bre_m = re.compile(to_bytes(regexp, errors='surrogate_or_strict'))
    if insertafter not in (None, 'BOF', 'EOF'):
        bre_ins = re.compile(to_bytes(insertafter, errors='surrogate_or_strict'))
    elif insertbefore not in (None, 'BOF'):
        bre_ins = re.compile(to_bytes(insertbefore, errors='surrogate_or_strict'))
    else:
        bre_ins = None
    index = [-1, -1]
    match = None
    exact_line_match = False
    b_line = to_bytes(line, errors='surrogate_or_strict')
    if regexp is not None:
        for (lineno, b_cur_line) in enumerate(b_lines):
            match_found = bre_m.search(b_cur_line)
            if match_found:
                index[0] = lineno
                match = match_found
                if firstmatch:
                    break
    if search_string is not None:
        for (lineno, b_cur_line) in enumerate(b_lines):
            match_found = to_bytes(search_string, errors='surrogate_or_strict') in b_cur_line
            if match_found:
                index[0] = lineno
                match = match_found
                if firstmatch:
                    break
    if not match:
        for (lineno, b_cur_line) in enumerate(b_lines):
            if b_line == b_cur_line.rstrip(b'\r\n'):
                index[0] = lineno
                exact_line_match = True
            elif bre_ins is not None and bre_ins.search(b_cur_line):
                if insertafter:
                    index[1] = lineno + 1
                    if firstmatch:
                        break
                if insertbefore:
                    index[1] = lineno
                    if firstmatch:
                        break
    msg = ''
    changed = False
    b_linesep = to_bytes(os.linesep, errors='surrogate_or_strict')
    if index[0] != -1:
        if backrefs and match:
            b_new_line = match.expand(b_line)
        else:
            b_new_line = b_line
        if not b_new_line.endswith(b_linesep):
            b_new_line += b_linesep
        if (regexp, search_string, match) == (None, None, None) and (not exact_line_match):
            if insertafter and insertafter != 'EOF':
                if b_lines and (not b_lines[-1][-1:] in (b'\n', b'\r')):
                    b_lines[-1] = b_lines[-1] + b_linesep
                if len(b_lines) == index[1]:
                    if b_lines[index[1] - 1].rstrip(b'\r\n') != b_line:
                        b_lines.append(b_line + b_linesep)
                        msg = 'line added'
                        changed = True
                elif b_lines[index[1]].rstrip(b'\r\n') != b_line:
                    b_lines.insert(index[1], b_line + b_linesep)
                    msg = 'line added'
                    changed = True
            elif insertbefore and insertbefore != 'BOF':
                if index[1] <= 0:
                    if b_lines[index[1]].rstrip(b'\r\n') != b_line:
                        b_lines.insert(index[1], b_line + b_linesep)
                        msg = 'line added'
                        changed = True
                elif b_lines[index[1] - 1].rstrip(b'\r\n') != b_line:
                    b_lines.insert(index[1], b_line + b_linesep)
                    msg = 'line added'
                    changed = True
        elif b_lines[index[0]] != b_new_line:
            b_lines[index[0]] = b_new_line
            msg = 'line replaced'
            changed = True
    elif backrefs:
        pass
    elif insertbefore == 'BOF' or insertafter == 'BOF':
        b_lines.insert(0, b_line + b_linesep)
        msg = 'line added'
        changed = True
    elif insertafter == 'EOF' or index[1] == -1:
        if b_lines and (not b_lines[-1][-1:] in (b'\n', b'\r')):
            b_lines.append(b_linesep)
        b_lines.append(b_line + b_linesep)
        msg = 'line added'
        changed = True
    elif insertafter and index[1] != -1:
        if len(b_lines) == index[1]:
            if b_lines[index[1] - 1].rstrip(b'\r\n') != b_line:
                b_lines.append(b_line + b_linesep)
                msg = 'line added'
                changed = True
        elif b_line != b_lines[index[1]].rstrip(b'\n\r'):
            b_lines.insert(index[1], b_line + b_linesep)
            msg = 'line added'
            changed = True
    else:
        b_lines.insert(index[1], b_line + b_linesep)
        msg = 'line added'
        changed = True
    if module._diff:
        diff['after'] = to_native(b''.join(b_lines))
    backupdest = ''
    if changed and (not module.check_mode):
        if backup and os.path.exists(b_dest):
            backupdest = module.backup_local(dest)
        write_changes(module, b_lines, dest)
    if module.check_mode and (not os.path.exists(b_dest)):
        module.exit_json(changed=changed, msg=msg, backup=backupdest, diff=diff)
    attr_diff = {}
    (msg, changed) = check_file_attrs(module, changed, msg, attr_diff)
    attr_diff['before_header'] = '%s (file attributes)' % dest
    attr_diff['after_header'] = '%s (file attributes)' % dest
    difflist = [diff, attr_diff]
    module.exit_json(changed=changed, msg=msg, backup=backupdest, diff=difflist)

def absent(module, dest, regexp, search_string, line, backup):
    if False:
        for i in range(10):
            print('nop')
    b_dest = to_bytes(dest, errors='surrogate_or_strict')
    if not os.path.exists(b_dest):
        module.exit_json(changed=False, msg='file not present')
    msg = ''
    diff = {'before': '', 'after': '', 'before_header': '%s (content)' % dest, 'after_header': '%s (content)' % dest}
    with open(b_dest, 'rb') as f:
        b_lines = f.readlines()
    if module._diff:
        diff['before'] = to_native(b''.join(b_lines))
    if regexp is not None:
        bre_c = re.compile(to_bytes(regexp, errors='surrogate_or_strict'))
    found = []
    b_line = to_bytes(line, errors='surrogate_or_strict')

    def matcher(b_cur_line):
        if False:
            while True:
                i = 10
        if regexp is not None:
            match_found = bre_c.search(b_cur_line)
        elif search_string is not None:
            match_found = to_bytes(search_string, errors='surrogate_or_strict') in b_cur_line
        else:
            match_found = b_line == b_cur_line.rstrip(b'\r\n')
        if match_found:
            found.append(b_cur_line)
        return not match_found
    b_lines = [l for l in b_lines if matcher(l)]
    changed = len(found) > 0
    if module._diff:
        diff['after'] = to_native(b''.join(b_lines))
    backupdest = ''
    if changed and (not module.check_mode):
        if backup:
            backupdest = module.backup_local(dest)
        write_changes(module, b_lines, dest)
    if changed:
        msg = '%s line(s) removed' % len(found)
    attr_diff = {}
    (msg, changed) = check_file_attrs(module, changed, msg, attr_diff)
    attr_diff['before_header'] = '%s (file attributes)' % dest
    attr_diff['after_header'] = '%s (file attributes)' % dest
    difflist = [diff, attr_diff]
    module.exit_json(changed=changed, found=len(found), msg=msg, backup=backupdest, diff=difflist)

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(path=dict(type='path', required=True, aliases=['dest', 'destfile', 'name']), state=dict(type='str', default='present', choices=['absent', 'present']), regexp=dict(type='str', aliases=['regex']), search_string=dict(type='str'), line=dict(type='str', aliases=['value']), insertafter=dict(type='str'), insertbefore=dict(type='str'), backrefs=dict(type='bool', default=False), create=dict(type='bool', default=False), backup=dict(type='bool', default=False), firstmatch=dict(type='bool', default=False), validate=dict(type='str')), mutually_exclusive=[['insertbefore', 'insertafter'], ['regexp', 'search_string'], ['backrefs', 'search_string']], add_file_common_args=True, supports_check_mode=True)
    params = module.params
    create = params['create']
    backup = params['backup']
    backrefs = params['backrefs']
    path = params['path']
    firstmatch = params['firstmatch']
    regexp = params['regexp']
    search_string = params['search_string']
    line = params['line']
    if '' in [regexp, search_string]:
        msg = 'The %s is an empty string, which will match every line in the file. This may have unintended consequences, such as replacing the last line in the file rather than appending.'
        param_name = 'search string'
        if regexp == '':
            param_name = 'regular expression'
            msg += " If this is desired, use '^' to match every line in the file and avoid this warning."
        module.warn(msg % param_name)
    b_path = to_bytes(path, errors='surrogate_or_strict')
    if os.path.isdir(b_path):
        module.fail_json(rc=256, msg='Path %s is a directory !' % path)
    if params['state'] == 'present':
        if backrefs and regexp is None:
            module.fail_json(msg='regexp is required with backrefs=true')
        if line is None:
            module.fail_json(msg='line is required with state=present')
        (ins_bef, ins_aft) = (params['insertbefore'], params['insertafter'])
        if ins_bef is None and ins_aft is None:
            ins_aft = 'EOF'
        present(module, path, regexp, search_string, line, ins_aft, ins_bef, create, backup, backrefs, firstmatch)
    else:
        if (regexp, search_string, line) == (None, None, None):
            module.fail_json(msg='one of line, search_string, or regexp is required with state=absent')
        absent(module, path, regexp, search_string, line, backup)
if __name__ == '__main__':
    main()