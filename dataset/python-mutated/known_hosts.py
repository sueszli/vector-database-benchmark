from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: known_hosts\nshort_description: Add or remove a host from the C(known_hosts) file\ndescription:\n   - The M(ansible.builtin.known_hosts) module lets you add or remove a host keys from the C(known_hosts) file.\n   - Starting at Ansible 2.2, multiple entries per host are allowed, but only one for each key type supported by ssh.\n     This is useful if you\'re going to want to use the M(ansible.builtin.git) module over ssh, for example.\n   - If you have a very large number of host keys to manage, you will find the M(ansible.builtin.template) module more useful.\nversion_added: "1.9"\noptions:\n  name:\n    aliases: [ \'host\' ]\n    description:\n      - The host to add or remove (must match a host specified in key). It will be converted to lowercase so that ssh-keygen can find it.\n      - Must match with <hostname> or <ip> present in key attribute.\n      - For custom SSH port, O(name) needs to specify port as well. See example section.\n    type: str\n    required: true\n  key:\n    description:\n      - The SSH public host key, as a string.\n      - Required if O(state=present), optional when O(state=absent), in which case all keys for the host are removed.\n      - The key must be in the right format for SSH (see sshd(8), section "SSH_KNOWN_HOSTS FILE FORMAT").\n      - Specifically, the key should not match the format that is found in an SSH pubkey file, but should rather have the hostname prepended to a\n        line that includes the pubkey, the same way that it would appear in the known_hosts file. The value prepended to the line must also match\n        the value of the name parameter.\n      - Should be of format C(<hostname[,IP]> ssh-rsa <pubkey>).\n      - For custom SSH port, O(key) needs to specify port as well. See example section.\n    type: str\n  path:\n    description:\n      - The known_hosts file to edit.\n      - The known_hosts file will be created if needed. The rest of the path must exist prior to running the module.\n    default: "~/.ssh/known_hosts"\n    type: path\n  hash_host:\n    description:\n      - Hash the hostname in the known_hosts file.\n    type: bool\n    default: "no"\n    version_added: "2.3"\n  state:\n    description:\n      - V(present) to add the host key.\n      - V(absent) to remove it.\n    choices: [ "absent", "present" ]\n    default: "present"\n    type: str\nattributes:\n  check_mode:\n    support: full\n  diff_mode:\n    support: full\n  platform:\n    platforms: posix\nextends_documentation_fragment:\n  - action_common_attributes\nauthor:\n- Matthew Vernon (@mcv21)\n'
EXAMPLES = '\n- name: Tell the host about our servers it might want to ssh to\n  ansible.builtin.known_hosts:\n    path: /etc/ssh/ssh_known_hosts\n    name: foo.com.invalid\n    key: "{{ lookup(\'ansible.builtin.file\', \'pubkeys/foo.com.invalid\') }}"\n\n- name: Another way to call known_hosts\n  ansible.builtin.known_hosts:\n    name: host1.example.com   # or 10.9.8.77\n    key: host1.example.com,10.9.8.77 ssh-rsa ASDeararAIUHI324324  # some key gibberish\n    path: /etc/ssh/ssh_known_hosts\n    state: present\n\n- name: Add host with custom SSH port\n  ansible.builtin.known_hosts:\n    name: \'[host1.example.com]:2222\'\n    key: \'[host1.example.com]:2222 ssh-rsa ASDeararAIUHI324324\' # some key gibberish\n    path: /etc/ssh/ssh_known_hosts\n    state: present\n'
import base64
import errno
import hashlib
import hmac
import os
import os.path
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native

def enforce_state(module, params):
    if False:
        print('Hello World!')
    '\n    Add or remove key.\n    '
    host = params['name'].lower()
    key = params.get('key', None)
    path = params.get('path')
    hash_host = params.get('hash_host')
    state = params.get('state')
    sshkeygen = module.get_bin_path('ssh-keygen', True)
    if not key and state != 'absent':
        module.fail_json(msg='No key specified when adding a host')
    if key and hash_host:
        key = hash_host_key(host, key)
    if key and (not key.endswith('\n')):
        key += '\n'
    sanity_check(module, host, key, sshkeygen)
    (found, replace_or_add, found_line) = search_for_host_key(module, host, key, path, sshkeygen)
    params['diff'] = compute_diff(path, found_line, replace_or_add, state, key)
    if state == 'absent' and (not found_line) and key:
        params['changed'] = False
        return params
    if module.check_mode:
        module.exit_json(changed=replace_or_add or (state == 'present') != found, diff=params['diff'])
    if found and (not key) and (state == 'absent'):
        module.run_command([sshkeygen, '-R', host, '-f', path], check_rc=True)
        params['changed'] = True
    if replace_or_add or found != (state == 'present'):
        try:
            inf = open(path, 'r')
        except IOError as e:
            if e.errno == errno.ENOENT:
                inf = None
            else:
                module.fail_json(msg='Failed to read %s: %s' % (path, str(e)))
        try:
            with tempfile.NamedTemporaryFile(mode='w+', dir=os.path.dirname(path), delete=False) as outf:
                if inf is not None:
                    for (line_number, line) in enumerate(inf):
                        if found_line == line_number + 1 and (replace_or_add or state == 'absent'):
                            continue
                        outf.write(line)
                    inf.close()
                if state == 'present':
                    outf.write(key)
        except (IOError, OSError) as e:
            module.fail_json(msg='Failed to write to file %s: %s' % (path, to_native(e)))
        else:
            module.atomic_move(outf.name, path)
        params['changed'] = True
    return params

def sanity_check(module, host, key, sshkeygen):
    if False:
        while True:
            i = 10
    'Check supplied key is sensible\n\n    host and key are parameters provided by the user; If the host\n    provided is inconsistent with the key supplied, then this function\n    quits, providing an error to the user.\n    sshkeygen is the path to ssh-keygen, found earlier with get_bin_path\n    '
    if not key:
        return
    if re.search('\\S+(\\s+)?,(\\s+)?', host):
        module.fail_json(msg='Comma separated list of names is not supported. Please pass a single name to lookup in the known_hosts file.')
    with tempfile.NamedTemporaryFile(mode='w+') as outf:
        try:
            outf.write(key)
            outf.flush()
        except IOError as e:
            module.fail_json(msg='Failed to write to temporary file %s: %s' % (outf.name, to_native(e)))
        sshkeygen_command = [sshkeygen, '-F', host, '-f', outf.name]
        (rc, stdout, stderr) = module.run_command(sshkeygen_command)
    if stdout == '':
        module.fail_json(msg='Host parameter does not match hashed host field in supplied key')

def search_for_host_key(module, host, key, path, sshkeygen):
    if False:
        for i in range(10):
            print('nop')
    "search_for_host_key(module,host,key,path,sshkeygen) -> (found,replace_or_add,found_line)\n\n    Looks up host and keytype in the known_hosts file path; if it's there, looks to see\n    if one of those entries matches key. Returns:\n    found (Boolean): is host found in path?\n    replace_or_add (Boolean): is the key in path different to that supplied by user?\n    found_line (int or None): the line where a key of the same type was found\n    if found=False, then replace is always False.\n    sshkeygen is the path to ssh-keygen, found earlier with get_bin_path\n    "
    if os.path.exists(path) is False:
        return (False, False, None)
    sshkeygen_command = [sshkeygen, '-F', host, '-f', path]
    (rc, stdout, stderr) = module.run_command(sshkeygen_command, check_rc=False)
    if stdout == '' and stderr == '' and (rc == 0 or rc == 1):
        return (False, False, None)
    if rc != 0:
        module.fail_json(msg="ssh-keygen failed (rc=%d, stdout='%s',stderr='%s')" % (rc, stdout, stderr))
    if not key:
        return (True, False, None)
    lines = stdout.split('\n')
    new_key = normalize_known_hosts_key(key)
    for (lnum, l) in enumerate(lines):
        if l == '':
            continue
        elif l[0] == '#':
            try:
                found_line = int(re.search('found: line (\\d+)', l).group(1))
            except IndexError:
                module.fail_json(msg="failed to parse output of ssh-keygen for line number: '%s'" % l)
        else:
            found_key = normalize_known_hosts_key(l)
            if new_key['host'][:3] == '|1|' and found_key['host'][:3] == '|1|':
                new_key['host'] = found_key['host']
            if new_key == found_key:
                return (True, False, found_line)
            elif new_key['type'] == found_key['type']:
                return (True, True, found_line)
    return (True, True, None)

def hash_host_key(host, key):
    if False:
        return 10
    hmac_key = os.urandom(20)
    hashed_host = hmac.new(hmac_key, to_bytes(host), hashlib.sha1).digest()
    parts = key.strip().split()
    i = 1 if parts[0][0] == '@' else 0
    parts[i] = '|1|%s|%s' % (to_native(base64.b64encode(hmac_key)), to_native(base64.b64encode(hashed_host)))
    return ' '.join(parts)

def normalize_known_hosts_key(key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform a key, either taken from a known_host file or provided by the\n    user, into a normalized form.\n    The host part (which might include multiple hostnames or be hashed) gets\n    replaced by the provided host. Also, any spurious information gets removed\n    from the end (like the username@host tag usually present in hostkeys, but\n    absent in known_hosts files)\n    '
    key = key.strip()
    k = key.split()
    d = dict()
    if k[0][0] == '@':
        d['options'] = k[0]
        d['host'] = k[1]
        d['type'] = k[2]
        d['key'] = k[3]
    else:
        d['host'] = k[0]
        d['type'] = k[1]
        d['key'] = k[2]
    return d

def compute_diff(path, found_line, replace_or_add, state, key):
    if False:
        for i in range(10):
            print('nop')
    diff = {'before_header': path, 'after_header': path, 'before': '', 'after': ''}
    try:
        inf = open(path, 'r')
    except IOError as e:
        if e.errno == errno.ENOENT:
            diff['before_header'] = '/dev/null'
    else:
        diff['before'] = inf.read()
        inf.close()
    lines = diff['before'].splitlines(1)
    if (replace_or_add or state == 'absent') and found_line is not None and (1 <= found_line <= len(lines)):
        del lines[found_line - 1]
    if state == 'present' and (replace_or_add or found_line is None):
        lines.append(key)
    diff['after'] = ''.join(lines)
    return diff

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(name=dict(required=True, type='str', aliases=['host']), key=dict(required=False, type='str', no_log=False), path=dict(default='~/.ssh/known_hosts', type='path'), hash_host=dict(required=False, type='bool', default=False), state=dict(default='present', choices=['absent', 'present'])), supports_check_mode=True)
    results = enforce_state(module, module.params)
    module.exit_json(**results)
if __name__ == '__main__':
    main()