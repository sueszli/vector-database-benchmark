from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: apt_key\nauthor:\n- Jayson Vantuyl (@jvantuyl)\nversion_added: "1.0"\nshort_description: Add or remove an apt key\ndescription:\n    - Add or remove an I(apt) key, optionally downloading it.\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    platform:\n        platforms: debian\nnotes:\n    - The apt-key command used by this module has been deprecated. See the L(Debian wiki,https://wiki.debian.org/DebianRepository/UseThirdParty) for details.\n      This module is kept for backwards compatibility for systems that still use apt-key as the main way to manage apt repository keys.\n    - As a sanity check, downloaded key id must match the one specified.\n    - "Use full fingerprint (40 characters) key ids to avoid key collisions.\n      To generate a full-fingerprint imported key: C(apt-key adv --list-public-keys --with-fingerprint --with-colons)."\n    - If you specify both the key id and the URL with O(state=present), the task can verify or add the key as needed.\n    - Adding a new key requires an apt cache update (e.g. using the M(ansible.builtin.apt) module\'s update_cache option).\nrequirements:\n    - gpg\nseealso:\n  - module: ansible.builtin.deb822_repository\noptions:\n    id:\n        description:\n            - The identifier of the key.\n            - Including this allows check mode to correctly report the changed state.\n            - If specifying a subkey\'s id be aware that apt-key does not understand how to remove keys via a subkey id.  Specify the primary key\'s id instead.\n            - This parameter is required when O(state) is set to V(absent).\n        type: str\n    data:\n        description:\n            - The keyfile contents to add to the keyring.\n        type: str\n    file:\n        description:\n            - The path to a keyfile on the remote server to add to the keyring.\n        type: path\n    keyring:\n        description:\n            - The full path to specific keyring file in C(/etc/apt/trusted.gpg.d/).\n        type: path\n        version_added: "1.3"\n    url:\n        description:\n            - The URL to retrieve key from.\n        type: str\n    keyserver:\n        description:\n            - The keyserver to retrieve key from.\n        type: str\n        version_added: "1.6"\n    state:\n        description:\n            - Ensures that the key is present (added) or absent (revoked).\n        type: str\n        choices: [ absent, present ]\n        default: present\n    validate_certs:\n        description:\n            - If V(false), SSL certificates for the target url will not be validated. This should only be used\n              on personally controlled sites using self-signed certificates.\n        type: bool\n        default: \'yes\'\n'
EXAMPLES = '\n- name: One way to avoid apt_key once it is removed from your distro, armored keys should use .asc extension, binary should use .gpg\n  block:\n    - name: somerepo | no apt key\n      ansible.builtin.get_url:\n        url: https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x36a1d7869245c8950f966e92d8576a8ba88d21e9\n        dest: /etc/apt/keyrings/myrepo.asc\n        checksum: sha256:bb42f0db45d46bab5f9ec619e1a47360b94c27142e57aa71f7050d08672309e0\n\n    - name: somerepo | apt source\n      ansible.builtin.apt_repository:\n        repo: "deb [arch=amd64 signed-by=/etc/apt/keyrings/myrepo.asc] https://download.example.com/linux/ubuntu {{ ansible_distribution_release }} stable"\n        state: present\n\n- name: Add an apt key by id from a keyserver\n  ansible.builtin.apt_key:\n    keyserver: keyserver.ubuntu.com\n    id: 36A1D7869245C8950F966E92D8576A8BA88D21E9\n\n- name: Add an Apt signing key, uses whichever key is at the URL\n  ansible.builtin.apt_key:\n    url: https://ftp-master.debian.org/keys/archive-key-6.0.asc\n    state: present\n\n- name: Add an Apt signing key, will not download if present\n  ansible.builtin.apt_key:\n    id: 9FED2BCBDCD29CDF762678CBAED4B06F473041FA\n    url: https://ftp-master.debian.org/keys/archive-key-6.0.asc\n    state: present\n\n- name: Remove a Apt specific signing key, leading 0x is valid\n  ansible.builtin.apt_key:\n    id: 0x9FED2BCBDCD29CDF762678CBAED4B06F473041FA\n    state: absent\n\n# Use armored file since utf-8 string is expected. Must be of "PGP PUBLIC KEY BLOCK" type.\n- name: Add a key from a file on the Ansible server\n  ansible.builtin.apt_key:\n    data: "{{ lookup(\'ansible.builtin.file\', \'apt.asc\') }}"\n    state: present\n\n- name: Add an Apt signing key to a specific keyring file\n  ansible.builtin.apt_key:\n    id: 9FED2BCBDCD29CDF762678CBAED4B06F473041FA\n    url: https://ftp-master.debian.org/keys/archive-key-6.0.asc\n    keyring: /etc/apt/trusted.gpg.d/debian.gpg\n\n- name: Add Apt signing key on remote server to keyring\n  ansible.builtin.apt_key:\n    id: 9FED2BCBDCD29CDF762678CBAED4B06F473041FA\n    file: /tmp/apt.gpg\n    state: present\n'
RETURN = '\nafter:\n    description: List of apt key ids or fingerprints after any modification\n    returned: on change\n    type: list\n    sample: ["D8576A8BA88D21E9", "3B4FE6ACC0B21F32", "D94AA3F0EFE21092", "871920D1991BC93C"]\nbefore:\n    description: List of apt key ids or fingprints before any modifications\n    returned: always\n    type: list\n    sample: ["3B4FE6ACC0B21F32", "D94AA3F0EFE21092", "871920D1991BC93C"]\nfp:\n    description: Fingerprint of the key to import\n    returned: always\n    type: str\n    sample: "D8576A8BA88D21E9"\nid:\n    description: key id from source\n    returned: always\n    type: str\n    sample: "36A1D7869245C8950F966E92D8576A8BA88D21E9"\nkey_id:\n    description: calculated key id, it should be same as \'id\', but can be different\n    returned: always\n    type: str\n    sample: "36A1D7869245C8950F966E92D8576A8BA88D21E9"\nshort_id:\n    description: calculated short key id\n    returned: always\n    type: str\n    sample: "A88D21E9"\n'
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
apt_key_bin = None
gpg_bin = None
locale = None

def lang_env(module):
    if False:
        while True:
            i = 10
    if not hasattr(lang_env, 'result'):
        locale = get_best_parsable_locale(module)
        lang_env.result = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
    return lang_env.result

def find_needed_binaries(module):
    if False:
        return 10
    global apt_key_bin
    global gpg_bin
    apt_key_bin = module.get_bin_path('apt-key', required=True)
    gpg_bin = module.get_bin_path('gpg', required=True)

def add_http_proxy(cmd):
    if False:
        print('Hello World!')
    for envvar in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy'):
        proxy = os.environ.get(envvar)
        if proxy:
            break
    if proxy:
        cmd += ' --keyserver-options http-proxy=%s' % proxy
    return cmd

def parse_key_id(key_id):
    if False:
        while True:
            i = 10
    'validate the key_id and break it into segments\n\n    :arg key_id: The key_id as supplied by the user.  A valid key_id will be\n        8, 16, or more hexadecimal chars with an optional leading ``0x``.\n    :returns: The portion of key_id suitable for apt-key del, the portion\n        suitable for comparisons with --list-public-keys, and the portion that\n        can be used with --recv-key.  If key_id is long enough, these will be\n        the last 8 characters of key_id, the last 16 characters, and all of\n        key_id.  If key_id is not long enough, some of the values will be the\n        same.\n\n    * apt-key del <= 1.10 has a bug with key_id != 8 chars\n    * apt-key adv --list-public-keys prints 16 chars\n    * apt-key adv --recv-key can take more chars\n\n    '
    int(to_native(key_id), 16)
    key_id = key_id.upper()
    if key_id.startswith('0X'):
        key_id = key_id[2:]
    key_id_len = len(key_id)
    if (key_id_len != 8 and key_id_len != 16) and key_id_len <= 16:
        raise ValueError('key_id must be 8, 16, or 16+ hexadecimal characters in length')
    short_key_id = key_id[-8:]
    fingerprint = key_id
    if key_id_len > 16:
        fingerprint = key_id[-16:]
    return (short_key_id, fingerprint, key_id)

def parse_output_for_keys(output, short_format=False):
    if False:
        for i in range(10):
            print('nop')
    found = []
    lines = to_native(output).split('\n')
    for line in lines:
        if (line.startswith('pub') or line.startswith('sub')) and 'expired' not in line:
            try:
                tokens = line.split()
                code = tokens[1]
                (len_type, real_code) = code.split('/')
            except (IndexError, ValueError):
                try:
                    tokens = line.split(':')
                    real_code = tokens[4]
                except (IndexError, ValueError):
                    continue
            found.append(real_code)
    if found and short_format:
        found = shorten_key_ids(found)
    return found

def all_keys(module, keyring, short_format):
    if False:
        return 10
    if keyring is not None:
        cmd = '%s --keyring %s adv --list-public-keys --keyid-format=long' % (apt_key_bin, keyring)
    else:
        cmd = '%s adv --list-public-keys --keyid-format=long' % apt_key_bin
    (rc, out, err) = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='Unable to list public keys', cmd=cmd, rc=rc, stdout=out, stderr=err)
    return parse_output_for_keys(out, short_format)

def shorten_key_ids(key_id_list):
    if False:
        print('Hello World!')
    "\n    Takes a list of key ids, and converts them to the 'short' format,\n    by reducing them to their last 8 characters.\n    "
    short = []
    for key in key_id_list:
        short.append(key[-8:])
    return short

def download_key(module, url):
    if False:
        i = 10
        return i + 15
    try:
        (rsp, info) = fetch_url(module, url, use_proxy=True)
        if info['status'] != 200:
            module.fail_json(msg='Failed to download key at %s: %s' % (url, info['msg']))
        return rsp.read()
    except Exception:
        module.fail_json(msg='error getting key id from url: %s' % url, traceback=format_exc())

def get_key_id_from_file(module, filename, data=None):
    if False:
        return 10
    native_data = to_native(data)
    is_armored = native_data.find('-----BEGIN PGP PUBLIC KEY BLOCK-----') >= 0
    key = None
    cmd = [gpg_bin, '--with-colons', filename]
    (rc, out, err) = module.run_command(cmd, environ_update=lang_env(module), data=native_data if is_armored else data, binary_data=not is_armored)
    if rc != 0:
        module.fail_json(msg="Unable to extract key from '%s'" % ('inline data' if data is not None else filename), stdout=out, stderr=err)
    keys = parse_output_for_keys(out)
    if keys:
        key = keys[0]
    return key

def get_key_id_from_data(module, data):
    if False:
        while True:
            i = 10
    return get_key_id_from_file(module, '-', data)

def import_key(module, keyring, keyserver, key_id):
    if False:
        while True:
            i = 10
    if keyring:
        cmd = '%s --keyring %s adv --no-tty --keyserver %s' % (apt_key_bin, keyring, keyserver)
    else:
        cmd = '%s adv --no-tty --keyserver %s' % (apt_key_bin, keyserver)
    cmd = add_http_proxy(cmd)
    cmd = '%s --recv %s' % (cmd, key_id)
    for retry in range(5):
        (rc, out, err) = module.run_command(cmd, environ_update=lang_env(module))
        if rc == 0:
            break
    else:
        if rc == 2 and 'not found on keyserver' in out:
            msg = 'Key %s not found on keyserver %s' % (key_id, keyserver)
            module.fail_json(cmd=cmd, msg=msg, forced_environment=lang_env(module))
        else:
            msg = 'Error fetching key %s from keyserver: %s' % (key_id, keyserver)
            module.fail_json(cmd=cmd, msg=msg, forced_environment=lang_env(module), rc=rc, stdout=out, stderr=err)
    return True

def add_key(module, keyfile, keyring, data=None):
    if False:
        return 10
    if data is not None:
        if keyring:
            cmd = '%s --keyring %s add -' % (apt_key_bin, keyring)
        else:
            cmd = '%s add -' % apt_key_bin
        (rc, out, err) = module.run_command(cmd, data=data, binary_data=True)
        if rc != 0:
            module.fail_json(msg='Unable to add a key from binary data', cmd=cmd, rc=rc, stdout=out, stderr=err)
    else:
        if keyring:
            cmd = '%s --keyring %s add %s' % (apt_key_bin, keyring, keyfile)
        else:
            cmd = '%s add %s' % (apt_key_bin, keyfile)
        (rc, out, err) = module.run_command(cmd)
        if rc != 0:
            module.fail_json(msg='Unable to add a key from file %s' % keyfile, cmd=cmd, rc=rc, keyfile=keyfile, stdout=out, stderr=err)
    return True

def remove_key(module, key_id, keyring):
    if False:
        print('Hello World!')
    if keyring:
        cmd = '%s --keyring %s del %s' % (apt_key_bin, keyring, key_id)
    else:
        cmd = '%s del %s' % (apt_key_bin, key_id)
    (rc, out, err) = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='Unable to remove a key with id %s' % key_id, cmd=cmd, rc=rc, key_id=key_id, stdout=out, stderr=err)
    return True

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec=dict(id=dict(type='str'), url=dict(type='str'), data=dict(type='str'), file=dict(type='path'), keyring=dict(type='path'), validate_certs=dict(type='bool', default=True), keyserver=dict(type='str'), state=dict(type='str', default='present', choices=['absent', 'present'])), supports_check_mode=True, mutually_exclusive=(('data', 'file', 'keyserver', 'url'),))
    key_id = module.params['id']
    url = module.params['url']
    data = module.params['data']
    filename = module.params['file']
    keyring = module.params['keyring']
    state = module.params['state']
    keyserver = module.params['keyserver']
    short_format = False
    short_key_id = None
    fingerprint = None
    error_no_error = 'apt-key did not return an error, but %s (check that the id is correct and *not* a subkey)'
    find_needed_binaries(module)
    r = {'changed': False}
    if not key_id:
        if keyserver:
            module.fail_json(msg='Missing key_id, required with keyserver.')
        if url:
            data = download_key(module, url)
        if filename:
            key_id = get_key_id_from_file(module, filename)
        elif data:
            key_id = get_key_id_from_data(module, data)
    r['id'] = key_id
    try:
        (short_key_id, fingerprint, key_id) = parse_key_id(key_id)
        r['short_id'] = short_key_id
        r['fp'] = fingerprint
        r['key_id'] = key_id
    except ValueError:
        module.fail_json(msg='Invalid key_id', **r)
    if not fingerprint:
        module.fail_json(msg='Unable to continue as we could not extract a valid fingerprint to compare against existing keys.', **r)
    if len(key_id) == 8:
        short_format = True
    r['before'] = keys = all_keys(module, keyring, short_format)
    keys2 = []
    if state == 'present':
        if short_format and short_key_id not in keys or (not short_format and fingerprint not in keys):
            r['changed'] = True
            if not module.check_mode:
                if filename:
                    add_key(module, filename, keyring)
                elif keyserver:
                    import_key(module, keyring, keyserver, key_id)
                elif data:
                    add_key(module, '-', keyring, data)
                elif url:
                    data = download_key(module, url)
                    add_key(module, '-', keyring, data)
                else:
                    module.fail_json(msg='No key to add ... how did i get here?!?!', **r)
                r['after'] = keys2 = all_keys(module, keyring, short_format)
                if short_format and short_key_id not in keys2 or (not short_format and fingerprint not in keys2):
                    module.fail_json(msg=error_no_error % 'failed to add the key', **r)
    elif state == 'absent':
        if not key_id:
            module.fail_json(msg='key is required to remove a key', **r)
        if fingerprint in keys:
            r['changed'] = True
            if not module.check_mode:
                if short_key_id is not None and remove_key(module, short_key_id, keyring):
                    r['after'] = keys2 = all_keys(module, keyring, short_format)
                    if fingerprint in keys2:
                        module.fail_json(msg=error_no_error % 'the key was not removed', **r)
                else:
                    module.fail_json(msg='error removing key_id', **r)
    module.exit_json(**r)
if __name__ == '__main__':
    main()