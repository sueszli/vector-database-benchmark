from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: rpm_key\nauthor:\n  - Hector Acosta (@hacosta) <hector.acosta@gazzang.com>\nshort_description: Adds or removes a gpg key from the rpm db\ndescription:\n  - Adds or removes (rpm --import) a gpg key to your rpm database.\nversion_added: "1.3"\noptions:\n    key:\n      description:\n        - Key that will be modified. Can be a url, a file on the managed node, or a keyid if the key\n          already exists in the database.\n      type: str\n      required: true\n    state:\n      description:\n        - If the key will be imported or removed from the rpm db.\n      type: str\n      default: present\n      choices: [ absent, present ]\n    validate_certs:\n      description:\n        - If V(false) and the O(key) is a url starting with V(https), SSL certificates will not be validated.\n        - This should only be used on personally controlled sites using self-signed certificates.\n      type: bool\n      default: \'yes\'\n    fingerprint:\n      description:\n        - The long-form fingerprint of the key being imported.\n        - This will be used to verify the specified key.\n      type: str\n      version_added: 2.9\nextends_documentation_fragment:\n    - action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    platform:\n        platforms: rhel\n'
EXAMPLES = '\n- name: Import a key from a url\n  ansible.builtin.rpm_key:\n    state: present\n    key: http://apt.sw.be/RPM-GPG-KEY.dag.txt\n\n- name: Import a key from a file\n  ansible.builtin.rpm_key:\n    state: present\n    key: /path/to/key.gpg\n\n- name: Ensure a key is not present in the db\n  ansible.builtin.rpm_key:\n    state: absent\n    key: DEADB33F\n\n- name: Verify the key, using a fingerprint, before import\n  ansible.builtin.rpm_key:\n    key: /path/to/RPM-GPG-KEY.dag.txt\n    fingerprint: EBC6 E12C 62B1 C734 026B  2122 A20E 5214 6B8D 79E6\n'
RETURN = '#'
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native

def is_pubkey(string):
    if False:
        print('Hello World!')
    'Verifies if string is a pubkey'
    pgp_regex = '.*?(-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----).*'
    return bool(re.match(pgp_regex, to_native(string, errors='surrogate_or_strict'), re.DOTALL))

class RpmKey(object):

    def __init__(self, module):
        if False:
            for i in range(10):
                print('nop')
        keyfile = None
        should_cleanup_keyfile = False
        self.module = module
        self.rpm = self.module.get_bin_path('rpm', True)
        state = module.params['state']
        key = module.params['key']
        fingerprint = module.params['fingerprint']
        if fingerprint:
            fingerprint = fingerprint.replace(' ', '').upper()
        self.gpg = self.module.get_bin_path('gpg')
        if not self.gpg:
            self.gpg = self.module.get_bin_path('gpg2', required=True)
        if '://' in key:
            keyfile = self.fetch_key(key)
            keyid = self.getkeyid(keyfile)
            should_cleanup_keyfile = True
        elif self.is_keyid(key):
            keyid = key
        elif os.path.isfile(key):
            keyfile = key
            keyid = self.getkeyid(keyfile)
        else:
            self.module.fail_json(msg='Not a valid key %s' % key)
        keyid = self.normalize_keyid(keyid)
        if state == 'present':
            if self.is_key_imported(keyid):
                module.exit_json(changed=False)
            else:
                if not keyfile:
                    self.module.fail_json(msg='When importing a key, a valid file must be given')
                if fingerprint:
                    has_fingerprint = self.getfingerprint(keyfile)
                    if fingerprint != has_fingerprint:
                        self.module.fail_json(msg="The specified fingerprint, '%s', does not match the key fingerprint '%s'" % (fingerprint, has_fingerprint))
                self.import_key(keyfile)
                if should_cleanup_keyfile:
                    self.module.cleanup(keyfile)
                module.exit_json(changed=True)
        elif self.is_key_imported(keyid):
            self.drop_key(keyid)
            module.exit_json(changed=True)
        else:
            module.exit_json(changed=False)

    def fetch_key(self, url):
        if False:
            i = 10
            return i + 15
        'Downloads a key from url, returns a valid path to a gpg key'
        (rsp, info) = fetch_url(self.module, url)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch key at %s , error was: %s' % (url, info['msg']))
        key = rsp.read()
        if not is_pubkey(key):
            self.module.fail_json(msg='Not a public key: %s' % url)
        (tmpfd, tmpname) = tempfile.mkstemp()
        self.module.add_cleanup_file(tmpname)
        tmpfile = os.fdopen(tmpfd, 'w+b')
        tmpfile.write(key)
        tmpfile.close()
        return tmpname

    def normalize_keyid(self, keyid):
        if False:
            print('Hello World!')
        "Ensure a keyid doesn't have a leading 0x, has leading or trailing whitespace, and make sure is uppercase"
        ret = keyid.strip().upper()
        if ret.startswith('0x'):
            return ret[2:]
        elif ret.startswith('0X'):
            return ret[2:]
        else:
            return ret

    def getkeyid(self, keyfile):
        if False:
            i = 10
            return i + 15
        (stdout, stderr) = self.execute_command([self.gpg, '--no-tty', '--batch', '--with-colons', '--fixed-list-mode', keyfile])
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('pub:'):
                return line.split(':')[4]
        self.module.fail_json(msg='Unexpected gpg output')

    def getfingerprint(self, keyfile):
        if False:
            return 10
        (stdout, stderr) = self.execute_command([self.gpg, '--no-tty', '--batch', '--with-colons', '--fixed-list-mode', '--with-fingerprint', keyfile])
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('fpr:'):
                return line.split(':')[9]
        self.module.fail_json(msg='Unexpected gpg output')

    def is_keyid(self, keystr):
        if False:
            while True:
                i = 10
        'Verifies if a key, as provided by the user is a keyid'
        return re.match('(0x)?[0-9a-f]{8}', keystr, flags=re.IGNORECASE)

    def execute_command(self, cmd):
        if False:
            while True:
                i = 10
        (rc, stdout, stderr) = self.module.run_command(cmd, use_unsafe_shell=True)
        if rc != 0:
            self.module.fail_json(msg=stderr)
        return (stdout, stderr)

    def is_key_imported(self, keyid):
        if False:
            i = 10
            return i + 15
        cmd = self.rpm + ' -q  gpg-pubkey'
        (rc, stdout, stderr) = self.module.run_command(cmd)
        if rc != 0:
            return False
        cmd += ' --qf "%{description}" | ' + self.gpg + ' --no-tty --batch --with-colons --fixed-list-mode -'
        (stdout, stderr) = self.execute_command(cmd)
        for line in stdout.splitlines():
            if keyid in line.split(':')[4]:
                return True
        return False

    def import_key(self, keyfile):
        if False:
            return 10
        if not self.module.check_mode:
            self.execute_command([self.rpm, '--import', keyfile])

    def drop_key(self, keyid):
        if False:
            for i in range(10):
                print('nop')
        if not self.module.check_mode:
            self.execute_command([self.rpm, '--erase', '--allmatches', 'gpg-pubkey-%s' % keyid[-8:].lower()])

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec=dict(state=dict(type='str', default='present', choices=['absent', 'present']), key=dict(type='str', required=True, no_log=False), fingerprint=dict(type='str'), validate_certs=dict(type='bool', default=True)), supports_check_mode=True)
    RpmKey(module)
if __name__ == '__main__':
    main()