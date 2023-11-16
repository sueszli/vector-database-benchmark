from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: subversion\nshort_description: Deploys a subversion repository\ndescription:\n   - Deploy given repository URL / revision to dest. If dest exists, update to the specified revision, otherwise perform a checkout.\nversion_added: "0.7"\nauthor:\n- Dane Summers (@dsummersl) <njharman@gmail.com>\noptions:\n  repo:\n    description:\n      - The subversion URL to the repository.\n    type: str\n    required: true\n    aliases: [ name, repository ]\n  dest:\n    description:\n      - Absolute path where the repository should be deployed.\n      - The destination directory must be specified unless O(checkout=no), O(update=no), and O(export=no).\n    type: path\n  revision:\n    description:\n      - Specific revision to checkout.\n    type: str\n    default: HEAD\n    aliases: [ rev, version ]\n  force:\n    description:\n      - If V(true), modified files will be discarded. If V(false), module will fail if it encounters modified files.\n        Prior to 1.9 the default was V(true).\n    type: bool\n    default: "no"\n  in_place:\n    description:\n      - If the directory exists, then the working copy will be checked-out over-the-top using\n        svn checkout --force; if force is specified then existing files with different content are reverted.\n    type: bool\n    default: "no"\n    version_added: "2.6"\n  username:\n    description:\n      - C(--username) parameter passed to svn.\n    type: str\n  password:\n    description:\n      - C(--password) parameter passed to svn when svn is less than version 1.10.0. This is not secure and\n        the password will be leaked to argv.\n      - C(--password-from-stdin) parameter when svn is greater or equal to version 1.10.0.\n    type: str\n  executable:\n    description:\n      - Path to svn executable to use. If not supplied,\n        the normal mechanism for resolving binary paths will be used.\n    type: path\n    version_added: "1.4"\n  checkout:\n    description:\n     - If V(false), do not check out the repository if it does not exist locally.\n    type: bool\n    default: "yes"\n    version_added: "2.3"\n  update:\n    description:\n     - If V(false), do not retrieve new revisions from the origin repository.\n    type: bool\n    default: "yes"\n    version_added: "2.3"\n  export:\n    description:\n      - If V(true), do export instead of checkout/update.\n    type: bool\n    default: "no"\n    version_added: "1.6"\n  switch:\n    description:\n      - If V(false), do not call svn switch before update.\n    default: "yes"\n    version_added: "2.0"\n    type: bool\n  validate_certs:\n    description:\n      - If V(false), passes the C(--trust-server-cert) flag to svn.\n      - If V(true), does not pass the flag.\n    default: "no"\n    version_added: "2.11"\n    type: bool\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    platform:\n        platforms: posix\nnotes:\n   - This module does not handle externals.\n\nrequirements:\n    - subversion (the command line tool with C(svn) entrypoint)\n'
EXAMPLES = '\n- name: Checkout subversion repository to specified folder\n  ansible.builtin.subversion:\n    repo: svn+ssh://an.example.org/path/to/repo\n    dest: /src/checkout\n\n- name: Export subversion directory to folder\n  ansible.builtin.subversion:\n    repo: svn+ssh://an.example.org/path/to/repo\n    dest: /src/export\n    export: yes\n\n- name: Get information about the repository whether or not it has already been cloned locally\n  ansible.builtin.subversion:\n    repo: svn+ssh://an.example.org/path/to/repo\n    dest: /src/checkout\n    checkout: no\n    update: no\n'
RETURN = '#'
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.compat.version import LooseVersion

class Subversion(object):
    REVISION_RE = '^\\w+\\s?:\\s+\\d+$'

    def __init__(self, module, dest, repo, revision, username, password, svn_path, validate_certs):
        if False:
            while True:
                i = 10
        self.module = module
        self.dest = dest
        self.repo = repo
        self.revision = revision
        self.username = username
        self.password = password
        self.svn_path = svn_path
        self.validate_certs = validate_certs

    def has_option_password_from_stdin(self):
        if False:
            return 10
        (rc, version, err) = self.module.run_command([self.svn_path, '--version', '--quiet'], check_rc=True)
        return LooseVersion(version) >= LooseVersion('1.10.0')

    def _exec(self, args, check_rc=True):
        if False:
            print('Hello World!')
        'Execute a subversion command, and return output. If check_rc is False, returns the return code instead of the output.'
        bits = [self.svn_path, '--non-interactive', '--no-auth-cache']
        if not self.validate_certs:
            bits.append('--trust-server-cert')
        stdin_data = None
        if self.username:
            bits.extend(['--username', self.username])
        if self.password:
            if self.has_option_password_from_stdin():
                bits.append('--password-from-stdin')
                stdin_data = self.password
            else:
                self.module.warn('The authentication provided will be used on the svn command line and is not secure. To securely pass credentials, upgrade svn to version 1.10.0 or greater.')
                bits.extend(['--password', self.password])
        bits.extend(args)
        (rc, out, err) = self.module.run_command(bits, check_rc, data=stdin_data)
        if check_rc:
            return out.splitlines()
        else:
            return rc

    def is_svn_repo(self):
        if False:
            i = 10
            return i + 15
        'Checks if path is a SVN Repo.'
        rc = self._exec(['info', self.dest], check_rc=False)
        return rc == 0

    def checkout(self, force=False):
        if False:
            for i in range(10):
                print('nop')
        'Creates new svn working directory if it does not already exist.'
        cmd = ['checkout']
        if force:
            cmd.append('--force')
        cmd.extend(['-r', self.revision, self.repo, self.dest])
        self._exec(cmd)

    def export(self, force=False):
        if False:
            i = 10
            return i + 15
        'Export svn repo to directory'
        cmd = ['export']
        if force:
            cmd.append('--force')
        cmd.extend(['-r', self.revision, self.repo, self.dest])
        self._exec(cmd)

    def switch(self):
        if False:
            while True:
                i = 10
        "Change working directory's repo."
        output = self._exec(['switch', '--revision', self.revision, self.repo, self.dest])
        for line in output:
            if re.search('^[ABDUCGE]\\s', line):
                return True
        return False

    def update(self):
        if False:
            return 10
        'Update existing svn working directory.'
        output = self._exec(['update', '-r', self.revision, self.dest])
        for line in output:
            if re.search('^[ABDUCGE]\\s', line):
                return True
        return False

    def revert(self):
        if False:
            while True:
                i = 10
        'Revert svn working directory.'
        output = self._exec(['revert', '-R', self.dest])
        for line in output:
            if re.search('^Reverted ', line) is None:
                return True
        return False

    def get_revision(self):
        if False:
            print('Hello World!')
        'Revision and URL of subversion working directory.'
        text = '\n'.join(self._exec(['info', self.dest]))
        rev = re.search(self.REVISION_RE, text, re.MULTILINE)
        if rev:
            rev = rev.group(0)
        else:
            rev = 'Unable to get revision'
        url = re.search('^URL\\s?:.*$', text, re.MULTILINE)
        if url:
            url = url.group(0)
        else:
            url = 'Unable to get URL'
        return (rev, url)

    def get_remote_revision(self):
        if False:
            for i in range(10):
                print('nop')
        'Revision and URL of subversion working directory.'
        text = '\n'.join(self._exec(['info', self.repo]))
        rev = re.search(self.REVISION_RE, text, re.MULTILINE)
        if rev:
            rev = rev.group(0)
        else:
            rev = 'Unable to get remote revision'
        return rev

    def has_local_mods(self):
        if False:
            for i in range(10):
                print('nop')
        'True if revisioned files have been added or modified. Unrevisioned files are ignored.'
        lines = self._exec(['status', '--quiet', '--ignore-externals', self.dest])
        regex = re.compile('^[^?X]')
        return len(list(filter(regex.match, lines))) > 0

    def needs_update(self):
        if False:
            while True:
                i = 10
        (curr, url) = self.get_revision()
        out2 = '\n'.join(self._exec(['info', '-r', self.revision, self.dest]))
        head = re.search(self.REVISION_RE, out2, re.MULTILINE)
        if head:
            head = head.group(0)
        else:
            head = 'Unable to get revision'
        rev1 = int(curr.split(':')[1].strip())
        rev2 = int(head.split(':')[1].strip())
        change = False
        if rev1 < rev2:
            change = True
        return (change, curr, head)

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict(dest=dict(type='path'), repo=dict(type='str', required=True, aliases=['name', 'repository']), revision=dict(type='str', default='HEAD', aliases=['rev', 'version']), force=dict(type='bool', default=False), username=dict(type='str'), password=dict(type='str', no_log=True), executable=dict(type='path'), export=dict(type='bool', default=False), checkout=dict(type='bool', default=True), update=dict(type='bool', default=True), switch=dict(type='bool', default=True), in_place=dict(type='bool', default=False), validate_certs=dict(type='bool', default=False)), supports_check_mode=True)
    dest = module.params['dest']
    repo = module.params['repo']
    revision = module.params['revision']
    force = module.params['force']
    username = module.params['username']
    password = module.params['password']
    svn_path = module.params['executable'] or module.get_bin_path('svn', True)
    export = module.params['export']
    switch = module.params['switch']
    checkout = module.params['checkout']
    update = module.params['update']
    in_place = module.params['in_place']
    validate_certs = module.params['validate_certs']
    locale = get_best_parsable_locale(module)
    module.run_command_environ_update = dict(LANG=locale, LC_MESSAGES=locale)
    if not dest and (checkout or update or export):
        module.fail_json(msg='the destination directory must be specified unless checkout=no, update=no, and export=no')
    svn = Subversion(module, dest, repo, revision, username, password, svn_path, validate_certs)
    if not export and (not update) and (not checkout):
        module.exit_json(changed=False, after=svn.get_remote_revision())
    if export or not os.path.exists(dest):
        before = None
        local_mods = False
        if module.check_mode:
            module.exit_json(changed=True)
        elif not export and (not checkout):
            module.exit_json(changed=False)
        if not export and checkout:
            svn.checkout()
            files_changed = True
        else:
            svn.export(force=force)
            files_changed = True
    elif svn.is_svn_repo():
        if not update:
            module.exit_json(changed=False)
        if module.check_mode:
            if svn.has_local_mods() and (not force):
                module.fail_json(msg='ERROR: modified files exist in the repository.')
            (check, before, after) = svn.needs_update()
            module.exit_json(changed=check, before=before, after=after)
        files_changed = False
        before = svn.get_revision()
        local_mods = svn.has_local_mods()
        if switch:
            files_changed = svn.switch() or files_changed
        if local_mods:
            if force:
                files_changed = svn.revert() or files_changed
            else:
                module.fail_json(msg='ERROR: modified files exist in the repository.')
        files_changed = svn.update() or files_changed
    elif in_place:
        before = None
        svn.checkout(force=True)
        files_changed = True
        local_mods = svn.has_local_mods()
        if local_mods and force:
            svn.revert()
    else:
        module.fail_json(msg='ERROR: %s folder already exists, but its not a subversion repository.' % (dest,))
    if export:
        module.exit_json(changed=True)
    else:
        after = svn.get_revision()
        changed = files_changed or local_mods
        module.exit_json(changed=changed, before=before, after=after)
if __name__ == '__main__':
    main()