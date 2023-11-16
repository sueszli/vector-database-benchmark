from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: apt_repository\nshort_description: Add and remove APT repositories\ndescription:\n    - Add or remove an APT repositories in Ubuntu and Debian.\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        platforms: debian\nnotes:\n    - This module supports Debian Squeeze (version 6) as well as its successors and derivatives.\nseealso:\n  - module: ansible.builtin.deb822_repository\noptions:\n    repo:\n        description:\n            - A source string for the repository.\n        type: str\n        required: true\n    state:\n        description:\n            - A source string state.\n        type: str\n        choices: [ absent, present ]\n        default: "present"\n    mode:\n        description:\n            - The octal mode for newly created files in sources.list.d.\n            - Default is what system uses (probably 0644).\n        type: raw\n        version_added: "1.6"\n    update_cache:\n        description:\n            - Run the equivalent of C(apt-get update) when a change occurs.  Cache updates are run after making changes.\n        type: bool\n        default: "yes"\n        aliases: [ update-cache ]\n    update_cache_retries:\n        description:\n        - Amount of retries if the cache update fails. Also see O(update_cache_retry_max_delay).\n        type: int\n        default: 5\n        version_added: \'2.10\'\n    update_cache_retry_max_delay:\n        description:\n        - Use an exponential backoff delay for each retry (see O(update_cache_retries)) up to this max delay in seconds.\n        type: int\n        default: 12\n        version_added: \'2.10\'\n    validate_certs:\n        description:\n            - If V(false), SSL certificates for the target repo will not be validated. This should only be used\n              on personally controlled sites using self-signed certificates.\n        type: bool\n        default: \'yes\'\n        version_added: \'1.8\'\n    filename:\n        description:\n            - Sets the name of the source list file in sources.list.d.\n              Defaults to a file name based on the repository source url.\n              The .list extension will be automatically added.\n        type: str\n        version_added: \'2.1\'\n    codename:\n        description:\n            - Override the distribution codename to use for PPA repositories.\n              Should usually only be set when working with a PPA on\n              a non-Ubuntu target (for example, Debian or Mint).\n        type: str\n        version_added: \'2.3\'\n    install_python_apt:\n        description:\n            - Whether to automatically try to install the Python apt library or not, if it is not already installed.\n              Without this library, the module does not work.\n            - Runs C(apt-get install python-apt) for Python 2, and C(apt-get install python3-apt) for Python 3.\n            - Only works with the system Python 2 or Python 3. If you are using a Python on the remote that is not\n               the system Python, set O(install_python_apt=false) and ensure that the Python apt library\n               for your Python version is installed some other way.\n        type: bool\n        default: true\nauthor:\n- Alexander Saltanov (@sashka)\nversion_added: "0.7"\nrequirements:\n   - python-apt (python 2)\n   - python3-apt (python 3)\n   - apt-key or gpg\n'
EXAMPLES = '\n- name: Add specified repository into sources list\n  ansible.builtin.apt_repository:\n    repo: deb http://archive.canonical.com/ubuntu hardy partner\n    state: present\n\n- name: Add specified repository into sources list using specified filename\n  ansible.builtin.apt_repository:\n    repo: deb http://dl.google.com/linux/chrome/deb/ stable main\n    state: present\n    filename: google-chrome\n\n- name: Add source repository into sources list\n  ansible.builtin.apt_repository:\n    repo: deb-src http://archive.canonical.com/ubuntu hardy partner\n    state: present\n\n- name: Remove specified repository from sources list\n  ansible.builtin.apt_repository:\n    repo: deb http://archive.canonical.com/ubuntu hardy partner\n    state: absent\n\n- name: Add nginx stable repository from PPA and install its signing key on Ubuntu target\n  ansible.builtin.apt_repository:\n    repo: ppa:nginx/stable\n\n- name: Add nginx stable repository from PPA and install its signing key on Debian target\n  ansible.builtin.apt_repository:\n    repo: \'ppa:nginx/stable\'\n    codename: trusty\n\n- name: One way to avoid apt_key once it is removed from your distro\n  block:\n    - name: somerepo |no apt key\n      ansible.builtin.get_url:\n        url: https://download.example.com/linux/ubuntu/gpg\n        dest: /etc/apt/keyrings/somerepo.asc\n\n    - name: somerepo | apt source\n      ansible.builtin.apt_repository:\n        repo: "deb [arch=amd64 signed-by=/etc/apt/keyrings/myrepo.asc] https://download.example.com/linux/ubuntu {{ ansible_distribution_release }} stable"\n        state: present\n'
RETURN = '\nrepo:\n  description: A source string for the repository\n  returned: always\n  type: str\n  sample: "deb https://artifacts.elastic.co/packages/6.x/apt stable main"\n\nsources_added:\n  description: List of sources added\n  returned: success, sources were added\n  type: list\n  sample: ["/etc/apt/sources.list.d/artifacts_elastic_co_packages_6_x_apt.list"]\n  version_added: "2.15"\n\nsources_removed:\n  description: List of sources removed\n  returned: success, sources were removed\n  type: list\n  sample: ["/etc/apt/sources.list.d/artifacts_elastic_co_packages_6_x_apt.list"]\n  version_added: "2.15"\n'
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
try:
    import apt
    import apt_pkg
    import aptsources.distro as aptsources_distro
    distro = aptsources_distro.get_distro()
    HAVE_PYTHON_APT = True
except ImportError:
    apt = apt_pkg = aptsources_distro = distro = None
    HAVE_PYTHON_APT = False
APT_KEY_DIRS = ['/etc/apt/keyrings', '/etc/apt/trusted.gpg.d', '/usr/share/keyrings']
DEFAULT_SOURCES_PERM = 420
VALID_SOURCE_TYPES = ('deb', 'deb-src')

def install_python_apt(module, apt_pkg_name):
    if False:
        for i in range(10):
            print('nop')
    if not module.check_mode:
        apt_get_path = module.get_bin_path('apt-get')
        if apt_get_path:
            (rc, so, se) = module.run_command([apt_get_path, 'update'])
            if rc != 0:
                module.fail_json(msg="Failed to auto-install %s. Error was: '%s'" % (apt_pkg_name, se.strip()))
            (rc, so, se) = module.run_command([apt_get_path, 'install', apt_pkg_name, '-y', '-q'])
            if rc != 0:
                module.fail_json(msg="Failed to auto-install %s. Error was: '%s'" % (apt_pkg_name, se.strip()))
    else:
        module.fail_json(msg='%s must be installed to use check mode' % apt_pkg_name)

class InvalidSource(Exception):
    pass

class SourcesList(object):

    def __init__(self, module):
        if False:
            i = 10
            return i + 15
        self.module = module
        self.files = {}
        self.new_repos = set()
        self.default_file = self._apt_cfg_file('Dir::Etc::sourcelist')
        if os.path.isfile(self.default_file):
            self.load(self.default_file)
        for file in glob.iglob('%s/*.list' % self._apt_cfg_dir('Dir::Etc::sourceparts')):
            self.load(file)

    def __iter__(self):
        if False:
            return 10
        'Simple iterator to go over all sources. Empty, non-source, and other not valid lines will be skipped.'
        for (file, sources) in self.files.items():
            for (n, valid, enabled, source, comment) in sources:
                if valid:
                    yield (file, n, enabled, source, comment)

    def _expand_path(self, filename):
        if False:
            while True:
                i = 10
        if '/' in filename:
            return filename
        else:
            return os.path.abspath(os.path.join(self._apt_cfg_dir('Dir::Etc::sourceparts'), filename))

    def _suggest_filename(self, line):
        if False:
            for i in range(10):
                print('nop')

        def _cleanup_filename(s):
            if False:
                i = 10
                return i + 15
            filename = self.module.params['filename']
            if filename is not None:
                return filename
            return '_'.join(re.sub('[^a-zA-Z0-9]', ' ', s).split())

        def _strip_username_password(s):
            if False:
                while True:
                    i = 10
            if '@' in s:
                s = s.split('@', 1)
                s = s[-1]
            return s
        line = re.sub('\\[[^\\]]+\\]', '', line)
        line = re.sub('\\w+://', '', line)
        parts = [part for part in line.split() if part not in VALID_SOURCE_TYPES]
        parts[0] = _strip_username_password(parts[0])
        return '%s.list' % _cleanup_filename(' '.join(parts[:1]))

    def _parse(self, line, raise_if_invalid_or_disabled=False):
        if False:
            return 10
        valid = False
        enabled = True
        source = ''
        comment = ''
        line = line.strip()
        if line.startswith('#'):
            enabled = False
            line = line[1:]
        i = line.find('#')
        if i > 0:
            comment = line[i + 1:].strip()
            line = line[:i]
        source = line.strip()
        if source:
            chunks = source.split()
            if chunks[0] in VALID_SOURCE_TYPES:
                valid = True
                source = ' '.join(chunks)
        if raise_if_invalid_or_disabled and (not valid or not enabled):
            raise InvalidSource(line)
        return (valid, enabled, source, comment)

    @staticmethod
    def _apt_cfg_file(filespec):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper for `apt_pkg` module for running with Python 2.5\n        '
        try:
            result = apt_pkg.config.find_file(filespec)
        except AttributeError:
            result = apt_pkg.Config.FindFile(filespec)
        return result

    @staticmethod
    def _apt_cfg_dir(dirspec):
        if False:
            while True:
                i = 10
        '\n        Wrapper for `apt_pkg` module for running with Python 2.5\n        '
        try:
            result = apt_pkg.config.find_dir(dirspec)
        except AttributeError:
            result = apt_pkg.Config.FindDir(dirspec)
        return result

    def load(self, file):
        if False:
            return 10
        group = []
        f = open(file, 'r')
        for (n, line) in enumerate(f):
            (valid, enabled, source, comment) = self._parse(line)
            group.append((n, valid, enabled, source, comment))
        self.files[file] = group

    def save(self):
        if False:
            while True:
                i = 10
        for (filename, sources) in list(self.files.items()):
            if sources:
                (d, fn) = os.path.split(filename)
                try:
                    os.makedirs(d)
                except OSError as ex:
                    if not os.path.isdir(d):
                        self.module.fail_json('Failed to create directory %s: %s' % (d, to_native(ex)))
                try:
                    (fd, tmp_path) = tempfile.mkstemp(prefix='.%s-' % fn, dir=d)
                except (OSError, IOError) as e:
                    self.module.fail_json(msg='Unable to create temp file at "%s" for apt source: %s' % (d, to_native(e)))
                f = os.fdopen(fd, 'w')
                for (n, valid, enabled, source, comment) in sources:
                    chunks = []
                    if not enabled:
                        chunks.append('# ')
                    chunks.append(source)
                    if comment:
                        chunks.append(' # ')
                        chunks.append(comment)
                    chunks.append('\n')
                    line = ''.join(chunks)
                    try:
                        f.write(line)
                    except IOError as ex:
                        self.module.fail_json(msg='Failed to write to file %s: %s' % (tmp_path, to_native(ex)))
                self.module.atomic_move(tmp_path, filename)
                if filename in self.new_repos:
                    this_mode = self.module.params.get('mode', DEFAULT_SOURCES_PERM)
                    self.module.set_mode_if_different(filename, this_mode, False)
            else:
                del self.files[filename]
                if os.path.exists(filename):
                    os.remove(filename)

    def dump(self):
        if False:
            return 10
        dumpstruct = {}
        for (filename, sources) in self.files.items():
            if sources:
                lines = []
                for (n, valid, enabled, source, comment) in sources:
                    chunks = []
                    if not enabled:
                        chunks.append('# ')
                    chunks.append(source)
                    if comment:
                        chunks.append(' # ')
                        chunks.append(comment)
                    chunks.append('\n')
                    lines.append(''.join(chunks))
                dumpstruct[filename] = ''.join(lines)
        return dumpstruct

    def _choice(self, new, old):
        if False:
            i = 10
            return i + 15
        if new is None:
            return old
        return new

    def modify(self, file, n, enabled=None, source=None, comment=None):
        if False:
            return 10
        "\n        This function to be used with iterator, so we don't care of invalid sources.\n        If source, enabled, or comment is None, original value from line ``n`` will be preserved.\n        "
        (valid, enabled_old, source_old, comment_old) = self.files[file][n][1:]
        self.files[file][n] = (n, valid, self._choice(enabled, enabled_old), self._choice(source, source_old), self._choice(comment, comment_old))

    def _add_valid_source(self, source_new, comment_new, file):
        if False:
            for i in range(10):
                print('nop')
        self.module.log('ading source file: %s | %s | %s' % (source_new, comment_new, file))
        found = False
        for (filename, n, enabled, source, comment) in self:
            if source == source_new:
                self.modify(filename, n, enabled=True)
                found = True
        if not found:
            if file is None:
                file = self.default_file
            else:
                file = self._expand_path(file)
            if file not in self.files:
                self.files[file] = []
            files = self.files[file]
            files.append((len(files), True, True, source_new, comment_new))
            self.new_repos.add(file)

    def add_source(self, line, comment='', file=None):
        if False:
            while True:
                i = 10
        source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._add_valid_source(source, comment, file=file or self._suggest_filename(source))

    def _remove_valid_source(self, source):
        if False:
            for i in range(10):
                print('nop')
        for (filename, n, enabled, src, comment) in self:
            if source == src and enabled:
                self.files[filename].pop(n)

    def remove_source(self, line):
        if False:
            print('Hello World!')
        source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._remove_valid_source(source)

class UbuntuSourcesList(SourcesList):
    LP_API = 'https://api.launchpad.net/1.0/~%s/+archive/%s'

    def __init__(self, module):
        if False:
            print('Hello World!')
        self.module = module
        self.codename = module.params['codename'] or distro.codename
        super(UbuntuSourcesList, self).__init__(module)
        self.apt_key_bin = self.module.get_bin_path('apt-key', required=False)
        self.gpg_bin = self.module.get_bin_path('gpg', required=False)
        if not self.apt_key_bin and (not self.gpg_bin):
            self.module.fail_json(msg='Either apt-key or gpg binary is required, but neither could be found')

    def __deepcopy__(self, memo=None):
        if False:
            while True:
                i = 10
        return UbuntuSourcesList(self.module)

    def _get_ppa_info(self, owner_name, ppa_name):
        if False:
            while True:
                i = 10
        lp_api = self.LP_API % (owner_name, ppa_name)
        headers = dict(Accept='application/json')
        (response, info) = fetch_url(self.module, lp_api, headers=headers)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch PPA information, error was: %s' % info['msg'])
        return json.loads(to_native(response.read()))

    def _expand_ppa(self, path):
        if False:
            for i in range(10):
                print('nop')
        ppa = path.split(':')[1]
        ppa_owner = ppa.split('/')[0]
        try:
            ppa_name = ppa.split('/')[1]
        except IndexError:
            ppa_name = 'ppa'
        line = 'deb http://ppa.launchpad.net/%s/%s/ubuntu %s main' % (ppa_owner, ppa_name, self.codename)
        return (line, ppa_owner, ppa_name)

    def _key_already_exists(self, key_fingerprint):
        if False:
            while True:
                i = 10
        if self.apt_key_bin:
            locale = get_best_parsable_locale(self.module)
            APT_ENV = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LC_CTYPE=locale)
            self.module.run_command_environ_update = APT_ENV
            (rc, out, err) = self.module.run_command([self.apt_key_bin, 'export', key_fingerprint], check_rc=True)
            found = bool(not err or 'nothing exported' not in err)
        else:
            found = self._gpg_key_exists(key_fingerprint)
        return found

    def _gpg_key_exists(self, key_fingerprint):
        if False:
            while True:
                i = 10
        found = False
        keyfiles = ['/etc/apt/trusted.gpg']
        for other_dir in APT_KEY_DIRS:
            keyfiles.extend([os.path.join(other_dir, x) for x in os.listdir(other_dir) if not x.startswith('.')])
        for key_file in keyfiles:
            if os.path.exists(key_file):
                try:
                    (rc, out, err) = self.module.run_command([self.gpg_bin, '--list-packets', key_file])
                except (IOError, OSError) as e:
                    self.debug('Could check key against file %s: %s' % (key_file, to_native(e)))
                    continue
                if key_fingerprint in out:
                    found = True
                    break
        return found

    def add_source(self, line, comment='', file=None):
        if False:
            while True:
                i = 10
        if line.startswith('ppa:'):
            (source, ppa_owner, ppa_name) = self._expand_ppa(line)
            if source in self.repos_urls:
                return
            info = self._get_ppa_info(ppa_owner, ppa_name)
            if not self._key_already_exists(info['signing_key_fingerprint']):
                keyfile = ''
                if not self.module.check_mode:
                    if self.apt_key_bin:
                        command = [self.apt_key_bin, 'adv', '--recv-keys', '--no-tty', '--keyserver', 'hkp://keyserver.ubuntu.com:80', info['signing_key_fingerprint']]
                    else:
                        for keydir in APT_KEY_DIRS:
                            if os.path.exists(keydir):
                                break
                        else:
                            self.module.fail_json('Unable to find any existing apt gpgp repo directories, tried the following: %s' % ', '.join(APT_KEY_DIRS))
                        keyfile = '%s/%s-%s-%s.gpg' % (keydir, os.path.basename(source).replace(' ', '-'), ppa_owner, ppa_name)
                        command = [self.gpg_bin, '--no-tty', '--keyserver', 'hkp://keyserver.ubuntu.com:80', '--export', info['signing_key_fingerprint']]
                    (rc, stdout, stderr) = self.module.run_command(command, check_rc=True, encoding=None)
                    if keyfile:
                        if len(stdout) == 0:
                            self.module.fail_json(msg='Unable to get required signing key', rc=rc, stderr=stderr, command=command)
                        try:
                            with open(keyfile, 'wb') as f:
                                f.write(stdout)
                            self.module.log('Added repo key "%s" for apt to file "%s"' % (info['signing_key_fingerprint'], keyfile))
                        except (OSError, IOError) as e:
                            self.module.fail_json(msg='Unable to add required signing key for%s ', rc=rc, stderr=stderr, error=to_native(e))
            file = file or self._suggest_filename('%s_%s' % (line, self.codename))
        else:
            source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
            file = file or self._suggest_filename(source)
        self._add_valid_source(source, comment, file)

    def remove_source(self, line):
        if False:
            for i in range(10):
                print('nop')
        if line.startswith('ppa:'):
            source = self._expand_ppa(line)[0]
        else:
            source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._remove_valid_source(source)

    @property
    def repos_urls(self):
        if False:
            return 10
        _repositories = []
        for parsed_repos in self.files.values():
            for parsed_repo in parsed_repos:
                valid = parsed_repo[1]
                enabled = parsed_repo[2]
                source_line = parsed_repo[3]
                if not valid or not enabled:
                    continue
                if source_line.startswith('ppa:'):
                    (source, ppa_owner, ppa_name) = self._expand_ppa(source_line)
                    _repositories.append(source)
                else:
                    _repositories.append(source_line)
        return _repositories

def revert_sources_list(sources_before, sources_after, sourceslist_before):
    if False:
        while True:
            i = 10
    'Revert the sourcelist files to their previous state.'
    for filename in set(sources_after.keys()).difference(sources_before.keys()):
        if os.path.exists(filename):
            os.remove(filename)
    sourceslist_before.save()

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(repo=dict(type='str', required=True), state=dict(type='str', default='present', choices=['absent', 'present']), mode=dict(type='raw'), update_cache=dict(type='bool', default=True, aliases=['update-cache']), update_cache_retries=dict(type='int', default=5), update_cache_retry_max_delay=dict(type='int', default=12), filename=dict(type='str'), install_python_apt=dict(type='bool', default=True), validate_certs=dict(type='bool', default=True), codename=dict(type='str')), supports_check_mode=True)
    params = module.params
    repo = module.params['repo']
    state = module.params['state']
    update_cache = module.params['update_cache']
    sourceslist = None
    if not HAVE_PYTHON_APT:
        apt_pkg_name = 'python3-apt' if PY3 else 'python-apt'
        if has_respawned():
            module.fail_json(msg='{0} must be installed and visible from {1}.'.format(apt_pkg_name, sys.executable))
        interpreters = ['/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python']
        interpreter = probe_interpreters_for_module(interpreters, 'apt')
        if interpreter:
            respawn_module(interpreter)
        if module.check_mode:
            module.fail_json(msg='%s must be installed to use check mode. If run normally this module can auto-install it.' % apt_pkg_name)
        if params['install_python_apt']:
            install_python_apt(module, apt_pkg_name)
        else:
            module.fail_json(msg='%s is not installed, and install_python_apt is False' % apt_pkg_name)
        interpreter = probe_interpreters_for_module(interpreters, 'apt')
        if interpreter:
            respawn_module(interpreter)
        else:
            module.fail_json(msg='{0} must be installed and visible from {1}.'.format(apt_pkg_name, sys.executable))
    if not repo:
        module.fail_json(msg="Please set argument 'repo' to a non-empty value")
    if isinstance(distro, aptsources_distro.Distribution):
        sourceslist = UbuntuSourcesList(module)
    else:
        module.fail_json(msg='Module apt_repository is not supported on target.')
    sourceslist_before = copy.deepcopy(sourceslist)
    sources_before = sourceslist.dump()
    try:
        if state == 'present':
            sourceslist.add_source(repo)
        elif state == 'absent':
            sourceslist.remove_source(repo)
    except InvalidSource as ex:
        module.fail_json(msg='Invalid repository string: %s' % to_native(ex))
    sources_after = sourceslist.dump()
    changed = sources_before != sources_after
    diff = []
    sources_added = set()
    sources_removed = set()
    if changed:
        sources_added = set(sources_after.keys()).difference(sources_before.keys())
        sources_removed = set(sources_before.keys()).difference(sources_after.keys())
        if module._diff:
            for filename in set(sources_added.union(sources_removed)):
                diff.append({'before': sources_before.get(filename, ''), 'after': sources_after.get(filename, ''), 'before_header': (filename, '/dev/null')[filename not in sources_before], 'after_header': (filename, '/dev/null')[filename not in sources_after]})
    if changed and (not module.check_mode):
        try:
            sourceslist.save()
            if update_cache:
                err = ''
                update_cache_retries = module.params.get('update_cache_retries')
                update_cache_retry_max_delay = module.params.get('update_cache_retry_max_delay')
                randomize = random.randint(0, 1000) / 1000.0
                for retry in range(update_cache_retries):
                    try:
                        cache = apt.Cache()
                        cache.update()
                        break
                    except apt.cache.FetchFailedException as e:
                        err = to_native(e)
                    delay = 2 ** retry + randomize
                    if delay > update_cache_retry_max_delay:
                        delay = update_cache_retry_max_delay + randomize
                    time.sleep(delay)
                else:
                    revert_sources_list(sources_before, sources_after, sourceslist_before)
                    module.fail_json(msg='Failed to update apt cache: %s' % (err if err else 'unknown reason'))
        except (OSError, IOError) as ex:
            revert_sources_list(sources_before, sources_after, sourceslist_before)
            module.fail_json(msg=to_native(ex))
    module.exit_json(changed=changed, repo=repo, sources_added=sources_added, sources_removed=sources_removed, state=state, diff=diff)
if __name__ == '__main__':
    main()