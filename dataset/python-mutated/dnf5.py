from __future__ import annotations
DOCUMENTATION = '\nmodule: dnf5\nauthor: Ansible Core Team\ndescription:\n  - Installs, upgrade, removes, and lists packages and groups with the I(dnf5) package manager.\n  - "WARNING: The I(dnf5) package manager is still under development and not all features that the existing M(ansible.builtin.dnf) module\n    provides are implemented in M(ansible.builtin.dnf5), please consult specific options for more information."\nshort_description: Manages packages with the I(dnf5) package manager\noptions:\n  name:\n    description:\n      - "A package name or package specifier with version, like C(name-1.0).\n        When using state=latest, this can be \'*\' which means run: dnf -y update.\n        You can also pass a url or a local path to an rpm file.\n        To operate on several packages this can accept a comma separated string of packages or a list of packages."\n      - Comparison operators for package version are valid here C(>), C(<), C(>=), C(<=). Example - C(name >= 1.0).\n        Spaces around the operator are required.\n      - You can also pass an absolute path for a binary which is provided by the package to install.\n        See examples for more information.\n    aliases:\n        - pkg\n    type: list\n    elements: str\n    default: []\n  list:\n    description:\n      - Various (non-idempotent) commands for usage with C(/usr/bin/ansible) and I(not) playbooks.\n        Use M(ansible.builtin.package_facts) instead of the O(list) argument as a best practice.\n    type: str\n  state:\n    description:\n      - Whether to install (V(present), V(latest)), or remove (V(absent)) a package.\n      - Default is V(None), however in effect the default action is V(present) unless the V(autoremove) option is\n        enabled for this module, then V(absent) is inferred.\n    choices: [\'absent\', \'present\', \'installed\', \'removed\', \'latest\']\n    type: str\n  enablerepo:\n    description:\n      - I(Repoid) of repositories to enable for the install/update operation.\n        These repos will not persist beyond the transaction.\n        When specifying multiple repos, separate them with a ",".\n    type: list\n    elements: str\n    default: []\n  disablerepo:\n    description:\n      - I(Repoid) of repositories to disable for the install/update operation.\n        These repos will not persist beyond the transaction.\n        When specifying multiple repos, separate them with a ",".\n    type: list\n    elements: str\n    default: []\n  conf_file:\n    description:\n      - The remote dnf configuration file to use for the transaction.\n    type: str\n  disable_gpg_check:\n    description:\n      - Whether to disable the GPG checking of signatures of packages being\n        installed. Has an effect only if O(state) is V(present) or V(latest).\n      - This setting affects packages installed from a repository as well as\n        "local" packages installed from the filesystem or a URL.\n    type: bool\n    default: \'no\'\n  installroot:\n    description:\n      - Specifies an alternative installroot, relative to which all packages\n        will be installed.\n    default: "/"\n    type: str\n  releasever:\n    description:\n      - Specifies an alternative release from which all packages will be\n        installed.\n    type: str\n  autoremove:\n    description:\n      - If V(true), removes all "leaf" packages from the system that were originally\n        installed as dependencies of user-installed packages but which are no longer\n        required by any such package. Should be used alone or when O(state) is V(absent)\n    type: bool\n    default: "no"\n  exclude:\n    description:\n      - Package name(s) to exclude when state=present, or latest. This can be a\n        list or a comma separated string.\n    type: list\n    elements: str\n    default: []\n  skip_broken:\n    description:\n      - Skip all unavailable packages or packages with broken dependencies\n        without raising an error. Equivalent to passing the --skip-broken option.\n    type: bool\n    default: "no"\n  update_cache:\n    description:\n      - Force dnf to check if cache is out of date and redownload if needed.\n        Has an effect only if O(state) is V(present) or V(latest).\n    type: bool\n    default: "no"\n    aliases: [ expire-cache ]\n  update_only:\n    description:\n      - When using latest, only update installed packages. Do not install packages.\n      - Has an effect only if O(state) is V(latest)\n    default: "no"\n    type: bool\n  security:\n    description:\n      - If set to V(true), and O(state=latest) then only installs updates that have been marked security related.\n      - Note that, similar to C(dnf upgrade-minimal), this filter applies to dependencies as well.\n    type: bool\n    default: "no"\n  bugfix:\n    description:\n      - If set to V(true), and O(state=latest) then only installs updates that have been marked bugfix related.\n      - Note that, similar to C(dnf upgrade-minimal), this filter applies to dependencies as well.\n    default: "no"\n    type: bool\n  enable_plugin:\n    description:\n      - This is currently a no-op as dnf5 itself does not implement this feature.\n      - I(Plugin) name to enable for the install/update operation.\n        The enabled plugin will not persist beyond the transaction.\n    type: list\n    elements: str\n    default: []\n  disable_plugin:\n    description:\n      - This is currently a no-op as dnf5 itself does not implement this feature.\n      - I(Plugin) name to disable for the install/update operation.\n        The disabled plugins will not persist beyond the transaction.\n    type: list\n    default: []\n    elements: str\n  disable_excludes:\n    description:\n      - Disable the excludes defined in DNF config files.\n      - If set to V(all), disables all excludes.\n      - If set to V(main), disable excludes defined in [main] in dnf.conf.\n      - If set to V(repoid), disable excludes defined for given repo id.\n    type: str\n  validate_certs:\n    description:\n      - This is effectively a no-op in the dnf5 module as dnf5 itself handles downloading a https url as the source of the rpm,\n        but is an accepted parameter for feature parity/compatibility with the M(ansible.builtin.yum) module.\n    type: bool\n    default: "yes"\n  sslverify:\n    description:\n      - Disables SSL validation of the repository server for this transaction.\n      - This should be set to V(false) if one of the configured repositories is using an untrusted or self-signed certificate.\n    type: bool\n    default: "yes"\n  allow_downgrade:\n    description:\n      - Specify if the named package and version is allowed to downgrade\n        a maybe already installed higher version of that package.\n        Note that setting allow_downgrade=True can make this module\n        behave in a non-idempotent way. The task could end up with a set\n        of packages that does not match the complete list of specified\n        packages to install (because dependencies between the downgraded\n        package and others can cause changes to the packages which were\n        in the earlier transaction).\n    type: bool\n    default: "no"\n  install_repoquery:\n    description:\n      - This is effectively a no-op in DNF as it is not needed with DNF, but is an accepted parameter for feature\n        parity/compatibility with the M(ansible.builtin.yum) module.\n    type: bool\n    default: "yes"\n  download_only:\n    description:\n      - Only download the packages, do not install them.\n    default: "no"\n    type: bool\n  lock_timeout:\n    description:\n      - This is currently a no-op as dnf5 does not provide an option to configure it.\n      - Amount of time to wait for the dnf lockfile to be freed.\n    required: false\n    default: 30\n    type: int\n  install_weak_deps:\n    description:\n      - Will also install all packages linked by a weak dependency relation.\n    type: bool\n    default: "yes"\n  download_dir:\n    description:\n      - Specifies an alternate directory to store packages.\n      - Has an effect only if O(download_only) is specified.\n    type: str\n  allowerasing:\n    description:\n      - If V(true) it allows  erasing  of  installed  packages to resolve dependencies.\n    required: false\n    type: bool\n    default: "no"\n  nobest:\n    description:\n      - Set best option to False, so that transactions are not limited to best candidates only.\n    required: false\n    type: bool\n    default: "no"\n  cacheonly:\n    description:\n      - Tells dnf to run entirely from system cache; does not download or update metadata.\n    type: bool\n    default: "no"\nextends_documentation_fragment:\n- action_common_attributes\n- action_common_attributes.flow\nattributes:\n    action:\n        details: In the case of dnf, it has 2 action plugins that use it under the hood, M(ansible.builtin.yum) and M(ansible.builtin.package).\n        support: partial\n    async:\n        support: none\n    bypass_host_loop:\n        support: none\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        platforms: rhel\nrequirements:\n  - "python3"\n  - "python3-libdnf5"\nversion_added: 2.15\n'
EXAMPLES = '\n- name: Install the latest version of Apache\n  ansible.builtin.dnf5:\n    name: httpd\n    state: latest\n\n- name: Install Apache >= 2.4\n  ansible.builtin.dnf5:\n    name: httpd >= 2.4\n    state: present\n\n- name: Install the latest version of Apache and MariaDB\n  ansible.builtin.dnf5:\n    name:\n      - httpd\n      - mariadb-server\n    state: latest\n\n- name: Remove the Apache package\n  ansible.builtin.dnf5:\n    name: httpd\n    state: absent\n\n- name: Install the latest version of Apache from the testing repo\n  ansible.builtin.dnf5:\n    name: httpd\n    enablerepo: testing\n    state: present\n\n- name: Upgrade all packages\n  ansible.builtin.dnf5:\n    name: "*"\n    state: latest\n\n- name: Update the webserver, depending on which is installed on the system. Do not install the other one\n  ansible.builtin.dnf5:\n    name:\n      - httpd\n      - nginx\n    state: latest\n    update_only: yes\n\n- name: Install the nginx rpm from a remote repo\n  ansible.builtin.dnf5:\n    name: \'http://nginx.org/packages/centos/6/noarch/RPMS/nginx-release-centos-6-0.el6.ngx.noarch.rpm\'\n    state: present\n\n- name: Install nginx rpm from a local file\n  ansible.builtin.dnf5:\n    name: /usr/local/src/nginx-release-centos-6-0.el6.ngx.noarch.rpm\n    state: present\n\n- name: Install Package based upon the file it provides\n  ansible.builtin.dnf5:\n    name: /usr/bin/cowsay\n    state: present\n\n- name: Install the \'Development tools\' package group\n  ansible.builtin.dnf5:\n    name: \'@Development tools\'\n    state: present\n\n- name: Autoremove unneeded packages installed as dependencies\n  ansible.builtin.dnf5:\n    autoremove: yes\n\n- name: Uninstall httpd but keep its dependencies\n  ansible.builtin.dnf5:\n    name: httpd\n    state: absent\n    autoremove: no\n'
RETURN = '\nmsg:\n  description: Additional information about the result\n  returned: always\n  type: str\n  sample: "Nothing to do"\nresults:\n  description: A list of the dnf transaction results\n  returned: success\n  type: list\n  sample: ["Installed: lsof-4.94.0-4.fc37.x86_64"]\nfailures:\n  description: A list of the dnf transaction failures\n  returned: failure\n  type: list\n  sample: ["Argument \'lsof\' matches only excluded packages."]\nrc:\n  description: For compatibility, 0 for success, 1 for failure\n  returned: always\n  type: int\n  sample: 0\n'
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
libdnf5 = None

def is_installed(base, spec):
    if False:
        print('Hello World!')
    settings = libdnf5.base.ResolveSpecSettings()
    query = libdnf5.rpm.PackageQuery(base)
    query.filter_installed()
    (match, nevra) = query.resolve_pkg_spec(spec, settings, True)
    return match

def is_newer_version_installed(base, spec):
    if False:
        while True:
            i = 10
    if '/' in spec:
        spec = spec.split('/')[-1]
        if spec.endswith('.rpm'):
            spec = spec[:-4]
    try:
        spec_nevra = next(iter(libdnf5.rpm.Nevra.parse(spec)))
    except (RuntimeError, StopIteration):
        return False
    spec_name = spec_nevra.get_name()
    v = spec_nevra.get_version()
    r = spec_nevra.get_release()
    if not v or not r:
        return False
    spec_evr = '{}:{}-{}'.format(spec_nevra.get_epoch() or '0', v, r)
    query = libdnf5.rpm.PackageQuery(base)
    query.filter_installed()
    query.filter_name([spec_name])
    query.filter_evr([spec_evr], libdnf5.common.QueryCmp_GT)
    return query.size() > 0

def package_to_dict(package):
    if False:
        return 10
    return {'nevra': package.get_nevra(), 'envra': package.get_nevra(), 'name': package.get_name(), 'arch': package.get_arch(), 'epoch': str(package.get_epoch()), 'release': package.get_release(), 'version': package.get_version(), 'repo': package.get_repo_id(), 'yumstate': 'installed' if package.is_installed() else 'available'}

def get_unneeded_pkgs(base):
    if False:
        while True:
            i = 10
    query = libdnf5.rpm.PackageQuery(base)
    query.filter_installed()
    query.filter_unneeded()
    for pkg in query:
        yield pkg

class Dnf5Module(YumDnf):

    def __init__(self, module):
        if False:
            while True:
                i = 10
        super(Dnf5Module, self).__init__(module)
        self._ensure_dnf()
        self.lockfile = ''
        self.pkg_mgr_name = 'dnf5'
        self.allowerasing = self.module.params['allowerasing']
        self.nobest = self.module.params['nobest']

    def _ensure_dnf(self):
        if False:
            for i in range(10):
                print('nop')
        locale = get_best_parsable_locale(self.module)
        os.environ['LC_ALL'] = os.environ['LC_MESSAGES'] = locale
        os.environ['LANGUAGE'] = os.environ['LANG'] = locale
        global libdnf5
        has_dnf = True
        try:
            import libdnf5
        except ImportError:
            has_dnf = False
        if has_dnf:
            return
        system_interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python']
        if not has_respawned():
            interpreter = probe_interpreters_for_module(system_interpreters, 'libdnf5')
            if interpreter:
                respawn_module(interpreter)
        self.module.fail_json(msg='Could not import the libdnf5 python module using {0} ({1}). Please install python3-libdnf5 package or ensure you have specified the correct ansible_python_interpreter. (attempted {2})'.format(sys.executable, sys.version.replace('\n', ''), system_interpreters), failures=[])

    def is_lockfile_pid_valid(self):
        if False:
            while True:
                i = 10
        return True

    def run(self):
        if False:
            i = 10
            return i + 15
        if sys.version_info.major < 3:
            self.module.fail_json(msg='The dnf5 module requires Python 3.', failures=[], rc=1)
        if not self.list and (not self.download_only) and (os.geteuid() != 0):
            self.module.fail_json(msg='This command has to be run under the root user.', failures=[], rc=1)
        if self.enable_plugin or self.disable_plugin:
            self.module.fail_json(msg='enable_plugin and disable_plugin options are not yet implemented in DNF5', failures=[], rc=1)
        base = libdnf5.base.Base()
        conf = base.get_config()
        if self.conf_file:
            conf.config_file_path = self.conf_file
        try:
            base.load_config_from_file()
        except RuntimeError as e:
            self.module.fail_json(msg=str(e), conf_file=self.conf_file, failures=[], rc=1)
        if self.releasever is not None:
            variables = base.get_vars()
            variables.set('releasever', self.releasever)
        if self.exclude:
            conf.excludepkgs = self.exclude
        if self.disable_excludes:
            if self.disable_excludes == 'all':
                self.disable_excludes = '*'
            conf.disable_excludes = self.disable_excludes
        conf.skip_broken = self.skip_broken
        conf.best = not self.nobest
        conf.install_weak_deps = self.install_weak_deps
        conf.gpgcheck = not self.disable_gpg_check
        conf.localpkg_gpgcheck = not self.disable_gpg_check
        conf.sslverify = self.sslverify
        conf.clean_requirements_on_remove = self.autoremove
        conf.installroot = self.installroot
        conf.use_host_config = True
        conf.cacheonly = 'all' if self.cacheonly else 'none'
        if self.download_dir:
            conf.destdir = self.download_dir
        base.setup()
        log_router = base.get_logger()
        global_logger = libdnf5.logger.GlobalLogger()
        global_logger.set(log_router.get(), libdnf5.logger.Logger.Level_DEBUG)
        logger = libdnf5.logger.create_file_logger(base)
        log_router.add_logger(logger)
        if self.update_cache:
            repo_query = libdnf5.repo.RepoQuery(base)
            repo_query.filter_type(libdnf5.repo.Repo.Type_AVAILABLE)
            for repo in repo_query:
                repo_dir = repo.get_cachedir()
                if os.path.exists(repo_dir):
                    repo_cache = libdnf5.repo.RepoCache(base, repo_dir)
                    repo_cache.write_attribute(libdnf5.repo.RepoCache.ATTRIBUTE_EXPIRED)
        sack = base.get_repo_sack()
        sack.create_repos_from_system_configuration()
        repo_query = libdnf5.repo.RepoQuery(base)
        if self.disablerepo:
            repo_query.filter_id(self.disablerepo, libdnf5.common.QueryCmp_IGLOB)
            for repo in repo_query:
                repo.disable()
        if self.enablerepo:
            repo_query.filter_id(self.enablerepo, libdnf5.common.QueryCmp_IGLOB)
            for repo in repo_query:
                repo.enable()
        sack.update_and_load_enabled_repos(True)
        if self.update_cache and (not self.names) and (not self.list):
            self.module.exit_json(msg='Cache updated', changed=False, results=[], rc=0)
        if self.list:
            command = self.list
            if command == 'updates':
                command = 'upgrades'
            if command in {'installed', 'upgrades', 'available'}:
                query = libdnf5.rpm.PackageQuery(base)
                getattr(query, 'filter_{}'.format(command))()
                results = [package_to_dict(package) for package in query]
            elif command in {'repos', 'repositories'}:
                query = libdnf5.repo.RepoQuery(base)
                query.filter_enabled(True)
                results = [{'repoid': repo.get_id(), 'state': 'enabled'} for repo in query]
            else:
                resolve_spec_settings = libdnf5.base.ResolveSpecSettings()
                query = libdnf5.rpm.PackageQuery(base)
                query.resolve_pkg_spec(command, resolve_spec_settings, True)
                results = [package_to_dict(package) for package in query]
            self.module.exit_json(msg='', results=results, rc=0)
        settings = libdnf5.base.GoalJobSettings()
        settings.group_with_name = True
        if self.bugfix or self.security:
            advisory_query = libdnf5.advisory.AdvisoryQuery(base)
            types = []
            if self.bugfix:
                types.append('bugfix')
            if self.security:
                types.append('security')
            advisory_query.filter_type(types)
            settings.set_advisory_filter(advisory_query)
        goal = libdnf5.base.Goal(base)
        results = []
        if self.names == ['*'] and self.state == 'latest':
            goal.add_rpm_upgrade(settings)
        elif self.state in {'install', 'present', 'latest'}:
            upgrade = self.state == 'latest'
            for spec in self.names:
                if is_newer_version_installed(base, spec):
                    if self.allow_downgrade:
                        if upgrade:
                            if is_installed(base, spec):
                                goal.add_upgrade(spec, settings)
                            else:
                                goal.add_install(spec, settings)
                        else:
                            goal.add_install(spec, settings)
                elif is_installed(base, spec):
                    if upgrade:
                        goal.add_upgrade(spec, settings)
                elif self.update_only:
                    results.append('Packages providing {} not installed due to update_only specified'.format(spec))
                else:
                    goal.add_install(spec, settings)
        elif self.state in {'absent', 'removed'}:
            for spec in self.names:
                try:
                    goal.add_remove(spec, settings)
                except RuntimeError as e:
                    self.module.fail_json(msg=str(e), failures=[], rc=1)
            if self.autoremove:
                for pkg in get_unneeded_pkgs(base):
                    goal.add_rpm_remove(pkg, settings)
        goal.set_allow_erasing(self.allowerasing)
        try:
            transaction = goal.resolve()
        except RuntimeError as e:
            self.module.fail_json(msg=str(e), failures=[], rc=1)
        if transaction.get_problems():
            failures = []
            for log_event in transaction.get_resolve_logs():
                if log_event.get_problem() == libdnf5.base.GoalProblem_NOT_FOUND and self.state in {'install', 'present', 'latest'}:
                    failures.append('No package {} available.'.format(log_event.get_spec()))
                else:
                    failures.append(log_event.to_string())
            if transaction.get_problems() & libdnf5.base.GoalProblem_SOLVER_ERROR != 0:
                msg = 'Depsolve Error occurred'
            else:
                msg = 'Failed to install some of the specified packages'
            self.module.fail_json(msg=msg, failures=failures, rc=1)
        actions_compat_map = {'Install': 'Installed', 'Remove': 'Removed', 'Replace': 'Installed', 'Upgrade': 'Installed', 'Replaced': 'Removed'}
        changed = bool(transaction.get_transaction_packages())
        for pkg in transaction.get_transaction_packages():
            if self.download_only:
                action = 'Downloaded'
            else:
                action = libdnf5.base.transaction.transaction_item_action_to_string(pkg.get_action())
            results.append('{}: {}'.format(actions_compat_map.get(action, action), pkg.get_package().get_nevra()))
        msg = ''
        if self.module.check_mode:
            if results:
                msg = 'Check mode: No changes made, but would have if not in check mode'
        else:
            transaction.download()
            if not self.download_only:
                transaction.set_description('ansible dnf5 module')
                result = transaction.run()
                if result == libdnf5.base.Transaction.TransactionRunResult_ERROR_GPG_CHECK:
                    self.module.fail_json(msg='Failed to validate GPG signatures: {}'.format(','.join(transaction.get_gpg_signature_problems())), failures=[], rc=1)
                elif result != libdnf5.base.Transaction.TransactionRunResult_SUCCESS:
                    self.module.fail_json(msg='Failed to install some of the specified packages', failures=['{}: {}'.format(transaction.transaction_result_to_string(result), log) for log in transaction.get_transaction_problems()], rc=1)
        if not msg and (not results):
            msg = 'Nothing to do'
        self.module.exit_json(results=results, changed=changed, msg=msg, rc=0)

def main():
    if False:
        for i in range(10):
            print('nop')
    yumdnf_argument_spec['argument_spec']['allowerasing'] = dict(default=False, type='bool')
    yumdnf_argument_spec['argument_spec']['nobest'] = dict(default=False, type='bool')
    Dnf5Module(AnsibleModule(**yumdnf_argument_spec)).run()
if __name__ == '__main__':
    main()