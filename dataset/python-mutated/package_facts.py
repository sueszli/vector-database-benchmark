from __future__ import annotations
DOCUMENTATION = '\nmodule: package_facts\nshort_description: Package information as facts\ndescription:\n  - Return information about installed packages as facts.\noptions:\n  manager:\n    description:\n      - The package manager used by the system so we can query the package information.\n      - Since 2.8 this is a list and can support multiple package managers per system.\n      - The \'portage\' and \'pkg\' options were added in version 2.8.\n      - The \'apk\' option was added in version 2.11.\n      - The \'pkg_info\' option was added in version 2.13.\n    default: [\'auto\']\n    choices: [\'auto\', \'rpm\', \'apt\', \'portage\', \'pkg\', \'pacman\', \'apk\', \'pkg_info\']\n    type: list\n    elements: str\n  strategy:\n    description:\n      - This option controls how the module queries the package managers on the system.\n        V(first) means it will return only information for the first supported package manager available.\n        V(all) will return information for all supported and available package managers on the system.\n    choices: [\'first\', \'all\']\n    default: \'first\'\n    type: str\n    version_added: "2.8"\nversion_added: "2.5"\nrequirements:\n    - For \'portage\' support it requires the C(qlist) utility, which is part of \'app-portage/portage-utils\'.\n    - For Debian-based systems C(python-apt) package must be installed on targeted hosts.\n    - For SUSE-based systems C(python3-rpm) package must be installed on targeted hosts.\n      This package is required because SUSE does not include RPM Python bindings by default.\nauthor:\n  - Matthew Jones (@matburt)\n  - Brian Coca (@bcoca)\n  - Adam Miller (@maxamillion)\nextends_documentation_fragment:\n  -  action_common_attributes\n  -  action_common_attributes.facts\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    facts:\n        support: full\n    platform:\n        platforms: posix\n'
EXAMPLES = '\n- name: Gather the package facts\n  ansible.builtin.package_facts:\n    manager: auto\n\n- name: Print the package facts\n  ansible.builtin.debug:\n    var: ansible_facts.packages\n\n- name: Check whether a package called foobar is installed\n  ansible.builtin.debug:\n    msg: "{{ ansible_facts.packages[\'foobar\'] | length }} versions of foobar are installed!"\n  when: "\'foobar\' in ansible_facts.packages"\n\n'
RETURN = '\nansible_facts:\n  description: Facts to add to ansible_facts.\n  returned: always\n  type: complex\n  contains:\n    packages:\n      description:\n        - Maps the package name to a non-empty list of dicts with package information.\n        - Every dict in the list corresponds to one installed version of the package.\n        - The fields described below are present for all package managers. Depending on the\n          package manager, there might be more fields for a package.\n      returned: when operating system level package manager is specified or auto detected manager\n      type: dict\n      contains:\n        name:\n          description: The package\'s name.\n          returned: always\n          type: str\n        version:\n          description: The package\'s version.\n          returned: always\n          type: str\n        source:\n          description: Where information on the package came from.\n          returned: always\n          type: str\n      sample: |-\n        {\n          "packages": {\n            "kernel": [\n              {\n                "name": "kernel",\n                "source": "rpm",\n                "version": "3.10.0",\n                ...\n              },\n              {\n                "name": "kernel",\n                "source": "rpm",\n                "version": "3.10.0",\n                ...\n              },\n              ...\n            ],\n            "kernel-tools": [\n              {\n                "name": "kernel-tools",\n                "source": "rpm",\n                "version": "3.10.0",\n                ...\n              }\n            ],\n            ...\n          }\n        }\n        # Sample rpm\n        {\n          "packages": {\n            "kernel": [\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel",\n                "release": "514.26.2.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              },\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel",\n                "release": "514.16.1.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              },\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel",\n                "release": "514.10.2.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              },\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel",\n                "release": "514.21.1.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              },\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel",\n                "release": "693.2.2.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              }\n            ],\n            "kernel-tools": [\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel-tools",\n                "release": "693.2.2.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              }\n            ],\n            "kernel-tools-libs": [\n              {\n                "arch": "x86_64",\n                "epoch": null,\n                "name": "kernel-tools-libs",\n                "release": "693.2.2.el7",\n                "source": "rpm",\n                "version": "3.10.0"\n              }\n            ],\n          }\n        }\n        # Sample deb\n        {\n          "packages": {\n            "libbz2-1.0": [\n              {\n                "version": "1.0.6-5",\n                "source": "apt",\n                "arch": "amd64",\n                "name": "libbz2-1.0"\n              }\n            ],\n            "patch": [\n              {\n                "version": "2.7.1-4ubuntu1",\n                "source": "apt",\n                "arch": "amd64",\n                "name": "patch"\n              }\n            ],\n          }\n        }\n        # Sample pkg_info\n        {\n          "packages": {\n            "curl": [\n              {\n                  "name": "curl",\n                  "source": "pkg_info",\n                  "version": "7.79.0"\n              }\n            ],\n            "intel-firmware": [\n              {\n                  "name": "intel-firmware",\n                  "source": "pkg_info",\n                  "version": "20210608v0"\n              }\n            ],\n          }\n        }\n'
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers

class RPM(LibMgr):
    LIB = 'rpm'

    def list_installed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._lib.TransactionSet().dbMatch()

    def get_package_details(self, package):
        if False:
            print('Hello World!')
        return dict(name=package[self._lib.RPMTAG_NAME], version=package[self._lib.RPMTAG_VERSION], release=package[self._lib.RPMTAG_RELEASE], epoch=package[self._lib.RPMTAG_EPOCH], arch=package[self._lib.RPMTAG_ARCH])

    def is_available(self):
        if False:
            while True:
                i = 10
        ' we expect the python bindings installed, but this gives warning if they are missing and we have rpm cli'
        we_have_lib = super(RPM, self).is_available()
        try:
            get_bin_path('rpm')
            if not we_have_lib and (not has_respawned()):
                interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2']
                interpreter_path = probe_interpreters_for_module(interpreters, self.LIB)
                if interpreter_path:
                    respawn_module(interpreter_path)
            if not we_have_lib:
                module.warn('Found "rpm" but %s' % missing_required_lib(self.LIB))
        except ValueError:
            pass
        return we_have_lib

class APT(LibMgr):
    LIB = 'apt'

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cache = None
        super(APT, self).__init__()

    @property
    def pkg_cache(self):
        if False:
            print('Hello World!')
        if self._cache is not None:
            return self._cache
        self._cache = self._lib.Cache()
        return self._cache

    def is_available(self):
        if False:
            for i in range(10):
                print('nop')
        ' we expect the python bindings installed, but if there is apt/apt-get give warning about missing bindings'
        we_have_lib = super(APT, self).is_available()
        if not we_have_lib:
            for exe in ('apt', 'apt-get', 'aptitude'):
                try:
                    get_bin_path(exe)
                except ValueError:
                    continue
                else:
                    if not has_respawned():
                        interpreters = ['/usr/bin/python3', '/usr/bin/python2']
                        interpreter_path = probe_interpreters_for_module(interpreters, self.LIB)
                        if interpreter_path:
                            respawn_module(interpreter_path)
                    module.warn('Found "%s" but %s' % (exe, missing_required_lib('apt')))
                    break
        return we_have_lib

    def list_installed(self):
        if False:
            return 10
        cache = self.pkg_cache
        return [pk for pk in cache.keys() if cache[pk].is_installed]

    def get_package_details(self, package):
        if False:
            for i in range(10):
                print('nop')
        ac_pkg = self.pkg_cache[package].installed
        return dict(name=package, version=ac_pkg.version, arch=ac_pkg.architecture, category=ac_pkg.section, origin=ac_pkg.origins[0].origin)

class PACMAN(CLIMgr):
    CLI = 'pacman'

    def list_installed(self):
        if False:
            for i in range(10):
                print('nop')
        locale = get_best_parsable_locale(module)
        (rc, out, err) = module.run_command([self._cli, '-Qi'], environ_update=dict(LC_ALL=locale))
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.split('\n\n')[:-1]

    def get_package_details(self, package):
        if False:
            i = 10
            return i + 15
        raw_pkg_details = {}
        last_detail = None
        for line in package.splitlines():
            m = re.match('([\\w ]*[\\w]) +: (.*)', line)
            if m:
                last_detail = m.group(1)
                raw_pkg_details[last_detail] = m.group(2)
            else:
                raw_pkg_details[last_detail] = raw_pkg_details[last_detail] + '  ' + line.lstrip()
        provides = None
        if raw_pkg_details['Provides'] != 'None':
            provides = [p.split('=')[0] for p in raw_pkg_details['Provides'].split('  ')]
        return {'name': raw_pkg_details['Name'], 'version': raw_pkg_details['Version'], 'arch': raw_pkg_details['Architecture'], 'provides': provides}

class PKG(CLIMgr):
    CLI = 'pkg'
    atoms = ['name', 'version', 'origin', 'installed', 'automatic', 'arch', 'category', 'prefix', 'vital']

    def list_installed(self):
        if False:
            while True:
                i = 10
        (rc, out, err) = module.run_command([self._cli, 'query', '%%%s' % '\t%'.join(['n', 'v', 'R', 't', 'a', 'q', 'o', 'p', 'V'])])
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.splitlines()

    def get_package_details(self, package):
        if False:
            return 10
        pkg = dict(zip(self.atoms, package.split('\t')))
        if 'arch' in pkg:
            try:
                pkg['arch'] = pkg['arch'].split(':')[2]
            except IndexError:
                pass
        if 'automatic' in pkg:
            pkg['automatic'] = bool(int(pkg['automatic']))
        if 'category' in pkg:
            pkg['category'] = pkg['category'].split('/', 1)[0]
        if 'version' in pkg:
            if ',' in pkg['version']:
                (pkg['version'], pkg['port_epoch']) = pkg['version'].split(',', 1)
            else:
                pkg['port_epoch'] = 0
            if '_' in pkg['version']:
                (pkg['version'], pkg['revision']) = pkg['version'].split('_', 1)
            else:
                pkg['revision'] = '0'
        if 'vital' in pkg:
            pkg['vital'] = bool(int(pkg['vital']))
        return pkg

class PORTAGE(CLIMgr):
    CLI = 'qlist'
    atoms = ['category', 'name', 'version', 'ebuild_revision', 'slots', 'prefixes', 'sufixes']

    def list_installed(self):
        if False:
            print('Hello World!')
        (rc, out, err) = module.run_command(' '.join([self._cli, '-Iv', '|', 'xargs', '-n', '1024', 'qatom']), use_unsafe_shell=True)
        if rc != 0:
            raise RuntimeError('Unable to list packages rc=%s : %s' % (rc, to_native(err)))
        return out.splitlines()

    def get_package_details(self, package):
        if False:
            return 10
        return dict(zip(self.atoms, package.split()))

class APK(CLIMgr):
    CLI = 'apk'

    def list_installed(self):
        if False:
            while True:
                i = 10
        (rc, out, err) = module.run_command([self._cli, 'info', '-v'])
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.splitlines()

    def get_package_details(self, package):
        if False:
            for i in range(10):
                print('nop')
        raw_pkg_details = {'name': package, 'version': '', 'release': ''}
        nvr = package.rsplit('-', 2)
        try:
            return {'name': nvr[0], 'version': nvr[1], 'release': nvr[2]}
        except IndexError:
            return raw_pkg_details

class PKG_INFO(CLIMgr):
    CLI = 'pkg_info'

    def list_installed(self):
        if False:
            return 10
        (rc, out, err) = module.run_command([self._cli, '-a'])
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.splitlines()

    def get_package_details(self, package):
        if False:
            while True:
                i = 10
        raw_pkg_details = {'name': package, 'version': ''}
        details = package.split(maxsplit=1)[0].rsplit('-', maxsplit=1)
        try:
            return {'name': details[0], 'version': details[1]}
        except IndexError:
            return raw_pkg_details

def main():
    if False:
        i = 10
        return i + 15
    PKG_MANAGERS = get_all_pkg_managers()
    PKG_MANAGER_NAMES = [x.lower() for x in PKG_MANAGERS.keys()]
    global module
    module = AnsibleModule(argument_spec=dict(manager={'type': 'list', 'elements': 'str', 'default': ['auto']}, strategy={'choices': ['first', 'all'], 'default': 'first'}), supports_check_mode=True)
    packages = {}
    results = {'ansible_facts': {}}
    managers = [x.lower() for x in module.params['manager']]
    strategy = module.params['strategy']
    if 'auto' in managers:
        managers.extend(PKG_MANAGER_NAMES)
        managers.remove('auto')
    unsupported = set(managers).difference(PKG_MANAGER_NAMES)
    if unsupported:
        if 'auto' in module.params['manager']:
            msg = 'Could not auto detect a usable package manager, check warnings for details.'
        else:
            msg = 'Unsupported package managers requested: %s' % ', '.join(unsupported)
        module.fail_json(msg=msg)
    found = 0
    seen = set()
    for pkgmgr in managers:
        if found and strategy == 'first':
            break
        if pkgmgr in seen:
            continue
        seen.add(pkgmgr)
        try:
            try:
                manager = PKG_MANAGERS[pkgmgr]()
                if manager.is_available():
                    found += 1
                    packages.update(manager.get_packages())
            except Exception as e:
                if pkgmgr in module.params['manager']:
                    module.warn('Requested package manager %s was not usable by this module: %s' % (pkgmgr, to_text(e)))
                continue
        except Exception as e:
            if pkgmgr in module.params['manager']:
                module.warn('Failed to retrieve packages with %s: %s' % (pkgmgr, to_text(e)))
    if found == 0:
        msg = 'Could not detect a supported package manager from the following list: %s, or the required Python library is not installed. Check warnings for details.' % managers
        module.fail_json(msg=msg)
    results['ansible_facts']['packages'] = packages
    module.exit_json(**results)
if __name__ == '__main__':
    main()