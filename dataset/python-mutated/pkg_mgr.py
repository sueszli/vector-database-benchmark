from __future__ import annotations
import os
import subprocess
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
PKG_MGRS = [{'path': '/usr/bin/rpm-ostree', 'name': 'atomic_container'}, {'path': '/usr/bin/yum', 'name': 'yum'}, {'path': '/usr/bin/dnf-3', 'name': 'dnf'}, {'path': '/usr/bin/dnf5', 'name': 'dnf5'}, {'path': '/usr/bin/apt-get', 'name': 'apt'}, {'path': '/usr/bin/zypper', 'name': 'zypper'}, {'path': '/usr/sbin/urpmi', 'name': 'urpmi'}, {'path': '/usr/bin/pacman', 'name': 'pacman'}, {'path': '/bin/opkg', 'name': 'opkg'}, {'path': '/usr/pkg/bin/pkgin', 'name': 'pkgin'}, {'path': '/opt/local/bin/pkgin', 'name': 'pkgin'}, {'path': '/opt/tools/bin/pkgin', 'name': 'pkgin'}, {'path': '/opt/local/bin/port', 'name': 'macports'}, {'path': '/usr/local/bin/brew', 'name': 'homebrew'}, {'path': '/opt/homebrew/bin/brew', 'name': 'homebrew'}, {'path': '/sbin/apk', 'name': 'apk'}, {'path': '/usr/sbin/pkg', 'name': 'pkgng'}, {'path': '/usr/sbin/swlist', 'name': 'swdepot'}, {'path': '/usr/bin/emerge', 'name': 'portage'}, {'path': '/usr/sbin/pkgadd', 'name': 'svr4pkg'}, {'path': '/usr/bin/pkg', 'name': 'pkg5'}, {'path': '/usr/bin/xbps-install', 'name': 'xbps'}, {'path': '/usr/local/sbin/pkg', 'name': 'pkgng'}, {'path': '/usr/bin/swupd', 'name': 'swupd'}, {'path': '/usr/sbin/sorcery', 'name': 'sorcery'}, {'path': '/usr/bin/installp', 'name': 'installp'}, {'path': '/QOpenSys/pkgs/bin/yum', 'name': 'yum'}]

class OpenBSDPkgMgrFactCollector(BaseFactCollector):
    name = 'pkg_mgr'
    _fact_ids = set()
    _platform = 'OpenBSD'

    def collect(self, module=None, collected_facts=None):
        if False:
            while True:
                i = 10
        return {'pkg_mgr': 'openbsd_pkg'}

class PkgMgrFactCollector(BaseFactCollector):
    name = 'pkg_mgr'
    _fact_ids = set()
    _platform = 'Generic'
    required_facts = set(['distribution'])

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(PkgMgrFactCollector, self).__init__(*args, **kwargs)
        self._default_unknown_pkg_mgr = 'unknown'

    def _check_rh_versions(self, pkg_mgr_name, collected_facts):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists('/run/ostree-booted'):
            return 'atomic_container'
        pkg_mgr_name = self._default_unknown_pkg_mgr
        for bin_path in ('/usr/bin/dnf', '/usr/bin/microdnf'):
            if os.path.exists(bin_path):
                pkg_mgr_name = 'dnf5' if os.path.realpath(bin_path) == '/usr/bin/dnf5' else 'dnf'
                break
        try:
            major_version = collected_facts['ansible_distribution_major_version']
            if collected_facts['ansible_distribution'] == 'Kylin Linux Advanced Server':
                major_version = major_version.lstrip('V')
            distro_major_ver = int(major_version)
        except ValueError:
            return self._default_unknown_pkg_mgr
        if (collected_facts['ansible_distribution'] == 'Fedora' and distro_major_ver < 23 or (collected_facts['ansible_distribution'] == 'Kylin Linux Advanced Server' and distro_major_ver < 10) or (collected_facts['ansible_distribution'] == 'Amazon' and distro_major_ver < 2022) or (collected_facts['ansible_distribution'] == 'TencentOS' and distro_major_ver < 3) or (distro_major_ver < 8)) and any((pm for pm in PKG_MGRS if pm['name'] == 'yum' and os.path.exists(pm['path']))):
            pkg_mgr_name = 'yum'
        return pkg_mgr_name

    def _check_apt_flavor(self, pkg_mgr_name):
        if False:
            for i in range(10):
                print('nop')
        rpm_query = '/usr/bin/rpm -q --whatprovides /usr/bin/apt-get'.split()
        if os.path.exists('/usr/bin/rpm'):
            with open(os.devnull, 'w') as null:
                try:
                    subprocess.check_call(rpm_query, stdout=null, stderr=null)
                    pkg_mgr_name = 'apt_rpm'
                except subprocess.CalledProcessError:
                    pkg_mgr_name = 'apt'
        return pkg_mgr_name

    def pkg_mgrs(self, collected_facts):
        if False:
            i = 10
            return i + 15
        if collected_facts['ansible_os_family'] == 'Altlinux':
            return filter(lambda pkg: pkg['path'] != '/usr/bin/pkg', PKG_MGRS)
        else:
            return PKG_MGRS

    def collect(self, module=None, collected_facts=None):
        if False:
            while True:
                i = 10
        collected_facts = collected_facts or {}
        pkg_mgr_name = self._default_unknown_pkg_mgr
        for pkg in self.pkg_mgrs(collected_facts):
            if os.path.exists(pkg['path']):
                pkg_mgr_name = pkg['name']
        if collected_facts['ansible_os_family'] == 'RedHat':
            pkg_mgr_name = self._check_rh_versions(pkg_mgr_name, collected_facts)
        elif collected_facts['ansible_os_family'] == 'Debian' and pkg_mgr_name != 'apt':
            pkg_mgr_name = 'apt'
        elif collected_facts['ansible_os_family'] == 'Altlinux':
            if pkg_mgr_name == 'apt':
                pkg_mgr_name = 'apt_rpm'
        if pkg_mgr_name == 'apt':
            pkg_mgr_name = self._check_apt_flavor(pkg_mgr_name)
        return {'pkg_mgr': pkg_mgr_name}