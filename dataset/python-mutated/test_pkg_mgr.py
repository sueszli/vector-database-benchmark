from __future__ import annotations
from ansible.module_utils.facts.system.pkg_mgr import PkgMgrFactCollector
_FEDORA_FACTS = {'ansible_distribution': 'Fedora', 'ansible_distribution_major_version': 38, 'ansible_os_family': 'RedHat'}
_KYLIN_FACTS = {'ansible_distribution': 'Kylin Linux Advanced Server', 'ansible_distribution_major_version': 'V10', 'ansible_os_family': 'RedHat'}

def test_default_dnf_version_detection_kylin_dnf4(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf', '/usr/bin/dnf-3'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/dnf': '/usr/bin/dnf-3'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_KYLIN_FACTS).get('pkg_mgr') == 'dnf'

def test_default_dnf_version_detection_fedora_dnf4(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf', '/usr/bin/dnf-3'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/dnf': '/usr/bin/dnf-3'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf'

def test_default_dnf_version_detection_fedora_dnf5(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf', '/usr/bin/dnf5'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/dnf': '/usr/bin/dnf5'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf5'

def test_default_dnf_version_detection_fedora_dnf4_both_installed(mocker):
    if False:
        return 10
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf', '/usr/bin/dnf-3', '/usr/bin/dnf5'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/dnf': '/usr/bin/dnf-3'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf'

def test_default_dnf_version_detection_fedora_dnf4_microdnf5_installed(mocker):
    if False:
        while True:
            i = 10
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf', '/usr/bin/microdnf', '/usr/bin/dnf-3', '/usr/bin/dnf5'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/dnf': '/usr/bin/dnf-3', '/usr/bin/microdnf': '/usr/bin/dnf5'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf'

def test_default_dnf_version_detection_fedora_dnf4_microdnf(mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('os.path.exists', lambda p: p == '/usr/bin/microdnf')
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf'

def test_default_dnf_version_detection_fedora_dnf5_microdnf(mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/microdnf', '/usr/bin/dnf5'))
    mocker.patch('os.path.realpath', lambda p: {'/usr/bin/microdnf': '/usr/bin/dnf5'}.get(p, p))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'dnf5'

def test_default_dnf_version_detection_fedora_no_default(mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('os.path.exists', lambda p: p in ('/usr/bin/dnf-3', '/usr/bin/dnf5'))
    assert PkgMgrFactCollector().collect(collected_facts=_FEDORA_FACTS).get('pkg_mgr') == 'unknown'