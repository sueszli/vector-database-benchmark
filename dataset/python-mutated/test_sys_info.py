from __future__ import annotations
import pytest
from unittest.mock import patch
import builtins
from ansible.module_utils.common.sys_info import get_distribution
from ansible.module_utils.common.sys_info import get_distribution_version
from ansible.module_utils.common.sys_info import get_platform_subclass
realimport = builtins.__import__

@pytest.fixture
def platform_linux(mocker):
    if False:
        return 10
    mocker.patch('platform.system', return_value='Linux')

@pytest.mark.parametrize(('system', 'dist'), (('Darwin', 'Darwin'), ('SunOS', 'Solaris'), ('FreeBSD', 'Freebsd')))
def test_get_distribution_not_linux(system, dist, mocker):
    if False:
        for i in range(10):
            print('nop')
    'For platforms other than Linux, return the distribution'
    mocker.patch('platform.system', return_value=system)
    mocker.patch('ansible.module_utils.common.sys_info.distro.id', return_value=dist)
    assert get_distribution() == dist

@pytest.mark.usefixtures('platform_linux')
class TestGetDistribution:
    """Tests for get_distribution that have to find something"""

    def test_distro_known(self):
        if False:
            while True:
                i = 10
        with patch('ansible.module_utils.distro.id', return_value='alpine'):
            assert get_distribution() == 'Alpine'
        with patch('ansible.module_utils.distro.id', return_value='arch'):
            assert get_distribution() == 'Arch'
        with patch('ansible.module_utils.distro.id', return_value='centos'):
            assert get_distribution() == 'Centos'
        with patch('ansible.module_utils.distro.id', return_value='clear-linux-os'):
            assert get_distribution() == 'Clear-linux-os'
        with patch('ansible.module_utils.distro.id', return_value='coreos'):
            assert get_distribution() == 'Coreos'
        with patch('ansible.module_utils.distro.id', return_value='debian'):
            assert get_distribution() == 'Debian'
        with patch('ansible.module_utils.distro.id', return_value='flatcar'):
            assert get_distribution() == 'Flatcar'
        with patch('ansible.module_utils.distro.id', return_value='linuxmint'):
            assert get_distribution() == 'Linuxmint'
        with patch('ansible.module_utils.distro.id', return_value='opensuse'):
            assert get_distribution() == 'Opensuse'
        with patch('ansible.module_utils.distro.id', return_value='oracle'):
            assert get_distribution() == 'Oracle'
        with patch('ansible.module_utils.distro.id', return_value='raspian'):
            assert get_distribution() == 'Raspian'
        with patch('ansible.module_utils.distro.id', return_value='rhel'):
            assert get_distribution() == 'Redhat'
        with patch('ansible.module_utils.distro.id', return_value='ubuntu'):
            assert get_distribution() == 'Ubuntu'
        with patch('ansible.module_utils.distro.id', return_value='virtuozzo'):
            assert get_distribution() == 'Virtuozzo'
        with patch('ansible.module_utils.distro.id', return_value='foo'):
            assert get_distribution() == 'Foo'

    def test_distro_unknown(self):
        if False:
            print('Hello World!')
        with patch('ansible.module_utils.distro.id', return_value=''):
            assert get_distribution() == 'OtherLinux'

    def test_distro_amazon_linux_short(self):
        if False:
            return 10
        with patch('ansible.module_utils.distro.id', return_value='amzn'):
            assert get_distribution() == 'Amazon'

    def test_distro_amazon_linux_long(self):
        if False:
            while True:
                i = 10
        with patch('ansible.module_utils.distro.id', return_value='amazon'):
            assert get_distribution() == 'Amazon'

@pytest.mark.parametrize(('system', 'version'), (('Darwin', '19.6.0'), ('SunOS', '11.4'), ('FreeBSD', '12.1')))
def test_get_distribution_version_not_linux(mocker, system, version):
    if False:
        return 10
    "If it's not Linux, then it has no distribution"
    mocker.patch('platform.system', return_value=system)
    mocker.patch('ansible.module_utils.common.sys_info.distro.version', return_value=version)
    assert get_distribution_version() == version

@pytest.mark.usefixtures('platform_linux')
def test_distro_found():
    if False:
        i = 10
        return i + 15
    with patch('ansible.module_utils.distro.version', return_value='1'):
        assert get_distribution_version() == '1'

class TestGetPlatformSubclass:

    class LinuxTest:
        pass

    class Foo(LinuxTest):
        platform = 'Linux'
        distribution = None

    class Bar(LinuxTest):
        platform = 'Linux'
        distribution = 'Bar'

    def test_not_linux(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('platform.system', return_value='Foo'):
            with patch('ansible.module_utils.common.sys_info.get_distribution', return_value=None):
                assert get_platform_subclass(self.LinuxTest) is self.LinuxTest

    @pytest.mark.usefixtures('platform_linux')
    def test_get_distribution_none(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('ansible.module_utils.common.sys_info.get_distribution', return_value=None):
            assert get_platform_subclass(self.LinuxTest) is self.Foo

    @pytest.mark.usefixtures('platform_linux')
    def test_get_distribution_found(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('ansible.module_utils.common.sys_info.get_distribution', return_value='Bar'):
            assert get_platform_subclass(self.LinuxTest) is self.Bar