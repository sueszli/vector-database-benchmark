import pytest
from collections import namedtuple
from chalice.utils import OSUtils
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.deploy.packager import Package
from chalice.deploy.packager import PipRunner
from chalice.deploy.packager import SubprocessPip
from chalice.deploy.packager import InvalidSourceDistributionNameError
from chalice.deploy.packager import NoSuchPackageError
from chalice.deploy.packager import PackageDownloadError
FakePipCall = namedtuple('FakePipEntry', ['args', 'env_vars', 'shim'])

class FakePip(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._calls = []
        self._returns = []

    def main(self, args, env_vars=None, shim=None):
        if False:
            print('Hello World!')
        self._calls.append(FakePipCall(args, env_vars, shim))
        if self._returns:
            return self._returns.pop(0)
        return (0, b'', b'')

    def add_return(self, return_pair):
        if False:
            return 10
        self._returns.append(return_pair)

    @property
    def calls(self):
        if False:
            i = 10
            return i + 15
        return self._calls

@pytest.fixture
def pip_factory():
    if False:
        return 10

    def create_pip_runner(osutils=None):
        if False:
            print('Hello World!')
        pip = FakePip()
        pip_runner = PipRunner(pip, osutils=osutils)
        return (pip, pip_runner)
    return create_pip_runner

class CustomEnv(OSUtils):

    def __init__(self, env):
        if False:
            return 10
        self._env = env

    def environ(self):
        if False:
            while True:
                i = 10
        return self._env

@pytest.fixture
def osutils():
    if False:
        while True:
            i = 10
    return OSUtils()

class FakePopen(object):

    def __init__(self, rc, out, err):
        if False:
            for i in range(10):
                print('nop')
        self.returncode = 0
        self._out = out
        self._err = err

    def communicate(self):
        if False:
            while True:
                i = 10
        return (self._out, self._err)

class FakePopenOSUtils(OSUtils):

    def __init__(self, processes):
        if False:
            for i in range(10):
                print('nop')
        self.popens = []
        self._processes = processes

    def popen(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.popens.append((args, kwargs))
        return self._processes.pop()

class TestPackage(object):

    def test_can_create_package_with_custom_osutils(self, osutils):
        if False:
            i = 10
            return i + 15
        pkg = Package('', 'foobar-1.0-py3-none-any.whl', osutils)
        assert pkg._osutils == osutils

    def test_wheel_package(self):
        if False:
            for i in range(10):
                print('nop')
        filename = 'foobar-1.0-py3-none-any.whl'
        pkg = Package('', filename)
        assert pkg.dist_type == 'wheel'
        assert pkg.filename == filename
        assert pkg.identifier == 'foobar==1.0'
        assert str(pkg) == 'foobar==1.0(wheel)'

    def test_invalid_package(self):
        if False:
            return 10
        with pytest.raises(InvalidSourceDistributionNameError):
            Package('', 'foobar.jpg')

    def test_diff_pkg_sdist_and_whl_do_not_collide(self):
        if False:
            i = 10
            return i + 15
        pkgs = set()
        pkgs.add(Package('', 'foobar-1.0-py3-none-any.whl'))
        pkgs.add(Package('', 'badbaz-1.0-py3-none-any.whl'))
        assert len(pkgs) == 2

    def test_same_pkg_is_eq(self):
        if False:
            print('Hello World!')
        pkg = Package('', 'foobar-1.0-py3-none-any.whl')
        assert pkg == pkg

    def test_pkg_is_eq_to_similar_pkg(self):
        if False:
            while True:
                i = 10
        pure_pkg = Package('', 'foobar-1.0-py3-none-any.whl')
        plat_pkg = Package('', 'foobar-1.0-py3-py36m-manylinux1_x86_64.whl')
        assert pure_pkg == plat_pkg

    def test_pkg_is_not_equal_to_different_type(self):
        if False:
            print('Hello World!')
        pkg = Package('', 'foobar-1.0-py3-none-any.whl')
        non_package_type = 1
        assert not pkg == non_package_type

    def test_pkg_repr(self):
        if False:
            while True:
                i = 10
        pkg = Package('', 'foobar-1.0-py3-none-any.whl')
        assert repr(pkg) == 'foobar==1.0(wheel)'

    def test_wheel_data_dir(self):
        if False:
            print('Hello World!')
        pkg = Package('', 'foobar-2.0-py3-none-any.whl')
        assert pkg.data_dir == 'foobar-2.0.data'

    def test_can_read_packages_with_underscore_in_name(self):
        if False:
            print('Hello World!')
        pkg = Package('', 'foo_bar-2.0-py3-none-any.whl')
        assert pkg.identifier == 'foo-bar==2.0'

    def test_can_read_packages_with_period_in_name(self):
        if False:
            i = 10
            return i + 15
        pkg = Package('', 'foo.bar-2.0-py3-none-any.whl')
        assert pkg.identifier == 'foo-bar==2.0'

    def test_can_normalize_data_dir(self):
        if False:
            i = 10
            return i + 15
        pkg = Package('', 'Foobar-2.0-py3-none-any.whl')
        assert pkg.data_dir == 'foobar-2.0.data'

    def test_can_normalize_dirname_comparisons(self):
        if False:
            i = 10
            return i + 15
        pkg = Package('', 'Foobar-2.0-py3-none-any.whl')
        assert pkg.matches_data_dir('Foobar-2.0.data')
        assert pkg.matches_data_dir('foobar-2.0.data')
        assert not pkg.matches_data_dir('other-2.0.data')
        assert not pkg.matches_data_dir('foobar-2.0.datastuff')
        assert not pkg.matches_data_dir('foobar-2.0')

class TestPipRunner(object):

    def test_does_propagate_env_vars(self, pip_factory):
        if False:
            return 10
        osutils = CustomEnv({'foo': 'bar'})
        (pip, runner) = pip_factory(osutils)
        wheel = 'foobar-1.2-py3-none-any.whl'
        directory = 'directory'
        runner.build_wheel(wheel, directory)
        call = pip.calls[0]
        assert 'foo' in call.env_vars
        assert call.env_vars['foo'] == 'bar'

    def test_build_wheel(self, pip_factory):
        if False:
            return 10
        (pip, runner) = pip_factory()
        wheel = 'foobar-1.0-py3-none-any.whl'
        directory = 'directory'
        runner.build_wheel(wheel, directory)
        assert len(pip.calls) == 1
        call = pip.calls[0]
        assert call.args == ['wheel', '--no-deps', '--wheel-dir', directory, wheel]
        for compile_env_var in pip_no_compile_c_env_vars:
            assert compile_env_var not in call.env_vars
        assert call.shim == ''

    def test_build_wheel_without_c_extensions(self, pip_factory):
        if False:
            for i in range(10):
                print('nop')
        (pip, runner) = pip_factory()
        wheel = 'foobar-1.0-py3-none-any.whl'
        directory = 'directory'
        runner.build_wheel(wheel, directory, compile_c=False)
        assert len(pip.calls) == 1
        call = pip.calls[0]
        assert call.args == ['wheel', '--no-deps', '--wheel-dir', directory, wheel]
        for compile_env_var in pip_no_compile_c_env_vars:
            assert compile_env_var in call.env_vars
        assert call.shim == pip_no_compile_c_shim

    def test_download_all_deps(self, pip_factory):
        if False:
            while True:
                i = 10
        (pip, runner) = pip_factory()
        runner.download_all_dependencies('requirements.txt', 'directory')
        assert len(pip.calls) == 1
        call = pip.calls[0]
        assert call.args == ['download', '-r', 'requirements.txt', '--dest', 'directory']
        assert call.env_vars is None
        assert call.shim is None

    def test_download_sdist(self, pip_factory):
        if False:
            return 10
        (pip, runner) = pip_factory()
        packages = ['foo', 'bar', 'baz']
        runner.download_sdists(packages, 'directory')
        expected_prefix = ['download', '--no-binary=:all:', '--no-deps', '--dest', 'directory']
        for (i, package) in enumerate(packages):
            assert pip.calls[i].args == expected_prefix + [package]
            assert pip.calls[i].env_vars is None
            assert pip.calls[i].shim is None

    def test_download_wheels(self, pip_factory):
        if False:
            i = 10
            return i + 15
        (pip, runner) = pip_factory()
        packages = ['foo', 'bar', 'baz']
        abi = 'cp37m'
        runner.download_manylinux_wheels(abi, packages, 'directory')
        expected_prefix = ['download', '--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64', '--implementation', 'cp', '--abi', abi, '--dest', 'directory']
        for (i, package) in enumerate(packages):
            assert pip.calls[i].args == expected_prefix + [package]
            assert pip.calls[i].env_vars is None
            assert pip.calls[i].shim is None

    def test_download_wheels_no_wheels(self, pip_factory):
        if False:
            for i in range(10):
                print('nop')
        (pip, runner) = pip_factory()
        runner.download_manylinux_wheels('cp36m', [], 'directory')
        assert len(pip.calls) == 0

    def test_does_find_local_directory(self, pip_factory):
        if False:
            return 10
        (pip, runner) = pip_factory()
        pip.add_return((0, b'Processing ../local-dir\n  Link is a directory, ignoring download_dir', b''))
        runner.download_all_dependencies('requirements.txt', 'directory')
        assert len(pip.calls) == 2
        assert pip.calls[1].args == ['wheel', '--no-deps', '--wheel-dir', 'directory', '../local-dir']

    def test_does_find_multiple_local_directories(self, pip_factory):
        if False:
            i = 10
            return i + 15
        (pip, runner) = pip_factory()
        pip.add_return((0, b'Processing ../local-dir-1\n  Link is a directory, ignoring download_dir\nsome pip output...\nProcessing ../local-dir-2\n  Link is a directory, ignoring download_dir', b''))
        runner.download_all_dependencies('requirements.txt', 'directory')
        assert len(pip.calls) == 3
        assert pip.calls[1].args == ['wheel', '--no-deps', '--wheel-dir', 'directory', '../local-dir-1']
        assert pip.calls[2].args == ['wheel', '--no-deps', '--wheel-dir', 'directory', '../local-dir-2']

    def test_raise_no_such_package_error(self, pip_factory):
        if False:
            for i in range(10):
                print('nop')
        (pip, runner) = pip_factory()
        pip.add_return((1, b'', b'Could not find a version that satisfies the requirement BadPackageName '))
        with pytest.raises(NoSuchPackageError) as einfo:
            runner.download_all_dependencies('requirements.txt', 'directory')
        assert str(einfo.value) == 'Could not satisfy the requirement: BadPackageName'

    def test_raise_other_unknown_error_during_downloads(self, pip_factory):
        if False:
            return 10
        (pip, runner) = pip_factory()
        pip.add_return((1, b'', b'SomeNetworkingError: Details here.'))
        with pytest.raises(PackageDownloadError) as einfo:
            runner.download_all_dependencies('requirements.txt', 'directory')
        assert str(einfo.value) == 'SomeNetworkingError: Details here.'

    def test_inject_unknown_error_if_no_stderr(self, pip_factory):
        if False:
            print('Hello World!')
        (pip, runner) = pip_factory()
        pip.add_return((1, None, None))
        with pytest.raises(PackageDownloadError) as einfo:
            runner.download_all_dependencies('requirements.txt', 'directory')
        assert str(einfo.value) == 'Unknown error'

class TestSubprocessPip(object):

    def test_does_use_custom_pip_import_string(self):
        if False:
            return 10
        fake_osutils = FakePopenOSUtils([FakePopen(0, '', '')])
        expected_import_statement = 'foobarbaz'
        pip = SubprocessPip(osutils=fake_osutils, import_string=expected_import_statement)
        pip.main(['--version'])
        pip_execution_string = fake_osutils.popens[0][0][0][2]
        import_statement = pip_execution_string.split(';')[1].strip()
        assert import_statement == expected_import_statement