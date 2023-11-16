"""Test class methods on Package objects.

This doesn't include methods on package *instances* (like do_install(),
etc.).  Only methods like ``possible_dependencies()`` that deal with the
static DSL metadata for packages.
"""
import os
import shutil
import pytest
import llnl.util.filesystem as fs
import spack.deptypes as dt
import spack.install_test
import spack.package_base
import spack.repo
from spack.build_systems.generic import Package
from spack.installer import InstallError

@pytest.fixture(scope='module')
def mpi_names(mock_repo_path):
    if False:
        i = 10
        return i + 15
    return [spec.name for spec in mock_repo_path.providers_for('mpi')]

@pytest.fixture()
def mpileaks_possible_deps(mock_packages, mpi_names):
    if False:
        return 10
    possible = {'callpath': set(['dyninst'] + mpi_names), 'low-priority-provider': set(), 'dyninst': set(['libdwarf', 'libelf']), 'fake': set(), 'intel-parallel-studio': set(), 'libdwarf': set(['libelf']), 'libelf': set(), 'mpich': set(), 'mpich2': set(), 'mpileaks': set(['callpath'] + mpi_names), 'multi-provider-mpi': set(), 'zmpi': set(['fake'])}
    return possible

def test_possible_dependencies(mock_packages, mpileaks_possible_deps):
    if False:
        for i in range(10):
            print('nop')
    pkg_cls = spack.repo.PATH.get_pkg_class('mpileaks')
    expanded_possible_deps = pkg_cls.possible_dependencies(expand_virtuals=True)
    assert mpileaks_possible_deps == expanded_possible_deps
    assert {'callpath': {'dyninst', 'mpi'}, 'dyninst': {'libdwarf', 'libelf'}, 'libdwarf': {'libelf'}, 'libelf': set(), 'mpi': set(), 'mpileaks': {'callpath', 'mpi'}} == pkg_cls.possible_dependencies(expand_virtuals=False)

def test_possible_direct_dependencies(mock_packages, mpileaks_possible_deps):
    if False:
        for i in range(10):
            print('nop')
    pkg_cls = spack.repo.PATH.get_pkg_class('mpileaks')
    deps = pkg_cls.possible_dependencies(transitive=False, expand_virtuals=False)
    assert {'callpath': set(), 'mpi': set(), 'mpileaks': {'callpath', 'mpi'}} == deps

def test_possible_dependencies_virtual(mock_packages, mpi_names):
    if False:
        return 10
    expected = dict(((name, set(spack.repo.PATH.get_pkg_class(name).dependencies)) for name in mpi_names))
    expected['fake'] = set()
    assert expected == spack.package_base.possible_dependencies('mpi', transitive=False)

def test_possible_dependencies_missing(mock_packages):
    if False:
        for i in range(10):
            print('nop')
    pkg_cls = spack.repo.PATH.get_pkg_class('missing-dependency')
    missing = {}
    pkg_cls.possible_dependencies(transitive=True, missing=missing)
    assert {'this-is-a-missing-dependency'} == missing['missing-dependency']

def test_possible_dependencies_with_deptypes(mock_packages):
    if False:
        i = 10
        return i + 15
    dtbuild1 = spack.repo.PATH.get_pkg_class('dtbuild1')
    assert {'dtbuild1': {'dtrun2', 'dtlink2'}, 'dtlink2': set(), 'dtrun2': set()} == dtbuild1.possible_dependencies(depflag=dt.LINK | dt.RUN)
    assert {'dtbuild1': {'dtbuild2', 'dtlink2'}, 'dtbuild2': set(), 'dtlink2': set()} == dtbuild1.possible_dependencies(depflag=dt.BUILD)
    assert {'dtbuild1': {'dtlink2'}, 'dtlink2': set()} == dtbuild1.possible_dependencies(depflag=dt.LINK)

def test_possible_dependencies_with_multiple_classes(mock_packages, mpileaks_possible_deps):
    if False:
        i = 10
        return i + 15
    pkgs = ['dt-diamond', 'mpileaks']
    expected = mpileaks_possible_deps.copy()
    expected.update({'dt-diamond': set(['dt-diamond-left', 'dt-diamond-right']), 'dt-diamond-left': set(['dt-diamond-bottom']), 'dt-diamond-right': set(['dt-diamond-bottom']), 'dt-diamond-bottom': set()})
    assert expected == spack.package_base.possible_dependencies(*pkgs)

def setup_install_test(source_paths, test_root):
    if False:
        return 10
    '\n    Set up the install test by creating sources and install test roots.\n\n    The convention used here is to create an empty file if the path name\n    ends with an extension otherwise, a directory is created.\n    '
    fs.mkdirp(test_root)
    for path in source_paths:
        if os.path.splitext(path)[1]:
            fs.touchp(path)
        else:
            fs.mkdirp(path)

@pytest.mark.parametrize('spec,sources,extras,expect', [('a', ['example/a.c'], ['example/a.c'], ['example/a.c']), ('b', ['test/b.cpp', 'test/b.hpp', 'example/b.txt'], ['test'], ['test/b.cpp', 'test/b.hpp']), ('c', ['examples/a.py', 'examples/b.py', 'examples/c.py', 'tests/d.py'], ['examples/b.py', 'tests'], ['examples/b.py', 'tests/d.py'])])
def test_cache_extra_sources(install_mockery, spec, sources, extras, expect):
    if False:
        while True:
            i = 10
    "Test the package's cache extra test sources helper function."
    s = spack.spec.Spec(spec).concretized()
    s.package.spec.concretize()
    source_path = s.package.stage.source_path
    srcs = [fs.join_path(source_path, src) for src in sources]
    test_root = spack.install_test.install_test_root(s.package)
    setup_install_test(srcs, test_root)
    emsg_dir = 'Expected {0} to be a directory'
    emsg_file = 'Expected {0} to be a file'
    for src in srcs:
        assert os.path.exists(src), 'Expected {0} to exist'.format(src)
        if os.path.splitext(src)[1]:
            assert os.path.isfile(src), emsg_file.format(src)
        else:
            assert os.path.isdir(src), emsg_dir.format(src)
    spack.install_test.cache_extra_test_sources(s.package, extras)
    src_dests = [fs.join_path(test_root, src) for src in sources]
    exp_dests = [fs.join_path(test_root, e) for e in expect]
    poss_dests = set(src_dests) | set(exp_dests)
    msg = 'Expected {0} to{1} exist'
    for pd in poss_dests:
        if pd in exp_dests:
            assert os.path.exists(pd), msg.format(pd, '')
            if os.path.splitext(pd)[1]:
                assert os.path.isfile(pd), emsg_file.format(pd)
            else:
                assert os.path.isdir(pd), emsg_dir.format(pd)
        else:
            assert not os.path.exists(pd), msg.format(pd, ' not')
    shutil.rmtree(os.path.dirname(source_path))

def test_cache_extra_sources_fails(install_mockery):
    if False:
        print('Hello World!')
    s = spack.spec.Spec('a').concretized()
    s.package.spec.concretize()
    with pytest.raises(InstallError) as exc_info:
        spack.install_test.cache_extra_test_sources(s.package, ['/a/b', 'no-such-file'])
    errors = str(exc_info.value)
    assert "'/a/b') must be relative" in errors
    assert "'no-such-file') for the copy does not exist" in errors

def test_package_exes_and_libs():
    if False:
        i = 10
        return i + 15
    with pytest.raises(spack.error.SpackError, match='defines both'):

        class BadDetectablePackage(spack.package.Package):
            executables = ['findme']
            libraries = ['libFindMe.a']

def test_package_url_and_urls():
    if False:
        for i in range(10):
            print('nop')

    class URLsPackage(spack.package.Package):
        url = 'https://www.example.com/url-package-1.0.tgz'
        urls = ['https://www.example.com/archive']
    s = spack.spec.Spec('a')
    with pytest.raises(ValueError, match='defines both'):
        URLsPackage(s)

def test_package_license():
    if False:
        return 10

    class LicensedPackage(spack.package.Package):
        extendees = None
        license_files = None
    s = spack.spec.Spec('a')
    pkg = LicensedPackage(s)
    assert pkg.global_license_file is None
    pkg.license_files = ['license.txt']
    assert os.path.basename(pkg.global_license_file) == pkg.license_files[0]

class BaseTestPackage(Package):
    extendees = None

def test_package_version_fails():
    if False:
        for i in range(10):
            print('nop')
    s = spack.spec.Spec('a')
    pkg = BaseTestPackage(s)
    with pytest.raises(ValueError, match='does not have a concrete version'):
        pkg.version()

def test_package_tester_fails():
    if False:
        return 10
    s = spack.spec.Spec('a')
    pkg = BaseTestPackage(s)
    with pytest.raises(ValueError, match='without concrete version'):
        pkg.tester()

def test_package_fetcher_fails():
    if False:
        return 10
    s = spack.spec.Spec('a')
    pkg = BaseTestPackage(s)
    with pytest.raises(ValueError, match='without concrete version'):
        pkg.fetcher

def test_package_no_extendees():
    if False:
        for i in range(10):
            print('nop')
    s = spack.spec.Spec('a')
    pkg = BaseTestPackage(s)
    assert pkg.extendee_args is None

def test_package_test_no_compilers(mock_packages, monkeypatch, capfd):
    if False:
        i = 10
        return i + 15

    def compilers(compiler, arch_spec):
        if False:
            for i in range(10):
                print('nop')
        return None
    monkeypatch.setattr(spack.compilers, 'compilers_for_spec', compilers)
    s = spack.spec.Spec('a')
    pkg = BaseTestPackage(s)
    pkg.test_requires_compiler = True
    pkg.do_test()
    error = capfd.readouterr()[1]
    assert 'Skipping tests for package' in error
    assert 'test requires missing compiler' in error

@pytest.mark.parametrize('msg,installed,purpose,expected', [('do-nothing', False, 'test: echo', 'do-nothing'), ('not installed', True, 'test: echo not installed', 'expected in prefix')])
def test_package_run_test_install(install_mockery_mutable_config, mock_fetch, capfd, msg, installed, purpose, expected):
    if False:
        while True:
            i = 10
    'Confirm expected outputs from run_test for installed/not installed exe.'
    s = spack.spec.Spec('trivial-smoke-test').concretized()
    pkg = s.package
    pkg.run_test('echo', msg, expected=[expected], installed=installed, purpose=purpose, work_dir='.')
    output = capfd.readouterr()[0]
    assert expected in output

@pytest.mark.parametrize('skip,failures,status', [(True, 0, str(spack.install_test.TestStatus.SKIPPED)), (False, 1, str(spack.install_test.TestStatus.FAILED))])
def test_package_run_test_missing(install_mockery_mutable_config, mock_fetch, capfd, skip, failures, status):
    if False:
        for i in range(10):
            print('nop')
    'Confirm expected results from run_test for missing exe when skip or not.'
    s = spack.spec.Spec('trivial-smoke-test').concretized()
    pkg = s.package
    pkg.run_test('no-possible-program', skip_missing=skip)
    output = capfd.readouterr()[0]
    assert len(pkg.tester.test_failures) == failures
    assert status in output

def test_package_run_test_fail_fast(install_mockery_mutable_config, mock_fetch):
    if False:
        return 10
    'Confirm expected exception when run_test with fail_fast enabled.'
    s = spack.spec.Spec('trivial-smoke-test').concretized()
    pkg = s.package
    with spack.config.override('config:fail_fast', True):
        with pytest.raises(spack.install_test.TestFailure, match='Failed to find executable'):
            pkg.run_test('no-possible-program')