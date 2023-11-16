import os
import pathlib
import pytest
import spack.build_systems.generic
import spack.config
import spack.error
import spack.package_base
import spack.repo
import spack.util.spack_yaml as syaml
import spack.version
from spack.solver.asp import InternalConcretizerError, UnsatisfiableSpecError
from spack.spec import Spec
from spack.util.url import path_to_file_url
pytestmark = [pytest.mark.not_on_windows('Windows uses old concretizer'), pytest.mark.only_clingo('Original concretizer does not support configuration requirements')]

def update_packages_config(conf_str):
    if False:
        print('Hello World!')
    conf = syaml.load_config(conf_str)
    spack.config.set('packages', conf['packages'], scope='concretize')
_pkgx = ('x', 'class X(Package):\n    version("1.1")\n    version("1.0")\n    version("0.9")\n\n    variant("shared", default=True,\n            description="Build shared libraries")\n\n    depends_on("y")\n')
_pkgy = ('y', 'class Y(Package):\n    version("2.5")\n    version("2.4")\n    version("2.3", deprecated=True)\n\n    variant("shared", default=True,\n            description="Build shared libraries")\n')
_pkgv = ('v', 'class V(Package):\n    version("2.1")\n    version("2.0")\n')
_pkgt = ('t', "class T(Package):\n    version('2.1')\n    version('2.0')\n\n    depends_on('u', when='@2.1:')\n")
_pkgu = ('u', "class U(Package):\n    version('1.1')\n    version('1.0')\n")

@pytest.fixture
def create_test_repo(tmpdir, mutable_config):
    if False:
        return 10
    repo_path = str(tmpdir)
    repo_yaml = tmpdir.join('repo.yaml')
    with open(str(repo_yaml), 'w') as f:
        f.write('repo:\n  namespace: testcfgrequirements\n')
    packages_dir = tmpdir.join('packages')
    for (pkg_name, pkg_str) in [_pkgx, _pkgy, _pkgv, _pkgt, _pkgu]:
        pkg_dir = packages_dir.ensure(pkg_name, dir=True)
        pkg_file = pkg_dir.join('package.py')
        with open(str(pkg_file), 'w') as f:
            f.write(pkg_str)
    yield spack.repo.Repo(repo_path)

@pytest.fixture
def test_repo(create_test_repo, monkeypatch, mock_stage):
    if False:
        for i in range(10):
            print('nop')
    with spack.repo.use_repositories(create_test_repo) as mock_repo_path:
        yield mock_repo_path

class MakeStage:

    def __init__(self, stage):
        if False:
            print('Hello World!')
        self.stage = stage

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.stage

@pytest.fixture
def fake_installs(monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    stage_path = str(tmpdir.ensure('fake-stage', dir=True))
    universal_unused_stage = spack.stage.DIYStage(stage_path)
    monkeypatch.setattr(spack.build_systems.generic.Package, '_make_stage', MakeStage(universal_unused_stage))

def test_one_package_multiple_reqs(concretize_scope, test_repo):
    if False:
        i = 10
        return i + 15
    conf_str = 'packages:\n  y:\n    require:\n    - "@2.4"\n    - "~shared"\n'
    update_packages_config(conf_str)
    y_spec = Spec('y').concretized()
    assert y_spec.satisfies('@2.4~shared')

def test_requirement_isnt_optional(concretize_scope, test_repo):
    if False:
        while True:
            i = 10
    'If a user spec requests something that directly conflicts\n    with a requirement, make sure we get an error.\n    '
    conf_str = 'packages:\n  x:\n    require: "@1.0"\n'
    update_packages_config(conf_str)
    with pytest.raises(UnsatisfiableSpecError):
        Spec('x@1.1').concretize()

def test_require_undefined_version(concretize_scope, test_repo):
    if False:
        for i in range(10):
            print('nop')
    "If a requirement specifies a numbered version that isn't in\n    the associated package.py and isn't part of a Git hash\n    equivalence (hash=number), then Spack should raise an error\n    (it is assumed this is a typo, and raising the error here\n    avoids a likely error when Spack attempts to fetch the version).\n    "
    conf_str = 'packages:\n  x:\n    require: "@1.2"\n'
    update_packages_config(conf_str)
    with pytest.raises(spack.config.ConfigError):
        Spec('x').concretize()

def test_require_truncated(concretize_scope, test_repo):
    if False:
        for i in range(10):
            print('nop')
    'A requirement specifies a version range, with satisfying\n    versions defined in the package.py. Make sure we choose one\n    of the defined versions (vs. allowing the requirement to\n    define a new version).\n    '
    conf_str = 'packages:\n  x:\n    require: "@1"\n'
    update_packages_config(conf_str)
    xspec = Spec('x').concretized()
    assert xspec.satisfies('@1.1')

def test_git_user_supplied_reference_satisfaction(concretize_scope, test_repo, mock_git_version_info, monkeypatch):
    if False:
        return 10
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    hash_eq_ver = Spec(f'v@{commits[0]}=2.2')
    hash_eq_ver_copy = Spec(f'v@{commits[0]}=2.2')
    just_hash = Spec(f'v@{commits[0]}')
    just_ver = Spec('v@=2.2')
    hash_eq_other_ver = Spec(f'v@{commits[0]}=2.3')
    assert not hash_eq_ver == just_hash
    assert not hash_eq_ver.satisfies(just_hash)
    assert not hash_eq_ver.intersects(just_hash)
    assert not hash_eq_ver.satisfies(just_ver)
    assert not just_ver.satisfies(hash_eq_ver)
    assert not hash_eq_ver.intersects(just_ver)
    assert hash_eq_ver != just_ver
    assert just_ver != hash_eq_ver
    assert not hash_eq_ver == just_ver
    assert not just_ver == hash_eq_ver
    assert not hash_eq_ver.satisfies(hash_eq_other_ver)
    assert not hash_eq_other_ver.satisfies(hash_eq_ver)
    assert not hash_eq_ver.intersects(hash_eq_other_ver)
    assert not hash_eq_other_ver.intersects(hash_eq_ver)
    assert hash_eq_ver != hash_eq_other_ver
    assert hash_eq_other_ver != hash_eq_ver
    assert not hash_eq_ver == hash_eq_other_ver
    assert not hash_eq_other_ver == hash_eq_ver
    assert hash_eq_ver == hash_eq_ver_copy
    assert not hash_eq_ver != hash_eq_ver_copy
    assert hash_eq_ver.satisfies(hash_eq_ver_copy)
    assert hash_eq_ver_copy.satisfies(hash_eq_ver)
    assert hash_eq_ver.intersects(hash_eq_ver_copy)
    assert hash_eq_ver_copy.intersects(hash_eq_ver)

def test_requirement_adds_new_version(concretize_scope, test_repo, mock_git_version_info, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    a_commit_hash = commits[0]
    conf_str = 'packages:\n  v:\n    require: "@{0}=2.2"\n'.format(a_commit_hash)
    update_packages_config(conf_str)
    s1 = Spec('v').concretized()
    assert s1.satisfies('@2.2')
    assert isinstance(s1.version, spack.version.GitVersion)
    assert s1.version.ref == a_commit_hash

def test_requirement_adds_version_satisfies(concretize_scope, test_repo, mock_git_version_info, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Make sure that new versions added by requirements are factored into\n    conditions. In this case create a new version that satisfies a\n    depends_on condition and make sure it is triggered (i.e. the\n    dependency is added).\n    '
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    s0 = Spec('t@2.0').concretized()
    assert not 'u' in s0
    conf_str = 'packages:\n  t:\n    require: "@{0}=2.2"\n'.format(commits[0])
    update_packages_config(conf_str)
    s1 = Spec('t').concretized()
    assert 'u' in s1
    assert s1.satisfies('@2.2')

@pytest.mark.parametrize('require_checksum', (True, False))
def test_requirement_adds_git_hash_version(require_checksum, concretize_scope, test_repo, mock_git_version_info, monkeypatch, working_env):
    if False:
        while True:
            i = 10
    if require_checksum:
        os.environ['SPACK_CONCRETIZER_REQUIRE_CHECKSUM'] = 'yes'
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    a_commit_hash = commits[0]
    conf_str = f'packages:\n  v:\n    require: "@{a_commit_hash}"\n'
    update_packages_config(conf_str)
    s1 = Spec('v').concretized()
    assert isinstance(s1.version, spack.version.GitVersion)
    assert s1.satisfies(f'v@{a_commit_hash}')

def test_requirement_adds_multiple_new_versions(concretize_scope, test_repo, mock_git_version_info, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    conf_str = f'packages:\n  v:\n    require:\n    - one_of: ["@{commits[0]}=2.2", "@{commits[1]}=2.3"]\n'
    update_packages_config(conf_str)
    assert Spec('v').concretized().satisfies(f'@{commits[0]}=2.2')
    assert Spec('v@2.3').concretized().satisfies(f'v@{commits[1]}=2.3')

def test_preference_adds_new_version(concretize_scope, test_repo, mock_git_version_info, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Normally a preference cannot define a new version, but that constraint\n    is ignored if the version is a Git hash-based version.\n    '
    (repo_path, filename, commits) = mock_git_version_info
    monkeypatch.setattr(spack.package_base.PackageBase, 'git', path_to_file_url(repo_path), raising=False)
    conf_str = f'packages:\n  v:\n    version: ["{commits[0]}=2.2", "{commits[1]}=2.3"]\n'
    update_packages_config(conf_str)
    assert Spec('v').concretized().satisfies(f'@{commits[0]}=2.2')
    assert Spec('v@2.3').concretized().satisfies(f'@{commits[1]}=2.3')
    s3 = Spec(f'v@{commits[1]}').concretized()
    assert s3.satisfies(f'v@{commits[1]}')
    assert not s3.satisfies('@2.3')

def test_external_adds_new_version_that_is_preferred(concretize_scope, test_repo):
    if False:
        while True:
            i = 10
    'Test that we can use a version, not declared in package recipe, as the\n    preferred version if that version appears in an external spec.\n    '
    conf_str = 'packages:\n  y:\n    version: ["2.7"]\n    externals:\n    - spec: y@2.7 # Not defined in y\n      prefix: /fake/nonexistent/path/\n    buildable: false\n'
    update_packages_config(conf_str)
    spec = Spec('x').concretized()
    assert spec['y'].satisfies('@2.7')
    assert spack.version.Version('2.7') not in spec['y'].package.versions

def test_requirement_is_successfully_applied(concretize_scope, test_repo):
    if False:
        while True:
            i = 10
    'If a simple requirement can be satisfied, make sure the\n    concretization succeeds and the requirement spec is applied.\n    '
    s1 = Spec('x').concretized()
    assert s1.satisfies('@1.1')
    conf_str = 'packages:\n  x:\n    require: "@1.0"\n'
    update_packages_config(conf_str)
    s2 = Spec('x').concretized()
    assert s2.satisfies('@1.0')

def test_multiple_packages_requirements_are_respected(concretize_scope, test_repo):
    if False:
        i = 10
        return i + 15
    'Apply requirements to two packages; make sure the concretization\n    succeeds and both requirements are respected.\n    '
    conf_str = 'packages:\n  x:\n    require: "@1.0"\n  y:\n    require: "@2.4"\n'
    update_packages_config(conf_str)
    spec = Spec('x').concretized()
    assert spec['x'].satisfies('@1.0')
    assert spec['y'].satisfies('@2.4')

def test_oneof(concretize_scope, test_repo):
    if False:
        i = 10
        return i + 15
    "'one_of' allows forcing the concretizer to satisfy one of\n    the specs in the group (but not all have to be satisfied).\n    "
    conf_str = 'packages:\n  y:\n    require:\n    - one_of: ["@2.4", "~shared"]\n'
    update_packages_config(conf_str)
    spec = Spec('x').concretized()
    assert spec['y'].satisfies('@2.4+shared')

def test_one_package_multiple_oneof_groups(concretize_scope, test_repo):
    if False:
        i = 10
        return i + 15
    "One package has two 'one_of' groups; check that both are\n    applied.\n    "
    conf_str = 'packages:\n  y:\n    require:\n    - one_of: ["@2.4%gcc", "@2.5%clang"]\n    - one_of: ["@2.5~shared", "@2.4+shared"]\n'
    update_packages_config(conf_str)
    s1 = Spec('y@2.5').concretized()
    assert s1.satisfies('%clang~shared')
    s2 = Spec('y@2.4').concretized()
    assert s2.satisfies('%gcc+shared')

@pytest.mark.regression('34241')
def test_require_cflags(concretize_scope, mock_packages):
    if False:
        return 10
    'Ensures that flags can be required from configuration.'
    conf_str = 'packages:\n  mpich2:\n    require: cflags="-g"\n  mpi:\n    require: mpich cflags="-O1"\n'
    update_packages_config(conf_str)
    spec_mpich2 = Spec('mpich2').concretized()
    assert spec_mpich2.satisfies('cflags=-g')
    spec_mpi = Spec('mpi').concretized()
    assert spec_mpi.satisfies('mpich cflags=-O1')

def test_requirements_for_package_that_is_not_needed(concretize_scope, test_repo):
    if False:
        print('Hello World!')
    'Specify requirements for specs that are not concretized or\n    a dependency of a concretized spec (in other words, none of\n    the requirements are used for the requested spec).\n    '
    conf_str = 'packages:\n  x:\n    require: "@1.0"\n  y:\n    require:\n    - one_of: ["@2.4%gcc", "@2.5%clang"]\n    - one_of: ["@2.5~shared", "@2.4+shared"]\n'
    update_packages_config(conf_str)
    s1 = Spec('v').concretized()
    assert s1.satisfies('@2.1')

def test_oneof_ordering(concretize_scope, test_repo):
    if False:
        i = 10
        return i + 15
    "Ensure that earlier elements of 'one_of' have higher priority.\n    This priority should override default priority (e.g. choosing\n    later versions).\n    "
    conf_str = 'packages:\n  y:\n    require:\n    - one_of: ["@2.4", "@2.5"]\n'
    update_packages_config(conf_str)
    s1 = Spec('y').concretized()
    assert s1.satisfies('@2.4')
    s2 = Spec('y@2.5').concretized()
    assert s2.satisfies('@2.5')

def test_reuse_oneof(concretize_scope, create_test_repo, mutable_database, fake_installs):
    if False:
        i = 10
        return i + 15
    conf_str = 'packages:\n  y:\n    require:\n    - one_of: ["@2.5", "%gcc"]\n'
    with spack.repo.use_repositories(create_test_repo):
        s1 = Spec('y@2.5%gcc').concretized()
        s1.package.do_install(fake=True, explicit=True)
        update_packages_config(conf_str)
        with spack.config.override('concretizer:reuse', True):
            s2 = Spec('y').concretized()
            assert not s2.satisfies('@2.5 %gcc')

@pytest.mark.parametrize('allow_deprecated,expected,not_expected', [(True, ['@=2.3', '%gcc'], []), (False, ['%gcc'], ['@=2.3'])])
def test_requirements_and_deprecated_versions(allow_deprecated, expected, not_expected, concretize_scope, test_repo):
    if False:
        for i in range(10):
            print('nop')
    'Tests the expected behavior of requirements and deprecated versions.\n\n    If deprecated versions are not allowed, concretization should just pick\n    the other requirement.\n\n    If deprecated versions are allowed, both requirements are honored.\n    '
    conf_str = 'packages:\n  y:\n    require:\n    - any_of: ["@=2.3", "%gcc"]\n'
    update_packages_config(conf_str)
    with spack.config.override('config:deprecated', allow_deprecated):
        s1 = Spec('y').concretized()
        for constrain in expected:
            assert s1.satisfies(constrain)
        for constrain in not_expected:
            assert not s1.satisfies(constrain)

@pytest.mark.parametrize('spec_str,requirement_str', [('x', '%gcc'), ('x', '%clang')])
def test_default_requirements_with_all(spec_str, requirement_str, concretize_scope, test_repo):
    if False:
        for i in range(10):
            print('nop')
    'Test that default requirements are applied to all packages.'
    conf_str = 'packages:\n  all:\n    require: "{}"\n'.format(requirement_str)
    update_packages_config(conf_str)
    spec = Spec(spec_str).concretized()
    for s in spec.traverse():
        assert s.satisfies(requirement_str)

@pytest.mark.parametrize('requirements,expectations', [(('%gcc', '%clang'), ('%gcc', '%clang')), (('%gcc ~shared', '@1.0'), ('%gcc ~shared', '@1.0 +shared'))])
def test_default_and_package_specific_requirements(concretize_scope, requirements, expectations, test_repo):
    if False:
        print('Hello World!')
    'Test that specific package requirements override default package requirements.'
    (generic_req, specific_req) = requirements
    (generic_exp, specific_exp) = expectations
    conf_str = 'packages:\n  all:\n    require: "{}"\n  x:\n    require: "{}"\n'.format(generic_req, specific_req)
    update_packages_config(conf_str)
    spec = Spec('x').concretized()
    assert spec.satisfies(specific_exp)
    for s in spec.traverse(root=False):
        assert s.satisfies(generic_exp)

@pytest.mark.parametrize('mpi_requirement', ['mpich', 'mpich2', 'zmpi'])
def test_requirements_on_virtual(mpi_requirement, concretize_scope, mock_packages):
    if False:
        for i in range(10):
            print('nop')
    conf_str = 'packages:\n  mpi:\n    require: "{}"\n'.format(mpi_requirement)
    update_packages_config(conf_str)
    spec = Spec('callpath').concretized()
    assert 'mpi' in spec
    assert mpi_requirement in spec

@pytest.mark.parametrize('mpi_requirement,specific_requirement', [('mpich', '@3.0.3'), ('mpich2', '%clang'), ('zmpi', '%gcc')])
def test_requirements_on_virtual_and_on_package(mpi_requirement, specific_requirement, concretize_scope, mock_packages):
    if False:
        print('Hello World!')
    conf_str = 'packages:\n  mpi:\n    require: "{0}"\n  {0}:\n    require: "{1}"\n'.format(mpi_requirement, specific_requirement)
    update_packages_config(conf_str)
    spec = Spec('callpath').concretized()
    assert 'mpi' in spec
    assert mpi_requirement in spec
    assert spec['mpi'].satisfies(specific_requirement)

def test_incompatible_virtual_requirements_raise(concretize_scope, mock_packages):
    if False:
        i = 10
        return i + 15
    conf_str = '    packages:\n      mpi:\n        require: "mpich"\n    '
    update_packages_config(conf_str)
    spec = Spec('callpath ^zmpi')
    with pytest.raises((UnsatisfiableSpecError, InternalConcretizerError)):
        spec.concretize()

def test_non_existing_variants_under_all(concretize_scope, mock_packages):
    if False:
        i = 10
        return i + 15
    conf_str = '    packages:\n      all:\n        require:\n        - any_of: ["~foo", "@:"]\n    '
    update_packages_config(conf_str)
    spec = Spec('callpath ^zmpi').concretized()
    assert '~foo' not in spec

@pytest.mark.parametrize('packages_yaml,spec_str,expected_satisfies', [('    packages:\n      all:\n        compiler: ["gcc", "clang"]\n\n      libelf:\n        require:\n        - one_of: ["%clang"]\n          when: "@0.8.13"\n', 'libelf', [('@0.8.13%clang', True), ('%gcc', False)]), ('    packages:\n      all:\n        compiler: ["gcc", "clang"]\n\n      libelf:\n        require:\n        - one_of: ["%clang"]\n          when: "@0.8.13"\n', 'libelf@0.8.12', [('%clang', False), ('%gcc', True)]), ('    packages:\n      all:\n        compiler: ["gcc", "clang"]\n\n      libelf:\n        require:\n        - spec: "%clang"\n          when: "@0.8.13"\n', 'libelf@0.8.12', [('%clang', False), ('%gcc', True)]), ('    packages:\n      all:\n        compiler: ["gcc", "clang"]\n\n      libelf:\n        require:\n        - spec: "@0.8.13"\n          when: "%clang"\n', 'libelf@0.8.13%gcc', [('%clang', False), ('%gcc', True), ('@0.8.13', True)])])
def test_conditional_requirements_from_packages_yaml(packages_yaml, spec_str, expected_satisfies, concretize_scope, mock_packages):
    if False:
        return 10
    'Test that conditional requirements are required when the condition is met,\n    and optional when the condition is not met.\n    '
    update_packages_config(packages_yaml)
    spec = Spec(spec_str).concretized()
    for (match_str, expected) in expected_satisfies:
        assert spec.satisfies(match_str) is expected

@pytest.mark.parametrize('packages_yaml,spec_str,expected_message', [('    packages:\n      mpileaks:\n        require:\n        - one_of: ["~debug"]\n          message: "debug is not allowed"\n', 'mpileaks+debug', 'debug is not allowed'), ('    packages:\n      libelf:\n        require:\n        - one_of: ["%clang"]\n          message: "can only be compiled with clang"\n', 'libelf%gcc', 'can only be compiled with clang'), ('        packages:\n          libelf:\n            require:\n            - one_of: ["%clang"]\n              when: platform=test\n              message: "can only be compiled with clang on the test platform"\n    ', 'libelf%gcc', 'can only be compiled with clang on '), ('            packages:\n              libelf:\n                require:\n                - spec: "%clang"\n                  when: platform=test\n                  message: "can only be compiled with clang on the test platform"\n        ', 'libelf%gcc', 'can only be compiled with clang on '), ('        packages:\n          libelf:\n            require:\n            - one_of: ["%clang", "%intel"]\n              when: platform=test\n              message: "can only be compiled with clang or intel on the test platform"\n    ', 'libelf%gcc', 'can only be compiled with clang or intel')])
def test_requirements_fail_with_custom_message(packages_yaml, spec_str, expected_message, concretize_scope, mock_packages):
    if False:
        while True:
            i = 10
    'Test that specs failing due to requirements not being satisfiable fail with a\n    custom error message.\n    '
    update_packages_config(packages_yaml)
    with pytest.raises(spack.error.SpackError, match=expected_message):
        Spec(spec_str).concretized()

def test_skip_requirement_when_default_requirement_condition_cannot_be_met(concretize_scope, mock_packages):
    if False:
        return 10
    "Tests that we can express a requirement condition under 'all' also in cases where\n    the corresponding condition spec mentions variants or versions that don't exist in the\n    package. For those packages the requirement rule is not emitted, since it can be\n    determined to be always false.\n    "
    packages_yaml = '\n        packages:\n          all:\n            require:\n            - one_of: ["%clang"]\n              when: "+shared"\n    '
    update_packages_config(packages_yaml)
    s = Spec('mpileaks').concretized()
    assert s.satisfies('%clang +shared')
    assert 'shared' not in s['callpath'].variants

def test_requires_directive(concretize_scope, mock_packages):
    if False:
        i = 10
        return i + 15
    compilers_yaml = pathlib.Path(concretize_scope) / 'compilers.yaml'
    compilers_yaml.write_text('\ncompilers::\n- compiler:\n    spec: gcc@12.0.0\n    paths:\n      cc: /usr/bin/clang-12\n      cxx: /usr/bin/clang++-12\n      f77: null\n      fc: null\n    operating_system: debian6\n    target: x86_64\n    modules: []\n')
    spack.config.CONFIG.clear_caches()
    s = Spec('requires_clang_or_gcc').concretized()
    assert s.satisfies('%gcc@12.0.0')
    with pytest.raises(spack.error.SpackError, match='can only be compiled with Clang'):
        Spec('requires_clang').concretized()