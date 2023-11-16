import copy
import os
import shutil
import pytest
from llnl.util.filesystem import mkdirp, touch, working_dir
import spack.config
import spack.repo
from spack.fetch_strategy import GitFetchStrategy
from spack.spec import Spec
from spack.stage import Stage
from spack.version import Version
_mock_transport_error = 'Mock HTTP transport error'

@pytest.fixture(params=[None, '1.8.5.2', '1.8.5.1', '1.7.10', '1.7.1', '1.7.0'])
def git_version(git, request, monkeypatch):
    if False:
        while True:
            i = 10
    'Tests GitFetchStrategy behavior for different git versions.\n\n    GitFetchStrategy tries to optimize using features of newer git\n    versions, but needs to work with older git versions.  To ensure code\n    paths for old versions still work, we fake it out here and make it\n    use the backward-compatibility code paths with newer git versions.\n    '
    real_git_version = spack.fetch_strategy.GitFetchStrategy.version_from_git(git)
    if request.param is None:
        yield real_git_version
    else:
        test_git_version = Version(request.param)
        if test_git_version > real_git_version:
            pytest.skip("Can't test clone logic for newer version of git.")
        monkeypatch.setattr(GitFetchStrategy, 'git_version', test_git_version)
        yield test_git_version

@pytest.fixture
def mock_bad_git(monkeypatch):
    if False:
        while True:
            i = 10
    '\n    Test GitFetchStrategy behavior with a bad git command for git >= 1.7.1\n    to trigger a SpackError.\n    '

    def bad_git(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Raise a SpackError with the transport message.'
        raise spack.error.SpackError(_mock_transport_error)
    monkeypatch.setattr(GitFetchStrategy, 'git', bad_git)
    monkeypatch.setattr(GitFetchStrategy, 'git_version', Version('1.7.1'))
    yield

def test_bad_git(tmpdir, mock_bad_git):
    if False:
        i = 10
        return i + 15
    'Trigger a SpackError when attempt a fetch with a bad git.'
    testpath = str(tmpdir)
    with pytest.raises(spack.error.SpackError):
        fetcher = GitFetchStrategy(git='file:///not-a-real-git-repo')
        with Stage(fetcher, path=testpath):
            fetcher.fetch()

@pytest.mark.parametrize('type_of_test', ['default', 'branch', 'tag', 'commit'])
@pytest.mark.parametrize('secure', [True, False])
def test_fetch(git, type_of_test, secure, mock_git_repository, default_mock_concretization, mutable_mock_repo, git_version, monkeypatch):
    if False:
        print('Hello World!')
    "Tries to:\n\n    1. Fetch the repo using a fetch strategy constructed with\n       supplied args (they depend on type_of_test).\n    2. Check if the test_file is in the checked out repository.\n    3. Assert that the repository is at the revision supplied.\n    4. Add and remove some files, then reset the repo, and\n       ensure it's all there again.\n    "
    t = mock_git_repository.checks[type_of_test]
    h = mock_git_repository.hash
    pkg_class = spack.repo.PATH.get_pkg_class('git-test')
    monkeypatch.delattr(pkg_class, 'git')
    s = default_mock_concretization('git-test')
    monkeypatch.setitem(s.package.versions, Version('git'), t.args)
    with s.package.stage:
        with spack.config.override('config:verify_ssl', secure):
            s.package.do_stage()
        with working_dir(s.package.stage.source_path):
            assert h('HEAD') == h(t.revision)
            file_path = os.path.join(s.package.stage.source_path, t.file)
            assert os.path.isdir(s.package.stage.source_path)
            assert os.path.isfile(file_path)
            os.unlink(file_path)
            assert not os.path.isfile(file_path)
            untracked_file = 'foobarbaz'
            touch(untracked_file)
            assert os.path.isfile(untracked_file)
            s.package.do_restage()
            assert not os.path.isfile(untracked_file)
            assert os.path.isdir(s.package.stage.source_path)
            assert os.path.isfile(file_path)
            assert h('HEAD') == h(t.revision)

@pytest.mark.disable_clean_stage_check
def test_fetch_pkg_attr_submodule_init(mock_git_repository, default_mock_concretization, mutable_mock_repo, monkeypatch, mock_stage):
    if False:
        print('Hello World!')
    "In this case the version() args do not contain a 'git' URL, so\n    the fetcher must be assembled using the Package-level 'git' attribute.\n    This test ensures that the submodules are properly initialized and the\n    expected branch file is present.\n    "
    t = mock_git_repository.checks['default-no-per-version-git']
    pkg_class = spack.repo.PATH.get_pkg_class('git-test')
    monkeypatch.setattr(pkg_class, 'git', mock_git_repository.url)
    s = default_mock_concretization('git-test')
    monkeypatch.setitem(s.package.versions, Version('git'), t.args)
    s.package.do_stage()
    collected_fnames = set()
    for (root, dirs, files) in os.walk(s.package.stage.source_path):
        collected_fnames.update(files)
    assert {'r0_file_0', 'r0_file_1', t.file} < collected_fnames

@pytest.mark.skipif(str(spack.platforms.host()) == 'windows', reason='Git fails to clone because the src/dst paths are too long: the name of the staging directory for ad-hoc Git commit versions is longer than other staged sources')
@pytest.mark.disable_clean_stage_check
def test_adhoc_version_submodules(mock_git_repository, config, mutable_mock_repo, monkeypatch, mock_stage):
    if False:
        while True:
            i = 10
    t = mock_git_repository.checks['tag']
    pkg_class = spack.repo.PATH.get_pkg_class('git-test')
    monkeypatch.setitem(pkg_class.versions, Version('git'), t.args)
    monkeypatch.setattr(pkg_class, 'git', 'file://%s' % mock_git_repository.path, raising=False)
    spec = Spec('git-test@{0}'.format(mock_git_repository.unversioned_commit))
    spec.concretize()
    spec.package.do_stage()
    collected_fnames = set()
    for (root, dirs, files) in os.walk(spec.package.stage.source_path):
        collected_fnames.update(files)
    assert set(['r0_file_0', 'r0_file_1']) < collected_fnames

@pytest.mark.parametrize('type_of_test', ['branch', 'commit'])
def test_debug_fetch(mock_packages, type_of_test, mock_git_repository, default_mock_concretization, monkeypatch):
    if False:
        print('Hello World!')
    'Fetch the repo with debug enabled.'
    t = mock_git_repository.checks[type_of_test]
    s = default_mock_concretization('git-test')
    monkeypatch.setitem(s.package.versions, Version('git'), t.args)
    with s.package.stage:
        with spack.config.override('config:debug', True):
            s.package.do_fetch()
            assert os.path.isdir(s.package.stage.source_path)

def test_git_extra_fetch(git, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    "Ensure a fetch after 'expanding' is effectively a no-op."
    testpath = str(tmpdir)
    fetcher = GitFetchStrategy(git='file:///not-a-real-git-repo')
    with Stage(fetcher, path=testpath) as stage:
        mkdirp(stage.source_path)
        fetcher.fetch()
        shutil.rmtree(stage.source_path)

def test_needs_stage(git):
    if False:
        for i in range(10):
            print('nop')
    'Trigger a NoStageError when attempt a fetch without a stage.'
    with pytest.raises(spack.fetch_strategy.NoStageError, match='set_stage.*before calling fetch'):
        fetcher = GitFetchStrategy(git='file:///not-a-real-git-repo')
        fetcher.fetch()

@pytest.mark.parametrize('get_full_repo', [True, False])
def test_get_full_repo(get_full_repo, git_version, mock_git_repository, default_mock_concretization, mutable_mock_repo, monkeypatch):
    if False:
        print('Hello World!')
    'Ensure that we can clone a full repository.'
    if git_version < Version('1.7.1'):
        pytest.skip('Not testing get_full_repo for older git {0}'.format(git_version))
    secure = True
    type_of_test = 'tag-branch'
    t = mock_git_repository.checks[type_of_test]
    s = default_mock_concretization('git-test')
    args = copy.copy(t.args)
    args['get_full_repo'] = get_full_repo
    monkeypatch.setitem(s.package.versions, Version('git'), args)
    with s.package.stage:
        with spack.config.override('config:verify_ssl', secure):
            s.package.do_stage()
            with working_dir(s.package.stage.source_path):
                branches = mock_git_repository.git_exe('branch', '-a', output=str).splitlines()
                nbranches = len(branches)
                commits = mock_git_repository.git_exe('log', '--graph', '--pretty=format:%h -%d %s (%ci) <%an>', '--abbrev-commit', output=str).splitlines()
                ncommits = len(commits)
        if get_full_repo:
            assert nbranches >= 5
            assert ncommits == 2
        else:
            assert nbranches == 2
            assert ncommits == 1

@pytest.mark.disable_clean_stage_check
@pytest.mark.parametrize('submodules', [True, False])
def test_gitsubmodule(submodules, mock_git_repository, default_mock_concretization, mutable_mock_repo, monkeypatch):
    if False:
        while True:
            i = 10
    '\n    Test GitFetchStrategy behavior with submodules. This package\n    has a `submodules` property which is always True: when a specific\n    version also indicates to include submodules, this should not\n    interfere; if the specific version explicitly requests that\n    submodules *not* be initialized, this should override the\n    Package-level request.\n    '
    type_of_test = 'tag-branch'
    t = mock_git_repository.checks[type_of_test]
    s = default_mock_concretization('git-test')
    args = copy.copy(t.args)
    args['submodules'] = submodules
    monkeypatch.setitem(s.package.versions, Version('git'), args)
    s.package.do_stage()
    with working_dir(s.package.stage.source_path):
        for submodule_count in range(2):
            file_path = os.path.join(s.package.stage.source_path, 'third_party/submodule{0}/r0_file_{0}'.format(submodule_count))
            if submodules:
                assert os.path.isfile(file_path)
            else:
                assert not os.path.isfile(file_path)

@pytest.mark.disable_clean_stage_check
def test_gitsubmodules_callable(mock_git_repository, default_mock_concretization, mutable_mock_repo, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test GitFetchStrategy behavior with submodules selected after concretization\n    '

    def submodules_callback(package):
        if False:
            i = 10
            return i + 15
        name = 'third_party/submodule0'
        return [name]
    type_of_test = 'tag-branch'
    t = mock_git_repository.checks[type_of_test]
    s = default_mock_concretization('git-test')
    args = copy.copy(t.args)
    args['submodules'] = submodules_callback
    monkeypatch.setitem(s.package.versions, Version('git'), args)
    s.package.do_stage()
    with working_dir(s.package.stage.source_path):
        file_path = os.path.join(s.package.stage.source_path, 'third_party/submodule0/r0_file_0')
        assert os.path.isfile(file_path)
        file_path = os.path.join(s.package.stage.source_path, 'third_party/submodule1/r0_file_1')
        assert not os.path.isfile(file_path)

@pytest.mark.disable_clean_stage_check
def test_gitsubmodules_delete(mock_git_repository, default_mock_concretization, mutable_mock_repo, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test GitFetchStrategy behavior with submodules_delete\n    '
    type_of_test = 'tag-branch'
    t = mock_git_repository.checks[type_of_test]
    s = default_mock_concretization('git-test')
    args = copy.copy(t.args)
    args['submodules'] = True
    args['submodules_delete'] = ['third_party/submodule0', 'third_party/submodule1']
    monkeypatch.setitem(s.package.versions, Version('git'), args)
    s.package.do_stage()
    with working_dir(s.package.stage.source_path):
        file_path = os.path.join(s.package.stage.source_path, 'third_party/submodule0')
        assert not os.path.isdir(file_path)
        file_path = os.path.join(s.package.stage.source_path, 'third_party/submodule1')
        assert not os.path.isdir(file_path)