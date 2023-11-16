from pathlib import Path
from typing import Optional
import pytest
from tests.lib import PipTestEnvironment, _change_test_package_version, _create_test_package, pyversion
from tests.lib.git_submodule_helpers import _change_test_package_submodule, _create_test_package_with_submodule, _pull_in_submodule_changes_to_module
from tests.lib.local_repos import local_checkout

def _get_editable_repo_dir(script: PipTestEnvironment, package_name: str) -> Path:
    if False:
        return 10
    '\n    Return the repository directory for an editable install.\n    '
    return script.venv_path / 'src' / package_name

def _get_editable_branch(script: PipTestEnvironment, package_name: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Return the current branch of an editable install.\n    '
    repo_dir = _get_editable_repo_dir(script, package_name)
    result = script.run('git', 'rev-parse', '--abbrev-ref', 'HEAD', cwd=repo_dir)
    return result.stdout.strip()

def _get_branch_remote(script: PipTestEnvironment, package_name: str, branch: str) -> str:
    if False:
        print('Hello World!')
    ' '
    repo_dir = _get_editable_repo_dir(script, package_name)
    result = script.run('git', 'config', f'branch.{branch}.remote', cwd=repo_dir)
    return result.stdout.strip()

def _github_checkout(url_path: str, tmpdir: Path, rev: Optional[str]=None, egg: Optional[str]=None, scheme: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    '\n    Call local_checkout() with a GitHub URL, and return the resulting URL.\n\n    Args:\n      url_path: the string used to create the package URL by filling in the\n        format string "git+{scheme}://github.com/{url_path}".\n      temp_dir: the pytest tmpdir value.\n      egg: an optional project name to append to the URL as the egg fragment,\n        prior to returning.\n      scheme: the scheme without the "git+" prefix. Defaults to "https".\n    '
    if scheme is None:
        scheme = 'https'
    url = f'git+{scheme}://github.com/{url_path}'
    local_url = local_checkout(url, tmpdir)
    if rev is not None:
        local_url += f'@{rev}'
    if egg is not None:
        local_url += f'#egg={egg}'
    return local_url

def _make_version_pkg_url(path: Path, rev: Optional[str]=None, name: str='version_pkg') -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a "git+file://" URL to the version_pkg test package.\n\n    Args:\n      path: a pathlib.Path object pointing to a Git repository\n        containing the version_pkg package.\n      rev: an optional revision to install like a branch name, tag, or SHA.\n    '
    file_url = path.as_uri()
    url_rev = '' if rev is None else f'@{rev}'
    url = f'git+{file_url}{url_rev}#egg={name}'
    return url

def _install_version_pkg_only(script: PipTestEnvironment, path: Path, rev: Optional[str]=None, allow_stderr_warning: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Install the version_pkg package in editable mode (without returning\n    the version).\n\n    Args:\n      path: a pathlib.Path object pointing to a Git repository\n        containing the package.\n      rev: an optional revision to install like a branch name or tag.\n    '
    version_pkg_url = _make_version_pkg_url(path, rev=rev)
    script.pip('install', '-e', version_pkg_url, allow_stderr_warning=allow_stderr_warning)

def _install_version_pkg(script: PipTestEnvironment, path: Path, rev: Optional[str]=None, allow_stderr_warning: bool=False) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Install the version_pkg package in editable mode, and return the version\n    installed.\n\n    Args:\n      path: a pathlib.Path object pointing to a Git repository\n        containing the package.\n      rev: an optional revision to install like a branch name or tag.\n    '
    _install_version_pkg_only(script, path, rev=rev, allow_stderr_warning=allow_stderr_warning)
    result = script.run('version_pkg')
    version = result.stdout.strip()
    return version

def test_git_install_again_after_changes(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    Test installing a repository a second time without specifying a revision,\n    and after updates to the remote repository.\n\n    This test also checks that no warning message like the following gets\n    logged on the update: "Did not find branch or tag ..., assuming ref or\n    revision."\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    version = _install_version_pkg(script, version_pkg_path)
    assert version == '0.1'
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path)
    assert version == 'some different version'

def test_git_install_branch_again_after_branch_changes(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Test installing a branch again after the branch is updated in the remote\n    repository.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    version = _install_version_pkg(script, version_pkg_path, rev='master')
    assert version == '0.1'
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path, rev='master')
    assert version == 'some different version'

@pytest.mark.network
def test_install_editable_from_git_with_https(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test cloning from Git with https.\n    '
    url_path = 'pypa/pip-test-package.git'
    local_url = _github_checkout(url_path, tmpdir, egg='pip-test-package')
    result = script.pip('install', '-e', local_url)
    result.assert_installed('pip-test-package', with_files=['.git'])

@pytest.mark.network
def test_install_noneditable_git(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    Test installing from a non-editable git URL with a given tag.\n    '
    result = script.pip('install', 'git+https://github.com/pypa/pip-test-package.git@0.1.1#egg=pip-test-package')
    dist_info_folder = script.site_packages / 'pip_test_package-0.1.1.dist-info'
    result.assert_installed('piptestpackage', without_egg_link=True, editable=False)
    result.did_create(dist_info_folder)

def test_git_with_sha1_revisions(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    Git backend should be able to install from SHA1 revisions\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    _change_test_package_version(script, version_pkg_path)
    sha1 = script.run('git', 'rev-parse', 'HEAD~1', cwd=version_pkg_path).stdout.strip()
    version = _install_version_pkg(script, version_pkg_path, rev=sha1)
    assert '0.1' == version

def test_git_with_short_sha1_revisions(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Git backend should be able to install from SHA1 revisions\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    _change_test_package_version(script, version_pkg_path)
    sha1 = script.run('git', 'rev-parse', 'HEAD~1', cwd=version_pkg_path).stdout.strip()[:7]
    version = _install_version_pkg(script, version_pkg_path, rev=sha1, allow_stderr_warning=True)
    assert '0.1' == version

def test_git_with_branch_name_as_revision(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Git backend should be able to install from branch names\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    branch = 'test_branch'
    script.run('git', 'checkout', '-b', branch, cwd=version_pkg_path)
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path, rev=branch)
    assert 'some different version' == version

def test_git_with_tag_name_as_revision(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    Git backend should be able to install from tag names\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    script.run('git', 'tag', 'test_tag', cwd=version_pkg_path)
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path, rev='test_tag')
    assert '0.1' == version

def _add_ref(script: PipTestEnvironment, path: Path, ref: str) -> None:
    if False:
        return 10
    '\n    Add a new ref to a repository at the given path.\n    '
    script.run('git', 'update-ref', ref, 'HEAD', cwd=path)

def test_git_install_ref(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    The Git backend should be able to install a ref with the first install.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    _add_ref(script, version_pkg_path, 'refs/foo/bar')
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path, rev='refs/foo/bar', allow_stderr_warning=True)
    assert '0.1' == version

def test_git_install_then_install_ref(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    The Git backend should be able to install a ref after a package has\n    already been installed.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    _add_ref(script, version_pkg_path, 'refs/foo/bar')
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path)
    assert 'some different version' == version
    version = _install_version_pkg(script, version_pkg_path, rev='refs/foo/bar', allow_stderr_warning=True)
    assert '0.1' == version

@pytest.mark.network
@pytest.mark.parametrize('rev, expected_sha', [('', '5547fa909e83df8bd743d3978d6667497983a4b7'), ('@0.1.1', '7d654e66c8fa7149c165ddeffa5b56bc06619458'), ('@65cf0a5bdd906ecf48a0ac241c17d656d2071d56', '65cf0a5bdd906ecf48a0ac241c17d656d2071d56')])
def test_install_git_logs_commit_sha(script: PipTestEnvironment, rev: str, expected_sha: str, tmpdir: Path) -> None:
    if False:
        print('Hello World!')
    '\n    Test installing from a git repository logs a commit SHA.\n    '
    url_path = 'pypa/pip-test-package.git'
    base_local_url = _github_checkout(url_path, tmpdir)
    local_url = f'{base_local_url}{rev}#egg=pip-test-package'
    result = script.pip('install', local_url)
    assert f'Resolved {base_local_url[4:]} to commit {expected_sha}' in result.stdout

@pytest.mark.network
def test_git_with_tag_name_and_update(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        return 10
    '\n    Test cloning a git repository and updating to a different version.\n    '
    url_path = 'pypa/pip-test-package.git'
    base_local_url = _github_checkout(url_path, tmpdir)
    local_url = f'{base_local_url}#egg=pip-test-package'
    result = script.pip('install', '-e', local_url)
    result.assert_installed('pip-test-package', with_files=['.git'])
    new_local_url = f'{base_local_url}@0.1.2#egg=pip-test-package'
    result = script.pip('install', '--global-option=--version', '-e', new_local_url, allow_stderr_warning=True)
    assert '0.1.2' in result.stdout

@pytest.mark.network
def test_git_branch_should_not_be_changed(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        print('Hello World!')
    '\n    Editable installations should not change branch\n    related to issue #32 and #161\n    '
    url_path = 'pypa/pip-test-package.git'
    local_url = _github_checkout(url_path, tmpdir, egg='pip-test-package')
    script.pip('install', '-e', local_url)
    branch = _get_editable_branch(script, 'pip-test-package')
    assert 'master' == branch

@pytest.mark.network
def test_git_with_non_editable_unpacking(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        return 10
    '\n    Test cloning a git repository from a non-editable URL with a given tag.\n    '
    url_path = 'pypa/pip-test-package.git'
    local_url = _github_checkout(url_path, tmpdir, rev='0.1.2', egg='pip-test-package')
    result = script.pip('install', '--global-option=--quiet', local_url, allow_stderr_warning=True)
    assert '0.1.2' in result.stdout

@pytest.mark.network
def test_git_with_editable_where_egg_contains_dev_string(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        print('Hello World!')
    '\n    Test cloning a git repository from an editable url which contains "dev"\n    string\n    '
    url_path = 'dcramer/django-devserver.git'
    local_url = _github_checkout(url_path, tmpdir, egg='django-devserver', scheme='https')
    result = script.pip('install', '-e', local_url)
    result.assert_installed('django-devserver', with_files=['.git'])

@pytest.mark.network
def test_git_with_non_editable_where_egg_contains_dev_string(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test cloning a git repository from a non-editable url which contains "dev"\n    string\n    '
    url_path = 'dcramer/django-devserver.git'
    local_url = _github_checkout(url_path, tmpdir, egg='django-devserver', scheme='https')
    result = script.pip('install', local_url)
    devserver_folder = script.site_packages / 'devserver'
    result.did_create(devserver_folder)

def test_git_with_ambiguous_revs(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Test git with two "names" (tag/branch) pointing to the same commit\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    version_pkg_url = _make_version_pkg_url(version_pkg_path, rev='0.1')
    script.run('git', 'tag', '0.1', cwd=version_pkg_path)
    result = script.pip('install', '-e', version_pkg_url)
    assert 'Could not find a tag or branch' not in result.stdout
    result.assert_installed('version-pkg', with_files=['.git'])

def test_editable__no_revision(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a basic install in editable mode specifying no revision.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    _install_version_pkg_only(script, version_pkg_path)
    branch = _get_editable_branch(script, 'version-pkg')
    assert branch == 'master'
    remote = _get_branch_remote(script, 'version-pkg', 'master')
    assert remote == 'origin'

def test_editable__branch_with_sha_same_as_default(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Test installing in editable mode a branch whose sha matches the sha\n    of the default branch, but is different from the default branch.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    script.run('git', 'branch', 'develop', cwd=version_pkg_path)
    _install_version_pkg_only(script, version_pkg_path, rev='develop')
    branch = _get_editable_branch(script, 'version-pkg')
    assert branch == 'develop'
    remote = _get_branch_remote(script, 'version-pkg', 'develop')
    assert remote == 'origin'

def test_editable__branch_with_sha_different_from_default(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    '\n    Test installing in editable mode a branch whose sha is different from\n    the sha of the default branch.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    script.run('git', 'branch', 'develop', cwd=version_pkg_path)
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path, rev='develop')
    assert version == '0.1'
    branch = _get_editable_branch(script, 'version-pkg')
    assert branch == 'develop'
    remote = _get_branch_remote(script, 'version-pkg', 'develop')
    assert remote == 'origin'

def test_editable__non_master_default_branch(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the branch you get after an editable install from a remote repo\n    with a non-master default branch.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    script.run('git', 'checkout', '-b', 'release', cwd=version_pkg_path)
    _install_version_pkg_only(script, version_pkg_path)
    branch = _get_editable_branch(script, 'version-pkg')
    assert branch == 'release'

def test_reinstalling_works_with_editable_non_master_branch(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Reinstalling an editable installation should not assume that the "master"\n    branch exists. See https://github.com/pypa/pip/issues/4448.\n    '
    version_pkg_path = _create_test_package(script.scratch_path)
    script.run('git', 'branch', '-m', 'foobar', cwd=version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path)
    assert '0.1' == version
    _change_test_package_version(script, version_pkg_path)
    version = _install_version_pkg(script, version_pkg_path)
    assert 'some different version' == version

@pytest.mark.skipif("sys.platform == 'win32'")
@pytest.mark.xfail(condition=True, reason='Git submodule against file: is not working; waiting for a good solution', run=True)
def test_check_submodule_addition(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Submodules are pulled in on install and updated on upgrade.\n    '
    (module_path, submodule_path) = _create_test_package_with_submodule(script, rel_path='testpkg/static')
    install_result = script.pip('install', '-e', f'git+{module_path.as_uri()}#egg=version_pkg')
    install_result.did_create(script.venv / 'src/version-pkg/testpkg/static/testfile')
    _change_test_package_submodule(script, submodule_path)
    _pull_in_submodule_changes_to_module(script, module_path, rel_path='testpkg/static')
    update_result = script.pip('install', '-e', f'git+{module_path.as_uri()}#egg=version_pkg', '--upgrade')
    update_result.did_create(script.venv / 'src/version-pkg/testpkg/static/testfile2')

def test_install_git_branch_not_cached(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Installing git urls with a branch revision does not cause wheel caching.\n    '
    PKG = 'gitbranchnotcached'
    repo_dir = _create_test_package(script.scratch_path, name=PKG)
    url = _make_version_pkg_url(repo_dir, rev='master', name=PKG)
    result = script.pip('install', url, '--only-binary=:all:')
    assert f'Successfully built {PKG}' in result.stdout, result.stdout
    script.pip('uninstall', '-y', PKG)
    result = script.pip('install', url)
    assert f'Successfully built {PKG}' in result.stdout, result.stdout

def test_install_git_sha_cached(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    '\n    Installing git urls with a sha revision does cause wheel caching.\n    '
    PKG = 'gitshacached'
    repo_dir = _create_test_package(script.scratch_path, name=PKG)
    commit = script.run('git', 'rev-parse', 'HEAD', cwd=repo_dir).stdout.strip()
    url = _make_version_pkg_url(repo_dir, rev=commit, name=PKG)
    result = script.pip('install', url)
    assert f'Successfully built {PKG}' in result.stdout, result.stdout
    script.pip('uninstall', '-y', PKG)
    result = script.pip('install', url)
    assert f'Successfully built {PKG}' not in result.stdout, result.stdout