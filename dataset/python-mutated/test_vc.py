import os
import subprocess as sp
import textwrap
from pathlib import Path
from unittest.mock import Mock
import pytest
from xonsh.prompt import vc
VC_BRANCH = {'git': {'master', 'main'}, 'hg': {'default'}, 'fossil': {'trunk'}}
VC_INIT: dict[str, list[list[str]]] = {'git': [['init']], 'hg': [['init']], 'fossil': [['init', 'test.fossil'], ['open', 'test.fossil']]}

@pytest.fixture(params=VC_BRANCH.keys())
def repo(request, tmpdir_factory):
    if False:
        while True:
            i = 10
    'Return a dict with vc and a temporary dir\n    that is a repository for testing.\n    '
    vc = request.param
    temp_dir = Path(tmpdir_factory.mktemp('dir'))
    os.chdir(temp_dir)
    try:
        for init_command in VC_INIT[vc]:
            sp.call([vc] + init_command)
    except FileNotFoundError:
        pytest.skip(f'cannot find {vc} executable')
    if vc == 'git':
        _init_git_repository(temp_dir)
    return {'vc': vc, 'dir': temp_dir}

def _init_git_repository(temp_dir):
    if False:
        print('Hello World!')
    git_config = temp_dir / '.git/config'
    git_config.write_text(textwrap.dedent('        [user]\n        name = me\n        email = my@email.address\n        [init]\n        defaultBranch = main\n        '))
    Path('test-file').touch()
    sp.call(['git', 'add', 'test-file'])
    sp.call(['git', 'commit', '-m', 'test commit'])

@pytest.fixture
def set_xenv(xession, monkeypatch):
    if False:
        return 10

    def _wrapper(path):
        if False:
            i = 10
            return i + 15
        xession.env.update(dict(VC_BRANCH_TIMEOUT=2, PWD=path))
        return xession
    return _wrapper

def test_test_repo(repo):
    if False:
        print('Hello World!')
    if repo['vc'] == 'fossil':
        metadata_file_names = {'.fslckout', '_FOSSIL_'}
        existing_files = {file.name for file in repo['dir'].iterdir()}
        assert existing_files.intersection(metadata_file_names)
    else:
        test_vc_dir = repo['dir'] / '.{}'.format(repo['vc'])
        assert test_vc_dir.is_dir()
    if repo['vc'] == 'git':
        test_file = repo['dir'] / 'test-file'
        assert test_file.exists()

def test_no_repo(tmpdir, set_xenv):
    if False:
        return 10
    set_xenv(tmpdir)
    assert vc.get_hg_branch() is None
    assert vc.get_git_branch() is None

def test_vc_get_branch(repo, set_xenv):
    if False:
        for i in range(10):
            print('nop')
    set_xenv(repo['dir'])
    get_branch = 'get_{}_branch'.format(repo['vc'])
    branch = getattr(vc, get_branch)()
    assert branch in VC_BRANCH[repo['vc']]
    if repo['vc'] == 'git':
        git_config = repo['dir'] / '.git/config'
        with git_config.open('a') as f:
            f.write('\n[color]\nbranch = always\ninteractive = always\n[color "branch"]\ncurrent = yellow reverse\n')
        branch = getattr(vc, get_branch)()
        assert branch in VC_BRANCH[repo['vc']]
        assert not branch.startswith('\x1b[')

def test_current_branch_calls_locate_binary_for_empty_cmds_cache(xession, monkeypatch):
    if False:
        print('Hello World!')
    cache = xession.commands_cache
    monkeypatch.setattr(cache, 'is_empty', Mock(return_value=True))
    monkeypatch.setattr(cache, 'locate_binary', Mock(return_value=''))
    vc.current_branch()
    assert cache.locate_binary.called

def test_current_branch_does_not_call_locate_binary_for_non_empty_cmds_cache(xession, monkeypatch):
    if False:
        print('Hello World!')
    cache = xession.commands_cache
    monkeypatch.setattr(cache, 'is_empty', Mock(return_value=False))
    monkeypatch.setattr(cache, 'locate_binary', Mock(return_value=''))
    monkeypatch.setattr(cache, 'lazy_locate_binary', Mock(return_value=''))
    vc.current_branch()
    assert not cache.locate_binary.called

def test_dirty_working_directory(repo, set_xenv):
    if False:
        print('Hello World!')
    get_dwd = '{}_dirty_working_directory'.format(repo['vc'])
    set_xenv(repo['dir'])
    Path('second-test-file').touch()
    assert not getattr(vc, get_dwd)()
    sp.call([repo['vc'], 'add', 'second-test-file'])
    assert getattr(vc, get_dwd)()

@pytest.mark.parametrize('include_untracked', [True, False])
def test_git_dirty_working_directory_includes_untracked(xession, fake_process, include_untracked):
    if False:
        return 10
    xession.env['VC_GIT_INCLUDE_UNTRACKED'] = include_untracked
    if include_untracked:
        fake_process.register_subprocess(command='git status --porcelain --untracked-files=normal'.split(), stdout=b'?? untracked-test-file')
    else:
        fake_process.register_subprocess(command='git status --porcelain --untracked-files=no'.split(), stdout=b'')
    assert vc.git_dirty_working_directory() == include_untracked