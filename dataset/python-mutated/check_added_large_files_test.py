from __future__ import annotations
import shutil
import pytest
from pre_commit_hooks.check_added_large_files import find_large_added_files
from pre_commit_hooks.check_added_large_files import main
from pre_commit_hooks.util import cmd_output
from testing.util import git_commit

def test_nothing_added(temp_git_dir):
    if False:
        for i in range(10):
            print('nop')
    with temp_git_dir.as_cwd():
        assert find_large_added_files(['f.py'], 0) == 0

def test_adding_something(temp_git_dir):
    if False:
        i = 10
        return i + 15
    with temp_git_dir.as_cwd():
        temp_git_dir.join('f.py').write("print('hello world')")
        cmd_output('git', 'add', 'f.py')
        assert find_large_added_files(['f.py'], 0) == 1

def test_add_something_giant(temp_git_dir):
    if False:
        while True:
            i = 10
    with temp_git_dir.as_cwd():
        temp_git_dir.join('f.py').write('a' * 10000)
        assert find_large_added_files(['f.py'], 0) == 0
        cmd_output('git', 'add', 'f.py')
        assert find_large_added_files(['f.py'], 0) == 1
        assert find_large_added_files(['f.py'], 9) == 1
        assert find_large_added_files(['f.py'], 10) == 0

def test_enforce_all(temp_git_dir):
    if False:
        i = 10
        return i + 15
    with temp_git_dir.as_cwd():
        temp_git_dir.join('f.py').write('a' * 10000)
        assert find_large_added_files(['f.py'], 0, enforce_all=True) == 1
        assert find_large_added_files(['f.py'], 0, enforce_all=False) == 0

def test_added_file_not_in_pre_commits_list(temp_git_dir):
    if False:
        while True:
            i = 10
    with temp_git_dir.as_cwd():
        temp_git_dir.join('f.py').write("print('hello world')")
        cmd_output('git', 'add', 'f.py')
        assert find_large_added_files(['g.py'], 0) == 0

def test_integration(temp_git_dir):
    if False:
        i = 10
        return i + 15
    with temp_git_dir.as_cwd():
        assert main(argv=[]) == 0
        temp_git_dir.join('f.py').write('a' * 10000)
        cmd_output('git', 'add', 'f.py')
        assert main(argv=['f.py']) == 0
        assert main(argv=['--maxkb', '9', 'f.py']) == 1

def has_gitlfs():
    if False:
        for i in range(10):
            print('nop')
    return shutil.which('git-lfs') is not None
xfailif_no_gitlfs = pytest.mark.xfail(not has_gitlfs(), reason='This test requires git-lfs')

@xfailif_no_gitlfs
def test_allows_gitlfs(temp_git_dir):
    if False:
        print('Hello World!')
    with temp_git_dir.as_cwd():
        cmd_output('git', 'lfs', 'install', '--local')
        temp_git_dir.join('f.py').write('a' * 10000)
        cmd_output('git', 'lfs', 'track', 'f.py')
        cmd_output('git', 'add', '--', '.')
        assert main(('--maxkb', '9', 'f.py')) == 0

@xfailif_no_gitlfs
def test_moves_with_gitlfs(temp_git_dir):
    if False:
        while True:
            i = 10
    with temp_git_dir.as_cwd():
        cmd_output('git', 'lfs', 'install', '--local')
        cmd_output('git', 'lfs', 'track', 'a.bin', 'b.bin')
        temp_git_dir.join('a.bin').write('a' * 10000)
        cmd_output('git', 'add', '--', '.')
        git_commit('-am', 'foo')
        cmd_output('git', 'mv', 'a.bin', 'b.bin')
        assert main(('--maxkb', '9', 'b.bin')) == 0

@xfailif_no_gitlfs
def test_enforce_allows_gitlfs(temp_git_dir):
    if False:
        i = 10
        return i + 15
    with temp_git_dir.as_cwd():
        cmd_output('git', 'lfs', 'install', '--local')
        temp_git_dir.join('f.py').write('a' * 10000)
        cmd_output('git', 'lfs', 'track', 'f.py')
        cmd_output('git', 'add', '--', '.')
        assert main(('--enforce-all', '--maxkb', '9', 'f.py')) == 0

@xfailif_no_gitlfs
def test_enforce_allows_gitlfs_after_commit(temp_git_dir):
    if False:
        return 10
    with temp_git_dir.as_cwd():
        cmd_output('git', 'lfs', 'install', '--local')
        temp_git_dir.join('f.py').write('a' * 10000)
        cmd_output('git', 'lfs', 'track', 'f.py')
        cmd_output('git', 'add', '--', '.')
        git_commit('-am', 'foo')
        assert main(('--enforce-all', '--maxkb', '9', 'f.py')) == 0