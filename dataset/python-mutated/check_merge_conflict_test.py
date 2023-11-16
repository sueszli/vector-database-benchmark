from __future__ import annotations
import os
import shutil
import pytest
from pre_commit_hooks.check_merge_conflict import main
from pre_commit_hooks.util import cmd_output
from testing.util import get_resource_path
from testing.util import git_commit

@pytest.fixture
def f1_is_a_conflict_file(tmpdir):
    if False:
        return 10
    repo1 = tmpdir.join('repo1')
    repo1_f1 = repo1.join('f1')
    repo2 = tmpdir.join('repo2')
    repo2_f1 = repo2.join('f1')
    cmd_output('git', 'init', '--', str(repo1))
    with repo1.as_cwd():
        repo1_f1.ensure()
        cmd_output('git', 'add', '.')
        git_commit('-m', 'commit1')
    cmd_output('git', 'clone', str(repo1), str(repo2))
    with repo1.as_cwd():
        repo1_f1.write('parent\n')
        git_commit('-am', 'master commit2')
    with repo2.as_cwd():
        repo2_f1.write('child\n')
        git_commit('-am', 'clone commit2')
        cmd_output('git', 'pull', '--no-rebase', retcode=None)
        f1 = repo2_f1.read()
        assert f1.startswith('<<<<<<< HEAD\nchild\n=======\nparent\n>>>>>>>') or f1.startswith('<<<<<<< HEAD\nchild\n||||||| merged common ancestors\n=======\nparent\n>>>>>>>') or f1.startswith('<<<<<<< HEAD\nparent\n=======\nchild\n>>>>>>>')
        assert os.path.exists(os.path.join('.git', 'MERGE_MSG'))
        yield repo2

@pytest.fixture
def repository_pending_merge(tmpdir):
    if False:
        while True:
            i = 10
    repo1 = tmpdir.join('repo1')
    repo1_f1 = repo1.join('f1')
    repo2 = tmpdir.join('repo2')
    repo2_f1 = repo2.join('f1')
    repo2_f2 = repo2.join('f2')
    cmd_output('git', 'init', str(repo1))
    with repo1.as_cwd():
        repo1_f1.ensure()
        cmd_output('git', 'add', '.')
        git_commit('-m', 'commit1')
    cmd_output('git', 'clone', str(repo1), str(repo2))
    with repo1.as_cwd():
        repo1_f1.write('parent\n')
        git_commit('-am', 'master commit2')
    with repo2.as_cwd():
        repo2_f2.write('child\n')
        cmd_output('git', 'add', '.')
        git_commit('-m', 'clone commit2')
        cmd_output('git', 'pull', '--no-commit', '--no-rebase')
        assert repo2_f1.read() == 'parent\n'
        assert repo2_f2.read() == 'child\n'
        assert os.path.exists(os.path.join('.git', 'MERGE_HEAD'))
        yield repo2

@pytest.mark.usefixtures('f1_is_a_conflict_file')
def test_merge_conflicts_git(capsys):
    if False:
        print('Hello World!')
    assert main(['f1']) == 1
    (out, _) = capsys.readouterr()
    assert out == "f1:1: Merge conflict string '<<<<<<<' found\nf1:3: Merge conflict string '=======' found\nf1:5: Merge conflict string '>>>>>>>' found\n"

@pytest.mark.parametrize('contents', (b'<<<<<<< HEAD\n', b'=======\n', b'>>>>>>> master\n'))
def test_merge_conflicts_failing(contents, repository_pending_merge):
    if False:
        while True:
            i = 10
    repository_pending_merge.join('f2').write_binary(contents)
    assert main(['f2']) == 1

@pytest.mark.parametrize('contents', (b'# <<<<<<< HEAD\n', b'# =======\n', b'import mod', b''))
def test_merge_conflicts_ok(contents, f1_is_a_conflict_file):
    if False:
        i = 10
        return i + 15
    f1_is_a_conflict_file.join('f1').write_binary(contents)
    assert main(['f1']) == 0

@pytest.mark.usefixtures('f1_is_a_conflict_file')
def test_ignores_binary_files():
    if False:
        for i in range(10):
            print('nop')
    shutil.copy(get_resource_path('img1.jpg'), 'f1')
    assert main(['f1']) == 0

def test_does_not_care_when_not_in_a_merge(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('README.md')
    f.write_binary(b'problem\n=======\n')
    assert main([str(f.realpath())]) == 0

def test_care_when_assumed_merge(tmpdir):
    if False:
        return 10
    f = tmpdir.join('README.md')
    f.write_binary(b'problem\n=======\n')
    assert main([str(f.realpath()), '--assume-in-merge']) == 1

def test_worktree_merge_conflicts(f1_is_a_conflict_file, tmpdir, capsys):
    if False:
        i = 10
        return i + 15
    worktree = tmpdir.join('worktree')
    cmd_output('git', 'worktree', 'add', str(worktree))
    with worktree.as_cwd():
        cmd_output('git', 'pull', '--no-rebase', 'origin', 'master', retcode=None)
        msg = f1_is_a_conflict_file.join('.git/worktrees/worktree/MERGE_MSG')
        assert msg.exists()
        test_merge_conflicts_git(capsys)