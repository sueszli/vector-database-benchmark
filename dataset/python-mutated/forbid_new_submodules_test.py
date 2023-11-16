from __future__ import annotations
import os
import subprocess
from unittest import mock
import pytest
from pre_commit_hooks.forbid_new_submodules import main
from testing.util import git_commit

@pytest.fixture
def git_dir_with_git_dir(tmpdir):
    if False:
        print('Hello World!')
    with tmpdir.as_cwd():
        subprocess.check_call(('git', 'init', '.'))
        git_commit('--allow-empty', '-m', 'init')
        subprocess.check_call(('git', 'init', 'foo'))
        git_commit('--allow-empty', '-m', 'init', cwd=str(tmpdir.join('foo')))
        yield

@pytest.mark.parametrize('cmd', (('git', 'submodule', 'add', './foo'), ('git', 'add', 'foo')))
def test_main_new_submodule(git_dir_with_git_dir, capsys, cmd):
    if False:
        for i in range(10):
            print('nop')
    subprocess.check_call(cmd)
    assert main(('random_non-related_file',)) == 0
    assert main(('foo',)) == 1
    (out, _) = capsys.readouterr()
    assert out.startswith('foo: new submodule introduced\n')

def test_main_new_submodule_committed(git_dir_with_git_dir, capsys):
    if False:
        i = 10
        return i + 15
    rev_parse_cmd = ('git', 'rev-parse', 'HEAD')
    from_ref = subprocess.check_output(rev_parse_cmd).decode().strip()
    subprocess.check_call(('git', 'submodule', 'add', './foo'))
    git_commit('-m', 'new submodule')
    to_ref = subprocess.check_output(rev_parse_cmd).decode().strip()
    with mock.patch.dict(os.environ, {'PRE_COMMIT_FROM_REF': from_ref, 'PRE_COMMIT_TO_REF': to_ref}):
        assert main(('random_non-related_file',)) == 0
        assert main(('foo',)) == 1
    (out, _) = capsys.readouterr()
    assert out.startswith('foo: new submodule introduced\n')

def test_main_no_new_submodule(git_dir_with_git_dir):
    if False:
        while True:
            i = 10
    open('test.py', 'a+').close()
    subprocess.check_call(('git', 'add', 'test.py'))
    assert main(('test.py',)) == 0