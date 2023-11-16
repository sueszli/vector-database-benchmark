"""
Tests for vcs.py
"""
import os
import os.path as osp
import sys
from spyder.utils import programs
import pytest
from spyder.config.base import running_in_ci
from spyder.utils.vcs import ActionToolNotFound, get_git_refs, get_git_remotes, get_git_revision, get_vcs_root, remote_to_url, run_vcs_tool
HERE = os.path.abspath(os.path.dirname(__file__))
skipnogit = pytest.mark.skipif(not get_vcs_root(HERE), reason='Not running from a git repo')

@skipnogit
@pytest.mark.skipif(running_in_ci(), reason='Not to be run outside of CIs')
def test_vcs_tool():
    if False:
        return 10
    if not os.name == 'nt':
        with pytest.raises(ActionToolNotFound):
            run_vcs_tool(osp.dirname(__file__), 'browse')
    else:
        assert run_vcs_tool(osp.dirname(__file__), 'browse')
        assert run_vcs_tool(osp.dirname(__file__), 'commit')

@skipnogit
def test_vcs_root(tmpdir):
    if False:
        i = 10
        return i + 15
    directory = tmpdir.mkdir('foo')
    assert get_vcs_root(str(directory)) == None
    assert get_vcs_root(osp.dirname(__file__)) != None

@skipnogit
def test_git_revision():
    if False:
        i = 10
        return i + 15
    root = get_vcs_root(osp.dirname(__file__))
    assert get_git_revision(osp.dirname(__file__)) == (None, None)
    assert all([isinstance(x, str) for x in get_git_revision(root)])

def test_no_git(monkeypatch):
    if False:
        return 10

    def mockreturn(program_name):
        if False:
            for i in range(10):
                print('nop')
        return None
    monkeypatch.setattr(programs, 'find_program', mockreturn)
    (branch_tags, branch, files_modified) = get_git_refs(__file__)
    assert len(branch_tags) == 0
    assert branch == ''
    assert len(files_modified) == 0

@skipnogit
def test_get_git_refs():
    if False:
        print('Hello World!')
    (branch_tags, branch, files_modified) = get_git_refs(__file__)
    assert bool(branch)
    assert len(files_modified) >= 0
    assert any(['master' in b or '4.x' in b for b in branch_tags])

@skipnogit
def test_get_git_remotes():
    if False:
        return 10
    remotes = get_git_remotes(HERE)
    assert 'origin' in remotes

@pytest.mark.parametrize('input_text, expected_output', [('https://github.com/neophnx/spyder.git', 'https://github.com/neophnx/spyder'), ('http://github.com/neophnx/spyder.git', 'http://github.com/neophnx/spyder'), ('git@github.com:goanpeca/spyder.git', 'https://github.com/goanpeca/spyder')])
def test_remote_to_url(input_text, expected_output):
    if False:
        print('Hello World!')
    output = remote_to_url(input_text)
    assert expected_output == output
if __name__ == '__main__':
    pytest.main()