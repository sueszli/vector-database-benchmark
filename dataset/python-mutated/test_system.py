from __future__ import annotations
import re
import shutil
import subprocess
import typing
import dulwich.repo
import pytest
from poetry.vcs.git.system import SystemGit
if typing.TYPE_CHECKING:
    from pathlib import Path
GIT_NOT_INSTALLLED = shutil.which('git') is None

def get_head_sha(cwd: Path) -> str:
    if False:
        return 10
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, text=True).strip()

class TempRepoFixture(typing.NamedTuple):
    path: Path
    repo: dulwich.repo.Repo
    init_commit: str
    head_commit: str

@pytest.fixture()
def temp_repo(tmp_path: Path) -> TempRepoFixture:
    if False:
        print('Hello World!')
    'Temporary repository with 2 commits'
    repo = dulwich.repo.Repo.init(str(tmp_path))
    (tmp_path / 'foo').write_text('foo')
    repo.stage(['foo'])
    init_commit = repo.do_commit(committer=b'User <user@example.com>', author=b'User <user@example.com>', message=b'init', no_verify=True)
    (tmp_path / 'foo').write_text('bar')
    repo.stage(['foo'])
    head_commit = repo.do_commit(committer=b'User <user@example.com>', author=b'User <user@example.com>', message=b'extra', no_verify=True)
    return TempRepoFixture(path=tmp_path, repo=repo, init_commit=init_commit.decode(), head_commit=head_commit.decode())

@pytest.mark.skipif(GIT_NOT_INSTALLLED, reason='These tests requires git cli')
class TestSystemGit:

    def test_clone_success(self, tmp_path: Path, temp_repo: TempRepoFixture) -> None:
        if False:
            while True:
                i = 10
        target_dir = tmp_path / 'test-repo'
        stdout = SystemGit.clone(temp_repo.path.as_uri(), target_dir)
        assert re.search("Cloning into '.+[\\\\/]test-repo'...", stdout)
        assert (target_dir / '.git').is_dir()

    def test_clone_invalid_parameter(self, tmp_path: Path) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(RuntimeError, match=re.escape('Invalid Git parameter: --upload-pack')):
            SystemGit.clone('--upload-pack=touch ./HELL', tmp_path)

    def test_checkout_1(self, temp_repo: TempRepoFixture) -> None:
        if False:
            for i in range(10):
                print('nop')
        SystemGit.checkout(temp_repo.init_commit[:12], temp_repo.path)
        assert get_head_sha(temp_repo.path) == temp_repo.init_commit

    def test_checkout_2(self, monkeypatch: pytest.MonkeyPatch, temp_repo: TempRepoFixture) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.chdir(temp_repo.path)
        SystemGit.checkout(temp_repo.init_commit[:12])
        assert get_head_sha(temp_repo.path) == temp_repo.init_commit