import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import appdirs
import pytest
from ..conftest import TESTS_PATH
from ..semgrep_runner import SEMGREP_BASE_COMMAND
from .public_repos import REPOS
SENTINEL_VALUE = 87518275812375164
LANGUAGE_SENTINELS = {'python': {'filename': 'sentinel.py', 'file_contents': f'sentinel = {SENTINEL_VALUE}'}, 'go': {'filename': 'sentinel.go', 'file_contents': f'package Sentinel\nconst sentinel = {SENTINEL_VALUE}'}, 'javascript': {'filename': 'sentinel.js', 'file_contents': f'sentinel = {SENTINEL_VALUE}'}, 'ruby': {'filename': 'sentinel.rb', 'file_contents': f'sentinel = {SENTINEL_VALUE}'}}
SENTINEL_PATTERN = f'$SENTINEL = {SENTINEL_VALUE}'

@contextlib.contextmanager
def chdir(dirname=None):
    if False:
        for i in range(10):
            print('nop')
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)

def assert_sentinel_results(repo_path, sentinel_path, language):
    if False:
        i = 10
        return i + 15
    cmd = SEMGREP_BASE_COMMAND + ['--disable-version-check', '--pattern', SENTINEL_PATTERN, '--lang', language, '--json', repo_path, '--metrics=off', '--optimizations=none']
    print(f'semgrep command: {cmd}')
    semgrep_run = subprocess.run(cmd, capture_output=True, encoding='utf-8')
    assert semgrep_run.returncode == 0
    output = json.loads(semgrep_run.stdout)
    assert output['errors'] == []
    assert len(output['results']) == 1
    assert output['results'][0]['path'] == str(sentinel_path)
REPO_CACHE = Path(os.environ.get('QA_TESTS_CACHE_PATH', appdirs.user_cache_dir('semgrep-qa-tests')))

def clone_github_repo(repo_url: str, retries: int=3):
    if False:
        print('Hello World!')
    '\n    Internal fixture function. Do not use directly, use the `clone_github_repo` fixture.\n    Wraps `_github_repo` function with retries. If the `_github_repo` throws an exception,\n    it will delete `repo_destination` and retry up to `retries` times.\n    '
    repo_dir = '-'.join(repo_url.split('/')[-2:])
    repo_destination = REPO_CACHE / repo_dir
    try:
        return _github_repo(repo_url, repo_destination)
    except (GitError, subprocess.CalledProcessError) as ex:
        print(f'Failed to clone github repo for tests {ex}')
        if repo_destination.exists():
            shutil.rmtree(repo_destination)
        if retries == 0:
            raise
        else:
            return clone_github_repo(repo_url, retries - 1)

class GitError(Exception):
    pass

def _github_repo(repo_url: str, repo_destination: Path):
    if False:
        i = 10
        return i + 15
    '\n    Internal fixture function. Use the `clone_github_repo` fixture.\n    Clones the github repo at repo_url into `repo_destination` and checks out `sha`.\n\n    If `repo_destination` already exists, it will validate that the correct repo is present at that location.\n    '
    if not repo_destination.exists():
        subprocess.check_output(['git', 'clone', '--depth=1', repo_url, repo_destination])
    with chdir(repo_destination):
        subprocess.check_output(['git', 'clean', '-fd'])
        subprocess.check_output(['git', 'reset', '--hard'])
        assert subprocess.check_output(['git', 'status', '--porcelain']).strip() == b'', 'repo must be clean'
    return repo_destination

@pytest.mark.slow
@pytest.mark.parametrize('repo', [repo.as_param() for repo in REPOS])
def test_semgrep_on_repo(monkeypatch, tmp_path, repo):
    if False:
        i = 10
        return i + 15
    (tmp_path / 'rules').symlink_to(Path(TESTS_PATH / 'qa' / 'rules').resolve())
    monkeypatch.chdir(tmp_path)
    repo_path = clone_github_repo(repo_url=repo.url)
    repo_languages = LANGUAGE_SENTINELS if repo.languages is None else {language: sentinel_info for (language, sentinel_info) in LANGUAGE_SENTINELS.items() if language in repo.languages}
    for (language, sentinel_info) in repo_languages.items():
        sentinel_path = repo_path / sentinel_info['filename']
        with sentinel_path.open('w') as sentinel_file:
            sentinel_file.write(sentinel_info['file_contents'])
        assert_sentinel_results(repo_path, sentinel_path, language)
    cmd = SEMGREP_BASE_COMMAND + ['--disable-version-check', '--config=rules/regex-sentinel.yaml', '--strict', '--json', '--metrics=off', '--optimizations=none', repo_path]
    print(f'semgrep command: {cmd}')
    res = subprocess.run(cmd, encoding='utf-8', capture_output=True)
    print('--- semgrep error output ---')
    print(res.stderr)
    print('----------------------------')
    print('--- semgrep standard output ---')
    print(res.stdout)
    print('-------------------------------')
    assert res.returncode == 0
    output = json.loads(res.stdout)
    assert output['results']
    assert len(output['results']) == len(repo_languages)
    assert output['errors'] == []