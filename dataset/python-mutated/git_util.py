import os
import re
from typing import Any, Optional, Tuple
from streamlit import util
GITHUB_HTTP_URL = '^https://(www\\.)?github.com/(.+)/(.+)(?:.git)?$'
GITHUB_SSH_URL = '^git@github.com:(.+)/(.+)(?:.git)?$'
MIN_GIT_VERSION = (2, 7, 0)

class GitRepo:

    def __init__(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.git_version: Optional[Tuple[int, ...]] = None
        try:
            import git
            git_package: Any = git
            self.repo = git_package.Repo(path, search_parent_directories=True)
            self.git_version = self.repo.git.version_info
            if self.git_version >= MIN_GIT_VERSION:
                git_root = self.repo.git.rev_parse('--show-toplevel')
                self.module = os.path.relpath(path, git_root)
        except Exception:
            self.repo = None

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return util.repr_(self)

    def is_valid(self) -> bool:
        if False:
            i = 10
            return i + 15
        "True if there's a git repo here, and git.version >= MIN_GIT_VERSION."
        return self.repo is not None and self.git_version is not None and (self.git_version >= MIN_GIT_VERSION)

    @property
    def tracking_branch(self):
        if False:
            while True:
                i = 10
        if not self.is_valid():
            return None
        if self.is_head_detached:
            return None
        return self.repo.active_branch.tracking_branch()

    @property
    def untracked_files(self):
        if False:
            while True:
                i = 10
        if not self.is_valid():
            return None
        return self.repo.untracked_files

    @property
    def is_head_detached(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_valid():
            return False
        return self.repo.head.is_detached

    @property
    def uncommitted_files(self):
        if False:
            while True:
                i = 10
        if not self.is_valid():
            return None
        return [item.a_path for item in self.repo.index.diff(None)]

    @property
    def ahead_commits(self):
        if False:
            i = 10
            return i + 15
        if not self.is_valid():
            return None
        try:
            (remote, branch_name) = self.get_tracking_branch_remote()
            remote_branch = '/'.join([remote.name, branch_name])
            return list(self.repo.iter_commits(f'{remote_branch}..{branch_name}'))
        except Exception:
            return list()

    def get_tracking_branch_remote(self):
        if False:
            return 10
        if not self.is_valid():
            return None
        tracking_branch = self.tracking_branch
        if tracking_branch is None:
            return None
        (remote_name, *branch) = tracking_branch.name.split('/')
        branch_name = '/'.join(branch)
        return (self.repo.remote(remote_name), branch_name)

    def is_github_repo(self):
        if False:
            i = 10
            return i + 15
        if not self.is_valid():
            return False
        remote_info = self.get_tracking_branch_remote()
        if remote_info is None:
            return False
        (remote, _branch) = remote_info
        for url in remote.urls:
            if re.match(GITHUB_HTTP_URL, url) is not None or re.match(GITHUB_SSH_URL, url) is not None:
                return True
        return False

    def get_repo_info(self):
        if False:
            i = 10
            return i + 15
        if not self.is_valid():
            return None
        remote_info = self.get_tracking_branch_remote()
        if remote_info is None:
            return None
        (remote, branch) = remote_info
        repo = None
        for url in remote.urls:
            https_matches = re.match(GITHUB_HTTP_URL, url)
            ssh_matches = re.match(GITHUB_SSH_URL, url)
            if https_matches is not None:
                repo = f'{https_matches.group(2)}/{https_matches.group(3)}'
                break
            if ssh_matches is not None:
                repo = f'{ssh_matches.group(1)}/{ssh_matches.group(2)}'
                break
        if repo is None:
            return None
        return (repo, branch, self.module)