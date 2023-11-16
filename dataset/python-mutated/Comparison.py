from __future__ import annotations
from typing import Any
import github.Commit
import github.File
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class Comparison(CompletableGithubObject):
    """
    This class represents Comparisons
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._ahead_by: Attribute[int] = NotSet
        self._base_commit: Attribute[github.Commit.Commit] = NotSet
        self._behind_by: Attribute[int] = NotSet
        self._commits: Attribute[list[github.Commit.Commit]] = NotSet
        self._diff_url: Attribute[str] = NotSet
        self._files: Attribute[list[github.File.File]] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._merge_base_commit: Attribute[github.Commit.Commit] = NotSet
        self._patch_url: Attribute[str] = NotSet
        self._permalink_url: Attribute[str] = NotSet
        self._status: Attribute[str] = NotSet
        self._total_commits: Attribute[int] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'url': self._url.value})

    @property
    def ahead_by(self) -> int:
        if False:
            return 10
        self._completeIfNotSet(self._ahead_by)
        return self._ahead_by.value

    @property
    def base_commit(self) -> github.Commit.Commit:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._base_commit)
        return self._base_commit.value

    @property
    def behind_by(self) -> int:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._behind_by)
        return self._behind_by.value

    @property
    def commits(self) -> list[github.Commit.Commit]:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._commits)
        return self._commits.value

    @property
    def diff_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._diff_url)
        return self._diff_url.value

    @property
    def files(self) -> list[github.File.File]:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._files)
        return self._files.value

    @property
    def html_url(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def merge_base_commit(self) -> github.Commit.Commit:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._merge_base_commit)
        return self._merge_base_commit.value

    @property
    def patch_url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._patch_url)
        return self._patch_url.value

    @property
    def permalink_url(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._permalink_url)
        return self._permalink_url.value

    @property
    def status(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._status)
        return self._status.value

    @property
    def total_commits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._total_commits)
        return self._total_commits.value

    @property
    def url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'ahead_by' in attributes:
            self._ahead_by = self._makeIntAttribute(attributes['ahead_by'])
        if 'base_commit' in attributes:
            self._base_commit = self._makeClassAttribute(github.Commit.Commit, attributes['base_commit'])
        if 'behind_by' in attributes:
            self._behind_by = self._makeIntAttribute(attributes['behind_by'])
        if 'commits' in attributes:
            self._commits = self._makeListOfClassesAttribute(github.Commit.Commit, attributes['commits'])
        if 'diff_url' in attributes:
            self._diff_url = self._makeStringAttribute(attributes['diff_url'])
        if 'files' in attributes:
            self._files = self._makeListOfClassesAttribute(github.File.File, attributes['files'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'merge_base_commit' in attributes:
            self._merge_base_commit = self._makeClassAttribute(github.Commit.Commit, attributes['merge_base_commit'])
        if 'patch_url' in attributes:
            self._patch_url = self._makeStringAttribute(attributes['patch_url'])
        if 'permalink_url' in attributes:
            self._permalink_url = self._makeStringAttribute(attributes['permalink_url'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])
        if 'total_commits' in attributes:
            self._total_commits = self._makeIntAttribute(attributes['total_commits'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])