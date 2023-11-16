from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CommitStats
import github.Gist
import github.GithubObject
import github.NamedUser
from github.GistFile import GistFile
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class GistHistoryState(CompletableGithubObject):
    """
    This class represents GistHistoryStates
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._change_status: Attribute[github.CommitStats.CommitStats] = NotSet
        self._comments: Attribute[int] = NotSet
        self._comments_url: Attribute[str] = NotSet
        self._commits_url: Attribute[str] = NotSet
        self._committed_at: Attribute[datetime] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._description: Attribute[str] = NotSet
        self._files: Attribute[dict[str, GistFile]] = NotSet
        self._forks: Attribute[list[github.Gist.Gist]] = NotSet
        self._forks_url: Attribute[str] = NotSet
        self._git_pull_url: Attribute[str] = NotSet
        self._git_push_url: Attribute[str] = NotSet
        self._history: Attribute[list[GistHistoryState]] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._id: Attribute[str] = NotSet
        self._owner: Attribute[github.NamedUser.NamedUser] = NotSet
        self._public: Attribute[bool] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet
        self._version: Attribute[str] = NotSet

    @property
    def change_status(self) -> github.CommitStats.CommitStats:
        if False:
            return 10
        self._completeIfNotSet(self._change_status)
        return self._change_status.value

    @property
    def comments(self) -> int:
        if False:
            return 10
        self._completeIfNotSet(self._comments)
        return self._comments.value

    @property
    def comments_url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._comments_url)
        return self._comments_url.value

    @property
    def commits_url(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._commits_url)
        return self._commits_url.value

    @property
    def committed_at(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._committed_at)
        return self._committed_at.value

    @property
    def created_at(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def description(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._description)
        return self._description.value

    @property
    def files(self) -> dict[str, GistFile]:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._files)
        return self._files.value

    @property
    def forks(self) -> list[github.Gist.Gist]:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._forks)
        return self._forks.value

    @property
    def forks_url(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._forks_url)
        return self._forks_url.value

    @property
    def git_pull_url(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._git_pull_url)
        return self._git_pull_url.value

    @property
    def git_push_url(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._git_push_url)
        return self._git_push_url.value

    @property
    def history(self) -> list[GistHistoryState]:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._history)
        return self._history.value

    @property
    def html_url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def id(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def owner(self) -> github.NamedUser.NamedUser:
        if False:
            return 10
        self._completeIfNotSet(self._owner)
        return self._owner.value

    @property
    def public(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._public)
        return self._public.value

    @property
    def updated_at(self) -> datetime:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def user(self) -> github.NamedUser.NamedUser:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._user)
        return self._user.value

    @property
    def version(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._version)
        return self._version.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'change_status' in attributes:
            self._change_status = self._makeClassAttribute(github.CommitStats.CommitStats, attributes['change_status'])
        if 'comments' in attributes:
            self._comments = self._makeIntAttribute(attributes['comments'])
        if 'comments_url' in attributes:
            self._comments_url = self._makeStringAttribute(attributes['comments_url'])
        if 'commits_url' in attributes:
            self._commits_url = self._makeStringAttribute(attributes['commits_url'])
        if 'committed_at' in attributes:
            self._committed_at = self._makeDatetimeAttribute(attributes['committed_at'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])
        if 'files' in attributes:
            self._files = self._makeDictOfStringsToClassesAttribute(github.GistFile.GistFile, attributes['files'])
        if 'forks' in attributes:
            self._forks = self._makeListOfClassesAttribute(github.Gist.Gist, attributes['forks'])
        if 'forks_url' in attributes:
            self._forks_url = self._makeStringAttribute(attributes['forks_url'])
        if 'git_pull_url' in attributes:
            self._git_pull_url = self._makeStringAttribute(attributes['git_pull_url'])
        if 'git_push_url' in attributes:
            self._git_push_url = self._makeStringAttribute(attributes['git_push_url'])
        if 'history' in attributes:
            self._history = self._makeListOfClassesAttribute(GistHistoryState, attributes['history'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'id' in attributes:
            self._id = self._makeStringAttribute(attributes['id'])
        if 'owner' in attributes:
            self._owner = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['owner'])
        if 'public' in attributes:
            self._public = self._makeBoolAttribute(attributes['public'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])
        if 'version' in attributes:
            self._version = self._makeStringAttribute(attributes['version'])