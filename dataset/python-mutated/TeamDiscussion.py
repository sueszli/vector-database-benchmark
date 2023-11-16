from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class TeamDiscussion(CompletableGithubObject):
    """
    This class represents TeamDiscussions. The reference can be found here https://docs.github.com/en/rest/reference/teams#discussions
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._author: Attribute[github.NamedUser.NamedUser] = NotSet
        self._body: Attribute[str] = NotSet
        self._body_html: Attribute[str] = NotSet
        self._body_version: Attribute[str] = NotSet
        self._comments_count: Attribute[int] = NotSet
        self._comments_url: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._last_edited_at: Attribute[datetime] = NotSet
        self._node_id: Attribute[str] = NotSet
        self._number: Attribute[int] = NotSet
        self._pinned: Attribute[bool] = NotSet
        self._private: Attribute[bool] = NotSet
        self._team_url: Attribute[str] = NotSet
        self._title: Attribute[str] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'number': self._number.value, 'title': self._title.value})

    @property
    def author(self) -> github.NamedUser.NamedUser:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._author)
        return self._author.value

    @property
    def body(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._body)
        return self._body.value

    @property
    def body_html(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._body_html)
        return self._body_html.value

    @property
    def body_version(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._body_version)
        return self._body_version.value

    @property
    def comments_count(self) -> int:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._comments_count)
        return self._comments_count.value

    @property
    def comments_url(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._comments_url)
        return self._comments_url.value

    @property
    def created_at(self) -> datetime:
        if False:
            return 10
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def html_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def last_edited_at(self) -> datetime:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._last_edited_at)
        return self._last_edited_at.value

    @property
    def node_id(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._node_id)
        return self._node_id.value

    @property
    def number(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._number)
        return self._number.value

    @property
    def pinned(self) -> bool:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._pinned)
        return self._pinned.value

    @property
    def private(self) -> bool:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._private)
        return self._private.value

    @property
    def team_url(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._team_url)
        return self._team_url.value

    @property
    def title(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._title)
        return self._title.value

    @property
    def updated_at(self) -> datetime:
        if False:
            return 10
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'author' in attributes:
            self._author = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['author'])
        if 'body' in attributes:
            self._body = self._makeStringAttribute(attributes['body'])
        if 'body_html' in attributes:
            self._body_html = self._makeStringAttribute(attributes['body_html'])
        if 'body_version' in attributes:
            self._body_version = self._makeStringAttribute(attributes['body_version'])
        if 'comments_count' in attributes:
            self._comments_count = self._makeIntAttribute(attributes['comments_count'])
        if 'comments_url' in attributes:
            self._comments_url = self._makeStringAttribute(attributes['comments_url'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'last_edited_at' in attributes:
            self._last_edited_at = self._makeDatetimeAttribute(attributes['last_edited_at'])
        if 'node_id' in attributes:
            self._node_id = self._makeStringAttribute(attributes['node_id'])
        if 'number' in attributes:
            self._number = self._makeIntAttribute(attributes['number'])
        if 'pinned' in attributes:
            self._pinned = self._makeBoolAttribute(attributes['pinned'])
        if 'private' in attributes:
            self._private = self._makeBoolAttribute(attributes['private'])
        if 'team_url' in attributes:
            self._team_url = self._makeStringAttribute(attributes['team_url'])
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])