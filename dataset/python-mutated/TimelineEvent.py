from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.TimelineEventSource
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class TimelineEvent(NonCompletableGithubObject):
    """
    This class represents IssueTimelineEvents. The reference can be found here https://docs.github.com/en/rest/reference/issues#timeline
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._actor: Attribute[github.NamedUser.NamedUser] = NotSet
        self._commit_id: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._event: Attribute[str] = NotSet
        self._id: Attribute[int] = NotSet
        self._node_id: Attribute[str] = NotSet
        self._commit_url: Attribute[str] = NotSet
        self._source: Attribute[github.TimelineEventSource.TimelineEventSource] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'id': self._id.value})

    @property
    def actor(self) -> github.NamedUser.NamedUser:
        if False:
            print('Hello World!')
        return self._actor.value

    @property
    def commit_id(self) -> str:
        if False:
            while True:
                i = 10
        return self._commit_id.value

    @property
    def created_at(self) -> datetime:
        if False:
            return 10
        return self._created_at.value

    @property
    def event(self) -> str:
        if False:
            print('Hello World!')
        return self._event.value

    @property
    def id(self) -> int:
        if False:
            print('Hello World!')
        return self._id.value

    @property
    def node_id(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._node_id.value

    @property
    def commit_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._commit_url.value

    @property
    def source(self) -> github.TimelineEventSource.TimelineEventSource | None:
        if False:
            while True:
                i = 10
        if self.event == 'cross-referenced' and self._source is not NotSet:
            return self._source.value
        return None

    @property
    def body(self) -> str | None:
        if False:
            print('Hello World!')
        if self.event == 'commented' and self._body is not NotSet:
            return self._body.value
        return None

    @property
    def author_association(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        if self.event == 'commented' and self._author_association is not NotSet:
            return self._author_association.value
        return None

    @property
    def url(self) -> str:
        if False:
            return 10
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        if 'actor' in attributes:
            self._actor = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['actor'])
        if 'commit_id' in attributes:
            self._commit_id = self._makeStringAttribute(attributes['commit_id'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'event' in attributes:
            self._event = self._makeStringAttribute(attributes['event'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'node_id' in attributes:
            self._node_id = self._makeStringAttribute(attributes['node_id'])
        if 'commit_url' in attributes:
            self._commit_url = self._makeStringAttribute(attributes['commit_url'])
        if 'source' in attributes:
            self._source = self._makeClassAttribute(github.TimelineEventSource.TimelineEventSource, attributes['source'])
        if 'body' in attributes:
            self._body = self._makeStringAttribute(attributes['body'])
        if 'author_association' in attributes:
            self._author_association = self._makeStringAttribute(attributes['author_association'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])