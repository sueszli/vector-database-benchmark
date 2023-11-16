from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class NotificationSubject(NonCompletableGithubObject):
    """
    This class represents Subjects of Notifications. The reference can be found here https://docs.github.com/en/rest/reference/activity#list-notifications-for-the-authenticated-user
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._title: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet
        self._latest_comment_url: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.get__repr__({'title': self._title.value})

    @property
    def title(self) -> str:
        if False:
            print('Hello World!')
        return self._title.value

    @property
    def url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._url.value

    @property
    def latest_comment_url(self) -> str:
        if False:
            while True:
                i = 10
        return self._latest_comment_url.value

    @property
    def type(self) -> str:
        if False:
            print('Hello World!')
        return self._type.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'latest_comment_url' in attributes:
            self._latest_comment_url = self._makeStringAttribute(attributes['latest_comment_url'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])