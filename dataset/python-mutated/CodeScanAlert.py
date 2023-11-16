from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CodeScanAlertInstance
import github.CodeScanRule
import github.CodeScanTool
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList

class CodeScanAlert(NonCompletableGithubObject):
    """
    This class represents alerts from code scanning.
    The reference can be found here https://docs.github.com/en/rest/reference/code-scanning.
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._number: Attribute[int] = NotSet
        self._rule: Attribute[github.CodeScanRule.CodeScanRule] = NotSet
        self._tool: Attribute[github.CodeScanTool.CodeScanTool] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._dismissed_at: Attribute[datetime | None] = NotSet
        self._dismissed_by: Attribute[github.NamedUser.NamedUser | None] = NotSet
        self._dismissed_reason: Attribute[str | None] = NotSet
        self._url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._instances_url: Attribute[str] = NotSet
        self._most_recent_instance: Attribute[github.CodeScanAlertInstance.CodeScanAlertInstance] = NotSet
        self._state: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'number': self.number})

    @property
    def number(self) -> int:
        if False:
            while True:
                i = 10
        return self._number.value

    @property
    def rule(self) -> github.CodeScanRule.CodeScanRule:
        if False:
            i = 10
            return i + 15
        return self._rule.value

    @property
    def tool(self) -> github.CodeScanTool.CodeScanTool:
        if False:
            print('Hello World!')
        return self._tool.value

    @property
    def created_at(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        return self._created_at.value

    @property
    def dismissed_at(self) -> datetime | None:
        if False:
            i = 10
            return i + 15
        return self._dismissed_at.value

    @property
    def dismissed_by(self) -> github.NamedUser.NamedUser | None:
        if False:
            i = 10
            return i + 15
        return self._dismissed_by.value

    @property
    def dismissed_reason(self) -> str | None:
        if False:
            print('Hello World!')
        return self._dismissed_reason.value

    @property
    def url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._url.value

    @property
    def html_url(self) -> str:
        if False:
            print('Hello World!')
        return self._html_url.value

    @property
    def instances_url(self) -> str:
        if False:
            print('Hello World!')
        return self._instances_url.value

    @property
    def most_recent_instance(self) -> github.CodeScanAlertInstance.CodeScanAlertInstance:
        if False:
            while True:
                i = 10
        return self._most_recent_instance.value

    @property
    def state(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._state.value

    def get_instances(self) -> PaginatedList[github.CodeScanAlertInstance.CodeScanAlertInstance]:
        if False:
            return 10
        '\n        :calls: `GET` on the URL for instances as provided by Github\n        '
        return PaginatedList(github.CodeScanAlertInstance.CodeScanAlertInstance, self._requester, self.instances_url, None)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'number' in attributes:
            self._number = self._makeIntAttribute(attributes['number'])
        if 'rule' in attributes:
            self._rule = self._makeClassAttribute(github.CodeScanRule.CodeScanRule, attributes['rule'])
        if 'tool' in attributes:
            self._tool = self._makeClassAttribute(github.CodeScanTool.CodeScanTool, attributes['tool'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'dismissed_at' in attributes:
            self._dismissed_at = self._makeDatetimeAttribute(attributes['dismissed_at'])
        if 'dismissed_by' in attributes:
            self._dismissed_by = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['dismissed_by'])
        if 'dismissed_reason' in attributes:
            self._dismissed_reason = self._makeStringAttribute(attributes['dismissed_reason'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'instances_url' in attributes:
            self._instances_url = self._makeStringAttribute(attributes['instances_url'])
        if 'most_recent_instance' in attributes:
            self._most_recent_instance = self._makeClassAttribute(github.CodeScanAlertInstance.CodeScanAlertInstance, attributes['most_recent_instance'])
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])