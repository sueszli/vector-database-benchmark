from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.NamedEnterpriseUser import NamedEnterpriseUser
from github.PaginatedList import PaginatedList

class EnterpriseConsumedLicenses(CompletableGithubObject):
    """
    This class represents license consumed by enterprises. The reference can be found here https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._total_seats_consumed: Attribute[int] = NotSet
        self._total_seats_purchased: Attribute[int] = NotSet
        self._enterprise: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'enterprise': self._enterprise.value})

    @property
    def total_seats_consumed(self) -> int:
        if False:
            print('Hello World!')
        return self._total_seats_consumed.value

    @property
    def total_seats_purchased(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._total_seats_purchased.value

    @property
    def enterprise(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._enterprise)
        return self._enterprise.value

    @property
    def url(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._url)
        return self._url.value

    def get_users(self) -> PaginatedList[NamedEnterpriseUser]:
        if False:
            print('Hello World!')
        '\n        :calls: `GET /enterprises/{enterprise}/consumed-licenses <https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses>`_\n        '
        url_parameters: Dict[str, Any] = {}
        return PaginatedList(NamedEnterpriseUser, self._requester, self.url, url_parameters, None, 'users', self.raw_data, self.raw_headers)

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'total_seats_consumed' in attributes:
            self._total_seats_consumed = self._makeIntAttribute(attributes['total_seats_consumed'])
        if 'total_seats_purchased' in attributes:
            self._total_seats_purchased = self._makeIntAttribute(attributes['total_seats_purchased'])
        if 'enterprise' in attributes:
            self._enterprise = self._makeStringAttribute(attributes['enterprise'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])