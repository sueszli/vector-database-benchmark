from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class IssuePullRequest(NonCompletableGithubObject):
    """
    This class represents IssuePullRequests
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._diff_url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._patch_url: Attribute[str] = NotSet

    @property
    def diff_url(self) -> str:
        if False:
            return 10
        return self._diff_url.value

    @property
    def html_url(self) -> str:
        if False:
            print('Hello World!')
        return self._html_url.value

    @property
    def patch_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._patch_url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'diff_url' in attributes:
            self._diff_url = self._makeStringAttribute(attributes['diff_url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'patch_url' in attributes:
            self._patch_url = self._makeStringAttribute(attributes['patch_url'])