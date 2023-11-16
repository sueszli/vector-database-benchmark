from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class EnvironmentDeploymentBranchPolicy(NonCompletableGithubObject):
    """
    This class represents a deployment branch policy for an environment. The reference can be found here https://docs.github.com/en/rest/reference/deployments#environments
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._protected_branches: Attribute[bool] = NotSet
        self._custom_branch_policies: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({})

    @property
    def protected_branches(self) -> bool:
        if False:
            return 10
        return self._protected_branches.value

    @property
    def custom_branch_policies(self) -> bool:
        if False:
            return 10
        return self._custom_branch_policies.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'protected_branches' in attributes:
            self._protected_branches = self._makeBoolAttribute(attributes['protected_branches'])
        if 'custom_branch_policies' in attributes:
            self._custom_branch_policies = self._makeBoolAttribute(attributes['custom_branch_policies'])

class EnvironmentDeploymentBranchPolicyParams:
    """
    This class presents the deployment branch policy parameters as can be configured for an Environment.
    """

    def __init__(self, protected_branches: bool=False, custom_branch_policies: bool=False):
        if False:
            return 10
        assert isinstance(protected_branches, bool)
        assert isinstance(custom_branch_policies, bool)
        self.protected_branches = protected_branches
        self.custom_branch_policies = custom_branch_policies

    def _asdict(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'protected_branches': self.protected_branches, 'custom_branch_policies': self.custom_branch_policies}