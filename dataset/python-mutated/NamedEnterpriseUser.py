from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class NamedEnterpriseUser(CompletableGithubObject):
    """
    This class represents NamedEnterpriseUsers. The reference can be found here https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._github_com_login: Attribute[str] = NotSet
        self._github_com_name: Attribute[str] = NotSet
        self._enterprise_server_user_ids: Attribute[list] = NotSet
        self._github_com_user: Attribute[bool] = NotSet
        self._enterprise_server_user: Attribute[bool] = NotSet
        self._visual_studio_subscription_user: Attribute[bool] = NotSet
        self._license_type: Attribute[str] = NotSet
        self._github_com_profile: Attribute[str] = NotSet
        self._github_com_member_roles: Attribute[list] = NotSet
        self._github_com_enterprise_roles: Attribute[list] = NotSet
        self._github_com_verified_domain_emails: Attribute[list] = NotSet
        self._github_com_saml_name_id: Attribute[str] = NotSet
        self._github_com_orgs_with_pending_invites: Attribute[list] = NotSet
        self._github_com_two_factor_auth: Attribute[bool] = NotSet
        self._enterprise_server_primary_emails: Attribute[list] = NotSet
        self._visual_studio_license_status: Attribute[str] = NotSet
        self._visual_studio_subscription_email: Attribute[str] = NotSet
        self._total_user_accounts: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'login': self._github_com_login.value})

    @property
    def github_com_login(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._github_com_login)
        return self._github_com_login.value

    @property
    def github_com_name(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._github_com_name)
        return self._github_com_name.value

    @property
    def enterprise_server_user_ids(self) -> list:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._enterprise_server_user_ids)
        return self._enterprise_server_user_ids.value

    @property
    def github_com_user(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._github_com_user)
        return self._github_com_user.value

    @property
    def enterprise_server_user(self) -> bool:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._enterprise_server_user)
        return self._enterprise_server_user.value

    @property
    def visual_studio_subscription_user(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._visual_studio_subscription_user)
        return self._visual_studio_subscription_user.value

    @property
    def license_type(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._license_type)
        return self._license_type.value

    @property
    def github_com_profile(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._github_com_profile)
        return self._github_com_profile.value

    @property
    def github_com_member_roles(self) -> list:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._github_com_member_roles)
        return self._github_com_member_roles.value

    @property
    def github_com_enterprise_roles(self) -> list:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._github_com_enterprise_roles)
        return self._github_com_enterprise_roles.value

    @property
    def github_com_verified_domain_emails(self) -> list:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._github_com_verified_domain_emails)
        return self._github_com_verified_domain_emails.value

    @property
    def github_com_saml_name_id(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._github_com_saml_name_id)
        return self._github_com_saml_name_id.value

    @property
    def github_com_orgs_with_pending_invites(self) -> list:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._github_com_orgs_with_pending_invites)
        return self._github_com_orgs_with_pending_invites.value

    @property
    def github_com_two_factor_auth(self) -> bool:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._github_com_two_factor_auth)
        return self._github_com_two_factor_auth.value

    @property
    def enterprise_server_primary_emails(self) -> list:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._enterprise_server_primary_emails)
        return self._enterprise_server_primary_emails.value

    @property
    def visual_studio_license_status(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._visual_studio_license_status)
        return self._visual_studio_license_status.value

    @property
    def visual_studio_subscription_email(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._visual_studio_subscription_email)
        return self._visual_studio_subscription_email.value

    @property
    def total_user_accounts(self) -> int:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._total_user_accounts)
        return self._total_user_accounts.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'github_com_login' in attributes:
            self._github_com_login = self._makeStringAttribute(attributes['github_com_login'])
        if 'github_com_name' in attributes:
            self._github_com_name = self._makeStringAttribute(attributes['github_com_name'])
        if 'enterprise_server_user_ids' in attributes:
            self._enterprise_server_user_ids = self._makeListOfStringsAttribute(attributes['enterprise_server_user_ids'])
        if 'github_com_user' in attributes:
            self._github_com_user = self._makeBoolAttribute(attributes['github_com_user'])
        if 'enterprise_server_user' in attributes:
            self._enterprise_server_user = self._makeBoolAttribute(attributes['enterprise_server_user'])
        if 'visual_studio_subscription_user' in attributes:
            self._visual_studio_subscription_user = self._makeBoolAttribute(attributes['visual_studio_subscription_user'])
        if 'license_type' in attributes:
            self._license_type = self._makeStringAttribute(attributes['license_type'])
        if 'github_com_profile' in attributes:
            self._github_com_profile = self._makeStringAttribute(attributes['github_com_profile'])
        if 'github_com_member_roles' in attributes:
            self._github_com_member_roles = self._makeListOfStringsAttribute(attributes['github_com_member_roles'])
        if 'github_com_enterprise_roles' in attributes:
            self._github_com_enterprise_roles = self._makeListOfStringsAttribute(attributes['github_com_enterprise_roles'])
        if 'github_com_verified_domain_emails' in attributes:
            self._github_com_verified_domain_emails = self._makeListOfStringsAttribute(attributes['github_com_verified_domain_emails'])
        if 'github_com_saml_name_id' in attributes:
            self._github_com_saml_name_id = self._makeStringAttribute(attributes['github_com_saml_name_id'])
        if 'github_com_orgs_with_pending_invites' in attributes:
            self._github_com_orgs_with_pending_invites = self._makeListOfStringsAttribute(attributes['github_com_orgs_with_pending_invites'])
        if 'github_com_two_factor_auth' in attributes:
            self._github_com_two_factor_auth = self._makeBoolAttribute(attributes['github_com_two_factor_auth'])
        if 'enterprise_server_primary_emails' in attributes:
            self._enterprise_server_primary_emails = self._makeListOfStringsAttribute(attributes['enterprise_server_primary_emails'])
        if 'visual_studio_license_status' in attributes:
            self._visual_studio_license_status = self._makeStringAttribute(attributes['visual_studio_license_status'])
        if 'visual_studio_subscription_email' in attributes:
            self._visual_studio_subscription_email = self._makeStringAttribute(attributes['visual_studio_subscription_email'])
        if 'total_user_accounts' in attributes:
            self._total_user_accounts = self._makeIntAttribute(attributes['total_user_accounts'])