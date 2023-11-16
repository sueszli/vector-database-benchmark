from functools import cached_property
from typing import Any, Mapping
import pytest
from sentry.auth.exceptions import IdentityNotValid
from sentry.auth.providers.oauth2 import OAuth2Provider
from sentry.models.authidentity import AuthIdentity
from sentry.models.authprovider import AuthProvider
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test

class DummyOAuth2Provider(OAuth2Provider):
    name = 'dummy'

    def get_client_id(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_client_secret(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_refresh_token_url(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def build_identity(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            return 10
        raise NotImplementedError

    def build_config(self, state):
        if False:
            return 10
        pass

@control_silo_test(stable=True)
class OAuth2ProviderTest(TestCase):

    @cached_property
    def auth_provider(self):
        if False:
            while True:
                i = 10
        return AuthProvider.objects.create(provider='oauth2', organization_id=self.organization.id)

    def test_refresh_identity_without_refresh_token(self):
        if False:
            while True:
                i = 10
        auth_identity = AuthIdentity.objects.create(auth_provider=self.auth_provider, user=self.user, data={'access_token': 'access_token'})
        provider = DummyOAuth2Provider(key=self.auth_provider.provider)
        with pytest.raises(IdentityNotValid):
            provider.refresh_identity(auth_identity)