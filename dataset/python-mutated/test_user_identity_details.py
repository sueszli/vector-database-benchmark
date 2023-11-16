from sentry.models.authidentity import AuthIdentity
from sentry.models.authprovider import AuthProvider
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class DeleteUserIdentityTest(APITestCase):
    endpoint = 'sentry-api-0-user-identity-details'
    method = 'delete'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        auth_provider = AuthProvider.objects.create(organization_id=self.organization.id, provider='dummy')
        auth_identity = AuthIdentity.objects.create(auth_provider=auth_provider, ident=self.user.email, user=self.user)
        self.get_success_response(self.user.id, auth_identity.id, status_code=204)
        assert not AuthIdentity.objects.filter(id=auth_identity.id).exists()