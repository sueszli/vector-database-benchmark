from sentry.models.identity import Identity, IdentityProvider, IdentityStatus
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class UserIdentityTest(APITestCase):
    endpoint = 'sentry-api-0-user-identity'
    method = 'get'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        idp = IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
        Identity.objects.create(external_id='UXXXXXXX1', idp=idp, user=self.user, status=IdentityStatus.VALID, scopes=[])
        response = self.get_success_response(self.user.id)
        assert response.data[0]['identityProvider']['type'] == 'slack'