from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test
class UserOrganizationsTest(APITestCase):
    endpoint = 'sentry-api-0-user-organizations'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            while True:
                i = 10
        organization_id = self.organization.id
        response = self.get_success_response('me')
        assert len(response.data) == 1
        assert response.data[0]['id'] == str(organization_id)