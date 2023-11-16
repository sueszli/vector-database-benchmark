from sentry.models.integrations.external_actor import ExternalActor
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class ExternalUserDetailsTest(APITestCase):
    endpoint = 'sentry-api-0-organization-external-user-details'
    method = 'put'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(user=self.user)
        self.external_user = self.create_external_user(self.user, self.organization, external_name='@NisanthanNanthakumar')

    def test_basic_delete(self):
        if False:
            print('Hello World!')
        self.get_success_response(self.organization.slug, self.external_user.id, method='delete')
        assert not ExternalActor.objects.filter(id=str(self.external_user.id)).exists()

    def test_basic_update(self):
        if False:
            return 10
        with self.feature({'organizations:integrations-codeowners': True}):
            data = {'externalName': '@new_username'}
            response = self.get_success_response(self.organization.slug, self.external_user.id, **data)
        assert response.data['id'] == str(self.external_user.id)
        assert response.data['externalName'] == '@new_username'

    def test_invalid_provider_update(self):
        if False:
            print('Hello World!')
        with self.feature({'organizations:integrations-codeowners': True}):
            data = {'provider': 'unknown'}
            response = self.get_error_response(self.organization.slug, self.external_user.id, status_code=400, **data)
        assert response.data == {'provider': ['"unknown" is not a valid choice.']}

    def test_delete_another_orgs_external_user(self):
        if False:
            while True:
                i = 10
        invalid_user = self.create_user()
        invalid_organization = self.create_organization(owner=invalid_user)
        self.login_as(user=invalid_user)
        resp = self.get_error_response(invalid_organization.slug, self.external_user.id, method='delete')
        assert resp.status_code == 404