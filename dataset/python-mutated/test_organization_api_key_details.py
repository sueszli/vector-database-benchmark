from sentry.hybridcloud.models import ApiKeyReplica
from sentry.models.apikey import ApiKey
from sentry.silo import SiloMode
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import assume_test_silo_mode, control_silo_test
DEFAULT_SCOPES = ['project:read', 'event:read', 'team:read', 'org:read', 'member:read']

class OrganizationApiKeyDetailsBase(APITestCase):
    endpoint = 'sentry-api-0-organization-api-key-details'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.login_as(self.user)
        self.api_key = ApiKey.objects.create(organization_id=self.organization.id, scope_list=DEFAULT_SCOPES)

@control_silo_test(stable=True)
class OrganizationApiKeyDetails(OrganizationApiKeyDetailsBase):

    def test_api_key_no_exist(self):
        if False:
            return 10
        self.get_error_response(self.organization.slug, 123456, status_code=404)

    def test_get_api_details(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_success_response(self.organization.slug, self.api_key.id)
        assert response.data.get('id') == str(self.api_key.id)

@control_silo_test(stable=True)
class OrganizationApiKeyDetailsPut(OrganizationApiKeyDetailsBase):
    method = 'put'

    def test_update_api_key_details(self):
        if False:
            return 10
        data = {'label': 'New Label', 'allowed_origins': 'sentry.io', 'scope_list': ['a', 'b', 'c', 'd']}
        self.get_success_response(self.organization.slug, self.api_key.id, **data)
        api_key = ApiKey.objects.get(id=self.api_key.id, organization_id=self.organization.id)
        assert api_key.label == 'New Label'
        assert api_key.allowed_origins == 'sentry.io'
        assert api_key.get_scopes() == ['a', 'b', 'c', 'd']

    def test_update_api_key_details_legacy_data(self):
        if False:
            while True:
                i = 10
        self.api_key.scope_list = '{event:read,member:read,org:read,project:read,team:read}'
        self.api_key.save()
        with assume_test_silo_mode(SiloMode.REGION):
            assert ApiKeyReplica.objects.get(apikey_id=self.api_key.id).get_scopes() == ['event:read', 'member:read', 'org:read', 'project:read', 'team:read']
        data = {'scope_list': ['a', 'b', 'c', 'd']}
        self.get_success_response(self.organization.slug, self.api_key.id, **data)
        api_key = ApiKey.objects.get(id=self.api_key.id, organization_id=self.organization.id)
        assert api_key.get_scopes() == ['a', 'b', 'c', 'd']

@control_silo_test(stable=True)
class OrganizationApiKeyDetailsDelete(OrganizationApiKeyDetailsBase):
    method = 'delete'

    def test_can_delete_api_key(self):
        if False:
            print('Hello World!')
        self.get_success_response(self.organization.slug, self.api_key.id)
        self.get_error_response(self.organization.slug, self.api_key.id, method='get', status_code=404)