import pytest
from sentry.testutils.cases import PermissionTestCase, TestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class OrganizationIntegrationSetupPermissionTest(PermissionTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.path = f'/organizations/{self.organization.slug}/integrations/example/setup/'

    @pytest.mark.xfail
    def test_manager_can_load(self):
        if False:
            print('Hello World!')
        self.assert_role_can_access(self.path, 'manager')

    @pytest.mark.xfail
    def test_owner_can_load(self):
        if False:
            print('Hello World!')
        self.assert_owner_can_access(self.path)

@control_silo_test(stable=True)
class OrganizationIntegrationSetupTest(TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.organization = self.create_organization(name='foo', owner=self.user)
        self.login_as(self.user)
        self.path = f'/organizations/{self.organization.slug}/integrations/example/setup/'

    def test_basic_flow(self):
        if False:
            while True:
                i = 10
        resp = self.client.get(self.path)
        assert resp.status_code == 200
        resp = self.client.post(self.path, data={'name': 'morty'})
        assert resp.status_code == 200
        assert b'morty' in resp.content