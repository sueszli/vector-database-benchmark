import pytest
from sentry.models.organization import Organization
from sentry.models.organizationmapping import OrganizationMapping
from sentry.testutils.cases import TestMigrations

@pytest.mark.skip('Migration is no longer runnable. Retain until migration is removed.')
class BackfillNotificationSettingTest(TestMigrations):
    migrate_from = '0471_dashboard_widget_description'
    migrate_to = '0472_delete_past_organization_mappings'

    def setup_initial_state(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_organization(name='foo')
        self.create_organization(name='bar')
        organizations = Organization.objects.filter()
        assert len(organizations) == 3
        organizations_mappings = OrganizationMapping.objects.filter()
        assert len(organizations_mappings) == 2

    def test_deleting_all_org_mappings(self):
        if False:
            i = 10
            return i + 15
        organization_mappings = OrganizationMapping.objects.filter()
        assert len(organization_mappings) == 0
        organizations = Organization.objects.filter()
        assert len(organizations) == 3