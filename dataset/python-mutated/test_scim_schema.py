from sentry.testutils.cases import SCIMTestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class SCIMSchemaEndpointTest(SCIMTestCase):
    endpoint = 'sentry-api-0-organization-scim-schema-index'

    def test_schema_200s(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_success_response(self.organization.slug)