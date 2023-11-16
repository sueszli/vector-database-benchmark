from sentry.models.integrations.integration_feature import IntegrationFeature, IntegrationTypes
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class IntegrationFeatureTest(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.sentry_app = self.create_sentry_app()
        self.integration_feature = IntegrationFeature.objects.get(target_id=self.sentry_app.id, target_type=IntegrationTypes.SENTRY_APP.value)

    def test_feature_str(self):
        if False:
            i = 10
            return i + 15
        assert self.integration_feature.feature_str() == 'integrations-api'

    def test_description(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.integration_feature.description == '%s can **utilize the Sentry API** to pull data or update resources in Sentry (with permissions granted, of course).' % self.sentry_app.name
        self.integration_feature.user_description = 'Custom description'
        self.integration_feature.save()
        assert self.integration_feature.description == 'Custom description'