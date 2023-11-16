from unittest.case import TestCase
import pytest
from environments.identities.models import Identity
from environments.models import Environment
from features.models import Feature, FeatureState
from integrations.rudderstack.models import RudderstackConfiguration
from integrations.rudderstack.rudderstack import RudderstackWrapper
from organisations.models import Organisation
from projects.models import Project

@pytest.mark.django_db
class RudderstackConfigurationTestCase(TestCase):

    def test_rudderstack_wrapper_generate_user_data(self):
        if False:
            for i in range(10):
                print('nop')
        rudderstack_config = RudderstackConfiguration(api_key='123key', base_url='https://api.rudderstack.com/')
        rudderstack_wrapper = RudderstackWrapper(rudderstack_config)
        organisation = Organisation.objects.create(name='Test Org')
        project = Project.objects.create(name='Test Project', organisation=organisation)
        environment = Environment.objects.create(name='Test Environment', project=project)
        identity = Identity.objects.create(identifier='user123', environment=environment)
        feature = Feature.objects.create(name='Test Feature', project=project)
        feature_states = FeatureState.objects.filter(feature=feature)
        user_data = rudderstack_wrapper.generate_user_data(identity=identity, feature_states=feature_states)
        assert user_data == {'user_id': identity.identifier, 'traits': {feature.name: False}}