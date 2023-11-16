import pytest
from ee.models.property_definition import EnterprisePropertyDefinition
from posthog.test.base import BaseTest

class TestPropertyDefinition(BaseTest):

    def test_errors_on_invalid_verified_by_type(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            EnterprisePropertyDefinition.objects.create(team=self.team, name='enterprise property', verified_by='Not user id')

    def test_default_verified_false(self):
        if False:
            while True:
                i = 10
        definition = EnterprisePropertyDefinition.objects.create(team=self.team, name='enterprise property')
        assert definition.verified is False