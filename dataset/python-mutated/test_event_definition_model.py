import pytest
from ee.models.event_definition import EnterpriseEventDefinition
from posthog.test.base import BaseTest

class TestEventDefinition(BaseTest):

    def test_errors_on_invalid_verified_by_type(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            EnterpriseEventDefinition.objects.create(team=self.team, name='enterprise event', owner=self.user, verified_by='Not user id')

    def test_default_verified_false(self):
        if False:
            print('Hello World!')
        eventDef = EnterpriseEventDefinition.objects.create(team=self.team, name='enterprise event', owner=self.user)
        assert eventDef.verified is False