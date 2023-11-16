"""Tests for fetching the features Oppia provides to its users."""
from __future__ import annotations
from core import feconf
from core.domain import opportunity_services
from core.domain import rights_manager
from core.domain import user_services
from core.tests import test_utils
from typing import Final

def exploration_features_url(exp_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns URL for getting which features the given exploration supports.'
    return '%s/%s' % (feconf.EXPLORATION_FEATURES_PREFIX, exp_id)

class ExplorationFeaturesTestBase(test_utils.GenericTestBase):
    """Does common exploration set up for testing feature handlers."""
    EXP_ID: Final = 'expId'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.save_new_valid_exploration(self.EXP_ID, editor_id, title='Explore!', end_state_name='END')
        editor_actions_info = user_services.get_user_actions_info(editor_id)
        rights_manager.publish_exploration(editor_actions_info, self.EXP_ID)

class ExplorationPlaythroughRecordingFeatureTest(ExplorationFeaturesTestBase):
    """Tests for fetching whether playthrough recording is enabled."""

    def test_can_record_playthroughs_in_curated_explorations(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.swap_to_always_return(opportunity_services, 'is_exploration_available_for_contribution', True):
            json_response = self.get_json(exploration_features_url(self.EXP_ID))
        self.assertTrue(json_response['exploration_is_curated'])

    def test_can_not_record_playthroughs_with_non_curated_exps(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.swap_to_always_return(opportunity_services, 'is_exploration_available_for_contribution', False):
            json_response = self.get_json(exploration_features_url(self.EXP_ID))
        self.assertFalse(json_response['exploration_is_curated'])