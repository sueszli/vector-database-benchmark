"""Tests for learner playlist services."""
from __future__ import annotations
from core import feconf
from core.domain import learner_playlist_services
from core.domain import learner_progress_services
from core.domain import subscription_services
from core.platform import models
from core.tests import test_utils
from typing import Final, List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])
MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT: Final = feconf.MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT

class LearnerPlaylistTests(test_utils.GenericTestBase):
    """Test the services related to learner playlist services."""
    EXP_ID_0: Final = '0_en_arch_bridges_in_england'
    EXP_ID_1: Final = '1_fi_arch_sillat_suomi'
    EXP_ID_2: Final = '2_en_welcome_introduce_oppia'
    EXP_ID_3: Final = '3_welcome_oppia'
    COL_ID_0: Final = '0_arch_bridges_in_england'
    COL_ID_1: Final = '1_welcome_introduce_oppia'
    COL_ID_2: Final = '2_welcome_introduce_oppia_interactions'
    COL_ID_3: Final = '3_welcome_oppia_collection'
    USER_EMAIL: Final = 'user@example.com'
    USER_USERNAME: Final = 'user'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.USER_EMAIL, self.USER_USERNAME)
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.user_id = self.get_user_id_from_email(self.USER_EMAIL)
        self.save_new_valid_exploration(self.EXP_ID_0, self.owner_id, title='Bridges in England', category='Architecture', language_code='en')
        self.save_new_valid_exploration(self.EXP_ID_1, self.owner_id, title='Sillat Suomi', category='Architecture', language_code='fi')
        self.save_new_valid_exploration(self.EXP_ID_2, self.user_id, title='Introduce Oppia', category='Welcome', language_code='en')
        self.save_new_valid_exploration(self.EXP_ID_3, self.owner_id, title='Welcome Oppia', category='Welcome', language_code='en')
        self.save_new_default_collection(self.COL_ID_0, self.owner_id, title='Bridges', category='Architecture')
        self.save_new_default_collection(self.COL_ID_1, self.owner_id, title='Introduce Oppia', category='Welcome')
        self.save_new_default_collection(self.COL_ID_2, self.user_id, title='Introduce Interactions in Oppia', category='Welcome')
        self.save_new_default_collection(self.COL_ID_3, self.owner_id, title='Welcome Oppia Collection', category='Welcome')

    def _get_all_learner_playlist_exp_ids(self, user_id: str) -> List[str]:
        if False:
            print('Hello World!')
        "Returns the list of all the exploration ids in the learner's playlist\n        corresponding to the given user id.\n        "
        learner_playlist_model = user_models.LearnerPlaylistModel.get(user_id, strict=False)
        if learner_playlist_model:
            exp_ids: List[str] = learner_playlist_model.exploration_ids
            return exp_ids
        else:
            return []

    def _get_all_learner_playlist_collection_ids(self, user_id: str) -> List[str]:
        if False:
            while True:
                i = 10
        "Returns the list of all the collection ids in the learner's playlist\n        corresponding to the given user id.\n        "
        learner_playlist_model = user_models.LearnerPlaylistModel.get(user_id, strict=False)
        if learner_playlist_model:
            collection_ids: List[str] = learner_playlist_model.collection_ids
            return collection_ids
        else:
            return []

    def test_subscribed_exploration_cannot_be_added_to_playlist(self) -> None:
        if False:
            return 10
        subscription_services.subscribe_to_exploration(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])

    def test_single_exploration_is_added_correctly_to_playlist(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_1, position_to_be_inserted=0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_1, self.EXP_ID_0])

    def test_multiple_explorations_are_added_correctly_to_playlist(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])

    def test_adding_exisiting_exploration_changes_order_of_explorations(self) -> None:
        if False:
            while True:
                i = 10
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0, position_to_be_inserted=1)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_1, self.EXP_ID_0])

    def test_incomplete_exploration_is_not_added_to_learner_playlist(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0])
        state_name = 'state_name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0])

    def test_number_of_explorations_cannot_exceed_max(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_ids = ['SAMPLE_EXP_ID_%s' % index for index in range(0, MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT)]
        for exp_id in exp_ids:
            learner_progress_services.add_exp_to_learner_playlist(self.user_id, exp_id)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), exp_ids)
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, 'SAMPLE_EXP_ID_MAX', position_to_be_inserted=MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), exp_ids)
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, 'SAMPLE_EXP_ID_MAX')
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), exp_ids)

    def test_subscribed_collection_cannot_be_added_to_playlist(self) -> None:
        if False:
            print('Hello World!')
        subscription_services.subscribe_to_collection(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])

    def test_single_collection_is_added_correctly_to_playlist(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_1, position_to_be_inserted=0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_1, self.COL_ID_0])

    def test_multiple_collections_are_added_correctly_to_playlist(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])

    def test_adding_existing_collection_changes_order_of_collections(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0, position_to_be_inserted=1)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_1, self.COL_ID_0])

    def test_number_of_collections_cannot_exceed_max(self) -> None:
        if False:
            print('Hello World!')
        col_ids = ['SAMPLE_COL_ID_%s' % index for index in range(0, MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT)]
        for col_id in col_ids:
            learner_progress_services.add_collection_to_learner_playlist(self.user_id, col_id)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), col_ids)
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, 'SAMPLE_COL_ID_MAX', position_to_be_inserted=MAX_LEARNER_PLAYLIST_ACTIVITY_COUNT)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), col_ids)
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, 'SAMPLE_COL_ID_MAX')
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), col_ids)

    def test_remove_exploration_from_learner_playlist(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_0)
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])
        learner_playlist_services.remove_exploration_from_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_1])
        learner_playlist_services.remove_exploration_from_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [self.EXP_ID_1])
        learner_playlist_services.remove_exploration_from_learner_playlist(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_learner_playlist_exp_ids(self.user_id), [])

    def test_remove_collection_from_learner_playlist(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_0)
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])
        learner_playlist_services.remove_collection_from_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_1])
        learner_playlist_services.remove_collection_from_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [self.COL_ID_1])
        learner_playlist_services.remove_collection_from_learner_playlist(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_learner_playlist_collection_ids(self.user_id), [])

    def test_get_all_exp_ids_in_learner_playlist(self) -> None:
        if False:
            return 10
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [])
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_0)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_0])
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_1)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])

    def test_get_all_learner_playlist_collection_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [])
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_0])
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_1)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_0, self.COL_ID_1])