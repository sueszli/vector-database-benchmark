"""Tests for learner progress services."""
from __future__ import annotations
import datetime
from core.constants import constants
from core.domain import collection_domain
from core.domain import collection_services
from core.domain import config_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import learner_goals_services
from core.domain import learner_playlist_services
from core.domain import learner_progress_services
from core.domain import rights_manager
from core.domain import story_domain
from core.domain import story_services
from core.domain import subtopic_page_domain
from core.domain import subtopic_page_services
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.domain import topic_services
from core.domain import user_services
from core.platform import models
from core.tests import test_utils
from typing import Final, List, TypedDict
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

class IncompleteExplorationDetailsDict(TypedDict):
    """Type for the incompletely played exploration's details dictionary."""
    timestamp: datetime.datetime
    state_name: str
    version: int

class LearnerProgressTests(test_utils.GenericTestBase):
    """Test the services related to tracking the progress of the learner."""
    EXP_ID_0: Final = '0_en_arch_bridges_in_england'
    EXP_ID_1: Final = '1_fi_arch_sillat_suomi'
    EXP_ID_2: Final = '2_en_welcome_introduce_oppia'
    EXP_ID_3: Final = '3_welcome_oppia'
    EXP_ID_4: Final = 'exp_4'
    EXP_ID_5: Final = 'exp_5'
    EXP_ID_6: Final = 'exp_6'
    EXP_ID_7: Final = 'exp_7'
    COL_ID_0: Final = '0_arch_bridges_in_england'
    COL_ID_1: Final = '1_welcome_introduce_oppia'
    COL_ID_2: Final = '2_welcome_introduce_oppia_interactions'
    COL_ID_3: Final = '3_welcome_oppia_collection'
    STORY_ID_0: Final = 'story_0'
    TOPIC_ID_0: Final = 'topic_0'
    STORY_ID_1: Final = 'story_1'
    STORY_ID_2: Final = 'story_2'
    STORY_ID_3: Final = 'story_3'
    TOPIC_ID_1: Final = 'topic_1'
    TOPIC_ID_2: Final = 'topic_2'
    TOPIC_ID_3: Final = 'topic_3'
    USER_EMAIL: Final = 'user@example.com'
    USER_USERNAME: Final = 'user'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.USER_EMAIL, self.USER_USERNAME)
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.signup(self.CURRICULUM_ADMIN_EMAIL, self.CURRICULUM_ADMIN_USERNAME)
        self.set_curriculum_admins([self.CURRICULUM_ADMIN_USERNAME])
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.user_id = self.get_user_id_from_email(self.USER_EMAIL)
        self.admin_id = self.get_user_id_from_email(self.CURRICULUM_ADMIN_EMAIL)
        self.save_new_valid_exploration(self.EXP_ID_0, self.owner_id, title='Bridges in England', category='Architecture', language_code='en')
        self.publish_exploration(self.owner_id, self.EXP_ID_0)
        self.save_new_valid_exploration(self.EXP_ID_1, self.owner_id, title='Sillat Suomi', category='Architecture', language_code='fi')
        self.publish_exploration(self.owner_id, self.EXP_ID_1)
        self.save_new_valid_exploration(self.EXP_ID_2, self.user_id, title='Introduce Oppia', category='Welcome', language_code='en')
        self.publish_exploration(self.user_id, self.EXP_ID_2)
        self.save_new_valid_exploration(self.EXP_ID_3, self.owner_id, title='Welcome Oppia', category='Welcome', language_code='en')
        self.publish_exploration(self.owner_id, self.EXP_ID_3)
        self.save_new_valid_exploration(self.EXP_ID_4, self.owner_id, title='A title', category='Art', language_code='en', correctness_feedback_enabled=True)
        self.publish_exploration(self.owner_id, self.EXP_ID_4)
        self.save_new_valid_exploration(self.EXP_ID_5, self.owner_id, title='Title', category='Art', language_code='en', correctness_feedback_enabled=True)
        self.publish_exploration(self.owner_id, self.EXP_ID_5)
        self.save_new_valid_exploration(self.EXP_ID_6, self.owner_id, title='A title', category='Art', language_code='en', correctness_feedback_enabled=True)
        self.publish_exploration(self.owner_id, self.EXP_ID_6)
        self.save_new_valid_exploration(self.EXP_ID_7, self.owner_id, title='A title', category='Art', language_code='en', correctness_feedback_enabled=True)
        self.publish_exploration(self.owner_id, self.EXP_ID_7)
        self.save_new_default_collection(self.COL_ID_0, self.owner_id, title='Bridges', category='Architecture')
        self.publish_collection(self.owner_id, self.COL_ID_0)
        self.save_new_default_collection(self.COL_ID_1, self.owner_id, title='Introduce Oppia', category='Welcome')
        self.publish_collection(self.owner_id, self.COL_ID_1)
        self.save_new_default_collection(self.COL_ID_2, self.user_id, title='Introduce Interactions in Oppia', category='Welcome')
        self.publish_collection(self.user_id, self.COL_ID_2)
        self.save_new_default_collection(self.COL_ID_3, self.owner_id, title='Welcome Oppia Collection', category='Welcome')
        self.publish_collection(self.owner_id, self.COL_ID_3)
        topic = topic_domain.Topic.create_default_topic(self.TOPIC_ID_0, 'topic', 'abbrev', 'description', 'fragm')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        subtopic_page = subtopic_page_domain.SubtopicPage.create_default_subtopic_page(1, self.TOPIC_ID_0)
        subtopic_page_services.save_subtopic_page(self.owner_id, subtopic_page, 'Added subtopic', [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_SUBTOPIC, 'subtopic_id': 1, 'title': 'Sample', 'url_fragment': 'dummy-fragment'})])
        topic_services.save_new_topic(self.owner_id, topic)
        self.save_new_story(self.STORY_ID_0, self.owner_id, self.TOPIC_ID_0)
        topic_services.add_canonical_story(self.owner_id, self.TOPIC_ID_0, self.STORY_ID_0)
        changelist = [story_domain.StoryChange({'cmd': story_domain.CMD_ADD_STORY_NODE, 'node_id': 'node_1', 'title': 'Title 1'}), story_domain.StoryChange({'cmd': story_domain.CMD_UPDATE_STORY_NODE_PROPERTY, 'property_name': story_domain.STORY_NODE_PROPERTY_EXPLORATION_ID, 'old_value': None, 'new_value': self.EXP_ID_4, 'node_id': 'node_1'})]
        story_services.update_story(self.owner_id, self.STORY_ID_0, changelist, 'Added node.')
        topic = topic_domain.Topic.create_default_topic(self.TOPIC_ID_1, 'topic 1', 'abbrev-one', 'description 1', 'fragm')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url-one')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        subtopic_page = subtopic_page_domain.SubtopicPage.create_default_subtopic_page(1, self.TOPIC_ID_1)
        subtopic_page_services.save_subtopic_page(self.owner_id, subtopic_page, 'Added subtopic', [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_SUBTOPIC, 'subtopic_id': 1, 'title': 'Sample', 'url_fragment': 'fragment'})])
        topic_services.save_new_topic(self.owner_id, topic)
        self.save_new_story(self.STORY_ID_1, self.owner_id, self.TOPIC_ID_1)
        topic_services.add_canonical_story(self.owner_id, self.TOPIC_ID_1, self.STORY_ID_1)
        changelist = [story_domain.StoryChange({'cmd': story_domain.CMD_ADD_STORY_NODE, 'node_id': 'node_1', 'title': 'Title 1'}), story_domain.StoryChange({'cmd': story_domain.CMD_UPDATE_STORY_NODE_PROPERTY, 'property_name': story_domain.STORY_NODE_PROPERTY_EXPLORATION_ID, 'old_value': None, 'new_value': self.EXP_ID_5, 'node_id': 'node_1'})]
        story_services.update_story(self.owner_id, self.STORY_ID_1, changelist, 'Added Node 1.')
        topic = topic_domain.Topic.create_default_topic(self.TOPIC_ID_2, 'topic 2', 'abbrev-two', 'description 2', 'fragm')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url-one')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        subtopic_page = subtopic_page_domain.SubtopicPage.create_default_subtopic_page(1, self.TOPIC_ID_2)
        subtopic_page_services.save_subtopic_page(self.owner_id, subtopic_page, 'Added subtopic', [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_SUBTOPIC, 'subtopic_id': 1, 'title': 'Sample', 'url_fragment': 'sample-fragment'})])
        topic_services.save_new_topic(self.owner_id, topic)
        self.save_new_story(self.STORY_ID_2, self.owner_id, self.TOPIC_ID_2)
        topic_services.add_canonical_story(self.owner_id, self.TOPIC_ID_2, self.STORY_ID_2)
        topic = topic_domain.Topic.create_default_topic(self.TOPIC_ID_3, 'topic 3', 'abbrev-three', 'description 3', 'fragm')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url-one')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        subtopic_page = subtopic_page_domain.SubtopicPage.create_default_subtopic_page(1, self.TOPIC_ID_3)
        subtopic_page_services.save_subtopic_page(self.owner_id, subtopic_page, 'Added subtopic', [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_SUBTOPIC, 'subtopic_id': 1, 'title': 'Sample', 'url_fragment': 'sample-fragment'})])
        topic_services.save_new_topic(self.owner_id, topic)
        self.save_new_story(self.STORY_ID_3, self.owner_id, self.TOPIC_ID_3)
        topic_services.add_canonical_story(self.owner_id, self.TOPIC_ID_3, self.STORY_ID_3)
        topic_services.publish_story(self.TOPIC_ID_0, self.STORY_ID_0, self.admin_id)
        topic_services.publish_topic(self.TOPIC_ID_0, self.admin_id)
        topic_services.publish_story(self.TOPIC_ID_1, self.STORY_ID_1, self.admin_id)
        topic_services.publish_topic(self.TOPIC_ID_1, self.admin_id)
        topic_services.publish_story(self.TOPIC_ID_2, self.STORY_ID_2, self.admin_id)
        topic_services.publish_topic(self.TOPIC_ID_2, self.admin_id)
        topic_services.publish_story(self.TOPIC_ID_3, self.STORY_ID_3, self.admin_id)
        topic_services.publish_topic(self.TOPIC_ID_3, self.admin_id)

    def _get_all_completed_exp_ids(self, user_id: str) -> List[str]:
        if False:
            return 10
        'Gets the ids of all the explorations completed by the learner\n        corresponding to the given user id.\n        '
        completed_activities_model = user_models.CompletedActivitiesModel.get(user_id, strict=False)
        if completed_activities_model:
            exploration_ids: List[str] = completed_activities_model.exploration_ids
            return exploration_ids
        else:
            return []

    def _get_all_completed_collection_ids(self, user_id: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Gets the ids of all the collections completed by the learner\n        corresponding to the given user id.\n        '
        completed_activities_model = user_models.CompletedActivitiesModel.get(user_id, strict=False)
        if completed_activities_model:
            collection_ids: List[str] = completed_activities_model.collection_ids
            return collection_ids
        else:
            return []

    def _get_all_completed_story_ids(self, user_id: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Gets the ids of all the stories completed by the learner\n        corresponding to the given user id.\n        '
        completed_activities_model = user_models.CompletedActivitiesModel.get(user_id, strict=False)
        if completed_activities_model:
            story_ids: List[str] = completed_activities_model.story_ids
            return story_ids
        else:
            return []

    def _get_all_learnt_topic_ids(self, user_id: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Gets the ids of all the topics learnt by the learner\n        corresponding to the given user id.\n        '
        completed_activities_model = user_models.CompletedActivitiesModel.get(user_id, strict=False)
        if completed_activities_model:
            learnt_topic_ids: List[str] = completed_activities_model.learnt_topic_ids
            return learnt_topic_ids
        else:
            return []

    def _get_all_incomplete_exp_ids(self, user_id: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Gets the ids of all the explorations not fully completed by the\n        learner corresponding to the given user id.\n        '
        incomplete_activities_model = user_models.IncompleteActivitiesModel.get(user_id, strict=False)
        if incomplete_activities_model:
            exploration_ids: List[str] = incomplete_activities_model.exploration_ids
            return exploration_ids
        else:
            return []

    def _get_incomplete_exp_details(self, user_id: str, exploration_id: str) -> IncompleteExplorationDetailsDict:
        if False:
            print('Hello World!')
        'Returns the dict containing all the exploration details that are\n        incompletely played by the learner corresponding to the given user id.\n        '
        incomplete_exploration_user_model = user_models.ExpUserLastPlaythroughModel.get(user_id, exploration_id)
        assert incomplete_exploration_user_model is not None
        return {'timestamp': incomplete_exploration_user_model.last_updated, 'state_name': incomplete_exploration_user_model.last_played_state_name, 'version': incomplete_exploration_user_model.last_played_exp_version}

    def _check_if_exp_details_match(self, actual_details: IncompleteExplorationDetailsDict, details_fetched_from_model: IncompleteExplorationDetailsDict) -> None:
        if False:
            while True:
                i = 10
        'Verifies the exploration details fetched from the model matches the\n        actual details.\n        '
        self.assertEqual(actual_details['state_name'], details_fetched_from_model['state_name'])
        self.assertEqual(actual_details['version'], details_fetched_from_model['version'])
        self.assertLess((actual_details['timestamp'] - details_fetched_from_model['timestamp']).total_seconds(), 10)

    def _get_all_incomplete_collection_ids(self, user_id: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns the list of all the collection ids that are incompletely\n        played by the learner corresponding to the given user id.\n        '
        incomplete_activities_model = user_models.IncompleteActivitiesModel.get(user_id, strict=False)
        if incomplete_activities_model:
            collection_ids: List[str] = incomplete_activities_model.collection_ids
            return collection_ids
        else:
            return []

    def _get_all_incomplete_story_ids(self, user_id: str) -> List[str]:
        if False:
            print('Hello World!')
        'Returns the list of all the story ids that are incompletely\n        played by the learner corresponding to the given user id.\n        '
        incomplete_activities_model = user_models.IncompleteActivitiesModel.get(user_id, strict=False)
        if incomplete_activities_model:
            story_ids: List[str] = incomplete_activities_model.story_ids
            return story_ids
        else:
            return []

    def _get_all_partially_learnt_topic_ids(self, user_id: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns the list of all the topics ids that are partially\n        learnt by the learner corresponding to the given user id.\n        '
        incomplete_activities_model = user_models.IncompleteActivitiesModel.get(user_id, strict=False)
        if incomplete_activities_model:
            learnt_topic_ids: List[str] = incomplete_activities_model.partially_learnt_topic_ids
            return learnt_topic_ids
        else:
            return []

    def test_mark_exploration_as_completed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0])
        state_name = 'state_name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_1])
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_3)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_3])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_3)
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1, self.EXP_ID_3])
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_2)
        self.assertEqual(self._get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1, self.EXP_ID_3])

    def test_mark_collection_as_completed(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_1])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [])
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_3)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_3])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_3)
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_3])
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_2)
        self.assertEqual(self._get_all_completed_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_3])

    def test_mark_story_as_completed(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_1])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [])
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])

    def test_mark_topic_as_learnt(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_1])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [])
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_2)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.user_id), [self.TOPIC_ID_2])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_2)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.user_id), [])

    def test_mark_exploration_as_incomplete(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [])
        state_name = u'state name'
        version = 1
        exp_details: IncompleteExplorationDetailsDict = {'timestamp': datetime.datetime.utcnow(), 'state_name': state_name, 'version': version}
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0])
        self._check_if_exp_details_match(self._get_incomplete_exp_details(self.user_id, self.EXP_ID_0), exp_details)
        state_name = u'new_state_name'
        version = 2
        modified_exp_details: IncompleteExplorationDetailsDict = {'timestamp': datetime.datetime.utcnow(), 'state_name': state_name, 'version': version}
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0])
        self._check_if_exp_details_match(self._get_incomplete_exp_details(self.user_id, self.EXP_ID_0), modified_exp_details)
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_1)
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_playlist_services.mark_exploration_to_be_played_later(self.user_id, self.EXP_ID_3)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_3])
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_3, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_3])
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [])
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_2, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_3])

    def test_mark_collection_as_incomplete(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_1)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0])
        learner_playlist_services.mark_collection_to_be_played_later(self.user_id, self.COL_ID_3)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_3])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_3)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_3])
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_2)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_3])

    def test_record_story_started(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0])

    def test_record_topic_started(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])

    def test_remove_exp_from_incomplete_list(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [])
        state_name: str = 'state name'
        version: int = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])
        learner_progress_services.remove_exp_from_incomplete_list(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_1])
        learner_progress_services.remove_exp_from_incomplete_list(self.user_id, self.EXP_ID_0)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_1])
        learner_progress_services.remove_exp_from_incomplete_list(self.user_id, self.EXP_ID_1)
        self.assertEqual(self._get_all_incomplete_exp_ids(self.user_id), [])

    def test_remove_collection_from_incomplete_list(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])
        learner_progress_services.remove_collection_from_incomplete_list(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_1])
        learner_progress_services.remove_collection_from_incomplete_list(self.user_id, self.COL_ID_0)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_1])
        learner_progress_services.remove_collection_from_incomplete_list(self.user_id, self.COL_ID_1)
        self.assertEqual(self._get_all_incomplete_collection_ids(self.user_id), [])

    def test_remove_story_from_incomplete_list(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_0)
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])
        learner_progress_services.remove_story_from_incomplete_list(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_1])
        learner_progress_services.remove_story_from_incomplete_list(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_1])
        learner_progress_services.remove_story_from_incomplete_list(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_incomplete_story_ids(self.user_id), [])

    def test_remove_topic_from_partially_learnt_list(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_partially_learnt_list(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_partially_learnt_list(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_partially_learnt_list(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_partially_learnt_topic_ids(self.user_id), [])

    def test_remove_story_from_completed_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])
        learner_progress_services.remove_story_from_completed_list(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_1])
        learner_progress_services.remove_story_from_completed_list(self.user_id, self.STORY_ID_0)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [self.STORY_ID_1])
        learner_progress_services.remove_story_from_completed_list(self.user_id, self.STORY_ID_1)
        self.assertEqual(self._get_all_completed_story_ids(self.user_id), [])

    def test_remove_topic_from_learnt_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_learnt_list(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_learnt_list(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_1])
        learner_progress_services.remove_topic_from_learnt_list(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_learnt_topic_ids(self.user_id), [])

    def test_get_all_completed_exp_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(learner_progress_services.get_all_completed_exp_ids(self.user_id), [])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        self.assertEqual(learner_progress_services.get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_1)
        self.assertEqual(learner_progress_services.get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])

    def test_unpublishing_completed_exploration_filters_it_out(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_1)
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_3)
        self.assertEqual(learner_progress_services.get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1, self.EXP_ID_3])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_3)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_3)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_exp_summaries = all_filtered_summaries.completed_exp_summaries
        self.assertEqual(completed_exp_summaries[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(completed_exp_summaries[1].id, '1_fi_arch_sillat_suomi')
        self.assertEqual(len(completed_exp_summaries), 2)

    def test_republishing_completed_exploration_filters_as_complete(self) -> None:
        if False:
            return 10
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        self.assertEqual(learner_progress_services.get_all_completed_exp_ids(self.user_id), [self.EXP_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_0)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_exp_summaries = all_filtered_summaries.completed_exp_summaries
        self.assertEqual(len(completed_exp_summaries), 0)
        self.publish_exploration(self.owner_id, self.EXP_ID_0)
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        public_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(public_exploration.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_exp_summaries = all_filtered_summaries.completed_exp_summaries
        self.assertEqual(completed_exp_summaries[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(len(completed_exp_summaries), 1)

    def test_get_all_completed_collection_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(learner_progress_services.get_all_completed_collection_ids(self.user_id), [])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_progress_services.get_all_completed_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_1)
        self.assertEqual(learner_progress_services.get_all_completed_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])

    def test_get_all_completed_story_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(learner_progress_services.get_all_completed_story_ids(self.user_id), [])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        self.assertEqual(learner_progress_services.get_all_completed_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        self.assertEqual(learner_progress_services.get_all_completed_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])

    def test_get_all_learnt_topic_ids(self) -> None:
        if False:
            return 10
        self.assertEqual(learner_progress_services.get_all_learnt_topic_ids(self.user_id), [])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(learner_progress_services.get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_progress_services.get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])

    def test_unpublishing_completed_collection_filters_it_out(self) -> None:
        if False:
            return 10
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_1)
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_3)
        self.assertEqual(learner_progress_services.get_all_completed_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_3])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_3)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_3)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_collection_summaries = all_filtered_summaries.completed_collection_summaries
        self.assertEqual(completed_collection_summaries[0].id, '0_arch_bridges_in_england')
        self.assertEqual(completed_collection_summaries[1].id, '1_welcome_introduce_oppia')
        self.assertEqual(len(completed_collection_summaries), 2)

    def test_republishing_completed_collection_filters_as_complete(self) -> None:
        if False:
            return 10
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_progress_services.get_all_completed_collection_ids(self.user_id), [self.COL_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_0)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_collection_summaries = all_filtered_summaries.completed_collection_summaries
        self.assertEqual(len(completed_collection_summaries), 0)
        self.publish_collection(self.owner_id, self.COL_ID_0)
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        public_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert public_collection is not None
        self.assertEqual(public_collection.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_collection_summaries = all_filtered_summaries.completed_collection_summaries
        self.assertEqual(completed_collection_summaries[0].id, '0_arch_bridges_in_england')
        self.assertEqual(len(completed_collection_summaries), 1)

    def test_unpublishing_completed_story_filters_it_out(self) -> None:
        if False:
            while True:
                i = 10
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_1, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        self.assertEqual(learner_progress_services.get_all_completed_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])
        topic_services.unpublish_story(self.TOPIC_ID_1, self.STORY_ID_1, self.admin_id)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_story_summaries = all_filtered_summaries.completed_story_summaries
        self.assertEqual(completed_story_summaries[0].id, self.STORY_ID_0)
        self.assertEqual(len(completed_story_summaries), 1)

    def test_unpublishing_learnt_topic_filters_it_out(self) -> None:
        if False:
            i = 10
            return i + 15
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_1, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_progress_services.get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        topic_services.unpublish_topic(self.TOPIC_ID_1, self.admin_id)
        topic_rights = topic_fetchers.get_topic_rights(self.TOPIC_ID_1)
        self.assertEqual(topic_rights.topic_is_published, False)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        learnt_topic_summaries = all_filtered_summaries.learnt_topic_summaries
        self.assertEqual(learnt_topic_summaries[0].id, self.TOPIC_ID_0)
        self.assertEqual(len(learnt_topic_summaries), 1)

    def test_deleting_a_story_filters_it_out_from_completed_list(self) -> None:
        if False:
            return 10
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_1, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        self.assertEqual(learner_progress_services.get_all_completed_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])
        story_services.delete_story(self.admin_id, self.STORY_ID_1)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        completed_story_summaries = all_filtered_summaries.completed_story_summaries
        self.assertEqual(completed_story_summaries[0].id, self.STORY_ID_0)
        self.assertEqual(len(completed_story_summaries), 1)

    def test_deleting_a_topic_filters_it_out_from_learnt_list(self) -> None:
        if False:
            while True:
                i = 10
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_1, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_1)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_progress_services.get_all_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        topic_services.delete_topic(self.admin_id, self.TOPIC_ID_1)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        learnt_topic_summaries = all_filtered_summaries.learnt_topic_summaries
        self.assertEqual(learnt_topic_summaries[0].id, self.TOPIC_ID_0)
        self.assertEqual(len(learnt_topic_summaries), 1)

    def test_get_all_incomplete_exp_ids(self) -> None:
        if False:
            return 10
        self.assertEqual(learner_progress_services.get_all_incomplete_exp_ids(self.user_id), [])
        state_name = 'state name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        self.assertEqual(learner_progress_services.get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0])
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        self.assertEqual(learner_progress_services.get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])

    def test_unpublishing_incomplete_exploration_filters_it_out(self) -> None:
        if False:
            return 10
        state_name = 'state name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_3, state_name, version)
        self.assertEqual(learner_progress_services.get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0, self.EXP_ID_1, self.EXP_ID_3])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_3)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_3)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_exp_summaries = all_filtered_summaries.incomplete_exp_summaries
        self.assertEqual(incomplete_exp_summaries[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(incomplete_exp_summaries[1].id, '1_fi_arch_sillat_suomi')
        self.assertEqual(len(incomplete_exp_summaries), 2)

    def test_republishing_incomplete_exploration_filters_as_incomplete(self) -> None:
        if False:
            return 10
        state_name = 'state name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        self.assertEqual(learner_progress_services.get_all_incomplete_exp_ids(self.user_id), [self.EXP_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_0)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_exp_summaries = all_filtered_summaries.incomplete_exp_summaries
        self.assertEqual(len(incomplete_exp_summaries), 0)
        self.publish_exploration(self.owner_id, self.EXP_ID_0)
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_0, state_name, version)
        public_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(public_exploration.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_exp_summaries = all_filtered_summaries.incomplete_exp_summaries
        self.assertEqual(incomplete_exp_summaries[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(len(incomplete_exp_summaries), 1)

    def test_get_all_incomplete_collection_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(learner_progress_services.get_all_incomplete_collection_ids(self.user_id), [])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_progress_services.get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0])
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        self.assertEqual(learner_progress_services.get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1])

    def test_get_all_incomplete_story_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(learner_progress_services.get_all_incomplete_story_ids(self.user_id), [])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_0)
        self.assertEqual(learner_progress_services.get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0])
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        self.assertEqual(learner_progress_services.get_all_incomplete_story_ids(self.user_id), [self.STORY_ID_0, self.STORY_ID_1])

    def test_get_all_partially_learnt_topic_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id), [])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])

    def test_get_all_and_untracked_topic_ids(self) -> None:
        if False:
            while True:
                i = 10
        self.login(self.CURRICULUM_ADMIN_EMAIL, is_super_admin=True)
        csrf_token = self.get_new_csrf_token()
        new_config_value = [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [self.TOPIC_ID_0, self.TOPIC_ID_1], 'course_details': '', 'topic_list_intro': ''}]
        payload = {'action': 'save_config_properties', 'new_config_property_values': {config_domain.CLASSROOM_PAGES_DATA.name: new_config_value}}
        self.post_json('/adminhandler', payload, csrf_token=csrf_token)
        self.logout()
        self.login(self.USER_EMAIL)
        partially_learnt_topic_ids = learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id)
        learnt_topic_ids = learner_progress_services.get_all_learnt_topic_ids(self.user_id)
        topic_ids_to_learn = learner_goals_services.get_all_topic_ids_to_learn(self.user_id)
        (all_topics, untracked_topics) = learner_progress_services.get_all_and_untracked_topic_ids_for_user(partially_learnt_topic_ids, learnt_topic_ids, topic_ids_to_learn)
        self.assertEqual(len(all_topics), 2)
        self.assertEqual(len(untracked_topics), 2)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        partially_learnt_topic_ids = learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id)
        learnt_topic_ids = learner_progress_services.get_all_learnt_topic_ids(self.user_id)
        topic_ids_to_learn = learner_goals_services.get_all_topic_ids_to_learn(self.user_id)
        (all_topics, untracked_topics) = learner_progress_services.get_all_and_untracked_topic_ids_for_user(partially_learnt_topic_ids, learnt_topic_ids, topic_ids_to_learn)
        self.assertEqual(len(all_topics), 2)
        self.assertEqual(len(untracked_topics), 1)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_1)
        partially_learnt_topic_ids = learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id)
        learnt_topic_ids = learner_progress_services.get_all_learnt_topic_ids(self.user_id)
        topic_ids_to_learn = learner_goals_services.get_all_topic_ids_to_learn(self.user_id)
        (all_topics, untracked_topics) = learner_progress_services.get_all_and_untracked_topic_ids_for_user(partially_learnt_topic_ids, learnt_topic_ids, topic_ids_to_learn)
        self.assertEqual(len(all_topics), 2)
        self.assertEqual(len(untracked_topics), 0)

    def test_unpublishing_incomplete_collection_filters_it_out(self) -> None:
        if False:
            return 10
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_3)
        self.assertEqual(learner_progress_services.get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_3])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_3)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_3)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_collection_summaries = all_filtered_summaries.incomplete_collection_summaries
        self.assertEqual(incomplete_collection_summaries[0].id, '0_arch_bridges_in_england')
        self.assertEqual(incomplete_collection_summaries[1].id, '1_welcome_introduce_oppia')
        self.assertEqual(len(incomplete_collection_summaries), 2)

    def test_republishing_incomplete_collection_filters_as_incomplete(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_progress_services.get_all_incomplete_collection_ids(self.user_id), [self.COL_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_0)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_collection_summaries = all_filtered_summaries.incomplete_collection_summaries
        self.assertEqual(len(incomplete_collection_summaries), 0)
        self.publish_collection(self.owner_id, self.COL_ID_0)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_0)
        public_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert public_collection is not None
        self.assertEqual(public_collection.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        incomplete_collection_summaries = all_filtered_summaries.incomplete_collection_summaries
        self.assertEqual(incomplete_collection_summaries[0].id, '0_arch_bridges_in_england')
        self.assertEqual(len(incomplete_collection_summaries), 1)

    def test_unpublishing_partially_learnt_topic_filters_it_out(self) -> None:
        if False:
            print('Hello World!')
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        topic_services.unpublish_topic(self.TOPIC_ID_1, self.admin_id)
        topic_rights = topic_fetchers.get_topic_rights(self.TOPIC_ID_1)
        self.assertEqual(topic_rights.topic_is_published, False)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        partially_learnt_topic_summaries = all_filtered_summaries.partially_learnt_topic_summaries
        self.assertEqual(partially_learnt_topic_summaries[0].id, self.TOPIC_ID_0)
        self.assertEqual(len(partially_learnt_topic_summaries), 1)

    def test_republishing_partially_learnt_topic_filters_as_incomplete(self) -> None:
        if False:
            print('Hello World!')
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(learner_progress_services.get_all_partially_learnt_topic_ids(self.user_id), [self.TOPIC_ID_0])
        topic_services.unpublish_topic(self.TOPIC_ID_0, self.admin_id)
        topic_rights = topic_fetchers.get_topic_rights(self.TOPIC_ID_0)
        self.assertEqual(topic_rights.topic_is_published, False)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        partially_learnt_topic_summaries = all_filtered_summaries.partially_learnt_topic_summaries
        self.assertEqual(len(partially_learnt_topic_summaries), 0)
        topic_services.publish_topic(self.TOPIC_ID_0, self.admin_id)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_0)
        topic_rights = topic_fetchers.get_topic_rights(self.TOPIC_ID_0)
        self.assertEqual(topic_rights.topic_is_published, True)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        partially_learnt_topic_summaries = all_filtered_summaries.partially_learnt_topic_summaries
        self.assertEqual(partially_learnt_topic_summaries[0].id, self.TOPIC_ID_0)
        self.assertEqual(len(partially_learnt_topic_summaries), 1)

    def test_removes_a_topic_from_topics_to_learn_list_when_topic_is_learnt(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.user_id), [])
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_0)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.user_id), [self.TOPIC_ID_0])
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        topics_to_learn = all_filtered_summaries.topics_to_learn_summaries
        self.assertEqual(len(topics_to_learn), 0)

    def test_unpublishing_topic_filters_it_out_from_topics_to_learn(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_0)
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_1)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.user_id), [self.TOPIC_ID_0, self.TOPIC_ID_1])
        topic_services.unpublish_topic(self.TOPIC_ID_0, self.admin_id)
        user_activity = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        topics_to_learn = all_filtered_summaries.topics_to_learn_summaries
        self.assertEqual(topics_to_learn[0].id, 'topic_1')
        self.assertEqual(len(topics_to_learn), 1)

    def test_unpublishing_exploration_filters_it_out_from_playlist(self) -> None:
        if False:
            print('Hello World!')
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_1)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_0, self.EXP_ID_1])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_1)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_1)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        exploration_playlist = all_filtered_summaries.exploration_playlist_summaries
        self.assertEqual(exploration_playlist[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(len(exploration_playlist), 1)

    def test_republishing_exploration_keeps_it_in_exploration_playlist(self) -> None:
        if False:
            print('Hello World!')
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        self.assertEqual(learner_playlist_services.get_all_exp_ids_in_learner_playlist(self.user_id), [self.EXP_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_exploration(system_user, self.EXP_ID_0)
        private_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(private_exploration.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        exploration_playlist = all_filtered_summaries.exploration_playlist_summaries
        self.assertEqual(len(exploration_playlist), 0)
        self.publish_exploration(self.owner_id, self.EXP_ID_0)
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_0)
        public_exploration = exp_fetchers.get_exploration_summary_by_id(self.EXP_ID_0)
        self.assertEqual(public_exploration.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_exploration_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        exploration_playlist = all_filtered_summaries.exploration_playlist_summaries
        self.assertEqual(exploration_playlist[0].id, '0_en_arch_bridges_in_england')
        self.assertEqual(len(exploration_playlist), 1)

    def test_unpublishing_collection_filters_it_out_from_playlist(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_1)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_0, self.COL_ID_1])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_1)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_1)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        collection_playlist = all_filtered_summaries.collection_playlist_summaries
        self.assertEqual(collection_playlist[0].id, '0_arch_bridges_in_england')
        self.assertEqual(len(collection_playlist), 1)

    def test_republishing_collection_keeps_it_in_collection_playlist(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        self.assertEqual(learner_playlist_services.get_all_collection_ids_in_learner_playlist(self.user_id), [self.COL_ID_0])
        system_user = user_services.get_system_user()
        rights_manager.unpublish_collection(system_user, self.COL_ID_0)
        private_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert private_collection is not None
        self.assertEqual(private_collection.status, constants.ACTIVITY_STATUS_PRIVATE)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        collection_playlist = all_filtered_summaries.collection_playlist_summaries
        self.assertEqual(len(collection_playlist), 0)
        self.publish_collection(self.owner_id, self.COL_ID_0)
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_0)
        public_collection = collection_services.get_collection_summary_by_id(self.COL_ID_0)
        assert public_collection is not None
        self.assertEqual(public_collection.status, constants.ACTIVITY_STATUS_PUBLIC)
        user_activity = learner_progress_services.get_collection_progress(self.user_id)
        all_filtered_summaries = user_activity[0]
        collection_playlist = all_filtered_summaries.collection_playlist_summaries
        self.assertEqual(collection_playlist[0].id, '0_arch_bridges_in_england')
        self.assertEqual(len(collection_playlist), 1)

    def test_get_ids_of_activities_in_learner_dashboard(self) -> None:
        if False:
            return 10
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        state_name = 'state name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_3)
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_3)
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_2)
        activity_ids = learner_progress_services.get_learner_dashboard_activities(self.user_id)
        self.assertEqual(activity_ids.completed_exploration_ids, [self.EXP_ID_0])
        self.assertEqual(activity_ids.completed_collection_ids, [self.COL_ID_0])
        self.assertEqual(activity_ids.completed_story_ids, [self.STORY_ID_0])
        self.assertEqual(activity_ids.learnt_topic_ids, [self.TOPIC_ID_0])
        self.assertEqual(activity_ids.incomplete_exploration_ids, [self.EXP_ID_1])
        self.assertEqual(activity_ids.incomplete_collection_ids, [self.COL_ID_1])
        self.assertEqual(activity_ids.partially_learnt_topic_ids, [self.TOPIC_ID_1])
        self.assertEqual(activity_ids.topic_ids_to_learn, [self.TOPIC_ID_2])
        self.assertEqual(activity_ids.exploration_playlist_ids, [self.EXP_ID_3])
        self.assertEqual(activity_ids.collection_playlist_ids, [self.COL_ID_3])

    def test_get_all_activity_progress(self) -> None:
        if False:
            while True:
                i = 10
        self.login(self.CURRICULUM_ADMIN_EMAIL, is_super_admin=True)
        csrf_token = self.get_new_csrf_token()
        new_config_value = [{'name': 'math', 'url_fragment': 'math', 'topic_ids': [self.TOPIC_ID_3], 'course_details': '', 'topic_list_intro': ''}]
        payload = {'action': 'save_config_properties', 'new_config_property_values': {config_domain.CLASSROOM_PAGES_DATA.name: new_config_value}}
        self.post_json('/adminhandler', payload, csrf_token=csrf_token)
        self.logout()
        learner_progress_services.mark_exploration_as_completed(self.user_id, self.EXP_ID_0)
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        story_services.record_completed_node_in_story_context(self.user_id, self.STORY_ID_0, 'node_1')
        learner_progress_services.mark_story_as_completed(self.user_id, self.STORY_ID_0)
        learner_progress_services.mark_topic_as_learnt(self.user_id, self.TOPIC_ID_0)
        state_name = 'state name'
        version = 1
        learner_progress_services.mark_exploration_as_incomplete(self.user_id, self.EXP_ID_1, state_name, version)
        learner_progress_services.mark_collection_as_incomplete(self.user_id, self.COL_ID_1)
        learner_progress_services.record_story_started(self.user_id, self.STORY_ID_1)
        learner_progress_services.record_topic_started(self.user_id, self.TOPIC_ID_1)
        learner_progress_services.add_exp_to_learner_playlist(self.user_id, self.EXP_ID_3)
        learner_progress_services.add_collection_to_learner_playlist(self.user_id, self.COL_ID_3)
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.user_id, self.TOPIC_ID_2)
        exploration_progress = learner_progress_services.get_exploration_progress(self.user_id)
        collection_progress = learner_progress_services.get_collection_progress(self.user_id)
        topics_and_stories_progress = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        incomplete_exp_summaries = exploration_progress[0].incomplete_exp_summaries
        incomplete_collection_summaries = collection_progress[0].incomplete_collection_summaries
        partially_learnt_topic_summaries = topics_and_stories_progress[0].partially_learnt_topic_summaries
        completed_exp_summaries = exploration_progress[0].completed_exp_summaries
        completed_collection_summaries = collection_progress[0].completed_collection_summaries
        completed_story_summaries = topics_and_stories_progress[0].completed_story_summaries
        learnt_topic_summaries = topics_and_stories_progress[0].learnt_topic_summaries
        topics_to_learn_summaries = topics_and_stories_progress[0].topics_to_learn_summaries
        all_topic_summaries = topics_and_stories_progress[0].all_topic_summaries
        untracked_topic_summaries = topics_and_stories_progress[0].untracked_topic_summaries
        exploration_playlist_summaries = exploration_progress[0].exploration_playlist_summaries
        collection_playlist_summaries = collection_progress[0].collection_playlist_summaries
        self.assertEqual(len(incomplete_exp_summaries), 1)
        self.assertEqual(len(incomplete_collection_summaries), 1)
        self.assertEqual(len(partially_learnt_topic_summaries), 1)
        self.assertEqual(len(completed_exp_summaries), 1)
        self.assertEqual(len(completed_collection_summaries), 1)
        self.assertEqual(len(completed_story_summaries), 1)
        self.assertEqual(len(learnt_topic_summaries), 1)
        self.assertEqual(len(topics_to_learn_summaries), 1)
        self.assertEqual(len(all_topic_summaries), 1)
        self.assertEqual(len(untracked_topic_summaries), 1)
        self.assertEqual(len(exploration_playlist_summaries), 1)
        self.assertEqual(len(collection_playlist_summaries), 1)
        self.assertEqual(incomplete_exp_summaries[0].title, 'Sillat Suomi')
        self.assertEqual(incomplete_collection_summaries[0].title, 'Introduce Oppia')
        self.assertEqual(partially_learnt_topic_summaries[0].name, 'topic 1')
        self.assertEqual(completed_exp_summaries[0].title, 'Bridges in England')
        self.assertEqual(completed_collection_summaries[0].title, 'Bridges')
        self.assertEqual(completed_story_summaries[0].title, 'Title')
        self.assertEqual(learnt_topic_summaries[0].name, 'topic')
        self.assertEqual(topics_to_learn_summaries[0].name, 'topic 2')
        self.assertEqual(untracked_topic_summaries[0].name, 'topic 3')
        self.assertEqual(all_topic_summaries[0].name, 'topic 3')
        self.assertEqual(exploration_playlist_summaries[0].title, 'Welcome Oppia')
        self.assertEqual(collection_playlist_summaries[0].title, 'Welcome Oppia Collection')
        exp_services.delete_exploration(self.owner_id, self.EXP_ID_0)
        exp_services.delete_exploration(self.owner_id, self.EXP_ID_1)
        exp_services.delete_exploration(self.owner_id, self.EXP_ID_3)
        collection_services.update_collection(self.owner_id, self.COL_ID_0, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': self.EXP_ID_2}], 'Add new exploration')
        topic_services.delete_topic(self.owner_id, self.TOPIC_ID_2)
        changelist = [story_domain.StoryChange({'cmd': story_domain.CMD_ADD_STORY_NODE, 'node_id': 'node_2', 'title': 'Title 2'}), story_domain.StoryChange({'cmd': story_domain.CMD_UPDATE_STORY_NODE_PROPERTY, 'property_name': story_domain.STORY_NODE_PROPERTY_EXPLORATION_ID, 'old_value': None, 'new_value': self.EXP_ID_6, 'node_id': 'node_2'})]
        story_services.update_story(self.owner_id, self.STORY_ID_0, changelist, 'Added node.')
        exploration_progress = learner_progress_services.get_exploration_progress(self.user_id)
        collection_progress = learner_progress_services.get_collection_progress(self.user_id)
        topics_and_stories_progress = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        self.assertEqual(len(exploration_progress[0].incomplete_exp_summaries), 0)
        self.assertEqual(exploration_progress[1]['completed_explorations'], 1)
        self.assertEqual(exploration_progress[1]['incomplete_explorations'], 1)
        self.assertEqual(exploration_progress[1]['exploration_playlist'], 1)
        self.assertEqual(topics_and_stories_progress[1]['topics_to_learn'], 1)
        incomplete_collection_summaries = collection_progress[0].incomplete_collection_summaries
        completed_story_summaries = topics_and_stories_progress[0].completed_story_summaries
        partially_learnt_topic_summaries = topics_and_stories_progress[0].partially_learnt_topic_summaries
        self.assertEqual(len(incomplete_collection_summaries), 2)
        self.assertEqual(incomplete_collection_summaries[1].title, 'Bridges')
        learner_progress_services.mark_collection_as_completed(self.user_id, self.COL_ID_0)
        self.assertEqual(len(completed_story_summaries), 0)
        self.assertEqual(len(partially_learnt_topic_summaries), 2)
        self.assertEqual(partially_learnt_topic_summaries[1].name, 'topic')
        collection_services.delete_collection(self.owner_id, self.COL_ID_0)
        collection_services.delete_collection(self.owner_id, self.COL_ID_1)
        collection_services.delete_collection(self.owner_id, self.COL_ID_3)
        topic_services.delete_topic(self.admin_id, self.TOPIC_ID_0)
        collection_progress = learner_progress_services.get_collection_progress(self.user_id)
        topics_and_stories_progress = learner_progress_services.get_topics_and_stories_progress(self.user_id)
        self.assertEqual(collection_progress[1]['completed_collections'], 1)
        self.assertEqual(collection_progress[1]['incomplete_collections'], 1)
        self.assertEqual(collection_progress[1]['collection_playlist'], 1)
        self.assertEqual(topics_and_stories_progress[1]['partially_learnt_topics'], 1)