"""Tests for learner goals services."""
from __future__ import annotations
from core import feconf
from core.constants import constants
from core.domain import learner_goals_services
from core.domain import learner_progress_services
from core.domain import topic_domain
from core.domain import topic_services
from core.platform import models
from core.tests import test_utils
from typing import Final, List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])
MAX_CURRENT_GOALS_COUNT: Final = feconf.MAX_CURRENT_GOALS_COUNT

class LearnerGoalsTests(test_utils.GenericTestBase):
    """Test the services related to learner goals services."""
    TOPIC_ID_1: Final = 'Topic_id_1'
    TOPIC_NAME_1: Final = 'Topic name 1'
    TOPIC_ID_2: Final = 'Topic_id_2'
    TOPIC_NAME_2: Final = 'Topic name 2'
    TOPIC_ID_3: Final = 'Topic_id_3'
    TOPIC_NAME_3: Final = 'Topic name 3'
    TOPIC_ID_4: Final = 'Topic_id_4'
    TOPIC_NAME_4: Final = 'Topic name 4'
    subtopic_1 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_2 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_3 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_4 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.signup(self.VIEWER_EMAIL, self.VIEWER_USERNAME)
        self.signup(self.CURRICULUM_ADMIN_EMAIL, self.CURRICULUM_ADMIN_USERNAME)
        self.viewer_id = self.get_user_id_from_email(self.VIEWER_EMAIL)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.curriculum_admin_id = self.get_user_id_from_email(self.CURRICULUM_ADMIN_EMAIL)
        self.set_curriculum_admins([self.CURRICULUM_ADMIN_USERNAME])
        self.save_new_topic(self.TOPIC_ID_1, self.owner_id, name=self.TOPIC_NAME_1, url_fragment='topic-one', description='A new topic', canonical_story_ids=[], additional_story_ids=[], uncategorized_skill_ids=[], subtopics=[self.subtopic_1], next_subtopic_id=1)
        topic_services.publish_topic(self.TOPIC_ID_1, self.curriculum_admin_id)
        self.save_new_topic(self.TOPIC_ID_2, self.owner_id, name=self.TOPIC_NAME_2, url_fragment='topic-two', description='A new topic', canonical_story_ids=[], additional_story_ids=[], uncategorized_skill_ids=[], subtopics=[self.subtopic_2], next_subtopic_id=1)
        topic_services.publish_topic(self.TOPIC_ID_2, self.curriculum_admin_id)
        self.save_new_topic(self.TOPIC_ID_3, self.owner_id, name=self.TOPIC_NAME_3, url_fragment='topic-three', description='A new topic', canonical_story_ids=[], additional_story_ids=[], uncategorized_skill_ids=[], subtopics=[self.subtopic_3], next_subtopic_id=1)
        topic_services.publish_topic(self.TOPIC_ID_3, self.curriculum_admin_id)
        self.save_new_topic(self.TOPIC_ID_4, self.owner_id, name=self.TOPIC_NAME_4, url_fragment='topic-four', description='A new topic', canonical_story_ids=[], additional_story_ids=[], uncategorized_skill_ids=[], subtopics=[self.subtopic_4], next_subtopic_id=1)
        topic_services.publish_topic(self.TOPIC_ID_4, self.curriculum_admin_id)

    def _get_all_topic_ids_to_learn(self, user_id: str) -> List[str]:
        if False:
            print('Hello World!')
        'Returns the list of all the topic ids to learn\n        corresponding to the given user id.\n        '
        learner_goals_model = user_models.LearnerGoalsModel.get(user_id, strict=False)
        if learner_goals_model:
            topic_ids: List[str] = learner_goals_model.topic_ids_to_learn
            return topic_ids
        else:
            return []

    def test_single_topic_is_added_correctly_to_learn(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])

    def test_multiple_topics_are_added_correctly_to_learn(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])

    def test_adding_exisiting_topic_is_not_added_again(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        with self.assertRaisesRegex(Exception, 'The topic id Topic_id_1 is already present in the learner goals'):
            learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)

    def test_completed_topic_is_not_added_to_learner_goals(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])
        learner_progress_services.mark_topic_as_learnt(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])

    def test_number_of_topics_cannot_exceed_max(self) -> None:
        if False:
            print('Hello World!')
        topic_ids = ['SAMPLE_TOPIC_ID_%s' % index for index in range(0, MAX_CURRENT_GOALS_COUNT)]
        for topic_id in topic_ids:
            learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, topic_id)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), topic_ids)
        learner_goals_services.mark_topic_to_learn(self.viewer_id, 'SAMPLE_TOPIC_ID_MAX')
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), topic_ids)

    def test_remove_topic_from_learner_goals(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_1)
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        learner_goals_services.remove_topics_from_learn_goal(self.viewer_id, [self.TOPIC_ID_1])
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_2])
        with self.assertRaisesRegex(Exception, 'The topic id Topic_id_1 is not present in LearnerGoalsModel'):
            learner_goals_services.remove_topics_from_learn_goal(self.viewer_id, [self.TOPIC_ID_1])
        learner_goals_services.remove_topics_from_learn_goal(self.viewer_id, [self.TOPIC_ID_2])
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])

    def test_get_all_topic_ids_in_learn(self) -> None:
        if False:
            return 10
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_1)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])

    def test_remove_topics_from_learn_goal_executed_correctly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_1)
        learner_goals_services.mark_topic_to_learn(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        learner_goals_services.remove_topics_from_learn_goal(self.viewer_id, [self.TOPIC_ID_1])
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_2])
        learner_goals = learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id)
        self.assertNotIn(self.TOPIC_ID_1, learner_goals)
        self.assertIn(self.TOPIC_ID_2, learner_goals)
        learner_goals_services.remove_topics_from_learn_goal(self.viewer_id, [self.TOPIC_ID_2])
        self.assertEqual(self._get_all_topic_ids_to_learn(self.viewer_id), [])
        learner_goals = learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id)
        self.assertNotIn(self.TOPIC_ID_1, learner_goals)
        self.assertNotIn(self.TOPIC_ID_2, learner_goals)

    def test_remove_topics_when_learner_goals_model_does_not_exist(self) -> None:
        if False:
            return 10
        non_existent_user_id = 'non_existent_user_id'
        self.assertIsNone(user_models.LearnerGoalsModel.get(non_existent_user_id, strict=False))
        learner_goals_services.remove_topics_from_learn_goal(non_existent_user_id, [self.TOPIC_ID_1])
        learner_goals = learner_goals_services.get_all_topic_ids_to_learn(non_existent_user_id)
        self.assertEqual(learner_goals, [])