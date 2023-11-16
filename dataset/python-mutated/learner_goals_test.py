"""Tests for the learner goals."""
from __future__ import annotations
from core import feconf
from core.constants import constants
from core.domain import learner_goals_services
from core.domain import learner_progress_services
from core.domain import topic_domain
from core.domain import topic_services
from core.tests import test_utils

class LearnerGoalsHandlerTests(test_utils.GenericTestBase):
    TOPIC_ID_1 = 'Topic_id_1'
    TOPIC_NAME_1 = 'Topic name 1'
    TOPIC_ID_2 = 'Topic_id_2'
    TOPIC_NAME_2 = 'Topic name 2'
    TOPIC_ID_3 = 'Topic_id_3'
    TOPIC_NAME_3 = 'Topic name 3'
    TOPIC_ID_4 = 'Topic_id_4'
    TOPIC_NAME_4 = 'Topic name 4'
    subtopic_1 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_2 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_3 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')
    subtopic_4 = topic_domain.Subtopic(0, 'Title 1', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-zero')

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
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

    def test_add_topic_to_learner_goal(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.VIEWER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_1), {}, csrf_token=csrf_token)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1])
        self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_2), {}, csrf_token=csrf_token)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        learner_progress_services.mark_topic_as_learnt(self.viewer_id, self.TOPIC_ID_3)
        response = self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_3), {}, csrf_token=csrf_token)
        self.assertEqual(response['belongs_to_learnt_list'], True)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        response = self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, 'InvalidActivityType', self.TOPIC_ID_1), {}, csrf_token=csrf_token, expected_status_int=400)
        self.assertIn("Received InvalidActivityType which is not in the allowed range of choices: ['learntopic']", response['error'])
        for topic_id in range(2, feconf.MAX_CURRENT_GOALS_COUNT + 1):
            self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, 'topic_id_%s' % topic_id), {}, csrf_token=csrf_token)
        response = self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, 'topic_id_%s' % str(feconf.MAX_CURRENT_GOALS_COUNT + 3)), {}, csrf_token=csrf_token)
        self.assertEqual(response['goals_limit_exceeded'], True)
        self.logout()

    def test_remove_topic_from_learner_goals(self) -> None:
        if False:
            print('Hello World!')
        self.login(self.VIEWER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_1)
        learner_progress_services.validate_and_add_topic_to_learn_goal(self.viewer_id, self.TOPIC_ID_2)
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_1, self.TOPIC_ID_2])
        self.delete_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_1))
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [self.TOPIC_ID_2])
        self.delete_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_2))
        self.assertEqual(learner_goals_services.get_all_topic_ids_to_learn(self.viewer_id), [])
        self.post_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, constants.ACTIVITY_TYPE_LEARN_TOPIC, self.TOPIC_ID_1), {}, csrf_token=csrf_token)
        response = self.delete_json('%s/%s/%s' % (feconf.LEARNER_GOALS_DATA_URL, 'InvalidActivityType', self.TOPIC_ID_1), expected_status_int=400)
        self.assertIn("Received InvalidActivityType which is not in the allowed range of choices: ['learntopic']", response['error'])
        self.logout()