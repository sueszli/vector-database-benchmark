"""Tests for learner group domain objects."""
from __future__ import annotations
from core.domain import learner_group_domain
from core.tests import test_utils

class LearnerGroupTest(test_utils.GenericTestBase):
    """Tests for LearnerGroup domain object."""
    VALID_LEARNER_GROUP = learner_group_domain.LearnerGroup('3232', 'title', 'description', ['user_1'], ['user_2', 'user_3', 'user_4'], ['user_5', 'user_6'], ['subtopic_1', 'subtopic_2'], ['story_1', 'story_2'])

    def test_initialization(self) -> None:
        if False:
            i = 10
            return i + 15
        learner_group = self.VALID_LEARNER_GROUP
        expected_learner_group_dict = {'group_id': '3232', 'title': 'title', 'description': 'description', 'facilitator_user_ids': ['user_1'], 'learner_user_ids': ['user_2', 'user_3', 'user_4'], 'invited_learner_user_ids': ['user_5', 'user_6'], 'subtopic_page_ids': ['subtopic_1', 'subtopic_2'], 'story_ids': ['story_1', 'story_2']}
        self.assertEqual(learner_group.group_id, '3232')
        self.assertEqual(learner_group.title, 'title')
        self.assertEqual(learner_group.description, 'description')
        self.assertEqual(learner_group.facilitator_user_ids, ['user_1'])
        self.assertEqual(learner_group.learner_user_ids, ['user_2', 'user_3', 'user_4'])
        self.assertEqual(learner_group.invited_learner_user_ids, ['user_5', 'user_6'])
        self.assertEqual(learner_group.subtopic_page_ids, ['subtopic_1', 'subtopic_2'])
        self.assertEqual(learner_group.story_ids, ['story_1', 'story_2'])
        self.assertEqual(learner_group.to_dict(), expected_learner_group_dict)

    def test_to_dict(self) -> None:
        if False:
            print('Hello World!')
        learner_group = self.VALID_LEARNER_GROUP
        expected_learner_group_dict = {'group_id': '3232', 'title': 'title', 'description': 'description', 'facilitator_user_ids': ['user_1'], 'learner_user_ids': ['user_2', 'user_3', 'user_4'], 'invited_learner_user_ids': ['user_5', 'user_6'], 'subtopic_page_ids': ['subtopic_1', 'subtopic_2'], 'story_ids': ['story_1', 'story_2']}
        self.assertEqual(learner_group.to_dict(), expected_learner_group_dict)

    def test_validation(self) -> None:
        if False:
            print('Hello World!')
        self._assert_validation_error(learner_group_domain.LearnerGroup('3232', 'title', 'description', [], ['user_2', 'user_3', 'user_4'], ['user_5', 'user_6'], ['subtopic_1', 'subtopic_2'], ['story_1', 'story_2']), 'Expected learner group to have at least one facilitator.')
        self._assert_validation_error(learner_group_domain.LearnerGroup('3232', 'title', 'description', ['user_1'], ['user_2', 'user_3', 'user_5'], ['user_5', 'user_6'], ['subtopic_1', 'subtopic_2'], ['story_1', 'story_2']), 'Learner group learner cannot be invited to join the group.')
        self._assert_validation_error(learner_group_domain.LearnerGroup('3232', 'title', 'description', ['user_1'], ['user_1', 'user_3', 'user_4'], ['user_5', 'user_6'], ['subtopic_1', 'subtopic_2'], ['story_1', 'story_2']), 'Learner group facilitator cannot be a learner of the group.')
        self._assert_validation_error(learner_group_domain.LearnerGroup('3232', 'title', 'description', ['user_1'], ['user_2', 'user_3', 'user_4'], ['user_1', 'user_6'], ['subtopic_1', 'subtopic_2'], ['story_1', 'story_2']), 'Learner group facilitator cannot be invited to join the group.')
        learner_group = self.VALID_LEARNER_GROUP
        learner_group.validate()