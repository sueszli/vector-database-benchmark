"""Domain objects for Learner Groups."""
from __future__ import annotations
from core import utils
from core.domain import story_domain
from core.domain import subtopic_page_domain
from typing import List, TypedDict

class LearnerGroupDict(TypedDict):
    """Dictionary for LearnerGroup domain object."""
    group_id: str
    title: str
    description: str
    facilitator_user_ids: List[str]
    learner_user_ids: List[str]
    invited_learner_user_ids: List[str]
    subtopic_page_ids: List[str]
    story_ids: List[str]

class LearnerGroup:
    """Domain object for learner group."""

    def __init__(self, group_id: str, title: str, description: str, facilitator_user_ids: List[str], learner_user_ids: List[str], invited_learner_user_ids: List[str], subtopic_page_ids: List[str], story_ids: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a LearnerGroup domain object.\n\n        Attributes:\n            group_id: str. The unique ID of the learner group.\n            title: str. The title of the learner group.\n            description: str. The description of the learner group.\n            facilitator_user_ids: List[str]. The list of user ids of\n                facilitators of the learner group.\n            learner_user_ids: List[str]. The list of user ids of learners\n                of the learner group.\n            invited_learner_user_ids: List[str]. The list of user ids of the\n                users invited to join the learner group as a learner.\n            subtopic_page_ids: List[str]. The list of subtopic page ids that\n                are part of the learner group syllabus. A subtopic page id is\n                depicted as topicId:subtopicId string.\n            story_ids: List[str]. The list of story ids of the learner group.\n        '
        self.group_id = group_id
        self.title = title
        self.description = description
        self.facilitator_user_ids = facilitator_user_ids
        self.learner_user_ids = learner_user_ids
        self.invited_learner_user_ids = invited_learner_user_ids
        self.subtopic_page_ids = subtopic_page_ids
        self.story_ids = story_ids

    def to_dict(self) -> LearnerGroupDict:
        if False:
            while True:
                i = 10
        'Convert the LearnerGroup domain instance into a dictionary\n        form with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the LearnerGroup class\n            information in a dictionary form.\n        '
        return {'group_id': self.group_id, 'title': self.title, 'description': self.description, 'facilitator_user_ids': self.facilitator_user_ids, 'learner_user_ids': self.learner_user_ids, 'invited_learner_user_ids': self.invited_learner_user_ids, 'subtopic_page_ids': self.subtopic_page_ids, 'story_ids': self.story_ids}

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the LearnerGroup domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the LearnerGroup\n                are invalid.\n        '
        if len(self.facilitator_user_ids) < 1:
            raise utils.ValidationError('Expected learner group to have at least one facilitator.')
        invited_learner_set = set(self.invited_learner_user_ids)
        learner_set = set(self.learner_user_ids)
        if len(invited_learner_set.intersection(learner_set)) > 0:
            raise utils.ValidationError('Learner group learner cannot be invited to join the group.')
        facilitator_set = set(self.facilitator_user_ids)
        if len(facilitator_set.intersection(learner_set)) > 0:
            raise utils.ValidationError('Learner group facilitator cannot be a learner of the group.')
        if len(facilitator_set.intersection(invited_learner_set)) > 0:
            raise utils.ValidationError('Learner group facilitator cannot be invited to join the group.')

class LearnerGroupSyllabusDict(TypedDict):
    """Dictionary reperesentation of learner group syllabus."""
    story_summary_dicts: List[story_domain.LearnerGroupSyllabusStorySummaryDict]
    subtopic_summary_dicts: List[subtopic_page_domain.SubtopicPageSummaryDict]