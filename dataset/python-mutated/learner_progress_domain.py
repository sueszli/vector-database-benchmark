"""Domain objects for learner progress."""
from __future__ import annotations
from core.domain import collection_domain
from core.domain import exp_domain
from core.domain import story_domain
from core.domain import topic_domain
from typing import Dict, List

class LearnerProgressInTopicsAndStories:
    """Domain object for the progress of the learner in topics and stories."""

    def __init__(self, partially_learnt_topic_summaries: List[topic_domain.TopicSummary], completed_story_summaries: List[story_domain.StorySummary], learnt_topic_summaries: List[topic_domain.TopicSummary], topics_to_learn_summaries: List[topic_domain.TopicSummary], all_topic_summaries: List[topic_domain.TopicSummary], untracked_topic_summaries: List[topic_domain.TopicSummary], completed_to_incomplete_story_titles: List[str], learnt_to_partially_learnt_topic_titles: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a LearnerProgress domain object.\n\n        Args:\n            partially_learnt_topic_summaries: list(TopicSummary). The\n                summaries of the topics partially learnt by the\n                learner.\n            completed_story_summaries: list(StorySummary). The\n                summaries of the stories completed by the learner.\n            learnt_topic_summaries: list(TopicSummary). The\n                summaries of the topics learnt by the learner.\n            topics_to_learn_summaries: list(TopicSummary). The\n                summaries of the topics to learn.\n            all_topic_summaries: list(TopicSummary). The summaries of the topics\n                in the edit goals.\n            untracked_topic_summaries: list(TopicSummary). The summaries of the\n                topics not tracked for the user.\n            completed_to_incomplete_story_titles: list(str).\n                The titles of summaries corresponding to those stories which\n                have been moved to the in progress section on account of new\n                nodes being added to them.\n            learnt_to_partially_learnt_topic_titles: list(str).\n                The titles of summaries corresponding to those topics which have\n                been moved to the in progress section on account of new\n                stories being added to them.\n        '
        self.partially_learnt_topic_summaries = partially_learnt_topic_summaries
        self.completed_story_summaries = completed_story_summaries
        self.learnt_topic_summaries = learnt_topic_summaries
        self.topics_to_learn_summaries = topics_to_learn_summaries
        self.all_topic_summaries = all_topic_summaries
        self.untracked_topic_summaries = untracked_topic_summaries
        self.completed_to_incomplete_stories = completed_to_incomplete_story_titles
        self.learnt_to_partially_learnt_topics = learnt_to_partially_learnt_topic_titles

class LearnerProgressInCollections:
    """Domain object for the progress of the learner in collections."""

    def __init__(self, incomplete_collection_summaries: List[collection_domain.CollectionSummary], completed_collection_summaries: List[collection_domain.CollectionSummary], collection_playlist: List[collection_domain.CollectionSummary], completed_to_incomplete_collection_titles: List[str]) -> None:
        if False:
            print('Hello World!')
        'Constructs a LearnerProgress domain object.\n\n        Args:\n            incomplete_collection_summaries: list(CollectionSummary). The\n                summaries of the collections partially completed by the\n                learner.\n            completed_collection_summaries: list(CollectionSummary). The\n                summaries of the collections partially completed by the learner.\n            collection_playlist: list(CollectionSummary). The summaries of the\n                collections in the learner playlist.\n            completed_to_incomplete_collection_titles: list(CollectionSummary).\n                The summaries corresponding to those collections which have\n                been moved to the in progress section on account of new\n                explorations being added to them.\n        '
        self.incomplete_collection_summaries = incomplete_collection_summaries
        self.completed_collection_summaries = completed_collection_summaries
        self.collection_playlist_summaries = collection_playlist
        self.completed_to_incomplete_collections = completed_to_incomplete_collection_titles

class LearnerProgressInExplorations:
    """Domain object for the progress of the learner in explorations."""

    def __init__(self, incomplete_exp_summaries: List[exp_domain.ExplorationSummary], completed_exp_summaries: List[exp_domain.ExplorationSummary], exploration_playlist: List[exp_domain.ExplorationSummary]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a LearnerProgress domain object.\n\n        Args:\n            incomplete_exp_summaries: list(ExplorationSummary). The summaries\n                of the explorations partially completed by the learner.\n            completed_exp_summaries: list(ExplorationSummary). The summaries of\n                the explorations partially completed by the learner.\n            exploration_playlist: list(ExplorationSummary). The summaries of the\n                explorations in the learner playlist.\n        '
        self.incomplete_exp_summaries = incomplete_exp_summaries
        self.completed_exp_summaries = completed_exp_summaries
        self.exploration_playlist_summaries = exploration_playlist

class ActivityIdsInLearnerDashboard:
    """Domain object for ids of the activities completed, currently being
    completed, in the playlist or goals of the user.
    """

    def __init__(self, completed_exploration_ids: List[str], completed_collection_ids: List[str], completed_story_ids: List[str], learnt_topic_ids: List[str], incomplete_exploration_ids: List[str], incomplete_collection_ids: List[str], partially_learnt_topic_ids: List[str], topic_ids_to_learn: List[str], all_topic_ids: List[str], untracked_topic_ids: List[str], exploration_playlist_ids: List[str], collection_playlist_ids: List[str]) -> None:
        if False:
            print('Hello World!')
        'Constructs a ActivityIdsInLearnerDashboard domain object.\n\n        Args:\n            completed_exploration_ids: list(str). The ids of the explorations\n                completed by the user.\n            completed_collection_ids: list(str). The ids of the collections\n                completed by the user.\n            completed_story_ids: list(str). The ids of the stories\n                completed by the user.\n            learnt_topic_ids: list(str). The ids of the topics\n                learnt by the user.\n            incomplete_exploration_ids: list(str). The ids of the explorations\n                currently in progress.\n            incomplete_collection_ids: list(str). The ids of the collections\n                currently in progress.\n            partially_learnt_topic_ids: list(str). The ids of the topics\n                partially learnt.\n            topic_ids_to_learn: list(str). The ids of the topics to learn.\n            all_topic_ids: list(str). The ids of the all the topics.\n            untracked_topic_ids: list(str). The ids of the untracked topics.\n            exploration_playlist_ids: list(str). The ids of the explorations\n                in the playlist of the user.\n            collection_playlist_ids: list(str). The ids of the collections\n                in the playlist of the user.\n        '
        self.completed_exploration_ids = completed_exploration_ids
        self.completed_collection_ids = completed_collection_ids
        self.completed_story_ids = completed_story_ids
        self.learnt_topic_ids = learnt_topic_ids
        self.incomplete_exploration_ids = incomplete_exploration_ids
        self.incomplete_collection_ids = incomplete_collection_ids
        self.partially_learnt_topic_ids = partially_learnt_topic_ids
        self.topic_ids_to_learn = topic_ids_to_learn
        self.all_topic_ids = all_topic_ids
        self.untracked_topic_ids = untracked_topic_ids
        self.exploration_playlist_ids = exploration_playlist_ids
        self.collection_playlist_ids = collection_playlist_ids

    def to_dict(self) -> Dict[str, List[str]]:
        if False:
            print('Hello World!')
        "Return dictionary representation of ActivityIdsInLearnerDashboard.\n\n        Returns:\n            dict. The keys of the dict are:\n                'completed_exploration_ids': list(str). The ids of the\n                    explorations that are completed.\n                'completed_collection_ids': list(str). The ids of the\n                    collections that are completed.\n                'completed_story_ids': list(str). The ids of the\n                    stories that are completed.\n                'learnt_topic_ids': list(str). The ids of the\n                    topics that are learnt.\n                'incomplete_exploration_ids': list(str). The ids of the\n                    explorations that are incomplete.\n                'incomplete_collection_ids': list(str). The ids of the\n                    collections that are incomplete.\n                'partially_learnt_topic_ids': list(str). The ids of the\n                    topics that are partially learnt.\n                'topic_ids_to_learn': list(str). The ids of the topics\n                    to learn.\n                'all_topic_ids': list(str). The ids of all the topics.\n                'untracked_topic_ids': list(str). The ids of the untracked\n                    topics.\n                'exploration_playlist_ids': list(str). The ids of the\n                    explorations that are in the playlist\n                'collection_playlist_ids': list(str). The ids of the\n                    collections that are in the playlist.\n        "
        return {'completed_exploration_ids': self.completed_exploration_ids, 'completed_collection_ids': self.completed_collection_ids, 'completed_story_ids': self.completed_story_ids, 'learnt_topic_ids': self.learnt_topic_ids, 'incomplete_exploration_ids': self.incomplete_exploration_ids, 'incomplete_collection_ids': self.incomplete_collection_ids, 'partially_learnt_topic_ids': self.partially_learnt_topic_ids, 'topic_ids_to_learn': self.topic_ids_to_learn, 'all_topic_ids': self.all_topic_ids, 'untracked_topic_ids': self.untracked_topic_ids, 'exploration_playlist_ids': self.exploration_playlist_ids, 'collection_playlist_ids': self.collection_playlist_ids}