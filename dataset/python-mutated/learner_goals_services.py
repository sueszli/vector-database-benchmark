"""Services for the learner goals feature of the learner dashboard."""
from __future__ import annotations
from core import feconf
from core.domain import user_domain
from core.platform import models
from typing import List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

def get_learner_goals_from_model(learner_goals_model: user_models.LearnerGoalsModel) -> user_domain.LearnerGoals:
    if False:
        for i in range(10):
            print('nop')
    'Returns the learner goals domain object given the learner goals\n    model loaded from the datastore.\n\n    Args:\n        learner_goals_model: LearnerGoalsModel. The\n            learner goals model from the datastore.\n\n    Returns:\n        LearnerGoals. The learner goals domain object corresponding to the\n        given model.\n    '
    return user_domain.LearnerGoals(learner_goals_model.id, learner_goals_model.topic_ids_to_learn, learner_goals_model.topic_ids_to_master)

def save_learner_goals(learner_goals: user_domain.LearnerGoals) -> None:
    if False:
        print('Hello World!')
    'Save a learner goals domain object as an LearnerGoalsModel entity\n    in the datastore.\n\n    Args:\n        learner_goals: LearnerGoals. The learner goals domain object to\n            be saved in the datastore.\n    '
    learner_goals_dict = learner_goals.to_dict()
    learner_goals_model = user_models.LearnerGoalsModel.get(learner_goals.id, strict=False)
    if learner_goals_model is not None:
        learner_goals_model.populate(**learner_goals_dict)
        learner_goals_model.update_timestamps()
        learner_goals_model.put()
    else:
        user_models.LearnerGoalsModel(id=learner_goals.id, **learner_goals_dict).put()

def mark_topic_to_learn(user_id: str, topic_id: str) -> bool:
    if False:
        print('Hello World!')
    'Adds the topic id to the learner goals of the user. If the count exceeds\n    feconf.MAX_CURRENT_GOALS_COUNT, the topic is not added.\n\n    Args:\n        user_id: str. The id of the user.\n        topic_id: str. The id of the topic to be added to the\n            learner goals.\n\n    Returns:\n        bool. The boolean indicates whether the learner goals limit\n        of the user has been exceeded.\n\n    Raises:\n        Exception. Given topic is already present.\n    '
    learner_goals_model = user_models.LearnerGoalsModel.get(user_id, strict=False)
    if not learner_goals_model:
        learner_goals_model = user_models.LearnerGoalsModel(id=user_id)
    learner_goals = get_learner_goals_from_model(learner_goals_model)
    goals_limit_exceeded = False
    topic_ids_count = len(learner_goals.topic_ids_to_learn)
    if topic_id not in learner_goals.topic_ids_to_learn:
        if topic_ids_count < feconf.MAX_CURRENT_GOALS_COUNT:
            learner_goals.add_topic_id_to_learn(topic_id)
        else:
            goals_limit_exceeded = True
        save_learner_goals(learner_goals)
        return goals_limit_exceeded
    else:
        raise Exception('The topic id %s is already present in the learner goals' % topic_id)

def remove_topics_from_learn_goal(user_id: str, topic_ids_to_remove: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Removes topics from the learner goals of the user (if present).\n\n    Args:\n        user_id: str. The id of the user.\n        topic_ids_to_remove: list(str). The ids of the topics to be removed.\n\n    Raises:\n        Exception. Given topic does not exist.\n    '
    learner_goals_model = user_models.LearnerGoalsModel.get(user_id, strict=False)
    if learner_goals_model:
        learner_goals = get_learner_goals_from_model(learner_goals_model)
        for topic_id in topic_ids_to_remove:
            if topic_id in learner_goals.topic_ids_to_learn:
                learner_goals.remove_topic_id_from_learn(topic_id)
            else:
                raise Exception('The topic id %s is not present in LearnerGoalsModel' % topic_id)
        save_learner_goals(learner_goals)

def get_all_topic_ids_to_learn(user_id: str) -> List[str]:
    if False:
        return 10
    'Returns a list with the ids of all the topics that are in the\n    goals of the user.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(str). A list of the ids of the topics that are in the\n        learner goals of the user.\n    '
    learner_goals_model = user_models.LearnerGoalsModel.get(user_id, strict=False)
    if learner_goals_model:
        learner_goals = get_learner_goals_from_model(learner_goals_model)
        return learner_goals.topic_ids_to_learn
    return []