"""Services for managing subscriptions."""
from __future__ import annotations
from core.platform import models
from typing import List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

def subscribe_to_thread(user_id: str, feedback_thread_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Subscribes a user to a feedback thread.\n\n    WARNING: Callers of this function should ensure that the user_id and\n    feedback_thread_id are valid.\n\n    Args:\n        user_id: str. The user ID of the new subscriber.\n        feedback_thread_id: str. The ID of the feedback thread.\n    '
    subscribe_to_threads(user_id, [feedback_thread_id])

def subscribe_to_threads(user_id: str, feedback_thread_ids: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Subscribes a user to feedback threads.\n\n    WARNING: Callers of this function should ensure that the user_id and\n    the feedback_thread_ids are valid.\n\n    Args:\n        user_id: str. The user ID of the new subscriber.\n        feedback_thread_ids: list(str). The IDs of the feedback threads.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if not subscriptions_model:
        subscriptions_model = user_models.UserSubscriptionsModel(id=user_id)
    current_feedback_thread_ids_set = set(subscriptions_model.general_feedback_thread_ids)
    feedback_thread_ids_to_add_to_subscriptions_model = list(set(feedback_thread_ids).difference(current_feedback_thread_ids_set))
    subscriptions_model.general_feedback_thread_ids.extend(feedback_thread_ids_to_add_to_subscriptions_model)
    subscriptions_model.update_timestamps()
    subscriptions_model.put()

def subscribe_to_exploration(user_id: str, exploration_id: str) -> None:
    if False:
        while True:
            i = 10
    'Subscribes a user to an exploration (and, therefore, indirectly to all\n    feedback threads for that exploration).\n\n    WARNING: Callers of this function should ensure that the user_id and\n    exploration_id are valid.\n\n    Args:\n        user_id: str. The user ID of the new subscriber.\n        exploration_id: str. The exploration ID.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if not subscriptions_model:
        subscriptions_model = user_models.UserSubscriptionsModel(id=user_id)
    if exploration_id not in subscriptions_model.exploration_ids:
        subscriptions_model.exploration_ids.append(exploration_id)
        subscriptions_model.update_timestamps()
        subscriptions_model.put()

def subscribe_to_creator(user_id: str, creator_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Subscribes a user (learner) to a creator.\n\n    WARNING: Callers of this function should ensure that the user_id and\n    creator_id are valid.\n\n    Args:\n        user_id: str. The user ID of the new subscriber.\n        creator_id: str. The user ID of the creator.\n\n    Raises:\n        Exception. The user ID of the new subscriber is same as the\n            user ID of the creator.\n    '
    if user_id == creator_id:
        raise Exception('User %s is not allowed to self subscribe.' % user_id)
    subscribers_model_creator = user_models.UserSubscribersModel.get(creator_id, strict=False)
    subscriptions_model_user = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if not subscribers_model_creator:
        subscribers_model_creator = user_models.UserSubscribersModel(id=creator_id)
    if not subscriptions_model_user:
        subscriptions_model_user = user_models.UserSubscriptionsModel(id=user_id)
    if user_id not in subscribers_model_creator.subscriber_ids:
        subscribers_model_creator.subscriber_ids.append(user_id)
        subscriptions_model_user.creator_ids.append(creator_id)
        subscribers_model_creator.update_timestamps()
        subscribers_model_creator.put()
        subscriptions_model_user.update_timestamps()
        subscriptions_model_user.put()

def unsubscribe_from_creator(user_id: str, creator_id: str) -> None:
    if False:
        return 10
    'Unsubscribe a user from a creator.\n\n    WARNING: Callers of this function should ensure that the user_id and\n    creator_id are valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n        creator_id: str. The user ID of the creator.\n    '
    subscribers_model_creator = user_models.UserSubscribersModel.get(creator_id)
    subscriptions_model_user = user_models.UserSubscriptionsModel.get(user_id)
    if user_id in subscribers_model_creator.subscriber_ids:
        subscribers_model_creator.subscriber_ids.remove(user_id)
        subscriptions_model_user.creator_ids.remove(creator_id)
        subscribers_model_creator.update_timestamps()
        subscribers_model_creator.put()
        subscriptions_model_user.update_timestamps()
        subscriptions_model_user.put()

def get_all_threads_subscribed_to(user_id: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Returns a list with ids of all the feedback and suggestion threads to\n    which the user is subscribed.\n\n    WARNING: Callers of this function should ensure that the user_id is valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n\n    Returns:\n        list(str). IDs of all the feedback and suggestion threads to\n        which the user is subscribed.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if subscriptions_model:
        feedback_thread_ids: List[str] = subscriptions_model.general_feedback_thread_ids
        return feedback_thread_ids
    else:
        return []

def get_all_creators_subscribed_to(user_id: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Returns a list with ids of all the creators to which this learner has\n    subscribed.\n\n    WARNING: Callers of this function should ensure that the user_id is valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n\n    Returns:\n        list(str). IDs of all the creators to which this learner has\n        subscribed.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if subscriptions_model:
        creator_ids: List[str] = subscriptions_model.creator_ids
        return creator_ids
    else:
        return []

def get_all_subscribers_of_creator(user_id: str) -> List[str]:
    if False:
        return 10
    'Returns a list with ids of all users who have subscribed to this\n    creator.\n\n    WARNING: Callers of this function should ensure that the user_id is valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n\n    Returns:\n        list(str). IDs of all users who have subscribed to this creator.\n    '
    subscribers_model = user_models.UserSubscribersModel.get(user_id, strict=False)
    if subscribers_model:
        subscriber_ids: List[str] = subscribers_model.subscriber_ids
        return subscriber_ids
    else:
        return []

def get_exploration_ids_subscribed_to(user_id: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Returns a list with ids of all explorations that the given user\n    subscribes to.\n\n    WARNING: Callers of this function should ensure that the user_id is valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n\n    Returns:\n        list(str). IDs of all explorations that the given user\n        subscribes to.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if subscriptions_model:
        exploration_ids: List[str] = subscriptions_model.exploration_ids
        return exploration_ids
    else:
        return []

def subscribe_to_collection(user_id: str, collection_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Subscribes a user to a collection.\n\n    WARNING: Callers of this function should ensure that the user_id and\n    collection_id are valid.\n\n    Args:\n        user_id: str. The user ID of the new subscriber.\n        collection_id: str. The collection ID.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if not subscriptions_model:
        subscriptions_model = user_models.UserSubscriptionsModel(id=user_id)
    if collection_id not in subscriptions_model.collection_ids:
        subscriptions_model.collection_ids.append(collection_id)
        subscriptions_model.update_timestamps()
        subscriptions_model.put()

def get_collection_ids_subscribed_to(user_id: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Returns a list with ids of all collections that the given user\n    subscribes to.\n\n    WARNING: Callers of this function should ensure that the user_id is valid.\n\n    Args:\n        user_id: str. The user ID of the subscriber.\n\n    Returns:\n        list(str). IDs of all collections that the given user\n        subscribes to.\n    '
    subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
    if subscriptions_model:
        collection_ids: List[str] = subscriptions_model.collection_ids
        return collection_ids
    else:
        return []