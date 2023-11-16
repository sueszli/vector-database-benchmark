"""Getter commands for learner group models."""
from __future__ import annotations
from core.domain import learner_group_domain
from core.domain import learner_group_services
from core.platform import models
from typing import List, Literal, Optional, Sequence, overload
MYPY = False
if MYPY:
    from mypy_imports import learner_group_models
    from mypy_imports import user_models
(learner_group_models, user_models) = models.Registry.import_models([models.Names.LEARNER_GROUP, models.Names.USER])

def get_new_learner_group_id() -> str:
    if False:
        while True:
            i = 10
    'Returns a new learner group id.\n\n    Returns:\n        str. A new learner group id.\n    '
    return learner_group_models.LearnerGroupModel.get_new_id()

@overload
def get_learner_group_by_id(group_id: str, *, strict: Literal[True]) -> learner_group_domain.LearnerGroup:
    if False:
        print('Hello World!')
    ...

@overload
def get_learner_group_by_id(group_id: str) -> Optional[learner_group_domain.LearnerGroup]:
    if False:
        print('Hello World!')
    ...

@overload
def get_learner_group_by_id(group_id: str, *, strict: Literal[False]) -> Optional[learner_group_domain.LearnerGroup]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_learner_group_by_id(group_id: str, strict: bool) -> Optional[learner_group_domain.LearnerGroup]:
    if False:
        return 10
    ...

def get_learner_group_by_id(group_id: str, strict: bool=False) -> Optional[learner_group_domain.LearnerGroup]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the learner group domain object given the learner group id.\n\n    Args:\n        group_id: str. The id of the learner group.\n        strict: bool. Whether to fail noisily if no LearnerGroupModel with the\n            given group_id exists in the datastore.\n\n    Returns:\n        LearnerGroup or None. The learner group domain object corresponding to\n        the given id or None if no learner group exists for the given group id.\n\n    Raises:\n        Exception. No LearnerGroupModel found for the given group_id.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=False)
    if not learner_group_model:
        if strict:
            raise Exception('No LearnerGroupModel found for the given group_id: %s' % group_id)
        return None
    return learner_group_services.get_learner_group_from_model(learner_group_model)

def get_learner_groups_of_facilitator(user_id: str) -> List[learner_group_domain.LearnerGroup]:
    if False:
        while True:
            i = 10
    'Returns a list of learner groups of the given facilitator.\n\n    Args:\n        user_id: str. The id of the facilitator.\n\n    Returns:\n        list(LearnerGroup). A list of learner groups of the given facilitator.\n    '
    learner_grp_models = learner_group_models.LearnerGroupModel.get_by_facilitator_id(user_id)
    if not learner_grp_models:
        return []
    return [learner_group_services.get_learner_group_from_model(model) for model in learner_grp_models]

@overload
def get_learner_group_models_by_ids(user_ids: List[str], *, strict: Literal[True]) -> List[user_models.LearnerGroupsUserModel]:
    if False:
        return 10
    ...

@overload
def get_learner_group_models_by_ids(user_ids: List[str]) -> List[Optional[user_models.LearnerGroupsUserModel]]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_learner_group_models_by_ids(user_ids: List[str], *, strict: Literal[False]) -> List[Optional[user_models.LearnerGroupsUserModel]]:
    if False:
        i = 10
        return i + 15
    ...

def get_learner_group_models_by_ids(user_ids: List[str], strict: bool=False) -> Sequence[Optional[user_models.LearnerGroupsUserModel]]:
    if False:
        i = 10
        return i + 15
    'Returns a list of learner_groups_user models matching the IDs provided.\n\n    Args:\n        user_ids: list(str). The user ids of the learners of the group.\n        strict: bool. Whether to fail noisily if no LearnerGroupsUserModel\n            exists with a given ID exists in the datastore.\n\n    Returns:\n        list(LearnerGroupsUserModel|None). The list of learner_groups_user\n        models corresponding to given ids.  If a LearnerGroupsUserModel does\n        not exist, the corresponding returned list element is None.\n\n    Raises:\n        Exception. No LearnerGroupsUserModel exists for the given user_id.\n    '
    learner_group_user_models = user_models.LearnerGroupsUserModel.get_multi(user_ids)
    if strict:
        for (index, learner_group_user_model) in enumerate(learner_group_user_models):
            if learner_group_user_model is None:
                raise Exception('No LearnerGroupsUserModel exists for the user_id: %s' % user_ids[index])
    return learner_group_user_models

def can_multi_learners_share_progress(user_ids: List[str], group_id: str) -> List[bool]:
    if False:
        print('Hello World!')
    'Returns the progress sharing permissions of the given users in the given\n    group.\n\n    Args:\n        user_ids: list(str). The user ids of the learners of the group.\n        group_id: str. The id of the learner group.\n\n    Returns:\n        list(bool). True if a user has progress sharing permission of the\n        given group as True, False otherwise.\n    '
    learner_group_user_models = get_learner_group_models_by_ids(user_ids, strict=True)
    progress_sharing_permissions: List[bool] = []
    for model in learner_group_user_models:
        for group_details in model.learner_groups_user_details:
            if group_details['group_id'] == group_id:
                progress_sharing_permissions.append(bool(group_details['progress_sharing_is_turned_on']))
                break
    return progress_sharing_permissions

def get_invited_learner_groups_of_learner(user_id: str) -> List[learner_group_domain.LearnerGroup]:
    if False:
        while True:
            i = 10
    'Returns a list of learner groups that the given learner has been\n    invited to join.\n\n    Args:\n        user_id: str. The id of the learner.\n\n    Returns:\n        list(LearnerGroup). A list of learner groups that the given learner\n        has been invited to join.\n    '
    learner_grp_models = learner_group_models.LearnerGroupModel.get_by_invited_learner_user_id(user_id)
    if not learner_grp_models:
        return []
    return [learner_group_services.get_learner_group_from_model(model) for model in learner_grp_models]

def get_learner_groups_joined_by_learner(user_id: str) -> List[learner_group_domain.LearnerGroup]:
    if False:
        i = 10
        return i + 15
    'Returns a list of learner groups that the given learner has joined.\n\n    Args:\n        user_id: str. The id of the learner.\n\n    Returns:\n        list(LearnerGroup). A list of learner groups that the given learner\n        is part of.\n    '
    learner_grp_models = learner_group_models.LearnerGroupModel.get_by_learner_user_id(user_id)
    if not learner_grp_models:
        return []
    return [learner_group_services.get_learner_group_from_model(model) for model in learner_grp_models]