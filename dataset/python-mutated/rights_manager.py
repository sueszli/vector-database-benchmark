"""Functions that manage rights for various user actions."""
from __future__ import annotations
import logging
from core import utils
from core.constants import constants
from core.domain import activity_services
from core.domain import change_domain
from core.domain import rights_domain
from core.domain import role_services
from core.domain import subscription_services
from core.domain import taskqueue_services
from core.domain import user_domain
from core.domain import user_services
from core.platform import models
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Union, overload
MYPY = False
if MYPY:
    from mypy_imports import collection_models
    from mypy_imports import datastore_services
    from mypy_imports import exp_models
(collection_models, exp_models) = models.Registry.import_models([models.Names.COLLECTION, models.Names.EXPLORATION])
datastore_services = models.Registry.import_datastore_services()

def get_activity_rights_from_model(activity_rights_model: Union[collection_models.CollectionRightsModel, exp_models.ExplorationRightsModel], activity_type: str) -> rights_domain.ActivityRights:
    if False:
        for i in range(10):
            print('nop')
    'Constructs an ActivityRights object from the given activity rights model.\n\n    Args:\n        activity_rights_model: ActivityRightsModel. Activity rights from the\n            datastore.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n\n    Returns:\n        ActivityRights. The rights object created from the model.\n    '
    cloned_from_value = None
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        assert isinstance(activity_rights_model, exp_models.ExplorationRightsModel)
        cloned_from_value = activity_rights_model.cloned_from
    return rights_domain.ActivityRights(activity_rights_model.id, activity_rights_model.owner_ids, activity_rights_model.editor_ids, activity_rights_model.voice_artist_ids, activity_rights_model.viewer_ids, community_owned=activity_rights_model.community_owned, cloned_from=cloned_from_value, status=activity_rights_model.status, viewable_if_private=activity_rights_model.viewable_if_private, first_published_msec=activity_rights_model.first_published_msec)

def _save_activity_rights(committer_id: str, activity_rights: rights_domain.ActivityRights, activity_type: str, commit_message: str, commit_cmds: Sequence[Mapping[str, change_domain.AcceptableChangeDictTypes]]) -> None:
    if False:
        return 10
    'Saves an ExplorationRights or CollectionRights domain object to the\n    datastore.\n\n    Args:\n        committer_id: str. ID of the committer.\n        activity_rights: ActivityRights. The rights object for the given\n            activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        commit_message: str. Descriptive message for the commit.\n        commit_cmds: list(dict). A list of commands describing what kind of\n            commit was done.\n    '
    activity_rights.validate()
    assert activity_type in (constants.ACTIVITY_TYPE_COLLECTION, constants.ACTIVITY_TYPE_EXPLORATION)
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        model: Union[exp_models.ExplorationRightsModel, collection_models.CollectionRightsModel] = exp_models.ExplorationRightsModel.get(activity_rights.id, strict=True)
    elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
        model = collection_models.CollectionRightsModel.get(activity_rights.id, strict=True)
    model.owner_ids = activity_rights.owner_ids
    model.editor_ids = activity_rights.editor_ids
    model.viewer_ids = activity_rights.viewer_ids
    model.voice_artist_ids = activity_rights.voice_artist_ids
    model.community_owned = activity_rights.community_owned
    model.status = activity_rights.status
    model.viewable_if_private = activity_rights.viewable_if_private
    model.first_published_msec = activity_rights.first_published_msec
    model.commit(committer_id, commit_message, commit_cmds)

def _update_exploration_summary(activity_rights: rights_domain.ActivityRights) -> None:
    if False:
        while True:
            i = 10
    'Updates the exploration summary for the activity associated with the\n    given rights object.\n\n    The ID of rights object is the same as the ID of associated activity.\n\n    Args:\n        activity_rights: ActivityRights. The rights object for the given\n            activity.\n    '
    from core.domain import exp_services
    exp_services.regenerate_exploration_and_contributors_summaries(activity_rights.id)

def _update_collection_summary(activity_rights: rights_domain.ActivityRights) -> None:
    if False:
        print('Hello World!')
    'Updates the collection summary for the given activity associated with\n    the given rights object.\n\n    The ID of rights object is the same as the ID of associated activity.\n\n    Args:\n        activity_rights: ActivityRights. The rights object for the given\n            activity.\n    '
    from core.domain import collection_services
    collection_services.regenerate_collection_and_contributors_summaries(activity_rights.id)

def _update_activity_summary(activity_type: str, activity_rights: rights_domain.ActivityRights) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the activity summary for the given activity associated with\n    the given rights object.\n\n    The ID of rights object is the same as the ID of associated activity.\n\n    Args:\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        activity_rights: ActivityRights. The rights object for the given\n            activity.\n    '
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        _update_exploration_summary(activity_rights)
    elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
        _update_collection_summary(activity_rights)

def create_new_exploration_rights(exploration_id: str, committer_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates a new exploration rights object and saves it to the datastore.\n    Subscribes the committer to the new exploration.\n\n    Args:\n        exploration_id: str. ID of the exploration.\n        committer_id: str. ID of the committer.\n    '
    exploration_rights = rights_domain.ActivityRights(exploration_id, [committer_id], [], [], [])
    commit_cmds: List[Dict[str, str]] = [{'cmd': rights_domain.CMD_CREATE_NEW}]
    exp_models.ExplorationRightsModel(id=exploration_rights.id, owner_ids=exploration_rights.owner_ids, editor_ids=exploration_rights.editor_ids, voice_artist_ids=exploration_rights.voice_artist_ids, viewer_ids=exploration_rights.viewer_ids, community_owned=exploration_rights.community_owned, status=exploration_rights.status, viewable_if_private=exploration_rights.viewable_if_private, first_published_msec=exploration_rights.first_published_msec).commit(committer_id, 'Created new exploration', commit_cmds)
    subscription_services.subscribe_to_exploration(committer_id, exploration_id)

@overload
def get_exploration_rights(exploration_id: str) -> rights_domain.ActivityRights:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_exploration_rights(exploration_id: str, *, strict: Literal[True]) -> rights_domain.ActivityRights:
    if False:
        while True:
            i = 10
    ...

@overload
def get_exploration_rights(exploration_id: str, *, strict: Literal[False]) -> Optional[rights_domain.ActivityRights]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_exploration_rights(exploration_id: str, *, strict: bool=False) -> Optional[rights_domain.ActivityRights]:
    if False:
        i = 10
        return i + 15
    ...

def get_exploration_rights(exploration_id: str, strict: bool=True) -> Optional[rights_domain.ActivityRights]:
    if False:
        return 10
    'Retrieves the rights for this exploration from the datastore.\n\n    Args:\n        exploration_id: str. ID of the exploration.\n        strict: bool. Whether to raise an error if there is no exploration\n            matching the given ID.\n\n    Returns:\n        ActivityRights. The rights object for the given exploration.\n\n    Raises:\n        EntityNotFoundError. The exploration with ID exploration_id was not\n            found in the datastore.\n    '
    model = exp_models.ExplorationRightsModel.get(exploration_id, strict=strict)
    if model is None:
        return None
    return get_activity_rights_from_model(model, constants.ACTIVITY_TYPE_EXPLORATION)

def get_multiple_exploration_rights_by_ids(exp_ids: List[str]) -> List[Optional[rights_domain.ActivityRights]]:
    if False:
        i = 10
        return i + 15
    'Returns a list of ActivityRights objects for given exploration ids.\n\n    Args:\n        exp_ids: list(str). List of exploration ids.\n\n    Returns:\n        list(ActivityRights or None). List of rights object --> ActivityRights\n        objects for existing exploration or None.\n    '
    exp_rights_models = exp_models.ExplorationRightsModel.get_multi(exp_ids)
    activity_rights_list: List[Optional[rights_domain.ActivityRights]] = []
    for model in exp_rights_models:
        if model is None:
            activity_rights_list.append(None)
        else:
            activity_rights_list.append(get_activity_rights_from_model(model, constants.ACTIVITY_TYPE_EXPLORATION))
    return activity_rights_list

def _get_activity_rights_where_user_is_owner(activity_type: str, user_id: str) -> List[rights_domain.ActivityRights]:
    if False:
        i = 10
        return i + 15
    'Returns a list of activity rights where the user is the owner.\n\n    Args:\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        user_id: str. The id of the user.\n\n    Returns:\n        list(ActivityRights). List of domain objects where the user has some\n        role.\n    '
    assert activity_type in (constants.ACTIVITY_TYPE_COLLECTION, constants.ACTIVITY_TYPE_EXPLORATION)
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        activity_rights_models: Sequence[Union[collection_models.CollectionRightsModel, exp_models.ExplorationRightsModel]] = exp_models.ExplorationRightsModel.query(datastore_services.any_of(exp_models.ExplorationRightsModel.owner_ids == user_id)).fetch()
    elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
        activity_rights_models = collection_models.CollectionRightsModel.query(datastore_services.any_of(collection_models.CollectionRightsModel.owner_ids == user_id)).fetch()
    return [get_activity_rights_from_model(activity_rights_model, activity_type) for activity_rights_model in activity_rights_models]

def get_exploration_rights_where_user_is_owner(user_id: str) -> List[rights_domain.ActivityRights]:
    if False:
        i = 10
        return i + 15
    'Returns a list of exploration rights where the user is the owner.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(ActivityRights). List of domain objects where the user is\n        the owner.\n    '
    return _get_activity_rights_where_user_is_owner(constants.ACTIVITY_TYPE_EXPLORATION, user_id)

def get_collection_rights_where_user_is_owner(user_id: str) -> List[rights_domain.ActivityRights]:
    if False:
        i = 10
        return i + 15
    'Returns a list of collection rights where the user is the owner.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(ActivityRights). List of domain objects where the user is\n        the owner.\n    '
    return _get_activity_rights_where_user_is_owner(constants.ACTIVITY_TYPE_COLLECTION, user_id)

def is_exploration_private(exploration_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether exploration is private.\n\n    Args:\n        exploration_id: str. ID of the exploration.\n\n    Returns:\n        bool. Whether the exploration is private or not.\n    '
    exploration_rights = get_exploration_rights(exploration_id)
    return exploration_rights.status == rights_domain.ACTIVITY_STATUS_PRIVATE

def is_exploration_public(exploration_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether exploration is public.\n\n    Args:\n        exploration_id: str. ID of the exploration.\n\n    Returns:\n        bool. Whether the exploration is public.\n    '
    exploration_rights = get_exploration_rights(exploration_id)
    return exploration_rights.status == rights_domain.ACTIVITY_STATUS_PUBLIC

def is_exploration_cloned(exploration_id: str) -> bool:
    if False:
        return 10
    'Returns whether the exploration is a clone of another exploration.\n\n    Args:\n        exploration_id: str. ID of the exploration.\n\n    Returns:\n        bool. Whether the exploration is a clone of another exploration.\n    '
    exploration_rights = get_exploration_rights(exploration_id)
    return bool(exploration_rights.cloned_from)

def create_new_collection_rights(collection_id: str, committer_id: str) -> None:
    if False:
        return 10
    'Creates a new collection rights object and saves it to the datastore.\n    Subscribes the committer to the new collection.\n\n    Args:\n        collection_id: str. ID of the collection.\n        committer_id: str. ID of the committer.\n    '
    collection_rights = rights_domain.ActivityRights(collection_id, [committer_id], [], [], [])
    commit_cmds = [{'cmd': rights_domain.CMD_CREATE_NEW}]
    collection_models.CollectionRightsModel(id=collection_rights.id, owner_ids=collection_rights.owner_ids, editor_ids=collection_rights.editor_ids, voice_artist_ids=collection_rights.voice_artist_ids, viewer_ids=collection_rights.viewer_ids, community_owned=collection_rights.community_owned, status=collection_rights.status, viewable_if_private=collection_rights.viewable_if_private, first_published_msec=collection_rights.first_published_msec).commit(committer_id, 'Created new collection', commit_cmds)
    subscription_services.subscribe_to_collection(committer_id, collection_id)

@overload
def get_collection_rights(collection_id: str) -> rights_domain.ActivityRights:
    if False:
        while True:
            i = 10
    ...

@overload
def get_collection_rights(collection_id: str, *, strict: Literal[True]) -> rights_domain.ActivityRights:
    if False:
        print('Hello World!')
    ...

@overload
def get_collection_rights(collection_id: str, *, strict: Literal[False]) -> Optional[rights_domain.ActivityRights]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_collection_rights(collection_id: str, *, strict: bool=False) -> Optional[rights_domain.ActivityRights]:
    if False:
        return 10
    ...

def get_collection_rights(collection_id: str, strict: bool=True) -> Optional[rights_domain.ActivityRights]:
    if False:
        print('Hello World!')
    'Retrieves the rights for this collection from the datastore.\n\n    Args:\n        collection_id: str. ID of the collection.\n        strict: bool. Whether to raise an error if ID is not found.\n\n    Returns:\n        ActivityRights. The rights object for the collection.\n\n    Raises:\n        EntityNotFoundError. The collection with ID collection_id is not found\n            in the datastore.\n    '
    model = collection_models.CollectionRightsModel.get(collection_id, strict=strict)
    if model is None:
        return None
    return get_activity_rights_from_model(model, constants.ACTIVITY_TYPE_COLLECTION)

def get_collection_owner_names(collection_id: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Retrieves the owners for this collection from the datastore.\n\n    Args:\n        collection_id: str. ID of the collection.\n\n    Returns:\n        list(str). Human-readable usernames (or truncated email addresses) of\n        owners for this collection.\n    '
    collection_rights = get_collection_rights(collection_id)
    return user_services.get_human_readable_user_ids(collection_rights.owner_ids)

def is_collection_private(collection_id: str) -> bool:
    if False:
        print('Hello World!')
    'Returns whether the collection is private.\n\n    Args:\n        collection_id: str. ID of the collection.\n\n    Returns:\n        bool. Whether the collection is private.\n    '
    collection_rights = get_collection_rights(collection_id)
    return collection_rights.status == rights_domain.ACTIVITY_STATUS_PRIVATE

def is_collection_public(collection_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the collection is public.\n\n    Args:\n        collection_id: str. ID of the collection.\n\n    Returns:\n        bool. Whether the collection is public.\n    '
    collection_rights = get_collection_rights(collection_id)
    return collection_rights.status == rights_domain.ACTIVITY_STATUS_PUBLIC

@overload
def _get_activity_rights(activity_type: str, activity_id: str, *, strict: Literal[True]) -> rights_domain.ActivityRights:
    if False:
        i = 10
        return i + 15
    ...

@overload
def _get_activity_rights(activity_type: str, activity_id: str) -> Optional[rights_domain.ActivityRights]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def _get_activity_rights(activity_type: str, activity_id: str, *, strict: Literal[False]) -> Optional[rights_domain.ActivityRights]:
    if False:
        print('Hello World!')
    ...

def _get_activity_rights(activity_type: str, activity_id: str, strict: bool=False) -> Optional[rights_domain.ActivityRights]:
    if False:
        i = 10
        return i + 15
    "Retrieves the rights object for the given activity\n    based on its type.\n\n    Args:\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        activity_id: str. ID of the activity.\n        strict: bool. Whether to fail noisily if the activity_rights\n            doesn't exist for the given activity_id.\n\n    Returns:\n        ActivityRights|None. The rights object associated with the given\n        activity, or None if no rights object exists.\n\n    Raises:\n        Exception. The activity_type provided is unknown.\n    "
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        activity_rights = get_exploration_rights(activity_id, strict=strict)
    elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
        activity_rights = get_collection_rights(activity_id, strict=strict)
    else:
        raise Exception('Cannot get activity rights for unknown activity type: %s' % activity_type)
    return activity_rights

def check_can_access_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        return 10
    'Checks whether the user can access given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the given activity can be accessed by the given user.\n    '
    if activity_rights is None:
        return False
    elif activity_rights.is_published():
        return bool(role_services.ACTION_PLAY_ANY_PUBLIC_ACTIVITY in user.actions)
    elif activity_rights.is_private():
        return bool(role_services.ACTION_PLAY_ANY_PRIVATE_ACTIVITY in user.actions or (user.user_id and (activity_rights.is_viewer(user.user_id) or activity_rights.is_owner(user.user_id) or activity_rights.is_editor(user.user_id) or activity_rights.is_voice_artist(user.user_id) or activity_rights.viewable_if_private)))
    return False

def check_can_edit_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks whether the user can edit given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the given user can edit this activity.\n    '
    if activity_rights is None:
        return False
    if role_services.ACTION_EDIT_OWNED_ACTIVITY not in user.actions:
        return False
    if user.user_id and (activity_rights.is_owner(user.user_id) or activity_rights.is_editor(user.user_id)):
        return True
    if activity_rights.community_owned or role_services.ACTION_EDIT_ANY_ACTIVITY in user.actions:
        return True
    if activity_rights.is_published() and role_services.ACTION_EDIT_ANY_PUBLIC_ACTIVITY in user.actions:
        return True
    return False

def check_can_voiceover_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        while True:
            i = 10
    'Checks whether the user can voiceover given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the given user can voiceover this activity.\n    '
    if activity_rights is None:
        return False
    if role_services.ACTION_EDIT_OWNED_ACTIVITY not in user.actions:
        return False
    if user.user_id and (activity_rights.is_owner(user.user_id) or activity_rights.is_editor(user.user_id) or activity_rights.is_voice_artist(user.user_id)):
        return True
    if activity_rights.community_owned or role_services.ACTION_EDIT_ANY_ACTIVITY in user.actions:
        return True
    if activity_rights.is_published() and role_services.ACTION_EDIT_ANY_PUBLIC_ACTIVITY in user.actions:
        return True
    return False

def check_can_manage_voice_artist_in_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check whether the user can manage voice artist for an activity.\n    Callers are expected to ensure that the activity is published when we are\n    adding voice artists.\n\n    Args:\n        user: UserActionInfo. Object having user_id, role, and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can assign voice artist.\n    '
    if activity_rights is None:
        return False
    return role_services.ACTION_CAN_MANAGE_VOICE_ARTIST in user.actions

def check_can_save_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks whether the user can save given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can save given activity.\n    '
    return check_can_edit_activity(user, activity_rights) or check_can_voiceover_activity(user, activity_rights)

def check_can_delete_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        print('Hello World!')
    'Checks whether the user can delete given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can delete given activity.\n    '
    if activity_rights is None:
        return False
    if user.user_id is None:
        return False
    if role_services.ACTION_DELETE_ANY_ACTIVITY in user.actions:
        return True
    elif activity_rights.is_private() and role_services.ACTION_DELETE_OWNED_PRIVATE_ACTIVITY in user.actions and activity_rights.is_owner(user.user_id):
        return True
    elif activity_rights.is_published() and role_services.ACTION_DELETE_ANY_PUBLIC_ACTIVITY in user.actions:
        return True
    return False

def check_can_modify_core_activity_roles(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        while True:
            i = 10
    'Checks whether the user can modify core roles for the given activity. The\n    core roles for an activity includes owner, editor etc.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can modify roles for given activity.\n    '
    if activity_rights is None:
        return False
    if user.user_id is None:
        return False
    if activity_rights.community_owned or activity_rights.cloned_from:
        return False
    if role_services.ACTION_MODIFY_CORE_ROLES_FOR_ANY_ACTIVITY in user.actions:
        return True
    if role_services.ACTION_MODIFY_CORE_ROLES_FOR_OWNED_ACTIVITY in user.actions:
        if activity_rights.is_owner(user.user_id):
            return True
    return False

def check_can_release_ownership(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        return 10
    'Checks whether the user can release ownership for given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can release ownership for given activity.\n    '
    if activity_rights is None:
        return False
    if activity_rights.is_private():
        return False
    return check_can_modify_core_activity_roles(user, activity_rights)

def check_can_publish_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        print('Hello World!')
    'Checks whether the user can publish given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can publish given activity.\n    '
    if activity_rights is None:
        return False
    if activity_rights.cloned_from:
        return False
    if activity_rights.is_published():
        return False
    if role_services.ACTION_PUBLISH_ANY_ACTIVITY in user.actions:
        return True
    if role_services.ACTION_PUBLISH_OWNED_ACTIVITY in user.actions:
        if user.user_id and activity_rights.is_owner(user.user_id):
            return True
    return False

def check_can_unpublish_activity(user: user_domain.UserActionsInfo, activity_rights: Optional[rights_domain.ActivityRights]) -> bool:
    if False:
        while True:
            i = 10
    'Checks whether the user can unpublish given activity.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        activity_rights: ActivityRights or None. Rights object for the given\n            activity.\n\n    Returns:\n        bool. Whether the user can unpublish given activity.\n    '
    if activity_rights is None:
        return False
    if activity_rights.community_owned:
        return False
    if activity_rights.is_published():
        if role_services.ACTION_UNPUBLISH_ANY_PUBLIC_ACTIVITY in user.actions:
            return True
    return False

def _assign_role(committer: user_domain.UserActionsInfo, assignee_id: str, new_role: str, activity_id: str, activity_type: str, allow_assigning_any_role: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Assigns a new role to the user.\n\n    Args:\n        committer: UserActionsInfo. UserActionInfo object for the user\n            who is performing the action.\n        assignee_id: str. ID of the user whose role is being changed.\n        new_role: str. The name of the new role: One of\n            ROLE_OWNER,\n            ROLE_EDITOR,\n            ROLE_VOICE_ARTIST,\n            ROLE_VIEWER.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        allow_assigning_any_role: bool. Whether to assign a role to the user\n            irrespective of whether they have any existing role in the activity.\n            The default value is false.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. The user already owns the activity.\n        Exception. The user can already edit the activity.\n        Exception. The user can already voiceover the activity.\n        Exception. The activity is already publicly editable.\n        Exception. The activity is already publicly translatable.\n        Exception. The user can already view the activity.\n        Exception. The activity is already publicly viewable.\n        Exception. The role is invalid.\n        Exception. No activity_rights exists for the given activity id.\n        Exception. Guest user is not allowed to assign roles.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to assign roles.')
    activity_rights = _get_activity_rights(activity_type, activity_id)
    if activity_rights is None:
        raise Exception('No activity_rights exists for the given activity_id: %s' % activity_id)
    if new_role == rights_domain.ROLE_VOICE_ARTIST and activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        if activity_rights.is_published():
            user_can_assign_role = check_can_manage_voice_artist_in_activity(committer, activity_rights)
        else:
            raise Exception('Could not assign voice artist to private activity.')
    else:
        user_can_assign_role = check_can_modify_core_activity_roles(committer, activity_rights)
    if not user_can_assign_role:
        logging.error('User %s tried to allow user %s to be a(n) %s of activity %s but was refused permission.' % (committer_id, assignee_id, new_role, activity_id))
        raise Exception('UnauthorizedUserException: Could not assign new role.')
    assignee_username = user_services.get_username(assignee_id)
    old_role = rights_domain.ROLE_NONE
    if new_role not in [rights_domain.ROLE_OWNER, rights_domain.ROLE_EDITOR, rights_domain.ROLE_VOICE_ARTIST, rights_domain.ROLE_VIEWER]:
        raise Exception('Invalid role: %s' % new_role)
    if allow_assigning_any_role:
        old_role = activity_rights.assign_new_role(assignee_id, new_role)
    elif new_role == rights_domain.ROLE_OWNER:
        if activity_rights.is_owner(assignee_id):
            raise Exception('This user already owns this %s.' % activity_type)
        activity_rights.owner_ids.append(assignee_id)
        if assignee_id in activity_rights.viewer_ids:
            activity_rights.viewer_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_VIEWER
        if assignee_id in activity_rights.editor_ids:
            activity_rights.editor_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_EDITOR
        if assignee_id in activity_rights.voice_artist_ids:
            activity_rights.voice_artist_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_VOICE_ARTIST
    elif new_role == rights_domain.ROLE_EDITOR:
        if activity_rights.is_editor(assignee_id) or activity_rights.is_owner(assignee_id):
            raise Exception('This user already can edit this %s.' % activity_type)
        activity_rights.editor_ids.append(assignee_id)
        if assignee_id in activity_rights.voice_artist_ids:
            activity_rights.voice_artist_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_VOICE_ARTIST
        if assignee_id in activity_rights.viewer_ids:
            activity_rights.viewer_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_VIEWER
    elif new_role == rights_domain.ROLE_VOICE_ARTIST:
        if activity_rights.is_editor(assignee_id) or activity_rights.is_voice_artist(assignee_id) or activity_rights.is_owner(assignee_id):
            raise Exception('This user already can voiceover this %s.' % activity_type)
        activity_rights.voice_artist_ids.append(assignee_id)
        if assignee_id in activity_rights.viewer_ids:
            activity_rights.viewer_ids.remove(assignee_id)
            old_role = rights_domain.ROLE_VIEWER
    elif new_role == rights_domain.ROLE_VIEWER:
        if activity_rights.is_owner(assignee_id) or activity_rights.is_editor(assignee_id) or activity_rights.is_viewer(assignee_id):
            raise Exception('This user already can view this %s.' % activity_type)
        if activity_rights.status != rights_domain.ACTIVITY_STATUS_PRIVATE:
            raise Exception('Public %ss can be viewed by anyone.' % activity_type)
        activity_rights.viewer_ids.append(assignee_id)
    commit_message = rights_domain.ASSIGN_ROLE_COMMIT_MESSAGE_TEMPLATE % (assignee_username, old_role, new_role)
    commit_cmds = [{'cmd': rights_domain.CMD_CHANGE_ROLE, 'assignee_id': assignee_id, 'old_role': old_role, 'new_role': new_role}]
    _save_activity_rights(committer_id, activity_rights, activity_type, commit_message, commit_cmds)
    _update_activity_summary(activity_type, activity_rights)

def _deassign_role(committer: user_domain.UserActionsInfo, removed_user_id: str, activity_id: str, activity_type: str) -> None:
    if False:
        while True:
            i = 10
    'Deassigns given user from their current role in the activity.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        removed_user_id: str. ID of the user who is being deassigned from\n            the activity.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n\n    Raises:\n        Exception. UnauthorizedUserException: Could not deassign role.\n        Exception. This user does not have any role for the given activity.\n        Exception. No activity_rights exists for the given activity id.\n        Exception. Guest user is not allowed to deassign roles.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to deassign roles.')
    activity_rights = _get_activity_rights(activity_type, activity_id)
    if activity_rights is None:
        raise Exception('No activity_rights exists for the given activity_id: %s' % activity_id)
    if activity_rights.is_voice_artist(removed_user_id) and activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        user_can_deassign_role = check_can_manage_voice_artist_in_activity(committer, activity_rights)
    else:
        user_can_deassign_role = check_can_modify_core_activity_roles(committer, activity_rights)
    if not user_can_deassign_role:
        logging.error('User %s tried to remove user %s from an activity %s but was refused permission.' % (committer_id, removed_user_id, activity_id))
        raise Exception('UnauthorizedUserException: Could not deassign role.')
    if activity_rights.is_owner(removed_user_id):
        old_role = rights_domain.ROLE_OWNER
        activity_rights.owner_ids.remove(removed_user_id)
    elif activity_rights.is_editor(removed_user_id):
        old_role = rights_domain.ROLE_EDITOR
        activity_rights.editor_ids.remove(removed_user_id)
    elif activity_rights.is_voice_artist(removed_user_id):
        old_role = rights_domain.ROLE_VOICE_ARTIST
        activity_rights.voice_artist_ids.remove(removed_user_id)
    elif activity_rights.is_viewer(removed_user_id):
        old_role = rights_domain.ROLE_VIEWER
        activity_rights.viewer_ids.remove(removed_user_id)
    else:
        raise Exception('This user does not have any role in %s with ID %s' % (activity_type, activity_id))
    assignee_username = user_services.get_usernames([removed_user_id])[0]
    if assignee_username is None:
        assignee_username = 'ANONYMOUS'
    commit_message = 'Remove %s from role %s for %s' % (assignee_username, old_role, activity_type)
    commit_cmds = [{'cmd': rights_domain.CMD_REMOVE_ROLE, 'removed_user_id': removed_user_id, 'old_role': old_role}]
    _save_activity_rights(committer_id, activity_rights, activity_type, commit_message, commit_cmds)
    _update_activity_summary(activity_type, activity_rights)

def _release_ownership_of_activity(committer: user_domain.UserActionsInfo, activity_id: str, activity_type: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Releases ownership of the given activity to the community.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user who\n            is performing the action.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n\n    Raises:\n        Exception. The committer does not have release rights.\n        Exception. The activity rights does not exist for the given activity_id.\n        Exception. Guest user is not allowed to release ownership of activity.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to release ownership of activity.')
    activity_rights = _get_activity_rights(activity_type, activity_id, strict=True)
    if not check_can_release_ownership(committer, activity_rights):
        logging.error('User %s tried to release ownership of %s %s but was refused permission.' % (committer_id, activity_type, activity_id))
        raise Exception('The ownership of this %s cannot be released.' % activity_type)
    activity_rights.community_owned = True
    activity_rights.owner_ids = []
    activity_rights.editor_ids = []
    activity_rights.viewer_ids = []
    activity_rights.voice_artist_ids = []
    commit_cmds = [{'cmd': rights_domain.CMD_RELEASE_OWNERSHIP}]
    _save_activity_rights(committer_id, activity_rights, activity_type, '%s ownership released to the community.' % activity_type, commit_cmds)
    _update_activity_summary(activity_type, activity_rights)

def _change_activity_status(committer_id: str, activity_id: str, activity_type: str, new_status: str, commit_message: str) -> None:
    if False:
        while True:
            i = 10
    'Changes the status of the given activity.\n\n    Args:\n        committer_id: str. ID of the user who is performing the update action.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n        new_status: str. The new status of the activity.\n        commit_message: str. The human-written commit message for this change.\n\n    Raises:\n        Exception. The activity rights does not exist for the given activity_id.\n    '
    activity_rights = _get_activity_rights(activity_type, activity_id, strict=True)
    old_status = activity_rights.status
    activity_rights.status = new_status
    if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
        cmd_type = rights_domain.CMD_CHANGE_EXPLORATION_STATUS
    elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
        cmd_type = rights_domain.CMD_CHANGE_COLLECTION_STATUS
    commit_cmds = [{'cmd': cmd_type, 'old_status': old_status, 'new_status': new_status}]
    if new_status != rights_domain.ACTIVITY_STATUS_PRIVATE:
        activity_rights.viewer_ids = []
        if activity_rights.first_published_msec is None:
            activity_rights.first_published_msec = utils.get_current_time_in_millisecs()
    _save_activity_rights(committer_id, activity_rights, activity_type, commit_message, commit_cmds)
    _update_activity_summary(activity_type, activity_rights)

def _publish_activity(committer: user_domain.UserActionsInfo, activity_id: str, activity_type: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Publishes the given activity.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n\n    Raises:\n        Exception. The committer does not have rights to publish the\n            activity.\n        Exception. Guest user is not allowed to publish activities.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to publish activities.')
    activity_rights = _get_activity_rights(activity_type, activity_id)
    if not check_can_publish_activity(committer, activity_rights):
        logging.error('User %s tried to publish %s %s but was refused permission.' % (committer_id, activity_type, activity_id))
        raise Exception('This %s cannot be published.' % activity_type)
    _change_activity_status(committer_id, activity_id, activity_type, rights_domain.ACTIVITY_STATUS_PUBLIC, '%s published.' % activity_type)

def _unpublish_activity(committer: user_domain.UserActionsInfo, activity_id: str, activity_type: str) -> None:
    if False:
        i = 10
        return i + 15
    'Unpublishes the given activity.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        activity_id: str. ID of the activity.\n        activity_type: str. The type of activity. Possible values:\n            constants.ACTIVITY_TYPE_EXPLORATION,\n            constants.ACTIVITY_TYPE_COLLECTION.\n\n    Raises:\n        Exception. The committer does not have rights to unpublish the\n            activity.\n        Exception. Guest user is not allowed to unpublish activities.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to unpublish activities.')
    activity_rights = _get_activity_rights(activity_type, activity_id)
    if not check_can_unpublish_activity(committer, activity_rights):
        logging.error('User %s tried to unpublish %s %s but was refused permission.' % (committer_id, activity_type, activity_id))
        raise Exception('This %s cannot be unpublished.' % activity_type)
    _change_activity_status(committer_id, activity_id, activity_type, rights_domain.ACTIVITY_STATUS_PRIVATE, '%s unpublished.' % activity_type)
    activity_services.remove_featured_activity(activity_type, activity_id)

def assign_role_for_exploration(committer: user_domain.UserActionsInfo, exploration_id: str, assignee_id: str, new_role: str) -> None:
    if False:
        print('Hello World!')
    'Assigns a user to the given role and subscribes the assignee to future\n    exploration updates.\n\n    The caller should ensure that assignee_id corresponds to a valid user in\n    the system.\n\n    Args:\n        committer: UserActionsInfo. The UserActionsInfo object for the\n            committer.\n        exploration_id: str. ID of the exploration.\n        assignee_id: str. ID of the user whose role is being changed.\n        new_role: str. The name of the new role: One of\n            ROLE_OWNER,\n            ROLE_EDITOR,\n            ROLE_VOICE_ARTIST.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _assign_role.\n    '
    _assign_role(committer, assignee_id, new_role, exploration_id, constants.ACTIVITY_TYPE_EXPLORATION, allow_assigning_any_role=True)
    if new_role in [rights_domain.ROLE_OWNER, rights_domain.ROLE_EDITOR, rights_domain.ROLE_VOICE_ARTIST]:
        subscription_services.subscribe_to_exploration(assignee_id, exploration_id)

def deassign_role_for_exploration(committer: user_domain.UserActionsInfo, exploration_id: str, removed_user_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deassigns a user from a given exploration.\n\n    The caller should ensure that assignee_id corresponds to a valid user in\n    the system.\n\n    Args:\n        committer: UserActionsInfo. The UserActionsInfo object for the\n            committer.\n        exploration_id: str. ID of the exploration.\n        removed_user_id: str. ID of the user whom is being deassigned from\n            the exploration.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _deassign_role.\n    '
    _deassign_role(committer, removed_user_id, exploration_id, constants.ACTIVITY_TYPE_EXPLORATION)

def release_ownership_of_exploration(committer: user_domain.UserActionsInfo, exploration_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Releases ownership of the given exploration to the community.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        exploration_id: str. ID of the exploration.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _release_ownership_of_activity.\n    '
    _release_ownership_of_activity(committer, exploration_id, constants.ACTIVITY_TYPE_EXPLORATION)

def set_private_viewability_of_exploration(committer: user_domain.UserActionsInfo, exploration_id: str, viewable_if_private: bool) -> None:
    if False:
        while True:
            i = 10
    "Sets the viewable_if_private attribute for the given exploration's rights\n    object.\n\n    If viewable_if_private is True, this allows a private exploration\n    to be viewed by anyone with the link.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        exploration_id: str. ID of the exploration.\n        viewable_if_private: bool. Whether the exploration should be made\n            viewable (by anyone with the link).\n\n    Raises:\n        Exception. The committer does not have the permission to perform change\n            action.\n        Exception. If the viewable_if_private property is already as desired.\n        Exception. Guest user is not allowed to set viewability of exploration.\n    "
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to set viewability of exploration.')
    exploration_rights = get_exploration_rights(exploration_id)
    if not check_can_publish_activity(committer, exploration_rights):
        logging.error('User %s tried to change private viewability of exploration %s but was refused permission.' % (committer_id, exploration_id))
        raise Exception('The viewability status of this exploration cannot be changed.')
    old_viewable_if_private = exploration_rights.viewable_if_private
    if old_viewable_if_private == viewable_if_private:
        raise Exception('Trying to change viewability status of this exploration to %s, but that is already the current value.' % viewable_if_private)
    exploration_rights.viewable_if_private = viewable_if_private
    commit_cmds: List[Dict[str, Union[str, bool]]] = [{'cmd': rights_domain.CMD_CHANGE_PRIVATE_VIEWABILITY, 'old_viewable_if_private': old_viewable_if_private, 'new_viewable_if_private': viewable_if_private}]
    commit_message = 'Made exploration viewable to anyone with the link.' if viewable_if_private else 'Made exploration viewable only to invited playtesters.'
    _save_activity_rights(committer_id, exploration_rights, constants.ACTIVITY_TYPE_EXPLORATION, commit_message, commit_cmds)
    _update_exploration_summary(exploration_rights)

def publish_exploration(committer: user_domain.UserActionsInfo, exploration_id: str) -> None:
    if False:
        return 10
    'Publishes the given exploration.\n\n    It is the responsibility of the caller to check that the exploration is\n    valid prior to publication.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        exploration_id: str. ID of the exploration.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _publish_activity.\n    '
    _publish_activity(committer, exploration_id, constants.ACTIVITY_TYPE_EXPLORATION)

def unpublish_exploration(committer: user_domain.UserActionsInfo, exploration_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Unpublishes the given exploration.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        exploration_id: str. ID of the exploration.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _unpublish_activity.\n    '
    _unpublish_activity(committer, exploration_id, constants.ACTIVITY_TYPE_EXPLORATION)
    taskqueue_services.defer(taskqueue_services.FUNCTION_ID_DELETE_EXPS_FROM_ACTIVITIES, taskqueue_services.QUEUE_NAME_ONE_OFF_JOBS, [exploration_id])

def assign_role_for_collection(committer: user_domain.UserActionsInfo, collection_id: str, assignee_id: str, new_role: str) -> None:
    if False:
        print('Hello World!')
    'Assign the given user to the given role and subscribes the assignee\n    to future collection updates.\n\n    The caller should ensure that assignee_id corresponds to a valid user in\n    the system.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        collection_id: str. ID of the collection.\n        assignee_id: str. ID of the user whose role is being changed.\n        new_role: str. The name of the new role: One of\n            ROLE_OWNER,\n            ROLE_EDITOR.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _assign_role.\n    '
    _assign_role(committer, assignee_id, new_role, collection_id, constants.ACTIVITY_TYPE_COLLECTION)
    if new_role in [rights_domain.ROLE_OWNER, rights_domain.ROLE_EDITOR]:
        subscription_services.subscribe_to_collection(assignee_id, collection_id)

def deassign_role_for_collection(committer: user_domain.UserActionsInfo, collection_id: str, removed_user_id: str) -> None:
    if False:
        return 10
    'Deassigns a user from a given collection.\n\n    The caller should ensure that assignee_id corresponds to a valid user in\n    the system.\n\n    Args:\n        committer: UserActionsInfo. The UserActionsInfo object for the\n            committer.\n        collection_id: str. ID of the collection.\n        removed_user_id: str. ID of the user whom is being deassigned from\n            the exploration.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _deassign_role.\n    '
    _deassign_role(committer, removed_user_id, collection_id, constants.ACTIVITY_TYPE_COLLECTION)

def release_ownership_of_collection(committer: user_domain.UserActionsInfo, collection_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Releases ownership of the given collection to the community.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        collection_id: str. ID of the collection.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _release_ownership_of_activity.\n    '
    _release_ownership_of_activity(committer, collection_id, constants.ACTIVITY_TYPE_COLLECTION)

def publish_collection(committer: user_domain.UserActionsInfo, collection_id: str) -> None:
    if False:
        while True:
            i = 10
    'Publishes the given collection.\n\n    It is the responsibility of the caller to check that the collection is\n    valid prior to publication.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        collection_id: str. ID of the collection.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _publish_activity.\n    '
    _publish_activity(committer, collection_id, constants.ACTIVITY_TYPE_COLLECTION)

def unpublish_collection(committer: user_domain.UserActionsInfo, collection_id: str) -> None:
    if False:
        return 10
    'Unpublishes the given collection.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        collection_id: str. ID of the collection.\n\n    Raises:\n        Exception. This could potentially throw an exception from\n            _unpublish_activity.\n    '
    _unpublish_activity(committer, collection_id, constants.ACTIVITY_TYPE_COLLECTION)