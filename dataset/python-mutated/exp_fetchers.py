"""Commands that can be used to fetch exploration related models.

All functions here should be agnostic of how ExplorationModel objects are
stored in the database. In particular, the various query methods should
delegate to the Exploration model class. This will enable the exploration
storage model to be changed without affecting this module and others above it.
"""
from __future__ import annotations
import copy
import logging
from core import feconf
from core.domain import caching_services
from core.domain import exp_domain
from core.domain import subscription_services
from core.domain import user_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, overload
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import exp_models
    from mypy_imports import user_models
(exp_models, user_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.USER])
datastore_services = models.Registry.import_datastore_services()

def _migrate_states_schema(versioned_exploration_states: exp_domain.VersionedExplorationStatesDict, init_state_name: str, language_code: str) -> Optional[int]:
    if False:
        print('Hello World!')
    'Holds the responsibility of performing a step-by-step, sequential update\n    of an exploration states structure based on the schema version of the input\n    exploration dictionary. This is very similar to the YAML conversion process\n    found in exp_domain.py and, in fact, many of the conversion functions for\n    states are also used in the YAML conversion pipeline. If the current\n    exploration states schema version changes\n    (feconf.CURRENT_STATE_SCHEMA_VERSION), a new conversion\n    function must be added and some code appended to this function to account\n    for that new version.\n\n    Args:\n        versioned_exploration_states: dict. A dict with two keys:\n            - states_schema_version: int. the states schema version for the\n                exploration.\n            - states: the dict of states comprising the exploration. The keys in\n                this dict are state names.\n        init_state_name: str. Name of initial state.\n        language_code: str. The language code of the exploration.\n\n    Returns:\n        None|int. The next content Id index for generating new content Id.\n\n    Raises:\n        Exception. The given states_schema_version is invalid.\n    '
    states_schema_version = versioned_exploration_states['states_schema_version']
    if not feconf.EARLIEST_SUPPORTED_STATE_SCHEMA_VERSION <= states_schema_version <= feconf.CURRENT_STATE_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v%d-v%d exploration state schemas at present.' % (feconf.EARLIEST_SUPPORTED_STATE_SCHEMA_VERSION, feconf.CURRENT_STATE_SCHEMA_VERSION))
    next_content_id_index = None
    while states_schema_version < feconf.CURRENT_STATE_SCHEMA_VERSION:
        if states_schema_version == 54:
            next_content_id_index = exp_domain.Exploration.update_states_from_model(versioned_exploration_states, states_schema_version, init_state_name, language_code)
        else:
            exp_domain.Exploration.update_states_from_model(versioned_exploration_states, states_schema_version, init_state_name, language_code)
        states_schema_version += 1
    return next_content_id_index

def get_new_exploration_id() -> str:
    if False:
        i = 10
        return i + 15
    'Returns a new exploration id.\n\n    Returns:\n        str. A new exploration id.\n    '
    return exp_models.ExplorationModel.get_new_id('')

def get_new_unique_progress_url_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns a new unique progress url id.\n\n    Returns:\n        str. A new unique progress url id.\n    '
    return exp_models.TransientCheckpointUrlModel.get_new_progress_id()

def get_multiple_versioned_exp_interaction_ids_mapping_by_version(exp_id: str, version_numbers: List[int]) -> List[exp_domain.VersionedExplorationInteractionIdsMapping]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of VersionedExplorationInteractionIdsMapping domain\n    objects corresponding to the specified versions.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        version_numbers: list(int). List of version numbers.\n\n    Returns:\n        list(VersionedExplorationInteractionIdsMapping). List of Exploration\n        domain objects.\n\n    Raises:\n        Exception. One or more of the given versions of the exploration could\n            not be converted to the latest schema version.\n    '
    versioned_exp_interaction_ids_mapping = []
    exploration_models = exp_models.ExplorationModel.get_multi_versions(exp_id, version_numbers)
    for (index, exploration_model) in enumerate(exploration_models):
        if exploration_model.states_schema_version != feconf.CURRENT_STATE_SCHEMA_VERSION:
            raise Exception('Exploration(id=%s, version=%s, states_schema_version=%s) does not match the latest schema version %s' % (exp_id, version_numbers[index], exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION))
        states_to_interaction_id_mapping = {}
        for state_name in exploration_model.states:
            states_to_interaction_id_mapping[state_name] = exploration_model.states[state_name]['interaction']['id']
        versioned_exp_interaction_ids_mapping.append(exp_domain.VersionedExplorationInteractionIdsMapping(exploration_model.version, states_to_interaction_id_mapping))
    return versioned_exp_interaction_ids_mapping

def get_exploration_from_model(exploration_model: exp_models.ExplorationModel, run_conversion: bool=True) -> exp_domain.Exploration:
    if False:
        while True:
            i = 10
    "Returns an Exploration domain object given an exploration model loaded\n    from the datastore.\n\n    If run_conversion is True, then the exploration's states schema version\n    will be checked against the current states schema version. If they do not\n    match, the exploration will be automatically updated to the latest states\n    schema version.\n\n    IMPORTANT NOTE TO DEVELOPERS: In general, run_conversion should never be\n    False. This option is only used for testing that the states schema version\n    migration works correctly, and it should never be changed otherwise.\n\n    Args:\n        exploration_model: ExplorationModel. An exploration storage model.\n        run_conversion: bool. When True, updates the exploration to the latest\n            states_schema_version if necessary.\n\n    Returns:\n        Exploration. The exploration domain object corresponding to the given\n        exploration model.\n    "
    versioned_exploration_states: exp_domain.VersionedExplorationStatesDict = {'states_schema_version': exploration_model.states_schema_version, 'states': copy.deepcopy(exploration_model.states)}
    init_state_name = exploration_model.init_state_name
    next_content_id_index = None
    language_code = exploration_model.language_code
    if run_conversion and exploration_model.states_schema_version != feconf.CURRENT_STATE_SCHEMA_VERSION:
        next_content_id_index = _migrate_states_schema(versioned_exploration_states, init_state_name, language_code)
    if next_content_id_index is not None:
        exploration_model.next_content_id_index = next_content_id_index
    return exp_domain.Exploration(exploration_model.id, exploration_model.title, exploration_model.category, exploration_model.objective, exploration_model.language_code, exploration_model.tags, exploration_model.blurb, exploration_model.author_notes, versioned_exploration_states['states_schema_version'], exploration_model.init_state_name, versioned_exploration_states['states'], exploration_model.param_specs, exploration_model.param_changes, exploration_model.version, exploration_model.auto_tts_enabled, exploration_model.correctness_feedback_enabled, exploration_model.next_content_id_index, exploration_model.edits_allowed, created_on=exploration_model.created_on, last_updated=exploration_model.last_updated)

@overload
def get_exploration_summary_by_id(exploration_id: str) -> exp_domain.ExplorationSummary:
    if False:
        while True:
            i = 10
    ...

@overload
def get_exploration_summary_by_id(exploration_id: str, *, strict: Literal[True]) -> exp_domain.ExplorationSummary:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_exploration_summary_by_id(exploration_id: str, *, strict: Literal[False]) -> Optional[exp_domain.ExplorationSummary]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_exploration_summary_by_id(exploration_id: str, strict: bool=True) -> Optional[exp_domain.ExplorationSummary]:
    if False:
        return 10
    'Returns a domain object representing an exploration summary.\n\n    Args:\n        exploration_id: str. The id of the ExplorationSummary to be returned.\n        strict: bool. Whether to fail noisily if no exploration with a given id\n            exists.\n\n    Returns:\n        ExplorationSummary|None. The summary domain object corresponding to the\n        given exploration, and none if no ExpSummaryModel exists for given id.\n    '
    exp_summary_model = exp_models.ExpSummaryModel.get(exploration_id, strict=strict)
    if exp_summary_model:
        exp_summary = get_exploration_summary_from_model(exp_summary_model)
        return exp_summary
    else:
        return None

def get_exploration_summaries_from_models(exp_summary_models: Sequence[exp_models.ExpSummaryModel]) -> Dict[str, exp_domain.ExplorationSummary]:
    if False:
        i = 10
        return i + 15
    'Returns a dict with ExplorationSummary domain objects as values,\n    keyed by their exploration id.\n\n    Args:\n        exp_summary_models: list(ExplorationSummary). List of ExplorationSummary\n            model instances.\n\n    Returns:\n        dict. The keys are exploration ids and the values are the corresponding\n        ExplorationSummary domain objects.\n    '
    exploration_summaries = [get_exploration_summary_from_model(exp_summary_model) for exp_summary_model in exp_summary_models]
    result = {}
    for exp_summary in exploration_summaries:
        result[exp_summary.id] = exp_summary
    return result

def get_exploration_summary_from_model(exp_summary_model: exp_models.ExpSummaryModel) -> exp_domain.ExplorationSummary:
    if False:
        print('Hello World!')
    'Returns an ExplorationSummary domain object.\n\n    Args:\n        exp_summary_model: ExplorationSummary. An ExplorationSummary model\n            instance.\n\n    Returns:\n        ExplorationSummary. The summary domain object correspoding to the\n        given exploration summary model.\n    '
    return exp_domain.ExplorationSummary(exp_summary_model.id, exp_summary_model.title, exp_summary_model.category, exp_summary_model.objective, exp_summary_model.language_code, exp_summary_model.tags, exp_summary_model.ratings, exp_summary_model.scaled_average_rating, exp_summary_model.status, exp_summary_model.community_owned, exp_summary_model.owner_ids, exp_summary_model.editor_ids, exp_summary_model.voice_artist_ids, exp_summary_model.viewer_ids, exp_summary_model.contributor_ids, exp_summary_model.contributors_summary, exp_summary_model.version, exp_summary_model.exploration_model_created_on, exp_summary_model.exploration_model_last_updated, exp_summary_model.first_published_msec, exp_summary_model.deleted)

def get_exploration_summaries_matching_ids(exp_ids: List[str]) -> List[Optional[exp_domain.ExplorationSummary]]:
    if False:
        return 10
    'Returns a list of ExplorationSummary domain objects (or None if the\n    corresponding summary does not exist) corresponding to the given\n    list of exploration ids.\n\n    Args:\n        exp_ids: list(str). List of exploration ids.\n\n    Returns:\n        list(ExplorationSummary|None). List of ExplorationSummary domain objects\n        corresponding to the given exploration ids. If an ExplorationSummary\n        does not exist, the corresponding returned list element is None.\n    '
    return [get_exploration_summary_from_model(model) if model else None for model in exp_models.ExpSummaryModel.get_multi(exp_ids)]

def get_exploration_summaries_subscribed_to(user_id: str) -> List[exp_domain.ExplorationSummary]:
    if False:
        i = 10
        return i + 15
    'Returns a list of ExplorationSummary domain objects that the user\n    subscribes to.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(ExplorationSummary). List of ExplorationSummary domain objects that\n        the user subscribes to.\n    '
    return [summary for summary in get_exploration_summaries_matching_ids(subscription_services.get_exploration_ids_subscribed_to(user_id)) if summary is not None]

@overload
def get_exploration_by_id(exploration_id: str) -> exp_domain.Exploration:
    if False:
        print('Hello World!')
    ...

@overload
def get_exploration_by_id(exploration_id: str, *, version: Optional[int]=None) -> exp_domain.Exploration:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_exploration_by_id(exploration_id: str, *, strict: Literal[True], version: Optional[int]=None) -> exp_domain.Exploration:
    if False:
        while True:
            i = 10
    ...

@overload
def get_exploration_by_id(exploration_id: str, *, strict: Literal[False], version: Optional[int]=None) -> Optional[exp_domain.Exploration]:
    if False:
        print('Hello World!')
    ...

def get_exploration_by_id(exploration_id: str, strict: bool=True, version: Optional[int]=None) -> Optional[exp_domain.Exploration]:
    if False:
        print('Hello World!')
    'Returns an Exploration domain object.\n\n    Args:\n        exploration_id: str. The id of the exploration to be returned.\n        strict: bool. Whether to fail noisily if no exploration with a given id\n            exists.\n        version: int or None. The version of the exploration to be returned.\n            If None, the latest version of the exploration is returned.\n\n    Returns:\n        Exploration|None. The domain object corresponding to the given\n        exploration.\n    '
    sub_namespace = str(version) if version else None
    cached_exploration = caching_services.get_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, sub_namespace, [exploration_id]).get(exploration_id)
    if cached_exploration is not None:
        return cached_exploration
    else:
        exploration_model = exp_models.ExplorationModel.get(exploration_id, strict=strict, version=version)
        if exploration_model:
            exploration = get_exploration_from_model(exploration_model)
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, sub_namespace, {exploration_id: exploration})
            return exploration
        else:
            return None

def get_multiple_explorations_by_id(exp_ids: List[str], strict: bool=True) -> Dict[str, exp_domain.Exploration]:
    if False:
        return 10
    'Returns a dict of domain objects representing explorations with the\n    given ids as keys. If an exp_id is not present, it is not included in the\n    return dict.\n\n    Args:\n        exp_ids: list(str). List of ids of the exploration to be returned.\n        strict: bool. If True, a ValueError is raised when any exploration id\n            is invalid.\n\n    Returns:\n        dict. Maps exploration ids to the corresponding Exploration domain\n        objects. Any invalid exploration ids are omitted.\n\n    Raises:\n        ValueError. When strict is True and at least one of the given exp_ids\n            is invalid.\n    '
    result = {}
    uncached = []
    cache_result = caching_services.get_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, exp_ids)
    for exp_obj in cache_result.values():
        result[exp_obj.id] = exp_obj
    for _id in exp_ids:
        if _id not in result:
            uncached.append(_id)
    db_exp_models = exp_models.ExplorationModel.get_multi(uncached)
    db_results_dict = {}
    not_found = []
    for (i, eid) in enumerate(uncached):
        model = db_exp_models[i]
        if model:
            exploration = get_exploration_from_model(model)
            db_results_dict[eid] = exploration
        else:
            logging.info('Tried to fetch exploration with id %s, but no such exploration exists in the datastore' % eid)
            not_found.append(eid)
    if strict and not_found:
        raise ValueError("Couldn't find explorations with the following ids:\n%s" % '\n'.join(not_found))
    cache_update = {eid: results for (eid, results) in db_results_dict.items() if results is not None}
    if cache_update:
        caching_services.set_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, cache_update)
    result.update(db_results_dict)
    return result

def get_exploration_summaries_where_user_has_role(user_id: str) -> List[exp_domain.ExplorationSummary]:
    if False:
        return 10
    'Returns a list of ExplorationSummary domain objects where the user has\n    some role.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(ExplorationSummary). List of ExplorationSummary domain objects\n        where the user has some role.\n    '
    exp_summary_models: Sequence[exp_models.ExpSummaryModel] = exp_models.ExpSummaryModel.query(datastore_services.any_of(exp_models.ExpSummaryModel.owner_ids == user_id, exp_models.ExpSummaryModel.editor_ids == user_id, exp_models.ExpSummaryModel.voice_artist_ids == user_id, exp_models.ExpSummaryModel.viewer_ids == user_id, exp_models.ExpSummaryModel.contributor_ids == user_id)).fetch()
    return [get_exploration_summary_from_model(exp_summary_model) for exp_summary_model in exp_summary_models]

def get_exploration_user_data(user_id: str, exp_id: str) -> Optional[user_domain.ExplorationUserData]:
    if False:
        print('Hello World!')
    'Returns an ExplorationUserData domain object.\n\n    Args:\n        user_id: str. The Id of the user.\n        exp_id: str. The Id of the exploration.\n\n    Returns:\n        ExplorationUserData or None. The domain object corresponding to the\n        given user and exploration. If the model corresponsing to given user\n        and exploration is not found, return None.\n    '
    exp_user_data_model = user_models.ExplorationUserDataModel.get(user_id, exp_id)
    if exp_user_data_model is None:
        return None
    return user_domain.ExplorationUserData(exp_user_data_model.user_id, exp_user_data_model.exploration_id, exp_user_data_model.rating, exp_user_data_model.rated_on, exp_user_data_model.draft_change_list, exp_user_data_model.draft_change_list_last_updated, exp_user_data_model.draft_change_list_exp_version, exp_user_data_model.draft_change_list_id, exp_user_data_model.mute_suggestion_notifications, exp_user_data_model.mute_feedback_notifications, exp_user_data_model.furthest_reached_checkpoint_exp_version, exp_user_data_model.furthest_reached_checkpoint_state_name, exp_user_data_model.most_recently_reached_checkpoint_exp_version, exp_user_data_model.most_recently_reached_checkpoint_state_name)

@overload
def get_logged_out_user_progress(unique_progress_url_id: str, *, strict: Literal[True]) -> exp_domain.TransientCheckpointUrl:
    if False:
        print('Hello World!')
    ...

@overload
def get_logged_out_user_progress(unique_progress_url_id: str) -> Optional[exp_domain.TransientCheckpointUrl]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_logged_out_user_progress(unique_progress_url_id: str, *, strict: Literal[False]) -> Optional[exp_domain.TransientCheckpointUrl]:
    if False:
        return 10
    ...

@overload
def get_logged_out_user_progress(unique_progress_url_id: str, *, strict: bool) -> Optional[exp_domain.TransientCheckpointUrl]:
    if False:
        return 10
    ...

def get_logged_out_user_progress(unique_progress_url_id: str, strict: bool=False) -> Optional[exp_domain.TransientCheckpointUrl]:
    if False:
        i = 10
        return i + 15
    'Returns an TransientCheckpointUrl domain object.\n\n    Args:\n        unique_progress_url_id: str. The 6 digit long unique id\n            assigned to the progress made by a logged-out user.\n        strict: bool. Whether to fail noisily if no TransientCheckpointUrlModel\n            with the given unique_progress_url_id exists in the datastore.\n\n    Returns:\n        TransientCheckpointUrl or None. The domain object corresponding to the\n        given unique_progress_url_id. If the model corresponding to given\n        unique_progress_url_id is not found, return None.\n    '
    logged_out_user_progress_model = exp_models.TransientCheckpointUrlModel.get(unique_progress_url_id, strict=strict)
    if logged_out_user_progress_model is None:
        return None
    return exp_domain.TransientCheckpointUrl(logged_out_user_progress_model.exploration_id, logged_out_user_progress_model.furthest_reached_checkpoint_state_name, logged_out_user_progress_model.furthest_reached_checkpoint_exp_version, logged_out_user_progress_model.most_recently_reached_checkpoint_state_name, logged_out_user_progress_model.most_recently_reached_checkpoint_exp_version)

def get_exploration_version_history(exp_id: str, exp_version: int) -> Optional[exp_domain.ExplorationVersionHistory]:
    if False:
        while True:
            i = 10
    'Returns an ExplorationVersionHistory domain object by fetching the\n    ExplorationVersionHistoryModel for the given exploration id and version.\n\n    Args:\n        exp_id: str. The id of the exploration.\n        exp_version: int. The version number of the exploration.\n\n    Returns:\n        ExplorationVersionHistory. The exploration version history domain\n        object for the ExplorationVersionHistoryModel corresponding to the\n        given exploration id and version.\n    '
    version_history_model_id = exp_models.ExplorationVersionHistoryModel.get_instance_id(exp_id, exp_version)
    version_history_model = exp_models.ExplorationVersionHistoryModel.get(version_history_model_id, strict=False)
    if version_history_model is None:
        return None
    return exp_domain.ExplorationVersionHistory(exp_id, exp_version, version_history_model.state_version_history, version_history_model.metadata_last_edited_version_number, version_history_model.metadata_last_edited_committer_id, version_history_model.committer_ids)