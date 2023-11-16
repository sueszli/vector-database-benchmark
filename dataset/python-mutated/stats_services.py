"""Services for exploration-related statistics."""
from __future__ import annotations
import copy
import datetime
import itertools
import logging
from core import feconf
from core import utils
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import question_services
from core.domain import stats_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, Union, cast, overload
MYPY = False
if MYPY:
    from core.domain import state_domain
    from mypy_imports import base_models
    from mypy_imports import stats_models
    from mypy_imports import transaction_services
(base_models, stats_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.STATISTICS])
transaction_services = models.Registry.import_transaction_services()

@overload
def get_playthrough_models_by_ids(playthrough_ids: List[str], *, strict: Literal[True]) -> List[stats_models.PlaythroughModel]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_playthrough_models_by_ids(playthrough_ids: List[str]) -> List[Optional[stats_models.PlaythroughModel]]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_playthrough_models_by_ids(playthrough_ids: List[str], *, strict: Literal[False]) -> List[Optional[stats_models.PlaythroughModel]]:
    if False:
        return 10
    ...

def get_playthrough_models_by_ids(playthrough_ids: List[str], strict: bool=False) -> Sequence[Optional[stats_models.PlaythroughModel]]:
    if False:
        print('Hello World!')
    'Returns a list of playthrough models matching the IDs provided.\n\n    Args:\n        playthrough_ids: list(str). List of IDs to get playthrough models for.\n        strict: bool. Whether to fail noisily if no playthrough model exists\n            with a given ID exists in the datastore.\n\n    Returns:\n        list(PlaythroughModel|None). The list of playthrough models\n        corresponding to given ids.  If a PlaythroughModel does not exist,\n        the corresponding returned list element is None.\n\n    Raises:\n        Exception. No PlaythroughModel exists for the given playthrough_id.\n    '
    playthrough_models = stats_models.PlaythroughModel.get_multi(playthrough_ids)
    if strict:
        for (index, playthrough_model) in enumerate(playthrough_models):
            if playthrough_model is None:
                raise Exception('No PlaythroughModel exists for the playthrough_id: %s' % playthrough_ids[index])
    return playthrough_models

def _migrate_to_latest_issue_schema(exp_issue_dict: stats_domain.ExplorationIssueDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Holds the responsibility of performing a step-by-step sequential update\n    of an exploration issue dict based on its schema version. If the current\n    issue schema version changes (stats_models.CURRENT_ISSUE_SCHEMA_VERSION), a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        exp_issue_dict: dict. Dict representing the exploration issue.\n\n    Raises:\n        Exception. The issue_schema_version is invalid.\n    '
    issue_schema_version = exp_issue_dict['schema_version']
    if issue_schema_version is None or issue_schema_version < 1:
        issue_schema_version = 0
    if not 0 <= issue_schema_version <= stats_models.CURRENT_ISSUE_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d and unversioned issue schemas at present.' % stats_models.CURRENT_ISSUE_SCHEMA_VERSION)
    while issue_schema_version < stats_models.CURRENT_ISSUE_SCHEMA_VERSION:
        stats_domain.ExplorationIssue.update_exp_issue_from_model(exp_issue_dict)
        issue_schema_version += 1

def _migrate_to_latest_action_schema(learner_action_dict: stats_domain.LearnerActionDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Holds the responsibility of performing a step-by-step sequential update\n    of an learner action dict based on its schema version. If the current action\n    schema version changes (stats_models.CURRENT_ACTION_SCHEMA_VERSION), a new\n    conversion function must be added and some code appended to this function to\n    account for that new version.\n\n    Args:\n        learner_action_dict: dict. Dict representing the learner action.\n\n    Raises:\n        Exception. The action_schema_version is invalid.\n    '
    action_schema_version = learner_action_dict['schema_version']
    if action_schema_version is None or action_schema_version < 1:
        action_schema_version = 0
    if not 0 <= action_schema_version <= stats_models.CURRENT_ACTION_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d and unversioned action schemas at present.' % stats_models.CURRENT_ACTION_SCHEMA_VERSION)
    while action_schema_version < stats_models.CURRENT_ACTION_SCHEMA_VERSION:
        stats_domain.LearnerAction.update_learner_action_from_model(learner_action_dict)
        action_schema_version += 1

def get_exploration_stats(exp_id: str, exp_version: int) -> stats_domain.ExplorationStats:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the ExplorationStats domain instance.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n\n    Returns:\n        ExplorationStats. The exploration stats domain object.\n    '
    exploration_stats = get_exploration_stats_by_id(exp_id, exp_version)
    if exploration_stats is None:
        exploration_stats = stats_domain.ExplorationStats.create_default(exp_id, exp_version, {})
    return exploration_stats

@transaction_services.run_in_transaction_wrapper
def _update_stats_transactional(exp_id: str, exp_version: int, aggregated_stats: stats_domain.AggregatedStatsDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates ExplorationStatsModel according to the dict containing aggregated\n    stats. The model GET and PUT must be done in a transaction to avoid loss of\n    updates that come in rapid succession.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        aggregated_stats: dict. Dict representing an ExplorationStatsModel\n            instance with stats aggregated in the frontend.\n\n    Raises:\n        Exception. ExplorationStatsModel does not exist.\n    '
    exploration = exp_fetchers.get_exploration_by_id(exp_id)
    if exploration.version != exp_version:
        logging.error('Trying to update stats for version %s of exploration %s, but the current version is %s.' % (exp_version, exp_id, exploration.version))
        return
    exp_stats = get_exploration_stats_by_id(exp_id, exp_version)
    if exp_stats is None:
        raise Exception('ExplorationStatsModel id="%s.%s" does not exist' % (exp_id, exp_version))
    try:
        stats_domain.SessionStateStats.validate_aggregated_stats_dict(aggregated_stats)
    except utils.ValidationError as e:
        logging.exception('Aggregated stats validation failed: %s', e)
        return
    exp_stats.num_starts_v2 += aggregated_stats['num_starts']
    exp_stats.num_completions_v2 += aggregated_stats['num_completions']
    exp_stats.num_actual_starts_v2 += aggregated_stats['num_actual_starts']
    state_stats_mapping = aggregated_stats['state_stats_mapping']
    for (state_name, stats) in state_stats_mapping.items():
        if state_name not in exp_stats.state_stats_mapping:
            if state_name == 'undefined':
                return
            raise Exception('ExplorationStatsModel id="%s.%s": state_stats_mapping[%r] does not exist' % (exp_id, exp_version, state_name))
        exp_stats.state_stats_mapping[state_name].aggregate_from(stats_domain.SessionStateStats.from_dict(stats))
    save_stats_model(exp_stats)

def update_stats(exp_id: str, exp_version: int, aggregated_stats: stats_domain.AggregatedStatsDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates ExplorationStatsModel according to the dict containing aggregated\n    stats.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        aggregated_stats: dict. Dict representing an ExplorationStatsModel\n            instance with stats aggregated in the frontend.\n    '
    _update_stats_transactional(exp_id, exp_version, aggregated_stats)

def get_stats_for_new_exploration(exp_id: str, exp_version: int, state_names: List[str]) -> stats_domain.ExplorationStats:
    if False:
        for i in range(10):
            print('nop')
    'Creates ExplorationStatsModel for the freshly created exploration and\n    sets all initial values to zero.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        state_names: list(str). State names of the exploration.\n\n    Returns:\n        ExplorationStats. The newly created exploration stats object.\n    '
    state_stats_mapping = {state_name: stats_domain.StateStats.create_default() for state_name in state_names}
    exploration_stats = stats_domain.ExplorationStats.create_default(exp_id, exp_version, state_stats_mapping)
    return exploration_stats

def get_stats_for_new_exp_version(exp_id: str, exp_version: int, state_names: List[str], exp_versions_diff: Optional[exp_domain.ExplorationVersionsDiff], revert_to_version: Optional[int]) -> stats_domain.ExplorationStats:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the ExplorationStatsModel for the old exp_version and makes any\n    required changes to the structure of the model. Then, a new\n    ExplorationStatsModel is created for the new exp_version. Note: This\n    function does not save the newly created model, it returns it. Callers\n    should explicitly save the model if required.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        state_names: list(str). State names of the exploration.\n        exp_versions_diff: ExplorationVersionsDiff|None. The domain object for\n            the exploration versions difference, None if it is a revert.\n        revert_to_version: int|None. If the change is a revert, the version.\n            Otherwise, None.\n\n    Returns:\n        ExplorationStats. The newly created exploration stats object.\n    '
    old_exp_stats = None
    old_exp_version = exp_version - 1
    new_exp_version = exp_version
    exploration_stats = get_exploration_stats_by_id(exp_id, old_exp_version)
    if exploration_stats is None:
        return get_stats_for_new_exploration(exp_id, new_exp_version, state_names)
    if revert_to_version:
        old_exp_stats = get_exploration_stats_by_id(exp_id, revert_to_version)
    return advance_version_of_exp_stats(new_exp_version, exp_versions_diff, exploration_stats, old_exp_stats, revert_to_version)

def advance_version_of_exp_stats(exp_version: int, exp_versions_diff: Optional[exp_domain.ExplorationVersionsDiff], exp_stats: stats_domain.ExplorationStats, reverted_exp_stats: Optional[stats_domain.ExplorationStats], revert_to_version: Optional[int]) -> stats_domain.ExplorationStats:
    if False:
        i = 10
        return i + 15
    'Makes required changes to the structure of ExplorationStatsModel of an\n    old exp_version and a new ExplorationStatsModel is created for the new\n    exp_version. Note: This function does not save the newly created model, it\n    returns it. Callers should explicitly save the model if required.\n\n    Args:\n        exp_version: int. Version of the exploration.\n        exp_versions_diff: ExplorationVersionsDiff|None. The domain object for\n            the exploration versions difference, None if it is a revert.\n        exp_stats: ExplorationStats. The ExplorationStats model.\n        reverted_exp_stats: ExplorationStats|None. The reverted\n            ExplorationStats model.\n        revert_to_version: int|None. If the change is a revert, the version.\n            Otherwise, None.\n\n    Returns:\n        ExplorationStats. The newly created exploration stats object.\n\n    Raises:\n        Exception. ExplorationVersionsDiff cannot be None when the change\n            is not a revert.\n    '
    if revert_to_version:
        if reverted_exp_stats:
            exp_stats.num_starts_v2 = reverted_exp_stats.num_starts_v2
            exp_stats.num_actual_starts_v2 = reverted_exp_stats.num_actual_starts_v2
            exp_stats.num_completions_v2 = reverted_exp_stats.num_completions_v2
            exp_stats.state_stats_mapping = reverted_exp_stats.state_stats_mapping
        exp_stats.exp_version = exp_version
        return exp_stats
    new_state_name_stats_mapping = {}
    if exp_versions_diff is None:
        raise Exception('ExplorationVersionsDiff cannot be None when the change is not a revert.')
    unchanged_state_names = set(utils.compute_list_difference(list(exp_stats.state_stats_mapping.keys()), exp_versions_diff.deleted_state_names + list(exp_versions_diff.new_to_old_state_names.values())))
    for state_name in unchanged_state_names:
        new_state_name_stats_mapping[state_name] = exp_stats.state_stats_mapping[state_name].clone()
    for state_name in exp_versions_diff.new_to_old_state_names:
        old_state_name = exp_versions_diff.new_to_old_state_names[state_name]
        new_state_name_stats_mapping[state_name] = exp_stats.state_stats_mapping[old_state_name].clone()
    for state_name in exp_versions_diff.added_state_names:
        new_state_name_stats_mapping[state_name] = stats_domain.StateStats.create_default()
    exp_stats.state_stats_mapping = new_state_name_stats_mapping
    exp_stats.exp_version = exp_version
    return exp_stats

def assign_playthrough_to_corresponding_issue(playthrough: stats_domain.Playthrough, exp_issues: stats_domain.ExplorationIssues, issue_schema_version: int) -> bool:
    if False:
        print('Hello World!')
    'Stores the given playthrough as a new model into its corresponding\n    exploration issue. When the corresponding exploration issue does not\n    exist, a new one is created.\n\n    Args:\n        playthrough: Playthrough. The playthrough domain object.\n        exp_issues: ExplorationIssues. The exploration issues domain object.\n        issue_schema_version: int. The version of the issue schema.\n\n    Returns:\n        bool. Whether the playthrough was stored successfully.\n    '
    issue = _get_corresponding_exp_issue(playthrough, exp_issues, issue_schema_version)
    if len(issue.playthrough_ids) < feconf.MAX_PLAYTHROUGHS_FOR_ISSUE:
        issue.playthrough_ids.append(stats_models.PlaythroughModel.create(playthrough.exp_id, playthrough.exp_version, playthrough.issue_type, playthrough.issue_customization_args, [action.to_dict() for action in playthrough.actions]))
        return True
    return False

def _get_corresponding_exp_issue(playthrough: stats_domain.Playthrough, exp_issues: stats_domain.ExplorationIssues, issue_schema_version: int) -> stats_domain.ExplorationIssue:
    if False:
        for i in range(10):
            print('nop')
    'Returns the unique exploration issue model expected to own the given\n    playthrough. If it does not exist yet, then it will be created.\n\n    Args:\n        playthrough: Playthrough. The playthrough domain object.\n        exp_issues: ExplorationIssues. The exploration issues domain object\n            which manages each individual exploration issue.\n        issue_schema_version: int. The version of the issue schema.\n\n    Returns:\n        ExplorationIssue. The corresponding exploration issue.\n    '
    for issue in exp_issues.unresolved_issues:
        if issue.issue_type == playthrough.issue_type:
            issue_customization_args = issue.issue_customization_args
            identifying_arg = feconf.CUSTOMIZATION_ARG_WHICH_IDENTIFIES_ISSUE[issue.issue_type]
            if issue_customization_args[identifying_arg] == playthrough.issue_customization_args[identifying_arg]:
                return issue
    issue = stats_domain.ExplorationIssue(playthrough.issue_type, playthrough.issue_customization_args, [], issue_schema_version, is_valid=True)
    exp_issues.unresolved_issues.append(issue)
    return issue

def create_exp_issues_for_new_exploration(exp_id: str, exp_version: int) -> None:
    if False:
        print('Hello World!')
    'Creates the ExplorationIssuesModel instance for the exploration.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n    '
    stats_models.ExplorationIssuesModel.create(exp_id, exp_version, [])

def get_updated_exp_issues_models_for_new_exp_version(exploration: exp_domain.Exploration, exp_versions_diff: Optional[exp_domain.ExplorationVersionsDiff], revert_to_version: Optional[int]) -> List[base_models.BaseModel]:
    if False:
        return 10
    'Retrieves the ExplorationIssuesModel for the old exp_version and makes\n    any required changes to the structure of the model.\n\n    Note: This method does not perform put operations on the models. The caller\n    of this method must do so.\n\n    Args:\n        exploration: Exploration. Domain object for the exploration.\n        exp_versions_diff: ExplorationVersionsDiff|None. The domain object for\n            the exploration versions difference, None if it is a revert.\n        revert_to_version: int|None. If the change is a revert, the version.\n            Otherwise, None.\n\n    Raises:\n        Exception. ExplorationVersionsDiff cannot be None when the change\n            is not a revert.\n\n    Returns:\n        list(BaseModel). A list of model instances related to exploration\n        issues that were updated.\n    '
    models_to_put: List[base_models.BaseModel] = []
    exp_issues = get_exp_issues(exploration.id, exploration.version - 1, strict=False)
    if exp_issues is None:
        instance_id = stats_models.ExplorationIssuesModel.get_entity_id(exploration.id, exploration.version - 1)
        models_to_put.append(stats_models.ExplorationIssuesModel(id=instance_id, exp_id=exploration.id, exp_version=exploration.version, unresolved_issues=[]))
        return models_to_put
    if revert_to_version:
        old_exp_issues = get_exp_issues(exploration.id, revert_to_version)
        exp_issues.unresolved_issues = old_exp_issues.unresolved_issues
        exp_issues.exp_version = exploration.version + 1
        models_to_put.append(get_exp_issues_model_from_domain_object(exp_issues))
        return models_to_put
    if exp_versions_diff is None:
        raise Exception('ExplorationVersionsDiff cannot be None when the change is not a revert.')
    deleted_state_names = exp_versions_diff.deleted_state_names
    old_to_new_state_names = exp_versions_diff.old_to_new_state_names
    playthrough_ids = list(itertools.chain.from_iterable((issue.playthrough_ids for issue in exp_issues.unresolved_issues)))
    playthrough_models = get_playthrough_models_by_ids(playthrough_ids, strict=True)
    updated_playthrough_models = []
    for playthrough_model in playthrough_models:
        playthrough = get_playthrough_from_model(playthrough_model)
        if 'state_names' in playthrough.issue_customization_args:
            state_names = cast(List[str], playthrough.issue_customization_args['state_names']['value'])
            playthrough.issue_customization_args['state_names']['value'] = [state_name if state_name not in old_to_new_state_names else old_to_new_state_names[state_name] for state_name in state_names]
        if 'state_name' in playthrough.issue_customization_args:
            state_name = cast(str, playthrough.issue_customization_args['state_name']['value'])
            playthrough.issue_customization_args['state_name']['value'] = state_name if state_name not in old_to_new_state_names else old_to_new_state_names[state_name]
        for action in playthrough.actions:
            action_customization_args = action.action_customization_args
            if 'state_name' in action_customization_args:
                state_name = cast(str, action_customization_args['state_name']['value'])
                action_customization_args['state_name']['value'] = state_name if state_name not in old_to_new_state_names else old_to_new_state_names[state_name]
            if 'dest_state_name' in action_customization_args:
                dest_state_name = cast(str, action_customization_args['dest_state_name']['value'])
                action_customization_args['dest_state_name']['value'] = dest_state_name if dest_state_name not in old_to_new_state_names else old_to_new_state_names[dest_state_name]
        playthrough_model.issue_customization_args = playthrough.issue_customization_args
        playthrough_model.actions = [action.to_dict() for action in playthrough.actions]
        updated_playthrough_models.append(playthrough_model)
    models_to_put.extend(updated_playthrough_models)
    for exp_issue in exp_issues.unresolved_issues:
        if 'state_names' in exp_issue.issue_customization_args:
            state_names = cast(List[str], exp_issue.issue_customization_args['state_names']['value'])
            if any((name in deleted_state_names for name in state_names)):
                exp_issue.is_valid = False
            exp_issue.issue_customization_args['state_names']['value'] = [state_name if state_name not in old_to_new_state_names else old_to_new_state_names[state_name] for state_name in state_names]
        if 'state_name' in exp_issue.issue_customization_args:
            state_name = cast(str, exp_issue.issue_customization_args['state_name']['value'])
            if state_name in deleted_state_names:
                exp_issue.is_valid = False
            exp_issue.issue_customization_args['state_name']['value'] = state_name if state_name not in old_to_new_state_names else old_to_new_state_names[state_name]
    exp_issues.exp_version += 1
    models_to_put.append(get_exp_issues_model_from_domain_object(exp_issues))
    return models_to_put

@overload
def get_exp_issues(exp_id: str, exp_version: int) -> stats_domain.ExplorationIssues:
    if False:
        while True:
            i = 10
    ...

@overload
def get_exp_issues(exp_id: str, exp_version: int, *, strict: Literal[True]) -> stats_domain.ExplorationIssues:
    if False:
        while True:
            i = 10
    ...

@overload
def get_exp_issues(exp_id: str, exp_version: int, *, strict: Literal[False]) -> Optional[stats_domain.ExplorationIssues]:
    if False:
        print('Hello World!')
    ...

@overload
def get_exp_issues(exp_id: str, exp_version: int, *, strict: bool=...) -> Optional[stats_domain.ExplorationIssues]:
    if False:
        i = 10
        return i + 15
    ...

def get_exp_issues(exp_id: str, exp_version: int, strict: bool=True) -> Optional[stats_domain.ExplorationIssues]:
    if False:
        i = 10
        return i + 15
    "Retrieves the ExplorationIssues domain object.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        strict: bool. Fails noisily if the model doesn't exist.\n\n    Returns:\n        ExplorationIssues|None. The domain object for exploration issues or None\n        if the exp_id is invalid.\n\n    Raises:\n        Exception. No ExplorationIssues model found for the given exp_id.\n    "
    exp_issues_model = stats_models.ExplorationIssuesModel.get_model(exp_id, exp_version)
    if exp_issues_model is None:
        if not strict:
            return None
        raise Exception('No ExplorationIssues model found for the given exp_id: %s' % exp_id)
    return get_exp_issues_from_model(exp_issues_model)

def get_playthrough_by_id(playthrough_id: str) -> Optional[stats_domain.Playthrough]:
    if False:
        print('Hello World!')
    'Retrieves the Playthrough domain object.\n\n    Args:\n        playthrough_id: str. ID of the playthrough.\n\n    Returns:\n        Playthrough|None. The domain object for the playthrough or None if the\n        playthrough_id is invalid.\n    '
    playthrough_model = stats_models.PlaythroughModel.get(playthrough_id, strict=False)
    if playthrough_model is None:
        return None
    return get_playthrough_from_model(playthrough_model)

def get_exploration_stats_by_id(exp_id: str, exp_version: int) -> Optional[stats_domain.ExplorationStats]:
    if False:
        print('Hello World!')
    'Retrieves the ExplorationStats domain object.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n\n    Returns:\n        ExplorationStats|None. The domain object for exploration statistics, or\n        None if no ExplorationStatsModel exists for the given id.\n\n    Raises:\n        Exception. Entity for class ExplorationStatsModel with id not found.\n    '
    exploration_stats = None
    exploration_stats_model = stats_models.ExplorationStatsModel.get_model(exp_id, exp_version)
    if exploration_stats_model is not None:
        exploration_stats = get_exploration_stats_from_model(exploration_stats_model)
    return exploration_stats

def get_multiple_exploration_stats_by_version(exp_id: str, version_numbers: List[int]) -> List[Optional[stats_domain.ExplorationStats]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of ExplorationStats domain objects corresponding to the\n    specified versions.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        version_numbers: list(int). List of version numbers.\n\n    Returns:\n        list(ExplorationStats|None). List of ExplorationStats domain class\n        instances.\n    '
    exploration_stats = []
    exploration_stats_models = stats_models.ExplorationStatsModel.get_multi_versions(exp_id, version_numbers)
    for exploration_stats_model in exploration_stats_models:
        exploration_stats.append(None if exploration_stats_model is None else get_exploration_stats_from_model(exploration_stats_model))
    return exploration_stats

def get_exp_issues_from_model(exp_issues_model: stats_models.ExplorationIssuesModel) -> stats_domain.ExplorationIssues:
    if False:
        while True:
            i = 10
    'Gets an ExplorationIssues domain object from an ExplorationIssuesModel\n    instance.\n\n    Args:\n        exp_issues_model: ExplorationIssuesModel. Exploration issues model in\n            datastore.\n\n    Returns:\n        ExplorationIssues. The domain object for exploration issues.\n    '
    unresolved_issues = []
    for unresolved_issue_dict in exp_issues_model.unresolved_issues:
        unresolved_issue_dict_copy = copy.deepcopy(unresolved_issue_dict)
        _migrate_to_latest_issue_schema(unresolved_issue_dict_copy)
        unresolved_issues.append(stats_domain.ExplorationIssue.from_dict(unresolved_issue_dict_copy))
    return stats_domain.ExplorationIssues(exp_issues_model.exp_id, exp_issues_model.exp_version, unresolved_issues)

def get_exploration_stats_from_model(exploration_stats_model: stats_models.ExplorationStatsModel) -> stats_domain.ExplorationStats:
    if False:
        print('Hello World!')
    'Gets an ExplorationStats domain object from an ExplorationStatsModel\n    instance.\n\n    Args:\n        exploration_stats_model: ExplorationStatsModel. Exploration statistics\n            model in datastore.\n\n    Returns:\n        ExplorationStats. The domain object for exploration statistics.\n    '
    new_state_stats_mapping = {state_name: stats_domain.StateStats.from_dict(exploration_stats_model.state_stats_mapping[state_name]) for state_name in exploration_stats_model.state_stats_mapping}
    return stats_domain.ExplorationStats(exploration_stats_model.exp_id, exploration_stats_model.exp_version, exploration_stats_model.num_starts_v1, exploration_stats_model.num_starts_v2, exploration_stats_model.num_actual_starts_v1, exploration_stats_model.num_actual_starts_v2, exploration_stats_model.num_completions_v1, exploration_stats_model.num_completions_v2, new_state_stats_mapping)

def get_playthrough_from_model(playthrough_model: stats_models.PlaythroughModel) -> stats_domain.Playthrough:
    if False:
        while True:
            i = 10
    'Gets a PlaythroughModel domain object from a PlaythroughModel instance.\n\n    Args:\n        playthrough_model: PlaythroughModel. Playthrough model in datastore.\n\n    Returns:\n        Playthrough. The domain object for a playthrough.\n    '
    actions = []
    for action_dict in playthrough_model.actions:
        _migrate_to_latest_action_schema(action_dict)
        actions.append(stats_domain.LearnerAction.from_dict(action_dict))
    return stats_domain.Playthrough(playthrough_model.exp_id, playthrough_model.exp_version, playthrough_model.issue_type, playthrough_model.issue_customization_args, actions)

def get_state_stats_mapping(exploration_stats: stats_domain.ExplorationStats) -> Dict[str, Dict[str, int]]:
    if False:
        print('Hello World!')
    'Returns the state stats mapping of the given exploration stats.\n\n    Args:\n        exploration_stats: ExplorationStats. Exploration statistics domain\n            object.\n\n    Returns:\n        dict. The state stats mapping of the given exploration stats.\n    '
    new_state_stats_mapping = {state_name: exploration_stats.state_stats_mapping[state_name].to_dict() for state_name in exploration_stats.state_stats_mapping}
    return new_state_stats_mapping

def create_stats_model(exploration_stats: stats_domain.ExplorationStats) -> str:
    if False:
        return 10
    'Creates an ExplorationStatsModel in datastore given an ExplorationStats\n    domain object.\n\n    Args:\n        exploration_stats: ExplorationStats. The domain object for exploration\n            statistics.\n\n    Returns:\n        str. ID of the datastore instance for ExplorationStatsModel.\n    '
    new_state_stats_mapping = get_state_stats_mapping(exploration_stats)
    instance_id = stats_models.ExplorationStatsModel.create(exploration_stats.exp_id, exploration_stats.exp_version, exploration_stats.num_starts_v1, exploration_stats.num_starts_v2, exploration_stats.num_actual_starts_v1, exploration_stats.num_actual_starts_v2, exploration_stats.num_completions_v1, exploration_stats.num_completions_v2, new_state_stats_mapping)
    return instance_id

def save_stats_model(exploration_stats: stats_domain.ExplorationStats) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the ExplorationStatsModel datastore instance with the passed\n    ExplorationStats domain object.\n\n    Args:\n        exploration_stats: ExplorationStats. The exploration statistics domain\n            object.\n\n    Raises:\n        Exception. No exploration stats model exists for the given exp_id.\n    '
    new_state_stats_mapping = {state_name: exploration_stats.state_stats_mapping[state_name].to_dict() for state_name in exploration_stats.state_stats_mapping}
    exploration_stats_model = stats_models.ExplorationStatsModel.get_model(exploration_stats.exp_id, exploration_stats.exp_version)
    if exploration_stats_model is None:
        raise Exception('No exploration stats model exists for the given exp_id.')
    exploration_stats_model.num_starts_v1 = exploration_stats.num_starts_v1
    exploration_stats_model.num_starts_v2 = exploration_stats.num_starts_v2
    exploration_stats_model.num_actual_starts_v1 = exploration_stats.num_actual_starts_v1
    exploration_stats_model.num_actual_starts_v2 = exploration_stats.num_actual_starts_v2
    exploration_stats_model.num_completions_v1 = exploration_stats.num_completions_v1
    exploration_stats_model.num_completions_v2 = exploration_stats.num_completions_v2
    exploration_stats_model.state_stats_mapping = new_state_stats_mapping
    exploration_stats_model.update_timestamps()
    exploration_stats_model.put()

def get_exp_issues_model_from_domain_object(exp_issues: stats_domain.ExplorationIssues) -> stats_models.ExplorationIssuesModel:
    if False:
        print('Hello World!')
    'Creates a new ExplorationIssuesModel instance.\n\n    Args:\n        exp_issues: ExplorationIssues. The exploration issues domain object.\n\n    Returns:\n        ExplorationIssuesModel. The ExplorationIssuesModel.\n    '
    unresolved_issues_dicts = [unresolved_issue.to_dict() for unresolved_issue in exp_issues.unresolved_issues]
    instance_id = stats_models.ExplorationIssuesModel.get_entity_id(exp_issues.exp_id, exp_issues.exp_version)
    return stats_models.ExplorationIssuesModel(id=instance_id, exp_id=exp_issues.exp_id, exp_version=exp_issues.exp_version, unresolved_issues=unresolved_issues_dicts)

def save_exp_issues_model(exp_issues: stats_domain.ExplorationIssues) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the ExplorationIssuesModel datastore instance with the passed\n    ExplorationIssues domain object.\n\n    Args:\n        exp_issues: ExplorationIssues. The exploration issues domain object.\n    '

    @transaction_services.run_in_transaction_wrapper
    def _save_exp_issues_model_transactional() -> None:
        if False:
            return 10
        'Implementation to be run in a transaction.'
        exp_issues_model = stats_models.ExplorationIssuesModel.get_model(exp_issues.exp_id, exp_issues.exp_version)
        if exp_issues_model is None:
            raise Exception('No ExplorationIssuesModel exists for the given exploration id.')
        exp_issues_model.exp_version = exp_issues.exp_version
        exp_issues_model.unresolved_issues = [issue.to_dict() for issue in exp_issues.unresolved_issues]
        exp_issues_model.update_timestamps()
        exp_issues_model.put()
    _save_exp_issues_model_transactional()

def get_exploration_stats_multi(exp_version_references: List[exp_domain.ExpVersionReference]) -> List[stats_domain.ExplorationStats]:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the exploration stats for the given explorations.\n\n    Args:\n        exp_version_references: list(ExpVersionReference). List of exploration\n            version reference domain objects.\n\n    Returns:\n        list(ExplorationStats). The list of exploration stats domain objects.\n    '
    exploration_stats_models = stats_models.ExplorationStatsModel.get_multi_stats_models(exp_version_references)
    exploration_stats_list = []
    for (index, exploration_stats_model) in enumerate(exploration_stats_models):
        if exploration_stats_model is None:
            exploration_stats_list.append(stats_domain.ExplorationStats.create_default(exp_version_references[index].exp_id, exp_version_references[index].version, {}))
        else:
            exploration_stats_list.append(get_exploration_stats_from_model(exploration_stats_model))
    return exploration_stats_list

def delete_playthroughs_multi(playthrough_ids: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes multiple playthrough instances.\n\n    Args:\n        playthrough_ids: list(str). List of playthrough IDs to be deleted.\n    '

    @transaction_services.run_in_transaction_wrapper
    def _delete_playthroughs_multi_transactional() -> None:
        if False:
            for i in range(10):
                print('nop')
        'Implementation to be run in a transaction.'
        playthrough_models = get_playthrough_models_by_ids(playthrough_ids, strict=True)
        filtered_playthrough_models = []
        for playthrough_model in playthrough_models:
            filtered_playthrough_models.append(playthrough_model)
        stats_models.PlaythroughModel.delete_multi(filtered_playthrough_models)
    _delete_playthroughs_multi_transactional()

def record_answer(exploration_id: str, exploration_version: int, state_name: str, interaction_id: str, submitted_answer: stats_domain.SubmittedAnswer) -> None:
    if False:
        while True:
            i = 10
    'Record an answer by storing it to the corresponding StateAnswers entity.\n\n    Args:\n        exploration_id: str. The exploration ID.\n        exploration_version: int. The version of the exploration.\n        state_name: str. The name of the state.\n        interaction_id: str. The ID of the interaction.\n        submitted_answer: SubmittedAnswer. The submitted answer.\n    '
    record_answers(exploration_id, exploration_version, state_name, interaction_id, [submitted_answer])

def record_answers(exploration_id: str, exploration_version: int, state_name: str, interaction_id: str, submitted_answer_list: List[stats_domain.SubmittedAnswer]) -> None:
    if False:
        i = 10
        return i + 15
    'Optimally record a group of answers using an already loaded exploration.\n    The submitted_answer_list is a list of SubmittedAnswer domain objects.\n\n    Args:\n        exploration_id: str. The exploration ID.\n        exploration_version: int. The version of the exploration.\n        state_name: str. The name of the state.\n        interaction_id: str. The ID of the interaction.\n        submitted_answer_list: list(SubmittedAnswer). The list of answers to be\n            recorded.\n    '
    state_answers = stats_domain.StateAnswers(exploration_id, exploration_version, state_name, interaction_id, submitted_answer_list)
    for submitted_answer in submitted_answer_list:
        submitted_answer.validate()
    stats_models.StateAnswersModel.insert_submitted_answers(state_answers.exploration_id, state_answers.exploration_version, state_answers.state_name, state_answers.interaction_id, state_answers.get_submitted_answer_dict_list())

def get_state_answers(exploration_id: str, exploration_version: int, state_name: str) -> Optional[stats_domain.StateAnswers]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a StateAnswers object containing all answers associated with the\n    specified exploration state, or None if no such answers have yet been\n    submitted.\n\n    Args:\n        exploration_id: str. The exploration ID.\n        exploration_version: int. The version of the exploration to fetch\n            answers for.\n        state_name: str. The name of the state to fetch answers for.\n\n    Returns:\n        StateAnswers or None. A StateAnswers object containing all answers\n        associated with the state, or None if no such answers exist.\n    '
    state_answers_models = stats_models.StateAnswersModel.get_all_models(exploration_id, exploration_version, state_name)
    if state_answers_models:
        main_state_answers_model = state_answers_models[0]
        submitted_answer_dict_list = itertools.chain.from_iterable([state_answers_model.submitted_answer_list for state_answers_model in state_answers_models])
        return stats_domain.StateAnswers(exploration_id, exploration_version, state_name, main_state_answers_model.interaction_id, [stats_domain.SubmittedAnswer.from_dict(submitted_answer_dict) for submitted_answer_dict in submitted_answer_dict_list], schema_version=main_state_answers_model.schema_version)
    else:
        return None

def get_sample_answers(exploration_id: str, exploration_version: int, state_name: str) -> List[state_domain.AcceptableCorrectAnswerTypes]:
    if False:
        return 10
    'Fetches a list of sample answers that were submitted to the specified\n    exploration state (at the given version of the exploration).\n\n    Args:\n        exploration_id: str. The exploration ID.\n        exploration_version: int. The version of the exploration to fetch\n            answers for.\n        state_name: str. The name of the state to fetch answers for.\n\n    Returns:\n        list(*). A list of some sample raw answers. At most 100 answers are\n        returned.\n    '
    answers_model = stats_models.StateAnswersModel.get_master_model(exploration_id, exploration_version, state_name)
    if answers_model is None:
        return []
    sample_answers = answers_model.submitted_answer_list[:100]
    return [stats_domain.SubmittedAnswer.from_dict(submitted_answer_dict).answer for submitted_answer_dict in sample_answers]

def get_state_reference_for_exploration(exp_id: str, state_name: str) -> str:
    if False:
        return 10
    'Returns the generated state reference for the given exploration id and\n    state name.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        state_name: str. Name of the state.\n\n    Returns:\n        str. The generated state reference.\n    '
    exploration = exp_fetchers.get_exploration_by_id(exp_id)
    if not exploration.has_state_name(state_name):
        raise utils.InvalidInputException('No state with the given state name was found in the exploration with id %s' % exp_id)
    return stats_models.LearnerAnswerDetailsModel.get_state_reference_for_exploration(exp_id, state_name)

def get_state_reference_for_question(question_id: str) -> str:
    if False:
        print('Hello World!')
    'Returns the generated state reference for the given question id.\n\n    Args:\n        question_id: str. ID of the question.\n\n    Returns:\n        str. The generated state reference.\n    '
    question = question_services.get_question_by_id(question_id, strict=False)
    if question is None:
        raise utils.InvalidInputException('No question with the given question id exists.')
    return stats_models.LearnerAnswerDetailsModel.get_state_reference_for_question(question_id)

def get_learner_answer_details_from_model(learner_answer_details_model: stats_models.LearnerAnswerDetailsModel) -> Optional[stats_domain.LearnerAnswerDetails]:
    if False:
        print('Hello World!')
    'Returns a LearnerAnswerDetails domain object given a\n    LearnerAnswerDetailsModel loaded from the datastore.\n\n    Args:\n        learner_answer_details_model: LearnerAnswerDetailsModel. The learner\n            answer details model loaded from the datastore.\n\n    Returns:\n        LearnerAnswerDetails|None. A LearnerAnswerDetails domain object\n        corresponding to the given model.\n    '
    return stats_domain.LearnerAnswerDetails(learner_answer_details_model.state_reference, learner_answer_details_model.entity_type, learner_answer_details_model.interaction_id, [stats_domain.LearnerAnswerInfo.from_dict(learner_answer_info_dict) for learner_answer_info_dict in learner_answer_details_model.learner_answer_info_list], learner_answer_details_model.learner_answer_info_schema_version, learner_answer_details_model.accumulated_answer_info_json_size_bytes)

def get_learner_answer_details(entity_type: str, state_reference: str) -> Optional[stats_domain.LearnerAnswerDetails]:
    if False:
        while True:
            i = 10
    "Returns a LearnerAnswerDetails domain object, with given entity_type and\n    state_name. This function checks in the datastore if the corresponding\n    LearnerAnswerDetailsModel exists, if not then None is returned.\n\n    Args:\n        entity_type: str. The type of entity i.e ENTITY_TYPE_EXPLORATION or\n            ENTITY_TYPE_QUESTION, which are declared in feconf.py.\n        state_reference: str. This is used to refer to a state in an exploration\n            or question. For an exploration the value will be equal to\n            'exp_id:state_name' and for question this will be equal to\n            'question_id'.\n\n    Returns:\n        Optional[LearnerAnswerDetails]. The learner answer domain object or\n        None if the model does not exist.\n    "
    learner_answer_details_model = stats_models.LearnerAnswerDetailsModel.get_model_instance(entity_type, state_reference)
    if learner_answer_details_model is not None:
        learner_answer_details = get_learner_answer_details_from_model(learner_answer_details_model)
        return learner_answer_details
    return None

def create_learner_answer_details_model_instance(learner_answer_details: stats_domain.LearnerAnswerDetails) -> None:
    if False:
        print('Hello World!')
    'Creates a new model instance from the given LearnerAnswerDetails domain\n    object.\n\n    Args:\n        learner_answer_details: LearnerAnswerDetails. The learner answer details\n            domain object.\n    '
    stats_models.LearnerAnswerDetailsModel.create_model_instance(learner_answer_details.entity_type, learner_answer_details.state_reference, learner_answer_details.interaction_id, learner_answer_details.learner_answer_info_list, learner_answer_details.learner_answer_info_schema_version, learner_answer_details.accumulated_answer_info_json_size_bytes)

def save_learner_answer_details(entity_type: str, state_reference: str, learner_answer_details: stats_domain.LearnerAnswerDetails) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Saves the LearnerAnswerDetails domain object in the datatstore, if the\n    model instance with the given entity_type and state_reference is found and\n    if the instance id of the model doesn't matches with the generated instance\n    id, then the earlier model is deleted and a new model instance is created.\n\n    Args:\n        entity_type: str. The type of entity i.e ENTITY_TYPE_EXPLORATION or\n            ENTITY_TYPE_QUESTION, which are declared in feconf.py.\n        state_reference: str. This is used to refer to a state in an exploration\n            or question. For an exploration the value will be equal to\n            'exp_id:state_name' and for question this will be equal to\n            'question_id'.\n        learner_answer_details: LearnerAnswerDetails. The learner answer details\n            domain object which is to be saved.\n    "
    learner_answer_details.validate()
    learner_answer_details_model = stats_models.LearnerAnswerDetailsModel.get_model_instance(entity_type, state_reference)
    if learner_answer_details_model is not None:
        instance_id = stats_models.LearnerAnswerDetailsModel.get_instance_id(learner_answer_details.entity_type, learner_answer_details.state_reference)
        if learner_answer_details_model.id == instance_id:
            learner_answer_details_model.learner_answer_info_list = [learner_answer_info.to_dict() for learner_answer_info in learner_answer_details.learner_answer_info_list]
            learner_answer_details_model.learner_answer_info_schema_version = learner_answer_details.learner_answer_info_schema_version
            learner_answer_details_model.accumulated_answer_info_json_size_bytes = learner_answer_details.accumulated_answer_info_json_size_bytes
            learner_answer_details_model.update_timestamps()
            learner_answer_details_model.put()
        else:
            learner_answer_details_model.delete()
            create_learner_answer_details_model_instance(learner_answer_details)
    else:
        create_learner_answer_details_model_instance(learner_answer_details)

def record_learner_answer_info(entity_type: str, state_reference: str, interaction_id: str, answer: Union[str, int, Dict[str, str], List[str]], answer_details: str) -> None:
    if False:
        i = 10
        return i + 15
    "Records the new learner answer info received from the learner in the\n    model and then saves it.\n\n    Args:\n        entity_type: str. The type of entity i.e ENTITY_TYPE_EXPLORATION or\n            ENTITY_TYPE_QUESTION, which are declared in feconf.py.\n        state_reference: str. This is used to refer to a state in an exploration\n            or question. For an exploration the value will be equal to\n            'exp_id:state_name' and for question this will be equal to\n            'question_id'.\n        interaction_id: str. The ID of the interaction.\n        answer: *(json-like). The answer which is submitted by the learner. The\n            actual type of answer depends on the interaction.\n        answer_details: str. The details the learner will submit when the\n            learner will be asked questions like 'Hey how did you land on this\n            answer', 'Why did you pick that answer' etc.\n    "
    learner_answer_details = get_learner_answer_details(entity_type, state_reference)
    if learner_answer_details is None:
        learner_answer_details = stats_domain.LearnerAnswerDetails(state_reference, entity_type, interaction_id, [], 0)
    learner_answer_info_id = stats_domain.LearnerAnswerInfo.get_new_learner_answer_info_id()
    learner_answer_info = stats_domain.LearnerAnswerInfo(learner_answer_info_id, answer, answer_details, datetime.datetime.utcnow())
    learner_answer_details.add_learner_answer_info(learner_answer_info)
    save_learner_answer_details(entity_type, state_reference, learner_answer_details)

def delete_learner_answer_info(entity_type: str, state_reference: str, learner_answer_info_id: str) -> None:
    if False:
        print('Hello World!')
    "Deletes the learner answer info in the model, and then saves it.\n\n    Args:\n        entity_type: str. The type of entity i.e ENTITY_TYPE_EXPLORATION or\n            ENTITY_TYPE_QUESTION, which are declared in feconf.py.\n        state_reference: str. This is used to refer to a state in an exploration\n            or question. For an exploration the value will be equal to\n            'exp_id:state_name' and for question this will be equal to\n            'question_id'.\n        learner_answer_info_id: str. The unique ID of the learner answer info\n            which needs to be deleted.\n    "
    learner_answer_details = get_learner_answer_details(entity_type, state_reference)
    if learner_answer_details is None:
        raise utils.InvalidInputException('No learner answer details found with the given state reference and entity')
    learner_answer_details.delete_learner_answer_info(learner_answer_info_id)
    save_learner_answer_details(entity_type, state_reference, learner_answer_details)

def update_state_reference(entity_type: str, old_state_reference: str, new_state_reference: str) -> None:
    if False:
        return 10
    'Updates the state_reference field of the LearnerAnswerDetails model\n    instance with the new_state_reference received and then saves the instance\n    in the datastore.\n\n    Args:\n        entity_type: str. The type of entity i.e ENTITY_TYPE_EXPLORATION or\n            ENTITY_TYPE_QUESTION, which are declared in feconf.py.\n        old_state_reference: str. The old state reference which needs to be\n            changed.\n        new_state_reference: str. The new state reference which needs to be\n            updated.\n    '
    learner_answer_details = get_learner_answer_details(entity_type, old_state_reference)
    if learner_answer_details is None:
        raise utils.InvalidInputException('No learner answer details found with the given state reference and entity')
    learner_answer_details.update_state_reference(new_state_reference)
    save_learner_answer_details(entity_type, old_state_reference, learner_answer_details)

def delete_learner_answer_details_for_exploration_state(exp_id: str, state_name: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the LearnerAnswerDetailsModel corresponding to the given\n    exploration ID and state name.\n\n    Args:\n        exp_id: str. The ID of the exploration.\n        state_name: str. The name of the state.\n    '
    state_reference = stats_models.LearnerAnswerDetailsModel.get_state_reference_for_exploration(exp_id, state_name)
    learner_answer_details_model = stats_models.LearnerAnswerDetailsModel.get_model_instance(feconf.ENTITY_TYPE_EXPLORATION, state_reference)
    if learner_answer_details_model is not None:
        learner_answer_details_model.delete()

def delete_learner_answer_details_for_question_state(question_id: str) -> None:
    if False:
        return 10
    'Deletes the LearnerAnswerDetailsModel for the given question ID.\n\n    Args:\n        question_id: str. The ID of the question.\n    '
    state_reference = stats_models.LearnerAnswerDetailsModel.get_state_reference_for_question(question_id)
    learner_answer_details_model = stats_models.LearnerAnswerDetailsModel.get_model_instance(feconf.ENTITY_TYPE_QUESTION, state_reference)
    if learner_answer_details_model is not None:
        learner_answer_details_model.delete()