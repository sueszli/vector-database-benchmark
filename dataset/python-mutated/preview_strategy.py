from typing import Any, Dict, Sequence
from sentry.issues.grouptype import GroupCategory
from sentry.models.organization import Organization
from sentry.snuba.dataset import Dataset
from sentry.snuba.events import Columns
'\nIssue category specific components to computing a preview for a set of rules.\n\nTo add support for a new issue category/dataset:\n    1. Update get_dataset_from_category\n    2. Add mapping from Dataset to snuba column name\n        a. The column name should be a field in sentry.snuba.events.Column\n    3. Add category-specific query params for get_update_kwargs_for_groups and get_update_kwargs_for_group\n'

def get_dataset_from_category(category: int, organization: Organization) -> Dataset:
    if False:
        for i in range(10):
            print('nop')
    if category == GroupCategory.ERROR.value:
        return Dataset.Events
    return Dataset.IssuePlatform
DATASET_TO_COLUMN_NAME: Dict[Dataset, str] = {Dataset.Events: 'event_name', Dataset.Transactions: 'transaction_name', Dataset.IssuePlatform: 'issue_platform_name'}

def get_dataset_columns(columns: Sequence[Columns]) -> Dict[Dataset, Sequence[str]]:
    if False:
        for i in range(10):
            print('nop')
    dataset_columns: Dict[Dataset, Sequence[str]] = {}
    for (dataset, column_name) in DATASET_TO_COLUMN_NAME.items():
        dataset_columns[dataset] = [getattr(column.value, column_name) for column in columns if getattr(column.value, column_name) is not None]
    return dataset_columns

def _events_from_groups_kwargs(group_ids: Sequence[int], kwargs: Dict[str, Any], has_issue_state_condition: bool=True) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    if has_issue_state_condition:
        kwargs['conditions'] = [('group_id', 'IN', group_ids)]
    return kwargs

def _transactions_from_groups_kwargs(group_ids: Sequence[int], kwargs: Dict[str, Any], has_issue_state_condition: bool=True) -> Dict[str, Any]:
    if False:
        return 10
    if has_issue_state_condition:
        kwargs['having'] = [('group_id', 'IN', group_ids)]
        kwargs['conditions'] = [[['hasAny', ['group_ids', ['array', group_ids]]], '=', 1]]
    if 'aggregations' not in kwargs:
        kwargs['aggregations'] = []
    kwargs['aggregations'].append(('arrayJoin', ['group_ids'], 'group_id'))
    return kwargs
"\nReturns the rows that contain the group id.\nIf there's a many-to-many relationship, the group id column should be arrayjoined.\nIf there are no issue state changes (causes no group ids), then do not filter by group ids.\n"

def get_update_kwargs_for_groups(dataset: Dataset, group_ids: Sequence[int], kwargs: Dict[str, Any], has_issue_state_condition: bool=True) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    if dataset == Dataset.Transactions:
        return _transactions_from_groups_kwargs(group_ids, kwargs, has_issue_state_condition)
    return _events_from_groups_kwargs(group_ids, kwargs, has_issue_state_condition)

def _events_from_group_kwargs(group_id: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    kwargs['conditions'] = [('group_id', '=', group_id)]
    return kwargs

def _transactions_from_group_kwargs(group_id: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    kwargs['conditions'] = [[['has', ['group_ids', group_id]], '=', 1]]
    return kwargs
'\nReturns the rows that reference the group id without arrayjoining.\n'

def get_update_kwargs_for_group(dataset: Dataset, group_id: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    if dataset == Dataset.Transactions:
        return _transactions_from_group_kwargs(group_id, kwargs)
    return _events_from_group_kwargs(group_id, kwargs)