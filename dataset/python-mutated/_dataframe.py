from __future__ import annotations
import collections
from typing import Any
from typing import DefaultDict
from typing import Set
import optuna
from optuna._imports import try_import
from optuna.trial._state import TrialState
with try_import() as _imports:
    import pandas as pd
if not _imports.is_successful():
    pd = object
__all__ = ['pd']

def _create_records_and_aggregate_column(study: 'optuna.Study', attrs: tuple[str, ...]) -> tuple[list[dict[tuple[str, str], Any]], list[tuple[str, str]]]:
    if False:
        return 10
    attrs_to_df_columns: dict[str, str] = {}
    for attr in attrs:
        if attr.startswith('_'):
            df_column = attr[1:]
        else:
            df_column = attr
        attrs_to_df_columns[attr] = df_column
    column_agg: DefaultDict[str, Set] = collections.defaultdict(set)
    non_nested_attr = ''
    metric_names = study.metric_names
    records = []
    for trial in study.get_trials(deepcopy=False):
        record = {}
        for (attr, df_column) in attrs_to_df_columns.items():
            value = getattr(trial, attr)
            if isinstance(value, TrialState):
                value = value.name
            if isinstance(value, dict):
                for (nested_attr, nested_value) in value.items():
                    record[df_column, nested_attr] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif attr == 'values':
                trial_values = [None] * len(study.directions) if value is None else value
                iterator = enumerate(trial_values) if metric_names is None else zip(metric_names, trial_values)
                for (nested_attr, nested_value) in iterator:
                    record[df_column, nested_attr] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif isinstance(value, list):
                for (nested_attr, nested_value) in enumerate(value):
                    record[df_column, nested_attr] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif attr == 'value':
                nested_attr = non_nested_attr if metric_names is None else metric_names[0]
                record[df_column, nested_attr] = value
                column_agg[attr].add((df_column, nested_attr))
            else:
                record[df_column, non_nested_attr] = value
                column_agg[attr].add((df_column, non_nested_attr))
        records.append(record)
    columns: list[tuple[str, str]] = sum((sorted(column_agg[k]) for k in attrs if k in column_agg), [])
    return (records, columns)

def _flatten_columns(columns: list[tuple[str, str]]) -> list[str]:
    if False:
        while True:
            i = 10
    return ['_'.join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns]

def _trials_dataframe(study: 'optuna.Study', attrs: tuple[str, ...], multi_index: bool) -> 'pd.DataFrame':
    if False:
        i = 10
        return i + 15
    _imports.check()
    if len(study.get_trials(deepcopy=False)) == 0:
        return pd.DataFrame()
    if 'value' in attrs and study._is_multi_objective():
        attrs = tuple(('values' if attr == 'value' else attr for attr in attrs))
    (records, columns) = _create_records_and_aggregate_column(study, attrs)
    df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))
    if not multi_index:
        df.columns = _flatten_columns(columns)
    return df