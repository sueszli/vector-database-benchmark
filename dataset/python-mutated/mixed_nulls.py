"""Module contains Mixed Nulls check."""
import math
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.utils.feature_importance import N_TOP_MESSAGE
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_percent, string_baseform
from deepchecks.utils.typing import Hashable
__all__ = ['MixedNulls']
DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', '\x00', '\x00\x00'}

@docstrings
class MixedNulls(SingleDatasetCheck, ReduceFeatureMixin):
    """Search for various types of null values, including string representations of null.

    Parameters
    ----------
    null_string_list : Iterable[str] , default: None
        List of strings to be considered alternative null representations
    check_nan : bool , default: True
        Whether to add to null list to check also NaN values
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    aggregation_method: t.Optional[str], default: 'max'
        {feature_aggregation_method_argument:2*indent}
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self, null_string_list: Iterable[str]=None, check_nan: bool=True, columns: Union[Hashable, List[Hashable], None]=None, ignore_columns: Union[Hashable, List[Hashable], None]=None, n_top_columns: int=10, aggregation_method: Optional[str]='max', n_samples: int=10000000, random_state: int=42, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.null_string_list = null_string_list
        self.check_nan = check_nan
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns
        self.aggregation_method = aggregation_method
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        if False:
            for i in range(10):
                print('nop')
        "Run check.\n\n        Returns\n        -------\n        CheckResult\n            Value is dict with columns as key, and dict of null values as value:\n            {column: {null_value: {count: x, percent: y}, ...}, ...}\n            display is DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column that\n            has more than 1 null values.\n        "
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        df = dataset.data
        df = select_from_dataframe(df, self.columns, self.ignore_columns)
        null_string_list = self._validate_null_string_list(self.null_string_list)
        feature_importance = context.feature_importance if context.feature_importance is not None else pd.Series(index=list(df.columns), dtype=object)
        display_array = []
        result_dict = {'n_samples': len(df), 'columns': {}, 'feature_importance': feature_importance}
        for column_name in list(df.columns):
            column_data = df[column_name]
            if is_categorical_dtype(column_data) is True:
                null_counts = {}
                for (value, count) in column_data.value_counts(dropna=False).to_dict().items():
                    if count > 0:
                        if pd.isna(value):
                            null_counts[nan_type(value)] = count
                        elif string_baseform(value) in null_string_list:
                            null_counts[repr(value).replace("'", '"')] = count
            else:
                string_null_counts = {repr(value).replace("'", '"'): count for (value, count) in column_data.value_counts(dropna=True).items() if string_baseform(value) in null_string_list}
                nan_data_counts = column_data[column_data.isna()].apply(nan_type).value_counts().to_dict()
                null_counts = {**string_null_counts, **nan_data_counts}
            result_dict['columns'][column_name] = {}
            for (null_value, count) in null_counts.items():
                percent = count / len(column_data)
                display_array.append([column_name, null_value, count, format_percent(percent)])
                result_dict['columns'][column_name][null_value] = {'count': count, 'percent': percent}
        if context.with_display and display_array:
            df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percent of data'])
            order = df_graph['Column Name'].value_counts(ascending=False).index[:self.n_top_columns]
            df_graph = df_graph.set_index(['Column Name', 'Value'])
            df_graph = df_graph.loc[order, :]
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph]
        else:
            display = None
        return CheckResult(result_dict, display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        'Return an aggregated drift score based on aggregation method defined.'
        feature_importance = check_result.value['feature_importance']
        if check_result.value['columns']:
            total_mixed_nulls = {column: [check_result.value['columns'][column][null_value]['count'] for null_value in check_result.value['columns'][column]] for column in check_result.value['columns']}
            total_mixed_nulls = {column: sum(total_mixed_nulls[column]) if len(total_mixed_nulls[column]) > 1 else 0 for column in total_mixed_nulls}
        else:
            total_mixed_nulls = {column: 0 for column in check_result.value['columns']}
        percent_mismatched = pd.Series({column: total_mixed_nulls[column] / check_result.value['n_samples'] for column in check_result.value['columns']})
        return self.feature_reduce(self.aggregation_method, percent_mismatched, feature_importance, 'Percent Mixed Nulls')

    def _validate_null_string_list(self, nsl) -> set:
        if False:
            return 10
        'Validate the object given is a list of strings. If null is given return default list of null values.\n\n        Parameters\n        ----------\n        nsl\n            Object to validate\n\n        Returns\n        -------\n        set\n            Returns list of null values as set object\n        '
        result: set
        if nsl:
            if not isinstance(nsl, Iterable):
                raise DeepchecksValueError('null_string_list must be an iterable')
            if len(nsl) == 0:
                raise DeepchecksValueError("null_string_list can't be empty list")
            if any((not isinstance(string, str) for string in nsl)):
                raise DeepchecksValueError("null_string_list must contain only items of type 'str'")
            result = set(nsl)
        else:
            result = set(DEFAULT_NULL_VALUES)
        return result

    def add_condition_different_nulls_less_equal_to(self, max_allowed_null_types: int=1):
        if False:
            return 10
        "Add condition - require column's number of different null values to be less or equal to threshold.\n\n        Parameters\n        ----------\n        max_allowed_null_types : int , default: 1\n            Number of different null value types which is the maximum allowed.\n        "

        def condition(result: Dict) -> ConditionResult:
            if False:
                print('Hello World!')
            not_passing_columns = [k for (k, v) in result['columns'].items() if len(v) > max_allowed_null_types]
            if not_passing_columns:
                details = f"Found {len(not_passing_columns)} out of {len(result['columns'])} columns with amount of null types above threshold: {not_passing_columns}"
                return ConditionResult(ConditionCategory.FAIL, details)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result['columns']))
        return self.add_condition(f'Number of different null types is less or equal to {max_allowed_null_types}', condition)

def nan_type(x):
    if False:
        print('Hello World!')
    if x is np.nan:
        return 'numpy.nan'
    elif x is pd.NA:
        return 'pandas.NA'
    elif x is pd.NaT:
        return 'pandas.NaT'
    elif isinstance(x, float) and math.isnan(x):
        return 'math.nan'
    return str(x)