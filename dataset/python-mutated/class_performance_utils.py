"""Module containing class performance condition utils."""
import typing as t
import numpy as np
import pandas as pd
from deepchecks.core import ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.strings import format_number, format_percent
__all__ = ['get_condition_test_performance_greater_than', 'get_condition_train_test_relative_degradation_less_than', 'get_condition_class_performance_imbalance_ratio_less_than']

def get_condition_test_performance_greater_than(min_score: float) -> t.Callable[[pd.DataFrame], ConditionResult]:
    if False:
        for i in range(10):
            print('nop')
    'Add condition - test metric scores are greater than the threshold.\n\n    Parameters\n    ----------\n    min_score : float\n        Minimum score to pass the check.\n\n    Returns\n    -------\n    Callable\n        the condition function\n    '

    def condition(check_result: pd.DataFrame):
        if False:
            while True:
                i = 10
        test_scores = check_result.loc[check_result['Dataset'] == 'Test']
        not_passed_test = test_scores.loc[test_scores['Value'] <= min_score]
        is_passed = len(not_passed_test) == 0
        has_classes = check_result.get('Class') is not None
        details = ''
        if not is_passed:
            details += f'Found {len(not_passed_test)} scores below threshold.\n'
        min_metric = test_scores.loc[test_scores['Value'].idxmin()]
        details += f"Found minimum score for {min_metric['Metric']} metric of value {format_number(min_metric['Value'])}"
        if has_classes:
            details += f" for class {min_metric.get('Class Name', min_metric['Class'])}."
        return ConditionResult(ConditionCategory.PASS if is_passed else ConditionCategory.FAIL, details)
    return condition

def get_condition_train_test_relative_degradation_less_than(threshold: float) -> t.Callable[[pd.DataFrame], ConditionResult]:
    if False:
        return 10
    'Add condition - test performance is not degraded by more than given percentage in train.\n\n    Parameters\n    ----------\n    threshold : float\n        maximum degradation ratio allowed (value between 0 and 1)\n\n    Returns\n    -------\n    Callable\n        the condition function\n    '

    def _ratio_of_change_calc(score_1, score_2):
        if False:
            print('Hello World!')
        if score_1 == 0:
            if score_2 == 0:
                return 0
            return 1
        return (score_1 - score_2) / abs(score_1)

    def condition(check_result: pd.DataFrame) -> ConditionResult:
        if False:
            print('Hello World!')
        test_scores = check_result.loc[check_result['Dataset'] == 'Test']
        train_scores = check_result.loc[check_result['Dataset'] == 'Train']
        max_degradation = ('', -np.inf)
        num_failures = 0

        def update_max_degradation(diffs, class_name):
            if False:
                i = 10
                return i + 15
            nonlocal max_degradation
            (max_scorer, max_diff) = get_dict_entry_by_value(diffs)
            if max_diff > max_degradation[1]:
                max_degradation_details = f'Found max degradation of {format_percent(max_diff)} for metric {max_scorer}'
                if class_name is not None:
                    max_degradation_details += f' and class {class_name}.'
                max_degradation = (max_degradation_details, max_diff)
        if 'Class' in check_result.columns and (not pd.isnull(check_result['Class']).all()):
            if 'Class Name' in check_result.columns and (not pd.isnull(check_result['Class Name']).all()):
                class_column = 'Class Name'
            else:
                class_column = 'Class'
            classes = check_result[class_column].unique()
        else:
            classes = None
        if classes is not None:
            for class_name in classes:
                test_scores_class = test_scores.loc[test_scores[class_column] == class_name]
                train_scores_class = train_scores.loc[train_scores[class_column] == class_name]
                test_scores_dict = dict(zip(test_scores_class['Metric'], test_scores_class['Value']))
                train_scores_dict = dict(zip(train_scores_class['Metric'], train_scores_class['Value']))
                if len(test_scores_dict) == 0 or len(train_scores_dict) == 0:
                    continue
                diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name]) for (score_name, score) in train_scores_dict.items()}
                update_max_degradation(diff, class_name)
                num_failures += len([v for v in diff.values() if v >= threshold])
        else:
            test_scores_dict = dict(zip(test_scores['Metric'], test_scores['Value']))
            train_scores_dict = dict(zip(train_scores['Metric'], train_scores['Value']))
            if not (len(test_scores_dict) == 0 or len(train_scores_dict) == 0):
                diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name]) for (score_name, score) in train_scores_dict.items()}
                update_max_degradation(diff, None)
                num_failures += len([v for v in diff.values() if v >= threshold])
        if num_failures > 0:
            message = f'{num_failures} scores failed. ' + max_degradation[0]
            return ConditionResult(ConditionCategory.FAIL, message)
        else:
            message = max_degradation[0]
            return ConditionResult(ConditionCategory.PASS, message)
    return condition

def get_condition_class_performance_imbalance_ratio_less_than(threshold: float, score: str) -> t.Callable[[pd.DataFrame], ConditionResult]:
    if False:
        print('Hello World!')
    'Add condition - relative ratio difference between highest-class and lowest-class is less than threshold.\n\n    Parameters\n    ----------\n    threshold : float\n        ratio difference threshold\n    score : str\n        limit score for condition\n\n    Returns\n    -------\n    Callable\n        the condition function\n    '

    def condition(check_result: pd.DataFrame) -> ConditionResult:
        if False:
            while True:
                i = 10
        if score not in set(check_result['Metric']):
            raise DeepchecksValueError(f'Data was not calculated using the scoring function: {score}')
        condition_passed = True
        datasets_details = []
        for dataset in ['Test', 'Train']:
            data = check_result.loc[(check_result['Dataset'] == dataset) & (check_result['Metric'] == score)]
            min_value_index = data['Value'].idxmin()
            min_row = data.loc[min_value_index]
            min_class_name = min_row.get('Class Name', min_row['Class'])
            min_value = min_row['Value']
            max_value_index = data['Value'].idxmax()
            max_row = data.loc[max_value_index]
            max_class_name = max_row.get('Class Name', max_row['Class'])
            max_value = max_row['Value']
            relative_difference = abs((min_value - max_value) / max_value)
            condition_passed &= relative_difference < threshold
            details = f'Relative ratio difference between highest and lowest in {dataset} dataset classes is {format_percent(relative_difference)}, using {score} metric. Lowest class - {min_class_name}: {format_number(min_value)}; Highest class - {max_class_name}: {format_number(max_value)}'
            datasets_details.append(details)
        category = ConditionCategory.PASS if condition_passed else ConditionCategory.FAIL
        return ConditionResult(category, details='\n'.join(datasets_details))
    return condition