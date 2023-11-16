"""Datasets size comparision check module."""
import typing as t
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
__all__ = ['DatasetsSizeComparison']
from deepchecks.utils.strings import format_number
T = t.TypeVar('T', bound='DatasetsSizeComparison')

class DatasetsSizeComparison(TrainTestCheck):
    """Verify test dataset size comparing it to the train dataset size."""

    def run_logic(self, context: Context) -> CheckResult:
        if False:
            while True:
                i = 10
        "Run check.\n\n        Returns\n        -------\n        CheckResult\n            with value of type pandas.DataFrame.\n            Value contains two keys, 'train' - size of the train dataset\n            and 'test' - size of the test dataset.\n\n        Raises\n        ------\n        DeepchecksValueError\n            if not dataset instances were provided.\n            if datasets are empty.\n        "
        train_dataset = context.train
        test_dataset = context.test
        sizes = {'Train': len(train_dataset), 'Test': len(test_dataset)}
        display = pd.DataFrame(sizes, index=['Size'])
        return CheckResult(value=sizes, display=display)

    def add_condition_test_size_greater_or_equal(self: T, value: int=100) -> T:
        if False:
            return 10
        'Add condition verifying that size of the test dataset is greater or equal to threshold.\n\n        Parameters\n        ----------\n        value : int , default: 100\n            minimal allowed test dataset size.\n\n        Returns\n        -------\n        Self\n            current instance of the DatasetsSizeComparison check.\n        '

        def condition(check_result: dict) -> ConditionResult:
            if False:
                print('Hello World!')
            details = f"Test dataset contains {check_result['Test']} samples"
            category = ConditionCategory.FAIL if check_result['Test'] <= value else ConditionCategory.PASS
            return ConditionResult(category, details)
        return self.add_condition(name=f'Test dataset size is greater or equal to {value}', condition_func=condition)

    def add_condition_test_train_size_ratio_greater_than(self: T, ratio: float=0.01) -> T:
        if False:
            i = 10
            return i + 15
        'Add condition verifying that test-train size ratio is greater than threshold.\n\n        Parameters\n        ----------\n        ratio : float , default: 0.01\n            minimal allowed test-train ratio.\n\n        Returns\n        -------\n        Self\n            current instance of the DatasetsSizeComparison check.\n        '

        def condition(check_result: dict) -> ConditionResult:
            if False:
                i = 10
                return i + 15
            test_train_ratio = check_result['Test'] / check_result['Train']
            details = f'Test-Train size ratio is {format_number(test_train_ratio)}'
            category = ConditionCategory.PASS if test_train_ratio > ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)
        return self.add_condition(name=f'Test-Train size ratio is greater than {ratio}', condition_func=condition)

    def add_condition_train_dataset_greater_or_equal_test(self: T) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Add condition verifying that train dataset is greater than test dataset.\n\n        Returns\n        -------\n        Self\n            current instance of the DatasetsSizeComparison check.\n        '

        def condition(check_result: dict) -> ConditionResult:
            if False:
                while True:
                    i = 10
            diff = check_result['Train'] - check_result['Test']
            if diff < 0:
                details = f'Train dataset is smaller than test dataset by {diff} samples'
                category = ConditionCategory.FAIL
            elif diff == 0:
                details = f"Train and test datasets both have {check_result['Train']} samples"
                category = ConditionCategory.PASS
            else:
                details = f'Train dataset is larger than test dataset by +{diff} samples'
                category = ConditionCategory.PASS
            return ConditionResult(category, details)
        return self.add_condition(name='Train dataset is greater or equal to test dataset', condition_func=condition)