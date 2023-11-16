"""The date_leakage check module."""
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_datetime, format_percent
__all__ = ['DateTrainTestLeakageDuplicates']

class DateTrainTestLeakageDuplicates(TrainTestCheck):
    """Check if test dates are present in train data.

    Parameters
    ----------
    n_to_show : int , default: 5
        Number of common dates to show.
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self, n_to_show: int=5, n_samples: int=10000000, random_state: int=42, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.n_to_show = n_to_show
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        if False:
            return 10
        "Run check.\n\n        Returns\n        -------\n        CheckResult\n            value is the ratio of date leakage.\n            data is html display of the checks' textual result.\n\n        Raises\n        ------\n        DeepchecksValueError\n            If one of the datasets is not a Dataset instance with an date\n        "
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        train_dataset.assert_datetime()
        train_date = train_dataset.datetime_col
        test_date = test_dataset.datetime_col
        date_intersection = tuple(set(train_date).intersection(test_date))
        if len(date_intersection) > 0:
            leakage_ratio = len([x for x in test_date if x in date_intersection]) / test_dataset.n_samples
            return_value = leakage_ratio
            if context.with_display:
                text = f'{format_percent(leakage_ratio)} of test data dates appear in training data'
                table = pd.DataFrame([[list((format_datetime(it) for it in date_intersection[:self.n_to_show]))]], index=['Sample of test dates in train:'], columns=['duplicate values'])
                display = [text, table]
            else:
                display = None
        else:
            display = None
            return_value = 0
        return CheckResult(value=return_value, header='Date Train-Test Leakage (duplicates)', display=display)

    def add_condition_leakage_ratio_less_or_equal(self, max_ratio: float=0):
        if False:
            while True:
                i = 10
        'Add condition - require leakage ratio to be less or equal to threshold.\n\n        Parameters\n        ----------\n        max_ratio : float , default: 0\n            Maximum ratio of leakage.\n        '

        def max_ratio_condition(result: float) -> ConditionResult:
            if False:
                i = 10
                return i + 15
            details = f'Found {format_percent(result)} leaked dates' if result > 0 else 'No leaked dates found'
            category = ConditionCategory.PASS if result <= max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)
        return self.add_condition(f'Date leakage ratio is less or equal to {format_percent(max_ratio)}', max_ratio_condition)