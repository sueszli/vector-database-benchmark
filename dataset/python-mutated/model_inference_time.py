"""The model inference time check module."""
import timeit
import typing as t
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number
__all__ = ['ModelInferenceTime']
MI = t.TypeVar('MI', bound='ModelInferenceTime')

class ModelInferenceTime(SingleDatasetCheck):
    """Measure model average inference time (in seconds) per sample.

    Parameters
    ----------
    n_samples : int , default: 1_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self, n_samples: int=1000, random_state: int=42, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.random_state = random_state
        if n_samples == 0 or n_samples < 0:
            raise DeepchecksValueError('n_samples cannot be le than 0!')

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        if False:
            return 10
        "Run check.\n\n        Returns\n        -------\n        CheckResult\n            value is of the type 'float' .\n\n        Raises\n        ------\n        DeepchecksValueError\n            If the test dataset is not a 'Dataset' instance with a label or\n            if 'model' is not a scikit-learn-compatible fitted estimator instance.\n        "
        dataset = context.get_data_by_kind(dataset_kind)
        model = context.model
        df = dataset.features_columns
        prediction_method = model.predict
        n_samples = len(df) if len(df) < self.n_samples else self.n_samples
        df = df.sample(n=n_samples, random_state=self.random_state)
        result = timeit.timeit('predict(*args)', globals={'predict': prediction_method, 'args': (df,)}, number=1)
        result = result / n_samples
        return CheckResult(value=result, display=f'Average model inference time for one sample (in seconds): {format_number(result, floating_point=8)}')

    def add_condition_inference_time_less_than(self: MI, value: float=0.001) -> MI:
        if False:
            for i in range(10):
                print('nop')
        'Add condition - the average model inference time (in seconds) per sample is less than threshold.\n\n        Parameters\n        ----------\n        value : float , default: 0.001\n            condition threshold.\n        Returns\n        -------\n        MI\n        '

        def condition(average_time: float) -> ConditionResult:
            if False:
                return 10
            details = f'Found average inference time (seconds): {format_number(average_time, floating_point=8)}'
            category = ConditionCategory.PASS if average_time < value else ConditionCategory.FAIL
            return ConditionResult(category=category, details=details)
        return self.add_condition(condition_func=condition, name=f'Average model inference time for one sample is less than {format_number(value, floating_point=8)}')