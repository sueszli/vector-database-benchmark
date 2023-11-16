"""The RegressionSystematicError check module."""
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number
__all__ = ['RegressionSystematicError']

class RegressionSystematicError(SingleDatasetCheck):
    """Check the regression systematic error.

    Parameters
    ----------
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self, n_samples: int=1000000, random_state: int=42, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('RegressionSystematicError check is deprecated and will be removed in future version, please use RegressionErrorDistribution check instead.', DeprecationWarning, stacklevel=2)
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        if False:
            return 10
        'Run check.\n\n        Returns\n        -------\n        CheckResult\n            value is a dict with rmse and mean prediction error.\n            display is box plot of the prediction error.\n\n        Raises\n        ------\n        DeepchecksValueError\n            If the object is not a Dataset instance with a label.\n        '
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        context.assert_regression_task()
        y_test = dataset.label_col
        x_test = dataset.features_columns
        y_pred = context.model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()
        if context.with_display:
            fig = go.Figure().add_trace(go.Box(x=diff, orientation='h', name='Model prediction error', hoverinfo='x', boxmean=True)).update_layout(title_text='Box plot of the model prediction error', height=500)
            display = ['Non-zero mean of the error distribution indicated the presents of systematic error in model predictions', fig]
        else:
            display = None
        return CheckResult(value={'rmse': rmse, 'mean_error': diff_mean}, display=display)

    def add_condition_systematic_error_ratio_to_rmse_less_than(self, max_ratio: float=0.01):
        if False:
            i = 10
            return i + 15
        'Add condition - require the absolute mean systematic error is less than (max_ratio * RMSE).\n\n        Parameters\n        ----------\n        max_ratio : float , default: 0.01\n            Maximum ratio\n        '

        def max_bias_condition(result: dict) -> ConditionResult:
            if False:
                print('Hello World!')
            rmse = result['rmse']
            mean_error = result['mean_error']
            ratio = abs(mean_error) / rmse
            details = f'Found bias ratio {format_number(ratio)}'
            category = ConditionCategory.PASS if ratio < max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)
        return self.add_condition(f'Bias ratio is less than {format_number(max_ratio)}', max_bias_condition)