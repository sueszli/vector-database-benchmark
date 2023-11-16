"""
Forecasting Model Explainer Base Class

A `_ForecastingModelExplainer` takes a fitted forecasting model as input and generates explanations for it.
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union
from darts import TimeSeries
from darts.explainability.explainability_result import _ExplainabilityResult
from darts.explainability.utils import process_horizons_and_targets, process_input
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
logger = get_logger(__name__)
MIN_BACKGROUND_SAMPLE = 10

class _ForecastingModelExplainer(ABC):

    @abstractmethod
    def __init__(self, model: ForecastingModel, background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, background_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, background_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, requires_background: bool=False, requires_covariates_encoding: bool=False, check_component_names: bool=False, test_stationarity: bool=False):
        if False:
            while True:
                i = 10
        "\n        The base class for forecasting model explainers. It defines the *minimal* behavior that all\n        forecasting model explainers support.\n\n        Naming:\n\n        - A background series is a `TimeSeries` with which to 'train' the `Explainer` model.\n        - A foreground series is the `TimeSeries` to explain using the fitted `Explainer` model.\n\n        Parameters\n        ----------\n        model\n            A `ForecastingModel` to be explained. It must be fitted first.\n        background_series\n            A series or list of series to *train* the `_ForecastingModelExplainer` along with any foreground series.\n            Consider using a reduced well-chosen background to reduce computation time.\n            Optional if `model` was fit on a single target series. By default, it is the `series` used at fitting time.\n            Mandatory if `model` was fit on multiple (sequence of) target series.\n        background_past_covariates\n            A past covariates series or list of series that the model needs once fitted.\n        background_future_covariates\n            A future covariates series or list of series that the model needs once fitted.\n        requires_background\n            Whether the explainer requires background series as an input. If `True`, raises an error if no background\n            series were provided and `model` was fit using multiple series.\n        requires_covariates_encoding\n            Whether to apply the model's encoders to the input covariates. This should only be `True` if the\n            Explainer will not call model methods `fit()` or `predict()` directly.\n        check_component_names\n            Whether to enforce that, in the case of multiple time series, all series of the same type (target and/or\n            *_covariates) must have the same component names.\n        test_stationarity\n            Whether to raise a warning if not all `background_series` are stationary.\n        "
        if not model._fit_called:
            raise_log(ValueError(f'The model must be fitted before instantiating a {self.__class__.__name__}.'), logger)
        self.model = model
        self.n: Optional[int] = getattr(self.model, 'output_chunk_length', None)
        (self.background_series, self.background_past_covariates, self.background_future_covariates, self.target_components, self.static_covariates_components, self.past_covariates_components, self.future_covariates_components) = process_input(model=model, input_type='background', series=background_series, past_covariates=background_past_covariates, future_covariates=background_future_covariates, fallback_series=model.training_series, fallback_past_covariates=model.past_covariate_series, fallback_future_covariates=model.future_covariate_series, check_component_names=check_component_names, requires_input=requires_background, requires_covariates_encoding=requires_covariates_encoding, test_stationarity=test_stationarity)
        self.requires_foreground = self.background_series is None
        self.requires_covariates_encoding = requires_covariates_encoding
        self.check_component_names = check_component_names
        self.test_stationarity = test_stationarity

    @abstractmethod
    def explain(self, foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, horizons: Optional[Sequence[int]]=None, target_components: Optional[Sequence[str]]=None) -> _ExplainabilityResult:
        if False:
            print('Hello World!')
        '\n        Explains a foreground time series, and returns a :class:`_ExplainabilityResult\n        <darts.explainability.explainability_result._ExplainabilityResult>` that can be used for downstream tasks.\n\n        Parameters\n        ----------\n        foreground_series\n            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.\n            If not provided, the background `TimeSeries` will be explained instead.\n        foreground_past_covariates\n            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.\n        foreground_future_covariates\n            Optionally, one or a sequence of future covariates `TimeSeries` if required by the forecasting model.\n        horizons\n            Optionally, an integer or sequence of integers representing the future time steps to be explained.\n            `1` corresponds to the first timestamp being forecasted.\n            All values must be `<=output_chunk_length` of the explained forecasting model.\n        target_components\n            Optionally, a string or sequence of strings with the target components to explain.\n\n        Returns\n        -------\n        _ExplainabilityResult\n            The explainability result.\n        '
        pass

    def _process_foreground(self, foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None):
        if False:
            print('Hello World!')
        return process_input(model=self.model, input_type='foreground', series=foreground_series, past_covariates=foreground_past_covariates, future_covariates=foreground_future_covariates, fallback_series=self.background_series, fallback_past_covariates=self.background_past_covariates, fallback_future_covariates=self.background_future_covariates, check_component_names=self.check_component_names, requires_input=self.requires_foreground, requires_covariates_encoding=self.requires_covariates_encoding, test_stationarity=self.test_stationarity)

    def _process_horizons_and_targets(self, horizons: Optional[Union[int, Sequence[int]]], target_components: Optional[Union[str, Sequence[str]]]) -> Tuple[Sequence[int], Sequence[str]]:
        if False:
            for i in range(10):
                print('nop')
        return process_horizons_and_targets(horizons=horizons, fallback_horizon=self.n, target_components=target_components, fallback_target_components=self.target_components, check_component_names=self.check_component_names)