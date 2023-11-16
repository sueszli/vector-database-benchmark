"""
Forecasting Anomaly Model
-------------------------

A ``ForecastingAnomalyModel`` wraps around a Darts forecasting model and one or several anomaly
scorer(s) to compute anomaly scores by comparing how actuals deviate from the model's forecasts.
"""
import inspect
from typing import Dict, Optional, Sequence, Union
import pandas as pd
from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _assert_same_length, _assert_timeseries, _to_list
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class ForecastingAnomalyModel(AnomalyModel):

    def __init__(self, model: ForecastingModel, scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]]):
        if False:
            return 10
        'Forecasting-based Anomaly Detection Model\n\n        The forecasting model may or may not be already fitted. The underlying assumption is that `model`\n        should be able to accurately forecast the series in the absence of anomalies. For this reason,\n        it is recommended to either provide a model that has already been fitted and evaluated to work\n        appropriately on a series without anomalies, or to ensure that a simple call to the :func:`fit()`\n        method of the model will be sufficient to train it to satisfactory performance on a series without anomalies.\n\n        Calling :func:`fit()` on the anomaly model will fit the underlying forecasting model only\n        if ``allow_model_training`` is set to ``True`` upon calling ``fit()``.\n        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.\n\n        Parameters\n        ----------\n        model\n            An instance of a Darts forecasting model.\n        scorer\n            One or multiple scorer(s) that will be used to compare the actual and predicted time series in order\n            to obtain an anomaly score ``TimeSeries``.\n            If a list of `N` scorers is given, the anomaly model will call each\n            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.\n        '
        raise_if_not(isinstance(model, ForecastingModel), f'Model must be a darts ForecastingModel not a {type(model)}.')
        self.model = model
        super().__init__(model=model, scorer=scorer)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, allow_model_training: bool=False, forecast_horizon: int=1, start: Union[pd.Timestamp, float, int]=0.5, num_samples: int=1, **model_fit_kwargs):
        if False:
            print('Hello World!')
        'Fit the underlying forecasting model (if applicable) and the fittable scorers, if any.\n\n        Train the model (if not already fitted and ``allow_model_training`` is set to True) and the\n        scorer(s) (if fittable) on the given time series.\n\n        Once the model is fitted, the series historical forecasts are computed,\n        representing what would have been forecasted by this model on the series.\n\n        The prediction and the series are then used to train the scorer(s).\n\n        Parameters\n        ----------\n        series\n            One or multiple (if the model supports it) target series to be\n            trained on (generally assumed to be anomaly-free).\n        past_covariates\n            Optional past-observed covariate series or sequence of series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            Optional future-known covariate series or sequence of series. This applies only if the model\n            supports future covariates.\n        allow_model_training\n            Boolean value that indicates if the forecasting model needs to be fitted on the given series.\n            If set to False, the model needs to be already fitted.\n            Default: False\n        forecast_horizon\n            The forecast horizon for the predictions.\n        start\n            The first point of time at which a prediction is computed for a future time.\n            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.\n            In the case of ``float``, the parameter will be treated as the proportion of the time series\n            that should lie before the first prediction point.\n            In the case of ``int``, the parameter will be treated as an integer index to the time index of\n            `series` that will be used as first prediction time.\n            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time\n            directly.\n            Default: 0.5\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for\n            deterministic models.\n        model_fit_kwargs\n            Parameters to be passed on to the forecast model ``fit()`` method.\n\n        Returns\n        -------\n        self\n            Fitted model\n        '
        raise_if_not(type(allow_model_training) is bool, f'`allow_model_training` must be Boolean, found type: {type(allow_model_training)}.')
        if not allow_model_training and (not self.scorers_are_trainable):
            logger.warning(f"The forecasting model {self.model.__class__.__name__} won't be trained" + ' because the parameter `allow_model_training` is set to False, and no scorer' + ' is fittable. ``.fit()`` method has no effect.')
            return
        list_series = _to_list(series)
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_series]), 'all input `series` must be of type Timeseries.')
        list_past_covariates = self._prepare_covariates(past_covariates, list_series, 'past')
        list_future_covariates = self._prepare_covariates(future_covariates, list_series, 'future')
        model_fit_kwargs['past_covariates'] = list_past_covariates
        model_fit_kwargs['future_covariates'] = list_future_covariates
        model_fit_kwargs = {k: v for (k, v) in model_fit_kwargs.items() if v}
        if allow_model_training:
            fit_signature_series = inspect.signature(self.model.fit).parameters['series'].annotation
            if 'Sequence[darts.timeseries.TimeSeries]' in str(fit_signature_series):
                self.model.fit(series=list_series, **model_fit_kwargs)
            else:
                raise_if_not(len(list_series) == 1, f'Forecasting model {self.model.__class__.__name__} only accepts a single time series' + ' for the training phase and not a sequence of multiple of time series.')
                self.model.fit(series=list_series[0], **model_fit_kwargs)
        else:
            raise_if_not(self.model._fit_called, f'Model {self.model.__class__.__name__} needs to be trained, consider training ' + 'it beforehand or setting ' + '`allow_model_training` to True (default: False). ' + 'The model will then be trained on the provided series.')
        if self.scorers_are_trainable:
            self._check_window_size(list_series, start)
            list_pred = []
            for (idx, series) in enumerate(list_series):
                if list_past_covariates is not None:
                    past_covariates = list_past_covariates[idx]
                if list_future_covariates is not None:
                    future_covariates = list_future_covariates[idx]
                list_pred.append(self._predict_with_forecasting(series, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, start=start, num_samples=num_samples))
        for scorer in self.scorers:
            if hasattr(scorer, 'fit'):
                scorer.fit_from_prediction(list_series, list_pred)
        return self

    def _prepare_covariates(self, covariates: Union[TimeSeries, Sequence[TimeSeries]], series: Sequence[TimeSeries], name_covariates: str) -> Sequence[TimeSeries]:
        if False:
            i = 10
            return i + 15
        'Convert `covariates` into Sequence, if not already, and checks if their length is equal to the one of `series`.\n\n        Parameters\n        ----------\n        covariates\n            Covariate ("future" or "past") of `series`.\n        series\n            The series to be trained on.\n        name_covariates\n            Internal parameter for error message, a string indicating if it is a "future" or "past" covariates.\n\n        Returns\n        -------\n        Sequence[TimeSeries]\n            Covariate time series\n        '
        if covariates is not None:
            list_covariates = _to_list(covariates)
            for covariates in list_covariates:
                _assert_timeseries(covariates, name_covariates + '_covariates input series')
            raise_if_not(len(list_covariates) == len(series), f'Number of {name_covariates}_covariates must match the number of given ' + f'series, found length {len(list_covariates)} and expected {len(series)}.')
        return list_covariates if covariates is not None else None

    def show_anomalies(self, series: TimeSeries, past_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, forecast_horizon: int=1, start: Union[pd.Timestamp, float, int]=0.5, num_samples: int=1, actual_anomalies: TimeSeries=None, names_of_scorers: Union[str, Sequence[str]]=None, title: str=None, metric: str=None):
        if False:
            i = 10
            return i + 15
        'Plot the results of the anomaly model.\n\n        Computes the score on the given series input and shows the different anomaly scores with respect to time.\n\n        The plot will be composed of the following:\n\n        - the series itself with the output of the forecasting model.\n        - the anomaly score for each scorer. The scorers with different windows will be separated.\n        - the actual anomalies, if given.\n\n        It is possible to:\n\n        - add a title to the figure with the parameter `title`\n        - give personalized names for the scorers with `names_of_scorers`\n        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR),\n            if the actual anomalies are provided.\n\n        Parameters\n        ----------\n        series\n            The series to visualize anomalies from.\n        past_covariates\n            An optional past-observed covariate series or sequence of series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            An optional future-known covariate series or sequence of series. This applies only if the model\n            supports future covariates.\n        forecast_horizon\n            The forecast horizon for the predictions.\n        start\n            The first point of time at which a prediction is computed for a future time.\n            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.\n            In the case of ``float``, the parameter will be treated as the proportion of the time series\n            that should lie before the first prediction point.\n            In the case of ``int``, the parameter will be treated as an integer index to the time index of\n            `series` that will be used as first prediction time.\n            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time\n            directly.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for\n            deterministic models.\n        actual_anomalies\n            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        names_of_scorers\n            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.\n        title\n            Title of the figure\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n        '
        if isinstance(series, Sequence):
            raise_if_not(len(series) == 1, f'`show_anomalies` expects one series, found a list of length {len(series)} as input.')
            series = series[0]
        raise_if_not(isinstance(series, TimeSeries), f'`show_anomalies` expects an input of type TimeSeries, found type: {type(series)}.')
        (anomaly_scores, model_output) = self.score(series, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, start=start, num_samples=num_samples, return_model_prediction=True)
        return self._show_anomalies(series, model_output=model_output, anomaly_scores=anomaly_scores, names_of_scorers=names_of_scorers, actual_anomalies=actual_anomalies, title=title, metric=metric)

    def score(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, forecast_horizon: int=1, start: Union[pd.Timestamp, float, int]=0.5, num_samples: int=1, return_model_prediction: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            for i in range(10):
                print('nop')
        'Compute anomaly score(s) for the given series.\n\n        Predicts the given target time series with the forecasting model, and applies the scorer(s)\n        on the prediction and the target input time series. Outputs the anomaly score of the given\n        input time series.\n\n        Parameters\n        ----------\n        series\n            The (sequence of) series to score on.\n        past_covariates\n            An optional past-observed covariate series or sequence of series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            An optional future-known covariate series or sequence of series. This applies only if the model\n            supports future covariates.\n        forecast_horizon\n            The forecast horizon for the predictions.\n        start\n            The first point of time at which a prediction is computed for a future time.\n            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.\n            In the case of ``float``, the parameter will be treated as the proportion of the time series\n            that should lie before the first prediction point.\n            In the case of ``int``, the parameter will be treated as an integer index to the time index of\n            `series` that will be used as first prediction time.\n            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time\n            directly. Default: 0.5\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for\n            deterministic models.\n        return_model_prediction\n            Boolean value indicating if the prediction of the model should be returned along the anomaly score\n            Default: False\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]\n            Anomaly scores series generated by the anomaly model scorers\n\n                - ``TimeSeries`` if `series` is a series, and the anomaly model contains one scorer.\n                - ``Sequence[TimeSeries]``\n\n                    * if `series` is a series, and the anomaly model contains multiple scorers,\n                      returns one series per scorer.\n                    * if `series` is a sequence, and the anomaly model contains one scorer,\n                      returns one series per series in the sequence.\n                - ``Sequence[Sequence[TimeSeries]]`` if `series` is a sequence, and the anomaly\n                  model contains multiple scorers. The outer sequence is over the series,\n                  and inner sequence is over the scorers.\n        '
        raise_if_not(type(return_model_prediction) is bool, f'`return_model_prediction` must be Boolean, found type: {type(return_model_prediction)}.')
        raise_if_not(self.model._fit_called, f'Model {self.model} has not been trained. Please call ``.fit()``.')
        list_series = _to_list(series)
        list_past_covariates = self._prepare_covariates(past_covariates, list_series, 'past')
        list_future_covariates = self._prepare_covariates(future_covariates, list_series, 'future')
        self._check_window_size(list_series, start)
        list_pred = []
        for (idx, s) in enumerate(list_series):
            if list_past_covariates is not None:
                past_covariates = list_past_covariates[idx]
            if list_future_covariates is not None:
                future_covariates = list_future_covariates[idx]
            list_pred.append(self._predict_with_forecasting(s, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, start=start, num_samples=num_samples))
        scores = list(zip(*[sc.score_from_prediction(list_series, list_pred) for sc in self.scorers]))
        if len(scores) == 1 and (not isinstance(series, Sequence)):
            scores = scores[0]
            if len(scores) == 1:
                scores = scores[0]
        if len(list_pred) == 1:
            list_pred = list_pred[0]
        if return_model_prediction:
            return (scores, list_pred)
        else:
            return scores

    def _check_window_size(self, series: Sequence[TimeSeries], start: Union[pd.Timestamp, float, int]):
        if False:
            print('Hello World!')
        'Checks if the parameters `window` of the scorers are smaller than the maximum window size allowed.\n        The maximum size allowed is equal to the output length of the .historical_forecast() applied on `series`.\n        It is defined by the parameter `start` and the series’ length.\n\n        Parameters\n        ----------\n        series\n            The series given to the .historical_forecast()\n        start\n            Parameter of the .historical_forecast(): first point of time at which a prediction is computed\n            for a future time.\n        '
        max_window = max((scorer.window for scorer in self.scorers))
        for s in series:
            max_possible_window = len(s.drop_before(s.get_timestamp_at_point(start))) + 1
            raise_if_not(max_window <= max_possible_window, f'Window size {max_window} is greater than the targeted series length {max_possible_window},' + f' must be lower or equal. Reduce window size, or reduce start value (start: {start}).')

    def _predict_with_forecasting(self, series: TimeSeries, past_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, forecast_horizon: int=1, start: Union[pd.Timestamp, float, int]=None, num_samples: int=1) -> TimeSeries:
        if False:
            return 10
        'Compute the historical forecasts that would have been obtained by this model on the `series`.\n\n        `retrain` is set to False if possible (this is not supported by all models). If set to True, it will always\n        re-train the model on the entire available history,\n\n        Parameters\n        ----------\n        series\n            The target time series to use to successively train and evaluate the historical forecasts.\n        past_covariates\n            An optional past-observed covariate series or sequence of series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            An optional future-known covariate series or sequence of series. This applies only if the model\n            supports future covariates.\n        forecast_horizon\n            The forecast horizon for the predictions\n        start\n            The first point of time at which a prediction is computed for a future time.\n            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.\n            In the case of ``float``, the parameter will be treated as the proportion of the time series\n            that should lie before the first prediction point.\n            In the case of ``int``, the parameter will be treated as an integer index to the time index of\n            `series` that will be used as first prediction time.\n            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time\n            directly.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for\n            deterministic models.\n\n        Returns\n        -------\n        TimeSeries\n            Single ``TimeSeries`` instance created from the last point of each individual forecast.\n        '
        if self.model._supports_non_retrainable_historical_forecasts:
            retrain = False
        else:
            retrain = True
        historical_forecasts_param = {'past_covariates': past_covariates, 'future_covariates': future_covariates, 'forecast_horizon': forecast_horizon, 'start': start, 'retrain': retrain, 'num_samples': num_samples, 'stride': 1, 'last_points_only': True, 'verbose': False}
        return self.model.historical_forecasts(series, **historical_forecasts_param)

    def eval_accuracy(self, actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]], series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, forecast_horizon: int=1, start: Union[pd.Timestamp, float, int]=0.5, num_samples: int=1, metric: str='AUC_ROC') -> Union[Dict[str, float], Dict[str, Sequence[float]], Sequence[Dict[str, float]], Sequence[Dict[str, Sequence[float]]]]:
        if False:
            i = 10
            return i + 15
        'Compute the accuracy of the anomaly scores computed by the model.\n\n        Predicts the `series` with the forecasting model, and applies the\n        scorer(s) on the predicted time series and the given target time series. Returns the\n        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).\n\n        Parameters\n        ----------\n        actual_anomalies\n            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        series\n            The (sequence of) series to predict anomalies on.\n        past_covariates\n            An optional past-observed covariate series or sequence of series. This applies only\n            if the model supports past covariates.\n        future_covariates\n            An optional future-known covariate series or sequence of series. This applies only\n            if the model supports future covariates.\n        forecast_horizon\n            The forecast horizon for the predictions.\n        start\n            The first point of time at which a prediction is computed for a future time.\n            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.\n            In the case of ``float``, the parameter will be treated as the proportion of the time series\n            that should lie before the first prediction point.\n            In the case of ``int``, the parameter will be treated as an integer index to the time index of\n            `series` that will be used as first prediction time.\n            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time\n            directly.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for\n            deterministic models.\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n\n        Returns\n        -------\n        Union[Dict[str, float], Dict[str, Sequence[float]], Sequence[Dict[str, float]],\n        Sequence[Dict[str, Sequence[float]]]]\n            Score for the time series.\n            A (sequence of) dictionary with the keys being the name of the scorers, and the values being the\n            metric results on the (sequence of) `series`. If the scorer treats every dimension independently\n            (by nature of the scorer or if its component_wise is set to True), the values of the dictionary\n            will be a Sequence containing the score for each dimension.\n        '
        list_actual_anomalies = _to_list(actual_anomalies)
        list_series = _to_list(series)
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_series]), 'all input `series` must be of type Timeseries.')
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_actual_anomalies]), 'all input `actual_anomalies` must be of type Timeseries.')
        _assert_same_length(list_actual_anomalies, list_series)
        self._check_univariate(list_actual_anomalies)
        list_anomaly_scores = self.score(series=list_series, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, start=start, num_samples=num_samples)
        acc_anomaly_scores = self._eval_accuracy_from_scores(list_actual_anomalies=list_actual_anomalies, list_anomaly_scores=list_anomaly_scores, metric=metric)
        if len(acc_anomaly_scores) == 1 and (not isinstance(series, Sequence)):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores