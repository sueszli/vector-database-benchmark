"""
Filtering Anomaly Model
-----------------------

A ``FilteringAnomalyModel`` wraps around a Darts filtering model and one or
several anomaly scorer(s) to compute anomaly scores
by comparing how actuals deviate from the model's predictions (filtered series).
"""
from typing import Dict, Sequence, Union
from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _assert_same_length, _to_list
from darts.logging import get_logger, raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class FilteringAnomalyModel(AnomalyModel):

    def __init__(self, model: FilteringModel, scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]]):
        if False:
            return 10
        'Filtering-based Anomaly Detection Model\n\n        The filtering model may or may not be already fitted. The underlying assumption is that this model\n        should be able to adequately filter the series in the absence of anomalies. For this reason,\n        it is recommended to either provide a model that has already been fitted and evaluated to work\n        appropriately on a series without anomalies, or to ensure that a simple call to the :func:`fit()`\n        function of the model will be sufficient to train it to satisfactory performance on series without anomalies.\n\n        Calling :func:`fit()` on the anomaly model will fit the underlying filtering model only\n        if ``allow_model_training`` is set to ``True`` upon calling ``fit()``.\n        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.\n\n        Parameters\n        ----------\n        filter\n            A filtering model from Darts that will be used to filter the actual time series\n        scorer\n            One or multiple scorer(s) that will be used to compare the actual and predicted time series in order\n            to obtain an anomaly score ``TimeSeries``.\n            If a list of `N` scorer is given, the anomaly model will call each\n            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.\n        '
        raise_if_not(isinstance(model, FilteringModel), f'`model` must be a darts.models.filtering not a {type(model)}.')
        self.filter = model
        super().__init__(model=model, scorer=scorer)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], allow_model_training: bool=False, **filter_fit_kwargs):
        if False:
            print('Hello World!')
        'Fit the underlying filtering model (if applicable) and the fittable scorers, if any.\n\n        Train the filter (if not already fitted and `allow_filter_training` is set to True)\n        and the scorer(s) on the given time series.\n\n        The filter model will be applied to the given series, and the results will be used\n        to train the scorer(s).\n\n        Parameters\n        ----------\n        series\n            The (sequence of) series to be trained on.\n        allow_model_training\n            Boolean value that indicates if the filtering model needs to be fitted on the given series.\n            If set to False, the model needs to be already fitted.\n            Default: False\n        filter_fit_kwargs\n            Parameters to be passed on to the filtering model ``fit()`` method.\n\n        Returns\n        -------\n        self\n            Fitted model\n        '
        raise_if_not(type(allow_model_training) is bool, f'`allow_filter_training` must be Boolean, found type: {type(allow_model_training)}.')
        if not allow_model_training and (not self.scorers_are_trainable):
            logger.warning(f'The filtering model {self.model.__class__.__name__} is not required to be trained' + ' because the parameter `allow_filter_training` is set to False, and no scorer' + ' fittable. The ``.fit()`` function has no effect.')
            return
        list_series = _to_list(series)
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_series]), 'all input `series` must be of type Timeseries.')
        if allow_model_training:
            if hasattr(self.filter, 'fit'):
                raise_if_not(len(list_series) == 1, f'Filter model {self.model.__class__.__name__} can only be fitted on a' + ' single time series, but multiple are provided.')
                self.filter.fit(list_series[0], **filter_fit_kwargs)
            else:
                raise ValueError('`allow_filter_training` was set to True, but the filter' + f' {self.model.__class__.__name__} has no fit() method.')
        else:
            pass
        if self.scorers_are_trainable:
            list_pred = [self.filter.filter(series) for series in list_series]
        for scorer in self.scorers:
            if hasattr(scorer, 'fit'):
                scorer.fit_from_prediction(list_series, list_pred)
        return self

    def show_anomalies(self, series: TimeSeries, actual_anomalies: TimeSeries=None, names_of_scorers: Union[str, Sequence[str]]=None, title: str=None, metric: str=None, **score_kwargs):
        if False:
            print('Hello World!')
        'Plot the results of the anomaly model.\n\n        Computes the score on the given series input and shows the different anomaly scores with respect to time.\n\n        The plot will be composed of the following:\n\n        - the series itself with the output of the filtering model\n        - the anomaly score of each scorer. The scorer with different windows will be separated.\n        - the actual anomalies, if given.\n\n        It is possible to:\n\n        - add a title to the figure with the parameter `title`\n        - give personalized names for the scorers with `names_of_scorers`\n        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies are given\n\n        Parameters\n        ----------\n        series\n            The series to visualize anomalies from.\n        actual_anomalies\n            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        names_of_scorers\n            Name of the scorers. Must be a list of length equal to the number of scorers in the anomaly_model.\n        title\n            Title of the figure\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n        score_kwargs\n            parameters for the `.score()` function\n        '
        if isinstance(series, Sequence):
            raise_if_not(len(series) == 1, f'`show_anomalies` expects one series, found a sequence of length {len(series)} as input.')
            series = series[0]
        (anomaly_scores, model_output) = self.score(series, return_model_prediction=True, **score_kwargs)
        return self._show_anomalies(series, model_output=model_output, anomaly_scores=anomaly_scores, names_of_scorers=names_of_scorers, actual_anomalies=actual_anomalies, title=title, metric=metric)

    def score(self, series: Union[TimeSeries, Sequence[TimeSeries]], return_model_prediction: bool=False, **filter_kwargs):
        if False:
            print('Hello World!')
        'Compute the anomaly score(s) for the given series.\n\n        Predicts the given target time series with the filtering model, and applies the scorer(s)\n        to compare the predicted (filtered) series and the provided series.\n\n        Outputs the anomaly score(s) of the provided time series.\n\n        Parameters\n        ----------\n        series\n            The (sequence of) series to score.\n        return_model_prediction\n            Boolean value indicating if the prediction of the model should be returned along the anomaly score\n            Default: False\n        filter_kwargs\n            parameters of the Darts `.filter()` filtering model\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]\n            Anomaly scores series generated by the anomaly model scorers\n\n                - ``TimeSeries`` if `series` is a series, and the anomaly model contains one scorer.\n                - ``Sequence[TimeSeries]``\n\n                    * If `series` is a series, and the anomaly model contains multiple scorers,\n                    returns one series per scorer.\n                    * If `series` is a sequence, and the anomaly model contains one scorer,\n                    returns one series per series in the sequence.\n                - ``Sequence[Sequence[TimeSeries]]`` if `series` is a sequence, and the anomaly\n                model contains multiple scorers.\n                The outer sequence is over the series, and inner sequence is over the scorers.\n        '
        raise_if_not(type(return_model_prediction) is bool, f'`return_model_prediction` must be Boolean, found type: {type(return_model_prediction)}.')
        list_series = _to_list(series)
        list_pred = [self.filter.filter(s, **filter_kwargs) for s in list_series]
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

    def eval_accuracy(self, actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]], series: Union[TimeSeries, Sequence[TimeSeries]], metric: str='AUC_ROC', **filter_kwargs) -> Union[Dict[str, float], Dict[str, Sequence[float]], Sequence[Dict[str, float]], Sequence[Dict[str, Sequence[float]]]]:
        if False:
            i = 10
            return i + 15
        'Compute the accuracy of the anomaly scores computed by the model.\n\n        Predicts the `series` with the filtering model, and applies the\n        scorer(s) on the filtered time series and the given target time series. Returns the\n        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).\n\n        Parameters\n        ----------\n        actual_anomalies\n            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        series\n            The (sequence of) series to predict anomalies on.\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n        filter_kwargs\n            parameters of the Darts `.filter()` filtering model\n\n        Returns\n        -------\n        Union[Dict[str, float], Dict[str, Sequence[float]], Sequence[Dict[str, float]],\n        Sequence[Dict[str, Sequence[float]]]]\n            Score for the time series.\n            A (sequence of) dictionary with the keys being the name of the scorers, and the values being the\n            metric results on the (sequence of) `series`. If the scorer treats every dimension independently\n            (by nature of the scorer or if its component_wise is set to True), the values of the dictionary\n            will be a Sequence containing the score for each dimension.\n        '
        (list_series, list_actual_anomalies) = (_to_list(series), _to_list(actual_anomalies))
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_series]), 'all input `series` must be of type Timeseries.')
        raise_if_not(all([isinstance(s, TimeSeries) for s in list_actual_anomalies]), 'all input `actual_anomalies` must be of type Timeseries.')
        _assert_same_length(list_series, list_actual_anomalies)
        self._check_univariate(list_actual_anomalies)
        list_anomaly_scores = self.score(series=list_series, **filter_kwargs)
        acc_anomaly_scores = self._eval_accuracy_from_scores(list_actual_anomalies=list_actual_anomalies, list_anomaly_scores=list_anomaly_scores, metric=metric)
        if len(acc_anomaly_scores) == 1 and (not isinstance(series, Sequence)):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores