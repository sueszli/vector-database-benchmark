"""
Scorers Base Classes
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union
import numpy as np
from darts import TimeSeries
from darts.ad.utils import _assert_same_length, _assert_timeseries, _intersect, _sanity_check_two_series, _to_list, eval_accuracy_from_scores, show_anomalies_from_scores
from darts.logging import get_logger, raise_if_not
logger = get_logger(__name__)

class AnomalyScorer(ABC):
    """Base class for all anomaly scorers"""

    def __init__(self, univariate_scorer: bool, window: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise_if_not(type(window) is int, f'Parameter `window` must be an integer, found type {type(window)}.')
        raise_if_not(window > 0, f'Parameter `window` must be stricly greater than 0, found size {window}.')
        self.window = window
        self.univariate_scorer = univariate_scorer

    def _check_univariate_scorer(self, actual_anomalies: Sequence[TimeSeries]):
        if False:
            print('Hello World!')
        "Checks if `actual_anomalies` contains only univariate series when the scorer has the\n        parameter 'univariate_scorer' set to True.\n\n        'univariate_scorer' is:\n            True -> when the function of the scorer ``score(series)`` (or, if applicable,\n                ``score_from_prediction(actual_series, pred_series)``) returns a univariate\n                anomaly score regardless of the input `series` (or, if applicable, `actual_series`\n                and `pred_series`).\n            False -> when the scorer will return a series that has the\n                same number of components as the input (can be univariate or multivariate).\n        "
        if self.univariate_scorer:
            raise_if_not(all([isinstance(s, TimeSeries) for s in actual_anomalies]), 'all series in `actual_anomalies` must be of type TimeSeries.')
            raise_if_not(all([s.width == 1 for s in actual_anomalies]), f'Scorer {self.__str__()} will return a univariate anomaly score series (width=1).' + ' Found a multivariate `actual_anomalies`.' + ' The evaluation of the accuracy cannot be computed between the two series.')

    def _check_window_size(self, series: TimeSeries):
        if False:
            while True:
                i = 10
        'Checks if the parameter window is less or equal than the length of the given series'
        raise_if_not(self.window <= len(series), f'Window size {self.window} is greater than the targeted series length {len(series)}, ' + 'must be lower or equal. Decrease the window size or increase the length series input' + ' to score on.')

    @property
    def is_probabilistic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether the scorer expects a probabilistic prediction for its first input.'
        return False

    def _assert_stochastic(self, series: TimeSeries, name_series: str):
        if False:
            print('Hello World!')
        'Checks if the series is stochastic (number of samples is higher than one).'
        raise_if_not(series.is_stochastic, f'Scorer {self.__str__()} is expecting `{name_series}` to be a stochastic timeseries' + f' (number of samples must be higher than 1, found: {series.n_samples}).')

    def _assert_deterministic(self, series: TimeSeries, name_series: str):
        if False:
            print('Hello World!')
        'Checks if the series is deterministic (number of samples is equal to one).'
        if not series.is_deterministic:
            logger.warning(f'Scorer {self.__str__()} is expecting `{name_series}` to be a (sequence of) deterministic' + f' timeseries (number of samples must be equal to 1, found: {series.n_samples}). The ' + 'series will be converted to a deterministic series by taking the median of the samples.')
            series = series.quantile_timeseries(quantile=0.5)
        return series

    @abstractmethod
    def __str__(self):
        if False:
            print('Hello World!')
        'returns the name of the scorer'
        pass

    def eval_accuracy_from_prediction(self, actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]], actual_series: Union[TimeSeries, Sequence[TimeSeries]], pred_series: Union[TimeSeries, Sequence[TimeSeries]], metric: str='AUC_ROC') -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        if False:
            print('Hello World!')
        'Computes the anomaly score between `actual_series` and `pred_series`, and returns the score\n        of an agnostic threshold metric.\n\n        Parameters\n        ----------\n        actual_anomalies\n            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        actual_series\n            The (sequence of) actual series.\n        pred_series\n            The (sequence of) predicted series.\n        metric\n            Optionally, metric function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n\n        Returns\n        -------\n        Union[float, Sequence[float], Sequence[Sequence[float]]]\n            Score of an agnostic threshold metric for the computed anomaly score\n                - ``float`` if `actual_series` and `actual_series` are univariate series (dimension=1).\n                - ``Sequence[float]``\n\n                    * if `actual_series` and `actual_series` are multivariate series (dimension>1),\n                    returns one value per dimension, or\n                    * if `actual_series` and `actual_series` are sequences of univariate series,\n                    returns one value per series\n                - ``Sequence[Sequence[float]]]`` if `actual_series` and `actual_series` are sequences\n                of multivariate series. Outer Sequence is over the sequence input and the inner\n                Sequence is over the dimensions of each element in the sequence input.\n        '
        actual_anomalies = _to_list(actual_anomalies)
        self._check_univariate_scorer(actual_anomalies)
        anomaly_score = self.score_from_prediction(actual_series, pred_series)
        return eval_accuracy_from_scores(actual_anomalies, anomaly_score, self.window, metric)

    @abstractmethod
    def score_from_prediction(self, actual_series: Any, pred_series: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        pass

    def show_anomalies_from_prediction(self, actual_series: TimeSeries, pred_series: TimeSeries, scorer_name: str=None, actual_anomalies: TimeSeries=None, title: str=None, metric: str=None):
        if False:
            print('Hello World!')
        'Plot the results of the scorer.\n\n        Computes the anomaly score on the two series. And plots the results.\n\n        The plot will be composed of the following:\n            - the actual_series and the pred_series.\n            - the anomaly score of the scorer.\n            - the actual anomalies, if given.\n\n        It is possible to:\n            - add a title to the figure with the parameter `title`\n            - give personalized name to the scorer with `scorer_name`\n            - show the results of a metric for the anomaly score (AUC_ROC or AUC_PR),\n              if the actual anomalies is provided.\n\n        Parameters\n        ----------\n        actual_series\n            The actual series to visualize anomalies from.\n        pred_series\n            The predicted series of `actual_series`.\n        actual_anomalies\n            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        scorer_name\n            Name of the scorer.\n        title\n            Title of the figure\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n        '
        if isinstance(actual_series, Sequence):
            raise_if_not(len(actual_series) == 1, '``show_anomalies_from_prediction`` expects only one series for `actual_series`,' + f' found a list of length {len(actual_series)} as input.')
            actual_series = actual_series[0]
        raise_if_not(isinstance(actual_series, TimeSeries), '``show_anomalies_from_prediction`` expects an input of type TimeSeries,' + f' found type {type(actual_series)} for `actual_series`.')
        if isinstance(pred_series, Sequence):
            raise_if_not(len(pred_series) == 1, '``show_anomalies_from_prediction`` expects one series for `pred_series`,' + f' found a list of length {len(pred_series)} as input.')
            pred_series = pred_series[0]
        raise_if_not(isinstance(pred_series, TimeSeries), '``show_anomalies_from_prediction`` expects an input of type TimeSeries,' + f' found type: {type(pred_series)} for `pred_series`.')
        anomaly_score = self.score_from_prediction(actual_series, pred_series)
        if title is None:
            title = f'Anomaly results by scorer {self.__str__()}'
        if scorer_name is None:
            scorer_name = [f'anomaly score by {self.__str__()}']
        return show_anomalies_from_scores(actual_series, model_output=pred_series, anomaly_scores=anomaly_score, window=self.window, names_of_scorers=scorer_name, actual_anomalies=actual_anomalies, title=title, metric=metric)

class NonFittableAnomalyScorer(AnomalyScorer):
    """Base class of anomaly scorers that do not need training."""

    def __init__(self, univariate_scorer, window) -> None:
        if False:
            print('Hello World!')
        super().__init__(univariate_scorer=univariate_scorer, window=window)
        self.trainable = False

    @abstractmethod
    def _score_core_from_prediction(self, series: Any) -> Any:
        if False:
            while True:
                i = 10
        pass

    def score_from_prediction(self, actual_series: Union[TimeSeries, Sequence[TimeSeries]], pred_series: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            while True:
                i = 10
        'Computes the anomaly score on the two (sequence of) series.\n\n        If a pair of sequences is given, they must contain the same number\n        of series. The scorer will score each pair of series independently\n        and return an anomaly score for each pair.\n\n        Parameters\n        ----------\n        actual_series:\n            The (sequence of) actual series.\n        pred_series\n            The (sequence of) predicted series.\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            (Sequence of) anomaly score time series\n        '
        (list_actual_series, list_pred_series) = (_to_list(actual_series), _to_list(pred_series))
        _assert_same_length(list_actual_series, list_pred_series)
        anomaly_scores = []
        for (s1, s2) in zip(list_actual_series, list_pred_series):
            _sanity_check_two_series(s1, s2)
            (s1, s2) = _intersect(s1, s2)
            self._check_window_size(s1)
            self._check_window_size(s2)
            anomaly_scores.append(self._score_core_from_prediction(s1, s2))
        if len(anomaly_scores) == 1 and (not isinstance(pred_series, Sequence)) and (not isinstance(actual_series, Sequence)):
            return anomaly_scores[0]
        else:
            return anomaly_scores

class FittableAnomalyScorer(AnomalyScorer):
    """Base class of scorers that do need training."""

    def __init__(self, univariate_scorer, window, diff_fn='abs_diff') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(univariate_scorer=univariate_scorer, window=window)
        self.trainable = True
        self._fit_called = False
        if diff_fn in {'abs_diff', 'diff'}:
            self.diff_fn = diff_fn
        else:
            raise ValueError(f"Metric should be 'diff' or 'abs_diff', found {diff_fn}")

    def check_if_fit_called(self):
        if False:
            print('Hello World!')
        'Checks if the scorer has been fitted before calling its `score()` function.'
        raise_if_not(self._fit_called, f'The Scorer {self.__str__()} has not been fitted yet. Call ``fit()`` first.')

    def eval_accuracy(self, actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]], series: Union[TimeSeries, Sequence[TimeSeries]], metric: str='AUC_ROC') -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        if False:
            while True:
                i = 10
        'Computes the anomaly score of the given time series, and returns the score\n        of an agnostic threshold metric.\n\n        Parameters\n        ----------\n        actual_anomalies\n            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        series\n            The (sequence of) series to detect anomalies from.\n        metric\n            Optionally, metric function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n\n        Returns\n        -------\n        Union[float, Sequence[float], Sequence[Sequence[float]]]\n            Score of an agnostic threshold metric for the computed anomaly score\n                - ``float`` if `series` is a univariate series (dimension=1).\n                - ``Sequence[float]``\n\n                    * if `series` is a multivariate series (dimension>1), returns one\n                    value per dimension, or\n                    * if `series` is a sequence of univariate series, returns one value\n                    per series\n                - ``Sequence[Sequence[float]]]`` if `series` is a sequence of multivariate\n                series. Outer Sequence is over the sequence input and the inner Sequence\n                is over the dimensions of each element in the sequence input.\n        '
        actual_anomalies = _to_list(actual_anomalies)
        self._check_univariate_scorer(actual_anomalies)
        anomaly_score = self.score(series)
        return eval_accuracy_from_scores(actual_anomalies, anomaly_score, self.window, metric)

    def score(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            for i in range(10):
                print('nop')
        'Computes the anomaly score on the given series.\n\n        If a sequence of series is given, the scorer will score each series independently\n        and return an anomaly score for each series in the sequence.\n\n        Parameters\n        ----------\n        series\n            The (sequence of) series to detect anomalies from.\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            (Sequence of) anomaly score time series\n        '
        self.check_if_fit_called()
        list_series = _to_list(series)
        anomaly_scores = []
        for s in list_series:
            _assert_timeseries(s)
            self._check_window_size(s)
            anomaly_scores.append(self._score_core(self._assert_deterministic(s, 'series')))
        if len(anomaly_scores) == 1 and (not isinstance(series, Sequence)):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def show_anomalies(self, series: TimeSeries, actual_anomalies: TimeSeries=None, scorer_name: str=None, title: str=None, metric: str=None):
        if False:
            print('Hello World!')
        'Plot the results of the scorer.\n\n        Computes the score on the given series input. And plots the results.\n\n        The plot will be composed of the following:\n            - the series itself.\n            - the anomaly score of the score.\n            - the actual anomalies, if given.\n\n        It is possible to:\n            - add a title to the figure with the parameter `title`\n            - give personalized name to the scorer with `scorer_name`\n            - show the results of a metric for the anomaly score (AUC_ROC or AUC_PR),\n            if the actual anomalies is provided.\n\n        Parameters\n        ----------\n        series\n            The series to visualize anomalies from.\n        actual_anomalies\n            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)\n        scorer_name\n            Name of the scorer.\n        title\n            Title of the figure\n        metric\n            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".\n            Default: "AUC_ROC"\n        '
        if isinstance(series, Sequence):
            raise_if_not(len(series) == 1, '``show_anomalies`` expects one series for `series`,' + f' found a list of length {len(series)} as input.')
            series = series[0]
        raise_if_not(isinstance(series, TimeSeries), '``show_anomalies`` expects an input of type TimeSeries,' + f' found type {type(series)} for `series`.')
        anomaly_score = self.score(series)
        if title is None:
            title = f'Anomaly results by scorer {self.__str__()}'
        if scorer_name is None:
            scorer_name = f'anomaly score by {self.__str__()}'
        return show_anomalies_from_scores(series, anomaly_scores=anomaly_score, window=self.window, names_of_scorers=scorer_name, actual_anomalies=actual_anomalies, title=title, metric=metric)

    def score_from_prediction(self, actual_series: Union[TimeSeries, Sequence[TimeSeries]], pred_series: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            for i in range(10):
                print('nop')
        'Computes the anomaly score on the two (sequence of) series.\n\n        The function ``diff_fn`` passed as a parameter to the scorer, will transform `pred_series` and `actual_series`\n        into one "difference" series. By default, ``diff_fn`` will compute the absolute difference\n        (Default: "abs_diff").\n        If actual_series and pred_series are sequences, ``diff_fn`` will be applied to all pairwise elements\n        of the sequences.\n\n        The scorer will then transform this series into an anomaly score. If a sequence of series is given,\n        the scorer will score each series independently and return an anomaly score for each series in the sequence.\n\n        Parameters\n        ----------\n        actual_series\n            The (sequence of) actual series.\n        pred_series\n            The (sequence of) predicted series.\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            (Sequence of) anomaly score time series\n        '
        self.check_if_fit_called()
        (list_actual_series, list_pred_series) = (_to_list(actual_series), _to_list(pred_series))
        _assert_same_length(list_actual_series, list_pred_series)
        anomaly_scores = []
        for (s1, s2) in zip(list_actual_series, list_pred_series):
            _sanity_check_two_series(s1, s2)
            s1 = self._assert_deterministic(s1, 'actual_series')
            s2 = self._assert_deterministic(s2, 'pred_series')
            diff = self._diff_series(s1, s2)
            self._check_window_size(diff)
            anomaly_scores.append(self.score(diff))
        if len(anomaly_scores) == 1 and (not isinstance(pred_series, Sequence)) and (not isinstance(actual_series, Sequence)):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        if False:
            i = 10
            return i + 15
        'Fits the scorer on the given time series input.\n\n        If sequence of series is given, the scorer will be fitted on the concatenation of the sequence.\n\n        The assumption is that the series `series` used for training are generally anomaly-free.\n\n        Parameters\n        ----------\n        series\n            The (sequence of) series with no anomalies.\n\n        Returns\n        -------\n        self\n            Fitted Scorer.\n        '
        list_series = _to_list(series)
        for (idx, s) in enumerate(list_series):
            _assert_timeseries(s)
            if idx == 0:
                self.width_trained_on = s.width
            else:
                raise_if_not(s.width == self.width_trained_on, 'series in `series` must have the same number of components,' + f' found number of components equal to {self.width_trained_on}' + f' at index 0 and {s.width} at index {idx}.')
            self._check_window_size(s)
            self._assert_deterministic(s, 'series')
        self._fit_core(list_series)
        self._fit_called = True

    def fit_from_prediction(self, actual_series: Union[TimeSeries, Sequence[TimeSeries]], pred_series: Union[TimeSeries, Sequence[TimeSeries]]):
        if False:
            for i in range(10):
                print('nop')
        'Fits the scorer on the two (sequence of) series.\n\n        The function ``diff_fn`` passed as a parameter to the scorer, will transform `pred_series` and `actual_series`\n        into one series. By default, ``diff_fn`` will compute the absolute difference (Default: "abs_diff").\n        If `pred_series` and `actual_series` are sequences, ``diff_fn`` will be applied to all pairwise elements\n        of the sequences.\n\n        The scorer will then be fitted on this (sequence of) series. If a sequence of series is given,\n        the scorer will be fitted on the concatenation of the sequence.\n\n        The scorer assumes that the (sequence of) actual_series is anomaly-free.\n\n        Parameters\n        ----------\n        actual_series\n            The (sequence of) actual series.\n        pred_series\n            The (sequence of) predicted series.\n\n        Returns\n        -------\n        self\n            Fitted Scorer.\n        '
        (list_actual_series, list_pred_series) = (_to_list(actual_series), _to_list(pred_series))
        _assert_same_length(list_actual_series, list_pred_series)
        list_fit_series = []
        for (s1, s2) in zip(list_actual_series, list_pred_series):
            _sanity_check_two_series(s1, s2)
            s1 = self._assert_deterministic(s1, 'actual_series')
            s2 = self._assert_deterministic(s2, 'pred_series')
            list_fit_series.append(self._diff_series(s1, s2))
        self.fit(list_fit_series)
        self._fit_called = True

    @abstractmethod
    def _fit_core(self, series: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def _score_core(self, series: Any) -> Any:
        if False:
            return 10
        pass

    def _diff_series(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        if False:
            print('Hello World!')
        'Applies the ``diff_fn`` to the two time series. Converts two time series into 1.\n\n        series_1 and series_2 must:\n            - have a non empty time intersection\n            - be of the same width W\n\n        Parameters\n        ----------\n        series_1\n            1st time series\n        series_2:\n            2nd time series\n\n        Returns\n        -------\n        TimeSeries\n            series of width W\n        '
        (series_1, series_2) = _intersect(series_1, series_2)
        if self.diff_fn == 'abs_diff':
            return (series_1 - series_2).map(lambda x: np.abs(x))
        elif self.diff_fn == 'diff':
            return series_1 - series_2
        else:
            raise ValueError(f"Metric should be 'diff' or 'abs_diff', found {self.diff_fn}")

class NLLScorer(NonFittableAnomalyScorer):
    """Parent class for all LikelihoodScorer"""

    def __init__(self, window) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(univariate_scorer=False, window=window)

    def _score_core_from_prediction(self, actual_series: TimeSeries, pred_series: TimeSeries) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        'For each timestamp of the inputs:\n            - the parameters of the considered distribution are fitted on the samples of the probabilistic time series\n            - the negative log-likelihood of the determinisitc time series values are computed\n\n        If the series is multivariate, the score will be computed on each component independently.\n\n        Parameters\n        ----------\n        actual_series:\n            A determinisict time series (number of samples per timestamp must be equal to 1)\n        pred_series\n            A probabilistic time series (number of samples per timestamp must be higher than 1)\n\n        Returns\n        -------\n        TimeSeries\n        '
        actual_series = self._assert_deterministic(actual_series, 'actual_series')
        self._assert_stochastic(pred_series, 'pred_series')
        np_actual_series = actual_series.all_values(copy=False)
        np_pred_series = pred_series.all_values(copy=False)
        np_anomaly_scores = []
        for component_idx in range(pred_series.width):
            np_anomaly_scores.append(self._score_core_nllikelihood(np_actual_series[:, component_idx].squeeze(-1), np_pred_series[:, component_idx]))
        anomaly_scores = TimeSeries.from_times_and_values(pred_series.time_index, list(zip(*np_anomaly_scores)))

        def _window_adjustment_series(series: TimeSeries) -> TimeSeries:
            if False:
                for i in range(10):
                    print('nop')
            'Slides a window of size self.window along the input series, and replaces the value of\n            the input time series by the mean of the values contained in the window (past self.window\n            points, including itself).\n            A series of length N will be transformed into a series of length N-self.window+1.\n            '
            if self.window == 1:
                return series
            else:
                return series.window_transform(transforms={'window': self.window, 'function': 'mean', 'mode': 'rolling', 'min_periods': self.window}, treat_na='dropna')
        return _window_adjustment_series(anomaly_scores)

    @property
    def is_probabilistic(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @abstractmethod
    def _score_core_nllikelihood(self, input_1: Any, input_2: Any) -> Any:
        if False:
            i = 10
            return i + 15
        'For each timestamp, the corresponding distribution is fitted on the probabilistic time-series\n        input_2, and returns the negative log-likelihood of the deterministic time-series input_1\n        given the distribution.\n        '
        pass