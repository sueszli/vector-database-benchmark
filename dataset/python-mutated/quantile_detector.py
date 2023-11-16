"""
Quantile Detector
-----------------

Flags anomalies that are beyond some quantiles of historical data.
This is similar to a threshold-based detector, where the thresholds are
computed as quantiles of historical data when the detector is fitted.
"""
from typing import Sequence, Union
import numpy as np
from darts.ad.detectors.detectors import FittableDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector
from darts.logging import raise_if, raise_if_not
from darts.timeseries import TimeSeries

class QuantileDetector(FittableDetector):

    def __init__(self, low_quantile: Union[Sequence[float], float, None]=None, high_quantile: Union[Sequence[float], float, None]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Flags values that are either\n        below or above the `low_quantile` and `high_quantile`\n        quantiles of historical data, respectively.\n\n        If a single value is provided for `low_quantile` or `high_quantile`, this same\n        value will be used across all components of the series.\n\n        If sequences of values are given for the parameters `low_quantile` and/or `high_quantile`,\n        they must be of the same length, matching the dimensionality of the series passed\n        to ``fit()``, or have a length of 1. In the latter case, this single value will be used\n        across all components of the series.\n\n        If either `low_quantile` or `high_quantile` is None, the corresponding bound will not be used.\n        However, at least one of the two must be set.\n\n        Parameters\n        ----------\n        low_quantile\n            (Sequence of) quantile of historical data below which a value is regarded as anomaly.\n            Must be between 0 and 1. If a sequence, must match the dimensionality of the series\n            this detector is applied to.\n        high_quantile\n            (Sequence of) quantile of historical data above which a value is regarded as anomaly.\n            Must be between 0 and 1. If a sequence, must match the dimensionality of the series\n            this detector is applied to.\n\n        Attributes\n        ----------\n        low_threshold\n            The (sequence of) lower quantile values.\n        high_threshold\n            The (sequence of) upper quantile values.\n        '
        super().__init__()
        raise_if(low_quantile is None and high_quantile is None, 'At least one parameter must be not None (`low` and `high` are both None).')

        def _prep_quantile(q):
            if False:
                print('Hello World!')
            return q.tolist() if isinstance(q, np.ndarray) else [q] if not isinstance(q, Sequence) else q
        low = _prep_quantile(low_quantile)
        high = _prep_quantile(high_quantile)
        for q in (low, high):
            raise_if_not(all([x is None or 0 <= x <= 1 for x in q]), 'Quantiles must be between 0 and 1, or None.')
        self.low_quantile = low * len(high) if len(low) == 1 else low
        self.high_quantile = high * len(low) if len(high) == 1 else high
        self.detector = None
        raise_if_not(len(self.low_quantile) == len(self.high_quantile), 'Parameters `low_quantile` and `high_quantile` must be of the same length,' + f' found `low`: {len(self.low_quantile)} and `high`: {len(self.high_quantile)}.')
        raise_if(all([lo is None for lo in self.low_quantile]) and all([hi is None for hi in self.high_quantile]), 'All provided quantile values are None.')
        raise_if_not(all([l <= h for (l, h) in zip(self.low_quantile, self.high_quantile) if l is not None and h is not None]), 'all values in `low_quantile` must be lower than or equal' + 'to their corresponding value in `high_quantile`.')

    def _fit_core(self, list_series: Sequence[TimeSeries]) -> None:
        if False:
            i = 10
            return i + 15
        raise_if(len(self.low_quantile) > 1 and len(self.low_quantile) != list_series[0].width, 'The number of components of input must be equal to the number' + ' of values given for `high_quantile` or/and `low_quantile`. Found number of ' + f'components equal to {list_series[0].width} and expected {len(self.low_quantile)}.')
        self.low_quantile = self.low_quantile * list_series[0].width if len(self.low_quantile) == 1 else self.low_quantile
        self.high_quantile = self.high_quantile * list_series[0].width if len(self.high_quantile) == 1 else self.high_quantile
        np_series = np.concatenate([series.all_values(copy=False) for series in list_series], axis=0)
        np_series = np.moveaxis(np_series, 2, 1)
        np_series = np_series.reshape(np_series.shape[0] * np_series.shape[1], -1)
        self.low_threshold = [np.quantile(np_series[:, i], q=lo, axis=0) if lo is not None else None for (i, lo) in enumerate(self.low_quantile)]
        self.high_threshold = [np.quantile(np_series[:, i], q=hi, axis=0) if hi is not None else None for (i, hi) in enumerate(self.high_quantile)]
        self.detector = ThresholdDetector(low_threshold=self.low_threshold, high_threshold=self.high_threshold)
        return self

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        return self.detector.detect(series)