"""
Horizon-Based Training Dataset
------------------------------
"""
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from .training_dataset import PastCovariatesTrainingDataset
from .utils import CovariateType
logger = get_logger(__name__)

class HorizonBasedDataset(PastCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, output_chunk_length: int=12, lh: Tuple[int, int]=(1, 3), lookback: int=3, use_static_covariates: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, future_target)\n        arrays,\n        in a way inspired by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.\n\n        The "past" series have length `lookback * output_chunk_length`, and the "future" series has length\n        `output_chunk_length`.\n\n        Given the horizon `output_chunk_length` of a model, this dataset will compute some "past/future"\n        splits as follows:\n        First a "forecast point" is selected in the the range of the last\n        `(min_lh * output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.\n        The "future" then consists in the following `output_chunk_length` points, and the "past" will be the preceding\n        `lookback * output_chunk_length` points.\n\n        All the series in the provided sequence must be long enough; i.e. have length at least\n        `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be at least 1\n        (to have targets of length exactly `1 * output_chunk_length`).\n        The target and covariates time series are sliced together using their time indexes for alignment.\n\n        The sampling is uniform both over the number of time series and the number of samples per series;\n        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N\n        time series in the sequence.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates:\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        output_chunk_length\n            The length of the "output" series emitted by the model\n        lh\n            A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.\n            For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points\n            before the end of the series. It is required that `min_lh >= 1`.\n        lookback:\n            A integer interval for the length of the input in the emitted input and output splits, expressed as a\n            multiple of `output_chunk_length`. For instance, `lookback=3` will emit "inputs" of lengths\n            `3 * output_chunk_length`.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.covariate_type = CovariateType.PAST
        self.output_chunk_length = output_chunk_length
        (self.min_lh, self.max_lh) = lh
        self.lookback = lookback
        raise_if_not(self.max_lh >= self.min_lh >= 1, 'The lh parameter should be an int tuple (min_lh, max_lh), with 1 <= min_lh <= max_lh')
        raise_if_not(covariates is None or len(self.target_series) == len(self.covariates), 'The provided sequence of target series must have the same length as the provided sequence of covariate series.')
        self.nr_samples_per_ts = (self.max_lh - self.min_lh) * self.output_chunk_length
        self.total_nr_samples = len(self.target_series) * self.nr_samples_per_ts
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the total number of possible (input, target) splits.\n        '
        return self.total_nr_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            i = 10
            return i + 15
        target_idx = idx // self.nr_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)
        raise_if_not(len(target_vals) >= (self.lookback + self.max_lh) * self.output_chunk_length, 'The dataset contains some input/target series that are shorter than `(lookback + max_lh) * H` ({}-th series)'.format(target_idx))
        lh_idx = idx - target_idx * self.nr_samples_per_ts
        end_of_output_idx = len(target_series) - ((self.min_lh - 1) * self.output_chunk_length + lh_idx)
        covariate_series = self.covariates[target_idx] if self.covariates is not None else None
        main_covariate_type = CovariateType.NONE if self.covariates is None else CovariateType.PAST
        shift = self.lookback * self.output_chunk_length
        input_chunk_length = shift
        (past_start, past_end, future_start, future_end, cov_start, cov_end) = self._memory_indexer(target_idx=target_idx, target_series=target_series, shift=shift, input_chunk_length=input_chunk_length, output_chunk_length=self.output_chunk_length, end_of_output_idx=end_of_output_idx, covariate_series=covariate_series, covariate_type=main_covariate_type)
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]
        covariate = None
        if self.covariates is not None:
            raise_if_not(cov_end <= len(covariate_series), f"The dataset contains 'past' covariates that don't extend far enough into the future. ({idx}-th sample)")
            covariate = covariate_series.random_component_values(copy=False)[cov_start:cov_end]
            raise_if_not(len(covariate) == len(past_target), "The dataset contains 'past' covariates whose time axis doesn't allow to obtain the input (or output) chunk relative to the target series.")
        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return (past_target, covariate, static_covariate, future_target)