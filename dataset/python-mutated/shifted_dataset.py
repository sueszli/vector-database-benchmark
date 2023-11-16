"""
Shifted Training Dataset
------------------------
"""
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from darts import TimeSeries
from darts.logging import raise_if_not
from .training_dataset import DualCovariatesTrainingDataset, FutureCovariatesTrainingDataset, MixedCovariatesTrainingDataset, PastCovariatesTrainingDataset, SplitCovariatesTrainingDataset, TrainingDataset
from .utils import CovariateType

class PastCovariatesShiftedDataset(PastCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, length: int=12, shift: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, future_target)\n        arrays, which all have length `length`.\n        The "future_target" is the "past_target" target shifted by `shift` time steps forward.\n        So if an emitted "past_target" (and "past_covariates") goes from position `i` to `i+length`,\n        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `length + shift`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        length\n            The length of the emitted past and future series.\n        shift\n            The number of time steps by which to shift the output relative to the input.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            print('Hello World!')
        return self.ds[idx]

class FutureCovariatesShiftedDataset(FutureCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, length: int=12, shift: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, future_target)\n        arrays, which all have length `length`.\n        The "future_target" is the "past_target" target shifted by `shift` time steps forward.\n        So if an emitted "past_target" goes from position `i` to `i+length`,\n        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.\n        The slicing future covariates matches that of future targets. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `length + shift`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        length\n            The length of the emitted past and future series.\n        shift\n            The number of time steps by which to shift the output relative to the input.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            i = 10
            return i + 15
        return self.ds[idx]

class DualCovariatesShiftedDataset(DualCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, length: int=12, shift: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            while True:
                i = 10
        '\n        A time series dataset containing tuples of\n        (past_target, historic_future_covariates, future_covariates, static_covariates, future_target)\n        arrays, which all have length `length`.\n        The "future_target" is the "past_target" target shifted by `shift` time steps forward.\n        So if an emitted "past_target" goes from position `i` to `i+length`,\n        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.\n        The slicing "future_covariates" matches that of "futuretarget" and the slicing of "historic_future_covariates"\n        matches that of "past_target". The slicing itself relies on time indexes to align the series if they have\n        unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `length + shift`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        length\n            The length of the emitted past and future series.\n        shift\n            The number of time steps by which to shift the output relative to the input.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.HISTORIC_FUTURE, use_static_covariates=use_static_covariates)
        self.ds_future = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            print('Hello World!')
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, future_covariate, _, _) = self.ds_future[idx]
        return (past_target, past_covariate, future_covariate, static_covariate, future_target)

class MixedCovariatesShiftedDataset(MixedCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, length: int=12, shift: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, past_covariates, historic_future_covariates,\n        future_covariates, static_covariates, future_target) arrays, which all have length `length`.\n        The "future_target" is the "past_target" target shifted by `shift` time steps forward.\n        So if an emitted "past_target" goes from position `i` to `i+length`,\n        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.\n        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `length + shift`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        past_covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        future_covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow\n            the same constraints as `past_covariates`.\n        length\n            The length of the emitted past and future series.\n        shift\n            The number of time steps by which to shift the output relative to the input.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=past_covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)
        self.ds_dual = DualCovariatesShiftedDataset(target_series=target_series, covariates=future_covariates, length=length, shift=shift, max_samples_per_ts=max_samples_per_ts, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            i = 10
            return i + 15
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, historic_future_covariate, future_covariate, _, _) = self.ds_dual[idx]
        return (past_target, past_covariate, historic_future_covariate, future_covariate, static_covariate, future_target)

class SplitCovariatesShiftedDataset(SplitCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, length: int=12, shift: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            while True:
                i = 10
        '\n        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,\n        future_target) arrays, which all have length `length`.\n        The "future_target" is the "past_target" target shifted by `shift` time steps forward.\n        So if an emitted "past_target" goes from position `i` to `i+length`,\n        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.\n        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `length + shift`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        past_covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        future_covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow\n            the same constraints as `past_covariates`.\n        length\n            The length of the emitted past and future series.\n        shift\n            The number of time steps by which to shift the output relative to the input.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=past_covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)
        self.ds_future = GenericShiftedDataset(target_series=target_series, covariates=future_covariates, input_chunk_length=length, output_chunk_length=length, shift=shift, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            return 10
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            while True:
                i = 10
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, future_covariate, _, _) = self.ds_future[idx]
        return (past_target, past_covariate, future_covariate, static_covariate, future_target)

class GenericShiftedDataset(TrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, shift: int=1, shift_covariates: bool=False, max_samples_per_ts: Optional[int]=None, covariate_type: CovariateType=CovariateType.NONE, use_static_covariates: bool=True):
        if False:
            return 10
        '\n        Contains (past_target, <X>_covariates, static_covariates, future_target), where "<X>" is past if\n        `shift_covariates = False` and future otherwise.\n        The past chunks have length `input_chunk_length` and the future chunks have length `output_chunk_length`.\n        The future chunks start `shift` after the past chunks\' start.\n\n        This is meant to be a "generic" dataset that can be used to build ShiftedDataset\'s\n        (when `input_chunk_length = output_chunk_length`), or SequenceDataset\'s (when `shift = input_chunk_length`).\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing covariates.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        shift\n            The number of time steps by which to shift the output chunks relative to the input chunks.\n        shift_covariates\n            Whether or not to shift the covariates forward the same way as the target.\n            FutureCovariatesModel\'s require this set to True, while PastCovariatesModel\'s require this set to False.\n        max_samples_per_ts\n            This is an upper bound on the number of (input, output, input_covariates) tuples that can be produced\n            per time series. It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        covariate_type\n            An instance of `CovariateType` describing the type of `covariates`.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.covariate_type = covariate_type
        raise_if_not(covariates is None or len(self.target_series) == len(self.covariates), 'The provided sequence of target series must have the same length as the provided sequence of covariate series.')
        (self.input_chunk_length, self.output_chunk_length) = (input_chunk_length, output_chunk_length)
        (self.shift, self.shift_covariates) = (shift, shift_covariates)
        self.max_samples_per_ts = max_samples_per_ts
        self.size_of_both_chunks = max(self.input_chunk_length, self.shift + self.output_chunk_length)
        if self.max_samples_per_ts is None:
            self.max_samples_per_ts = max((len(ts) for ts in self.target_series)) - self.size_of_both_chunks + 1
        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ideal_nr_samples

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if False:
            i = 10
            return i + 15
        target_idx = idx // self.max_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)
        n_samples_in_ts = len(target_vals) - self.size_of_both_chunks + 1
        raise_if_not(n_samples_in_ts >= 1, 'The dataset contains some time series that are too short to contain `max(self.input_chunk_length, self.shift + self.output_chunk_length)` ({}-th series)'.format(target_idx))
        end_of_output_idx = len(target_series) - (idx - target_idx * self.max_samples_per_ts) % n_samples_in_ts
        covariate_series = self.covariates[target_idx] if self.covariates is not None else None
        main_covariate_type = CovariateType.NONE
        if self.covariates is not None:
            main_covariate_type = CovariateType.FUTURE if self.shift_covariates else CovariateType.PAST
        (past_start, past_end, future_start, future_end, covariate_start, covariate_end) = self._memory_indexer(target_idx=target_idx, target_series=target_series, shift=self.shift, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, end_of_output_idx=end_of_output_idx, covariate_series=covariate_series, covariate_type=main_covariate_type)
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]
        covariate = None
        if self.covariates is not None:
            raise_if_not(covariate_end <= len(covariate_series), f"The dataset contains {main_covariate_type.value} covariates that don't extend far enough into the future. ({idx}-th sample)")
            covariate = covariate_series.random_component_values(copy=False)[covariate_start:covariate_end]
            raise_if_not(len(covariate) == (self.output_chunk_length if self.shift_covariates else self.input_chunk_length), f"The dataset contains {main_covariate_type.value} covariates whose time axis doesn't allow to obtain the input (or output) chunk relative to the target series.")
        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return (past_target, covariate, static_covariate, future_target)