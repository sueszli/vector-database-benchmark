"""
Sequential Training Dataset
---------------------------
"""
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from darts import TimeSeries
from .shifted_dataset import GenericShiftedDataset
from .training_dataset import DualCovariatesTrainingDataset, FutureCovariatesTrainingDataset, MixedCovariatesTrainingDataset, PastCovariatesTrainingDataset, SplitCovariatesTrainingDataset
from .utils import CovariateType

class PastCovariatesSequentialDataset(PastCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, future_target).\n        The "past" series have length `input_chunk_length` and the "future" series have\n        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.\n        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `input_chunk_length + output_chunk_length`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            print('Hello World!')
        return self.ds[idx]

class FutureCovariatesSequentialDataset(FutureCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, future_target).\n        The "past" series have length `input_chunk_length` and the "future" series have\n        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.\n        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `input_chunk_length + output_chunk_length`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            i = 10
            return i + 15
        return self.ds[idx]

class DualCovariatesSequentialDataset(DualCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        A time series dataset containing tuples of\n        (past_target, historic_future_covariates, future_covariates, static_covariates, future_target).\n        The "past" series (incl `historic_future_covariates`) have length `input_chunk_length`\n        and the "future" series have length `output_chunk_length`. The "future" series are immediately consecutive\n        to the "past" series. The slicing of past and future covariates matches that of past and future targets,\n        respectively. The slicing itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `input_chunk_length + output_chunk_length`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.HISTORIC_FUTURE, use_static_covariates=use_static_covariates)
        self.ds_future = GenericShiftedDataset(target_series=target_series, covariates=covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            print('Hello World!')
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, future_covariate, _, _) = self.ds_future[idx]
        return (past_target, past_covariate, future_covariate, static_covariate, future_target)

class MixedCovariatesSequentialDataset(MixedCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            return 10
        '\n        A time series dataset containing tuples of\n        (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates, future_target).\n        The "past" series (incl `historic_future_covariates`) have length `input_chunk_length`\n        and the "future" series have length `output_chunk_length`. The "future" series are immediately consecutive\n        to the "past" series. The slicing of past and future covariates matches that of past and future targets,\n        respectively. The slicing itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `input_chunk_length + output_chunk_length`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        past_covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        future_covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow\n            the same constraints as `past_covariates`.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=past_covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)
        self.ds_dual = DualCovariatesSequentialDataset(target_series=target_series, covariates=future_covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            print('Hello World!')
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, historic_future_covariate, future_covariate, _, _) = self.ds_dual[idx]
        return (past_target, past_covariate, historic_future_covariate, future_covariate, static_covariate, future_target)

class SplitCovariatesSequentialDataset(SplitCovariatesTrainingDataset):

    def __init__(self, target_series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, input_chunk_length: int=12, output_chunk_length: int=1, max_samples_per_ts: Optional[int]=None, use_static_covariates: bool=True):
        if False:
            print('Hello World!')
        '\n        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,\n        future_target).\n        The "past" series have length `input_chunk_length` and the "future" series have\n        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.\n        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing\n        itself relies on time indexes to align the series if they have unequal lengths.\n\n        Each series must be long enough to contain at least one (input, output) pair; i.e., each\n        series must have length at least `input_chunk_length + output_chunk_length`.\n        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.\n\n        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has\n        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different\n        lengths, they will contain different numbers of slices. Therefore, some particular slices may\n        be sampled more often than others if they belong to shorter time series.\n\n        Parameters\n        ----------\n        target_series\n            One or a sequence of target `TimeSeries`.\n        past_covariates\n            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,\n            the provided sequence must have the same length as that of `target_series`. Moreover, all\n            covariates in the sequence must have a time span large enough to contain all the required slices.\n            The joint slicing of the target and covariates is relying on the time axes of both series.\n        future_covariates\n            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow\n            the same constraints as `past_covariates`.\n        input_chunk_length\n            The length of the emitted past series.\n        output_chunk_length\n            The length of the emitted future series.\n        max_samples_per_ts\n            This is an upper bound on the number of tuples that can be produced per time series.\n            It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        use_static_covariates\n            Whether to use/include static covariate data from input series.\n        '
        super().__init__()
        self.ds_past = GenericShiftedDataset(target_series=target_series, covariates=past_covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=False, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.PAST, use_static_covariates=use_static_covariates)
        self.ds_future = GenericShiftedDataset(target_series=target_series, covariates=future_covariates, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, shift=input_chunk_length, shift_covariates=True, max_samples_per_ts=max_samples_per_ts, covariate_type=CovariateType.FUTURE, use_static_covariates=use_static_covariates)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if False:
            i = 10
            return i + 15
        (past_target, past_covariate, static_covariate, future_target) = self.ds_past[idx]
        (_, future_covariate, _, _) = self.ds_future[idx]
        return (past_target, past_covariate, future_covariate, static_covariate, future_target)