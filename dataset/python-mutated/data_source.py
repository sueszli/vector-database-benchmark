from abc import ABC, abstractmethod
from typing import List, Tuple
import dask.dataframe as dd
import pandas as pd
from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.audio_utils import is_audio_score
from ludwig.utils.automl.utils import avg_num_tokens
from ludwig.utils.image_utils import is_image_score
from ludwig.utils.misc_utils import memoized_method
from ludwig.utils.types import DataFrame

@DeveloperAPI
class DataSource(ABC):

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abstractmethod
    def get_dtype(self, column: str) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abstractmethod
    def get_distinct_values(self, column: str, max_values_to_return: int) -> Tuple[int, List[str], float]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abstractmethod
    def get_nonnull_values(self, column: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abstractmethod
    def get_avg_num_tokens(self, column: str) -> int:
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def is_string_type(self, dtype: str) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def size_bytes(self) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

@DeveloperAPI
class DataframeSourceMixin:
    df: DataFrame

    @property
    def columns(self) -> List[str]:
        if False:
            print('Hello World!')
        return self.df.columns

    def get_dtype(self, column: str) -> str:
        if False:
            return 10
        return self.df[column].dtype.name

    def get_distinct_values(self, column, max_values_to_return: int) -> Tuple[int, List[str], float]:
        if False:
            i = 10
            return i + 15
        unique_values = self.df[column].dropna().unique()
        num_unique_values = len(unique_values)
        unique_values_counts = self.df[column].value_counts()
        if len(unique_values_counts) != 0:
            unique_majority_values = unique_values_counts[unique_values_counts.idxmax()]
            unique_minority_values = unique_values_counts[unique_values_counts.idxmin()]
            unique_values_balance = unique_minority_values / unique_majority_values
        else:
            unique_values_balance = 1.0
        return (num_unique_values, unique_values[:max_values_to_return], unique_values_balance)

    def get_nonnull_values(self, column: str) -> int:
        if False:
            while True:
                i = 10
        return len(self.df[column].notnull())

    def get_image_values(self, column: str, sample_size: int=10) -> int:
        if False:
            while True:
                i = 10
        return int(sum((is_image_score(x) for x in self.df[column].head(sample_size))))

    def get_audio_values(self, column: str, sample_size: int=10) -> int:
        if False:
            print('Hello World!')
        return int(sum((is_audio_score(x) for x in self.df[column].head(sample_size))))

    def get_avg_num_tokens(self, column: str) -> int:
        if False:
            return 10
        return avg_num_tokens(self.df[column])

    def is_string_type(self, dtype: str) -> bool:
        if False:
            while True:
                i = 10
        return dtype in ['str', 'string', 'object']

    def size_bytes(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return sum(self.df.memory_usage(deep=True))

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.df)

@DeveloperAPI
class DataframeSource(DataframeSourceMixin, DataSource):

    def __init__(self, df):
        if False:
            while True:
                i = 10
        self.df = df

@DeveloperAPI
class DaskDataSource(DataframeSource):

    @memoized_method(maxsize=1)
    def get_sample(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        return self.df.head(10000)

    @property
    def sample(self) -> pd.DataFrame:
        if False:
            return 10
        return self.get_sample()

    def get_distinct_values(self, column, max_values_to_return) -> Tuple[int, List[str], float]:
        if False:
            print('Hello World!')
        unique_values = self.df[column].drop_duplicates().dropna().persist()
        num_unique_values = len(unique_values)
        imbalance_ratio = 1.0
        return (num_unique_values, unique_values.head(max_values_to_return), imbalance_ratio)

    def get_nonnull_values(self, column) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.df[column].notnull().sum().compute()

    def get_image_values(self, column: str, sample_size: int=10) -> int:
        if False:
            while True:
                i = 10
        return int(sum((is_image_score(x) for x in self.sample[column].head(sample_size))))

    def get_audio_values(self, column: str, sample_size: int=10) -> int:
        if False:
            i = 10
            return i + 15
        return int(sum((is_audio_score(x) for x in self.sample[column].head(sample_size))))

    def get_avg_num_tokens(self, column) -> int:
        if False:
            for i in range(10):
                print('nop')
        return avg_num_tokens(self.sample[column])

@DeveloperAPI
def wrap_data_source(df: DataFrame) -> DataSource:
    if False:
        while True:
            i = 10
    if isinstance(df, dd.core.DataFrame):
        return DaskDataSource(df)
    return DataframeSource(df)