import hashlib
import os
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas as pd
import requests
from darts import TimeSeries
from darts.logging import get_logger
logger = get_logger(__name__)

@dataclass
class DatasetLoaderMetadata:
    name: str
    uri: str
    hash: str
    header_time: Optional[str]
    format_time: Optional[str] = None
    freq: Optional[str] = None
    pre_process_zipped_csv_fn: Optional[Callable] = None
    pre_process_csv_fn: Optional[Callable] = None
    multivariate: Optional[bool] = None

class DatasetLoadingException(BaseException):
    pass

class DatasetLoader(ABC):
    """
    Class that downloads a dataset and caches it locally.
    Assumes that the file can be downloaded (i.e. publicly available via a URI)
    """
    _DEFAULT_DIRECTORY = Path(os.path.join(Path.home(), Path('.darts/datasets/')))

    def __init__(self, metadata: DatasetLoaderMetadata, root_path: Optional[Path]=None):
        if False:
            for i in range(10):
                print('nop')
        self._metadata: DatasetLoaderMetadata = metadata
        if root_path is None:
            self._root_path: Path = DatasetLoader._DEFAULT_DIRECTORY
        else:
            self._root_path: Path = root_path

    def load(self) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load the dataset in memory, as a TimeSeries.\n        Downloads the dataset if it is not present already\n\n        Raises\n        -------\n        DatasetLoadingException\n            If loading fails (MD5 Checksum is invalid, Download failed, Reading from disk failed)\n\n        Returns\n        -------\n        time_series: TimeSeries\n            A TimeSeries object that contains the dataset\n        '
        if not self._is_already_downloaded():
            if self._metadata.uri.endswith('.zip'):
                self._download_zip_dataset()
            else:
                self._download_dataset()
        self._check_dataset_integrity_or_raise()
        return self._load_from_disk(self._get_path_dataset(), self._metadata)

    def _check_dataset_integrity_or_raise(self):
        if False:
            print('Hello World!')
        '\n        Ensures that the dataset exists and its MD5 checksum matches the expected hash.\n\n        Raises\n        -------\n        DatasetLoadingException\n            if checks fail\n\n        Returns\n        -------\n        '
        if not self._is_already_downloaded():
            raise DatasetLoadingException(f'Checking md5 checksum of a absent file: {self._get_path_dataset()}')
        with open(self._get_path_dataset(), 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
            if md5_hash != self._metadata.hash:
                raise DatasetLoadingException(f'Expected hash for {self._get_path_dataset()}: {self._metadata.hash}, got: {md5_hash}')

    def _download_dataset(self):
        if False:
            while True:
                i = 10
        '\n        Downloads the dataset in the root_path directory\n\n        Raises\n        -------\n        DatasetLoadingException\n            if downloading or writing the file to disk fails\n\n        Returns\n        -------\n        '
        if self._metadata.pre_process_zipped_csv_fn:
            logger.warning('Loading a CSV file does not use the pre_process_zipped_csv_fn')
        os.makedirs(self._root_path, exist_ok=True)
        try:
            request = requests.get(self._metadata.uri)
            with open(self._get_path_dataset(), 'wb') as f:
                f.write(request.content)
        except Exception as e:
            raise DatasetLoadingException('Could not download the dataset. Reason:' + e.__repr__()) from None
        if self._metadata.pre_process_csv_fn is not None:
            self._metadata.pre_process_csv_fn(self._get_path_dataset())

    def _download_zip_dataset(self):
        if False:
            return 10
        if self._metadata.pre_process_csv_fn:
            logger.warning('Loading a ZIP file does not use the pre_process_csv_fn')
        os.makedirs(self._root_path, exist_ok=True)
        try:
            request = requests.get(self._metadata.uri)
            with tempfile.TemporaryFile() as tf:
                tf.write(request.content)
                with tempfile.TemporaryDirectory() as td:
                    with zipfile.ZipFile(tf, 'r') as zip_ref:
                        zip_ref.extractall(td)
                        self._metadata.pre_process_zipped_csv_fn(td, self._get_path_dataset())
        except Exception as e:
            raise DatasetLoadingException('Could not download the dataset. Reason:' + e.__repr__()) from None

    @abstractmethod
    def _load_from_disk(self, path_to_file: Path, metadata: DatasetLoaderMetadata) -> TimeSeries:
        if False:
            return 10
        "\n        Given a Path to the file and a DataLoaderMetadata object, return a TimeSeries\n        One can assume that the file exists and its MD5 checksum has been verified before this function is called\n\n        Parameters\n        ----------\n        path_to_file: Path\n            A Path object where the dataset is located\n        metadata: Metadata\n            The dataset's metadata\n\n        Returns\n        -------\n        time_series: TimeSeries\n            a TimeSeries object that contains the whole dataset\n        "
        pass

    def _get_path_dataset(self) -> Path:
        if False:
            i = 10
            return i + 15
        return Path(os.path.join(self._root_path, self._metadata.name))

    def _is_already_downloaded(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return os.path.isfile(self._get_path_dataset())

    def _format_time_column(self, df):
        if False:
            while True:
                i = 10
        df[self._metadata.header_time] = pd.to_datetime(df[self._metadata.header_time], format=self._metadata.format_time, errors='raise')
        return df

class DatasetLoaderCSV(DatasetLoader):

    def __init__(self, metadata: DatasetLoaderMetadata, root_path: Optional[Path]=None):
        if False:
            print('Hello World!')
        super().__init__(metadata, root_path)

    def _load_from_disk(self, path_to_file: Path, metadata: DatasetLoaderMetadata) -> Union[TimeSeries, List[TimeSeries]]:
        if False:
            i = 10
            return i + 15
        df = pd.read_csv(path_to_file)
        if metadata.header_time is not None:
            df = self._format_time_column(df)
            series = TimeSeries.from_dataframe(df=df, time_col=metadata.header_time, freq=metadata.freq)
        else:
            df.sort_index(inplace=True)
            series = TimeSeries.from_dataframe(df)
        if self._metadata.multivariate is not None and self._metadata.multivariate is False:
            try:
                series = self._to_multi_series(series.pd_dataframe())
            except Exception as e:
                raise DatasetLoadingException('Could not convert to multi-series. Reason:' + e.__repr__()) from None
        return series