from typing import Mapping, Sequence, Union
from modelscope.msdatasets.auth.auth_config import BaseAuthConfig
from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.msdatasets.meta.data_meta_config import DataMetaConfig
from modelscope.utils.constant import DownloadMode, Hubs

class DatasetContextConfig:
    """Context configuration of dataset."""

    def __init__(self, dataset_name: Union[str, list], namespace: str, version: str, subset_name: str, split: Union[str, list], target: str, hub: Hubs, data_dir: str, data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]], download_mode: DownloadMode, cache_root_dir: str, use_streaming: bool, stream_batch_size: int, **kwargs):
        if False:
            return 10
        self._download_config = None
        self._data_meta_config = None
        self._config_kwargs = kwargs
        self._dataset_version_cache_root_dir = None
        self._auth_config = None
        self._global_meta_lock_file_path = None
        self._global_data_lock_file_path = None
        self.hub = hub
        self.download_mode = download_mode
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.version = version
        self.subset_name = subset_name
        self.split = split
        self.target = target
        self.data_dir = data_dir
        self.data_files = data_files
        self.cache_root_dir = cache_root_dir
        self.use_streaming = use_streaming
        self.stream_batch_size = stream_batch_size
        self.download_virgo_files: bool = False

    @property
    def config_kwargs(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return self._config_kwargs

    @config_kwargs.setter
    def config_kwargs(self, val: dict):
        if False:
            while True:
                i = 10
        self._config_kwargs = val

    @property
    def download_config(self) -> DataDownloadConfig:
        if False:
            for i in range(10):
                print('nop')
        return self._download_config

    @download_config.setter
    def download_config(self, val: DataDownloadConfig):
        if False:
            while True:
                i = 10
        self._download_config = val

    @property
    def data_meta_config(self) -> DataMetaConfig:
        if False:
            return 10
        return self._data_meta_config

    @data_meta_config.setter
    def data_meta_config(self, val: DataMetaConfig):
        if False:
            return 10
        self._data_meta_config = val

    @property
    def dataset_version_cache_root_dir(self) -> str:
        if False:
            print('Hello World!')
        return self._dataset_version_cache_root_dir

    @dataset_version_cache_root_dir.setter
    def dataset_version_cache_root_dir(self, val: str):
        if False:
            for i in range(10):
                print('nop')
        self._dataset_version_cache_root_dir = val

    @property
    def global_meta_lock_file_path(self) -> str:
        if False:
            return 10
        return self._global_meta_lock_file_path

    @global_meta_lock_file_path.setter
    def global_meta_lock_file_path(self, val: str):
        if False:
            return 10
        self._global_meta_lock_file_path = val

    @property
    def global_data_lock_file_path(self) -> str:
        if False:
            while True:
                i = 10
        return self._global_data_lock_file_path

    @global_data_lock_file_path.setter
    def global_data_lock_file_path(self, val: str):
        if False:
            while True:
                i = 10
        self._global_data_lock_file_path = val

    @property
    def auth_config(self) -> BaseAuthConfig:
        if False:
            for i in range(10):
                print('nop')
        return self._auth_config

    @auth_config.setter
    def auth_config(self, val: BaseAuthConfig):
        if False:
            return 10
        self._auth_config = val