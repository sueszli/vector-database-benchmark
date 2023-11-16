"""GeoJSONDataSet loads and saves data to a local geojson file. The
underlying functionality is supported by geopandas, so it supports all
allowed geopandas (pandas) options for loading and saving geosjon files.
"""
import copy
from pathlib import PurePosixPath
from typing import Any, Dict, Union
import fsspec
import geopandas as gpd
from kedro.io.core import AbstractVersionedDataset, DatasetError, Version, get_filepath_str, get_protocol_and_path

class GeoJSONDataSet(AbstractVersionedDataset[gpd.GeoDataFrame, Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]]):
    """``GeoJSONDataSet`` loads/saves data to a GeoJSON file using an underlying filesystem
    (eg: local, S3, GCS).
    The underlying functionality is supported by geopandas, so it supports all
    allowed geopandas (pandas) options for loading and saving GeoJSON files.

    Example:
    ::

        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> from kedro.extras.datasets.geopandas import GeoJSONDataSet
        >>>
        >>> data = gpd.GeoDataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]}, geometry=[Point(1,1), Point(2,4)])
        >>> data_set = GeoJSONDataSet(filepath="test.geojson", save_args=None)
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>>
        >>> assert data.equals(reloaded)

    """
    DEFAULT_LOAD_ARGS = {}
    DEFAULT_SAVE_ARGS = {'driver': 'GeoJSON'}

    def __init__(self, filepath: str, load_args: Dict[str, Any]=None, save_args: Dict[str, Any]=None, version: Version=None, credentials: Dict[str, Any]=None, fs_args: Dict[str, Any]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a new instance of ``GeoJSONDataSet`` pointing to a concrete GeoJSON file\n        on a specific filesystem fsspec.\n\n        Args:\n\n            filepath: Filepath in POSIX format to a GeoJSON file prefixed with a protocol like\n                `s3://`. If prefix is not provided `file` protocol (local filesystem) will be used.\n                The prefix should be any protocol supported by ``fsspec``.\n                Note: `http(s)` doesn\'t support versioning.\n            load_args: GeoPandas options for loading GeoJSON files.\n                Here you can find all available arguments:\n                https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html\n            save_args: GeoPandas options for saving geojson files.\n                Here you can find all available arguments:\n                https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html\n                The default_save_arg driver is \'GeoJSON\', all others preserved.\n            version: If specified, should be an instance of\n                ``kedro.io.core.Version``. If its ``load`` attribute is\n                None, the latest version will be loaded. If its ``save``\n            credentials: credentials required to access the underlying filesystem.\n                Eg. for ``GCFileSystem`` it would look like `{\'token\': None}`.\n            fs_args: Extra arguments to pass into underlying filesystem class constructor\n                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as\n                to pass to the filesystem\'s `open` method through nested keys\n                `open_args_load` and `open_args_save`.\n                Here you can find all available arguments for `open`:\n                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.open\n                All defaults are preserved, except `mode`, which is set to `wb` when saving.\n        '
        _fs_args = copy.deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop('open_args_load', {})
        _fs_open_args_save = _fs_args.pop('open_args_save', {})
        _credentials = copy.deepcopy(credentials) or {}
        (protocol, path) = get_protocol_and_path(filepath, version)
        self._protocol = protocol
        if protocol == 'file':
            _fs_args.setdefault('auto_mkdir', True)
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)
        super().__init__(filepath=PurePosixPath(path), version=version, exists_function=self._fs.exists, glob_function=self._fs.glob)
        self._load_args = copy.deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = copy.deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        _fs_open_args_save.setdefault('mode', 'wb')
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

    def _load(self) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
        if False:
            return 10
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
            return gpd.read_file(fs_file, **self._load_args)

    def _save(self, data: gpd.GeoDataFrame) -> None:
        if False:
            print('Hello World!')
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            data.to_file(fs_file, **self._save_args)
        self.invalidate_cache()

    def _exists(self) -> bool:
        if False:
            print('Hello World!')
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return self._fs.exists(load_path)

    def _describe(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'filepath': self._filepath, 'protocol': self._load_args, 'save_args': self._save_args, 'version': self._version}

    def _release(self) -> None:
        if False:
            return 10
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        if False:
            print('Hello World!')
        'Invalidate underlying filesystem cache.'
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)