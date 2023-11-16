"""``ExcelDataSet`` loads/saves data from/to a Excel file using an underlying
filesystem (e.g.: local, S3, GCS). It uses pandas to handle the Excel file.
"""
import logging
from copy import deepcopy
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Dict, Union
import fsspec
import pandas as pd
from kedro.io.core import PROTOCOL_DELIMITER, AbstractVersionedDataset, DatasetError, Version, get_filepath_str, get_protocol_and_path
logger = logging.getLogger(__name__)

class ExcelDataSet(AbstractVersionedDataset[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Union[pd.DataFrame, Dict[str, pd.DataFrame]]]):
    """``ExcelDataSet`` loads/saves data from/to a Excel file using an underlying
    filesystem (e.g.: local, S3, GCS). It uses pandas to handle the Excel file.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:

    .. code-block:: yaml

        rockets:
          type: pandas.ExcelDataSet
          filepath: gcs://your_bucket/rockets.xlsx
          fs_args:
            project: my-project
          credentials: my_gcp_credentials
          save_args:
            sheet_name: Sheet1
          load_args:
            sheet_name: Sheet1

        shuttles:
          type: pandas.ExcelDataSet
          filepath: data/01_raw/shuttles.xlsx

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from kedro.extras.datasets.pandas import ExcelDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>>
        >>> data_set = ExcelDataSet(filepath="test.xlsx")
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>> assert data.equals(reloaded)

    To save a multi-sheet Excel file, no special ``save_args`` are required.
    Instead, return a dictionary of ``Dict[str, pd.DataFrame]`` where the string
    keys are your sheet names.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_
    for a multi-sheet Excel file:

    .. code-block:: yaml

        trains:
          type: pandas.ExcelDataSet
          filepath: data/02_intermediate/company/trains.xlsx
          load_args:
            sheet_name: [Sheet1, Sheet2, Sheet3]

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_
    for a multi-sheet Excel file:
    ::

        >>> from kedro.extras.datasets.pandas import ExcelDataSet
        >>> import pandas as pd
        >>>
        >>> dataframe = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>> another_dataframe = pd.DataFrame({"x": [10, 20], "y": ["hello", "world"]})
        >>> multiframe = {"Sheet1": dataframe, "Sheet2": another_dataframe}
        >>> data_set = ExcelDataSet(filepath="test.xlsx", load_args = {"sheet_name": None})
        >>> data_set.save(multiframe)
        >>> reloaded = data_set.load()
        >>> assert multiframe["Sheet1"].equals(reloaded["Sheet1"])
        >>> assert multiframe["Sheet2"].equals(reloaded["Sheet2"])

    """
    DEFAULT_LOAD_ARGS = {'engine': 'openpyxl'}
    DEFAULT_SAVE_ARGS = {'index': False}

    def __init__(self, filepath: str, engine: str='openpyxl', load_args: Dict[str, Any]=None, save_args: Dict[str, Any]=None, version: Version=None, credentials: Dict[str, Any]=None, fs_args: Dict[str, Any]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new instance of ``ExcelDataSet`` pointing to a concrete Excel file\n        on a specific filesystem.\n\n        Args:\n            filepath: Filepath in POSIX format to a Excel file prefixed with a protocol like\n                `s3://`. If prefix is not provided, `file` protocol (local filesystem) will be used.\n                The prefix should be any protocol supported by ``fsspec``.\n                Note: `http(s)` doesn\'t support versioning.\n            engine: The engine used to write to Excel files. The default\n                engine is \'openpyxl\'.\n            load_args: Pandas options for loading Excel files.\n                Here you can find all available arguments:\n                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html\n                All defaults are preserved, but "engine", which is set to "openpyxl".\n                Supports multi-sheet Excel files (include `sheet_name = None` in `load_args`).\n            save_args: Pandas options for saving Excel files.\n                Here you can find all available arguments:\n                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_excel.html\n                All defaults are preserved, but "index", which is set to False.\n                If you would like to specify options for the `ExcelWriter`,\n                you can include them under the "writer" key. Here you can\n                find all available arguments:\n                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.ExcelWriter.html\n            version: If specified, should be an instance of\n                ``kedro.io.core.Version``. If its ``load`` attribute is\n                None, the latest version will be loaded. If its ``save``\n                attribute is None, save version will be autogenerated.\n            credentials: Credentials required to get access to the underlying filesystem.\n                E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.\n            fs_args: Extra arguments to pass into underlying filesystem class constructor\n                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``).\n\n        Raises:\n            DatasetError: If versioning is enabled while in append mode.\n        '
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        (protocol, path) = get_protocol_and_path(filepath, version)
        if protocol == 'file':
            _fs_args.setdefault('auto_mkdir', True)
        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)
        super().__init__(filepath=PurePosixPath(path), version=version, exists_function=self._fs.exists, glob_function=self._fs.glob)
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        self._writer_args = self._save_args.pop('writer', {})
        self._writer_args.setdefault('engine', engine or 'openpyxl')
        if version and self._writer_args.get('mode') == 'a':
            raise DatasetError("'ExcelDataSet' doesn't support versioning in append mode.")
        if 'storage_options' in self._save_args or 'storage_options' in self._load_args:
            logger.warning("Dropping 'storage_options' for %s, please specify them under 'fs_args' or 'credentials'.", self._filepath)
            self._save_args.pop('storage_options', None)
            self._load_args.pop('storage_options', None)

    def _describe(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return {'filepath': self._filepath, 'protocol': self._protocol, 'load_args': self._load_args, 'save_args': self._save_args, 'writer_args': self._writer_args, 'version': self._version}

    def _load(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if False:
            while True:
                i = 10
        load_path = str(self._get_load_path())
        if self._protocol == 'file':
            return pd.read_excel(load_path, **self._load_args)
        load_path = f'{self._protocol}{PROTOCOL_DELIMITER}{load_path}'
        return pd.read_excel(load_path, storage_options=self._storage_options, **self._load_args)

    def _save(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
        if False:
            i = 10
            return i + 15
        output = BytesIO()
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with pd.ExcelWriter(output, **self._writer_args) as writer:
            if isinstance(data, dict):
                for (sheet_name, sheet_data) in data.items():
                    sheet_data.to_excel(writer, sheet_name=sheet_name, **self._save_args)
            else:
                data.to_excel(writer, **self._save_args)
        with self._fs.open(save_path, mode='wb') as fs_file:
            fs_file.write(output.getvalue())
        self._invalidate_cache()

    def _exists(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return self._fs.exists(load_path)

    def _release(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        if False:
            return 10
        'Invalidate underlying filesystem caches.'
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)