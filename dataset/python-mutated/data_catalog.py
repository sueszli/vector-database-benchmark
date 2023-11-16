"""``DataCatalog`` stores instances of ``AbstractDataset`` implementations to
provide ``load`` and ``save`` capabilities from anywhere in the program. To
use a ``DataCatalog``, you need to instantiate it with a dictionary of data
sets. Then it will act as a single point of reference for your calls,
relaying load and save functions to the underlying data sets.
"""
from __future__ import annotations
import copy
import difflib
import logging
import re
from collections import defaultdict
from typing import Any, Dict
from parse import parse
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, AbstractVersionedDataset, DatasetAlreadyExistsError, DatasetError, DatasetNotFoundError, Version, generate_timestamp
from kedro.io.memory_dataset import MemoryDataset
Patterns = Dict[str, Dict[str, Any]]
CATALOG_KEY = 'catalog'
CREDENTIALS_KEY = 'credentials'
WORDS_REGEX_PATTERN = re.compile('\\W+')

def _get_credentials(credentials_name: str, credentials: dict[str, Any]) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Return a set of credentials from the provided credentials dict.\n\n    Args:\n        credentials_name: Credentials name.\n        credentials: A dictionary with all credentials.\n\n    Returns:\n        The set of requested credentials.\n\n    Raises:\n        KeyError: When a data set with the given name has not yet been\n            registered.\n\n    '
    try:
        return credentials[credentials_name]
    except KeyError as exc:
        raise KeyError(f"Unable to find credentials '{credentials_name}': check your data catalog and credentials configuration. See https://kedro.readthedocs.io/en/stable/kedro.io.DataCatalog.html for an example.") from exc

def _resolve_credentials(config: dict[str, Any], credentials: dict[str, Any]) -> dict[str, Any]:
    if False:
        print('Hello World!')
    'Return the dataset configuration where credentials are resolved using\n    credentials dictionary provided.\n\n    Args:\n        config: Original dataset config, which may contain unresolved credentials.\n        credentials: A dictionary with all credentials.\n\n    Returns:\n        The dataset config, where all the credentials are successfully resolved.\n    '
    config = copy.deepcopy(config)

    def _map_value(key: str, value: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if key == CREDENTIALS_KEY and isinstance(value, str):
            return _get_credentials(value, credentials)
        if isinstance(value, dict):
            return {k: _map_value(k, v) for (k, v) in value.items()}
        return value
    return {k: _map_value(k, v) for (k, v) in config.items()}

def _sub_nonword_chars(data_set_name: str) -> str:
    if False:
        print('Hello World!')
    'Replace non-word characters in data set names since Kedro 0.16.2.\n\n    Args:\n        data_set_name: The data set name registered in the data catalog.\n\n    Returns:\n        The name used in `DataCatalog.datasets`.\n    '
    return re.sub(WORDS_REGEX_PATTERN, '__', data_set_name)

class _FrozenDatasets:
    """Helper class to access underlying loaded datasets."""

    def __init__(self, *datasets_collections: _FrozenDatasets | dict[str, AbstractDataset]):
        if False:
            return 10
        'Return a _FrozenDatasets instance from some datasets collections.\n        Each collection could either be another _FrozenDatasets or a dictionary.\n        '
        for collection in datasets_collections:
            if isinstance(collection, _FrozenDatasets):
                self.__dict__.update(collection.__dict__)
            else:
                self.__dict__.update({_sub_nonword_chars(dataset_name): dataset for (dataset_name, dataset) in collection.items()})

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Operation not allowed! '
        if key in self.__dict__:
            msg += 'Please change datasets through configuration.'
        else:
            msg += 'Please use DataCatalog.add() instead.'
        raise AttributeError(msg)

class DataCatalog:
    """``DataCatalog`` stores instances of ``AbstractDataset`` implementations
    to provide ``load`` and ``save`` capabilities from anywhere in the
    program. To use a ``DataCatalog``, you need to instantiate it with
    a dictionary of data sets. Then it will act as a single point of reference
    for your calls, relaying load and save functions
    to the underlying data sets.
    """

    def __init__(self, data_sets: dict[str, AbstractDataset]=None, feed_dict: dict[str, Any]=None, layers: dict[str, set[str]]=None, dataset_patterns: Patterns=None, load_versions: dict[str, str]=None, save_version: str=None) -> None:
        if False:
            return 10
        '``DataCatalog`` stores instances of ``AbstractDataset``\n        implementations to provide ``load`` and ``save`` capabilities from\n        anywhere in the program. To use a ``DataCatalog``, you need to\n        instantiate it with a dictionary of data sets. Then it will act as a\n        single point of reference for your calls, relaying load and save\n        functions to the underlying data sets.\n\n        Args:\n            data_sets: A dictionary of data set names and data set instances.\n            feed_dict: A feed dict with data to be added in memory.\n            layers: A dictionary of data set layers. It maps a layer name\n                to a set of data set names, according to the\n                data engineering convention. For more details, see\n                https://docs.kedro.org/en/stable/resources/glossary.html#layers-data-engineering-convention\n            dataset_patterns: A dictionary of data set factory patterns\n                and corresponding data set configuration\n            load_versions: A mapping between data set names and versions\n                to load. Has no effect on data sets without enabled versioning.\n            save_version: Version string to be used for ``save`` operations\n                by all data sets with enabled versioning. It must: a) be a\n                case-insensitive string that conforms with operating system\n                filename limitations, b) always return the latest version when\n                sorted in lexicographical order.\n\n        Example:\n        ::\n\n            >>> from kedro.extras.datasets.pandas import CSVDataSet\n            >>>\n            >>> cars = CSVDataSet(filepath="cars.csv",\n            >>>                   load_args=None,\n            >>>                   save_args={"index": False})\n            >>> io = DataCatalog(data_sets={\'cars\': cars})\n        '
        self._data_sets = dict(data_sets or {})
        self.datasets = _FrozenDatasets(self._data_sets)
        self.layers = layers
        self._dataset_patterns = dataset_patterns or {}
        self._load_versions = load_versions or {}
        self._save_version = save_version
        if feed_dict:
            self.add_feed_dict(feed_dict)

    @property
    def _logger(self):
        if False:
            return 10
        return logging.getLogger(__name__)

    @classmethod
    def from_config(cls, catalog: dict[str, dict[str, Any]] | None, credentials: dict[str, dict[str, Any]]=None, load_versions: dict[str, str]=None, save_version: str=None) -> DataCatalog:
        if False:
            print('Hello World!')
        'Create a ``DataCatalog`` instance from configuration. This is a\n        factory method used to provide developers with a way to instantiate\n        ``DataCatalog`` with configuration parsed from configuration files.\n\n        Args:\n            catalog: A dictionary whose keys are the data set names and\n                the values are dictionaries with the constructor arguments\n                for classes implementing ``AbstractDataset``. The data set\n                class to be loaded is specified with the key ``type`` and their\n                fully qualified class name. All ``kedro.io`` data set can be\n                specified by their class name only, i.e. their module name\n                can be omitted.\n            credentials: A dictionary containing credentials for different\n                data sets. Use the ``credentials`` key in a ``AbstractDataset``\n                to refer to the appropriate credentials as shown in the example\n                below.\n            load_versions: A mapping between dataset names and versions\n                to load. Has no effect on data sets without enabled versioning.\n            save_version: Version string to be used for ``save`` operations\n                by all data sets with enabled versioning. It must: a) be a\n                case-insensitive string that conforms with operating system\n                filename limitations, b) always return the latest version when\n                sorted in lexicographical order.\n\n        Returns:\n            An instantiated ``DataCatalog`` containing all specified\n            data sets, created and ready to use.\n\n        Raises:\n            DatasetError: When the method fails to create any of the data\n                sets from their config.\n            DatasetNotFoundError: When `load_versions` refers to a dataset that doesn\'t\n                exist in the catalog.\n\n        Example:\n        ::\n\n            >>> config = {\n            >>>     "cars": {\n            >>>         "type": "pandas.CSVDataSet",\n            >>>         "filepath": "cars.csv",\n            >>>         "save_args": {\n            >>>             "index": False\n            >>>         }\n            >>>     },\n            >>>     "boats": {\n            >>>         "type": "pandas.CSVDataSet",\n            >>>         "filepath": "s3://aws-bucket-name/boats.csv",\n            >>>         "credentials": "boats_credentials",\n            >>>         "save_args": {\n            >>>             "index": False\n            >>>         }\n            >>>     }\n            >>> }\n            >>>\n            >>> credentials = {\n            >>>     "boats_credentials": {\n            >>>         "client_kwargs": {\n            >>>             "aws_access_key_id": "<your key id>",\n            >>>             "aws_secret_access_key": "<your secret>"\n            >>>         }\n            >>>      }\n            >>> }\n            >>>\n            >>> catalog = DataCatalog.from_config(config, credentials)\n            >>>\n            >>> df = catalog.load("cars")\n            >>> catalog.save("boats", df)\n        '
        data_sets = {}
        dataset_patterns = {}
        catalog = copy.deepcopy(catalog) or {}
        credentials = copy.deepcopy(credentials) or {}
        save_version = save_version or generate_timestamp()
        load_versions = copy.deepcopy(load_versions) or {}
        layers: dict[str, set[str]] = defaultdict(set)
        for (ds_name, ds_config) in catalog.items():
            ds_config = _resolve_credentials(ds_config, credentials)
            if cls._is_pattern(ds_name):
                dataset_patterns[ds_name] = ds_config
            else:
                if 'layer' in ds_config:
                    import warnings
                    warnings.warn("Defining the 'layer' attribute at the top level is deprecated and will be removed in Kedro 0.19.0. Please move 'layer' inside the 'metadata' -> 'kedro-viz' attributes. See https://docs.kedro.org/en/latest/visualisation/kedro-viz_visualisation.html#visualise-layers for more information.", KedroDeprecationWarning)
                ds_layer = ds_config.pop('layer', None)
                if ds_layer is not None:
                    layers[ds_layer].add(ds_name)
                data_sets[ds_name] = AbstractDataset.from_config(ds_name, ds_config, load_versions.get(ds_name), save_version)
        dataset_layers = layers or None
        sorted_patterns = cls._sort_patterns(dataset_patterns)
        missing_keys = [key for key in load_versions.keys() if not (key in catalog or cls._match_pattern(sorted_patterns, key))]
        if missing_keys:
            raise DatasetNotFoundError(f"'load_versions' keys [{', '.join(sorted(missing_keys))}] are not found in the catalog.")
        return cls(data_sets=data_sets, layers=dataset_layers, dataset_patterns=sorted_patterns, load_versions=load_versions, save_version=save_version)

    @staticmethod
    def _is_pattern(pattern: str):
        if False:
            for i in range(10):
                print('nop')
        "Check if a given string is a pattern. Assume that any name with '{' is a pattern."
        return '{' in pattern

    @staticmethod
    def _match_pattern(data_set_patterns: Patterns, data_set_name: str) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Match a dataset name against patterns in a dictionary.'
        matches = (pattern for pattern in data_set_patterns.keys() if parse(pattern, data_set_name))
        return next(matches, None)

    @classmethod
    def _sort_patterns(cls, data_set_patterns: Patterns) -> dict[str, dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Sort a dictionary of dataset patterns according to parsing rules.\n\n        In order:\n\n        1. Decreasing specificity (number of characters outside the curly brackets)\n        2. Decreasing number of placeholders (number of curly bracket pairs)\n        3. Alphabetically\n        '
        sorted_keys = sorted(data_set_patterns, key=lambda pattern: (-cls._specificity(pattern), -pattern.count('{'), pattern))
        return {key: data_set_patterns[key] for key in sorted_keys}

    @staticmethod
    def _specificity(pattern: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Helper function to check the length of exactly matched characters not inside brackets.\n\n        Example:\n        ::\n\n            >>> specificity("{namespace}.companies") = 10\n            >>> specificity("{namespace}.{dataset}") = 1\n            >>> specificity("france.companies") = 16\n        '
        result = re.sub('\\{.*?\\}', '', pattern)
        return len(result)

    def _get_dataset(self, data_set_name: str, version: Version=None, suggest: bool=True) -> AbstractDataset:
        if False:
            print('Hello World!')
        matched_pattern = self._match_pattern(self._dataset_patterns, data_set_name)
        if data_set_name not in self._data_sets and matched_pattern:
            config_copy = copy.deepcopy(self._dataset_patterns[matched_pattern])
            data_set_config = self._resolve_config(data_set_name, matched_pattern, config_copy)
            ds_layer = data_set_config.pop('layer', None)
            if ds_layer:
                self.layers = self.layers or {}
                self.layers.setdefault(ds_layer, set()).add(data_set_name)
            data_set = AbstractDataset.from_config(data_set_name, data_set_config, self._load_versions.get(data_set_name), self._save_version)
            if self._specificity(matched_pattern) == 0:
                self._logger.warning("Config from the dataset factory pattern '%s' in the catalog will be used to override the default MemoryDataset creation for the dataset '%s'", matched_pattern, data_set_name)
            self.add(data_set_name, data_set)
        if data_set_name not in self._data_sets:
            error_msg = f"Dataset '{data_set_name}' not found in the catalog"
            if suggest:
                matches = difflib.get_close_matches(data_set_name, self._data_sets.keys())
                if matches:
                    suggestions = ', '.join(matches)
                    error_msg += f' - did you mean one of these instead: {suggestions}'
            raise DatasetNotFoundError(error_msg)
        data_set = self._data_sets[data_set_name]
        if version and isinstance(data_set, AbstractVersionedDataset):
            data_set = data_set._copy(_version=version)
        return data_set

    def __contains__(self, data_set_name):
        if False:
            print('Hello World!')
        'Check if an item is in the catalog as a materialised dataset or pattern'
        matched_pattern = self._match_pattern(self._dataset_patterns, data_set_name)
        if data_set_name in self._data_sets or matched_pattern:
            return True
        return False

    @classmethod
    def _resolve_config(cls, data_set_name: str, matched_pattern: str, config: dict) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Get resolved AbstractDataset from a factory config'
        result = parse(matched_pattern, data_set_name)
        if isinstance(config, dict):
            for (key, value) in config.items():
                config[key] = cls._resolve_config(data_set_name, matched_pattern, value)
        elif isinstance(config, (list, tuple)):
            config = [cls._resolve_config(data_set_name, matched_pattern, value) for value in config]
        elif isinstance(config, str) and '}' in config:
            try:
                config = str(config).format_map(result.named)
            except KeyError as exc:
                raise DatasetError(f"Unable to resolve '{config}' from the pattern '{matched_pattern}'. Keys used in the configuration should be present in the dataset factory pattern.") from exc
        return config

    def load(self, name: str, version: str=None) -> Any:
        if False:
            print('Hello World!')
        'Loads a registered data set.\n\n        Args:\n            name: A data set to be loaded.\n            version: Optional argument for concrete data version to be loaded.\n                Works only with versioned datasets.\n\n        Returns:\n            The loaded data as configured.\n\n        Raises:\n            DatasetNotFoundError: When a data set with the given name\n                has not yet been registered.\n\n        Example:\n        ::\n\n            >>> from kedro.io import DataCatalog\n            >>> from kedro.extras.datasets.pandas import CSVDataSet\n            >>>\n            >>> cars = CSVDataSet(filepath="cars.csv",\n            >>>                   load_args=None,\n            >>>                   save_args={"index": False})\n            >>> io = DataCatalog(data_sets={\'cars\': cars})\n            >>>\n            >>> df = io.load("cars")\n        '
        load_version = Version(version, None) if version else None
        dataset = self._get_dataset(name, version=load_version)
        self._logger.info("Loading data from '%s' (%s)...", name, type(dataset).__name__)
        result = dataset.load()
        return result

    def save(self, name: str, data: Any) -> None:
        if False:
            return 10
        'Save data to a registered data set.\n\n        Args:\n            name: A data set to be saved to.\n            data: A data object to be saved as configured in the registered\n                data set.\n\n        Raises:\n            DatasetNotFoundError: When a data set with the given name\n                has not yet been registered.\n\n        Example:\n        ::\n\n            >>> import pandas as pd\n            >>>\n            >>> from kedro.extras.datasets.pandas import CSVDataSet\n            >>>\n            >>> cars = CSVDataSet(filepath="cars.csv",\n            >>>                   load_args=None,\n            >>>                   save_args={"index": False})\n            >>> io = DataCatalog(data_sets={\'cars\': cars})\n            >>>\n            >>> df = pd.DataFrame({\'col1\': [1, 2],\n            >>>                    \'col2\': [4, 5],\n            >>>                    \'col3\': [5, 6]})\n            >>> io.save("cars", df)\n        '
        dataset = self._get_dataset(name)
        self._logger.info("Saving data to '%s' (%s)...", name, type(dataset).__name__)
        dataset.save(data)

    def exists(self, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether registered data set exists by calling its `exists()`\n        method. Raises a warning and returns False if `exists()` is not\n        implemented.\n\n        Args:\n            name: A data set to be checked.\n\n        Returns:\n            Whether the data set output exists.\n\n        '
        try:
            dataset = self._get_dataset(name)
        except DatasetNotFoundError:
            return False
        return dataset.exists()

    def release(self, name: str):
        if False:
            return 10
        'Release any cached data associated with a data set\n\n        Args:\n            name: A data set to be checked.\n\n        Raises:\n            DatasetNotFoundError: When a data set with the given name\n                has not yet been registered.\n        '
        dataset = self._get_dataset(name)
        dataset.release()

    def add(self, data_set_name: str, data_set: AbstractDataset, replace: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a new ``AbstractDataset`` object to the ``DataCatalog``.\n\n        Args:\n            data_set_name: A unique data set name which has not been\n                registered yet.\n            data_set: A data set object to be associated with the given data\n                set name.\n            replace: Specifies whether to replace an existing dataset\n                with the same name is allowed.\n\n        Raises:\n            DatasetAlreadyExistsError: When a data set with the same name\n                has already been registered.\n\n        Example:\n        ::\n\n            >>> from kedro.extras.datasets.pandas import CSVDataSet\n            >>>\n            >>> io = DataCatalog(data_sets={\n            >>>                   \'cars\': CSVDataSet(filepath="cars.csv")\n            >>>                  })\n            >>>\n            >>> io.add("boats", CSVDataSet(filepath="boats.csv"))\n        '
        if data_set_name in self._data_sets:
            if replace:
                self._logger.warning("Replacing dataset '%s'", data_set_name)
            else:
                raise DatasetAlreadyExistsError(f"Dataset '{data_set_name}' has already been registered")
        self._data_sets[data_set_name] = data_set
        self.datasets = _FrozenDatasets(self.datasets, {data_set_name: data_set})

    def add_all(self, data_sets: dict[str, AbstractDataset], replace: bool=False) -> None:
        if False:
            return 10
        'Adds a group of new data sets to the ``DataCatalog``.\n\n        Args:\n            data_sets: A dictionary of dataset names and dataset\n                instances.\n            replace: Specifies whether to replace an existing dataset\n                with the same name is allowed.\n\n        Raises:\n            DatasetAlreadyExistsError: When a data set with the same name\n                has already been registered.\n\n        Example:\n        ::\n\n            >>> from kedro.extras.datasets.pandas import CSVDataSet, ParquetDataSet\n            >>>\n            >>> io = DataCatalog(data_sets={\n            >>>                   "cars": CSVDataSet(filepath="cars.csv")\n            >>>                  })\n            >>> additional = {\n            >>>     "planes": ParquetDataSet("planes.parq"),\n            >>>     "boats": CSVDataSet(filepath="boats.csv")\n            >>> }\n            >>>\n            >>> io.add_all(additional)\n            >>>\n            >>> assert io.list() == ["cars", "planes", "boats"]\n        '
        for (name, data_set) in data_sets.items():
            self.add(name, data_set, replace)

    def add_feed_dict(self, feed_dict: dict[str, Any], replace: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Adds instances of ``MemoryDataset``, containing the data provided\n        through feed_dict.\n\n        Args:\n            feed_dict: A feed dict with data to be added in memory.\n            replace: Specifies whether to replace an existing dataset\n                with the same name is allowed.\n\n        Example:\n        ::\n\n            >>> import pandas as pd\n            >>>\n            >>> df = pd.DataFrame({\'col1\': [1, 2],\n            >>>                    \'col2\': [4, 5],\n            >>>                    \'col3\': [5, 6]})\n            >>>\n            >>> io = DataCatalog()\n            >>> io.add_feed_dict({\n            >>>     \'data\': df\n            >>> }, replace=True)\n            >>>\n            >>> assert io.load("data").equals(df)\n        '
        for data_set_name in feed_dict:
            if isinstance(feed_dict[data_set_name], AbstractDataset):
                data_set = feed_dict[data_set_name]
            else:
                data_set = MemoryDataset(data=feed_dict[data_set_name])
            self.add(data_set_name, data_set, replace)

    def list(self, regex_search: str | None=None) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        "\n        List of all dataset names registered in the catalog.\n        This can be filtered by providing an optional regular expression\n        which will only return matching keys.\n\n        Args:\n            regex_search: An optional regular expression which can be provided\n                to limit the data sets returned by a particular pattern.\n        Returns:\n            A list of dataset names available which match the\n            `regex_search` criteria (if provided). All data set names are returned\n            by default.\n\n        Raises:\n            SyntaxError: When an invalid regex filter is provided.\n\n        Example:\n        ::\n\n            >>> io = DataCatalog()\n            >>> # get data sets where the substring 'raw' is present\n            >>> raw_data = io.list(regex_search='raw')\n            >>> # get data sets which start with 'prm' or 'feat'\n            >>> feat_eng_data = io.list(regex_search='^(prm|feat)')\n            >>> # get data sets which end with 'time_series'\n            >>> models = io.list(regex_search='.+time_series$')\n        "
        if regex_search is None:
            return list(self._data_sets.keys())
        if not regex_search.strip():
            self._logger.warning('The empty string will not match any data sets')
            return []
        try:
            pattern = re.compile(regex_search, flags=re.IGNORECASE)
        except re.error as exc:
            raise SyntaxError(f"Invalid regular expression provided: '{regex_search}'") from exc
        return [dset_name for dset_name in self._data_sets if pattern.search(dset_name)]

    def shallow_copy(self) -> DataCatalog:
        if False:
            return 10
        'Returns a shallow copy of the current object.\n\n        Returns:\n            Copy of the current object.\n        '
        return DataCatalog(data_sets=self._data_sets, layers=self.layers, dataset_patterns=self._dataset_patterns, load_versions=self._load_versions, save_version=self._save_version)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return (self._data_sets, self.layers, self._dataset_patterns) == (other._data_sets, other.layers, other._dataset_patterns)

    def confirm(self, name: str) -> None:
        if False:
            while True:
                i = 10
        'Confirm a dataset by its name.\n\n        Args:\n            name: Name of the dataset.\n        Raises:\n            DatasetError: When the dataset does not have `confirm` method.\n\n        '
        self._logger.info("Confirming dataset '%s'", name)
        data_set = self._get_dataset(name)
        if hasattr(data_set, 'confirm'):
            data_set.confirm()
        else:
            raise DatasetError(f"Dataset '{name}' does not have 'confirm' method")