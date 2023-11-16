import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
if TYPE_CHECKING:
    import pyarrow

@DeveloperAPI
class PartitionStyle(str, Enum):
    """Supported dataset partition styles.

    Inherits from `str` to simplify plain text serialization/deserialization.

    Examples:
        >>> # Serialize to JSON text.
        >>> json.dumps(PartitionStyle.HIVE)  # doctest: +SKIP
        '"hive"'

        >>> # Deserialize from JSON text.
        >>> PartitionStyle(json.loads('"hive"'))  # doctest: +SKIP
        <PartitionStyle.HIVE: 'hive'>
    """
    HIVE = 'hive'
    DIRECTORY = 'dir'

@DeveloperAPI
@dataclass
class Partitioning:
    """Partition scheme used to describe path-based partitions.

    Path-based partition formats embed all partition keys and values directly in
    their dataset file paths.

    For example, to read a dataset with
    `Hive-style partitions <https://athena.guide/articles/hive-style-partitioning/>`_:

        >>> import ray
        >>> from ray.data.datasource.partitioning import Partitioning
        >>> ds = ray.data.read_csv(
        ...     "s3://anonymous@ray-example-data/iris.csv",
        ...     partitioning=Partitioning("hive"),
        ... )

    Instead, if your files are arranged in a directory structure such as:

    .. code::

        root/dog/dog_0.jpeg
        root/dog/dog_1.jpeg
        ...

        root/cat/cat_0.jpeg
        root/cat/cat_1.jpeg
        ...

    Then you can use directory-based partitioning:

        >>> import ray
        >>> from ray.data.datasource.partitioning import Partitioning
        >>> root = "s3://anonymous@air-example-data/cifar-10/images"
        >>> partitioning = Partitioning("dir", field_names=["class"], base_dir=root)
        >>> ds = ray.data.read_images(root, partitioning=partitioning)
    """
    style: PartitionStyle
    base_dir: Optional[str] = None
    field_names: Optional[List[str]] = None
    filesystem: Optional['pyarrow.fs.FileSystem'] = None

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.base_dir is None:
            self.base_dir = ''
        self._normalized_base_dir = None
        self._resolved_filesystem = None

    @property
    def normalized_base_dir(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the base directory normalized for compatibility with a filesystem.'
        if self._normalized_base_dir is None:
            self._normalize_base_dir()
        return self._normalized_base_dir

    @property
    def resolved_filesystem(self) -> 'pyarrow.fs.FileSystem':
        if False:
            return 10
        'Returns the filesystem resolved for compatibility with a base directory.'
        if self._resolved_filesystem is None:
            self._normalize_base_dir()
        return self._resolved_filesystem

    def _normalize_base_dir(self):
        if False:
            i = 10
            return i + 15
        'Normalizes the partition base directory for compatibility with the\n        given filesystem.\n\n        This should be called once a filesystem has been resolved to ensure that this\n        base directory is correctly discovered at the root of all partitioned file\n        paths.\n        '
        from ray.data.datasource.path_util import _resolve_paths_and_filesystem
        (paths, self._resolved_filesystem) = _resolve_paths_and_filesystem(self.base_dir, self.filesystem)
        assert len(paths) == 1, f'Expected 1 normalized base directory, but found {len(paths)}'
        normalized_base_dir = paths[0]
        if len(normalized_base_dir) and (not normalized_base_dir.endswith('/')):
            normalized_base_dir += '/'
        self._normalized_base_dir = normalized_base_dir

@DeveloperAPI
class PathPartitionParser:
    """Partition parser for path-based partition formats.

    Path-based partition formats embed all partition keys and values directly in
    their dataset file paths.

    Two path partition formats are currently supported - `HIVE` and `DIRECTORY`.

    For `HIVE` Partitioning, all partition directories under the base directory
    will be discovered based on `{key1}={value1}/{key2}={value2}` naming
    conventions. Key/value pairs do not need to be presented in the same
    order across all paths. Directory names nested under the base directory that
    don't follow this naming condition will be considered unpartitioned. If a
    partition filter is defined, then it will be called with an empty input
    dictionary for each unpartitioned file.

    For `DIRECTORY` Partitioning, all directories under the base directory will
    be interpreted as partition values of the form `{value1}/{value2}`. An
    accompanying ordered list of partition field names must also be provided,
    where the order and length of all partition values must match the order and
    length of field names. Files stored directly in the base directory will
    be considered unpartitioned. If a partition filter is defined, then it will
    be called with an empty input dictionary for each unpartitioned file. For
    example, if the base directory is `"foo"`, then `"foo.csv"` and `"foo/bar.csv"`
    would be considered unpartitioned files but `"foo/bar/baz.csv"` would be associated
    with partition `"bar"`. If the base directory is undefined, then `"foo.csv"` would
    be unpartitioned, `"foo/bar.csv"` would be associated with partition `"foo"`, and
    "foo/bar/baz.csv" would be associated with partition `("foo", "bar")`.
    """

    @staticmethod
    def of(style: PartitionStyle=PartitionStyle.HIVE, base_dir: Optional[str]=None, field_names: Optional[List[str]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None) -> 'PathPartitionParser':
        if False:
            print('Hello World!')
        'Creates a path-based partition parser using a flattened argument list.\n\n        Args:\n            style: The partition style - may be either HIVE or DIRECTORY.\n            base_dir: "/"-delimited base directory to start searching for partitions\n                (exclusive). File paths outside of this directory will be considered\n                unpartitioned. Specify `None` or an empty string to search for\n                partitions in all file path directories.\n            field_names: The partition key names. Required for DIRECTORY partitioning.\n                Optional for HIVE partitioning. When non-empty, the order and length of\n                partition key field names must match the order and length of partition\n                directories discovered. Partition key field names are not required to\n                exist in the dataset schema.\n            filesystem: Filesystem that will be used for partition path file I/O.\n\n        Returns:\n            The new path-based partition parser.\n        '
        scheme = Partitioning(style, base_dir, field_names, filesystem)
        return PathPartitionParser(scheme)

    def __init__(self, partitioning: Partitioning):
        if False:
            return 10
        "Creates a path-based partition parser.\n\n        Args:\n            partitioning: The path-based partition scheme. The parser starts\n                searching for partitions from this scheme's base directory. File paths\n                outside the base directory will be considered unpartitioned. If the\n                base directory is `None` or an empty string then this will search for\n                partitions in all file path directories. Field names are required for\n                DIRECTORY partitioning, and optional for HIVE partitioning. When\n                non-empty, the order and length of partition key field names must match\n                the order and length of partition directories discovered.\n        "
        style = partitioning.style
        field_names = partitioning.field_names
        if style == PartitionStyle.DIRECTORY and (not field_names):
            raise ValueError('Directory partitioning requires a corresponding list of partition key field names. Please retry your request with one or more field names specified.')
        parsers = {PartitionStyle.HIVE: self._parse_hive_path, PartitionStyle.DIRECTORY: self._parse_dir_path}
        self._parser_fn: Callable[[str], Dict[str, str]] = parsers.get(style)
        if self._parser_fn is None:
            raise ValueError(f'Unsupported partition style: {style}. Supported styles: {parsers.keys()}')
        self._scheme = partitioning

    def __call__(self, path: str) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'Parses partition keys and values from a single file path.\n\n        Args:\n            path: Input file path to parse.\n        Returns:\n            Dictionary mapping directory partition keys to values from the input file\n            path. Returns an empty dictionary for unpartitioned files.\n        '
        dir_path = self._dir_path_trim_base(path)
        if dir_path is None:
            return {}
        return self._parser_fn(dir_path)

    @property
    def scheme(self) -> Partitioning:
        if False:
            i = 10
            return i + 15
        'Returns the partitioning for this parser.'
        return self._scheme

    def _dir_path_trim_base(self, path: str) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Trims the normalized base directory and returns the directory path.\n\n        Returns None if the path does not start with the normalized base directory.\n        Simply returns the directory path if the base directory is undefined.\n        '
        if not path.startswith(self._scheme.normalized_base_dir):
            return None
        path = path[len(self._scheme.normalized_base_dir):]
        return posixpath.dirname(path)

    def _parse_hive_path(self, dir_path: str) -> Dict[str, str]:
        if False:
            print('Hello World!')
        'Hive partition path parser.\n\n        Returns a dictionary mapping partition keys to values given a hive-style\n        partition path of the form "{key1}={value1}/{key2}={value2}/..." or an empty\n        dictionary for unpartitioned files.\n        '
        dirs = [d for d in dir_path.split('/') if d and d.count('=') == 1]
        kv_pairs = [d.split('=') for d in dirs] if dirs else []
        field_names = self._scheme.field_names
        if field_names and kv_pairs:
            if len(kv_pairs) != len(field_names):
                raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(kv_pairs)}: {kv_pairs}.')
            for (i, field_name) in enumerate(field_names):
                if kv_pairs[i][0] != field_name:
                    raise ValueError(f'Expected partition key {field_name} but found {kv_pairs[i][0]}')
        return dict(kv_pairs)

    def _parse_dir_path(self, dir_path: str) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        'Directory partition path parser.\n\n        Returns a dictionary mapping directory partition keys to values from a\n        partition path of the form "{value1}/{value2}/..." or an empty dictionary for\n        unpartitioned files.\n\n        Requires a corresponding ordered list of partition key field names to map the\n        correct key to each value.\n        '
        dirs = [d for d in dir_path.split('/') if d]
        field_names = self._scheme.field_names
        if dirs and len(dirs) != len(field_names):
            raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(dirs)}: {dirs}.')
        if not dirs:
            return {}
        return {field: directory for (field, directory) in zip(field_names, dirs) if field is not None}

@PublicAPI(stability='beta')
class PathPartitionFilter:
    """Partition filter for path-based partition formats.

    Used to explicitly keep or reject files based on a custom filter function that
    takes partition keys and values parsed from the file's path as input.
    """

    @staticmethod
    def of(filter_fn: Callable[[Dict[str, str]], bool], style: PartitionStyle=PartitionStyle.HIVE, base_dir: Optional[str]=None, field_names: Optional[List[str]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None) -> 'PathPartitionFilter':
        if False:
            while True:
                i = 10
        'Creates a path-based partition filter using a flattened argument list.\n\n        Args:\n            filter_fn: Callback used to filter partitions. Takes a dictionary mapping\n                partition keys to values as input. Unpartitioned files are denoted with\n                an empty input dictionary. Returns `True` to read a file for that\n                partition or `False` to skip it. Partition keys and values are always\n                strings read from the filesystem path. For example, this removes all\n                unpartitioned files:\n\n                .. code:: python\n\n                    lambda d: True if d else False\n\n                This raises an assertion error for any unpartitioned file found:\n\n                .. code:: python\n\n                    def do_assert(val, msg):\n                        assert val, msg\n\n                    lambda d: do_assert(d, "Expected all files to be partitioned!")\n\n                And this only reads files from January, 2022 partitions:\n\n                .. code:: python\n\n                    lambda d: d["month"] == "January" and d["year"] == "2022"\n\n            style: The partition style - may be either HIVE or DIRECTORY.\n            base_dir: "/"-delimited base directory to start searching for partitions\n                (exclusive). File paths outside of this directory will be considered\n                unpartitioned. Specify `None` or an empty string to search for\n                partitions in all file path directories.\n            field_names: The partition key names. Required for DIRECTORY partitioning.\n                Optional for HIVE partitioning. When non-empty, the order and length of\n                partition key field names must match the order and length of partition\n                directories discovered. Partition key field names are not required to\n                exist in the dataset schema.\n            filesystem: Filesystem that will be used for partition path file I/O.\n\n        Returns:\n            The new path-based partition filter.\n        '
        scheme = Partitioning(style, base_dir, field_names, filesystem)
        path_partition_parser = PathPartitionParser(scheme)
        return PathPartitionFilter(path_partition_parser, filter_fn)

    def __init__(self, path_partition_parser: PathPartitionParser, filter_fn: Callable[[Dict[str, str]], bool]):
        if False:
            for i in range(10):
                print('nop')
        'Creates a new path-based partition filter based on a parser.\n\n        Args:\n            path_partition_parser: The path-based partition parser.\n            filter_fn: Callback used to filter partitions. Takes a dictionary mapping\n                partition keys to values as input. Unpartitioned files are denoted with\n                an empty input dictionary. Returns `True` to read a file for that\n                partition or `False` to skip it. Partition keys and values are always\n                strings read from the filesystem path. For example, this removes all\n                unpartitioned files:\n                ``lambda d: True if d else False``\n                This raises an assertion error for any unpartitioned file found:\n                ``lambda d: assert d, "Expected all files to be partitioned!"``\n                And this only reads files from January, 2022 partitions:\n                ``lambda d: d["month"] == "January" and d["year"] == "2022"``\n        '
        self._parser = path_partition_parser
        self._filter_fn = filter_fn

    def __call__(self, paths: List[str]) -> List[str]:
        if False:
            return 10
        "Returns all paths that pass this partition scheme's partition filter.\n\n        If no partition filter is set, then returns all input paths. If a base\n        directory is set, then only paths under this base directory will be parsed\n        for partitions. All paths outside of this base directory will automatically\n        be considered unpartitioned, and passed into the filter function as empty\n        dictionaries.\n\n        Also normalizes the partition base directory for compatibility with the\n        given filesystem before applying the filter.\n\n        Args:\n            paths: Paths to pass through the partition filter function. All\n                paths should be normalized for compatibility with the given\n                filesystem.\n        Returns:\n            List of paths that pass the partition filter, or all paths if no\n            partition filter is defined.\n        "
        filtered_paths = paths
        if self._filter_fn is not None:
            filtered_paths = [path for path in paths if self._filter_fn(self._parser(path))]
        return filtered_paths

    @property
    def parser(self) -> PathPartitionParser:
        if False:
            return 10
        'Returns the path partition parser for this filter.'
        return self._parser