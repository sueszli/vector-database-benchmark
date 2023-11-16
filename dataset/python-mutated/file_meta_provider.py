import itertools
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
if TYPE_CHECKING:
    import pyarrow
logger = logging.getLogger(__name__)

@DeveloperAPI
class FileMetadataProvider:
    """Abstract callable that provides metadata for the files of a single dataset block.

    Current subclasses:
        - :class:`BaseFileMetadataProvider`
        - :class:`ParquetMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], **kwargs) -> BlockMetadata:
        if False:
            return 10
        'Resolves and returns block metadata for files in the given paths.\n\n        All file paths provided should belong to a single dataset block.\n\n        Args:\n            paths: The file paths for a single dataset block.\n            schema: The user-provided or inferred schema for the given paths,\n                if any.\n\n        Returns:\n            BlockMetadata aggregated across the given paths.\n        '
        raise NotImplementedError

    def __call__(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], **kwargs) -> BlockMetadata:
        if False:
            return 10
        return self._get_block_metadata(paths, schema, **kwargs)

@DeveloperAPI
class BaseFileMetadataProvider(FileMetadataProvider):
    """Abstract callable that provides metadata for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations that reuse the base :meth:`~ray.data.Datasource.prepare_read`
    method.

    Also supports file and file size discovery in input directory paths.

    Current subclasses:
        - :class:`DefaultFileMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, rows_per_file: Optional[int], file_sizes: List[Optional[int]]) -> BlockMetadata:
        if False:
            print('Hello World!')
        'Resolves and returns block metadata for files of a single dataset block.\n\n        Args:\n            paths: The file paths for a single dataset block. These\n                paths will always be a subset of those previously returned from\n                :meth:`.expand_paths`.\n            schema: The user-provided or inferred schema for the given file\n                paths, if any.\n            rows_per_file: The fixed number of rows per input file, or None.\n            file_sizes: Optional file size per input file previously returned\n                from :meth:`.expand_paths`, where `file_sizes[i]` holds the size of\n                the file at `paths[i]`.\n\n        Returns:\n            BlockMetadata aggregated across the given file paths.\n        '
        raise NotImplementedError

    def expand_paths(self, paths: List[str], filesystem: Optional['pyarrow.fs.FileSystem'], partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        if False:
            print('Hello World!')
        'Expands all paths into concrete file paths by walking directories.\n\n        Also returns a sidecar of file sizes.\n\n        The input paths must be normalized for compatibility with the input\n        filesystem prior to invocation.\n\n        Args:\n            paths: A list of file and/or directory paths compatible with the\n                given filesystem.\n            filesystem: The filesystem implementation that should be used for\n                expanding all paths and reading their files.\n            ignore_missing_paths: If True, ignores any file paths in ``paths`` that\n                are not found. Defaults to False.\n\n        Returns:\n            An iterator of `(file_path, file_size)` pairs. None may be returned for the\n            file size if it is either unknown or will be fetched later by\n            `_get_block_metadata()`, but the length of\n            both lists must be equal.\n        '
        raise NotImplementedError

@DeveloperAPI
class DefaultFileMetadataProvider(BaseFileMetadataProvider):
    """Default metadata provider for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations that reuse the base `prepare_read` method.

    Calculates block size in bytes as the sum of its constituent file sizes,
    and assumes a fixed number of rows per file.
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, rows_per_file: Optional[int], file_sizes: List[Optional[int]]) -> BlockMetadata:
        if False:
            i = 10
            return i + 15
        if rows_per_file is None:
            num_rows = None
        else:
            num_rows = len(paths) * rows_per_file
        return BlockMetadata(num_rows=num_rows, size_bytes=None if None in file_sizes else int(sum(file_sizes)), schema=schema, input_files=paths, exec_stats=None)

    def expand_paths(self, paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        if False:
            while True:
                i = 10
        yield from _expand_paths(paths, filesystem, partitioning, ignore_missing_paths)

@DeveloperAPI
class FastFileMetadataProvider(DefaultFileMetadataProvider):
    """Fast Metadata provider for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations.

    Offers improved performance vs.
    :class:`DefaultFileMetadataProvider`
    by skipping directory path expansion and file size collection.
    While this performance improvement may be negligible for local filesystems,
    it can be substantial for cloud storage service providers.

    This should only be used when all input paths exist and are known to be files.
    """

    def expand_paths(self, paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        if False:
            return 10
        if ignore_missing_paths:
            raise ValueError('`ignore_missing_paths` cannot be set when used with `FastFileMetadataProvider`. All paths must exist when using `FastFileMetadataProvider`.')
        logger.warning(f'Skipping expansion of {len(paths)} path(s). If your paths contain directories or if file size collection is required, try rerunning this read with `meta_provider=DefaultFileMetadataProvider()`.')
        yield from zip(paths, itertools.repeat(None, len(paths)))

@DeveloperAPI
class ParquetMetadataProvider(FileMetadataProvider):
    """Abstract callable that provides block metadata for Arrow Parquet file fragments.

    All file fragments should belong to a single dataset block.

    Supports optional pre-fetching of ordered metadata for all file fragments in
    a single batch to help optimize metadata resolution.

    Current subclasses:
        - :class:`~ray.data.datasource.file_meta_provider.DefaultParquetMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, num_fragments: int, prefetched_metadata: Optional[List[Any]]) -> BlockMetadata:
        if False:
            i = 10
            return i + 15
        'Resolves and returns block metadata for files of a single dataset block.\n\n        Args:\n            paths: The file paths for a single dataset block.\n            schema: The user-provided or inferred schema for the given file\n                paths, if any.\n            num_fragments: The number of Parquet file fragments derived from the input\n                file paths.\n            prefetched_metadata: Metadata previously returned from\n                `prefetch_file_metadata()` for each file fragment, where\n                `prefetched_metadata[i]` contains the metadata for `fragments[i]`.\n\n        Returns:\n            BlockMetadata aggregated across the given file paths.\n        '
        raise NotImplementedError

    def prefetch_file_metadata(self, fragments: List['pyarrow.dataset.ParquetFileFragment'], **ray_remote_args) -> Optional[List[Any]]:
        if False:
            print('Hello World!')
        "Pre-fetches file metadata for all Parquet file fragments in a single batch.\n\n        Subsets of the metadata returned will be provided as input to\n        subsequent calls to :meth:`~FileMetadataProvider._get_block_metadata` together\n        with their corresponding Parquet file fragments.\n\n        Implementations that don't support pre-fetching file metadata shouldn't\n        override this method.\n\n        Args:\n            fragments: The Parquet file fragments to fetch metadata for.\n\n        Returns:\n            Metadata resolved for each input file fragment, or `None`. Metadata\n            must be returned in the same order as all input file fragments, such\n            that `metadata[i]` always contains the metadata for `fragments[i]`.\n        "
        return None

@DeveloperAPI
class DefaultParquetMetadataProvider(ParquetMetadataProvider):
    """The default file metadata provider for ParquetDatasource.

    Aggregates total block bytes and number of rows using the Parquet file metadata
    associated with a list of Arrow Parquet dataset file fragments.
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, num_fragments: int, prefetched_metadata: Optional[List['pyarrow.parquet.FileMetaData']]) -> BlockMetadata:
        if False:
            print('Hello World!')
        if prefetched_metadata is not None and len(prefetched_metadata) == num_fragments and all((m is not None for m in prefetched_metadata)):
            block_metadata = BlockMetadata(num_rows=sum((m.num_rows for m in prefetched_metadata)), size_bytes=sum((sum((m.row_group(i).total_byte_size for i in range(m.num_row_groups))) for m in prefetched_metadata)), schema=schema, input_files=paths, exec_stats=None)
        else:
            block_metadata = BlockMetadata(num_rows=None, size_bytes=None, schema=schema, input_files=paths, exec_stats=None)
        return block_metadata

    def prefetch_file_metadata(self, fragments: List['pyarrow.dataset.ParquetFileFragment'], **ray_remote_args) -> Optional[List['pyarrow.parquet.FileMetaData']]:
        if False:
            for i in range(10):
                print('nop')
        from ray.data.datasource.parquet_datasource import FRAGMENTS_PER_META_FETCH, PARALLELIZE_META_FETCH_THRESHOLD, _fetch_metadata, _fetch_metadata_serialization_wrapper, _SerializedFragment
        if len(fragments) > PARALLELIZE_META_FETCH_THRESHOLD:
            fragments = [_SerializedFragment(fragment) for fragment in fragments]
            return list(_fetch_metadata_parallel(fragments, _fetch_metadata_serialization_wrapper, FRAGMENTS_PER_META_FETCH, **ray_remote_args))
        else:
            return _fetch_metadata(fragments)

def _handle_read_os_error(error: OSError, paths: Union[str, List[str]]) -> str:
    if False:
        print('Hello World!')
    aws_error_pattern = '^(?:(.*)AWS Error \\[code \\d+\\]: No response body\\.(.*))|(?:(.*)AWS Error UNKNOWN \\(HTTP status 400\\) during HeadObject operation: No response body\\.(.*))|(?:(.*)AWS Error ACCESS_DENIED during HeadObject operation: No response body\\.(.*))$'
    if re.match(aws_error_pattern, str(error)):
        if isinstance(paths, str):
            paths = f'"{paths}"'
        raise OSError(f'Failing to read AWS S3 file(s): {paths}. Please check that file exists and has properly configured access. You can also run AWS CLI command to get more detailed error message (e.g., aws s3 ls <file-name>). See https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/index.html and https://docs.ray.io/en/latest/data/creating-datasets.html#reading-from-remote-storage for more information.')
    else:
        raise error

def _expand_paths(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning], ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    if False:
        return 10
    'Get the file sizes for all provided file paths.'
    from pyarrow.fs import LocalFileSystem
    from ray.data.datasource.file_based_datasource import FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD
    from ray.data.datasource.path_util import _unwrap_protocol
    if len(paths) < FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD or isinstance(filesystem, LocalFileSystem):
        yield from _get_file_infos_serial(paths, filesystem, ignore_missing_paths)
    else:
        common_path = os.path.commonpath(paths)
        if partitioning is not None and common_path == _unwrap_protocol(partitioning.base_dir) or all((str(pathlib.Path(path).parent) == common_path for path in paths)):
            yield from _get_file_infos_common_path_prefix(paths, common_path, filesystem, ignore_missing_paths)
        else:
            yield from _get_file_infos_parallel(paths, filesystem, ignore_missing_paths)

def _get_file_infos_serial(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    if False:
        while True:
            i = 10
    for path in paths:
        yield from _get_file_infos(path, filesystem, ignore_missing_paths)

def _get_file_infos_common_path_prefix(paths: List[str], common_path: str, filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    if False:
        i = 10
        return i + 15
    path_to_size = {path: None for path in paths}
    for (path, file_size) in _get_file_infos(common_path, filesystem, ignore_missing_paths):
        if path in path_to_size:
            path_to_size[path] = file_size
    have_missing_path = False
    for path in paths:
        if path_to_size[path] is None:
            logger.debug(f'Finding path {path} not have file size metadata. Fall back to get files metadata in parallel for all paths.')
            have_missing_path = True
            break
    if have_missing_path:
        yield from _get_file_infos_parallel(paths, filesystem, ignore_missing_paths)
    else:
        for path in paths:
            yield (path, path_to_size[path])

def _get_file_infos_parallel(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    if False:
        for i in range(10):
            print('nop')
    from ray.data.datasource.file_based_datasource import PATHS_PER_FILE_SIZE_FETCH_TASK, _unwrap_s3_serialization_workaround, _wrap_s3_serialization_workaround
    logger.warning(f'Expanding {len(paths)} path(s). This may be a HIGH LATENCY operation on some cloud storage services. Moving all the paths to a common parent directory will lead to faster metadata fetching.')
    filesystem = _wrap_s3_serialization_workaround(filesystem)

    def _file_infos_fetcher(paths: List[str]) -> List[Tuple[str, int]]:
        if False:
            i = 10
            return i + 15
        fs = _unwrap_s3_serialization_workaround(filesystem)
        return list(itertools.chain.from_iterable((_get_file_infos(path, fs, ignore_missing_paths) for path in paths)))
    yield from _fetch_metadata_parallel(paths, _file_infos_fetcher, PATHS_PER_FILE_SIZE_FETCH_TASK)
Uri = TypeVar('Uri')
Meta = TypeVar('Meta')

def _fetch_metadata_parallel(uris: List[Uri], fetch_func: Callable[[List[Uri]], List[Meta]], desired_uris_per_task: int, **ray_remote_args) -> Iterator[Meta]:
    if False:
        while True:
            i = 10
    'Fetch file metadata in parallel using Ray tasks.'
    remote_fetch_func = cached_remote_fn(fetch_func, num_cpus=0.5)
    if ray_remote_args:
        remote_fetch_func = remote_fetch_func.options(**ray_remote_args)
    parallelism = max(len(uris) // desired_uris_per_task, 2)
    metadata_fetch_bar = ProgressBar('Metadata Fetch Progress', total=parallelism)
    fetch_tasks = []
    for uri_chunk in np.array_split(uris, parallelism):
        if len(uri_chunk) == 0:
            continue
        fetch_tasks.append(remote_fetch_func.remote(uri_chunk))
    results = metadata_fetch_bar.fetch_until_complete(fetch_tasks)
    yield from itertools.chain.from_iterable(results)

def _get_file_infos(path: str, filesystem: 'pyarrow.fs.FileSystem', ignore_missing_path: bool=False) -> List[Tuple[str, int]]:
    if False:
        for i in range(10):
            print('nop')
    'Get the file info for all files at or under the provided path.'
    from pyarrow.fs import FileType
    file_infos = []
    try:
        file_info = filesystem.get_file_info(path)
    except OSError as e:
        _handle_read_os_error(e, path)
    if file_info.type == FileType.Directory:
        for (file_path, file_size) in _expand_directory(path, filesystem):
            file_infos.append((file_path, file_size))
    elif file_info.type == FileType.File:
        file_infos.append((path, file_info.size))
    elif file_info.type == FileType.NotFound and ignore_missing_path:
        pass
    else:
        raise FileNotFoundError(path)
    return file_infos

def _expand_directory(path: str, filesystem: 'pyarrow.fs.FileSystem', exclude_prefixes: Optional[List[str]]=None, ignore_missing_path: bool=False) -> List[Tuple[str, int]]:
    if False:
        print('Hello World!')
    '\n    Expand the provided directory path to a list of file paths.\n\n    Args:\n        path: The directory path to expand.\n        filesystem: The filesystem implementation that should be used for\n            reading these files.\n        exclude_prefixes: The file relative path prefixes that should be\n            excluded from the returned file set. Default excluded prefixes are\n            "." and "_".\n\n    Returns:\n        An iterator of (file_path, file_size) tuples.\n    '
    if exclude_prefixes is None:
        exclude_prefixes = ['.', '_']
    from pyarrow.fs import FileSelector
    selector = FileSelector(path, recursive=True, allow_not_found=ignore_missing_path)
    files = filesystem.get_file_info(selector)
    base_path = selector.base_dir
    out = []
    for file_ in files:
        if not file_.is_file:
            continue
        file_path = file_.path
        if not file_path.startswith(base_path):
            continue
        relative = file_path[len(base_path):]
        if any((relative.startswith(prefix) for prefix in exclude_prefixes)):
            continue
        out.append((file_path, file_.size))
    return sorted(out)