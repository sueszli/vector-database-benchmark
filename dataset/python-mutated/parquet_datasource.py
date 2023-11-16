import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Literal, Optional, Union
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.util import _check_pyarrow_version, _is_local_scheme
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource._default_metadata_providers import get_generic_metadata_provider
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.file_meta_provider import DefaultParquetMetadataProvider, ParquetMetadataProvider, _handle_read_os_error
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import _has_file_extension, _resolve_paths_and_filesystem
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    import pyarrow
    from pyarrow.dataset import ParquetFileFragment
logger = logging.getLogger(__name__)
FRAGMENTS_PER_META_FETCH = 6
PARALLELIZE_META_FETCH_THRESHOLD = 24
PARQUET_READER_ROW_BATCH_SIZE = 10000
FILE_READING_RETRY = 8
PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT = 5
PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND = 2
PARQUET_ENCODING_RATIO_ESTIMATE_SAMPLING_RATIO = 0.01
PARQUET_ENCODING_RATIO_ESTIMATE_MIN_NUM_SAMPLES = 2
PARQUET_ENCODING_RATIO_ESTIMATE_MAX_NUM_SAMPLES = 10
PARQUET_ENCODING_RATIO_ESTIMATE_NUM_ROWS = 1024

class _SerializedFragment:

    def __init__(self, frag: 'ParquetFileFragment'):
        if False:
            i = 10
            return i + 15
        self._data = cloudpickle.dumps((frag.format, frag.path, frag.filesystem, frag.partition_expression))

    def deserialize(self) -> 'ParquetFileFragment':
        if False:
            while True:
                i = 10
        import pyarrow.fs
        (file_format, path, filesystem, partition_expression) = cloudpickle.loads(self._data)
        return file_format.make_fragment(path, filesystem, partition_expression)

def _deserialize_fragments(serialized_fragments: List[_SerializedFragment]) -> List['pyarrow._dataset.ParquetFileFragment']:
    if False:
        for i in range(10):
            print('nop')
    return [p.deserialize() for p in serialized_fragments]

def _deserialize_fragments_with_retry(serialized_fragments: List[_SerializedFragment]) -> List['pyarrow._dataset.ParquetFileFragment']:
    if False:
        for i in range(10):
            print('nop')
    min_interval = 0
    final_exception = None
    for i in range(FILE_READING_RETRY):
        try:
            return _deserialize_fragments(serialized_fragments)
        except Exception as e:
            import random
            import time
            retry_timing = '' if i == FILE_READING_RETRY - 1 else f'Retry after {min_interval} sec. '
            log_only_show_in_1st_retry = '' if i else f'If earlier read attempt threw certain Exception, it may or may not be an issue depends on these retries succeed or not. serialized_fragments:{serialized_fragments}'
            logger.exception(f'{i + 1}th attempt to deserialize ParquetFileFragment failed. {retry_timing}{log_only_show_in_1st_retry}')
            if not min_interval:
                min_interval = 1 + random.random()
            time.sleep(min_interval)
            min_interval = min_interval * 2
            final_exception = e
    raise final_exception

@PublicAPI
class ParquetDatasource(Datasource):
    """Parquet datasource, for reading and writing Parquet files.

    The primary difference from ParquetBaseDatasource is that this uses
    PyArrow's `ParquetDataset` abstraction for dataset reads, and thus offers
    automatic Arrow dataset schema inference and row count collection at the
    cost of some potential performance and/or compatibility penalties.
    """

    def __init__(self, paths: Union[str, List[str]], *, columns: Optional[List[str]]=None, dataset_kwargs: Optional[Dict[str, Any]]=None, to_batch_kwargs: Optional[Dict[str, Any]]=None, _block_udf: Optional[Callable[[Block], Block]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None, schema: Optional[Union[type, 'pyarrow.lib.Schema']]=None, meta_provider: ParquetMetadataProvider=DefaultParquetMetadataProvider(), partition_filter: PathPartitionFilter=None, shuffle: Union[Literal['files'], None]=None, file_extensions: Optional[List[str]]=None):
        if False:
            i = 10
            return i + 15
        _check_pyarrow_version()
        import pyarrow as pa
        import pyarrow.parquet as pq
        self._supports_distributed_reads = not _is_local_scheme(paths)
        if not self._supports_distributed_reads and ray.util.client.ray.is_connected():
            raise ValueError("Because you're using Ray Client, read tasks scheduled on the Ray cluster can't access your local files. To fix this issue, store files in cloud storage or a distributed filesystem like NFS.")
        self._local_scheduling = None
        if not self._supports_distributed_reads:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            self._local_scheduling = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        (paths, filesystem) = _resolve_paths_and_filesystem(paths, filesystem)
        if partition_filter is not None or file_extensions is not None:
            default_meta_provider = get_generic_metadata_provider(file_extensions=None)
            (expanded_paths, _) = map(list, zip(*default_meta_provider.expand_paths(paths, filesystem)))
            paths = list(expanded_paths)
            if partition_filter is not None:
                paths = partition_filter(paths)
            if file_extensions is not None:
                paths = [path for path in paths if _has_file_extension(path, file_extensions)]
            filtered_paths = set(expanded_paths) - set(paths)
            if filtered_paths:
                logger.info(f'Filtered out the following paths: {filtered_paths}')
        if len(paths) == 1:
            paths = paths[0]
        if dataset_kwargs is None:
            dataset_kwargs = {}
        try:
            pq_ds = pq.ParquetDataset(paths, **dataset_kwargs, filesystem=filesystem, use_legacy_dataset=False)
        except OSError as e:
            _handle_read_os_error(e, paths)
        if schema is None:
            schema = pq_ds.schema
        if columns:
            schema = pa.schema([schema.field(column) for column in columns], schema.metadata)
        if _block_udf is not None:
            dummy_table = schema.empty_table()
            try:
                inferred_schema = _block_udf(dummy_table).schema
                inferred_schema = inferred_schema.with_metadata(schema.metadata)
            except Exception:
                logger.debug('Failed to infer schema of dataset by passing dummy table through UDF due to the following exception:', exc_info=True)
                inferred_schema = schema
        else:
            inferred_schema = schema
        try:
            prefetch_remote_args = {}
            if self._local_scheduling:
                prefetch_remote_args['scheduling_strategy'] = self._local_scheduling
            self._metadata = meta_provider.prefetch_file_metadata(pq_ds.fragments, **prefetch_remote_args) or []
        except OSError as e:
            _handle_read_os_error(e, paths)
        if to_batch_kwargs is None:
            to_batch_kwargs = {}
        self._pq_fragments = [_SerializedFragment(p) for p in pq_ds.fragments]
        self._pq_paths = [p.path for p in pq_ds.fragments]
        self._meta_provider = meta_provider
        self._inferred_schema = inferred_schema
        self._block_udf = _block_udf
        self._to_batches_kwargs = to_batch_kwargs
        self._columns = columns
        self._schema = schema
        self._encoding_ratio = self._estimate_files_encoding_ratio()
        self._file_metadata_shuffler = None
        if shuffle == 'files':
            self._file_metadata_shuffler = np.random.default_rng()

    def estimate_inmemory_data_size(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        total_size = 0
        for file_metadata in self._metadata:
            for row_group_idx in range(file_metadata.num_row_groups):
                row_group_metadata = file_metadata.row_group(row_group_idx)
                total_size += row_group_metadata.total_byte_size
        return total_size * self._encoding_ratio

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        if False:
            for i in range(10):
                print('nop')
        pq_metadata = self._metadata
        if len(pq_metadata) < len(self._pq_fragments):
            pq_metadata += [None] * (len(self._pq_fragments) - len(pq_metadata))
        if self._file_metadata_shuffler is not None:
            files_metadata = list(zip(self._pq_fragments, self._pq_paths, pq_metadata))
            shuffled_files_metadata = [files_metadata[i] for i in self._file_metadata_shuffler.permutation(len(files_metadata))]
            (pq_fragments, pq_paths, pq_metadata) = list(map(list, zip(*shuffled_files_metadata)))
        else:
            (pq_fragments, pq_paths, pq_metadata) = (self._pq_fragments, self._pq_paths, pq_metadata)
        read_tasks = []
        for (fragments, paths, metadata) in zip(np.array_split(pq_fragments, parallelism), np.array_split(pq_paths, parallelism), np.array_split(pq_metadata, parallelism)):
            if len(fragments) <= 0:
                continue
            meta = self._meta_provider(paths, self._inferred_schema, num_fragments=len(fragments), prefetched_metadata=metadata)
            if self._to_batches_kwargs.get('filter') is not None:
                meta.num_rows = None
            if meta.size_bytes is not None:
                meta.size_bytes = int(meta.size_bytes * self._encoding_ratio)
            if meta.num_rows and meta.size_bytes:
                row_size = meta.size_bytes / meta.num_rows
                max_parquet_reader_row_batch_size_bytes = DataContext.get_current().target_max_block_size // 10
                default_read_batch_size_rows = max(1, min(PARQUET_READER_ROW_BATCH_SIZE, max_parquet_reader_row_batch_size_bytes // row_size))
            else:
                default_read_batch_size_rows = PARQUET_READER_ROW_BATCH_SIZE
            (block_udf, to_batches_kwargs, columns, schema) = (self._block_udf, self._to_batches_kwargs, self._columns, self._schema)
            read_tasks.append(ReadTask(lambda f=fragments: _read_fragments(block_udf, to_batches_kwargs, default_read_batch_size_rows, columns, schema, f), meta))
        return read_tasks

    def _estimate_files_encoding_ratio(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Return an estimate of the Parquet files encoding ratio.\n\n        To avoid OOMs, it is safer to return an over-estimate than an underestimate.\n        '
        if not DataContext.get_current().decoding_size_estimation:
            return PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
        num_files = len(self._pq_fragments)
        num_samples = int(num_files * PARQUET_ENCODING_RATIO_ESTIMATE_SAMPLING_RATIO)
        min_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MIN_NUM_SAMPLES, num_files)
        max_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MAX_NUM_SAMPLES, num_files)
        num_samples = max(min(num_samples, max_num_samples), min_num_samples)
        file_samples = [self._pq_fragments[idx] for idx in np.linspace(0, num_files - 1, num_samples).astype(int).tolist()]
        sample_fragment = cached_remote_fn(_sample_fragment)
        futures = []
        scheduling = self._local_scheduling or 'SPREAD'
        for sample in file_samples:
            futures.append(sample_fragment.options(scheduling_strategy=scheduling).remote(self._to_batches_kwargs, self._columns, self._schema, sample))
        sample_bar = ProgressBar('Parquet Files Sample', len(futures))
        sample_ratios = sample_bar.fetch_until_complete(futures)
        sample_bar.close()
        ratio = np.mean(sample_ratios)
        logger.debug(f'Estimated Parquet encoding ratio from sampling is {ratio}.')
        return max(ratio, PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND)

    def get_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a human-readable name for this datasource.\n        This will be used as the names of the read tasks.\n        Note: overrides the base `ParquetBaseDatasource` method.\n        '
        return 'Parquet'

    @property
    def supports_distributed_reads(self) -> bool:
        if False:
            while True:
                i = 10
        return self._supports_distributed_reads

def _read_fragments(block_udf, to_batches_kwargs, default_read_batch_size_rows, columns, schema, serialized_fragments: List[_SerializedFragment]) -> Iterator['pyarrow.Table']:
    if False:
        i = 10
        return i + 15
    from ray.data.extensions.tensor_extension import ArrowTensorType
    fragments: List['pyarrow._dataset.ParquetFileFragment'] = _deserialize_fragments_with_retry(serialized_fragments)
    assert len(fragments) > 0
    import pyarrow as pa
    from pyarrow.dataset import _get_partition_keys
    logger.debug(f'Reading {len(fragments)} parquet fragments')
    use_threads = to_batches_kwargs.pop('use_threads', False)
    batch_size = to_batches_kwargs.pop('batch_size', default_read_batch_size_rows)
    for fragment in fragments:
        part = _get_partition_keys(fragment.partition_expression)
        batches = fragment.to_batches(use_threads=use_threads, columns=columns, schema=schema, batch_size=batch_size, **to_batches_kwargs)
        for batch in batches:
            table = pa.Table.from_batches([batch], schema=schema)
            if part:
                for (col, value) in part.items():
                    table = table.set_column(table.schema.get_field_index(col), col, pa.array([value] * len(table)))
            if table.num_rows > 0:
                if block_udf is not None:
                    yield block_udf(table)
                else:
                    yield table

def _fetch_metadata_serialization_wrapper(fragments: List[_SerializedFragment]) -> List['pyarrow.parquet.FileMetaData']:
    if False:
        return 10
    fragments: List['pyarrow._dataset.ParquetFileFragment'] = _deserialize_fragments_with_retry(fragments)
    return _fetch_metadata(fragments)

def _fetch_metadata(fragments: List['pyarrow.dataset.ParquetFileFragment']) -> List['pyarrow.parquet.FileMetaData']:
    if False:
        print('Hello World!')
    fragment_metadata = []
    for f in fragments:
        try:
            fragment_metadata.append(f.metadata)
        except AttributeError:
            break
    return fragment_metadata

def _sample_fragment(to_batches_kwargs, columns, schema, file_fragment: _SerializedFragment) -> float:
    if False:
        print('Hello World!')
    fragment = _deserialize_fragments_with_retry([file_fragment])[0]
    fragment = fragment.subset(row_group_ids=[0])
    batch_size = max(min(fragment.metadata.num_rows, PARQUET_ENCODING_RATIO_ESTIMATE_NUM_ROWS), 1)
    to_batches_kwargs.pop('batch_size', None)
    batches = fragment.to_batches(columns=columns, schema=schema, batch_size=batch_size, **to_batches_kwargs)
    try:
        batch = next(batches)
    except StopIteration:
        ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    else:
        if batch.num_rows > 0:
            in_memory_size = batch.nbytes / batch.num_rows
            metadata = fragment.metadata
            total_size = 0
            for idx in range(metadata.num_row_groups):
                total_size += metadata.row_group(idx).total_byte_size
            file_size = total_size / metadata.num_rows
            ratio = in_memory_size / file_size
        else:
            ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    logger.debug(f'Estimated Parquet encoding ratio is {ratio} for fragment {fragment} with batch size {batch_size}.')
    return ratio