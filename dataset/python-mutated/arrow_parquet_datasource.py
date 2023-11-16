"""Ray ArrowParquetDatasource Module.

This module is pulled from Ray's [ParquetDatasource]
(https://github.com/ray-project/ray/blob/ray-2.0.0/python/ray/data/datasource/parquet_datasource.py) with a few changes
and customized to ensure compatibility with AWS SDK for pandas behavior. Changes from the original implementation,
are documented in the comments and marked with (AWS SDK for pandas) prefix.
"""
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import numpy as np
import pyarrow.fs
from pyarrow.dataset import ParquetFileFragment
from pyarrow.lib import Schema
from ray import cloudpickle
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data._internal.progress_bar import ProgressBar
from ray.data.block import Block, BlockAccessor
from ray.data.context import DatasetContext
from ray.data.datasource import Reader, ReadTask
from ray.data.datasource.file_based_datasource import _resolve_paths_and_filesystem
from ray.data.datasource.file_meta_provider import DefaultParquetMetadataProvider, ParquetMetadataProvider, _handle_read_os_error
from awswrangler import exceptions
from awswrangler._arrow import _add_table_partitions, _df_to_table
from awswrangler.distributed.ray import ray_remote
from awswrangler.distributed.ray.datasources.arrow_parquet_base_datasource import ArrowParquetBaseDatasource
from awswrangler.s3._write import _COMPRESSION_2_EXT
_logger: logging.Logger = logging.getLogger(__name__)
PARQUET_READER_ROW_BATCH_SIZE = 100000
FILE_READING_RETRY = 8
PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT = 5
PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND = 2
PARQUET_ENCODING_RATIO_ESTIMATE_SAMPLING_RATIO = 0.01
PARQUET_ENCODING_RATIO_ESTIMATE_MIN_NUM_SAMPLES = 2
PARQUET_ENCODING_RATIO_ESTIMATE_MAX_NUM_SAMPLES = 10
PARQUET_ENCODING_RATIO_ESTIMATE_NUM_ROWS = 5

class ArrowParquetDatasource(ArrowParquetBaseDatasource):
    """(AWS SDK for pandas) Parquet datasource, for reading and writing Parquet files.

    The following are the changes to the original Ray implementation:
    1. Added handling of additional parameters `dtype`, `index`, `compression` and added the ability
       to pass through additional `pyarrow_additional_kwargs` and `s3_additional_kwargs` for writes.
    3. Added `dataset` and `path_root` parameters to allow user to control loading partitions
       relative to the root S3 prefix.
    """

    def create_reader(self, **kwargs: Dict[str, Any]) -> Reader:
        if False:
            while True:
                i = 10
        'Return a Reader for the given read arguments.'
        return _ArrowParquetDatasourceReader(**kwargs)

    def _write_block(self, f: 'pyarrow.NativeFile', block: BlockAccessor, pandas_kwargs: Optional[Dict[str, Any]], **writer_args: Any) -> None:
        if False:
            return 10
        'Write a block to S3.'
        import pyarrow as pa
        schema: pa.Schema = writer_args.get('schema', None)
        dtype: Optional[Dict[str, str]] = writer_args.get('dtype', None)
        index: bool = writer_args.get('index', False)
        compression: Optional[str] = writer_args.get('compression', None)
        pyarrow_additional_kwargs: Optional[Dict[str, Any]] = writer_args.get('pyarrow_additional_kwargs', {})
        pa.parquet.write_table(_df_to_table(block.to_pandas(), schema=schema, index=index, dtype=dtype), f, compression=compression, **pyarrow_additional_kwargs)

    def _get_file_suffix(self, file_format: str, compression: Optional[str]) -> str:
        if False:
            while True:
                i = 10
        if compression is not None:
            return f'{_COMPRESSION_2_EXT.get(compression)[1:]}.{file_format}'
        return file_format

class _SerializedPiece:

    def __init__(self, frag: ParquetFileFragment):
        if False:
            return 10
        self._data = cloudpickle.dumps((frag.format, frag.path, frag.filesystem, frag.partition_expression))

    def deserialize(self) -> ParquetFileFragment:
        if False:
            print('Hello World!')
        'Implicitly trigger S3 subsystem initialization by importing pyarrow.fs.'
        import pyarrow.fs
        (file_format, path, filesystem, partition_expression) = cloudpickle.loads(self._data)
        return file_format.make_fragment(path, filesystem, partition_expression)

def _deserialize_pieces(serialized_pieces: List[_SerializedPiece]) -> List[ParquetFileFragment]:
    if False:
        print('Hello World!')
    return [p.deserialize() for p in serialized_pieces]

def _deserialize_pieces_with_retry(serialized_pieces: List[_SerializedPiece]) -> List[ParquetFileFragment]:
    if False:
        print('Hello World!')
    min_interval: float = 0
    final_exception: Optional[Exception] = None
    for i in range(FILE_READING_RETRY):
        try:
            return _deserialize_pieces(serialized_pieces)
        except Exception as e:
            import random
            import time
            retry_timing = '' if i == FILE_READING_RETRY - 1 else f'Retry after {min_interval} sec. '
            log_only_show_in_1st_retry = '' if i else f'If earlier read attempt threw certain Exception, it may or may not be an issue depends on these retries succeed or not. serialized_pieces:{serialized_pieces}'
            _logger.exception('%sth attempt to deserialize ParquetFileFragment failed. %s %s', i + 1, retry_timing, log_only_show_in_1st_retry)
            if not min_interval:
                min_interval = 1 + random.random()
            time.sleep(min_interval)
            min_interval = min_interval * 2
            final_exception = e
    raise final_exception

class _ArrowParquetDatasourceReader(Reader):

    def __init__(self, paths: Union[str, List[str]], local_uri: bool=False, filesystem: Optional['pyarrow.fs.FileSystem']=None, columns: Optional[List[str]]=None, schema: Optional[Schema]=None, meta_provider: ParquetMetadataProvider=DefaultParquetMetadataProvider(), _block_udf: Optional[Callable[[Block], Block]]=None, **reader_args: Any):
        if False:
            while True:
                i = 10
        import pyarrow as pa
        import pyarrow.parquet as pq
        (paths, filesystem) = _resolve_paths_and_filesystem(paths, filesystem)
        if len(paths) == 1:
            paths = paths[0]
        self._local_scheduling = None
        if local_uri:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            self._local_scheduling = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        dataset_kwargs = reader_args.pop('dataset_kwargs', {})
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
                _logger.debug('Failed to infer schema of dataset by passing dummy table through UDF due to the following exception:', exc_info=True)
                inferred_schema = schema
        else:
            inferred_schema = schema
        try:
            prefetch_remote_args = {}
            if self._local_scheduling:
                prefetch_remote_args['scheduling_strategy'] = self._local_scheduling
            self._metadata = meta_provider.prefetch_file_metadata(pq_ds.pieces, **prefetch_remote_args) or []
        except OSError as e:
            _handle_read_os_error(e, paths)
        except pyarrow.ArrowInvalid as ex:
            if 'Parquet file size is 0 bytes' in str(ex):
                raise exceptions.InvalidFile(f'Invalid Parquet file. {str(ex)}')
            raise
        self._pq_ds = pq_ds
        self._meta_provider = meta_provider
        self._inferred_schema = inferred_schema
        self._block_udf = _block_udf
        self._reader_args = reader_args
        self._columns = columns
        self._schema = schema
        self._encoding_ratio = self._estimate_files_encoding_ratio()

    def estimate_inmemory_data_size(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Estimate data size.'
        total_size: int = 0
        for file_metadata in self._metadata:
            for row_group_idx in range(file_metadata.num_row_groups):
                row_group_metadata = file_metadata.row_group(row_group_idx)
                total_size += row_group_metadata.total_byte_size
        return total_size * self._encoding_ratio

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        if False:
            i = 10
            return i + 15
        "Override the base class FileBasedDatasource.get_read_tasks().\n\n        Required in order to leverage pyarrow's ParquetDataset abstraction,\n        which simplifies partitioning logic.\n        "
        read_tasks = []
        (block_udf, reader_args, columns, schema) = (self._block_udf, self._reader_args, self._columns, self._schema)
        for (pieces, metadata) in zip(np.array_split(self._pq_ds.pieces, parallelism), np.array_split(self._metadata, parallelism)):
            if len(pieces) <= 0:
                continue
            serialized_pieces = [_SerializedPiece(p) for p in pieces]
            input_files = [p.path for p in pieces]
            meta = self._meta_provider(input_files, self._inferred_schema, pieces=pieces, prefetched_metadata=metadata)
            if meta.size_bytes is not None:
                meta.size_bytes = int(meta.size_bytes * self._encoding_ratio)
            read_tasks.append(ReadTask(lambda p=serialized_pieces: _read_pieces(block_udf, reader_args, columns, schema, p), meta))
        return read_tasks

    def _estimate_files_encoding_ratio(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Return an estimate of the Parquet files encoding ratio.\n\n        To avoid OOMs, it is safer to return an over-estimate than an underestimate.\n        '
        if not DatasetContext.get_current().decoding_size_estimation:
            return PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
        num_files = len(self._pq_ds.pieces)
        num_samples = int(num_files * PARQUET_ENCODING_RATIO_ESTIMATE_SAMPLING_RATIO)
        min_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MIN_NUM_SAMPLES, num_files)
        max_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MAX_NUM_SAMPLES, num_files)
        num_samples = max(min(num_samples, max_num_samples), min_num_samples)
        file_samples = [self._pq_ds.pieces[idx] for idx in np.linspace(0, num_files - 1, num_samples).astype(int).tolist()]
        futures = []
        sample_piece = ray_remote(scheduling_strategy=self._local_scheduling or 'SPREAD')(_sample_piece)
        for sample in file_samples:
            serialized_sample = _SerializedPiece(sample)
            futures.append(sample_piece(self._reader_args, self._columns, self._schema, serialized_sample))
        sample_bar = ProgressBar('Parquet Files Sample', len(futures))
        sample_ratios = sample_bar.fetch_until_complete(futures)
        sample_bar.close()
        ratio = np.mean(sample_ratios)
        _logger.debug(f'Estimated Parquet encoding ratio from sampling is {ratio}.')
        return max(ratio, PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND)

def _read_pieces(block_udf: Optional[Callable[[Block], Block]], reader_args: Any, columns: Optional[List[str]], schema: Optional[Union[type, 'pyarrow.lib.Schema']], serialized_pieces: List[_SerializedPiece]) -> Iterator['pyarrow.Table']:
    if False:
        return 10
    pieces: List[ParquetFileFragment] = _deserialize_pieces_with_retry(serialized_pieces)
    assert len(pieces) > 0
    import pyarrow as pa
    ctx = DatasetContext.get_current()
    output_buffer = BlockOutputBuffer(block_udf=block_udf, target_max_block_size=ctx.target_max_block_size)
    _logger.debug('Reading %s parquet pieces', len(pieces))
    use_threads = reader_args.pop('use_threads', False)
    path_root = reader_args.pop('path_root', None)
    for piece in pieces:
        batches = piece.to_batches(use_threads=use_threads, columns=columns, schema=schema, batch_size=PARQUET_READER_ROW_BATCH_SIZE, **reader_args)
        for batch in batches:
            table = _add_table_partitions(table=pa.Table.from_batches([batch], schema=schema), path=f's3://{piece.path}', path_root=path_root)
            if table.num_rows > 0:
                output_buffer.add_block(table)
                if output_buffer.has_next():
                    yield output_buffer.next()
    output_buffer.finalize()
    if output_buffer.has_next():
        yield output_buffer.next()

def _sample_piece(reader_args: Any, columns: Optional[List[str]], schema: Optional[Union[type, 'pyarrow.lib.Schema']], file_piece: _SerializedPiece) -> float:
    if False:
        while True:
            i = 10
    piece = _deserialize_pieces_with_retry([file_piece])[0]
    piece = piece.subset(row_group_ids=[0])
    batch_size = max(min(piece.metadata.num_rows, PARQUET_ENCODING_RATIO_ESTIMATE_NUM_ROWS), 1)
    reader_args.pop('batch_size', None)
    reader_args.pop('path_root', None)
    batches = piece.to_batches(columns=columns, schema=schema, batch_size=batch_size, **reader_args)
    try:
        batch = next(batches)
    except StopIteration:
        ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    else:
        if batch.num_rows > 0:
            in_memory_size = batch.nbytes / batch.num_rows
            metadata = piece.metadata
            total_size = 0
            for idx in range(metadata.num_row_groups):
                total_size += metadata.row_group(idx).total_byte_size
            file_size = total_size / metadata.num_rows
            ratio = in_memory_size / file_size
        else:
            ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    _logger.debug(f'Estimated Parquet encoding ratio is {ratio} for piece {piece} with batch size {batch_size}.')
    return ratio