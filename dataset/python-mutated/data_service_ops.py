"""Python API for executing a tf.data.Dataset using a tf.data service."""
import enum
import functools
from typing import Callable
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
COMPRESSION_AUTO = 'AUTO'
COMPRESSION_NONE = None
_PARALLEL_EPOCHS = 'parallel_epochs'
_DISTRIBUTED_EPOCH = 'distributed_epoch'

@tf_export('data.experimental.service.ShardingPolicy')
class ShardingPolicy(enum.IntEnum):
    """Specifies how to shard data among tf.data service workers.

  OFF: No sharding will be performed. Each worker produces the entire dataset
  without any sharding. With this mode, the best practice is to shuffle the
  dataset nondeterministically so that workers process the dataset in different
  orders. If workers are restarted or join the cluster mid-job, they will begin
  processing the dataset from the beginning.

  DYNAMIC: The input dataset is dynamically split among workers at runtime. Each
  worker gets the next split when it reads data from the dispatcher. Data is
  produced non-deterministically in this mode. Dynamic sharding works well with
  varying-sized tf.data service clusters, e.g., when you need to auto-scale your
  workers. Dynamic sharding provides at-most once visitation guarantees. No
  examples will be repeated, but some may be missed if a tf.data service worker
  gets restarted while processing a file.

  The following are static sharding policies. The semantics are similar to
  `tf.data.experimental.AutoShardPolicy`. These policies require:
  * The tf.data service cluster is configured with a fixed list of workers
    in DispatcherConfig.
  * Each client only reads from the local tf.data service worker.

  If a worker is restarted while performing static sharding, the worker will
  begin processing its shard again from the beginning.

  FILE: Shards by input files (i.e. each worker will get a fixed set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.

  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partition the dataset elements, the dataset needs to
  produce elements in a deterministic order.

  FILE_OR_DATA: Attempts FILE-based sharding, falling back to DATA-based
  sharding on failure.

  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """
    OFF = 0
    DYNAMIC = 1
    FILE = 2
    DATA = 3
    FILE_OR_DATA = 4
    HINT = 5

    def _to_proto(self) -> data_service_pb2.ProcessingModeDef.ShardingPolicy:
        if False:
            for i in range(10):
                print('nop')
        'Converts the policy to ProcessingModeDef proto enum.'
        if self == ShardingPolicy.OFF:
            return data_service_pb2.ProcessingModeDef.OFF
        if self == ShardingPolicy.DYNAMIC:
            return data_service_pb2.ProcessingModeDef.DYNAMIC
        if self == ShardingPolicy.FILE:
            return data_service_pb2.ProcessingModeDef.FILE
        if self == ShardingPolicy.DATA:
            return data_service_pb2.ProcessingModeDef.DATA
        if self == ShardingPolicy.FILE_OR_DATA:
            return data_service_pb2.ProcessingModeDef.FILE_OR_DATA
        if self == ShardingPolicy.HINT:
            return data_service_pb2.ProcessingModeDef.HINT
        raise ValueError(f'Unable to convert sharding policy {self!r} to proto.')

@tf_export('data.experimental.service.CrossTrainerCache')
class CrossTrainerCache:
    """Options related to the tf.data service cross trainer cache.

  This is used to enable cross-trainer cache when distributing a dataset. For
  example:

  ```
  dataset = dataset.apply(tf.data.experimental.service.distribute(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      job_name="job",
      cross_trainer_cache=data_service_ops.CrossTrainerCache(
          trainer_id=trainer_id())))
  ```

  For more details, refer to
  https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers.
  """

    def __init__(self, trainer_id):
        if False:
            print('Hello World!')
        'Constructs a CrossTrainerCache.\n\n    Args:\n      trainer_id: Each training job has a unique ID. Once a job has consumed\n      data, the data remains in the cache and is re-used by jobs with different\n      `trainer_id`s. Requests with the same `trainer_id` do not re-use data.\n\n    Raises:\n      ValueError if `trainer_id` is empty.\n    '
        if not trainer_id:
            raise ValueError('tf.data service cross-trainer cache requires a non-empty trainer ID.')
        self.trainer_id = trainer_id

    def _to_proto(self) -> data_service_pb2.CrossTrainerCacheOptions:
        if False:
            while True:
                i = 10
        return data_service_pb2.CrossTrainerCacheOptions(trainer_id=self.trainer_id)

def _get_validated_sharding_policy(processing_mode) -> ShardingPolicy:
    if False:
        i = 10
        return i + 15
    'Validates `processing_mode` and converts it to ShardingPolicy.'
    if isinstance(processing_mode, ShardingPolicy):
        return processing_mode
    if processing_mode == _PARALLEL_EPOCHS:
        return ShardingPolicy.OFF
    if processing_mode == _DISTRIBUTED_EPOCH:
        return ShardingPolicy.DYNAMIC
    raise ValueError(f'tf.data service processing mode should be a `tf.data.experimental.service.ShardingPolicy`, `"parallel_epochs"`, or `"distributed_epoch"`. Got {processing_mode!r}.')

def _validate_job_name(job_name) -> None:
    if False:
        return 10
    if job_name is None:
        return
    if not isinstance(job_name, str):
        raise ValueError(f'`job_name` must be a string, but `job_name` was of type {type(job_name)}. job_name={job_name}')
    if not job_name:
        raise ValueError('`job_name` must not be empty')

def _validate_compression(compression) -> None:
    if False:
        print('Hello World!')
    valid_compressions = [COMPRESSION_AUTO, COMPRESSION_NONE]
    if compression not in valid_compressions:
        raise ValueError(f'Invalid `compression` argument: {compression}. Must be one of {valid_compressions}.')

def _get_compression_proto(compression) -> data_service_pb2.DataServiceMetadata.Compression:
    if False:
        i = 10
        return i + 15
    if compression == COMPRESSION_AUTO:
        return data_service_pb2.DataServiceMetadata.COMPRESSION_SNAPPY
    if compression == COMPRESSION_NONE:
        return data_service_pb2.DataServiceMetadata.COMPRESSION_OFF
    raise ValueError(f'Invalid `compression` argument: {compression}. Must be one of {[COMPRESSION_AUTO, COMPRESSION_NONE]}.')

def _to_tensor(dataset_id) -> tensor.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Converts `dataset_id` to Tensor.'
    if isinstance(dataset_id, tensor.Tensor):
        return dataset_id
    if isinstance(dataset_id, str) or isinstance(dataset_id, bytes):
        return ops.convert_to_tensor(dataset_id, dtype=dtypes.string, name='dataset_id')
    return ops.convert_to_tensor(dataset_id, dtype=dtypes.int64, name='dataset_id')

def _to_string(dataset_id) -> str:
    if False:
        while True:
            i = 10
    'Converts `dataset_id` to string.'
    if isinstance(dataset_id, tensor.Tensor):
        return dataset_id if dataset_id.dtype == dtypes.string else string_ops.as_string(dataset_id)
    return dataset_id.decode() if isinstance(dataset_id, bytes) else str(dataset_id)

class _DataServiceDatasetV2(dataset_ops.DatasetSource):
    """A `Dataset` that reads elements from the tf.data service."""

    def __init__(self, dataset_id, processing_mode, address, element_spec, protocol, data_transfer_protocol, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, cross_trainer_cache=None, target_workers='AUTO'):
        if False:
            return 10
        'Constructs a _DataServiceDatasetV2.\n\n    Args:\n      dataset_id: The dataset id for the dataset to read from.\n      processing_mode: A `tf.data.experimental.service.ShardingPolicy`\n        specifying how to shard the dataset among tf.data workers. See\n        `tf.data.experimental.service.ShardingPolicy` for details. For backwards\n        compatibility, `processing_mode` may also be set to the strings\n        `"parallel_epochs"` or `"distributed_epoch"`, which are respectively\n        equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.\n      address: The tf.data service address, e.g. "localhost:5000".\n      element_spec: The dataset element spec for the dataset to read from.\n      protocol: The protocol to use for communicating with the tf.data service,\n        e.g. "grpc".\n      data_transfer_protocol: (Optional.) The protocol to use for transferring\n        data with the tf.data service. By default, data is transferred using\n        gRPC.\n      job_name: (Optional.) The name of the job. If provided, it must be a\n        non-empty string or Tensor. This argument makes it possible for multiple\n        datasets to share the same job. The default behavior is that the dataset\n        creates anonymous, exclusively owned jobs.\n      consumer_index: (Optional.) The index of the consumer in the range from\n        `0` to `num_consumers`. Must be specified alongside `num_consumers`.\n        When specified, consumers will read from the job in a strict round-robin\n        order, instead of the default first-come-first-served order.\n      num_consumers: (Optional.) The number of consumers which will consume from\n        the job. Must be specified alongside `consumer_index`. When specified,\n        consumers will read from the job in a strict round-robin order, instead\n        of the default first-come-first-served order. When `num_consumers` is\n        specified, the dataset must have infinite cardinality to prevent a\n        producer from running out of data early and causing consumers to go out\n        of sync.\n      max_outstanding_requests: (Optional.) A limit on how many elements may be\n        requested at the same time. You can use this option to control the\n        amount of memory used, since `distribute` won\'t use more than\n        `element_size` * `max_outstanding_requests` of memory.\n      task_refresh_interval_hint_ms: (Optional.) A hint for how often to query\n        the dispatcher for task changes.\n      cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is\n        provided, dataset iteration will be shared across concurrently running\n        trainers. See\n        https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers\n        for details.\n      target_workers: (Optional.) Which workers to read from. If `"AUTO"`,\n        tf.data runtime decides which workers to read from. If `"ANY"`, reads\n        from any tf.data service workers. If `"LOCAL"`, only reads from local\n        in-processs tf.data service workers. `"AUTO"` works well for most cases,\n        while users can specify other targets. For example, `"LOCAL"` helps\n        avoid RPCs and data copy if every TF worker colocates with a tf.data\n        service worker. Consumers of a shared job must use the same\n        `target_workers`. Defaults to `"AUTO"`.\n    '
        if consumer_index is None != num_consumers is None:
            raise ValueError('Must either set both `consumer_index` and `num_consumers`, or neither. ', f'consumer_index={consumer_index}, num_consumers={num_consumers}')
        if num_consumers is not None and job_name is None:
            raise ValueError(f'`job_name` must be set when setting `num_consumers`. num_consumers was set to {num_consumers}.')
        processing_mode_def = data_service_pb2.ProcessingModeDef(sharding_policy=_get_validated_sharding_policy(processing_mode)._to_proto())
        if job_name is None:
            job_name = ''
        if max_outstanding_requests is None:
            max_outstanding_requests = dataset_ops.AUTOTUNE
        if task_refresh_interval_hint_ms is None:
            task_refresh_interval_hint_ms = dataset_ops.AUTOTUNE
        self._dataset_id = _to_tensor(dataset_id)
        self._processing_mode = ops.convert_to_tensor(processing_mode_def.SerializeToString(), dtype=dtypes.string, name='processing_mode')
        self._address = ops.convert_to_tensor(address, dtype=dtypes.string, name='address')
        self._protocol = ops.convert_to_tensor(protocol, dtype=dtypes.string, name='protocol')
        self._job_name = ops.convert_to_tensor(job_name, dtype=dtypes.string, name='job_name')
        self._consumer_index = ops.convert_to_tensor(-1 if consumer_index is None else consumer_index, dtype=dtypes.int64, name='consumer_index')
        self._num_consumers = ops.convert_to_tensor(-1 if num_consumers is None else num_consumers, dtype=dtypes.int64, name='num_consumers')
        self._max_outstanding_requests = ops.convert_to_tensor(max_outstanding_requests, dtype=dtypes.int64, name='max_outstanding_requests')
        self._element_spec = element_spec
        uncompress_func = structured_function.StructuredFunctionWrapper(lambda x: compression_ops.uncompress(x, output_spec=element_spec), transformation_name='DataServiceDataset.uncompress()', input_structure=tensor.TensorSpec(shape=(), dtype=dtypes.variant))
        cross_trainer_cache_options = cross_trainer_cache._to_proto().SerializeToString() if cross_trainer_cache else None
        compat_kwargs = {}
        if data_transfer_protocol is not None:
            compat_kwargs['data_transfer_protocol'] = data_transfer_protocol
        uncompress = True
        variant_tensor = gen_experimental_dataset_ops.data_service_dataset_v4(dataset_id=self._dataset_id, processing_mode=self._processing_mode, address=self._address, protocol=self._protocol, job_name=self._job_name, consumer_index=self._consumer_index, num_consumers=self._num_consumers, max_outstanding_requests=self._max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, iteration_counter=gen_experimental_dataset_ops.dummy_iteration_counter(), target_workers=target_workers, uncompress=uncompress, uncompress_fn=uncompress_func.function, cross_trainer_cache_options=cross_trainer_cache_options, **compat_kwargs, **self._flat_structure)
        super(_DataServiceDatasetV2, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return self._element_spec

class _DataServiceDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` that executes its input through the tf.data service."""

    @functools.wraps(_DataServiceDatasetV2.__init__)
    def __init__(self, dataset_id, processing_mode, address, element_spec, protocol, data_transfer_protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, task_refresh_interval_hint_ms, cross_trainer_cache, target_workers):
        if False:
            i = 10
            return i + 15
        self._wrapped = _DataServiceDatasetV2(dataset_id=dataset_id, processing_mode=processing_mode, address=address, element_spec=element_spec, protocol=protocol, data_transfer_protocol=data_transfer_protocol, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
        super(_DataServiceDatasetV1, self).__init__(self._wrapped)
if tf2.enabled():
    _DataServiceDataset = _DataServiceDatasetV2
else:
    _DataServiceDataset = _DataServiceDatasetV1

def _parse_service(service) -> tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    'Converts a tf.data service string into a (protocol, address) tuple.\n\n  Args:\n    service: A string in the format "protocol://address" or just "address". If\n      the string is only an address, the default protocol will be used.\n\n  Returns:\n    The (protocol, address) tuple\n  '
    if not isinstance(service, str):
        raise ValueError(f'`service` must be a string, but `service` was of type {type(service)}. service={service}')
    if not service:
        raise ValueError('`service` must not be empty')
    parts = service.split('://')
    if len(parts) == 2:
        (protocol, address) = parts
    elif len(parts) == 1:
        address = parts[0]
        protocol = _pywrap_utils.TF_DATA_DefaultProtocol()
    else:
        raise ValueError(f"Malformed `service` string has multiple '://': {service}.")
    return (protocol, address)

def _distribute(processing_mode, service, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, data_transfer_protocol=None, compression='AUTO', cross_trainer_cache=None, target_workers='AUTO') -> Callable[dataset_ops.Dataset, dataset_ops.Dataset]:
    if False:
        i = 10
        return i + 15
    'A transformation that moves dataset processing to the tf.data service.\n\n  This transformation is similar to `distribute`, but supports additional\n  parameters which we do not yet want to add to the public Python API.\n\n  Args:\n    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying\n      how to shard the dataset among tf.data workers. See\n      `tf.data.experimental.service.ShardingPolicy` for details. For backwards\n      compatibility, `processing_mode` may also be set to the strings\n      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively\n      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    job_name: (Optional.) The name of the job. If provided, it must be a\n      non-empty string. This argument makes it possible for multiple datasets to\n      share the same job. The default behavior is that the dataset creates\n      anonymous, exclusively owned jobs.\n    consumer_index: (Optional.) The index of the consumer in the range from `0`\n      to `num_consumers`. Must be specified alongside `num_consumers`. When\n      specified, consumers will read from the job in a strict round-robin order,\n      instead of the default first-come-first-served order.\n    num_consumers: (Optional.) The number of consumers which will consume from\n      the job. Must be specified alongside `consumer_index`. When specified,\n      consumers will read from the job in a strict round-robin order, instead of\n      the default first-come-first-served order. When `num_consumers` is\n      specified, the dataset must have infinite cardinality to prevent a\n      producer from running out of data early and causing consumers to go out of\n      sync.\n    max_outstanding_requests: (Optional.) A limit on how many elements may be\n      requested at the same time. You can use this option to control the amount\n      of memory used, since `distribute` won\'t use more than `element_size` *\n      `max_outstanding_requests` of memory.\n    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the\n      dispatcher for task changes.\n    data_transfer_protocol: (Optional.) The protocol to use for transferring\n      data with the tf.data service. By default, data is transferred using gRPC.\n    compression: How to compress the dataset\'s elements before transferring them\n      over the network. "AUTO" leaves the decision of how to compress up to the\n      tf.data service runtime. `None` indicates not to compress.\n    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is\n      provided, dataset iteration will be shared across concurrently running\n      trainers. See\n      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers\n      for details.\n    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data\n      runtime decides which workers to read from. If `"ANY"`, reads from any\n      tf.data service workers. If `"LOCAL"`, only reads from local in-processs\n      tf.data service workers. `"AUTO"` works well for most cases, while users\n      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and\n      data copy if every TF worker colocates with a tf.data service worker.\n      Consumers of a shared job must use the same `target_workers`. Defaults to\n      `"AUTO"`.\n\n  Returns:\n    Dataset: A `Dataset` of the elements produced by the data service.\n  '
    processing_mode = _get_validated_sharding_policy(processing_mode)
    _validate_compression(compression)

    def _apply_fn(dataset) -> dataset_ops.Dataset:
        if False:
            while True:
                i = 10
        dataset_id = _register_dataset(service, dataset, compression=compression)
        return _from_dataset_id(processing_mode, service, dataset_id, dataset.element_spec, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, data_transfer_protocol=data_transfer_protocol, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
    return _apply_fn

@tf_export('data.experimental.service.distribute')
def distribute(processing_mode, service, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, data_transfer_protocol=None, compression='AUTO', cross_trainer_cache=None, target_workers='AUTO') -> Callable[dataset_ops.Dataset, dataset_ops.Dataset]:
    if False:
        print('Hello World!')
    'A transformation that moves dataset processing to the tf.data service.\n\n  When you iterate over a dataset containing the `distribute` transformation,\n  the tf.data service creates a "job" which produces data for the dataset\n  iteration.\n\n  The tf.data service uses a cluster of workers to prepare data for training\n  your model.\n  The `processing_mode` argument to `tf.data.experimental.service.distribute`\n  describes how to leverage multiple workers to process the input dataset.\n  Currently, there are two processing modes to choose from: "distributed_epoch"\n  and "parallel_epochs".\n\n  "distributed_epoch" means that the dataset will be split across all tf.data\n  service workers.\n  The dispatcher produces "splits" for the dataset and sends them to workers for\n  further processing. For example, if a dataset begins with a list of filenames,\n  the dispatcher will iterate through the filenames and send the filenames to\n  tf.data workers, which will perform the rest of the dataset transformations on\n  those files. "distributed_epoch" is useful when your model needs to see each\n  element of the dataset exactly once, or if it needs to see the data in a\n  generally-sequential order. "distributed_epoch" only works for datasets with\n  splittable sources, such as `Dataset.from_tensor_slices`,\n  `Dataset.list_files`, or `Dataset.range`.\n\n  "parallel_epochs" means that the entire input dataset will be processed\n  independently by each of the tf.data service workers.\n  For this reason, it is important to shuffle data (e.g. filenames)\n  non-deterministically, so that each worker will process the elements of the\n  dataset in a different order. "parallel_epochs" can be used to distribute\n  datasets that aren\'t splittable.\n\n  With two workers, "parallel_epochs" will produce every element of the dataset\n  twice:\n\n  >>> dispatcher = tf.data.experimental.service.DispatchServer()\n  >>> dispatcher_address = dispatcher.target.split("://")[1]\n  >>> # Start two workers\n  >>> workers = [\n  ...     tf.data.experimental.service.WorkerServer(\n  ...         tf.data.experimental.service.WorkerConfig(\n  ...             dispatcher_address=dispatcher_address)) for _ in range(2)\n  ... ]\n  >>> dataset = tf.data.Dataset.range(10)\n  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(\n  ...     processing_mode="parallel_epochs", service=dispatcher.target))\n  >>> print(sorted(list(dataset.as_numpy_iterator())))\n  [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]\n\n  "distributed_epoch", on the other hand, will still produce each element once:\n\n  >>> dispatcher = tf.data.experimental.service.DispatchServer()\n  >>> dispatcher_address = dispatcher.target.split("://")[1]\n  >>> workers = [\n  ...     tf.data.experimental.service.WorkerServer(\n  ...         tf.data.experimental.service.WorkerConfig(\n  ...             dispatcher_address=dispatcher_address)) for _ in range(2)\n  ... ]\n  >>> dataset = tf.data.Dataset.range(10)\n  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(\n  ...     processing_mode="distributed_epoch", service=dispatcher.target))\n  >>> print(sorted(list(dataset.as_numpy_iterator())))\n  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n  When using `apply(tf.data.experimental.service.distribute(...))`, the dataset\n  before the `apply` transformation executes within the tf.data service, while\n  the operations after `apply` happen within the local process.\n\n  >>> dispatcher = tf.data.experimental.service.DispatchServer()\n  >>> dispatcher_address = dispatcher.target.split("://")[1]\n  >>> workers = [\n  ...     tf.data.experimental.service.WorkerServer(\n  ...         tf.data.experimental.service.WorkerConfig(\n  ...             dispatcher_address=dispatcher_address)) for _ in range(2)\n  ... ]\n  >>> dataset = tf.data.Dataset.range(5)\n  >>> dataset = dataset.map(lambda x: x*x)\n  >>> dataset = dataset.apply(\n  ...    tf.data.experimental.service.distribute("parallel_epochs",\n  ...                                            dispatcher.target))\n  >>> dataset = dataset.map(lambda x: x+1)\n  >>> print(sorted(list(dataset.as_numpy_iterator())))\n  [1, 1, 2, 2, 5, 5, 10, 10, 17, 17]\n\n  In the above example, the dataset operations (before applying the `distribute`\n  function on the elements) will be executed on the tf.data workers,\n  and the elements are provided over RPC. The remaining transformations\n  (after the call to `distribute`) will be executed locally. The dispatcher\n  and the workers will bind to usused free ports (which are chosen at random),\n  in order to communicate with each other. However, to bind them to specific\n  ports, the `port` parameter can be passed.\n\n  The `job_name` argument allows jobs to be shared across multiple\n  datasets. Instead of each dataset creating its own job, all\n  datasets with the same `job_name` will consume from the same job. A new job\n  will be created for each iteration of the dataset (with each repetition of\n  `Dataset.repeat` counting as a new iteration). Suppose the `DispatchServer`\n  is serving on `localhost:5000` and two training workers (in either a single\n  client or multi-client setup) iterate over the below dataset, and there is a\n  single tf.data worker:\n\n  ```\n  range5_dataset = tf.data.Dataset.range(5)\n  dataset = range5_dataset.apply(tf.data.experimental.service.distribute(\n      "parallel_epochs", "localhost:5000", job_name="my_job_name"))\n  for iteration in range(3):\n    print(list(dataset))\n  ```\n\n  The elements of each job will be split between the two processes, with\n  elements being consumed by the processes on a first-come first-served basis.\n  One possible result is that process 1 prints\n\n  ```\n  [0, 2, 4]\n  [0, 1, 3]\n  [1]\n  ```\n\n  and process 2 prints\n\n  ```\n  [1, 3]\n  [2, 4]\n  [0, 2, 3, 4]\n  ```\n\n  Job names must not be re-used across different training jobs within the\n  lifetime of the tf.data service. In general, the tf.data service is expected\n  to live for the duration of a single training job.\n  To use the tf.data service with multiple training jobs, make sure to use\n  different job names to avoid conflicts. For example, suppose a training job\n  calls `distribute` with `job_name="job"` and reads until end of input. If\n  another independent job connects to the same tf.data service and tries to read\n  from `job_name="job"`, it will immediately receive end of input, without\n  getting any data.\n\n  **Coordinated data read**\n\n  By default, when multiple consumers read from the same job, they receive data\n  on a first-come first-served basis. In some use cases, it is advantageous to\n  coordinate the consumers. At each step, consumers read data from the same\n  worker.\n\n  For example, the tf.data service can be used to coordinate example sizes\n  across a cluster during synchronous training, so that during each step all\n  replicas train on similar-sized elements. To achieve this, define a dataset\n  which generates rounds of `num_consumers` consecutive similar-sized batches,\n  then enable coordinated reads by setting `consumer_index` and `num_consumers`.\n\n  NOTE: To keep consumers in sync, round robin data consumption requires that\n  the dataset have infinite cardinality. You can get this by adding `.repeat()`\n  at the end of the dataset definition.\n\n  **Keras and Distribution Strategies**\n\n  The dataset produced by the `distribute` transformation can be passed to\n  Keras\' `Model.fit` or Distribution Strategy\'s\n  `tf.distribute.Strategy.experimental_distribute_dataset` like any other\n  `tf.data.Dataset`. We recommend setting a `job_name` on the call to\n  `distribute` so that if there are multiple workers, they read data from the\n  same job. Note that the autosharding normally performed by\n  `experimental_distribute_dataset` will be disabled when setting a `job_name`,\n  since sharing the job already results in splitting data across the workers.\n  When using a shared job, data will be dynamically balanced across workers, so\n  that they reach end of input about the same time. This results in better\n  worker utilization than with autosharding, where each worker processes an\n  independent set of files, and some workers may run out of data earlier than\n  others.\n\n  Args:\n    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying\n      how to shard the dataset among tf.data workers. See\n      `tf.data.experimental.service.ShardingPolicy` for details. For backwards\n      compatibility, `processing_mode` may also be set to the strings\n      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively\n      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    job_name: (Optional.) The name of the job. If provided, it must be a\n      non-empty string. This argument makes it possible for multiple datasets to\n      share the same job. The default behavior is that the dataset creates\n      anonymous, exclusively owned jobs.\n    consumer_index: (Optional.) The index of the consumer in the range from `0`\n      to `num_consumers`. Must be specified alongside `num_consumers`. When\n      specified, consumers will read from the job in a strict round-robin order,\n      instead of the default first-come-first-served order.\n    num_consumers: (Optional.) The number of consumers which will consume from\n      the job. Must be specified alongside `consumer_index`. When specified,\n      consumers will read from the job in a strict round-robin order, instead of\n      the default first-come-first-served order. When `num_consumers` is\n      specified, the dataset must have infinite cardinality to prevent a\n      producer from running out of data early and causing consumers to go out of\n      sync.\n    max_outstanding_requests: (Optional.) A limit on how many elements may be\n      requested at the same time. You can use this option to control the amount\n      of memory used, since `distribute` won\'t use more than `element_size` *\n      `max_outstanding_requests` of memory.\n    data_transfer_protocol: (Optional.) The protocol to use for transferring\n      data with the tf.data service. By default, data is transferred using gRPC.\n    compression: How to compress the dataset\'s elements before transferring them\n      over the network. "AUTO" leaves the decision of how to compress up to the\n      tf.data service runtime. `None` indicates not to compress.\n    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is\n      provided, dataset iteration will be shared across concurrently running\n      trainers. See\n      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers\n      for details.\n    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data\n      runtime decides which workers to read from. If `"ANY"`, reads from any\n      tf.data service workers. If `"LOCAL"`, only reads from local in-processs\n      tf.data service workers. `"AUTO"` works well for most cases, while users\n      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and\n      data copy if every TF worker colocates with a tf.data service worker.\n      Consumers of a shared job must use the same `target_workers`. Defaults to\n      `"AUTO"`.\n\n  Returns:\n    Dataset: A `Dataset` of the elements produced by the data service.\n  '
    _validate_job_name(job_name)
    return _distribute(processing_mode=processing_mode, service=service, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, data_transfer_protocol=data_transfer_protocol, compression=compression, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)

def _register_dataset(service, dataset, compression, dataset_id=None) -> tensor.Tensor:
    if False:
        while True:
            i = 10
    'Registers a dataset with the tf.data service.\n\n  This transformation is similar to `register_dataset`, but supports additional\n  parameters which we do not yet want to add to the public Python API.\n\n  Args:\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    dataset: A `tf.data.Dataset` to register with the tf.data service.\n    compression: How to compress the dataset\'s elements before transferring them\n      over the network. "AUTO" leaves the decision of how to compress up to the\n      tf.data service runtime. `None` indicates not to compress.\n    dataset_id: (Optional.) By default, tf.data service generates a unique\n      (string) ID for each registered dataset. If a `dataset_id` is provided, it\n      will use the specified ID. If a dataset with a matching ID already exists,\n      no new dataset is registered. This is useful if multiple training jobs\n      want to (re)use the same dataset for training. In this case, they can\n      register the dataset with the same dataset ID.\n\n  Returns:\n    A scalar string tensor representing the dataset ID.\n  '
    _validate_compression(compression)
    if isinstance(service, tuple):
        (protocol, address) = service
    else:
        (protocol, address) = _parse_service(service)
    external_state_policy = dataset.options().experimental_external_state_policy
    if external_state_policy is None:
        external_state_policy = ExternalStatePolicy.WARN
    encoded_spec = None
    if context.executing_eagerly():
        encoded_spec = nested_structure_coder.encode_structure(dataset.element_spec).SerializeToString()
    if compression == COMPRESSION_AUTO:
        dataset = dataset.map(lambda *x: compression_ops.compress(x), num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset._apply_debug_options()
    metadata = data_service_pb2.DataServiceMetadata(element_spec=encoded_spec, compression=_get_compression_proto(compression))
    return gen_experimental_dataset_ops.register_dataset_v2(dataset._variant_tensor, address=address, protocol=protocol, external_state_policy=external_state_policy.value, requested_dataset_id=dataset_id, metadata=metadata.SerializeToString())

@tf_export('data.experimental.service.register_dataset')
def register_dataset(service, dataset, compression='AUTO', dataset_id=None) -> tensor.Tensor:
    if False:
        print('Hello World!')
    'Registers a dataset with the tf.data service.\n\n  `register_dataset` registers a dataset with the tf.data service so that\n  datasets can be created later with\n  `tf.data.experimental.service.from_dataset_id`. This is useful when the\n  dataset\n  is registered by one process, then used in another process. When the same\n  process is both registering and reading from the dataset, it is simpler to use\n  `tf.data.experimental.service.distribute` instead.\n\n  If the dataset is already registered with the tf.data service,\n  `register_dataset` returns the already-registered dataset\'s id.\n\n  >>> dispatcher = tf.data.experimental.service.DispatchServer()\n  >>> dispatcher_address = dispatcher.target.split("://")[1]\n  >>> worker = tf.data.experimental.service.WorkerServer(\n  ...     tf.data.experimental.service.WorkerConfig(\n  ...         dispatcher_address=dispatcher_address))\n  >>> dataset = tf.data.Dataset.range(10)\n  >>> dataset_id = tf.data.experimental.service.register_dataset(\n  ...     dispatcher.target, dataset)\n  >>> dataset = tf.data.experimental.service.from_dataset_id(\n  ...     processing_mode="parallel_epochs",\n  ...     service=dispatcher.target,\n  ...     dataset_id=dataset_id,\n  ...     element_spec=dataset.element_spec)\n  >>> print(list(dataset.as_numpy_iterator()))\n  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n  Args:\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    dataset: A `tf.data.Dataset` to register with the tf.data service.\n    compression: (Optional.) How to compress the dataset\'s elements before\n      transferring them over the network. "AUTO" leaves the decision of how to\n      compress up to the tf.data service runtime. `None` indicates not to\n      compress.\n    dataset_id: (Optional.) By default, tf.data service generates a unique\n      (string) ID for each registered dataset. If a `dataset_id` is provided, it\n      will use the specified ID. If a dataset with a matching ID already exists,\n      no new dataset is registered. This is useful if multiple training jobs\n      want to (re)use the same dataset for training. In this case, they can\n      register the dataset with the same dataset ID.\n\n  Returns:\n    A scalar string tensor representing the dataset ID.\n  '
    return _register_dataset(service, dataset, compression, dataset_id)

def _from_dataset_id(processing_mode, service, dataset_id, element_spec, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, data_transfer_protocol=None, cross_trainer_cache=None, target_workers='AUTO') -> dataset_ops.Dataset:
    if False:
        while True:
            i = 10
    'Creates a dataset which reads data from the tf.data service.\n\n  This transformation is similar to `from_dataset_id`, but supports additional\n  parameters which we do not yet want to add to the public Python API.\n\n  Args:\n    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying\n      how to shard the dataset among tf.data workers. See\n      `tf.data.experimental.service.ShardingPolicy` for details. For backwards\n      compatibility, `processing_mode` may also be set to the strings\n      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively\n      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    dataset_id: The id of the dataset to read from. This id is returned by\n      `register_dataset` when the dataset is registered with the tf.data\n      service.\n    element_spec: A nested structure of `tf.TypeSpec`s representing the type of\n      elements produced by the dataset. This argument is only required inside a\n      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec\n      for a given dataset.\n    job_name: (Optional.) The name of the job. If provided, it must be a\n      non-empty string or tensor. This argument makes it possible for multiple\n      datasets to share the same job. The default behavior is that the dataset\n      creates anonymous, exclusively owned jobs.\n    consumer_index: (Optional.) The index of the consumer in the range from `0`\n      to `num_consumers`. Must be specified alongside `num_consumers`. When\n      specified, consumers will read from the job in a strict round-robin order,\n      instead of the default first-come-first-served order.\n    num_consumers: (Optional.) The number of consumers which will consume from\n      the job. Must be specified alongside `consumer_index`. When specified,\n      consumers will read from the job in a strict round-robin order, instead of\n      the default first-come-first-served order. When `num_consumers` is\n      specified, the dataset must have infinite cardinality to prevent a\n      producer from running out of data early and causing consumers to go out of\n      sync.\n    max_outstanding_requests: (Optional.) A limit on how many elements may be\n      requested at the same time. You can use this option to control the amount\n      of memory used, since `distribute` won\'t use more than `element_size` *\n      `max_outstanding_requests` of memory.\n    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the\n      dispatcher for task changes.\n    data_transfer_protocol: (Optional.) The protocol to use for transferring\n      data with the tf.data service. By default, data is transferred using gRPC.\n    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is\n      provided, dataset iteration will be shared across concurrently running\n      trainers. See\n      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers\n      for details.\n    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data\n      runtime decides which workers to read from. If `"ANY"`, reads from any\n      tf.data service workers. If `"LOCAL"`, only reads from local in-processs\n      tf.data service workers. `"AUTO"` works well for most cases, while users\n      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and\n      data copy if every TF worker colocates with a tf.data service worker.\n      Consumers of a shared job must use the same `target_workers`. Defaults to\n      `"AUTO"`.\n\n  Returns:\n    A `tf.data.Dataset` which reads from the tf.data service.\n  '

    def _get_element_spec():
        if False:
            while True:
                i = 10
        'Fetches the element spec from the server.'
        data_service_metadata = None
        dataset_id_val = tensor_util.constant_value(dataset_id)
        try:
            data_service_metadata = _pywrap_server_lib.TF_DATA_GetDataServiceMetadataByID(dataset_id_val, address, protocol)
        except NotImplementedError as err:
            raise ValueError('The tf.data service is running an earlier version of TensorFlow that requires specifying `element_spec` as an argument to `from_dataset_id`. Please either supply an element spec or update the tf.data service to the latest version.') from err
        except RuntimeError:
            pass
        if not data_service_metadata or not data_service_metadata.element_spec:
            dataset_id_val = tensor_util.constant_value(dataset_id)
            raise ValueError(f'Failed to fetch element spec for dataset id {dataset_id_val} from tf.data service. If the dataset was registered in graph mode or inside a tf.function, the `element_spec` must be specified as an argument to `from_dataset_id`.')
        struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
        struct_pb.ParseFromString(data_service_metadata.element_spec)
        return nested_structure_coder.decode_proto(struct_pb)
    processing_mode = _get_validated_sharding_policy(processing_mode)
    if isinstance(service, tuple):
        (protocol, address) = service
    else:
        (protocol, address) = _parse_service(service)
    if job_name is not None:
        if not isinstance(job_name, str) and (not isinstance(job_name, tensor.Tensor)):
            raise ValueError(f'`job_name` must be a string or Tensor, but `job_name` was of type {type(job_name)}. job_name={job_name}.')
    if not element_spec:
        if not context.executing_eagerly():
            raise ValueError('In graph mode `element_spec` must be provided manually.')
        element_spec = _get_element_spec()
    dataset = _DataServiceDataset(dataset_id=dataset_id, processing_mode=processing_mode, address=address, element_spec=element_spec, protocol=protocol, data_transfer_protocol=data_transfer_protocol, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
    if job_name is not None:
        options = options_lib.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
    return dataset

@tf_export('data.experimental.service.from_dataset_id')
def from_dataset_id(processing_mode, service, dataset_id, element_spec=None, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, data_transfer_protocol=None, cross_trainer_cache=None, target_workers='AUTO') -> dataset_ops.Dataset:
    if False:
        return 10
    'Creates a dataset which reads data from the tf.data service.\n\n  This is useful when the dataset is registered by one process, then used in\n  another process. When the same process is both registering and reading from\n  the dataset, it is simpler to use `tf.data.experimental.service.distribute`\n  instead.\n\n  Before using `from_dataset_id`, the dataset must have been registered with the\n  tf.data service using `tf.data.experimental.service.register_dataset`.\n  `register_dataset` returns a dataset id for the registered dataset. That is\n  the `dataset_id` which should be passed to `from_dataset_id`.\n\n  The `element_spec` argument indicates the `tf.TypeSpec`s for the elements\n  produced by the dataset. Currently `element_spec` must be explicitly\n  specified, and match the dataset registered under `dataset_id`. `element_spec`\n  defaults to `None` so that in the future we can support automatically\n  discovering the `element_spec` by querying the tf.data service.\n\n  `tf.data.experimental.service.distribute` is a convenience method which\n  combines `register_dataset` and `from_dataset_id` into a dataset\n  transformation.\n  See the documentation for `tf.data.experimental.service.distribute` for more\n  detail about how `from_dataset_id` works.\n\n  >>> dispatcher = tf.data.experimental.service.DispatchServer()\n  >>> dispatcher_address = dispatcher.target.split("://")[1]\n  >>> worker = tf.data.experimental.service.WorkerServer(\n  ...     tf.data.experimental.service.WorkerConfig(\n  ...         dispatcher_address=dispatcher_address))\n  >>> dataset = tf.data.Dataset.range(10)\n  >>> dataset_id = tf.data.experimental.service.register_dataset(\n  ...     dispatcher.target, dataset)\n  >>> dataset = tf.data.experimental.service.from_dataset_id(\n  ...     processing_mode="parallel_epochs",\n  ...     service=dispatcher.target,\n  ...     dataset_id=dataset_id,\n  ...     element_spec=dataset.element_spec)\n  >>> print(list(dataset.as_numpy_iterator()))\n  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n  Args:\n    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying\n      how to shard the dataset among tf.data workers. See\n      `tf.data.experimental.service.ShardingPolicy` for details. For backwards\n      compatibility, `processing_mode` may also be set to the strings\n      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively\n      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.\n    service: A string or a tuple indicating how to connect to the tf.data\n      service. If it\'s a string, it should be in the format\n      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher\n        address and `<protocol>` can optionally be used to override the default\n        protocol to use. If it\'s a tuple, it should be (protocol, address).\n    dataset_id: The id of the dataset to read from. This id is returned by\n      `register_dataset` when the dataset is registered with the tf.data\n      service.\n    element_spec: A nested structure of `tf.TypeSpec`s representing the type of\n      elements produced by the dataset. This argument is only required inside a\n      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec\n      for a given dataset.\n    job_name: (Optional.) The name of the job. If provided, it must be a\n      non-empty string. This argument makes it possible for multiple datasets to\n      share the same job. The default behavior is that the dataset creates\n      anonymous, exclusively owned jobs.\n    consumer_index: (Optional.) The index of the consumer in the range from `0`\n      to `num_consumers`. Must be specified alongside `num_consumers`. When\n      specified, consumers will read from the job in a strict round-robin order,\n      instead of the default first-come-first-served order.\n    num_consumers: (Optional.) The number of consumers which will consume from\n      the job. Must be specified alongside `consumer_index`. When specified,\n      consumers will read from the job in a strict round-robin order, instead of\n      the default first-come-first-served order. When `num_consumers` is\n      specified, the dataset must have infinite cardinality to prevent a\n      producer from running out of data early and causing consumers to go out of\n      sync.\n    max_outstanding_requests: (Optional.) A limit on how many elements may be\n      requested at the same time. You can use this option to control the amount\n      of memory used, since `distribute` won\'t use more than `element_size` *\n      `max_outstanding_requests` of memory.\n    data_transfer_protocol: (Optional.) The protocol to use for transferring\n      data with the tf.data service. By default, data is transferred using gRPC.\n    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is\n      provided, dataset iteration will be shared across concurrently running\n      trainers. See\n      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers\n      for details.\n    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data\n      runtime decides which workers to read from. If `"ANY"`, reads from any\n      tf.data service workers. If `"LOCAL"`, only reads from local in-processs\n      tf.data service workers. `"AUTO"` works well for most cases, while users\n      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and\n      data copy if every TF worker colocates with a tf.data service worker.\n      Consumers of a shared job must use the same `target_workers`. Defaults to\n      `"AUTO"`.\n\n  Returns:\n    A `tf.data.Dataset` which reads from the tf.data service.\n  '
    _validate_job_name(job_name)
    if job_name is not None:
        job_name = string_ops.string_join(['dataset_id=', _to_string(dataset_id), job_name], '/')
    return _from_dataset_id(processing_mode=processing_mode, service=service, dataset_id=dataset_id, element_spec=element_spec, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, data_transfer_protocol=data_transfer_protocol, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)