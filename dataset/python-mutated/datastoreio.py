"""
A connector for reading from and writing to Google Cloud Datastore.

This module uses the newer google-cloud-datastore client package. Its API was
different enough to require extensive changes to this and associated modules.

**Updates to the I/O connector code**

For any significant updates to this I/O connector, please consider involving
corresponding code reviewers mentioned in
https://github.com/apache/beam/blob/master/sdks/python/OWNERS
"""
import logging
import time
from apache_beam import typehints
from apache_beam.internal.metrics.metric import ServiceCallMetric
from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.io.gcp import resource_identifiers
from apache_beam.io.gcp.datastore.v1new import helper
from apache_beam.io.gcp.datastore.v1new import query_splitter
from apache_beam.io.gcp.datastore.v1new import types
from apache_beam.io.gcp.datastore.v1new import util
from apache_beam.io.gcp.datastore.v1new.rampup_throttling_fn import RampupThrottlingFn
from apache_beam.metrics import monitoring_infos
from apache_beam.metrics.metric import Metrics
from apache_beam.transforms import Create
from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
from apache_beam.transforms import Reshuffle
from apache_beam.utils import retry
try:
    from apitools.base.py.exceptions import HttpError
    from google.api_core.exceptions import ClientError, GoogleAPICallError
except ImportError:
    pass
__all__ = ['ReadFromDatastore', 'WriteToDatastore', 'DeleteFromDatastore']
_LOGGER = logging.getLogger(__name__)

@typehints.with_output_types(types.Entity)
class ReadFromDatastore(PTransform):
    """A ``PTransform`` for querying Google Cloud Datastore.

  To read a ``PCollection[Entity]`` from a Cloud Datastore ``Query``, use
  the ``ReadFromDatastore`` transform by providing a `query` to
  read from. The project and optional namespace are set in the query.
  The query will be split into multiple queries to allow for parallelism. The
  degree of parallelism is automatically determined, but can be overridden by
  setting `num_splits` to a value of 1 or greater.

  Note: Normally, a runner will read from Cloud Datastore in parallel across
  many workers. However, when the `query` is configured with a `limit` or if the
  query contains inequality filters like `GREATER_THAN, LESS_THAN` etc., then
  all the returned results will be read by a single worker in order to ensure
  correct data. Since data is read from a single worker, this could have
  significant impact on the performance of the job. Using a
  :class:`~apache_beam.transforms.util.Reshuffle` transform after the read in
  this case might be beneficial for parallelizing work across workers.

  The semantics for query splitting is defined below:
    1. If `num_splits` is equal to 0, then the number of splits will be chosen
    dynamically at runtime based on the query data size.

    2. Any value of `num_splits` greater than
    `ReadFromDatastore._NUM_QUERY_SPLITS_MAX` will be capped at that value.

    3. If the `query` has a user limit set, or contains inequality filters, then
    `num_splits` will be ignored and no split will be performed.

    4. Under certain cases Cloud Datastore is unable to split query to the
    requested number of splits. In such cases we just use whatever Cloud
    Datastore returns.

  See https://developers.google.com/datastore/ for more details on Google Cloud
  Datastore.
  """
    _NUM_QUERY_SPLITS_MAX = 50000
    _NUM_QUERY_SPLITS_MIN = 12
    _DEFAULT_BUNDLE_SIZE_BYTES = 64 * 1024 * 1024

    def __init__(self, query, num_splits=0):
        if False:
            return 10
        'Initialize the `ReadFromDatastore` transform.\n\n    This transform outputs elements of type\n    :class:`~apache_beam.io.gcp.datastore.v1new.types.Entity`.\n\n    Args:\n      query: (:class:`~apache_beam.io.gcp.datastore.v1new.types.Query`) query\n        used to fetch entities.\n      num_splits: (:class:`int`) (optional) Number of splits for the query.\n    '
        super().__init__()
        if not query.project:
            raise ValueError('query.project cannot be empty')
        if not query:
            raise ValueError('query cannot be empty')
        if num_splits < 0:
            raise ValueError('num_splits must be greater than or equal 0')
        self._project = query.project
        self._datastore_namespace = query.namespace
        self._query = query
        self._num_splits = num_splits

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll.pipeline | 'UserQuery' >> Create([self._query]) | 'SplitQuery' >> ParDo(ReadFromDatastore._SplitQueryFn(self._num_splits)) | Reshuffle() | 'Read' >> ParDo(ReadFromDatastore._QueryFn())

    def display_data(self):
        if False:
            while True:
                i = 10
        disp_data = {'project': self._query.project, 'query': str(self._query), 'num_splits': self._num_splits}
        if self._datastore_namespace is not None:
            disp_data['namespace'] = self._datastore_namespace
        return disp_data

    @typehints.with_input_types(types.Query)
    @typehints.with_output_types(types.Query)
    class _SplitQueryFn(DoFn):
        """A `DoFn` that splits a given query into multiple sub-queries."""

        def __init__(self, num_splits):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self._num_splits = num_splits

        def process(self, query, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            client = helper.get_client(query.project, query.namespace)
            try:
                query_splitter.validate_split(query)
                if self._num_splits == 0:
                    estimated_num_splits = self.get_estimated_num_splits(client, query)
                else:
                    estimated_num_splits = self._num_splits
                _LOGGER.info('Splitting the query into %d splits', estimated_num_splits)
                query_splits = query_splitter.get_splits(client, query, estimated_num_splits)
            except query_splitter.QuerySplitterError:
                _LOGGER.info('Unable to parallelize the given query: %s', query, exc_info=True)
                query_splits = [query]
            return query_splits

        def display_data(self):
            if False:
                for i in range(10):
                    print('nop')
            disp_data = {'num_splits': self._num_splits}
            return disp_data

        @staticmethod
        def query_latest_statistics_timestamp(client):
            if False:
                for i in range(10):
                    print('nop')
            'Fetches the latest timestamp of statistics from Cloud Datastore.\n\n      Cloud Datastore system tables with statistics are periodically updated.\n      This method fetches the latest timestamp (in microseconds) of statistics\n      update using the `__Stat_Total__` table.\n      '
            if client.namespace is None:
                kind = '__Stat_Total__'
            else:
                kind = '__Stat_Ns_Total__'
            query = client.query(kind=kind, order=['-timestamp'])
            entities = list(query.fetch(limit=1))
            if not entities:
                raise RuntimeError('Datastore total statistics unavailable.')
            return entities[0]['timestamp']

        @staticmethod
        def get_estimated_size_bytes(client, query):
            if False:
                for i in range(10):
                    print('nop')
            "Get the estimated size of the data returned by this instance's query.\n\n      Cloud Datastore provides no way to get a good estimate of how large the\n      result of a query is going to be. Hence we use the __Stat_Kind__ system\n      table to get size of the entire kind as an approximate estimate, assuming\n      exactly 1 kind is specified in the query.\n      See https://cloud.google.com/datastore/docs/concepts/stats.\n      "
            kind_name = query.kind
            latest_timestamp = ReadFromDatastore._SplitQueryFn.query_latest_statistics_timestamp(client)
            _LOGGER.info('Latest stats timestamp for kind %s is %s', kind_name, latest_timestamp)
            if client.namespace is None:
                kind = '__Stat_Kind__'
            else:
                kind = '__Stat_Ns_Kind__'
            query = client.query(kind=kind)
            query.add_filter('kind_name', '=', kind_name)
            query.add_filter('timestamp', '=', latest_timestamp)
            entities = list(query.fetch(limit=1))
            if not entities:
                raise RuntimeError('Datastore statistics for kind %s unavailable' % kind_name)
            return entities[0]['entity_bytes']

        @staticmethod
        def get_estimated_num_splits(client, query):
            if False:
                for i in range(10):
                    print('nop')
            'Computes the number of splits to be performed on the query.'
            try:
                estimated_size_bytes = ReadFromDatastore._SplitQueryFn.get_estimated_size_bytes(client, query)
                _LOGGER.info('Estimated size bytes for query: %s', estimated_size_bytes)
                num_splits = int(min(ReadFromDatastore._NUM_QUERY_SPLITS_MAX, round(float(estimated_size_bytes) / ReadFromDatastore._DEFAULT_BUNDLE_SIZE_BYTES)))
            except Exception as e:
                _LOGGER.warning('Failed to fetch estimated size bytes: %s', e)
                num_splits = ReadFromDatastore._NUM_QUERY_SPLITS_MIN
            return max(num_splits, ReadFromDatastore._NUM_QUERY_SPLITS_MIN)

    @typehints.with_input_types(types.Query)
    @typehints.with_output_types(types.Entity)
    class _QueryFn(DoFn):
        """A DoFn that fetches entities from Cloud Datastore, for a given query."""

        def process(self, query, *unused_args, **unused_kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if query.namespace is None:
                query.namespace = ''
            _client = helper.get_client(query.project, query.namespace)
            client_query = query._to_client_query(_client)
            resource = resource_identifiers.DatastoreNamespace(query.project, query.namespace)
            labels = {monitoring_infos.SERVICE_LABEL: 'Datastore', monitoring_infos.METHOD_LABEL: 'BatchDatastoreRead', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.DATASTORE_NAMESPACE_LABEL: query.namespace, monitoring_infos.DATASTORE_PROJECT_ID_LABEL: query.project, monitoring_infos.STATUS_LABEL: 'ok'}
            service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
            try:
                for client_entity in client_query.fetch(query.limit):
                    yield types.Entity.from_client_entity(client_entity)
                service_call_metric.call('ok')
            except (ClientError, GoogleAPICallError) as e:
                service_call_metric.call(e.code.value)
                raise
            except HttpError as e:
                service_call_metric.call(e)
                raise

class _Mutate(PTransform):
    """A ``PTransform`` that writes mutations to Cloud Datastore.

  Only idempotent Datastore mutation operations (upsert and delete) are
  supported, as the commits are retried when failures occur.
  """
    _DEFAULT_HINT_NUM_WORKERS = 500

    def __init__(self, mutate_fn, throttle_rampup=True, hint_num_workers=_DEFAULT_HINT_NUM_WORKERS):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a Mutate transform.\n\n     Args:\n       mutate_fn: Instance of `DatastoreMutateFn` to use.\n       throttle_rampup: Whether to enforce a gradual ramp-up.\n       hint_num_workers: A hint for the expected number of workers, used to\n                         estimate appropriate limits during ramp-up throttling.\n     '
        self._mutate_fn = mutate_fn
        self._throttle_rampup = throttle_rampup
        self._hint_num_workers = hint_num_workers

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        if self._throttle_rampup:
            throttling_fn = RampupThrottlingFn(self._hint_num_workers)
            pcoll = pcoll | 'Enforce throttling during ramp-up' >> ParDo(throttling_fn)
        return pcoll | 'Write Batch to Datastore' >> ParDo(self._mutate_fn)

    class DatastoreMutateFn(DoFn):
        """A ``DoFn`` that write mutations to Datastore.

    Mutations are written in batches, where the maximum batch size is
    `util.WRITE_BATCH_SIZE`.

    Commits are non-transactional. If a commit fails because of a conflict over
    an entity group, the commit will be retried. This means that the mutation
    should be idempotent (`upsert` and `delete` mutations) to prevent duplicate
    data or errors.
    """

        def __init__(self, project):
            if False:
                print('Hello World!')
            '\n      Args:\n        project: (str) cloud project id\n      '
            self._project = project
            self._client = None
            self._rpc_successes = Metrics.counter(_Mutate.DatastoreMutateFn, 'datastoreRpcSuccesses')
            self._rpc_errors = Metrics.counter(_Mutate.DatastoreMutateFn, 'datastoreRpcErrors')
            self._throttled_secs = Metrics.counter(_Mutate.DatastoreMutateFn, 'cumulativeThrottlingSeconds')
            self._throttler = AdaptiveThrottler(window_ms=120000, bucket_ms=1000, overload_ratio=1.25)

        def _update_rpc_stats(self, successes=0, errors=0, throttled_secs=0):
            if False:
                return 10
            self._rpc_successes.inc(successes)
            self._rpc_errors.inc(errors)
            self._throttled_secs.inc(throttled_secs)

        def start_bundle(self):
            if False:
                while True:
                    i = 10
            self._client = helper.get_client(self._project, namespace=None)
            self._init_batch()
            self._batch_sizer = util.DynamicBatchSizer()
            self._target_batch_size = self._batch_sizer.get_batch_size(time.time() * 1000)

        def element_to_client_batch_item(self, element):
            if False:
                while True:
                    i = 10
            raise NotImplementedError

        def add_to_batch(self, client_batch_item):
            if False:
                return 10
            raise NotImplementedError

        @retry.with_exponential_backoff(num_retries=5, retry_filter=helper.retry_on_rpc_error)
        def write_mutations(self, throttler, rpc_stats_callback, throttle_delay=1):
            if False:
                while True:
                    i = 10
            'Writes a batch of mutations to Cloud Datastore.\n\n      If a commit fails, it will be retried up to 5 times. All mutations in the\n      batch will be committed again, even if the commit was partially\n      successful. If the retry limit is exceeded, the last exception from\n      Cloud Datastore will be raised.\n\n      Assumes that the Datastore client library does not perform any retries on\n      commits. It has not been determined how such retries would interact with\n      the retries and throttler used here.\n      See ``google.cloud.datastore_v1.gapic.datastore_client_config`` for\n      retry config.\n\n      Args:\n        rpc_stats_callback: a function to call with arguments `successes` and\n            `failures` and `throttled_secs`; this is called to record successful\n            and failed RPCs to Datastore and time spent waiting for throttling.\n        throttler: (``apache_beam.io.gcp.datastore.v1new.adaptive_throttler.\n          AdaptiveThrottler``)\n          Throttler instance used to select requests to be throttled.\n        throttle_delay: (:class:`float`) time in seconds to sleep when\n            throttled.\n\n      Returns:\n        (int) The latency of the successful RPC in milliseconds.\n      '
            while throttler.throttle_request(time.time() * 1000):
                _LOGGER.info('Delaying request for %ds due to previous failures', throttle_delay)
                time.sleep(throttle_delay)
                rpc_stats_callback(throttled_secs=throttle_delay)
            if self._batch is None:
                self._batch = self._client.batch()
                self._batch.begin()
                for element in self._batch_elements:
                    self.add_to_batch(element)
            resource = resource_identifiers.DatastoreNamespace(self._project, '')
            labels = {monitoring_infos.SERVICE_LABEL: 'Datastore', monitoring_infos.METHOD_LABEL: 'BatchDatastoreWrite', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.DATASTORE_NAMESPACE_LABEL: '', monitoring_infos.DATASTORE_PROJECT_ID_LABEL: self._project, monitoring_infos.STATUS_LABEL: 'ok'}
            service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
            try:
                start_time = time.time()
                self._batch.commit()
                end_time = time.time()
                service_call_metric.call('ok')
                rpc_stats_callback(successes=1)
                throttler.successful_request(start_time * 1000)
                commit_time_ms = int((end_time - start_time) * 1000)
                return commit_time_ms
            except (ClientError, GoogleAPICallError) as e:
                self._batch = None
                service_call_metric.call(e.code.value)
                rpc_stats_callback(errors=1)
                raise
            except HttpError as e:
                service_call_metric.call(e)
                rpc_stats_callback(errors=1)
                raise

        def process(self, element):
            if False:
                while True:
                    i = 10
            client_element = self.element_to_client_batch_item(element)
            self._batch_elements.append(client_element)
            self.add_to_batch(client_element)
            self._batch_bytes_size += self._batch.mutations[-1]._pb.ByteSize()
            if len(self._batch.mutations) >= self._target_batch_size or self._batch_bytes_size > util.WRITE_BATCH_MAX_BYTES_SIZE:
                self._flush_batch()

        def finish_bundle(self):
            if False:
                for i in range(10):
                    print('nop')
            if self._batch_elements:
                self._flush_batch()

        def _init_batch(self):
            if False:
                i = 10
                return i + 15
            self._batch_bytes_size = 0
            self._batch = self._client.batch()
            self._batch.begin()
            self._batch_elements = []

        def _flush_batch(self):
            if False:
                for i in range(10):
                    print('nop')
            latency_ms = self.write_mutations(self._throttler, rpc_stats_callback=self._update_rpc_stats, throttle_delay=util.WRITE_BATCH_TARGET_LATENCY_MS // 1000)
            _LOGGER.debug('Successfully wrote %d mutations in %dms.', len(self._batch.mutations), latency_ms)
            now = time.time() * 1000
            self._batch_sizer.report_latency(now, latency_ms, len(self._batch.mutations))
            self._target_batch_size = self._batch_sizer.get_batch_size(now)
            self._init_batch()

@typehints.with_input_types(types.Entity)
class WriteToDatastore(_Mutate):
    """
  Writes elements of type
  :class:`~apache_beam.io.gcp.datastore.v1new.types.Entity` to Cloud Datastore.

  Entity keys must be complete. The ``project`` field in each key must match the
  project ID passed to this transform. If ``project`` field in entity or
  property key is empty then it is filled with the project ID passed to this
  transform.
  """

    def __init__(self, project, throttle_rampup=True, hint_num_workers=_Mutate._DEFAULT_HINT_NUM_WORKERS):
        if False:
            while True:
                i = 10
        'Initialize the `WriteToDatastore` transform.\n\n    Args:\n      project: (:class:`str`) The ID of the project to write entities to.\n      throttle_rampup: Whether to enforce a gradual ramp-up.\n      hint_num_workers: A hint for the expected number of workers, used to\n                        estimate appropriate limits during ramp-up throttling.\n    '
        mutate_fn = WriteToDatastore._DatastoreWriteFn(project)
        super().__init__(mutate_fn, throttle_rampup, hint_num_workers)

    class _DatastoreWriteFn(_Mutate.DatastoreMutateFn):

        def element_to_client_batch_item(self, element):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(element, types.Entity):
                raise ValueError('apache_beam.io.gcp.datastore.v1new.datastoreio.Entity expected, got: %s' % type(element))
            if not element.key.project:
                element.key.project = self._project
            client_entity = element.to_client_entity()
            if client_entity.key.is_partial:
                raise ValueError('Entities to be written to Cloud Datastore must have complete keys:\n%s' % client_entity)
            return client_entity

        def add_to_batch(self, client_entity):
            if False:
                while True:
                    i = 10
            self._batch.put(client_entity)

        def display_data(self):
            if False:
                i = 10
                return i + 15
            return {'mutation': 'Write (upsert)', 'project': self._project}

@typehints.with_input_types(types.Key)
class DeleteFromDatastore(_Mutate):
    """
  Deletes elements matching input
  :class:`~apache_beam.io.gcp.datastore.v1new.types.Key` elements from Cloud
  Datastore.

  Keys must be complete. The ``project`` field in each key must match the
  project ID passed to this transform. If ``project`` field in key is empty then
  it is filled with the project ID passed to this transform.
  """

    def __init__(self, project, throttle_rampup=True, hint_num_workers=_Mutate._DEFAULT_HINT_NUM_WORKERS):
        if False:
            while True:
                i = 10
        'Initialize the `DeleteFromDatastore` transform.\n\n    Args:\n      project: (:class:`str`) The ID of the project from which the entities will\n        be deleted.\n      throttle_rampup: Whether to enforce a gradual ramp-up.\n      hint_num_workers: A hint for the expected number of workers, used to\n                        estimate appropriate limits during ramp-up throttling.\n    '
        mutate_fn = DeleteFromDatastore._DatastoreDeleteFn(project)
        super().__init__(mutate_fn, throttle_rampup, hint_num_workers)

    class _DatastoreDeleteFn(_Mutate.DatastoreMutateFn):

        def element_to_client_batch_item(self, element):
            if False:
                return 10
            if not isinstance(element, types.Key):
                raise ValueError('apache_beam.io.gcp.datastore.v1new.datastoreio.Key expected, got: %s' % type(element))
            if not element.project:
                element.project = self._project
            client_key = element.to_client_key()
            if client_key.is_partial:
                raise ValueError('Keys to be deleted from Cloud Datastore must be complete:\n%s' % client_key)
            return client_key

        def add_to_batch(self, client_key):
            if False:
                while True:
                    i = 10
            self._batch.delete(client_key)

        def display_data(self):
            if False:
                for i in range(10):
                    print('nop')
            return {'mutation': 'Delete', 'project': self._project}