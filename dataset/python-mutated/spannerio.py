"""Google Cloud Spanner IO

Experimental; no backwards-compatibility guarantees.

This is an experimental module for reading and writing data from Google Cloud
Spanner. Visit: https://cloud.google.com/spanner for more details.

Reading Data from Cloud Spanner.

To read from Cloud Spanner apply ReadFromSpanner transformation. It will
return a PCollection, where each element represents an individual row returned
from the read operation. Both Query and Read APIs are supported.

ReadFromSpanner relies on the ReadOperation objects which is exposed by the
SpannerIO API. ReadOperation holds the immutable data which is responsible to
execute batch and naive reads on Cloud Spanner. This is done for more
convenient programming.

ReadFromSpanner reads from Cloud Spanner by providing either an 'sql' param
in the constructor or 'table' name with 'columns' as list. For example:::

  records = (pipeline
            | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
            sql='Select * from users'))

  records = (pipeline
            | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
            table='users', columns=['id', 'name', 'email']))

You can also perform multiple reads by providing a list of ReadOperations
to the ReadFromSpanner transform constructor. ReadOperation exposes two static
methods. Use 'query' to perform sql based reads, 'table' to perform read from
table name. For example:::

  read_operations = [
                      ReadOperation.table(table='customers', columns=['name',
                      'email']),
                      ReadOperation.table(table='vendors', columns=['name',
                      'email']),
                    ]
  all_users = pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
        read_operations=read_operations)

  ...OR...

  read_operations = [
                      ReadOperation.query(sql='Select name, email from
                      customers'),
                      ReadOperation.query(
                        sql='Select * from users where id <= @user_id',
                        params={'user_id': 100},
                        params_type={'user_id': param_types.INT64}
                      ),
                    ]
  # `params_types` are instance of `google.cloud.spanner.param_types`
  all_users = pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
        read_operations=read_operations)

For more information, please review the docs on class ReadOperation.

User can also able to provide the ReadOperation in form of PCollection via
pipeline. For example:::

  users = (pipeline
           | beam.Create([ReadOperation...])
           | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME))

User may also create cloud spanner transaction from the transform called
`create_transaction` which is available in the SpannerIO API.

The transform is guaranteed to be executed on a consistent snapshot of data,
utilizing the power of read only transactions. Staleness of data can be
controlled by providing the `read_timestamp` or `exact_staleness` param values
in the constructor.

This transform requires root of the pipeline (PBegin) and returns PTransform
which is passed later to the `ReadFromSpanner` constructor. `ReadFromSpanner`
pass this transaction PTransform as a singleton side input to the
`_NaiveSpannerReadDoFn` containing 'session_id' and 'transaction_id'.
For example:::

  transaction = (pipeline | create_transaction(TEST_PROJECT_ID,
                                              TEST_INSTANCE_ID,
                                              DB_NAME))

  users = pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
        sql='Select * from users', transaction=transaction)

  tweets = pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME,
        sql='Select * from tweets', transaction=transaction)

For further details of this transform, please review the docs on the
:meth:`create_transaction` method available in the SpannerIO API.

ReadFromSpanner takes this transform in the constructor and pass this to the
read pipeline as the singleton side input.

Writing Data to Cloud Spanner.

The WriteToSpanner transform writes to Cloud Spanner by executing a
collection a input rows (WriteMutation). The mutations are grouped into
batches for efficiency.

WriteToSpanner transform relies on the WriteMutation objects which is exposed
by the SpannerIO API. WriteMutation have five static methods (insert, update,
insert_or_update, replace, delete). These methods returns the instance of the
_Mutator object which contains the mutation type and the Spanner Mutation
object. For more details, review the docs of the class SpannerIO.WriteMutation.
For example:::

  mutations = [
                WriteMutation.insert(table='user', columns=('name', 'email'),
                values=[('sara', 'sara@dev.com')])
              ]
  _ = (p
       | beam.Create(mutations)
       | WriteToSpanner(
          project_id=SPANNER_PROJECT_ID,
          instance_id=SPANNER_INSTANCE_ID,
          database_id=SPANNER_DATABASE_NAME)
        )

You can also create WriteMutation via calling its constructor. For example:::

  mutations = [
      WriteMutation(insert='users', columns=('name', 'email'),
                    values=[('sara", 'sara@example.com')])
  ]

For more information, review the docs available on WriteMutation class.

WriteToSpanner transform also takes three batching parameters (max_number_rows,
max_number_cells and max_batch_size_bytes). By default, max_number_rows is set
to 50 rows, max_number_cells is set to 500 cells and max_batch_size_bytes is
set to 1MB (1048576 bytes). These parameter used to reduce the number of
transactions sent to spanner by grouping the mutation into batches. Setting
these param values either to smaller value or zero to disable batching.
Unlike the Java connector, this connector does not create batches of
transactions sorted by table and primary key.

WriteToSpanner transforms starts with the grouping into batches. The first step
in this process is to make the mutation groups of the WriteMutation
objects and then filtering them into batchable and unbatchable mutation
groups. There are three batching parameters (max_number_cells, max_number_rows
& max_batch_size_bytes). We calculated th mutation byte size from the method
available in the `google.cloud.spanner_v1.proto.mutation_pb2.Mutation.ByteSize`.
if the mutation rows, cells or byte size are larger than value of the any
batching parameters param, it will be tagged as "unbatchable" mutation. After
this all the batchable mutation are merged into a single mutation group whos
size is not larger than the "max_batch_size_bytes", after this process, all the
mutation groups together to process. If the Mutation references a table or
column does not exits, it will cause a exception and fails the entire pipeline.
"""
import typing
from collections import deque
from collections import namedtuple
from apache_beam import Create
from apache_beam import DoFn
from apache_beam import Flatten
from apache_beam import ParDo
from apache_beam import Reshuffle
from apache_beam.internal.metrics.metric import ServiceCallMetric
from apache_beam.io.gcp import resource_identifiers
from apache_beam.metrics import Metrics
from apache_beam.metrics import monitoring_infos
from apache_beam.pvalue import AsSingleton
from apache_beam.pvalue import PBegin
from apache_beam.pvalue import TaggedOutput
from apache_beam.transforms import PTransform
from apache_beam.transforms import ptransform_fn
from apache_beam.transforms import window
from apache_beam.transforms.display import DisplayDataItem
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types
try:
    from google.cloud.spanner import Client
    from google.cloud.spanner import KeySet
    from google.cloud.spanner_v1 import batch
    from google.cloud.spanner_v1.database import BatchSnapshot
    from google.api_core.exceptions import ClientError, GoogleAPICallError
    from apitools.base.py.exceptions import HttpError
except ImportError:
    Client = None
    KeySet = None
    BatchSnapshot = None
try:
    from google.cloud.spanner_v1 import Mutation
except ImportError:
    try:
        from google.cloud.spanner_v1.proto.mutation_pb2 import Mutation
    except ImportError:
        pass
__all__ = ['create_transaction', 'ReadFromSpanner', 'ReadOperation', 'WriteToSpanner', 'WriteMutation', 'MutationGroup']

class _SPANNER_TRANSACTION(namedtuple('SPANNER_TRANSACTION', ['transaction'])):
    """
  Holds the spanner transaction details.
  """
    __slots__ = ()

class ReadOperation(namedtuple('ReadOperation', ['is_sql', 'is_table', 'read_operation', 'kwargs'])):
    """
  Encapsulates a spanner read operation.
  """
    __slots__ = ()

    @classmethod
    def query(cls, sql, params=None, param_types=None):
        if False:
            i = 10
            return i + 15
        '\n    A convenient method to construct ReadOperation from sql query.\n\n    Args:\n      sql: SQL query statement\n      params: (optional) values for parameter replacement. Keys must match the\n        names used in sql\n      param_types: (optional) maps explicit types for one or more param values;\n        required if parameters are passed.\n    '
        if params:
            assert param_types is not None
        return cls(is_sql=True, is_table=False, read_operation='process_query_batch', kwargs={'sql': sql, 'params': params, 'param_types': param_types})

    @classmethod
    def table(cls, table, columns, index='', keyset=None):
        if False:
            for i in range(10):
                print('nop')
        "\n    A convenient method to construct ReadOperation from table.\n\n    Args:\n      table: name of the table from which to fetch data.\n      columns: names of columns to be retrieved.\n      index: (optional) name of index to use, rather than the table's primary\n        key.\n      keyset: (optional) `KeySet` keys / ranges identifying rows to be\n        retrieved.\n    "
        keyset = keyset or KeySet(all_=True)
        if not isinstance(keyset, KeySet):
            raise ValueError('keyset must be an instance of class google.cloud.spanner.KeySet')
        return cls(is_sql=False, is_table=True, read_operation='process_read_batch', kwargs={'table': table, 'columns': columns, 'index': index, 'keyset': keyset})

class _BeamSpannerConfiguration(namedtuple('_BeamSpannerConfiguration', ['project', 'instance', 'database', 'table', 'query_name', 'credentials', 'pool', 'snapshot_read_timestamp', 'snapshot_exact_staleness'])):
    """
  A namedtuple holds the immutable data of the connection string to the cloud
  spanner.
  """

    @property
    def snapshot_options(self):
        if False:
            return 10
        snapshot_options = {}
        if self.snapshot_exact_staleness:
            snapshot_options['exact_staleness'] = self.snapshot_exact_staleness
        if self.snapshot_read_timestamp:
            snapshot_options['read_timestamp'] = self.snapshot_read_timestamp
        return snapshot_options

@with_input_types(ReadOperation, _SPANNER_TRANSACTION)
@with_output_types(typing.List[typing.Any])
class _NaiveSpannerReadDoFn(DoFn):

    def __init__(self, spanner_configuration):
        if False:
            return 10
        '\n    A naive version of Spanner read which uses the transaction API of the\n    cloud spanner.\n    https://googleapis.dev/python/spanner/latest/transaction-api.html\n    In Naive reads, this transform performs single reads, where as the\n    Batch reads use the spanner partitioning query to create batches.\n\n    Args:\n      spanner_configuration: (_BeamSpannerConfiguration) Connection details to\n        connect with cloud spanner.\n    '
        self._spanner_configuration = spanner_configuration
        self._snapshot = None
        self._session = None
        self.base_labels = {monitoring_infos.SERVICE_LABEL: 'Spanner', monitoring_infos.METHOD_LABEL: 'Read', monitoring_infos.SPANNER_PROJECT_ID: self._spanner_configuration.project, monitoring_infos.SPANNER_DATABASE_ID: self._spanner_configuration.database}

    def _table_metric(self, table_id, status):
        if False:
            i = 10
            return i + 15
        database_id = self._spanner_configuration.database
        project_id = self._spanner_configuration.project
        resource = resource_identifiers.SpannerTable(project_id, database_id, table_id)
        labels = {**self.base_labels, monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.SPANNER_TABLE_ID: table_id}
        service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
        service_call_metric.call(str(status))

    def _query_metric(self, query_name, status):
        if False:
            print('Hello World!')
        project_id = self._spanner_configuration.project
        resource = resource_identifiers.SpannerSqlQuery(project_id, query_name)
        labels = {**self.base_labels, monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.SPANNER_QUERY_NAME: query_name}
        service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
        service_call_metric.call(str(status))

    def _get_session(self):
        if False:
            print('Hello World!')
        if self._session is None:
            session = self._session = self._database.session()
            session.create()
        return self._session

    def _close_session(self):
        if False:
            print('Hello World!')
        if self._session is not None:
            self._session.delete()

    def setup(self):
        if False:
            i = 10
            return i + 15
        spanner_client = Client(self._spanner_configuration.project)
        instance = spanner_client.instance(self._spanner_configuration.instance)
        self._database = instance.database(self._spanner_configuration.database, pool=self._spanner_configuration.pool)

    def process(self, element, spanner_transaction):
        if False:
            while True:
                i = 10
        if not isinstance(spanner_transaction, _SPANNER_TRANSACTION):
            raise ValueError('Invalid transaction object: %s. It should be instance of SPANNER_TRANSACTION object created by spannerio.create_transaction transform.' % type(spanner_transaction))
        transaction_info = spanner_transaction.transaction
        self._snapshot = BatchSnapshot.from_dict(self._database, transaction_info)
        with self._get_session().transaction() as transaction:
            table_id = self._spanner_configuration.table
            query_name = self._spanner_configuration.query_name or ''
            if element.is_sql is True:
                transaction_read = transaction.execute_sql
                metric_action = self._query_metric
                metric_id = query_name
            elif element.is_table is True:
                transaction_read = transaction.read
                metric_action = self._table_metric
                metric_id = table_id
            else:
                raise ValueError('ReadOperation is improperly configure: %s' % str(element))
            try:
                for row in transaction_read(**element.kwargs):
                    yield row
                metric_action(metric_id, 'ok')
            except (ClientError, GoogleAPICallError) as e:
                metric_action(metric_id, e.code.value)
                raise
            except HttpError as e:
                metric_action(metric_id, e)
                raise

@with_input_types(ReadOperation)
@with_output_types(typing.Dict[typing.Any, typing.Any])
class _CreateReadPartitions(DoFn):
    """
  A DoFn to create partitions. Uses the Partitioning API (PartitionRead /
  PartitionQuery) request to start a partitioned query operation. Returns a
  list of batch information needed to perform the actual queries.

  If the element is the instance of :class:`ReadOperation` is to perform sql
  query, `PartitionQuery` API is used the create partitions and returns mappings
  of information used perform actual partitioned reads via
  :meth:`process_query_batch`.

  If the element is the instance of :class:`ReadOperation` is to perform read
  from table, `PartitionRead` API is used the create partitions and returns
  mappings of information used perform actual partitioned reads via
  :meth:`process_read_batch`.
  """

    def __init__(self, spanner_configuration):
        if False:
            print('Hello World!')
        self._spanner_configuration = spanner_configuration

    def setup(self):
        if False:
            i = 10
            return i + 15
        spanner_client = Client(project=self._spanner_configuration.project, credentials=self._spanner_configuration.credentials)
        instance = spanner_client.instance(self._spanner_configuration.instance)
        self._database = instance.database(self._spanner_configuration.database, pool=self._spanner_configuration.pool)
        self._snapshot = self._database.batch_snapshot(**self._spanner_configuration.snapshot_options)
        self._snapshot_dict = self._snapshot.to_dict()

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        if element.is_sql is True:
            partitioning_action = self._snapshot.generate_query_batches
        elif element.is_table is True:
            partitioning_action = self._snapshot.generate_read_batches
        else:
            raise ValueError('ReadOperation is improperly configure: %s' % str(element))
        for p in partitioning_action(**element.kwargs):
            yield {'is_sql': element.is_sql, 'is_table': element.is_table, 'read_operation': element.read_operation, 'partitions': p, 'transaction_info': self._snapshot_dict}

@with_input_types(int)
@with_output_types(_SPANNER_TRANSACTION)
class _CreateTransactionFn(DoFn):
    """
  A DoFn to create the transaction of cloud spanner.
  It connects to the database and and returns the transaction_id and session_id
  by using the batch_snapshot.to_dict() method available in the google cloud
  spanner sdk.

  https://googleapis.dev/python/spanner/latest/database-api.html?highlight=
  batch_snapshot#google.cloud.spanner_v1.database.BatchSnapshot.to_dict
  """

    def __init__(self, project_id, instance_id, database_id, credentials, pool, read_timestamp, exact_staleness):
        if False:
            print('Hello World!')
        self._project_id = project_id
        self._instance_id = instance_id
        self._database_id = database_id
        self._credentials = credentials
        self._pool = pool
        self._snapshot_options = {}
        if read_timestamp:
            self._snapshot_options['read_timestamp'] = read_timestamp
        if exact_staleness:
            self._snapshot_options['exact_staleness'] = exact_staleness
        self._snapshot = None

    def setup(self):
        if False:
            return 10
        self._spanner_client = Client(project=self._project_id, credentials=self._credentials)
        self._instance = self._spanner_client.instance(self._instance_id)
        self._database = self._instance.database(self._database_id, pool=self._pool)

    def process(self, element, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._snapshot = self._database.batch_snapshot(**self._snapshot_options)
        return [_SPANNER_TRANSACTION(self._snapshot.to_dict())]

@ptransform_fn
def create_transaction(pbegin, project_id, instance_id, database_id, credentials=None, pool=None, read_timestamp=None, exact_staleness=None):
    if False:
        i = 10
        return i + 15
    '\n  A PTransform method to create a batch transaction.\n\n  Args:\n    pbegin: Root of the pipeline\n    project_id: Cloud spanner project id. Be sure to use the Project ID,\n      not the Project Number.\n    instance_id: Cloud spanner instance id.\n    database_id: Cloud spanner database id.\n    credentials: (optional) The authorization credentials to attach to requests.\n      These credentials identify this application to the service.\n      If none are specified, the client will attempt to ascertain\n      the credentials from the environment.\n    pool: (optional) session pool to be used by database. If not passed,\n      Spanner Cloud SDK uses the BurstyPool by default.\n      `google.cloud.spanner.BurstyPool`. Ref:\n      https://googleapis.dev/python/spanner/latest/database-api.html?#google.\n      cloud.spanner_v1.database.Database\n    read_timestamp: (optional) An instance of the `datetime.datetime` object to\n      execute all reads at the given timestamp.\n    exact_staleness: (optional) An instance of the `datetime.timedelta`\n      object. These timestamp bounds execute reads at a user-specified\n      timestamp.\n  '
    assert isinstance(pbegin, PBegin)
    return pbegin | Create([1]) | ParDo(_CreateTransactionFn(project_id, instance_id, database_id, credentials, pool, read_timestamp, exact_staleness))

@with_input_types(typing.Dict[typing.Any, typing.Any])
@with_output_types(typing.List[typing.Any])
class _ReadFromPartitionFn(DoFn):
    """
  A DoFn to perform reads from the partition.
  """

    def __init__(self, spanner_configuration):
        if False:
            i = 10
            return i + 15
        self._spanner_configuration = spanner_configuration
        self.base_labels = {monitoring_infos.SERVICE_LABEL: 'Spanner', monitoring_infos.METHOD_LABEL: 'Read', monitoring_infos.SPANNER_PROJECT_ID: self._spanner_configuration.project, monitoring_infos.SPANNER_DATABASE_ID: self._spanner_configuration.database}
        self.service_metric = None

    def _table_metric(self, table_id):
        if False:
            print('Hello World!')
        database_id = self._spanner_configuration.database
        project_id = self._spanner_configuration.project
        resource = resource_identifiers.SpannerTable(project_id, database_id, table_id)
        labels = {**self.base_labels, monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.SPANNER_TABLE_ID: table_id}
        service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
        return service_call_metric

    def _query_metric(self, query_name):
        if False:
            print('Hello World!')
        project_id = self._spanner_configuration.project
        resource = resource_identifiers.SpannerSqlQuery(project_id, query_name)
        labels = {**self.base_labels, monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.SPANNER_QUERY_NAME: query_name}
        service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
        return service_call_metric

    def setup(self):
        if False:
            i = 10
            return i + 15
        spanner_client = Client(self._spanner_configuration.project)
        instance = spanner_client.instance(self._spanner_configuration.instance)
        self._database = instance.database(self._spanner_configuration.database, pool=self._spanner_configuration.pool)
        self._snapshot = self._database.batch_snapshot(**self._spanner_configuration.snapshot_options)

    def process(self, element):
        if False:
            i = 10
            return i + 15
        self._snapshot = BatchSnapshot.from_dict(self._database, element['transaction_info'])
        table_id = self._spanner_configuration.table
        query_name = self._spanner_configuration.query_name or ''
        if element['is_sql'] is True:
            read_action = self._snapshot.process_query_batch
            self.service_metric = self._query_metric(query_name)
        elif element['is_table'] is True:
            read_action = self._snapshot.process_read_batch
            self.service_metric = self._table_metric(table_id)
        else:
            raise ValueError('ReadOperation is improperly configure: %s' % str(element))
        try:
            for row in read_action(element['partitions']):
                yield row
            self.service_metric.call('ok')
        except (ClientError, GoogleAPICallError) as e:
            self.service_metric(str(e.code.value))
            raise
        except HttpError as e:
            self.service_metric(str(e))
            raise

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        if self._snapshot:
            self._snapshot.close()

class ReadFromSpanner(PTransform):
    """
  A PTransform to perform reads from cloud spanner.
  ReadFromSpanner uses BatchAPI to perform all read operations.
  """

    def __init__(self, project_id, instance_id, database_id, pool=None, read_timestamp=None, exact_staleness=None, credentials=None, sql=None, params=None, param_types=None, table=None, query_name=None, columns=None, index='', keyset=None, read_operations=None, transaction=None):
        if False:
            return 10
        "\n    A PTransform that uses Spanner Batch API to perform reads.\n\n    Args:\n      project_id: Cloud spanner project id. Be sure to use the Project ID,\n        not the Project Number.\n      instance_id: Cloud spanner instance id.\n      database_id: Cloud spanner database id.\n      pool: (optional) session pool to be used by database. If not passed,\n        Spanner Cloud SDK uses the BurstyPool by default.\n        `google.cloud.spanner.BurstyPool`. Ref:\n        https://googleapis.dev/python/spanner/latest/database-api.html?#google.\n        cloud.spanner_v1.database.Database\n      read_timestamp: (optional) An instance of the `datetime.datetime` object\n        to execute all reads at the given timestamp. By default, set to `None`.\n      exact_staleness: (optional) An instance of the `datetime.timedelta`\n        object. These timestamp bounds execute reads at a user-specified\n        timestamp. By default, set to `None`.\n      credentials: (optional) The authorization credentials to attach to\n        requests. These credentials identify this application to the service.\n        If none are specified, the client will attempt to ascertain\n        the credentials from the environment. By default, set to `None`.\n      sql: (optional) SQL query statement.\n      params: (optional) Values for parameter replacement. Keys must match the\n        names used in sql. By default, set to `None`.\n      param_types: (optional) maps explicit types for one or more param values;\n        required if params are passed. By default, set to `None`.\n      table: (optional) Name of the table from which to fetch data. By\n        default, set to `None`.\n      columns: (optional) List of names of columns to be retrieved; required if\n        the table is passed. By default, set to `None`.\n      index: (optional) name of index to use, rather than the table's primary\n        key. By default, set to `None`.\n      keyset: (optional) keys / ranges identifying rows to be retrieved. By\n        default, set to `None`.\n      read_operations: (optional) List of the objects of :class:`ReadOperation`\n        to perform read all. By default, set to `None`.\n      transaction: (optional) PTransform of the :meth:`create_transaction` to\n        perform naive read on cloud spanner. By default, set to `None`.\n    "
        self._configuration = _BeamSpannerConfiguration(project=project_id, instance=instance_id, database=database_id, table=table, query_name=query_name, credentials=credentials, pool=pool, snapshot_read_timestamp=read_timestamp, snapshot_exact_staleness=exact_staleness)
        self._read_operations = read_operations
        self._transaction = transaction
        if self._read_operations is None:
            if table is not None:
                if columns is None:
                    raise ValueError('Columns are required with the table name.')
                self._read_operations = [ReadOperation.table(table=table, columns=columns, index=index, keyset=keyset)]
            elif sql is not None:
                self._read_operations = [ReadOperation.query(sql=sql, params=params, param_types=param_types)]

    def expand(self, pbegin):
        if False:
            print('Hello World!')
        if self._read_operations is not None and isinstance(pbegin, PBegin):
            pcoll = pbegin.pipeline | Create(self._read_operations)
        elif not isinstance(pbegin, PBegin):
            if self._read_operations is not None:
                raise ValueError('Read operation in the constructor only works with the root of the pipeline.')
            pcoll = pbegin
        else:
            raise ValueError('Spanner required read operation, sql or table with columns.')
        if self._transaction is None:
            p = pcoll | 'Generate Partitions' >> ParDo(_CreateReadPartitions(spanner_configuration=self._configuration)) | 'Reshuffle' >> Reshuffle() | 'Read From Partitions' >> ParDo(_ReadFromPartitionFn(spanner_configuration=self._configuration))
        else:
            p = pcoll | 'Reshuffle' >> Reshuffle().with_input_types(ReadOperation) | 'Perform Read' >> ParDo(_NaiveSpannerReadDoFn(spanner_configuration=self._configuration), AsSingleton(self._transaction))
        return p

    def display_data(self):
        if False:
            while True:
                i = 10
        res = {}
        sql = []
        table = []
        if self._read_operations is not None:
            for ro in self._read_operations:
                if ro.is_sql is True:
                    sql.append(ro.kwargs)
                elif ro.is_table is True:
                    table.append(ro.kwargs)
            if sql:
                res['sql'] = DisplayDataItem(str(sql), label='Sql')
            if table:
                res['table'] = DisplayDataItem(str(table), label='Table')
        if self._transaction:
            res['transaction'] = DisplayDataItem(str(self._transaction), label='transaction')
        return res

class WriteToSpanner(PTransform):

    def __init__(self, project_id, instance_id, database_id, pool=None, credentials=None, max_batch_size_bytes=1048576, max_number_rows=50, max_number_cells=500):
        if False:
            return 10
        '\n    A PTransform to write onto Google Cloud Spanner.\n\n    Args:\n      project_id: Cloud spanner project id. Be sure to use the Project ID,\n        not the Project Number.\n      instance_id: Cloud spanner instance id.\n      database_id: Cloud spanner database id.\n      max_batch_size_bytes: (optional) Split the mutations into batches to\n        reduce the number of transaction sent to Spanner. By default it is\n        set to 1 MB (1048576 Bytes).\n      max_number_rows: (optional) Split the mutations into batches to\n        reduce the number of transaction sent to Spanner. By default it is\n        set to 50 rows per batch.\n      max_number_cells: (optional) Split the mutations into batches to\n        reduce the number of transaction sent to Spanner. By default it is\n        set to 500 cells per batch.\n    '
        self._configuration = _BeamSpannerConfiguration(project=project_id, instance=instance_id, database=database_id, table=None, query_name=None, credentials=credentials, pool=pool, snapshot_read_timestamp=None, snapshot_exact_staleness=None)
        self._max_batch_size_bytes = max_batch_size_bytes
        self._max_number_rows = max_number_rows
        self._max_number_cells = max_number_cells
        self._database_id = database_id
        self._project_id = project_id
        self._instance_id = instance_id
        self._pool = pool

    def display_data(self):
        if False:
            return 10
        res = {'project_id': DisplayDataItem(self._project_id, label='Project Id'), 'instance_id': DisplayDataItem(self._instance_id, label='Instance Id'), 'pool': DisplayDataItem(str(self._pool), label='Pool'), 'database': DisplayDataItem(self._database_id, label='Database'), 'batch_size': DisplayDataItem(self._max_batch_size_bytes, label='Batch Size'), 'max_number_rows': DisplayDataItem(self._max_number_rows, label='Max Rows'), 'max_number_cells': DisplayDataItem(self._max_number_cells, label='Max Cells')}
        return res

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | 'make batches' >> _WriteGroup(max_batch_size_bytes=self._max_batch_size_bytes, max_number_rows=self._max_number_rows, max_number_cells=self._max_number_cells) | 'Writing to spanner' >> ParDo(_WriteToSpannerDoFn(self._configuration))

class _Mutator(namedtuple('_Mutator', ['mutation', 'operation', 'kwargs', 'rows', 'cells'])):
    __slots__ = ()

    @property
    def byte_size(self):
        if False:
            while True:
                i = 10
        if hasattr(self.mutation, '_pb'):
            return self.mutation._pb.ByteSize()
        else:
            return self.mutation.ByteSize()

class MutationGroup(deque):
    """
  A Bundle of Spanner Mutations (_Mutator).
  """

    @property
    def info(self):
        if False:
            return 10
        cells = 0
        rows = 0
        bytes = 0
        for m in self.__iter__():
            bytes += m.byte_size
            rows += m.rows
            cells += m.cells
        return {'rows': rows, 'cells': cells, 'byte_size': bytes}

    def primary(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self.__iter__())

class WriteMutation(object):
    _OPERATION_DELETE = 'delete'
    _OPERATION_INSERT = 'insert'
    _OPERATION_INSERT_OR_UPDATE = 'insert_or_update'
    _OPERATION_REPLACE = 'replace'
    _OPERATION_UPDATE = 'update'

    def __init__(self, insert=None, update=None, insert_or_update=None, replace=None, delete=None, columns=None, values=None, keyset=None):
        if False:
            while True:
                i = 10
        '\n    A convenient class to create Spanner Mutations for Write. User can provide\n    the operation via constructor or via static methods.\n\n    Note: If a user passing the operation via construction, make sure that it\n    will only accept one operation at a time. For example, if a user passing\n    a table name in the `insert` parameter, and he also passes the `update`\n    parameter value, this will cause an error.\n\n    Args:\n      insert: (Optional) Name of the table in which rows will be inserted.\n      update: (Optional) Name of the table in which existing rows will be\n        updated.\n      insert_or_update: (Optional) Table name in which rows will be written.\n        Like insert, except that if the row already exists, then its column\n        values are overwritten with the ones provided. Any column values not\n        explicitly written are preserved.\n      replace: (Optional) Table name in which rows will be replaced. Like\n        insert, except that if the row already exists, it is deleted, and the\n        column values provided are inserted instead. Unlike `insert_or_update`,\n        this means any values not explicitly written become `NULL`.\n      delete: (Optional) Table name from which rows will be deleted. Succeeds\n        whether or not the named rows were present.\n      columns: The names of the columns in table to be written. The list of\n        columns must contain enough columns to allow Cloud Spanner to derive\n        values for all primary key columns in the row(s) to be modified.\n      values: The values to be written. `values` can contain more than one\n        list of values. If it does, then multiple rows are written, one for\n        each entry in `values`. Each list in `values` must have exactly as\n        many entries as there are entries in columns above. Sending multiple\n        lists is equivalent to sending multiple Mutations, each containing one\n        `values` entry and repeating table and columns.\n      keyset: (Optional) The primary keys of the rows within table to delete.\n        Delete is idempotent. The transaction will succeed even if some or\n        all rows do not exist.\n    '
        self._columns = columns
        self._values = values
        self._keyset = keyset
        self._insert = insert
        self._update = update
        self._insert_or_update = insert_or_update
        self._replace = replace
        self._delete = delete
        if sum([1 for x in [self._insert, self._update, self._insert_or_update, self._replace, self._delete] if x is not None]) != 1:
            raise ValueError('No or more than one write mutation operation provided: <%s: %s>' % (self.__class__.__name__, str(self.__dict__)))

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._insert is not None:
            return WriteMutation.insert(table=self._insert, columns=self._columns, values=self._values)
        elif self._update is not None:
            return WriteMutation.update(table=self._update, columns=self._columns, values=self._values)
        elif self._insert_or_update is not None:
            return WriteMutation.insert_or_update(table=self._insert_or_update, columns=self._columns, values=self._values)
        elif self._replace is not None:
            return WriteMutation.replace(table=self._replace, columns=self._columns, values=self._values)
        elif self._delete is not None:
            return WriteMutation.delete(table=self._delete, keyset=self._keyset)

    @staticmethod
    def insert(table, columns, values):
        if False:
            while True:
                i = 10
        'Insert one or more new table rows.\n\n    Args:\n      table: Name of the table to be modified.\n      columns: Name of the table columns to be modified.\n      values: Values to be modified.\n    '
        rows = len(values)
        cells = len(columns) * len(values)
        return _Mutator(mutation=Mutation(insert=batch._make_write_pb(table, columns, values)), operation=WriteMutation._OPERATION_INSERT, rows=rows, cells=cells, kwargs={'table': table, 'columns': columns, 'values': values})

    @staticmethod
    def update(table, columns, values):
        if False:
            i = 10
            return i + 15
        'Update one or more existing table rows.\n\n    Args:\n      table: Name of the table to be modified.\n      columns: Name of the table columns to be modified.\n      values: Values to be modified.\n    '
        rows = len(values)
        cells = len(columns) * len(values)
        return _Mutator(mutation=Mutation(update=batch._make_write_pb(table, columns, values)), operation=WriteMutation._OPERATION_UPDATE, rows=rows, cells=cells, kwargs={'table': table, 'columns': columns, 'values': values})

    @staticmethod
    def insert_or_update(table, columns, values):
        if False:
            i = 10
            return i + 15
        'Insert/update one or more table rows.\n    Args:\n      table: Name of the table to be modified.\n      columns: Name of the table columns to be modified.\n      values: Values to be modified.\n    '
        rows = len(values)
        cells = len(columns) * len(values)
        return _Mutator(mutation=Mutation(insert_or_update=batch._make_write_pb(table, columns, values)), operation=WriteMutation._OPERATION_INSERT_OR_UPDATE, rows=rows, cells=cells, kwargs={'table': table, 'columns': columns, 'values': values})

    @staticmethod
    def replace(table, columns, values):
        if False:
            print('Hello World!')
        'Replace one or more table rows.\n\n    Args:\n      table: Name of the table to be modified.\n      columns: Name of the table columns to be modified.\n      values: Values to be modified.\n    '
        rows = len(values)
        cells = len(columns) * len(values)
        return _Mutator(mutation=Mutation(replace=batch._make_write_pb(table, columns, values)), operation=WriteMutation._OPERATION_REPLACE, rows=rows, cells=cells, kwargs={'table': table, 'columns': columns, 'values': values})

    @staticmethod
    def delete(table, keyset):
        if False:
            print('Hello World!')
        'Delete one or more table rows.\n\n    Args:\n      table: Name of the table to be modified.\n      keyset: Keys/ranges identifying rows to delete.\n    '
        delete = Mutation.Delete(table=table, key_set=keyset._to_pb())
        return _Mutator(mutation=Mutation(delete=delete), rows=0, cells=0, operation=WriteMutation._OPERATION_DELETE, kwargs={'table': table, 'keyset': keyset})

@with_input_types(typing.Union[MutationGroup, TaggedOutput])
@with_output_types(MutationGroup)
class _BatchFn(DoFn):
    """
  Batches mutations together.
  """

    def __init__(self, max_batch_size_bytes, max_number_rows, max_number_cells):
        if False:
            return 10
        self._max_batch_size_bytes = max_batch_size_bytes
        self._max_number_rows = max_number_rows
        self._max_number_cells = max_number_cells

    def start_bundle(self):
        if False:
            while True:
                i = 10
        self._batch = MutationGroup()
        self._size_in_bytes = 0
        self._rows = 0
        self._cells = 0

    def _reset_count(self):
        if False:
            for i in range(10):
                print('nop')
        self._batch = MutationGroup()
        self._size_in_bytes = 0
        self._rows = 0
        self._cells = 0

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        mg_info = element.info
        if mg_info['byte_size'] + self._size_in_bytes > self._max_batch_size_bytes or mg_info['cells'] + self._cells > self._max_number_cells or mg_info['rows'] + self._rows > self._max_number_rows:
            if self._batch:
                yield self._batch
            self._reset_count()
        self._batch.extend(element)
        self._size_in_bytes += mg_info['byte_size']
        self._rows += mg_info['rows']
        self._cells += mg_info['cells']

    def finish_bundle(self):
        if False:
            print('Hello World!')
        if self._batch is not None:
            yield window.GlobalWindows.windowed_value(self._batch)
            self._batch = None

@with_input_types(MutationGroup)
@with_output_types(MutationGroup)
class _BatchableFilterFn(DoFn):
    """
  Filters MutationGroups larger than the batch size to the output tagged with
  OUTPUT_TAG_UNBATCHABLE.
  """
    OUTPUT_TAG_UNBATCHABLE = 'unbatchable'

    def __init__(self, max_batch_size_bytes, max_number_rows, max_number_cells):
        if False:
            for i in range(10):
                print('nop')
        self._max_batch_size_bytes = max_batch_size_bytes
        self._max_number_rows = max_number_rows
        self._max_number_cells = max_number_cells
        self._batchable = None
        self._unbatchable = None

    def process(self, element):
        if False:
            print('Hello World!')
        if element.primary().operation == WriteMutation._OPERATION_DELETE:
            yield TaggedOutput(_BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE, element)
        else:
            mg_info = element.info
            if mg_info['byte_size'] > self._max_batch_size_bytes or mg_info['cells'] > self._max_number_cells or mg_info['rows'] > self._max_number_rows:
                yield TaggedOutput(_BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE, element)
            else:
                yield element

class _WriteToSpannerDoFn(DoFn):

    def __init__(self, spanner_configuration):
        if False:
            print('Hello World!')
        self._spanner_configuration = spanner_configuration
        self._db_instance = None
        self.batches = Metrics.counter(self.__class__, 'SpannerBatches')
        self.base_labels = {monitoring_infos.SERVICE_LABEL: 'Spanner', monitoring_infos.METHOD_LABEL: 'Write', monitoring_infos.SPANNER_PROJECT_ID: spanner_configuration.project, monitoring_infos.SPANNER_DATABASE_ID: spanner_configuration.database}
        self.service_metrics = {}

    def _register_table_metric(self, table_id):
        if False:
            for i in range(10):
                print('nop')
        if table_id in self.service_metrics:
            return
        database_id = self._spanner_configuration.database
        project_id = self._spanner_configuration.project
        resource = resource_identifiers.SpannerTable(project_id, database_id, table_id)
        labels = {**self.base_labels, monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.SPANNER_TABLE_ID: table_id}
        service_call_metric = ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)
        self.service_metrics[table_id] = service_call_metric

    def setup(self):
        if False:
            print('Hello World!')
        spanner_client = Client(self._spanner_configuration.project)
        instance = spanner_client.instance(self._spanner_configuration.instance)
        self._db_instance = instance.database(self._spanner_configuration.database, pool=self._spanner_configuration.pool)

    def start_bundle(self):
        if False:
            print('Hello World!')
        self.service_metrics = {}

    def process(self, element):
        if False:
            print('Hello World!')
        self.batches.inc()
        try:
            with self._db_instance.batch() as b:
                for m in element:
                    table_id = m.kwargs['table']
                    self._register_table_metric(table_id)
                    if m.operation == WriteMutation._OPERATION_DELETE:
                        batch_func = b.delete
                    elif m.operation == WriteMutation._OPERATION_REPLACE:
                        batch_func = b.replace
                    elif m.operation == WriteMutation._OPERATION_INSERT_OR_UPDATE:
                        batch_func = b.insert_or_update
                    elif m.operation == WriteMutation._OPERATION_INSERT:
                        batch_func = b.insert
                    elif m.operation == WriteMutation._OPERATION_UPDATE:
                        batch_func = b.update
                    else:
                        raise ValueError('Unknown operation action: %s' % m.operation)
                    batch_func(**m.kwargs)
        except (ClientError, GoogleAPICallError) as e:
            for service_metric in self.service_metrics.values():
                service_metric.call(str(e.code.value))
            raise
        except HttpError as e:
            for service_metric in self.service_metrics.values():
                service_metric.call(str(e))
            raise
        else:
            for service_metric in self.service_metrics.values():
                service_metric.call('ok')

@with_input_types(typing.Union[MutationGroup, _Mutator])
@with_output_types(MutationGroup)
class _MakeMutationGroupsFn(DoFn):
    """
  Make Mutation group object if the element is the instance of _Mutator.
  """

    def process(self, element):
        if False:
            print('Hello World!')
        if isinstance(element, MutationGroup):
            yield element
        elif isinstance(element, _Mutator):
            yield MutationGroup([element])
        else:
            raise ValueError('Invalid object type: %s. Object must be an instance of MutationGroup or WriteMutations' % str(element))

class _WriteGroup(PTransform):

    def __init__(self, max_batch_size_bytes, max_number_rows, max_number_cells):
        if False:
            while True:
                i = 10
        self._max_batch_size_bytes = max_batch_size_bytes
        self._max_number_rows = max_number_rows
        self._max_number_cells = max_number_cells

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        filter_batchable_mutations = pcoll | 'Making mutation groups' >> ParDo(_MakeMutationGroupsFn()) | 'Filtering Batchable Mutations' >> ParDo(_BatchableFilterFn(max_batch_size_bytes=self._max_batch_size_bytes, max_number_rows=self._max_number_rows, max_number_cells=self._max_number_cells)).with_outputs(_BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE, main='batchable')
        batching_batchables = filter_batchable_mutations['batchable'] | ParDo(_BatchFn(max_batch_size_bytes=self._max_batch_size_bytes, max_number_rows=self._max_number_rows, max_number_cells=self._max_number_cells))
        return (batching_batchables, filter_batchable_mutations[_BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE]) | 'Merging batchable and unbatchable' >> Flatten()