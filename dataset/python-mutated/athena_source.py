from typing import Callable, Dict, Iterable, Optional, Tuple
from feast import type_map
from feast.data_source import DataSource
from feast.errors import DataSourceNoNameException, DataSourceNotFoundException
from feast.feature_logging import LoggingDestination
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.protos.feast.core.FeatureService_pb2 import LoggingConfig as LoggingConfigProto
from feast.protos.feast.core.SavedDataset_pb2 import SavedDatasetStorage as SavedDatasetStorageProto
from feast.repo_config import RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.value_type import ValueType

class AthenaSource(DataSource):

    def __init__(self, *, timestamp_field: Optional[str]='', table: Optional[str]=None, database: Optional[str]=None, data_source: Optional[str]=None, created_timestamp_column: Optional[str]=None, field_mapping: Optional[Dict[str, str]]=None, date_partition_column: Optional[str]=None, query: Optional[str]=None, name: Optional[str]=None, description: Optional[str]='', tags: Optional[Dict[str, str]]=None, owner: Optional[str]=''):
        if False:
            return 10
        "\n        Creates a AthenaSource object.\n\n        Args:\n            timestamp_field : event timestamp column.\n            table (optional): Athena table where the features are stored. Exactly one of 'table'\n                and 'query' must be specified.\n            database: Athena Database Name\n            data_source (optional): Athena data source\n            created_timestamp_column (optional): Timestamp column indicating when the\n                row was created, used for deduplicating rows.\n            field_mapping (optional): A dictionary mapping of column names in this data\n                source to column names in a feature table or view.\n            date_partition_column : Timestamp column used for partitioning.\n            query (optional): The query to be executed to obtain the features. Exactly one of 'table'\n                and 'query' must be specified.\n            name (optional): Name for the source. Defaults to the table if not specified, in which\n                case the table must be specified.\n            description (optional): A human-readable description.\n            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.\n            owner (optional): The owner of the athena source, typically the email of the primary\n                maintainer.\n        "
        _database = 'default' if table and (not database) else database
        self.athena_options = AthenaOptions(table=table, query=query, database=_database, data_source=data_source)
        if table is None and query is None:
            raise ValueError('No "table" argument provided.')
        if name is None and table is None:
            raise DataSourceNoNameException()
        _name = name or table
        assert _name
        super().__init__(name=_name if _name else '', timestamp_field=timestamp_field, created_timestamp_column=created_timestamp_column, field_mapping=field_mapping, date_partition_column=date_partition_column, description=description, tags=tags, owner=owner)

    @staticmethod
    def from_proto(data_source: DataSourceProto):
        if False:
            print('Hello World!')
        '\n        Creates a AthenaSource from a protobuf representation of a AthenaSource.\n\n        Args:\n            data_source: A protobuf representation of a AthenaSource\n\n        Returns:\n            A AthenaSource object based on the data_source protobuf.\n        '
        return AthenaSource(name=data_source.name, timestamp_field=data_source.timestamp_field, table=data_source.athena_options.table, database=data_source.athena_options.database, data_source=data_source.athena_options.data_source, created_timestamp_column=data_source.created_timestamp_column, field_mapping=dict(data_source.field_mapping), date_partition_column=data_source.date_partition_column, query=data_source.athena_options.query, description=data_source.description, tags=dict(data_source.tags))

    def __hash__(self):
        if False:
            print('Hello World!')
        return super().__hash__()

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, AthenaSource):
            raise TypeError('Comparisons should only involve AthenaSource class objects.')
        return super().__eq__(other) and self.athena_options.table == other.athena_options.table and (self.athena_options.query == other.athena_options.query) and (self.athena_options.database == other.athena_options.database) and (self.athena_options.data_source == other.athena_options.data_source)

    @property
    def table(self):
        if False:
            return 10
        'Returns the table of this Athena source.'
        return self.athena_options.table

    @property
    def database(self):
        if False:
            print('Hello World!')
        'Returns the database of this Athena source.'
        return self.athena_options.database

    @property
    def query(self):
        if False:
            return 10
        'Returns the Athena query of this Athena source.'
        return self.athena_options.query

    @property
    def data_source(self):
        if False:
            while True:
                i = 10
        'Returns the Athena data_source of this Athena source.'
        return self.athena_options.data_source

    def to_proto(self) -> DataSourceProto:
        if False:
            i = 10
            return i + 15
        '\n        Converts a RedshiftSource object to its protobuf representation.\n\n        Returns:\n            A DataSourceProto object.\n        '
        data_source_proto = DataSourceProto(type=DataSourceProto.BATCH_ATHENA, name=self.name, timestamp_field=self.timestamp_field, created_timestamp_column=self.created_timestamp_column, field_mapping=self.field_mapping, date_partition_column=self.date_partition_column, description=self.description, tags=self.tags, athena_options=self.athena_options.to_proto())
        return data_source_proto

    def validate(self, config: RepoConfig):
        if False:
            i = 10
            return i + 15
        self.get_table_column_names_and_types(config)

    def get_table_query_string(self, config: Optional[RepoConfig]=None) -> str:
        if False:
            i = 10
            return i + 15
        'Returns a string that can directly be used to reference this table in SQL.'
        if self.table:
            data_source = self.data_source
            database = self.database
            if config:
                data_source = config.offline_store.data_source
                database = config.offline_store.database
            return f'"{data_source}"."{database}"."{self.table}"'
        else:
            return f'({self.query})'

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        if False:
            for i in range(10):
                print('nop')
        return type_map.athena_to_feast_value_type

    def get_table_column_names_and_types(self, config: RepoConfig) -> Iterable[Tuple[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a mapping of column names to types for this Athena source.\n\n        Args:\n            config: A RepoConfig describing the feature repo\n        '
        from botocore.exceptions import ClientError
        from feast.infra.offline_stores.contrib.athena_offline_store.athena import AthenaOfflineStoreConfig
        from feast.infra.utils import aws_utils
        assert isinstance(config.offline_store, AthenaOfflineStoreConfig)
        client = aws_utils.get_athena_data_client(config.offline_store.region)
        if self.table:
            try:
                table = client.get_table_metadata(CatalogName=self.data_source, DatabaseName=self.database, TableName=self.table)
            except ClientError as e:
                raise aws_utils.AthenaError(e)
            if len(table['TableMetadata']['Columns']) == 0:
                raise DataSourceNotFoundException(self.table)
            columns = table['TableMetadata']['Columns']
        else:
            statement_id = aws_utils.execute_athena_query(client, config.offline_store.data_source, config.offline_store.database, config.offline_store.workgroup, f'SELECT * FROM ({self.query}) LIMIT 1')
            columns = aws_utils.get_athena_query_result(client, statement_id)['ResultSetMetadata']['ColumnInfo']
        return [(column['Name'], column['Type'].upper()) for column in columns]

class AthenaOptions:
    """
    Configuration options for a Athena data source.
    """

    def __init__(self, table: Optional[str], query: Optional[str], database: Optional[str], data_source: Optional[str]):
        if False:
            i = 10
            return i + 15
        self.table = table or ''
        self.query = query or ''
        self.database = database or ''
        self.data_source = data_source or ''

    @classmethod
    def from_proto(cls, athena_options_proto: DataSourceProto.AthenaOptions):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a AthenaOptions from a protobuf representation of a Athena option.\n\n        Args:\n            athena_options_proto: A protobuf representation of a DataSource\n\n        Returns:\n            A AthenaOptions object based on the athena_options protobuf.\n        '
        athena_options = cls(table=athena_options_proto.table, query=athena_options_proto.query, database=athena_options_proto.database, data_source=athena_options_proto.data_source)
        return athena_options

    def to_proto(self) -> DataSourceProto.AthenaOptions:
        if False:
            while True:
                i = 10
        '\n        Converts an AthenaOptionsProto object to its protobuf representation.\n\n        Returns:\n            A AthenaOptionsProto protobuf.\n        '
        athena_options_proto = DataSourceProto.AthenaOptions(table=self.table, query=self.query, database=self.database, data_source=self.data_source)
        return athena_options_proto

class SavedDatasetAthenaStorage(SavedDatasetStorage):
    _proto_attr_name = 'athena_storage'
    athena_options: AthenaOptions

    def __init__(self, table_ref: str, query: str=None, database: str=None, data_source: str=None):
        if False:
            print('Hello World!')
        self.athena_options = AthenaOptions(table=table_ref, query=query, database=database, data_source=data_source)

    @staticmethod
    def from_proto(storage_proto: SavedDatasetStorageProto) -> SavedDatasetStorage:
        if False:
            while True:
                i = 10
        return SavedDatasetAthenaStorage(table_ref=AthenaOptions.from_proto(storage_proto.athena_storage).table)

    def to_proto(self) -> SavedDatasetStorageProto:
        if False:
            i = 10
            return i + 15
        return SavedDatasetStorageProto(athena_storage=self.athena_options.to_proto())

    def to_data_source(self) -> DataSource:
        if False:
            i = 10
            return i + 15
        return AthenaSource(table=self.athena_options.table)

class AthenaLoggingDestination(LoggingDestination):
    _proto_kind = 'athena_destination'
    table_name: str

    def __init__(self, *, table_name: str):
        if False:
            while True:
                i = 10
        self.table_name = table_name

    @classmethod
    def from_proto(cls, config_proto: LoggingConfigProto) -> 'LoggingDestination':
        if False:
            print('Hello World!')
        return AthenaLoggingDestination(table_name=config_proto.athena_destination.table_name)

    def to_proto(self) -> LoggingConfigProto:
        if False:
            while True:
                i = 10
        return LoggingConfigProto(athena_destination=LoggingConfigProto.AthenaDestination(table_name=self.table_name))

    def to_data_source(self) -> DataSource:
        if False:
            return 10
        return AthenaSource(table=self.table_name)